"""Per-clip frame analysis: what's sharp, colourful, moving, steady, well exposed, and where the subject is."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import control
from .media import rotate_filter
from .config import DATA_DIR, FFMPEG
from .looks import LUMA, auto_exposure, to_display

ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
VERSION = 4
BINS = 24
_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def cache_path(clip: dict) -> Path:
    return ANALYSIS_DIR / f"{clip['id']}.json"


def load(clip: dict) -> dict | None:
    p = cache_path(clip)
    if p.exists():
        data = json.loads(p.read_text())
        if data.get("version") == VERSION and data.get("profile") == clip["profile"] \
                and data.get("rotate_fix", 0) == clip.get("rotate_fix", 0):
            return data
    return None


def decode_frames(src: str, w: int, h: int, fps: float, hw: bool = True, rot: int = 0):
    cmd = [FFMPEG, "-v", "error"]
    if hw:
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", src, "-an", "-vf", f"{rotate_filter(rot)}fps={fps},scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = control.run(cmd)
    if r.returncode != 0 and hw:
        return decode_frames(src, w, h, fps, hw=False, rot=rot)
    raw = np.frombuffer(r.stdout, dtype=np.uint8)
    n = raw.size // (w * h * 3)
    return raw[: n * w * h * 3].reshape(n, h, w, 3)


def audio_rms(src: str, step: float) -> np.ndarray:
    r = control.run([FFMPEG, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"])
    pcm = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768
    hop = max(1, int(8000 * step))
    n = pcm.size // hop
    if n == 0:
        return np.zeros(0)
    return np.sqrt((pcm[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)


def colorfulness(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def analyze_clip(clip: dict, force: bool = False) -> dict:
    if not force and (cached := load(clip)):
        return cached
    src = clip.get("proxy") or clip["path"]
    dur = clip["duration"]
    fps = 3.0 if dur <= 180 else (1.5 if dur <= 600 else 0.75)
    aspect = clip["width"] / max(1, clip["height"])
    w = 256 if aspect >= 1 else 144
    h = int(round(w / aspect / 2) * 2)
    rot = clip.get("rotate_fix", 0)
    frames = decode_frames(src, w, h, fps, rot=rot)
    if len(frames) == 0:
        frames = decode_frames(clip["path"], w, h, fps, hw=False, rot=rot)
    n = len(frames)
    step = 1.0 / fps
    rms = audio_rms(clip["path"], step) if clip.get("has_audio") else np.zeros(0)

    profile = clip["profile"]
    f32 = frames.astype(np.float32) / 255.0
    ev = auto_exposure([f32[i] for i in np.linspace(0, n - 1, min(n, 6)).astype(int)], profile) if n else 0.0

    out = {k: [] for k in ("t", "sharp", "bright", "clip_hi", "clip_lo", "color", "motion", "dx", "dy", "faces", "face_x", "face_y", "audio", "colx", "coly", "sig")}
    prev_gray = None
    for i in range(n):
        disp = to_display(f32[i], profile, ev)
        gray8 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        grayf = gray8.astype(np.float32) / 255.0
        luma = disp @ LUMA
        out["t"].append(round(i * step + step / 2, 3))
        out["sharp"].append(float(cv2.Laplacian(gray8, cv2.CV_64F).var()))
        out["bright"].append(float(np.median(luma)))
        out["clip_hi"].append(float((luma > 0.97).mean()))
        out["clip_lo"].append(float((luma < 0.03).mean()))
        out["color"].append(colorfulness(disp))
        edges = np.abs(cv2.Sobel(grayf, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(grayf, cv2.CV_32F, 0, 1))
        if prev_gray is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray.astype(np.float64), grayf.astype(np.float64))
            shifted = cv2.warpAffine(prev_gray, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
            diff = np.abs(grayf - prev_gray)
            resid = cv2.GaussianBlur(np.abs(grayf - shifted), (5, 5), 0)
            out["motion"].append(float(diff.mean()))
            out["dx"].append(float(dx) / w)
            out["dy"].append(float(dy) / w)
            energy = edges * 0.3 + np.maximum(resid - np.percentile(resid, 75), 0) * 12.0
        else:
            out["motion"].append(0.0)
            out["dx"].append(0.0)
            out["dy"].append(0.0)
            energy = edges * 0.3
        faces = []
        if i % 2 == 0:
            faces = _face.detectMultiScale(gray8, scaleFactor=1.15, minNeighbors=5, minSize=(14, 14))
        if len(faces):
            big = max(faces, key=lambda f: f[2] * f[3])
            out["faces"].append(int(len(faces)))
            out["face_x"].append(round(float((big[0] + big[2] / 2) / w), 3))
            out["face_y"].append(round(float((big[1] + big[3] / 2) / h), 3))
            fx, fy = int(big[0] + big[2] / 2), int(big[1] + big[3] / 2)
            energy[max(0, fy - big[3]):fy + big[3], max(0, fx - big[2]):fx + big[2]] += 3.0
        else:
            out["faces"].append(0)
            out["face_x"].append(None)
            out["face_y"].append(None)
        col = energy.sum(axis=0)
        row = energy.sum(axis=1)
        out["colx"].append(np.round(_bin(col, BINS), 4).tolist())
        out["coly"].append(np.round(_bin(row, BINS), 4).tolist())
        tiny = cv2.resize(disp, (4, 3), interpolation=cv2.INTER_AREA).reshape(-1)
        out["sig"].append(np.round(tiny, 3).tolist())
        out["audio"].append(float(rms[i]) if i < len(rms) else 0.0)
        prev_gray = grayf

    contrast = float(np.percentile(f32.mean(-1), 95) - np.percentile(f32.mean(-1), 5)) if n else 0
    chroma = float((f32.max(-1) - f32.min(-1)).mean()) if n else 0
    result = {
        "version": VERSION,
        "clip_id": clip["id"],
        "profile": profile,
        "rotate_fix": rot,
        "fps": fps,
        "ev": ev,
        "log_suspect": profile == "rec709" and contrast < 0.55 and chroma < 0.09 and clip.get("bit_depth") == 10,
        **{k: v for k, v in out.items()},
    }
    tmp = cache_path(clip).with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(cache_path(clip))
    return result


def _bin(v: np.ndarray, bins: int) -> np.ndarray:
    idx = np.linspace(0, len(v), bins + 1).astype(int)
    s = np.array([v[a:b].sum() for a, b in zip(idx, idx[1:])])
    return s / (s.sum() + 1e-9)
