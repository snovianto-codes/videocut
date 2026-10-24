"""Generate a fake multi-day, multi-camera trip for testing (procedural scenes, camera moves, log profiles)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import EXIFTOOL, FFMPEG  # noqa: E402

BW, BH = 2880, 1620


def noise(h, w, scale, rng):
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255


def ridge(w, base, amp, rng, rough=6):
    x = np.linspace(0, 1, w)
    y = np.full(w, base, np.float32)
    for k in range(1, rough + 1):
        y += amp / k * np.sin(2 * np.pi * (k * 1.3 * x + rng.random()))
    return y


def scene(kind: str, rng) -> Image.Image:
    H, W = BH, BW
    yy = np.linspace(0, 1, H)[:, None, None]
    pal = {
        "beach": ((0.35, 0.62, 0.92), (0.80, 0.90, 0.98)),
        "sunset": ((0.25, 0.20, 0.45), (1.00, 0.55, 0.25)),
        "forest": ((0.45, 0.65, 0.85), (0.85, 0.92, 0.95)),
        "mountain": ((0.30, 0.50, 0.80), (0.85, 0.90, 0.95)),
        "city": ((0.02, 0.03, 0.10), (0.12, 0.10, 0.22)),
        "temple": ((0.50, 0.70, 0.90), (0.95, 0.85, 0.70)),
    }[kind]
    top, hor = np.array(pal[0]), np.array(pal[1])
    img = top + (hor - top) * yy ** 0.8
    img = np.broadcast_to(img, (H, W, 3)).copy()
    tex = noise(H, W, 3, rng)[..., None]
    fine = noise(H, W, 1, rng)[..., None]
    hz = int(H * 0.55)
    cols = np.arange(W)
    if kind in ("beach", "sunset"):
        sea = np.array((0.05, 0.35, 0.55) if kind == "beach" else (0.35, 0.18, 0.25))
        img[hz:] = sea * (0.7 + 0.5 * tex[hz:]) + 0.15 * fine[hz:]
        sand_line = ridge(W, H * 0.78, H * 0.03, rng)
        mask = np.arange(H)[:, None] > sand_line[None, :]
        sand = np.array((0.93, 0.82, 0.62) if kind == "beach" else (0.55, 0.35, 0.30))
        img[mask] = (sand * (0.8 + 0.3 * tex) + 0.1 * fine)[mask]
        if kind == "sunset":
            cy, cx = int(H * 0.48), int(W * 0.6)
            d = np.hypot(np.arange(H)[:, None] - cy, cols[None, :] - cx)
            img += (np.exp(-(d / 90) ** 2) * 1.5 + np.exp(-(d / 600) ** 2) * 0.35)[..., None] * np.array((1.0, 0.7, 0.3))
    elif kind in ("forest", "mountain", "temple"):
        for layer, (base, amp, color) in enumerate([
            (0.45, 0.10, (0.55, 0.62, 0.75)),
            (0.58, 0.07, (0.25, 0.42, 0.30) if kind != "mountain" else (0.42, 0.45, 0.55)),
            (0.72, 0.05, (0.10, 0.30, 0.12) if kind != "mountain" else (0.25, 0.28, 0.25)),
        ]):
            line = ridge(W, H * base, H * amp, rng, rough=8 + layer * 4)
            mask = np.arange(H)[:, None] > line[None, :]
            img[mask] = (np.array(color) * (0.7 + 0.5 * tex) + 0.08 * fine)[mask]
        if kind == "mountain":
            snow = ridge(W, H * 0.40, H * 0.10, rng)
            line = ridge(W, H * 0.45, H * 0.10, rng)
            m = (np.arange(H)[:, None] > line[None, :] - 1) & (np.arange(H)[:, None] < snow[None, :] + H * 0.08)
            img[m] = img[m] * 0.4 + 0.6
    elif kind == "city":
        x = 0
        while x < W:
            bw = rng.integers(80, 220)
            bh = rng.integers(int(H * 0.25), int(H * 0.75))
            img[H - bh:, x:x + bw] = np.array((0.04, 0.05, 0.08)) + 0.03 * tex[H - bh:, x:x + bw]
            for wy in range(H - bh + 20, H - 10, 34):
                for wx in range(x + 10, x + bw - 14, 26):
                    if rng.random() < 0.45:
                        img[wy:wy + 14, wx:wx + 10] = rng.choice([(1.0, 0.85, 0.5), (0.6, 0.85, 1.0), (1.0, 0.4, 0.6)])
            x += bw + rng.integers(4, 30)
    if kind == "temple":
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        cx, base = int(W * 0.5), int(H * 0.7)
        for k in range(5):
            w2 = 520 - k * 90
            d.rectangle([cx - w2 // 2, base - (k + 1) * 110, cx + w2 // 2, base - k * 110 - 20], fill=(150, 70, 45))
            d.polygon([(cx - w2 // 2 - 60, base - k * 110 - 20), (cx + w2 // 2 + 60, base - k * 110 - 20), (cx, base - k * 110 - 70)], fill=(70, 40, 35))
        return im.filter(ImageFilter.SHARPEN)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.SHARPEN)


def encode_cube(profile: str, path: Path, n: int = 33):
    """Display sRGB -> camera log RGB (legal-range normalised), the inverse of backend.looks."""
    axis = np.linspace(0, 1, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    rgb = np.stack([r, g, b], -1).reshape(-1, 3)
    lin = rgb ** 2.2 * 0.9
    if profile == "slog3":
        cv = np.where(lin >= 0.01125, (420 + np.log10((lin + 0.01) / 0.19) * 261.5), lin * (171.2102946929 - 95) / 0.01125 + 95)
        out = (cv - 64) / 876
    elif profile == "flog":
        y = np.where(lin >= 0.00089, 0.344676 * np.log10(0.555556 * lin + 0.009468) + 0.790453, 8.735631 * lin + 0.092864)
        out = (y * 1023 - 64) / 876
    elif profile == "dlog":
        y = np.where(lin <= 0.0078, 6.025 * lin + 0.0929, np.log10(lin * 0.9892 + 0.0108) * 0.256663 + 0.584555)
        out = (y * 1023 - 64) / 876
    else:
        out = rgb
    out = np.clip(out, 0, 1)
    path.write_text(f"LUT_3D_SIZE {n}\n" + "\n".join(f"{a:.6f} {bb:.6f} {c:.6f}" for a, bb, c in out) + "\n")


CLIPS = [
    # name, scene, start time, duration, fps, profile, make/model, gps, portrait, shake, pan, exposure
    ("Day1/C0001.MP4", "beach", "2026-09-12T09:10:00+08:00", 9, 50, "slog3", ("SONY", "ILCE-7M4"), (-8.81, 115.09), False, 0.5, 1, 1.0),
    ("Day1/C0002.MP4", "beach", "2026-09-12T09:14:30+08:00", 7, 50, "slog3", ("SONY", "ILCE-7M4"), (-8.81, 115.09), False, 3.0, -1, 1.0),
    ("Day1/C0003.MP4", "temple", "2026-09-12T11:02:00+08:00", 8, 25, "slog3", ("SONY", "ILCE-7M4"), (-8.83, 115.08), False, 0.4, 1, 1.0),
    ("Day1/DSCF0101.MOV", "sunset", "2026-09-12T18:05:00+08:00", 10, 30, "flog", ("FUJIFILM", "X-T5"), (-8.71, 115.17), False, 0.8, -1, 1.0),
    ("Day1/DSCF0102.MOV", "sunset", "2026-09-12T18:12:00+08:00", 6, 30, "flog", ("FUJIFILM", "X-T5"), (-8.71, 115.17), True, 0.6, 1, 1.0),
    ("Day2/DJI_20260913081500_0001_D.MP4", "forest", "2026-09-13T08:15:00", 12, 60, "rec709", ("DJI", "FC8482"), (-8.43, 115.28), False, 0.0, 1, 1.0),
    ("Day2/DJI_20260913082200_0002_D.MP4", "mountain", "2026-09-13T08:22:00", 10, 60, "dlog", ("DJI", "FC8482"), (-8.42, 115.29), False, 0.0, -1, 1.0),
    ("Day2/C0010.MP4", "forest", "2026-09-13T10:40:00+08:00", 8, 25, "rec709", ("SONY", "ILCE-7M4"), (-8.43, 115.27), False, 1.2, 1, 0.45),
    ("Day3/C0020.MP4", "city", "2026-09-14T20:30:00+08:00", 9, 25, "rec709", ("SONY", "ILCE-7M4"), (-8.67, 115.21), False, 0.9, 1, 1.0),
    ("Day3/C0021.MP4", "city", "2026-09-14T20:41:00+08:00", 7, 25, "rec709", ("SONY", "ILCE-7M4"), (-8.67, 115.21), True, 0.7, -1, 1.0),
]


def make_clip(root: Path, spec, rng, work: Path):
    name, kind, when, dur, fps, profile, (make, model), gps, portrait, shake, pan, expo = spec
    out = root / name
    out.parent.mkdir(parents=True, exist_ok=True)
    still = work / f"{out.stem}.png"
    scene(kind, rng).save(still)
    ow, oh = (1080, 1920) if portrait else (1920, 1080)
    cw, ch = (int(BH * 9 / 16), BH) if portrait else (int(BW * 0.72), int(BW * 0.72 * 9 / 16))
    sx = f"{shake}*(6*sin(t*11)+4*sin(t*23+1)+3*sin(t*41))"
    sy = f"{shake}*(5*sin(t*13+2)+3*sin(t*31))"
    span = BW - cw - 40
    x = f"20+{span}*({0.15 if pan > 0 else 0.85}+{0.6 * pan}*t/{dur})+{sx}"
    y = f"({BH - ch})/2+{sy}"
    cube = work / f"enc_{profile}.cube"
    encode_cube(profile, cube)
    subj = "drawbox=x='iw*0.2+iw*0.5*t/{d}':y='ih*0.62+10*sin(t*6)':w=iw*0.035:h=ih*0.09:color=0xC04020@1:t=fill".format(d=dur)
    vf = (f"crop={cw}:{ch}:x='{x}':y='{y}',scale={ow}:{oh},{subj},noise=alls=5:allf=t,"
          f"eq=brightness={(expo - 1) * 0.4:.3f},lut3d=file={cube},format=yuv420p")
    tags = ["-metadata", f"creation_time={when.split('+')[0]}Z" if "+" in when else f"creation_time={when}Z"]
    cmd = [FFMPEG, "-y", "-v", "error", "-loop", "1", "-framerate", str(fps), "-i", str(still),
           "-f", "lavfi", "-i", f"anoisesrc=d={dur}:c=pink:a=0.03,lowpass=f=1500",
           "-f", "lavfi", "-i", f"sine=f=660:d={dur},volume='if(between(mod(t,4),2,2.25),0.25,0)':eval=frame",
           "-filter_complex", f"[0:v]{vf}[v];[1:a][2:a]amix=inputs=2,aformat=channel_layouts=stereo[a]",
           "-map", "[v]", "-map", "[a]", "-t", str(dur), "-r", str(fps),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", *tags, str(out)]
    subprocess.run(cmd, check=True)
    ex = [EXIFTOOL, "-q", "-overwrite_original", f"-Keys:Make={make}", f"-Keys:Model={model}"]
    if "+" in when:
        ex.append(f"-Keys:CreationDate={when.replace('-', ':', 2).replace('T', ' ')}")
    if profile in ("slog3", "flog"):
        ex.append(f"-Keys:Description=Gamma {'S-Log3' if profile == 'slog3' else 'F-Log'}")
    if not make.startswith("DJI"):
        ex.append(f"-Keys:GPSCoordinates={gps[0]}, {gps[1]}, 0")
    subprocess.run([*ex, str(out)], check=True)
    if make.startswith("DJI"):
        color = "dlog" if profile == "dlog" else "default"
        out.with_suffix(".SRT").write_text(
            f"1\n00:00:00,000 --> 00:00:00,033\n<font size=\"28\">FrameCnt: 1\n[iso: 100] [shutter: 1/500.0] [fnum: 1.7] "
            f"[ev: 0] [color_md: {color}] [focal_len: 24.00] [latitude: {gps[0]}] [longitude: {gps[1]}] "
            f"[rel_alt: 85.300 abs_alt: 412.110] </font>\n")
    print("made", out)


def make_music(path: Path, seconds: int = 70, bpm: int = 112):
    beat = 60 / bpm
    expr = (f"0.8*sin(2*PI*55*t)*exp(-18*mod(t,{beat:.5f}))"
            f"+0.25*sin(2*PI*220*t)*exp(-6*mod(t+{beat / 2:.5f},{beat:.5f}))"
            f"+0.15*sin(2*PI*330*t+sin(2*PI*0.25*t))")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d={seconds}",
                    "-af", "afade=t=in:d=2", "-ac", "2", "-b:a", "192k", str(path)], check=True)
    print("made", path)


SONG_FORM = [("intro", 8), ("build", 4), ("drop", 8), ("break", 4), ("drop", 8), ("outro", 4)]


def synth_song(bpm: int = 120, sr: int = 22050, seed: int = 3) -> tuple[np.ndarray, int, list[float]]:
    """A structured EDM-ish track. Returns (mono audio, sr, drop start times)."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bars = sum(n for _, n in SONG_FORM)
    n = int(bars * 4 * beat * sr) + sr
    y = np.zeros(n, np.float32)
    t_note = np.arange(int(beat * sr)) / sr
    chords = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6), (196.0, 246.9, 311.1)]

    def add(start, sig):
        i = int(start * sr)
        j = min(n, i + len(sig))
        y[i:j] += sig[: j - i]

    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-t_note * 30)) * t_note) * np.exp(-t_note * 9)
    hat = rng.standard_normal(int(0.05 * sr)) * np.exp(-np.arange(int(0.05 * sr)) / sr * 80)
    clap = rng.standard_normal(int(0.15 * sr)) * np.exp(-np.arange(int(0.15 * sr)) / sr * 25)
    crash = rng.standard_normal(int(1.5 * sr)) * np.exp(-np.arange(int(1.5 * sr)) / sr * 2.5)
    drops, bar_i = [], 0
    for name, count in SONG_FORM:
        for k in range(count):
            t0 = bar_i * 4 * beat
            ch = chords[(bar_i // 2) % len(chords)]
            tb = np.arange(int(4 * beat * sr)) / sr
            pad_level = {"intro": 0.06, "build": 0.08, "drop": 0.13, "break": 0.09, "outro": 0.05}[name]
            pad = sum(np.sin(2 * np.pi * f * tb) for f in ch) * pad_level * np.minimum(1, tb * 8)
            add(t0, pad)
            if name == "drop" and k == 0:
                drops.append(t0)
                add(t0, crash * 0.5)
            for b in range(4):
                tt = t0 + b * beat
                if name in ("intro", "outro", "break"):
                    add(tt + beat / 2, hat * 0.05)
                if name == "build":
                    rolls = 2 ** min(3, k)
                    for r in range(rolls):
                        add(tt + r * beat / rolls, clap * 0.08 * (0.4 + 0.6 * (k + b / 4) / count))
                if name == "drop":
                    add(tt, kick * (0.95 if b == 0 else 0.7))
                    add(tt + beat / 2, hat * 0.15)
                    if b in (1, 3):
                        add(tt, clap * 0.3)
                    if b == 0:
                        add(tt, np.sin(2 * np.pi * ch[0] / 4 * t_note) * np.exp(-t_note * 2) * 0.5)
            if name == "build":
                sweep = rng.standard_normal(len(tb)) * 0.04 * ((k * 4 * beat + tb) / (count * 4 * beat)) ** 2
                add(t0, sweep)
            bar_i += 1
    y /= np.abs(y).max() + 1e-9
    return y * 0.9, sr, drops


def make_song(path: Path, bpm: int = 120):
    import soundfile as sf

    y, sr, drops = synth_song(bpm)
    wav = path.with_suffix(".wav")
    sf.write(wav, y, sr)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(wav), "-ac", "2", "-b:a", "192k", str(path)], check=True)
    wav.unlink()
    print("made", path, "drops at", drops)


if __name__ == "__main__":
    if sys.argv[1] == "--song":
        make_song(Path(sys.argv[2]))
        sys.exit()
    root = Path(sys.argv[1])
    work = root / ".work"
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for spec in CLIPS:
        make_clip(root, spec, rng, work)
    if len(sys.argv) > 2:
        make_music(Path(sys.argv[2]))
