"""Probe camera files: capture time, GPS, camera, orientation, colour profile."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess

from . import control
from datetime import datetime, timezone
from pathlib import Path

from .config import EXIFTOOL, FFPROBE, VIDEO_EXTS

DJI_NAME = re.compile(r"DJI_(\d{14})_")
TZ_DATE = re.compile(r"(\d{4})[:-](\d{2})[:-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?([+-]\d{2}:?\d{2}|Z)?")


def clip_id(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]


def find_videos(folder: Path) -> list[Path]:
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"):
            out.append(p)
    return sorted(out)


def ffprobe(path: Path) -> dict:
    r = control.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                    text=True, check=True)
    return json.loads(r.stdout)


def exif_batch(paths: list[Path]) -> dict[str, dict]:
    if not paths:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(paths), 50):
        chunk = paths[i:i + 50]
        r = control.run([EXIFTOOL, "-j", "-n", "-G1", "-api", "LargeFileSupport=1", *map(str, chunk)], text=True)
        try:
            for item in json.loads(r.stdout or "[]"):
                out[str(Path(item["SourceFile"]).resolve())] = item
        except json.JSONDecodeError:
            pass
    return out


def _parse_date(s: str) -> tuple[datetime, bool] | None:
    """Return (naive datetime, has_tz_offset). Offset dates are wall-clock local."""
    m = TZ_DATE.search(str(s))
    if not m:
        return None
    y, mo, d, h, mi, se, tz = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(se))
    except ValueError:
        return None
    if dt.year < 2000:
        return None
    return dt, bool(tz and tz != "Z")


def pick_capture_time(path: Path, exif: dict, make: str) -> tuple[datetime, str]:
    m = DJI_NAME.search(path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), "filename"

    offset_dates, local_dates, utc_dates = [], [], []
    for key, val in exif.items():
        k = key.split(":")[-1].lower()
        if "date" not in k or "modif" in k or k.startswith("file"):
            continue
        parsed = _parse_date(val)
        if not parsed:
            continue
        dt, has_tz = parsed
        if has_tz:
            offset_dates.append(dt)
        elif "original" in k or key.startswith(("XML", "Sony", "FUJIFILM", "ExifIFD")):
            local_dates.append(dt)
        elif key.startswith(("QuickTime", "Track", "Media")) and k in ("createdate", "trackcreatedate", "mediacreatedate"):
            utc_dates.append(dt)
    if offset_dates:
        return min(offset_dates), "metadata+tz"
    if local_dates:
        return min(local_dates), "metadata"
    if utc_dates:
        dt = min(utc_dates)
        if make.lower().startswith("dji"):
            return dt, "metadata"  # DJI stores local time in the UTC field
        return dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None), "utc->local"
    return datetime.fromtimestamp(path.stat().st_mtime), "file-mtime"


SRT_LAT = re.compile(r"latitude\s*:\s*(-?\d+\.\d+)", re.I)
SRT_LON = re.compile(r"longitude\s*:\s*(-?\d+\.\d+)", re.I)
SRT_GPS_OLD = re.compile(r"GPS\s*\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)")
SRT_COLOR = re.compile(r"color_md\s*:\s*([\w-]+)", re.I)
SRT_ALT = re.compile(r"rel_alt\s*:\s*(-?\d+\.?\d*)", re.I)


def read_dji_srt(path: Path) -> dict:
    for cand in (path.with_suffix(".SRT"), path.with_suffix(".srt")):
        if cand.exists():
            text = cand.read_text(errors="ignore")[:20000]
            info: dict = {}
            lat, lon = SRT_LAT.search(text), SRT_LON.search(text)
            if lat and lon:
                info["gps"] = (float(lat.group(1)), float(lon.group(1)))
            else:
                old = SRT_GPS_OLD.search(text)
                if old:
                    info["gps"] = (float(old.group(2)), float(old.group(1)))
            if (c := SRT_COLOR.search(text)):
                info["color_md"] = c.group(1).lower()
            if (a := SRT_ALT.search(text)):
                info["rel_alt"] = float(a.group(1))
            return info
    return {}


def detect_profile(exif: dict, vstream: dict, make: str, srt: dict) -> str:
    """Return one of: rec709, slog3, slog2, flog, flog2, dlog, dlogm, hlg."""
    blob = " ".join(str(v) for v in exif.values()).lower().replace("_", "-")
    color_md = srt.get("color_md", "")
    if "dlog-m" in color_md or "d-log m" in blob or "dlog-m" in blob or "d-log-m" in blob:
        return "dlogm"
    if "dlog" in color_md or "d-log" in blob:
        return "dlog"
    if "s-log3" in blob or "slog3" in blob:
        return "slog3"
    if "s-log2" in blob or "slog2" in blob:
        return "slog2"
    if "f-log2" in blob or "flog2" in blob:
        return "flog2"
    if "f-log" in blob or "flog" in blob:
        return "flog"
    trc = vstream.get("color_transfer", "")
    if trc == "arib-std-b67" or " hlg" in blob or "hybrid log" in blob:
        return "hlg"
    return "rec709"


def _gps(exif: dict) -> tuple[float, float] | None:
    lat = lon = None
    for k, v in exif.items():
        kk = k.split(":")[-1].lower()
        if kk in ("gpslatitude", "locationlatitude") and isinstance(v, (int, float)):
            lat = v
        elif kk in ("gpslongitude", "locationlongitude") and isinstance(v, (int, float)):
            lon = v
        elif kk in ("gpscoordinates", "gpsposition") and isinstance(v, str):
            parts = v.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                except ValueError:
                    pass
    if lat is None or lon is None or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
        return None
    return (float(lat), float(lon))


def _fps(s: str) -> float:
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _rotation(v: dict) -> int:
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    try:
        return int(v.get("tags", {}).get("rotate", 0)) % 360
    except ValueError:
        return 0


def probe_clip(path: Path, exif: dict) -> dict | None:
    try:
        info = ffprobe(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1]
    if not vstreams:
        return None
    v = vstreams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    duration = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    if duration < 1.0:
        return None

    make = str(next((val for k, val in exif.items() if k.endswith(":Make")), "")).strip()
    model = str(next((val for k, val in exif.items() if k.endswith(":Model")), "")).strip()
    if not make:
        enc = str(info.get("format", {}).get("tags", {}).get("encoder", ""))
        if path.name.upper().startswith("DJI") or "dji" in enc.lower():
            make = "DJI"
        elif path.name.upper().startswith("DSCF"):
            make = "FUJIFILM"
        elif path.name.upper().startswith(("C0", "DSC")):
            make = "SONY"

    srt = read_dji_srt(path) if make.upper().startswith("DJI") or path.name.upper().startswith("DJI") else {}
    captured, time_src = pick_capture_time(path, exif, make)
    rot = _rotation(v)
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    if rot in (90, 270):
        w, h = h, w
    fps = _fps(v.get("avg_frame_rate", "")) or _fps(v.get("r_frame_rate", ""))
    gps = srt.get("gps") or _gps(exif)
    is_drone = bool(srt.get("rel_alt") is not None and abs(srt.get("rel_alt", 0)) > 2) or model.upper().startswith("FC")
    proxy = None
    for cand in (path.with_suffix(".LRF"), path.with_suffix(".lrf")):
        if cand.exists():
            proxy = str(cand)

    return {
        "id": clip_id(path),
        "path": str(path),
        "name": path.name,
        "proxy": proxy,
        "captured": captured.isoformat(),
        "time_source": time_src,
        "duration": round(duration, 3),
        "width": w,
        "height": h,
        "rotation": rot,
        "fps": round(fps, 3),
        "codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt", ""),
        "bit_depth": 10 if "10" in v.get("pix_fmt", "") else 8,
        "color_space": v.get("color_space", ""),
        "color_transfer": v.get("color_transfer", ""),
        "color_range": "pc" if v.get("color_range") == "pc" else "tv",
        "has_audio": has_audio,
        "make": make,
        "model": model,
        "camera": camera_label(make, model, is_drone),
        "is_drone": is_drone,
        "gps": list(gps) if gps else None,
        "profile": detect_profile(exif, v, make, srt),
        "orientation": orientation_of(w, h),
    }


def camera_label(make: str, model: str, is_drone: bool) -> str:
    mk = make.upper()
    if mk.startswith("DJI"):
        return f"DJI {'drone' if is_drone else model or 'camera'}".strip()
    if mk.startswith("SONY"):
        return f"Sony {model}".strip()
    if mk.startswith("FUJI"):
        return f"Fujifilm {model}".strip()
    return f"{make} {model}".strip() or "Unknown camera"


def orientation_of(w: int, h: int) -> str:
    return "portrait" if h > w else "landscape" if w > h else "square"


def set_rotate_fix(clip: dict, deg: int) -> dict:
    """Extra clockwise rotation on top of the camera's own tag, for clips the camera tagged wrongly."""
    deg %= 360
    if (deg - clip.get("rotate_fix", 0)) % 180 == 90:
        clip["width"], clip["height"] = clip["height"], clip["width"]
    clip["rotate_fix"] = deg
    clip["orientation"] = orientation_of(clip["width"], clip["height"])
    return clip


def rotate_filter(deg: int) -> str:
    """ffmpeg filter for a clockwise rotation, with a trailing comma, or '' for none."""
    f = {90: "transpose=clock", 180: "hflip,vflip", 270: "transpose=cclock"}.get(deg % 360, "")
    return f + "," if f else ""
