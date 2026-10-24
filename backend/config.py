import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VIDEOCUT_DATA", ROOT / "data"))
EXPORT_DIR = Path(os.environ.get("VIDEOCUT_EXPORTS", Path.home() / "Movies" / "VideoCut"))
MUSIC_DIR = Path(os.environ.get("VIDEOCUT_MUSIC", ROOT / "music"))
LUT_DIR = ROOT / "luts"


def _tool(name: str) -> str:
    # Homebrew's build is preferred: older Anaconda ffmpeg lacks xfade.
    for cand in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if Path(cand).exists():
            return cand
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found; install with `brew install {name}`")
    return found


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")
EXIFTOOL = shutil.which("exiftool") or "/opt/homebrew/bin/exiftool"

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mts", ".m2ts", ".mxf", ".avi", ".mkv"}

FONT_TITLE = "/System/Library/Fonts/Avenir Next.ttc"
FONT_FALLBACK = "/System/Library/Fonts/Helvetica.ttc"

AI_MODEL = os.environ.get("VIDEOCUT_MODEL", "claude-opus-5-5")

for d in (DATA_DIR, EXPORT_DIR, MUSIC_DIR, LUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
