#!/bin/bash
# Start VideoCut on http://127.0.0.1:8430
cd "$(dirname "$0")"
# Lets Pillow find Homebrew's fribidi for flag and joined emoji in text overlays.
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8430}"
