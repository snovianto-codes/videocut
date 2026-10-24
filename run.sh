#!/bin/bash
# Start VideoCut on http://127.0.0.1:8430
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8430}"
