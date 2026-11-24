#!/bin/bash
set -eu
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v python3 >/dev/null || ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
    echo "請依 README.md 安裝 Python 3、NumPy 與 FFmpeg，然後重開。"; read -r -p "按 Enter 結束。" reply; exit 1
fi
if [ -x ".venv/bin/python" ]; then py=".venv/bin/python"; else py="$(command -v python3)"; fi
if ! "$py" -c 'import numpy' >/dev/null 2>&1; then
    echo "請依 README.md 安裝 NumPy，然後重開。"; read -r -p "按 Enter 結束。" reply; exit 1
fi
exec "$py" scripts/launch.py
