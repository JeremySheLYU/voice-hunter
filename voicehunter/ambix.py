"""Decode declared ACN/SN3D AmbiX to virtual coincident left/right cardioids."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .media import FORMATS, MAX_SECONDS, MAX_UPLOAD, SAMPLE_RATE, tool, write_wav


def decode_ambix(path, *, half_angle=45):
    path = Path(path).resolve()
    if not math.isfinite(half_angle) or not 0 < half_angle <= 90:
        raise ValueError("夾角的一半須大於 0° 且不超過 90°。")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD:
        raise ValueError("請選擇 160 MiB 以內的 AmbiX 音檔。")
    source_options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS]
    probe = json.loads(tool(["ffprobe", "-v", "error", *source_options, "-select_streams", "a:0",
                             "-show_entries", "stream=channels,duration:format=duration", "-of", "json", str(path)]))
    streams = probe.get("streams", [])
    if not streams or streams[0].get("channels") not in (4, 9, 16, 25):
        raise ValueError("需要 4、9、16 或 25 聲道且已確認為 ACN/SN3D 的 AmbiX。")
    declared = streams[0].get("duration") or probe.get("format", {}).get("duration")
    if declared not in (None, "N/A") and not 0 < float(declared) <= MAX_SECONDS + .02:
        raise ValueError("請選擇 60 秒以內的 AmbiX 音檔。")
    angle = math.radians(half_angle)
    x = .5*math.cos(angle)
    y = .5*math.sin(angle)
    pan = f"pan=stereo|c0=0.5*c0+{x:.12f}*c3+{y:.12f}*c1|c1=0.5*c0+{x:.12f}*c3-{y:.12f}*c1"
    raw = tool(["ffmpeg", "-nostdin", "-v", "error", *source_options, "-i", str(path),
                "-map", "0:a:0", "-vn", "-af", pan, "-t", str(MAX_SECONDS+.12),
                "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"])
    stereo = np.frombuffer(raw, dtype="<f4").reshape(-1,2)
    if len(stereo) > round((MAX_SECONDS+.02)*SAMPLE_RATE):
        raise ValueError("請選擇 60 秒以內的 AmbiX 音檔。")
    if not len(stereo) or not np.isfinite(stereo).all():
        raise ValueError("無法解碼有限雙聲道樣本。")
    return stereo


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog="Requires known ACN/SN3D channel order. At higher orders this projection uses W/Y/X only.")
    p.add_argument("source", type=Path)
    p.add_argument("--layout", required=True, choices=["acn-sn3d"], help="confirm input channel convention")
    p.add_argument("--half-angle", type=float, default=45, help="virtual microphone axes at ±angle degrees (default ±45)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        signal = decode_ambix(args.source, half_angle=args.half_angle)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".ambix-",dir=args.out.parent) as tmp:
            output = Path(tmp)/"stereo.wav"
            write_wav(output, signal, stem=True)
            os.link(output, args.out)
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(1, f"{exc}\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
