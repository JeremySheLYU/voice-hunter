"""Local media I/O for short, sample-aligned stereo clips."""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 48000
MAX_SECONDS = 60
MAX_UPLOAD = 160 * 1024 * 1024
EXTENSIONS = {".mov", ".mp4", ".m4v", ".m4a", ".wav", ".aif", ".aiff", ".flac", ".mp3", ".aac", ".ogg", ".webm", ".mkv"}
FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,wav,aiff,flac,mp3,aac,ogg,matroska,webm"


def tool(argv, timeout=65):
    try:
        return subprocess.run(argv, capture_output=True, check=True, timeout=timeout).stdout
    except FileNotFoundError as exc:
        raise ValueError(f"找不到 {argv[0]}，請先安裝 FFmpeg。") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{argv[0]} 無法讀寫這個媒體檔。請試用原始 MOV、MP4 或 WAV。") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("媒體處理超時，請試用較短片段。") from exc


def load_media(path):
    path = Path(path).resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD:
        raise ValueError("請選擇非空、大小在 160 MiB 以內的媒體檔。")
    flags = ["-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS]
    probe = json.loads(tool(["ffprobe", *flags, "-show_entries",
                             "format=start_time,duration:stream=codec_type,codec_name,channels,sample_rate,"
                             "start_time,duration:stream_disposition=attached_pic",
                             "-of", "json", str(path)]))
    streams = probe.get("streams", [])
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if a is None or a.get("channels") != 2:
        raise ValueError("請選擇含雙聲道音訊的影片或錄音。這版的輸入是 stereo。")
    container = probe.get("format", {})
    duration = _timestamp(container.get("duration"))
    audio_duration = _timestamp(a.get("duration"))
    origin = _timestamp(container.get("start_time")) or 0.0
    audio_start = _timestamp(a.get("start_time"))
    offset = (audio_start - origin) if audio_start is not None else 0.0
    if any(v is not None and v < 0 for v in (audio_duration, duration)):
        raise ValueError("媒體的音訊時長無效。")
    if any(v is not None and v > MAX_SECONDS + 0.02 for v in (audio_duration, duration, abs(offset))):
        raise ValueError(f"目前每段最多 {MAX_SECONDS} 秒，請先裁短。")
    # Normalize to the movie timeline. A track that enters late gets leading
    # silence; timestamp gaps remain gaps in the preview, stems and remix.
    raw = tool(["ffmpeg", "-nostdin", *flags, "-i", str(path), "-map", "0:a:0",
                "-vn", "-sn", "-dn", "-af", f"aresample={SAMPLE_RATE}:async=1:first_pts=0",
                "-t", str(MAX_SECONDS + 0.1), "-ac", "2",
                "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"])
    x = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    # AAC may emit a final padded frame. Trim only a small codec tail and use
    # the audio end time; a video track may continue after all sound has ended.
    end = max(0.0, offset + audio_duration) if audio_duration is not None else duration
    if end is not None and end > 0 and 0 < len(x) / SAMPLE_RATE - end < .05:
        x = x[:round(end * SAMPLE_RATE)]
    if not len(x) or not np.isfinite(x).all() or len(x) / SAMPLE_RATE > MAX_SECONDS + 0.02:
        raise ValueError("音訊資料為空、過長或包含無效取樣。")
    return x, {"codec": a["codec_name"], "source_rate": int(a["sample_rate"]),
               "has_video": any(s.get("codec_type") == "video" and
                                not s.get("disposition", {}).get("attached_pic") for s in streams),
               "container_duration": duration, "audio_start_offset": offset, "source_audio_duration": audio_duration}


def _timestamp(value):
    if value is None or value == "N/A":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("媒體時間戳無效。") from exc
    if not math.isfinite(seconds):
        raise ValueError("媒體時間戳無效。")
    return seconds


def waveform(x, bins=250):
    edges = np.linspace(0, len(x), min(bins, len(x)) + 1, dtype=int)
    vals = np.maximum.reduceat(np.max(np.abs(x), axis=1), edges[:-1])
    vals /= max(0.001, float(np.max(vals)))
    return np.round(vals, 4).tolist()


def write_wav(path, x, *, stem=False):
    x = np.ascontiguousarray(x, dtype="<f4")
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("算法輸出的聲道或數值無效。")
    path = Path(path)
    tmp = path.with_suffix(".part")
    if stem:
        # IEEE float WAV retains over-unity peaks for additive reconstruction.
        data = x.tobytes()
        fmt = struct.pack("<HHIIHH", 3, 2, SAMPLE_RATE, SAMPLE_RATE * 8, 8, 32)
        chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<II", 4, len(x))
        with open(tmp, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 12 + len(chunks) + len(data)) + b"WAVE" + chunks)
            f.write(b"data" + struct.pack("<I", len(data)))
            f.write(data)
        gain = 1.
    else:
        peak = float(np.max(np.abs(x)))
        gain = 1. / max(1., peak)
        samples = np.rint(np.clip(x * gain, -1., 1.) * 8388607).astype("<i4")
        payload = samples.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        with wave.open(str(tmp), "wb") as f:
            f.setparams((2, 3, SAMPLE_RATE, len(x), "NONE", "not compressed"))
            f.writeframes(payload)
    os.replace(tmp, path)
    return gain


def mux_video(source, audio, destination):
    # MOV retains the original video codec (including iPhone HEVC).
    tool(["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
          "-format_whitelist", FORMATS, "-i", str(source), "-i", str(audio),
          "-map", "0:V:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
          "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mov", str(destination)])
