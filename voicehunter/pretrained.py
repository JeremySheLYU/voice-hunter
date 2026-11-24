"""Pretrained speech enhancement and optional locally installed Demucs inference.

RNNoise runs offline with the original, redistributable Xiph model via FFmpeg.
Its mono speech estimate is a guide for stereo projection in engine.py.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


_MODELS = Path(__file__).parent / "models"
_MODEL_HASH = "6b8943dc4a9b6b24425873992a44f29c0577503276456af46a8854774faeb294"
_FRAME = 480  # 10 ms at the model's native rate.
_MUSIC = {"vocals": "歌聲", "drums": "鼓", "bass": "貝斯", "other": "其他伴奏"}
_WORKER = Path(__file__).with_name("demucs_worker.py")


def _run_neural_filter(audio):
    # arnndn works in 480-sample frames. One frame of trailing silence flushes
    # the analysis/synthesis latency. Trim the leading delay and padded tail.
    audio = np.asarray(audio, dtype="<f4").reshape(-1)
    padded = np.pad(audio, (0, _FRAME + (-len(audio) % _FRAME)))
    command = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", "pipe:0",
               "-af", "arnndn=m=std.rnnn", "-ar", "48000", "-ac", "1",
               "-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(command, cwd=_MODELS, input=padded.tobytes(), capture_output=True,
                                check=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("人聲模型無法執行。請確認 FFmpeg 內建 arnndn，且模型檔已隨專案安裝。") from exc
    out = np.frombuffer(result.stdout, dtype="<f4")
    if len(out) < len(audio) + _FRAME:
        raise ValueError("人聲模型產生的聲軌長度不足。")
    return out[_FRAME:_FRAME + len(audio)].copy()


@lru_cache(maxsize=1)
def _rnnoise_status():
    try:
        if hashlib.sha256((_MODELS / "std.rnnn").read_bytes()).hexdigest() != _MODEL_HASH:
            return False, "內建權重檔驗證失敗"
        _run_neural_filter(np.zeros(960, dtype=np.float32))
        return True, ""
    except (OSError, ValueError) as exc:
        return False, str(exc)


def _demucs_config():
    # Optional worker may use a separate Python environment from the local app.
    # Users explicitly opt in by supplying an official trusted .th checkpoint.
    model = os.getenv("VOICEHUNTER_DEMUCS_CHECKPOINT", "")
    python = os.getenv("VOICEHUNTER_DEMUCS_PYTHON", sys.executable)
    return python, str(Path(model).expanduser().absolute()) if model else ""


@lru_cache(maxsize=4)
def _demucs_status(python, checkpoint):
    if not checkpoint or not Path(checkpoint).is_file():
        return False, "未配置本地四軌 Demucs 權重"
    try:
        probe = subprocess.run([python, str(_WORKER), "--probe", checkpoint],
                               capture_output=True, timeout=60)
        if probe.returncode:
            return False, "請檢查 Demucs 環境及相容的四軌模型"
    except (OSError, subprocess.SubprocessError):
        return False, "無法載入本地 Demucs"
    return True, ""


def capabilities():
    ready, reason = _rnnoise_status()
    music_ready, music_reason = _demucs_status(*_demucs_config())
    return [
        {"id": "rnnoise", "label": "自動提取人聲 · RNNoise", "kind": "choice",
         "options": [{"value": "voice", "label": "所有說話者的人聲"}],
         "available": ready, "reason": reason},
        {"id": "demucs", "label": "音樂分軌 · Demucs", "kind": "choice",
         "options": [{"value": key, "label": label} for key, label in _MUSIC.items()],
         "available": music_ready, "reason": music_reason},
    ]


def _demucs_target(audio, sample_rate, target):
    if target not in _MUSIC:
        raise ValueError("請從可用的音樂分軌中選一項。")
    python, checkpoint = _demucs_config()
    ready, reason = _demucs_status(python, checkpoint)
    if not ready:
        raise ValueError(reason)
    with tempfile.TemporaryDirectory(prefix="voicehunter-music-") as folder:
        root = Path(folder)
        np.save(root / "input.npy", audio, allow_pickle=False)
        try:
            subprocess.run([python, str(_WORKER), "--run", checkpoint, target,
                            str(sample_rate), str(root)], check=True, capture_output=True,
                           timeout=900)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("本地 Demucs 模型未能完成音樂分軌。") from exc
        output = np.load(root / "target.npy", allow_pickle=False)
    if output.shape != audio.shape or not np.isfinite(output).all():
        raise ValueError("音樂模型輸出的音訊無效。")
    return output, audio - output


def separate_pretrained(audio_stereo, sample_rate, model, target, progress=None):
    """Return aligned target and residual waveform estimates at 48 kHz."""
    x = np.asarray(audio_stereo, dtype=np.float32)
    if sample_rate != 48000 or x.ndim != 2 or x.shape[1] != 2 or not len(x):
        raise ValueError("預訓練分軌模組需要 48 kHz 雙聲道輸入。")
    if not np.isfinite(x).all():
        raise ValueError("音訊含有無效樣本。")
    if model == "demucs":
        return _demucs_target(x, sample_rate, target)
    if model != "rnnoise" or target not in ("voice", ""):
        raise ValueError("未知的模型或聲源。")
    ready, reason = _rnnoise_status()
    if not ready:
        raise ValueError(reason)
    # When opposite-polarity channels largely cancel at the midpoint, the
    # higher-energy channel is a better guide. The renderer still uses both.
    guide = np.mean(x, axis=1)
    powers = np.mean(x.astype(np.float64) ** 2, axis=0)
    if np.mean(guide.astype(np.float64) ** 2) < 0.04 * float(np.max(powers)):
        guide = x[:, int(np.argmax(powers))]
    speech = _run_neural_filter(guide)
    return speech, guide - speech
