"""Estimate a target's contribution to an existing stereo recording.

Example mode learns nonnegative spectral dictionaries from the current clip.
Auto mode uses Apple's local classifier to choose examples and guide the same
separator in time. Its neural model detects events; its waveform separation is
statistical. Both modes apply one shared real mask to the two original channels.
"""
from __future__ import annotations

import atexit
from functools import lru_cache
import json
import platform
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

from .spatial import extract_stereo, stft


N_FFT = 2048
HOP = 512
EPS = 1e-8
EVENT_GROUPS = {
    "speech": ["speech", "shout", "whispering"],
    "piano": ["piano", "electric_piano"],
    "guitar": ["guitar", "electric_guitar", "bass_guitar", "acoustic_guitar",
               "steel_guitar_slide_guitar", "guitar_tapping", "guitar_strum"],
    "music": ["music"],
    "wind": ["wind", "wind_rustling_leaves"],
    "wind_mic": ["wind_noise_microphone"],
    "train": ["rail_transport", "train", "train_whistle", "train_horn",
              "railroad_car", "train_wheels_squealing"],
}


_build_lock = threading.Lock()


@lru_cache(maxsize=1)
def _event_helper():
    """Cache one compiled native helper per process, inside a private temporary directory."""
    if platform.system() != "Darwin" or not shutil.which("clang"):
        raise RuntimeError("自動辨識需要 macOS 和 Apple 命令列開發工具，可改用示範片段模式。")
    tmp = tempfile.TemporaryDirectory(prefix="voicehunter-native-")
    binary = Path(tmp.name) / "events"
    source = Path(__file__).with_name("sound_events.m")
    try:
        cmd = ["/usr/bin/clang", "-fobjc-arc", "-O2", "-fmodules-cache-path=" + str(Path(tmp.name) / "modules"),
               "-framework", "Foundation", "-framework", "CoreMedia", "-framework", "SoundAnalysis",
               str(source), "-o", str(binary)]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        probe = subprocess.run([str(binary), "--labels"], capture_output=True, check=True, timeout=20)
        if "speech" not in json.loads(probe.stdout):
            raise RuntimeError("系統聲音分類模型未提供所需的標籤。")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as err:
        tmp.cleanup()
        raise RuntimeError("無法啟動 Apple 聲音辨識，請檢查命令列開發工具，或改用示範片段。") from err
    atexit.register(tmp.cleanup)
    return binary


@lru_cache(maxsize=1)
def auto_available():
    """Probe the native classifier once; unsupported platforms retain example mode."""
    try:
        with _build_lock:
            _event_helper()
        return True
    except RuntimeError:
        return False


def _native_windows(audio, sample_rate, target):
    """Run the system classifier locally. Each recording uses a private temporary file."""
    groups = [target] if isinstance(target, str) else target
    if not isinstance(groups, (tuple, list)) or not groups or any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError("請選擇要辨識的聲音類別。")
    labels = list(dict.fromkeys(label for group in groups for label in EVENT_GROUPS.get(group, [group])))
    with _build_lock:
        binary = _event_helper()
    with tempfile.TemporaryDirectory(prefix="voicehunter-events-") as tmp:
        wav = Path(tmp) / "input.wav"
        mono = np.mean(audio, axis=1)
        if np.linalg.norm(mono) < 0.08 * max(np.linalg.norm(audio[:, i]) for i in range(2)):
            mono = audio[:, int(np.argmax(np.mean(audio ** 2, axis=0)))]
        with wave.open(str(wav), "wb") as dest:
            dest.setparams((1, 2, int(sample_rate), 0, "NONE", "not compressed"))
            dest.writeframes((np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes())
        result = subprocess.run([str(binary), str(wav), *labels], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Apple 模型無法辨識此錄音或不支援所選類別，請改用示範片段。")
        return json.loads(result.stdout)


def _span_frames(span, times, duration, label):
    if span is None or len(span) != 2:
        raise ValueError(f"請選擇{label}的示範區間（秒）。")
    start, end = map(float, span)
    if not (0 <= start < end <= duration + 1e-6):
        raise ValueError(f"{label}的起點和終點必須在錄音範圍內。")
    if end - start < 0.15 - 1e-8:
        raise ValueError(f"{label}示範片段至少需要 0.15 秒。")
    frames = (times >= start) & (times < end)
    if frames.sum() < 3:
        raise ValueError(f"請選較長的{label}示範片段。")
    return frames


def _dot(left, right):
    # Some macOS Accelerate kernels set spurious CPU exception flags even for
    # finite GEMM results. Check the product itself before continuing.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        result = np.dot(left, right)
    if not np.isfinite(result).all():
        raise ArithmeticError("Spectral fitting produced a non-finite matrix.")
    return result


def _dictionary(features, select, rank=7):
    """Small NMF with deterministic, time-distributed initialization."""
    examples = features[:, select]
    levels = np.linalg.norm(examples, axis=0)
    if not levels.size or levels.max() < EPS:
        raise ValueError("示範片段沒有可分析的聲音，請改選有聲音的區間。")
    active = np.flatnonzero(levels > max(EPS, levels.max() * 0.015))
    if active.size < 3:
        raise ValueError("示範片段內的可聽聲音過短，請延長區間。")
    active = active[np.linspace(0, active.size - 1, min(240, active.size), dtype=int)]
    v = np.ascontiguousarray(examples[:, active] / (levels[active] + EPS))
    atoms = min(rank, max(1, v.shape[1] // 4))
    indices = np.linspace(0, v.shape[1] - 1, atoms, dtype=int)
    d = np.ascontiguousarray((v[:, indices] + 0.03 * v.mean(axis=1, keepdims=True)) + 1e-6)
    d /= np.maximum(np.linalg.norm(d, axis=0), EPS)
    h = np.maximum(_dot(d.T, v), 1e-6)
    for _ in range(28):
        h *= _dot(d.T, v) / np.maximum(_dot(_dot(d.T, d), h), EPS)
        d *= _dot(v, h.T) / np.maximum(_dot(d, _dot(h, h.T)), EPS)
        norms = np.maximum(np.linalg.norm(d, axis=0), EPS)
        d /= norms
        h *= norms[:, None]
    return d


def _source_mask(features, target_dict, bg_dict, probability=None):
    dictionary = np.ascontiguousarray(np.column_stack([target_dict, bg_dict]))
    target_rank = target_dict.shape[1]
    gram = _dot(dictionary.T, dictionary)
    out = np.empty_like(features)
    for left in range(0, features.shape[1], 384):
        right = min(left + 384, features.shape[1])
        activation = np.maximum(_dot(dictionary.T, features[:, left:right]), EPS)
        h = activation.copy()
        for _ in range(42):
            h *= activation / np.maximum(_dot(gram, h), EPS)
        a = _dot(target_dict, h[:target_rank])
        b = _dot(bg_dict, h[target_rank:])
        if probability is not None:
            # A gentle, bounded prior. The classifier labels a window, not every
            # frequency in the window. Keep time-frequency evidence in the mix.
            prior = 0.20 + 0.60 * probability[left:right]
            a *= np.sqrt(prior / (1 - prior))[None, :]
        a = np.square(a)
        out[:, left:right] = a / np.maximum(a + np.square(b), EPS)
    return out


def _background_without_span(features, valid, dictionary):
    """Use less target-like frames outside the reference for the other dictionary."""
    indices = np.flatnonzero(valid)
    if indices.size < 6:
        raise ValueError("請縮短目標片段，或另選一段背景示範。")
    spectra = features[:, indices]
    norm = np.linalg.norm(spectra, axis=0)
    # Do not accidentally learn the 'background' dictionary from digital silence.
    audible = norm > max(EPS, float(np.max(norm)) * 0.015)
    if audible.sum() < 3:
        raise ValueError("目標片段以外只剩靜音，請另選錄音或區間。")
    indices = indices[audible]
    similarity = np.max(_dot(dictionary.T, spectra[:, audible]), axis=0) / np.maximum(norm[audible], EPS)
    count = min(indices.size, max(4, int(indices.size * 0.35)))
    return np.sort(indices[np.argsort(similarity)[:count]])


def _automatic_selection(audio, sr, times, target):
    return _selection_from_result(_native_windows(audio, sr, target), times)


def _selection_from_result(result, times):
    windows = result["windows"]
    if len(windows) < 3:
        raise ValueError("錄音過短，請選擇至少 3 秒且有聲音變化的片段，或改用示範模式。")
    score = np.array([w["score"] for w in windows])
    mid = [w["start"] + w["duration"] / 2 for w in windows]
    peak = float(score.max())
    threshold = 0.18
    if peak < threshold:
        raise ValueError(f"沒有找到足夠明確的目標聲音（最高偵測分數 {peak:.2f}）。可改用示範片段。")
    # Exclude the last padded frame beyond the audio duration.
    p = np.interp(times, mid, score).astype(np.float32)
    low = float(np.quantile(p, 0.24))
    if float(p.max()) - low < 0.085:
        raise ValueError("整段錄音的辨識分數太相近，請手動標示目標和背景示範。")
    # Capture brief events too: they may occupy far less than 20% of a long clip.
    high = max(float(np.quantile(p, 0.78)), low + 0.48 * (float(p.max()) - low))
    foreground, background = p >= high, p <= low
    return foreground, background, p, {
        "classifier": "Apple SoundAnalysis v1 (time labels only)",
        "labels": result["labels"],
        "peak_event_score": round(peak, 4),
        "event_windows": [{k: round(float(w[k]), 4) for k in ("start", "duration", "score")} for w in windows],
    }


def separate(audio, sr, *, method="example", target="speech", target_span=None, background_span=None):
    """Return target, additive residual (both [samples, 2]) and JSON-safe metadata.

    In example mode the target interval should contain predominantly the wanted
    sound. An optional separate background interval helps learn competing sounds.
    Automatic mode requires the target to vary over time and local macOS tooling.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2 or not x.shape[0] or not np.isfinite(x).all():
        raise ValueError("需要有效且非空的左右兩聲道音訊。")
    if not isinstance(sr, (int, np.integer)) or not 8000 <= sr <= 192000:
        raise ValueError("支援的取樣率是 8,000 到 192,000 Hz。")
    if np.max(np.abs(x)) < EPS:
        raise ValueError("這段錄音是靜音。")
    duration = len(x) / sr
    mix = stft(x, n_fft=N_FFT, hop=HOP)
    power = (np.abs(mix[:, :, 0]) ** 2 + np.abs(mix[:, :, 1]) ** 2) * 0.5
    del mix
    magnitude = np.sqrt(power, out=power).astype(np.float32, copy=False)
    # Partial spectral whitening improves faint/high-frequency feature matching.
    frequency_scale = np.mean(magnitude, axis=1, keepdims=True)
    frequency_scale = np.sqrt(np.maximum(frequency_scale, max(EPS, magnitude.max() * 1e-4)))
    features = np.ascontiguousarray(magnitude / frequency_scale)
    del magnitude, power
    times = np.arange(features.shape[1]) * HOP / sr
    p = None
    metadata = {"method": method, "target": str(target), "n_fft": N_FFT,
                "hop": HOP, "separator": "clip-trained nonnegative spectral dictionaries / shared stereo mask"}
    if method == "auto":
        if target_span is not None or background_span is not None:
            raise ValueError("如要指定時間區間，請選示範片段模式。")
        fg, bg, p, event_metadata = _automatic_selection(x, sr, times, target)
        metadata.update(event_metadata)
        metadata["note"] = "Apple 模型定位聲音出現的時段，再用這段錄音學習其頻譜特徵並提取左右聲道。"
    elif method == "example":
        fg = _span_frames(target_span, times, duration, "目標")
        bg = None if background_span is None else _span_frames(background_span, times, duration, "背景")
        if bg is not None and max(target_span[0], background_span[0]) < min(target_span[1], background_span[1]):
            raise ValueError("目標和背景示範請選互不重疊的區間。")
        metadata["note"] = "根據示範片段學習目標的頻譜特徵，估計整段錄音中相似的聲音。"
        metadata.update({"target_span_seconds": list(map(float, target_span)),
                         "background_span_seconds": None if background_span is None else list(map(float, background_span))})
    else:
        raise ValueError(f"不支援的分軌模式 {method}。")
    d1 = _dictionary(features, fg)
    if bg is None:
        bg = _background_without_span(features, ~fg, d1)
    d2 = _dictionary(features, bg)
    mask = _source_mask(features, d1, d2, p)
    target_audio, residual = extract_stereo(x, mask, n_fft=N_FFT, hop=HOP)
    return target_audio, residual, metadata
