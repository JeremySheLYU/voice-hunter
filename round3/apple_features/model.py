"""Local five-category classifier using Apple's sound-event scores on separated objects."""
from functools import lru_cache
from pathlib import Path
import json
import subprocess

import numpy as np

from round2.semantic.semantic_candidate import CLASSES, RATE, candidates
from voicehunter.separation import _native_windows, _event_helper, _build_lock, EVENT_GROUPS
from voicehunter.spatial import _samples

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def label_order():
    with _build_lock:
        helper = _event_helper()
    return tuple(sorted(json.loads(subprocess.check_output([str(helper), '--labels']))))


def describe(mono):
    """Mean and upper quantile of class evidence on voiced and non-voiced sounds alike."""
    mono = np.asarray(mono, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono.astype('f8')**2)))
    if not np.isfinite(rms) or rms < 1e-7:
        raise ValueError('音量不足，無法辨識。')
    scale = min(.08/rms, .98/max(float(np.max(abs(mono))), 1e-8))
    sig = np.repeat((mono * scale)[:, None], 2, axis=1)
    labels = label_order()
    windows = _native_windows(sig, RATE, list(labels))['windows']
    if not windows:
        raise ValueError('聲音太短，無法評估。')
    probs = np.array([[w['class_scores'].get(label, 0.) for label in labels] for w in windows])
    return np.stack((probs.mean(axis=0), np.quantile(probs,.85,axis=0)))


def normalized_features(stats):
    stats = np.sqrt(np.maximum(stats, 0.))
    features = stats.reshape((len(stats), -1))
    return features / np.maximum(1e-9, np.linalg.norm(features, axis=1, keepdims=True))


def prior(stats, labels):
    expanded = [set(EVENT_GROUPS.get(name, [name])) for name in CLASSES]
    return np.array([np.max(stats[:,[i for i,label in enumerate(labels) if label in group]]) for group in expanded])


def fit(x, labels, reg):
    basis = normalized_features(x)
    kernel = (basis @ basis.T)**2
    targets = np.eye(len(CLASSES))[labels]
    return basis, np.linalg.solve(kernel + reg*np.eye(len(basis)), targets)


def estimate(stat, model):
    feature = normalized_features(stat[None])[0]
    supervised = ((model['basis'] @ feature)**2) @ model['weights']
    native = prior(stat, tuple(map(str, model['labels'])))
    blend = float(model['blend'])
    return (1.-blend)*supervised + blend*native


@lru_cache(maxsize=1)
def load():
    with np.load(HERE/'weights.npz',allow_pickle=False) as data:
        model = dict(data)
    if tuple(model['labels']) != label_order():
        raise RuntimeError('系統的聲音分類標籤與訓練版本不同。')
    return model


def prepare(audio, sr):
    if sr != RATE:
        raise ValueError('需要 48 kHz 音訊。')
    sources, mixing, _, _ = candidates(_samples(audio, stereo=True))
    return sources, mixing, np.array([describe(part) for part in sources.T])


def predict_scene(audio, sr, requests):
    requests = tuple(requests)
    if any(x not in CLASSES for x in requests):
        raise ValueError('未知聲音類別。')
    try:
        sources, mixing, stats = prepare(audio,sr)
    except ValueError:
        return {req:None for req in requests}
    model = load()
    scores = np.stack([estimate(stat,model) for stat in stats])
    out = {}
    for req in requests:
        k = CLASSES.index(req)
        side = int(np.argmax(scores[:,k]))
        out[req] = sources[:,side,None]*mixing[:,side] if scores[side,k]>=float(model['threshold']) else None
    return out


def separate(audio, sr, target):
    stem = predict_scene(audio, sr, [target])[target]
    if stem is None:
        raise ValueError('此錄音未能可靠分出指定聲音。')
    return stem
