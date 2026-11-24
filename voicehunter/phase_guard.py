"""Fade a locally cancelling point-source target back to its original stereo."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def phase_threshold():
    path = Path(__file__).with_name("models") / "phase_retention.json"
    return float(json.loads(path.read_text())["threshold"])


def window_power(target, rate):
    window = max(2, round(.04*rate))
    centers = np.arange(0, len(target), max(1, window//2))
    left = np.maximum(centers-window//2, 0)
    right = np.minimum(centers+window//2, len(target))
    mid = target.mean(axis=1, dtype=np.float64)
    channel_power = (target[:,0].astype('f8')**2 + target[:,1].astype('f8')**2)*.5
    def local_mean(signal):
        prefix = np.empty(len(signal)+1)
        prefix[0]=0.
        np.cumsum(signal, out=prefix[1:])
        return (prefix[right]-prefix[left]) / (right-left)
    energy = local_mean(channel_power)
    ratio = local_mean(mid*mid)/np.maximum(energy, 1e-12)
    return centers, energy, ratio


def fallback_curve(target, rate, threshold):
    centers, energy, ratio = window_power(target, rate)
    # Ramp to original stereo over the last 20% of the retention threshold.
    weights = np.clip((threshold-ratio)/(.2*threshold), 0., 1.)
    weights[energy < 1e-7] = 0.
    return np.interp(np.arange(len(target)), centers, weights).astype('f4')
