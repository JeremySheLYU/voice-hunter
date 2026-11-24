"""Check lateralisation cues on an impulse and two calibrated sinusoidal tones."""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicehunter.binaural import render_headphones


def run():
    sr = 48_000
    impulse = np.zeros((2048, 2), np.float32)
    impulse[100, :] = 1
    background = np.zeros_like(impulse)
    right = render_headphones(impulse, background, sr, azimuth_deg=45)
    left = render_headphones(impulse, background, sr, azimuth_deg=-45)
    assert np.array_equal(left, right[:, ::-1])
    assert np.array_equal(render_headphones(impulse, background, sr), impulse)
    assert np.array_equal(right[:, 1], impulse[:, 1])
    arrivals = [int(np.flatnonzero(abs(right[:, ch]) > 1e-6)[0]) for ch in (0, 1)]
    expected_delay = sr * 0.0875 / 343 * (math.pi / 4 + math.sqrt(0.5))
    assert abs((arrivals[0] - arrivals[1]) - expected_delay) < 1

    def band_level(frequency):
        values = np.sin(2 * np.pi * frequency * np.arange(sr // 4) / sr).astype(np.float32)
        signal = np.column_stack([values, values])
        y = render_headphones(signal, np.zeros_like(signal), sr, azimuth_deg=45)
        powers = np.mean(y[1000:, :] ** 2, axis=0)
        return 10 * np.log10(powers[0] / powers[1])

    low = band_level(150)
    high = band_level(6000)
    assert abs(low) < 0.3 and high < low - 5
    rng = np.random.default_rng(42)
    ambience = rng.normal(0, 0.05, size=impulse.shape).astype(np.float32)
    assert np.allclose(render_headphones(impulse, ambience, sr, azimuth_deg=45) - right, ambience, atol=1e-7)
    anti = np.column_stack([impulse[:, 0], -impulse[:, 1]])
    try:
        render_headphones(anti, background, sr)
    except ValueError:
        pass
    else:
        raise AssertionError("Renderer accepted a cancelling point source")
    print(f"±45° mirror and neutral pass. Far-ear onset delay {arrivals[0]-arrivals[1]} samples "
          f"at 48 kHz, 150 Hz far/near {low:.2f} dB, 6 kHz {high:.2f} dB.")


if __name__ == "__main__":
    run()
