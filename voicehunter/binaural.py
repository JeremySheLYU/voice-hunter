"""Horizontal headphone renderer with analytical or CIPIC-calibrated lateral cues."""
import math

import numpy as np

from .spatial import remix, _samples


def _far_ear_filter(sample_rate, angle):
    # Woodworth's delay for a rigid spherical head in its frontal hemisphere.
    head_radius = 0.0875
    speed_of_sound = 343.0
    delay = sample_rate * head_radius / speed_of_sound * (angle + math.sin(angle))
    samples = int(math.floor(delay))
    fraction = delay - samples

    # A DC-normalised one-pole low-pass in parallel with a direct branch yields
    # a bounded high shelf: 0 dB at DC, down to -12 dB at Nyquist for a side source.
    cutoff = 1400.0
    k = math.tan(math.pi * cutoff / sample_rate)
    pole = (1.0 - k) / (1.0 + k)
    first = k / (1.0 + k)
    count = max(2, int(math.ceil(math.log(1e-6) / math.log(pole))) + 1)
    low_pass = np.empty(count)
    low_pass[0] = first
    low_pass[1:] = first * (1 + pole) * pole ** np.arange(count - 1)
    low_pass /= low_pass.sum()
    high_gain = 10.0 ** (-12.0 * math.sin(angle) / 20.0)
    shadow = (1.0 - high_gain) * low_pass
    shadow[0] += high_gain

    # Four-tap causal Lagrange interpolation keeps fractional ITD at low
    # frequencies without wrap-around or external DSP dependencies.
    f = fraction
    interpolation = np.array([-(f-1)*(f-2)*(f-3)/6, f*(f-2)*(f-3)/2,
                              -f*(f-1)*(f-3)/2, f*(f-1)*(f-2)/6])
    return samples, np.convolve(interpolation, shadow).astype(np.float32)


def render_headphones(target, residual, sr, *, azimuth_deg=0, width=1, target_gain_db=0, profile="analytic"):
    """Place a target's mono midpoint at -90..+90 degrees. Return stereo float32.

    Use profile="measured" for CIPIC-calibrated horizontal interaural cues.
    Frames stay aligned with video. The near ear starts at the original onset;
    delay/filter tails past the end are truncated to retain the original length.
    The stereo residual comes through at its original position; width adjusts
    only its side component. With a 0-degree setting both ears receive the
    foreground midpoint. The measured profile crossfades severely cancelling intervals to the original
    stereo foreground. Those intervals keep their captured stereo image.
    """
    target = _samples(target, stereo=True)
    residual = _samples(residual, stereo=True)
    if target.shape != residual.shape:
        raise ValueError("目標和環境音軌必須等長。")
    if not isinstance(sr, (int, np.integer)) or not 8000 <= sr <= 192000:
        raise ValueError("取樣率須在 8 到 192 kHz 之間。")
    if not math.isfinite(azimuth_deg) or abs(azimuth_deg) > 90:
        raise ValueError("耳機聲源角度須在左方 −90° 到右方 +90° 之間。")
    if profile not in ("analytic", "measured"):
        raise ValueError("未識別的耳機聲學設定。")
    source = target.mean(axis=1)
    channel_energy = np.mean(np.square(target), dtype=np.float64)
    mid_energy = np.mean(np.square(source), dtype=np.float64)
    if profile == "analytic" and channel_energy > 1e-14 and mid_energy < 0.03 * channel_energy:
        raise ValueError("這條目標軌的左右聲道接近反相，合成單一聲源會抵銷。請選立體聲模式保留原錄音。")

    if azimuth_deg == 0:
        foreground = np.column_stack([source, source])
    else:
        if profile == "measured":
            from .hrtf_cues import measured_far_filter
            delay, kernel = measured_far_filter(sr, abs(azimuth_deg))
        else:
            delay, kernel = _far_ear_filter(sr, math.radians(abs(azimuth_deg)))
        far_signal = np.zeros_like(source)
        if delay < len(source):
            filtered = np.convolve(source, kernel, mode="full")
            far_signal[delay:] = filtered[:len(source) - delay]
        foreground = np.column_stack([far_signal, source] if azimuth_deg > 0 else [source, far_signal])
    if profile == "measured":
        from .phase_guard import fallback_curve, phase_threshold
        weight = fallback_curve(target, sr, phase_threshold())[:, None]
        foreground = foreground * (1 - weight) + target * weight
    return remix(foreground, residual, width=width, target_gain_db=target_gain_db)
