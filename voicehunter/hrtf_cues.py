"""Symmetric front-horizontal ear-cue profile calibrated from CIPIC HRIRs.

Copyright (c) 2001 The Regents of the University of California. All Rights
Reserved. See models/CIPIC-NOTICE.txt for training-data use and notices.
Only pooled interaural cues are modelled; the near ear stays dry.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def load_profile():
    path = Path(__file__).parent / "models" / "cipic_horizontal_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def relative_phase_delay_us(near, far, rate):
    """Interaural group delay in the lower speech band, measured on actual FIRs."""
    spectrum_length = 4096
    freq = np.fft.rfftfreq(spectrum_length, 1 / rate)
    band = (freq >= 300) & (freq <= 1200)
    cross = np.fft.rfft(far, spectrum_length) * np.conj(np.fft.rfft(near, spectrum_length))
    omega = (2 * np.pi) * freq[band]
    omega -= np.mean(omega)
    phase = np.unwrap(np.angle(cross[band]))
    return float(-1e6 * np.dot(omega, phase) / np.dot(omega, omega))


def measured_far_filter(rate, angle_degrees, *, profile=None):
    """Build minimum-phase far-ear spectral shading and compensate its delay."""
    if profile is None:
        profile = load_profile()
    angles = np.asarray(profile["angles_deg"])
    delays = np.asarray(profile["lowband_delay_us"])
    ild = np.asarray(profile["near_far_ild_db"])
    band_frequencies = np.asarray(profile["bands_hz"])
    band_db = np.array([np.interp(angle_degrees, angles, column) for column in ild.T])
    target_delay_us = np.interp(angle_degrees, angles, delays)
    # Regularized, minimum-phase relative response. Both extreme ends are held
    # constant outside the measured bands instead of extrapolating notches.
    n_fft = max(512, 1 << (round(rate*.024)-1).bit_length())
    freq = np.fft.rfftfreq(n_fft, 1 / rate)
    log_gain = np.interp(np.log(np.maximum(freq, 1)), np.log(band_frequencies), -band_db) * (math.log(10)/20)
    cep = np.fft.irfft(log_gain)
    cep[1:n_fft//2] *= 2
    cep[n_fft//2+1:] = 0
    impulse = np.fft.irfft(np.exp(np.fft.rfft(cep)))[:max(8,round(rate*.004))].copy()
    fade = min(len(impulse), max(3, round(rate*.0005)))
    impulse[-fade:] *= np.linspace(1,0,fade)
    # The minimum-phase shadow already carries some low-frequency delay.
    # Only add the remaining positive differential delay.
    delay = max(0.0, (target_delay_us - relative_phase_delay_us([1.0], impulse, rate)) * rate/1e6)
    integer = math.floor(delay)
    f = delay - integer
    frac_kernel = [-(f-1)*(f-2)*(f-3)/6, f*(f-2)*(f-3)/2,
                   -f*(f-1)*(f-3)/2, f*(f-1)*(f-2)/6]
    return integer, np.convolve(impulse, frac_kernel).astype(np.float32)
