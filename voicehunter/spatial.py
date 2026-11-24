"""Stereo source-image reconstruction and remixing. Arrays use [sample, channel].

Frames are centred at k * hop (including k=0 and k=ceil(samples/hop)),
zero padded outside the clip and analysed with a periodic square-root Hann.
Spectra have shape [frequency, frame, channel]. Masks have shape [frequency,
frame]; each mask value is shared by L and R. This retains the mixture's
interchannel amplitude and phase relationships within each selected FFT bin.
When two sources share a bin, neither source's spatial cues are guaranteed.
"""

import math

import numpy as np


_BLOCK_FRAMES = 128


def _samples(audio, *, stereo=False):
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1 and not stereo:
        samples = samples[:, None]
    if samples.ndim != 2 or not len(samples):
        raise ValueError("Audio must have shape [samples, channels], with at least one sample.")
    if samples.shape[1] not in ((2,) if stereo else (1, 2)):
        raise ValueError("Expected stereo audio." if stereo else "Expected one or two channels.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples.")
    return samples


def _window(n_fft, hop):
    if not isinstance(n_fft, (int, np.integer)) or not isinstance(hop, (int, np.integer)):
        raise ValueError("FFT size and hop must be integers.")
    if n_fft < 4 or n_fft % 2 or not 1 <= hop <= n_fft // 2:
        raise ValueError("Use an even FFT size >= 4 and a hop between 1 and half the FFT size.")
    return np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)


def _frames(samples, n_fft, hop):
    """View of centred frames, shaped [frame, channel, sample within frame]."""
    trailing = (-len(samples)) % hop
    padded = np.pad(samples, ((n_fft // 2, n_fft // 2 + trailing), (0, 0)))
    return np.lib.stride_tricks.sliding_window_view(padded, n_fft, axis=0)[::hop]


def _spectrum(frames, window):
    return np.fft.rfft(frames * window, axis=-1)


def stft(audio, n_fft=2048, hop=512):
    """Compute a complex [frequency, frame, channel] STFT for mono or stereo input."""
    samples = _samples(audio)
    window = _window(n_fft, hop)
    frames = _frames(samples, n_fft, hop)
    out = np.empty((n_fft // 2 + 1, len(frames), samples.shape[1]), dtype=np.complex64)
    for start in range(0, len(frames), _BLOCK_FRAMES):
        stop = min(start + _BLOCK_FRAMES, len(frames))
        out[:, start:stop] = _spectrum(frames[start:stop], window).transpose(2, 0, 1)
    return out


def mask_from_estimates(target_mono, residual_mono, *, n_fft=2048, hop=512):
    """Soft power-ratio mask from sample-aligned estimates at the mixture rate.

    Accepts mono models' estimates or two-channel estimates; in the latter
    case channel powers are averaged to avoid cancellation when deriving a mask.
    """
    target = _samples(target_mono)
    residual = _samples(residual_mono)
    if len(target) != len(residual):
        raise ValueError("Target and residual must have equal sample counts.")
    window = _window(n_fft, hop)
    target_frames = _frames(target, n_fft, hop)
    residual_frames = _frames(residual, n_fft, hop)
    mask = np.zeros((n_fft // 2 + 1, len(target_frames)), dtype=np.float32)
    for start in range(0, len(target_frames), _BLOCK_FRAMES):
        stop = min(start + _BLOCK_FRAMES, len(target_frames))
        target_power = (np.abs(_spectrum(target_frames[start:stop], window)) ** 2).mean(axis=1)
        residual_power = (np.abs(_spectrum(residual_frames[start:stop], window)) ** 2).mean(axis=1)
        total = target_power + residual_power
        np.divide(target_power, total, out=mask[:, start:stop].T, where=total > 0)
    return mask


def extract_stereo(audio, mask, *, n_fft=2048, hop=512):
    """Estimate target and residual stereo images from a shared [0, 1] mask.

    Both arrays have the input length and original time alignment. Residual is
    the sample-wise difference so neutral recombination is accurate to float
    rounding. Transform temporaries are bounded even on long clips.
    """
    samples = _samples(audio, stereo=True)
    window = _window(n_fft, hop)
    frames = _frames(samples, n_fft, hop)
    mask = np.asarray(mask, dtype=np.float32)
    expected = (n_fft // 2 + 1, len(frames))
    if mask.shape != expected:
        raise ValueError(f"Expected mask shape {expected}, received {mask.shape}.")
    if not np.isfinite(mask).all() or np.min(mask) < 0 or np.max(mask) > 1:
        raise ValueError("A separation mask must contain finite values between 0 and 1.")

    padded_length = n_fft + (len(frames) - 1) * hop
    target = np.zeros((padded_length, 2), dtype=np.float32)
    normalization = np.zeros(padded_length, dtype=np.float32)
    squared_window = window * window
    for start in range(0, len(frames), _BLOCK_FRAMES):
        stop = min(start + _BLOCK_FRAMES, len(frames))
        spectrum = _spectrum(frames[start:stop], window)
        spectrum *= mask[:, start:stop].T[:, None, :]
        block = np.fft.irfft(spectrum, n=n_fft, axis=-1) * window
        for index, frame in enumerate(block, start):
            offset = index * hop
            target[offset:offset + n_fft] += frame.T
            normalization[offset:offset + n_fft] += squared_window
    interior = slice(n_fft // 2, n_fft // 2 + len(samples))
    target = target[interior]
    target /= normalization[interior, None]
    return target, samples - target


def remix(target, residual, *, target_gain_db=0.0, pan=0.0, width=1.0):
    """Apply target gain, relative target pan and residual mid/side width.

    Pan -1/+1 moves the entire target to the left/right. Zero retains the
    target's stereo image. It is a constant-sum crossfeed: folding to mono
    gives the same level at every pan. Side gain applies only to the residual,
    giving a mono background at 0, its captured width at 1, and extra width up
    to 2. It cannot add stereo cues to an identical-channel residual.
    Output is unclipped float32. Apply peak headroom on final audition/export.
    """
    target = _samples(target, stereo=True)
    residual = _samples(residual, stereo=True)
    if target.shape != residual.shape:
        raise ValueError("Target and residual stereo arrays must have equal length.")
    if not math.isfinite(pan) or abs(pan) > 1:
        raise ValueError("Pan must be between -1 (left) and +1 (right).")
    if not math.isfinite(width) or not 0 <= width <= 2:
        raise ValueError("Background width must be between 0 and 2.")
    if not math.isfinite(target_gain_db) or not -120 <= target_gain_db <= 36:
        raise ValueError("Target gain must be between -120 and +36 dB.")

    out = target * np.float32(10 ** (target_gain_db / 20))
    if pan:
        farther, nearer = (0, 1) if pan > 0 else (1, 0)
        transferred = abs(pan) * out[:, farther]
        out[:, farther] -= transferred
        out[:, nearer] += transferred
    out += residual
    if width != 1:
        side_change = ((width - 1) / 2) * (residual[:, 0] - residual[:, 1])
        out[:, 0] += side_change
        out[:, 1] -= side_change
    return out


def peak_safe(audio, ceiling_db=-1.0):
    """Return (float32 audio, linear gain) with optional whole-clip peak trim.

    Only attenuates, equally in both channels. This is sample-peak headroom,
    not a true-peak limiter or a loudness matcher.
    """
    samples = _samples(audio, stereo=True)
    if not math.isfinite(ceiling_db) or not -60 <= ceiling_db <= 0:
        raise ValueError("Peak ceiling must be between -60 and 0 dBFS.")
    peak = float(np.max(np.abs(samples)))
    ceiling = 10 ** (ceiling_db / 20)
    gain = min(1.0, ceiling / peak) if peak else 1.0
    return (samples if gain == 1 else samples * np.float32(gain)), gain
