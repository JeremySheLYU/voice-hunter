"""Two-speaker, example-guided stereo separation for fixed microphone/source positions.

A short solo example of each talker is needed. A calibrated two-channel spatial
filter extracts speaker A. The second track contains B plus any remaining room
sound. This is a spatial mixing estimate, not voice identification.
"""
import numpy as np

from .spatial import stft


def _inverse_stft(spectrum, n_samples, n_fft, hop):
    """Invert the centred, square-root Hann transform of voicehunter.spatial."""
    window = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)
    samples = n_fft + (spectrum.shape[1] - 1) * hop
    summed = np.zeros((samples, spectrum.shape[2]), dtype=np.float32)
    scale = np.zeros(samples, dtype=np.float32)
    square = np.square(window)
    for left in range(0, spectrum.shape[1], 128):
        subset = spectrum[:, left:left + 128, :]
        frames = np.fft.irfft(subset, n=n_fft, axis=0).transpose(1, 0, 2)
        for offset, frame in enumerate(frames):
            start = (left + offset) * hop
            summed[start:start + n_fft] += frame * window[:, None]
            scale[start:start + n_fft] += square
    keep = slice(n_fft // 2, n_fft // 2 + n_samples)
    return summed[keep] / np.maximum(scale[keep, None], 1.e-8)


def _calibration_span(pair, sample_rate, samples, label):
    if pair is None or len(pair) != 2:
        raise ValueError(f"請選一段只有 {label} 說話的片段。")
    begin, end = map(float, pair)
    if not 0 <= begin < end <= samples / sample_rate + 1e-6 or end - begin < 0.35:
        raise ValueError(f"{label} 的單獨參考片段至少需 0.35 秒，且須在錄音範圍內。")
    return int(round(begin * sample_rate)), min(samples, int(round(end * sample_rate)))


def _steering(solo, n_fft, hop):
    # Estimate a rank-one spatial image via the leading complex covariance
    # eigenvector. Its arbitrary common phase cancels in the projection.
    spec = stft(solo, n_fft, hop)
    if np.max(np.abs(solo)) < 1e-5:
        raise ValueError("參考片段太安靜，請選只有一人清楚說話的時段。")
    covariance = np.einsum('ftc,ftd->fcd', spec, spec.conj(), optimize=True)
    # Smooth only over adjacent bins, weighted by energy for stability in gaps.
    pad = np.pad(covariance, ((1, 1), (0, 0), (0, 0)), mode='edge')
    covariance = (pad[:-2] + 2 * pad[1:-1] + pad[2:]) / (4 * spec.shape[1])
    vals, vecs = np.linalg.eigh(covariance)
    power = np.maximum(vals[:, 1], 0.)
    coherence = np.maximum(0., vals[:, 1] - vals[:, 0]) / np.maximum(power, 1.e-12)
    return vecs[:, :, 1], coherence, power


def separate_speakers(audio, sr, *, target_span, background_span):
    """Return A estimate, additive B/ambient residual, diagnostics, at input rate."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("請使用可讀取的左右雙聲道音訊。")
    if not isinstance(sr, (int, np.integer)) or not 8000 <= sr <= 192000:
        raise ValueError("取樣率須在 8 至 192 kHz 之間。")
    span_a = _calibration_span(target_span, sr, len(x), "A")
    span_b = _calibration_span(background_span, sr, len(x), "B")
    if max(span_a[0], span_b[0]) < min(span_a[1], span_b[1]):
        raise ValueError("A 和 B 的單獨參考段不能重疊。")

    n_fft = max(256, int(2 ** round(np.log2(sr * 0.032))))
    hop = n_fft // 4
    a, coh_a, pa = _steering(x[slice(*span_a)], n_fft, hop)
    b, coh_b, pb = _steering(x[slice(*span_b)], n_fft, hop)
    # The projection of A onto the nullspace of B gives a demixing vector.
    # norm2 near zero means the two images cannot be distinguished reliably.
    perpendicular = a - b * np.sum(b.conj() * a, axis=1)[:, None]
    norm2 = np.sum(abs(perpendicular) ** 2, axis=1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    speech_band = (freqs >= 100.) & (freqs <= 6500.)
    weight = np.sqrt(pa * pb) * speech_band
    if weight.sum() <= 1.e-12:
        raise ValueError("語音頻帶太弱，請重選兩段有清楚人聲的片段。")
    contrast = float(np.average(norm2, weights=weight))
    coherence = float(np.average(np.minimum(coh_a, coh_b), weights=weight))
    if contrast < 0.025 or coherence < 0.36:
        raise ValueError("兩段聲音的左右位置太接近，或反射太強。請重選各自單獨、位置穩定的例段，或改用「範例分離」。")

    ridge = 0.004 + (1 - np.minimum(coh_a, coh_b)) * 0.05
    reliability = (norm2 / (norm2 + 0.018)) * np.minimum(coh_a, coh_b)
    # Where spatial cues are weak, shrink towards a conservative spectral
    # ratio estimated from the two examples instead of amplifying cancellation.
    a_share = pa / np.maximum(pa + pb, 1.e-12)
    demix = perpendicular.conj() / (norm2 + ridge)[:, None]
    proj = a[:, :, None] * demix[:, None, :]
    proj *= reliability[:, None, None]
    proj += np.eye(2, dtype=np.float32)[None, :, :] * ((1 - reliability) * a_share)[:, None, None]

    # Bound memory on long recordings. Only retain the interior of each block
    # after OLA so its borders use the same true samples as an unchunked STFT.
    target = np.empty_like(x)
    block_size = hop * max(256, int(sr * 8 / hop))
    for left in range(0, len(x), block_size):
        right = min(left + block_size, len(x))
        start, stop = max(left - n_fft, 0), min(right + n_fft, len(x))
        spectra = stft(x[start:stop], n_fft, hop)
        separated = np.einsum('fij,ftj->fti', proj, spectra, optimize=True)
        segment = _inverse_stft(separated, stop - start, n_fft=n_fft, hop=hop)
        target[left:right] = segment[left - start:right - start]
    return target, x - target, {
        "method": "speakers", "target": "A", "separator": "two-example regularized stereo spatial demixer",
        "target_span_seconds": list(map(float, target_span)),
        "background_span_seconds": list(map(float, background_span)),
        "spatial_contrast": round(contrast, 4), "reference_coherence": round(coherence, 4),
        "note": "已按參考段的左右聲像分出 A。另一軌包含 B 與其餘環境聲。移動、同方向與強反射會增加串音。",
    }


def separate_moving_speakers(audio, sr, *, target_span, background_span):
    """Reassign the fixed spatial tracks using enrollment-voice similarity."""
    x = np.asarray(audio, dtype=np.float32)
    target, residual, metadata = separate_speakers(x, sr, target_span=target_span, background_span=background_span)
    span_a = _calibration_span(target_span, sr, len(x), "A")
    span_b = _calibration_span(background_span, sr, len(x), "B")
    # In a solo reference, the wrong output should carry considerably less
    # energy than the intended one. If this basic condition fails, the tracks
    # cannot provide reliable signals for an exchange of voice identities.
    isolation = []
    for bounds, correct, other in [(span_a, target, residual), (span_b, residual, target)]:
        segment = slice(*bounds)
        wanted = np.sum(correct[segment] ** 2, dtype=np.float64)
        leak = np.sum(other[segment] ** 2, dtype=np.float64)
        isolation.append(float(10 * np.log10((wanted + 1e-16) / (leak + 1e-16))))
    if min(isolation) < 8.0:
        raise ValueError("兩人的單獨參考段在分軌後仍有較多串音，難以可靠追蹤身分。請改選更乾淨、較長的單人片段，或使用範例分離。")

    from .speaker_identity import follow_reference_voices
    target, identity = follow_reference_voices(x, target, sr, span_a, span_b)
    metadata.update(identity)
    metadata.update(method="speakers_moving",
                    separator="two-example stereo demixer with trained voice-distance reassignment",
                    enrollment_separation_db=[round(value, 2) for value in isolation],
                    note="已比對參考人聲與兩個方向的輸出。兩人音色接近、共用同一方位或強反射時仍可能跟錯人。")
    return target, x - target, metadata
