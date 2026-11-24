"""Reference-conditioned two-channel Wiener filters for stationary source images."""
import json
from pathlib import Path

import numpy as np

from . import spatial


def parameters():
    model = Path(__file__).with_name("models") / "joint_spatial.json"
    data = json.loads(model.read_text(encoding="utf-8"))
    return float(data["temperature"]), float(data["spectral_exponent"])


def _smooth(values, axis):
    """Short triangular average with edge replication."""
    pad = [(0, 0)] * values.ndim
    pad[axis] = (2, 2)
    padded = np.moveaxis(np.pad(values, pad, mode="edge"), axis, 0)
    averaged = (padded[:-4] + 2*padded[1:-3] + 3*padded[2:-2] + 2*padded[3:-1] + padded[4:]) / 9
    return np.moveaxis(averaged, 0, axis)


def _covariance(left, right):
    return np.abs(left)**2, np.abs(right)**2, left * np.conj(right)


def reference_templates(spectrum, spans, sample_rate, hop):
    """Return unit-trace per-frequency covariances, or reject weak spatial evidence."""
    times = np.arange(spectrum.shape[1]) * (hop / sample_rate)
    templates = []
    for start, end in spans:
        subset = spectrum[:, (times >= start) & (times < end)]
        if subset.shape[1] < 3:
            return None
        cov = np.stack([_smooth(v.mean(axis=1), 0) for v in _covariance(subset[:,:,0], subset[:,:,1])], axis=1)
        energy = (cov[:, 0] + cov[:, 1]).real
        energy_sum = float(energy.sum())
        if energy_sum <= 1e-15:
            return None
        # Weakly-excited frequency bands borrow the broadband image, including
        # its sign; well-supported bins retain their observed interchannel phase.
        ridge = max(float(energy.max()) * 1e-5, 1e-20)
        cov = (cov + ridge * (cov.sum(axis=0) / energy_sum)) / (energy[:,None] + ridge)
        l, r, c = cov.T
        dominance = np.sqrt((l.real-r.real)**2 + 4*np.abs(c)**2)
        if float(np.average(dominance, weights=energy)) < 0.88:
            return None
        templates.append(cov)
    for i, first in enumerate(templates):
        for second in templates[:i]:
            # Trace inner product is one for identical coherent unit-trace images.
            overlap = (first[:,0]*second[:,0] + first[:,1]*second[:,1]
                       + 2*first[:,2]*np.conj(second[:,2])).real
            if np.median(1 - overlap) < 0.035:
                return None
    return templates


def reconstruct(spectrum, masks, templates, length, n_fft, hop, temperature, exponent):
    """Blockwise 2x2 multichannel Wiener reconstruction with an unassigned bed."""
    window = spatial._window(n_fft, hop)
    frame_count = spectrum.shape[1]
    padded_length = n_fft + (frame_count - 1) * hop
    outputs = [np.zeros((padded_length, 2), np.float32) for _ in masks]
    norm = np.zeros(padded_length, np.float32)
    squared = window**2
    for start in range(0, frame_count, 80):
        stop = min(frame_count, start+80)
        # The smoothing has two frames of support on each side. Supply those
        # frames at every block boundary so the result is independent of chunks.
        low = max(0, start-2)
        high = min(frame_count, stop+2)
        block = spectrum[:, low:high]
        selection = slice(start-low, stop-low)
        ll, rr, lr = [_smooth(_smooth(v, 1), 0)[:,selection] for v in _covariance(block[:,:,0], block[:,:,1])]
        trace = np.maximum(ll+rr, 1e-15)
        distances = []
        for cov in templates:
            a, b, c = cov.T
            aligned = a.real[:,None]*ll + b.real[:,None]*rr + 2*(np.conj(c)[:,None]*lr).real
            distances.append(np.clip(1-aligned/trace, 0, 1))
        nearest = np.minimum.reduce(distances)
        priors = []
        for mask, distance in zip(masks, distances):
            spectral = mask[:, start:stop]
            priors.append((0.025+spectral)**exponent * np.exp(-(distance-nearest)/temperature))
        total_prior = np.maximum(np.add.reduce(priors), 1e-8)
        unexplained = 0.008 + 0.65*np.maximum(0, nearest-0.045)
        lsum, rsum = unexplained.copy(), unexplained.copy()
        csum = np.zeros_like(lsum, dtype=np.complex64)
        for prior, cov in zip(priors, templates):
            prior /= total_prior
            a, b, c = cov.T
            lsum += prior*a.real[:,None]
            rsum += prior*b.real[:,None]
            csum += prior*c[:,None]
        inv_det = 1 / np.maximum(lsum*rsum-np.abs(csum)**2, 1e-8)
        x = spectrum[:,start:stop]
        left = (rsum*x[:,:,0] - csum*x[:,:,1]) * inv_det
        right = (lsum*x[:,:,1] - np.conj(csum)*x[:,:,0]) * inv_det
        for dest, cov, prior in zip(outputs, templates, priors):
            a, b, c = cov.T
            y = np.stack((prior*(a.real[:,None]*left + c[:,None]*right),
                          prior*(np.conj(c[:,None])*left + b.real[:,None]*right)),axis=-1)
            frames = np.fft.irfft(y, n=n_fft, axis=0).transpose(1, 0, 2)
            for index, frame in enumerate(frames, start):
                offset = index*hop
                dest[offset:offset+n_fft] += frame*window[:,None]
        for index in range(start, stop):
            norm[index*hop:index*hop+n_fft] += squared
    keep = slice(n_fft//2, n_fft//2 + length)
    cropped = [out[keep] for out in outputs]
    for output in cropped:
        output /= norm[keep,None]
    return cropped
