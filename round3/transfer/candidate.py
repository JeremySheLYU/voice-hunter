"""YAMNet transfer head for two separable stereo source images.

The five outputs form a closed vocabulary. Inference never reads the ground
truth number of present classes, their identities or an example of a target.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from round2.semantic.semantic_candidate import candidates
from voicehunter.spatial import _samples

HERE = Path(__file__).resolve().parent
CLASSES = ("guitar", "piano", "speech", "train", "wind")
AUDIOSET_FAMILIES = ((135, 136, 137, 138, 139, 140),
                     (148, 149), (0, 1, 2, 3, 5, 12),
                     tuple(range(322, 329)), (277, 278, 279))
RATE = 48000
# A 127-tap low-pass FIR prevents frequencies above 8 kHz folding into
# YAMNet's 16 kHz waveform input during 3:1 integer decimation.
time_grid = np.arange(-63, 64)
kernel = np.sinc(time_grid * .31) * .31 * np.kaiser(127, 7.5)
LOWPASS = np.asarray(kernel / kernel.sum(), dtype='f4')


class Encoder:
    def __init__(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(HERE / "yamnet.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"])

    def encode(self, signal):
        samples = np.convolve(signal, LOWPASS, mode='same')[::3].astype('f4')
        rms = np.sqrt(np.mean(samples.astype('f8') ** 2))
        if rms > 1e-7:
            samples *= min(.08 / rms, 1 / max(abs(samples).max(), 1e-8))
        scores, latent = self.session.run(["output_0", "output_1"], {"waveform": samples})
        # Weight active windows above silence. Train at the same loudness.
        weights = np.maximum(1e-3, 1.0 - scores[:, 494])
        weights /= np.sum(weights)
        vector = np.sqrt(np.maximum(np.sum(latent * weights[:, None], axis=0), 0))
        vector /= max(np.linalg.norm(vector), 1e-8)
        semantic = np.stack([scores[:, cols].max(axis=1) for cols in AUDIOSET_FAMILIES], axis=1)
        prior = np.sum(semantic * weights[:, None], axis=0)
        return vector, np.log(prior + .0001)


def fit_head(embeddings, labels):
    """Class-balanced kernel ridge regression on fixed unit-norm embeddings."""
    inputs = np.asarray(embeddings, dtype='f8')
    kernel = (inputs @ inputs.T).clip(0, 1) ** 3
    targets = np.eye(len(CLASSES))[labels]
    class_counts = targets.sum(axis=0)
    ridge = .16 * class_counts[labels] / np.median(class_counts)
    alpha = np.linalg.solve(kernel + np.diag(ridge), targets)
    return {"anchors": inputs.astype('f4'), "alpha": alpha.astype('f4')}


def stem_logits(vectors, priors, head, prior_weight):
    kernel = np.clip(vectors @ head["anchors"].T, 0, 1) ** 3
    return kernel @ head["alpha"] * 6 + prior_weight * priors


def select_stem(logits, label, threshold):
    # The model is deliberately calibrated to a requested label, and may abstain.
    centred = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(centred)
    probs /= probs.sum(axis=1, keepdims=True)
    k = CLASSES.index(label)
    idx = int(np.argmax(probs[:, k]))
    return idx if probs[idx, k] >= threshold else None


@dataclass
class Prepared:
    separated: object
    matrix: object
    logits: object
    threshold: float

    def predict(self, label):
        if label not in CLASSES:
            raise ValueError(f"Unsupported label: {label}")
        if self.logits is None:
            return None
        i = select_stem(self.logits, label, self.threshold)
        if i is None:
            return None
        return self.separated[:, i, None] * self.matrix[:, i]


class Predictor:
    def __init__(self):
        self.encoder = Encoder()
        with np.load(HERE / 'head.npz', allow_pickle=False) as f:
            self.head = dict(f)

    def prepare(self, stereo, sr=RATE):
        if sr != RATE:
            raise ValueError("This stereo separator expects 48 kHz.")
        audio = _samples(stereo, stereo=True)
        try:
            mono, matrix, _, _ = candidates(audio)
        except ValueError:
            return Prepared(None, None, None, float(self.head['threshold']))
        representations = [self.encoder.encode(m) for m in mono.T]
        vectors, priors = [np.stack(v) for v in zip(*representations)]
        logits = stem_logits(vectors, priors, self.head, float(self.head['prior_weight']))
        return Prepared(mono, matrix, logits, float(self.head['threshold']))
