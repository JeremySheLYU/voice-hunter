"""Map frozen candidates with heterogeneous inference APIs to the stress runner's protocol."""
import os
from round3.evaluation.evaluate import CLASSES, predictor


def load_model():
    return predictor(os.environ["VOICEHUNTER_STRESS_CANDIDATE"])


def predict_scene(audio, sr, model):
    if sr != 48000:
        raise ValueError("This benchmark runs at 48 kHz")
    return model(audio, CLASSES)
