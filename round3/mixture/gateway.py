"""Gate two-source joint diagonalization using the measured spatial fit.

The estimator requires two resolvable and nearly instantaneous stereo images.
It rejects a poor fit before assigning names to images.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from round3.mixture import jointdiag
from round3.mixture.mix_model import predict_raw,confidence
from round3.transfer.candidate import CLASSES,Encoder,HERE as TRANSFER_DIR,stem_logits

HERE=Path(__file__).resolve().parent

@dataclass
class Prepared:
    images: object
    scores: object
    choices: object
    threshold: float

    def predict(self,label):
        k=CLASSES.index(label)
        if self.images is None or self.scores[k]<self.threshold:return None
        return self.images[self.choices[k]]


class Predictor:
    def __init__(self):
        self.encoder=Encoder()
        with np.load(TRANSFER_DIR/'head.npz',allow_pickle=False) as f:self.stem_head=dict(f)
        with np.load(HERE/'mix_head.npz',allow_pickle=False) as f:self.mix_head=dict(f)
        self.config=json.loads((HERE/'gateway_params.json').read_text())

    def prepare(self,stereo,sr=48000):
        if sr!=48000:raise ValueError('Requires 48 kHz audio')
        try:images,quality=jointdiag.image_pair(stereo,True)
        except ValueError:return Prepared(None,None,None,0)
        if quality[0]>.16 or quality[1]<.05:
            return Prepared(None,None,None,0)
        rep=[self.encoder.encode(s[:,np.argmax(np.mean(s**2,axis=0))]) for s in images]
        z,p=[np.stack(xs) for xs in zip(*rep)]
        logit=stem_logits(z,p,self.stem_head,float(self.config['stem_weight']))
        prob=np.exp(logit-logit.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
        weight=self.config['mixture_weight']
        if weight:
            raw,prior=predict_raw(self.encoder,self.mix_head,stereo)
            scene=confidence(raw,prior,self.mix_head)
            scores=(1-weight)*prob.max(axis=0)+weight*scene
        else:scores=prob.max(axis=0)
        return Prepared(images,scores,prob.argmax(axis=0),self.config['threshold'])


@lru_cache(maxsize=1)
def get_predictor():return Predictor()


def predict_scene(audio,sr,unused=None):
    result=get_predictor().prepare(audio,sr)
    return {c:result.predict(c) for c in CLASSES}


def separate(audio,sr,target):
    result=get_predictor().prepare(audio,sr).predict(target)
    if result is None:raise ValueError('Target or spatial contrast insufficient')
    return result
