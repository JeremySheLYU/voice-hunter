"""Evaluate optional transfer recognition after a blind spatial separation.

The target is never used to estimate source images. The direct-pan route is
used only when channel phases are consistent, else a convolutive route is
attempted. No location, clean stem or target prevalence is provided.
"""
from dataclasses import dataclass

import numpy as np

from round3.transfer.candidate import Encoder, RATE, CLASSES, HERE as HEAD_DIR, stem_logits, select_stem
from round3.mixture import model
from round2.semantic.semantic_candidate import candidates


def estimate_images(x, sr):
    if sr!=RATE:raise ValueError('48 kHz input required')
    spec=model.stft(x,1024,384)
    product=spec[:,:,0] * np.conj(spec[:,:,1])
    phase_spread=np.sum(abs(product.imag))/max(1e-12,np.sum(abs(product)))
    if phase_spread < .18:
        try:
            mono,matrix,angles,focus=candidates(x)
            return np.stack([mono[:,i,None]*matrix[None,:,i] for i in range(2)]), 'direct'
        except ValueError:
            pass
    return model.demix(x), 'room'


@dataclass
class Result:
    images: object
    logits: object
    threshold: float
    method: str

    def predict(self,label):
        if self.images is None:return None
        index=select_stem(self.logits,label,self.threshold)
        return None if index is None else self.images[index]


class Predictor:
    def __init__(self,head=None):
        self.encoder=Encoder()
        with np.load(head or HEAD_DIR/'head.npz', allow_pickle=False) as f:
            self.head=dict(f)

    def prepare(self,stereo,sr=RATE):
        try:
            images,method=estimate_images(stereo,sr)
        except ValueError:
            return Result(None,None,float(self.head['threshold']),'no contrast')
        vectors=[];priors=[]
        for image in images:
            signal=image[:,np.argmax(np.mean(image**2,axis=0))]
            vec,prior=self.encoder.encode(signal)
            vectors.append(vec);priors.append(prior)
        logits=stem_logits(np.stack(vectors),np.stack(priors),self.head,float(self.head['prior_weight']))
        return Result(images,logits,float(self.head['threshold']),method)
