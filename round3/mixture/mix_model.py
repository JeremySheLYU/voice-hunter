"""Train a multilabel transfer head on generated mixtures of the public train split."""
from functools import lru_cache
import numpy as np
from round3.mixture import model,masker
from round3.transfer.candidate import Encoder, CLASSES
from dataclasses import dataclass


def predict_raw(encoder,head,audio):
    channel=int(np.argmax(np.mean(audio**2,axis=0)))
    z,prior=encoder.encode(audio[:,channel])
    kernel=np.clip(z@head['anchor'].T,0,1)**3
    return kernel@head['alpha'], prior


def confidence(raw,prior,head):
    return raw + head['prior_weight']*prior


@dataclass
class Prepared:
    mixture: object
    scores: object
    threshold: float
    images: object=None

    def predict(self,target):
        k=CLASSES.index(target)
        if self.scores[k]<self.threshold:return None
        if self.images is None:self.images=model.nmf_components(self.mixture,model.params())[0]
        return masker.output(self.mixture,self.images,k,self.scores)


class Predictor:
    def __init__(self):
        self.enc=Encoder()
        with np.load(model.HERE/'mix_head.npz',allow_pickle=False) as f:self.head=dict(f)

    def prepare(self,audio,sr):
        if sr!=model.RATE:raise ValueError('Requires 48 kHz')
        r,p=predict_raw(self.enc,self.head,model._samples(audio,stereo=True))
        return Prepared(audio,confidence(r,p,self.head),float(self.head['threshold']))


@lru_cache(maxsize=1)
def get_predictor():return Predictor()


def separate(audio,sr,target):
    result=get_predictor().prepare(audio,sr).predict(target)
    if result is None:raise ValueError('Target category below validation confidence')
    return result
