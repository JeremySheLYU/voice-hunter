"""OOF multi-model classification of credible stereo source images."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from round3.transformer import model as ast
from round3.apple_features import model as apple
from round3.fusion import model as fusion
from round3.mixture.jointdiag import image_pair
from round3.transfer.candidate import CLASSES,RATE

HERE=Path(__file__).resolve().parent


def softmax(x):
    ex=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return ex/ex.sum(axis=-1,keepdims=True)


def stem_evidence(a,ap,y,p,w):
    a=np.atleast_2d(a)
    u=a-w['ast_center'];u/=np.linalg.norm(u,axis=1,keepdims=True)+.001
    ascore=(.5+.5*(u@w['ast_basis'].T))**2 @ w['ast_coef']+.2
    afam=np.stack([a[:,list(cols)].max(axis=1) for cols in ast.FAMILIES],axis=1)
    apnorm=apple.normalized_features(ap)
    apscore=(apnorm@w['app_basis'].T)**2@w['app_coef']
    app_prior=np.stack([apple.prior(s,w['label_order']) for s in ap])
    ys=np.clip(y@w['yam_basis'].T,0,1)**3 @ w['yam_coef']
    return np.stack((ascore,apscore,ys,softmax(afam),app_prior,softmax(p)),axis=1)


def posterior(x,w,smoothing):
    centered=x-x.mean(axis=2,keepdims=True)
    scaled=(centered / w['scale'][None,:,None]).clip(-6.,6.)
    logits=(scaled*(w['shared'][:,None]+w['individual'])).sum(axis=1)+w['intercept']
    fixed=softmax(5.0*(x[:,0]*.50+x[:,1]*.30+x[:,2]*.20))
    return smoothing*fixed+(1.-smoothing)*softmax(logits)


@lru_cache(maxsize=1)
def load_model():
    with np.load(HERE/'weights.npz',allow_pickle=False) as f:return dict(f)


def predict_scene(audio,sr=48000,model=None):
    if sr!=RATE:raise ValueError('Requires 48 kHz stereo')
    w=load_model() if model is None else model
    try:
        images,quality=image_pair(audio,True)
        if quality[0]>.16 or quality[1]<.05:return dict.fromkeys(CLASSES,None)
        try:
            stems,matrix=fusion.demix(audio);images=[stems[:,i,None]*matrix[None,:,i] for i in (0,1)]
        except ValueError:
            stems=np.stack([v[:,np.argmax(np.mean(v**2,axis=0))] for v in images],axis=1)
        enc=fusion.encoder()
        apples=np.stack([apple.describe(z) for z in stems.T])
        tf=[enc.encode(z) for z in stems.T];y,p=[np.stack(t) for t in zip(*tf)]
        aa=np.stack([ast.encode(s) for s in stems.T])
    except ValueError:return dict.fromkeys(CLASSES,None)
    x=stem_evidence(aa,apples,y,p,w)
    score=posterior(x,w,float(w['smoothing']));out={}
    for j,key in enumerate(CLASSES):
        k=int(score[:,j].argmax());out[key]=images[k] if score[k,j]>=w['thresholds'][j] else None
    return out
