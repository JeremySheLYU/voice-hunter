"""Fuse two locally computed sound representations; only route separable stereo sources."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from round2.semantic.semantic_candidate import CLASSES, RATE, candidates
from round3.apple_features import model as apple
from round3.transfer import candidate as transfer
from voicehunter.spatial import _samples

HERE=Path(__file__).resolve().parent

def covariance_demix(x):
    """Fallback for stationary pan pairs using simultaneous 2x2 covariance diagonalization."""
    data=x[::4].astype('f8')
    size=480
    if len(data)<6*size:raise ValueError('錄音過短。')
    data -= data.mean(axis=0)
    data = data[:len(data)//size*size]
    eig, base=np.linalg.eigh(data.T @ data / len(data))
    if eig[0] < max(eig[1]*.004,1e-12):raise ValueError('方向過近。')
    whiten=base / np.sqrt(eig)
    frames=(data@whiten).reshape(-1,size,2)
    cov=np.einsum('bti,btj->bij',frames,frames)/size
    anis=np.column_stack((cov[:,0,0]-cov[:,1,1],2*cov[:,1,0]))
    energy=cov[:,0,0]+cov[:,1,1]
    keep=energy>np.quantile(energy,.75)*.04
    anis=anis[keep]/(energy[keep,None]+.25)
    eig2, vec=np.linalg.eigh(anis.T @ anis)
    if eig2[1]<1e-8 or eig2[0]>.16*eig2[1]:raise ValueError('兩個方向隨時間不穩定。')
    angle=np.arctan2(vec[1,-1],vec[0,-1])/2
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    matrix=(base*np.sqrt(eig))@rot
    matrix*=np.sign(matrix.sum(axis=0))
    matrix/=np.linalg.norm(matrix,axis=0)
    directions=np.rad2deg(np.arctan2(matrix[1],matrix[0]))
    if min(directions)<-3 or max(directions)>93 or np.ptp(directions)<15:raise ValueError('沒有兩個可靠的左右聲像。')
    matrix=matrix[:,np.argsort(directions)].astype('f4')
    return (x@np.linalg.inv(matrix).T).astype('f4'),matrix


def demix(audio):
    try:
        s,m,_,_=candidates(audio)
    except ValueError:
        s,m=covariance_demix(audio)
    return s,m


@lru_cache(maxsize=1)
def encoder():
    return transfer.Encoder()


@lru_cache(maxsize=1)
def tf_head():
    with np.load(transfer.HERE/'head.npz',allow_pickle=False) as f:
        return dict(f)


@lru_cache(maxsize=1)
def calibration():
    with np.load(HERE/'calibration.npz',allow_pickle=False) as f:
        return dict(f)


def probability(apple_scores, transfer_logits, blend):
    norm=transfer_logits-np.max(transfer_logits,axis=1,keepdims=True)
    exp=np.exp(norm)
    tf=exp/exp.sum(axis=1,keepdims=True)
    return (1.-blend)*apple_scores+blend*tf


def stream_scores(stems):
    model=apple.load()
    stats=[apple.describe(s) for s in stems.T]
    reps=[encoder().encode(s) for s in stems.T]
    vec,priors=map(np.stack,zip(*reps))
    head=tf_head()
    return (np.stack([apple.estimate(stat,model) for stat in stats]),
            transfer.stem_logits(vec,priors,head,float(head['prior_weight'])))


def predict_scene(audio,sr,requests):
    if sr!=RATE:raise ValueError('需要 48 kHz 音訊。')
    if any(k not in CLASSES for k in requests):raise ValueError('未知聲音類別。')
    try:
        stems,matrix=demix(_samples(audio,stereo=True))
        scores=stream_scores(stems)
    except ValueError:
        return {req:None for req in requests}
    params=calibration()
    probs=probability(*scores,float(params['blend']))
    out={}
    for key in requests:
        k=CLASSES.index(key); i=np.argmax(probs[:,k])
        out[key]=stems[:,i,None]*matrix[:,i] if probs[i,k]>=params['threshold'][k] else None
    return out


def separate(audio,sr,target):
    stem=predict_scene(audio,sr,[target])[target]
    if stem is None:raise ValueError('指定聲音沒有足夠證據。')
    return stem
