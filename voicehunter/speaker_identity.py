"""Compare unnamed acoustic tracks with two short, user-selected voice references."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

RATE = 8000
FRAME = 320
STEP = 80


@lru_cache(maxsize=16)
def _kernel(sr):
    radius = max(6, int(np.ceil(5 * sr / RATE)))
    axis = np.arange(-radius, radius+1)
    taps = np.sinc(axis * (.94 * RATE/sr)) * np.hamming(len(axis))
    return taps / taps.sum()


def _analysis_audio(x, sr):
    if x.ndim == 2:
        # Evaluate the stronger microphone; opposed channels may cancel at mid.
        x = x[:, int(np.argmax(np.sum(x*x, axis=0, dtype=np.float64)))]
    if sr != RATE:
        if sr > RATE:
            x = np.convolve(x, _kernel(sr), mode='same')
        x = np.interp(np.arange(round(len(x)*RATE/sr))*(sr/RATE), np.arange(len(x)), x)
    return np.ascontiguousarray(x, dtype='f4')


@lru_cache(maxsize=1)
def _filters():
    hz = np.fft.rfftfreq(512, 1/RATE)
    edges = 700 * np.expm1(np.linspace(np.log1p(65/700), np.log1p(3850/700), 26))
    mel = []
    for low, mid, high in zip(edges,edges[1:],edges[2:]):
        weights = np.maximum(0.,np.minimum((hz-low)/(mid-low), (high-hz)/(high-mid)))
        mel.append(weights/max(weights.sum(),1e-8))
    dct = np.cos(np.pi/24 * (np.arange(24)+.5)[:,None] * np.arange(1,14)[None,:])
    return np.asarray(mel, dtype='f4').T, (dct/np.sqrt(12)).astype('f4')


def frames(x, sr):
    """Rows of 10 ms local energy, voiced periodicity, pitch, and 13 cepstra."""
    signal = _analysis_audio(x,sr)
    if len(signal)<FRAME:
        signal = np.pad(signal,(0,FRAME-len(signal)))
    y = np.lib.stride_tricks.sliding_window_view(signal,FRAME)[::STEP]
    y = y - y.mean(axis=1, keepdims=True)
    power = np.mean(y*y,axis=1)
    mag = abs(np.fft.rfft(y*np.hanning(FRAME).astype('f4'),512))**2
    mel,dct = _filters()
    cep = np.log(np.maximum(mag @ mel, np.maximum(mag.max(axis=1,keepdims=True)*1e-7,1e-12))) @ dct
    ac = np.fft.irfft(abs(np.fft.rfft(y,1024))**2,1024)[:, :128]
    lags = np.arange(18,118)  # 68–444 Hz
    cum = np.pad(np.cumsum(y*y,axis=1), ((0,0),(1,0)))
    energy = np.sqrt(np.maximum(cum[:, FRAME-lags]*(cum[:,-1,None]-cum[:,lags]), 1e-16))
    ncc = np.clip(ac[:,lags]/energy, -1,1)
    peaks = (ncc[:,1:-1] >= ncc[:,:-2]) & (ncc[:,1:-1] > ncc[:,2:])
    validity = peaks & (ncc[:,1:-1] >= ncc.max(axis=1,keepdims=True)*.90)
    choice = np.where(validity.any(axis=1),validity.argmax(axis=1)+1,ncc.argmax(axis=1))
    periodicity = ncc[np.arange(len(choice)),choice]
    lpitch = np.log(RATE/lags[choice])
    return np.column_stack((power,periodicity,lpitch,cep)).astype('f4')


def summarize(f):
    if len(f)<8:
        return None
    power = f[:,0]
    select = power>max(1e-11,float(np.quantile(power,.90))*.012)
    if select.sum()<8:
        return None
    active = f[select]
    voiced = active[active[:,1]>.6]
    if len(voiced)<4:
        return None
    cep = active[:,3:]
    return np.concatenate([np.quantile(voiced[:,2],[.2,.5,.8]),
                           [active[:,1].mean()], np.mean(cep,axis=0), np.std(cep,axis=0)])


@lru_cache(maxsize=1)
def _metric():
    path=Path(__file__).with_name('models')/'speaker-distance.json'
    spec=json.loads(path.read_text(encoding='utf-8'))
    return np.asarray(spec['absolute_feature_weights'], dtype=np.float64)


def follow_reference_voices(mixture, left_track, sr, span_a, span_b):
    """Offline, conservative reassignment of the two spatial outputs.

    This can follow an exchange of the two enrolled acoustic paths. It does
    not constitute general separation of two moving, co-located speakers.
    """
    refs = [summarize(frames(mixture[slice(*span)],sr)) for span in (span_a,span_b)]
    if any(ref is None for ref in refs):
        return left_track, {'reassigned_seconds':0., 'identity_evidence':'weak_reference'}
    weights = _metric()
    dist = lambda a,b: np.sum(weights*np.abs(a-b))
    ref_distance = dist(*refs)
    if ref_distance<.35:
        return left_track, {'reassigned_seconds':0., 'identity_evidence':'similar_references'}
    right_track=mixture-left_track
    # Bound the FFT intermediate size for clips of up to one minute.
    total=len(mixture)/sr
    centers=np.arange(0,total+.10,.18)
    preference=np.full(len(centers), np.nan)
    for block_start in np.arange(0,total,5.4):
        ix=np.flatnonzero((centers >= block_start-1e-7)&(centers < block_start+5.4-1e-7))
        if not len(ix):
            continue
        first=max(0.,block_start-.55)
        end=min(total,block_start+5.95)
        lo,hi=int(round(first*sr)),int(round(end*sr))
        candidates=[frames(s[lo:hi],sr) for s in (left_track,right_track)]
        for i in ix:
            center=centers[i]
            start=max(0,round((center-first-.46)*100))
            stop=round((center-first+.46)*100)
            window=[f[start:stop] for f in candidates]
            if any(len(w)<15 for w in window):
                continue
            strengths=[float(np.mean(w[:,0])) for w in window]
            if min(strengths) < max(1e-10,max(strengths)*.012):
                continue
            vals=[summarize(w) for w in window]
            if any(v is None for v in vals):
                continue
            a,b=refs
            a_votes=(dist(vals[0],a)-dist(vals[0],b))/ref_distance
            b_votes=(dist(vals[1],b)-dist(vals[1],a))/ref_distance
            preference[i]=.5*(a_votes+b_votes)

    # A talker does not switch identity because a syllable is missing. Fit
    # the most likely assignment sequence offline using a two-state path
    # model. A transfer has a fixed cost, so weak or contradictory short
    # windows cannot toggle the output back and forth.
    anchors=np.array([any(a/sr-.10 <= c <= b/sr+.10 for a,b in (span_a,span_b)) for c in centers])
    likelihood=np.clip(np.nan_to_num(preference,nan=0.),-.85,.85)-.035
    costs=np.array([0.,-.70])
    back=np.zeros((len(centers),2),dtype=np.int8)
    for i, evidence in enumerate(likelihood):
        transitions=costs[:,None]-np.array([[0.,.7],[.7,0.]])
        back[i]=np.argmax(transitions,axis=0)
        costs=np.max(transitions,axis=0)+[0.,evidence]
        if anchors[i]:costs[1]=-1e6
    swap=np.zeros(len(centers),dtype=bool)
    state=int(costs.argmax())
    for i in range(len(centers)-1,-1,-1):
        swap[i]=state==1
        state=int(back[i,state])
    info={'reassigned_seconds':0.,'identity_evidence':'no_supported_exchange'}
    if not swap.any():
        return left_track,info
    alpha=np.interp(np.arange(len(mixture))/sr,centers,swap.astype(float)).astype('f4')[:,None]
    info.update(reassigned_seconds=round(float(alpha.sum(dtype=np.float64)/sr),3),
                identity_evidence='exchanged_paths')
    return left_track*(1-alpha)+right_track*alpha,info
