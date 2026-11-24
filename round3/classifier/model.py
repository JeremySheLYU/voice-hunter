"""Five-class local audio classifier for a pair of stationary stereo sources."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from round2.semantic.semantic_candidate import CLASSES, candidates


RATE = 24000
HOP = 240
FFT = 1024
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def transforms():
    hz = np.fft.rfftfreq(FFT, 1 / RATE)
    mel = 2595 * np.log10(1 + hz / 700)
    stops = np.linspace(mel[1], mel[-1], 42)
    ramps = (mel[None, :] - stops[:-1, None]) / np.diff(stops)[:, None]
    filters = np.minimum(ramps[:-1], 1 - ramps[1:]).clip(0)
    filters /= filters.sum(axis=1, keepdims=True).clip(1e-10)
    cep = np.cos(np.pi * (np.arange(40)[None, :] + .5) * np.arange(1, 16)[:, None] / 40)
    return hz, filters.astype('f4'), cep.astype('f4')


@lru_cache(maxsize=1)
def antialias():
    ix = np.arange(-31, 32)
    kernel = .46 * np.sinc(.46 * ix) * np.hamming(len(ix))
    return (kernel / kernel.sum()).astype('f4')


def at_rate(x, sr):
    x = np.asarray(x, dtype='f4')
    if x.ndim != 1 or not len(x):
        raise ValueError("Expected non-empty one-channel signal")
    if sr == RATE:
        return x
    if sr == RATE * 2:
        return np.convolve(x, antialias(), mode='same')[::2]
    # The media backend and all mixture tests explicitly work at 48 kHz.
    raise ValueError("Expected 48 kHz or 24 kHz audio")


def _summarize(a):
    return np.concatenate([np.mean(a, axis=0), np.std(a, axis=0),
                           *np.quantile(a, [.15, .85], axis=0)])


def components(signal):
    """Frame-level tonal, spectral shape, onset and periodicity statistics."""
    signal = np.ascontiguousarray(signal)
    if len(signal) < FFT:
        raise ValueError("Insufficient audio")
    frames = np.lib.stride_tricks.sliding_window_view(signal, FFT)[::HOP]
    hz, filters, cep = transforms()
    mag = np.abs(np.fft.rfft(frames * np.hanning(FFT).astype('f4'))).astype('f4')
    power = mag**2
    levels = power.mean(axis=1)
    if levels.max() < 1e-13:
        raise ValueError("Silent source")
    power /= levels.max()
    bandpower = (power @ filters.T).clip(1e-11)
    bandlog = np.log(bandpower)
    weighted = np.log(np.maximum(bandpower, bandpower.max(axis=1, keepdims=True) * 1e-6))
    profile = weighted - weighted.mean(axis=1, keepdims=True)
    cepstra = profile @ cep.T / 10
    # ACF is normalised to zero-lag energy in each Hann-windowed frame.
    acf = np.fft.irfft(power, n=FFT).astype('f4')
    acf /= acf[:, :1].copy().clip(1e-8)
    lags = np.linspace(np.log(27), np.log(300), 11)
    edges = np.round(np.exp(lags)).astype(int)
    periodicity = np.stack([acf[:, a:b].max(axis=1) for a,b in zip(edges[:-1], edges[1:])], axis=1)
    contrasts = []
    shares = []
    for a,b in ((50,200),(200,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,12001)):
        chunk = power[:, (hz >= a) & (hz < b)]
        avg = chunk.mean(axis=1)
        contrasts.append(np.log((chunk.max(axis=1) + 1e-8) / (avg + 1e-8)))
        shares.append(np.log((avg + 1e-8) / (power.mean(axis=1) + 1e-8)))
    # Peak pitch-class concentrations and how quickly the spectral shape changes
    # complement spectral averages for plucked notes, voices and broadband beds.
    idx = np.flatnonzero((hz >= 150) & (hz <= 4800))
    chroma = np.rint(69 + 12 * np.log2(hz[idx] / 440)).astype(int) % 12
    pitch_energy = np.maximum(mag[:,idx] - .5 * (mag[:,idx-1] + mag[:,idx+1]), 0)
    chroma_power = np.stack([pitch_energy[:,chroma==i].sum(axis=1) for i in range(12)], axis=1)
    chroma_power /= chroma_power.sum(axis=1, keepdims=True).clip(1e-9)
    chroma_sorted = np.sort(chroma_power, axis=1)[:, -4:]
    texture = np.column_stack((periodicity, *contrasts, *shares, chroma_sorted))
    return profile, cepstra, texture, bandlog, levels / levels.max()


def describe_frames(chunks, start, stop):
    profile, cep, texture, bands, amp = (m[start:stop] for m in chunks)
    active = amp > max(2e-6, np.quantile(amp,.85)*.008)
    if active.sum() < 6:
        return None
    result = [_summarize(profile[active]), _summarize(cep[active]), _summarize(texture[active])]
    # Dynamics across real adjacent frames, with silent holds excluded from means.
    for lag in (1, 6, 18):
        delta = abs(bands[lag:] - bands[:-lag])
        use = active[lag:] & active[:-lag]
        valid = delta[use] if np.any(use) else np.zeros((1,bands.shape[1]))
        result.extend((valid.mean(axis=0), np.quantile(valid, .85, axis=0)))
    envelope = np.log(amp.clip(1e-7))
    result.append(np.quantile(envelope, [.05,.15,.3,.5,.7,.85,.95]) - np.quantile(envelope,.9))
    return np.concatenate(result).astype('f4')


def describe(signal, sr):
    samples = at_rate(signal,sr)
    chunks = components(samples)
    n = len(chunks[-1])
    # Overlapping local examples retain intermittent changes; include a global summary.
    width = min(n,155)
    starts = np.unique(np.linspace(0,n-width,min(8,1+max(0,n-width)//85+int(n>width)),dtype=int))
    windows = [(0,n)] + [(i,i+width) for i in starts] if n>width else [(0,n)]
    feats = [describe_frames(chunks,a,b) for a,b in windows]
    good = [f for f in feats if f is not None]
    if not good:
        raise ValueError("No usable active part")
    return np.stack(good)


def _dist(x,basis):
    return np.maximum(np.mean(x*x,axis=1)[:,None] + np.mean(basis*basis,axis=1)
                      - 2 * (x @ basis.T) / x.shape[1], 0.)


def design(x,basis):
    d = _dist(x,basis)
    return np.column_stack((np.exp(-d*.5), np.exp(-d*2.5), x * .25 / np.sqrt(x.shape[1]),
                            np.ones(len(x))))


def fit(features, labels, seed=171):
    x = np.concatenate(features)
    weights = np.concatenate([np.repeat(1/len(g), len(g)) for g in features])
    y = np.repeat(labels, [len(g) for g in features])
    counts = np.bincount(labels, minlength=len(CLASSES))
    if (counts==0).any():
        raise ValueError("All five classes require training recordings")
    weights *= len(labels)/(len(CLASSES)*counts[y])
    center=np.average(x,axis=0,weights=weights)
    scale=np.sqrt(np.average((x-center)**2,axis=0,weights=weights)).clip(.03)
    z=np.clip((x-center)/scale,-4,4).astype('f4')
    gen=np.random.default_rng(seed)
    chosen=sorted(gen.choice(len(z),min(360,len(z)),replace=False,p=weights/weights.sum()))
    basis=z[chosen]
    mat=design(z,basis)
    cov=(mat.T*weights) @ mat
    cross=(mat.T*weights) @ np.eye(len(CLASSES))[y]
    coef=np.linalg.solve(cov+.3*np.eye(len(cov)),cross)
    return dict(center=center.astype('f4'), scale=scale.astype('f4'),basis=basis,coef=coef.astype('f4'))


def scores_from_features(feats, model):
    z=np.clip((feats-model['center'])/model['scale'],-4,4)
    pooled=design(z,model['basis']) @ model['coef']
    return .35*pooled[0]+.65*np.mean(pooled[1:] if len(pooled)>1 else pooled,axis=0)


@lru_cache(maxsize=1)
def load_model():
    with np.load(HERE/'weights.npz',allow_pickle=False) as f:
        return dict(f)


def predict_mono(mono,sr,model=None):
    """Five uncalibrated decision scores, in CLASSES order."""
    if model is None: model=load_model()
    return scores_from_features(describe(mono,sr),model)


def analyse(audio,sr=48000,model=None):
    if sr != 48000:
        raise ValueError("Stereo analysis expects 48 kHz")
    sources,mixing,angles,focus=candidates(np.asarray(audio,dtype='f4'))
    probs=np.stack([predict_mono(s,sr,model) for s in sources.T])
    return sources,mixing,probs,angles,focus


def predict_scene(audio,sr=48000,model=None):
    if model is None: model=load_model()
    result=dict.fromkeys(CLASSES)
    try:
        stems,mixing,scores,_,_=analyse(audio,sr,model)
    except ValueError:
        return result
    for j,label in enumerate(CLASSES):
        i=int(np.argmax(scores[:,j]))
        if scores[i,j]>=float(model['threshold']):
            result[label]=stems[:,i,None]*mixing[:,i]
    return result


def separate(audio,sr,target):
    if target not in CLASSES:
        raise ValueError("Unknown requested class")
    result=predict_scene(audio,sr)[target]
    if result is None:
        raise ValueError("Insufficient evidence for requested source")
    return result
