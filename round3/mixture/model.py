"""Blind convolutive two-channel demixer and supervised spectral mask fallback."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from round2.semantic import semantic_candidate as old
from voicehunter.spatial import _samples, stft, extract_stereo

HERE = Path(__file__).resolve().parent
CLASSES = old.CLASSES
RATE = 48000
NFFT = 3072
HOP = 768


def project_back(spec, size):
    """Synthesize a complex source image with the centred square-root Hann."""
    win = np.sqrt(np.hanning(NFFT + 1)[:-1]).astype('f4')
    block = (np.fft.irfft(spec.transpose(1, 2, 0), n=NFFT) * win).astype('f4')
    result = np.zeros(((len(block)-1)*HOP + NFFT, 2), dtype='f4')
    norm = np.zeros(len(result), dtype='f4')
    for i, frame in enumerate(block):
        k = i * HOP
        result[k:k+NFFT] += frame.T
        norm[k:k+NFFT] += win**2
    valid = slice(NFFT//2, NFFT//2 + size)
    return result[valid] / norm[valid,None]


def demix(audio, iterations=12):
    """2x2 weighted independent vector analysis with frequency-specific mixing."""
    x = _samples(audio, stereo=True)
    spec = stft(x, NFFT, HOP)
    left, right = spec[:,:,0], spec[:,:,1]
    power = abs(left)**2 + abs(right)**2
    f = len(power)
    energy = power.mean(axis=1).clip(1e-9)
    cross = left * np.conj(right)
    coherence = np.abs(cross.sum(axis=1))**2 / (
        (abs(left)**2).sum(axis=1) * (abs(right)**2).sum(axis=1) + 1e-20)
    # If redundant channels fit a single transfer almost everywhere, the
    # two-source filter is singular and there is no usable spatial cue.
    coherent_fraction = np.sum(energy * (coherence > .995)) / energy.sum()
    if coherent_fraction > .95:
        raise ValueError("Two-channel spatial contrast is insufficient")
    # Separately initialize each frequency from left/right energy ratios.
    # Each selected time-frequency group contributes a rank-one estimate.
    observations = spec.transpose(0,2,1)
    ild = (abs(left)**2 - abs(right)**2) / np.maximum(power,1e-12)
    split = np.median(ild,axis=1,keepdims=True)
    axes = []
    for pick in ((ild>split), (ild<=split)):
        selected = observations * np.sqrt(pick[:,None,:])
        cov = selected @ selected.conj().transpose(0,2,1)
        _,vec = np.linalg.eigh(cov)
        axes.append(vec[:,:,-1])
    matrix = np.stack(axes,axis=2).astype('c8')
    near_singular = abs(matrix[:,0,0]*matrix[:,1,1] - matrix[:,0,1]*matrix[:,1,0]) < .14
    matrix[near_singular] = np.eye(2,dtype='c8')
    w = np.linalg.inv(matrix)
    y = w @ observations
    for _ in range(iterations):
        # Mean over frequency couples the demixing permutation across bands.
        active = (energy > energy.max()*1e-5) & (np.fft.rfftfreq(NFFT,1/RATE) < 16000)
        power_y = abs(y[active])**2
        norms = np.mean(power_y / power_y.mean(axis=2,keepdims=True).clip(1e-12),axis=0).clip(1e-7)**-.5
        for k in (0,1):
            sample = y[:,k,:]
            mag = abs(sample)**2
            numerator = np.mean(y * np.conj(sample[:,None,:]) * norms[None,:,:],axis=2)
            denom = (mag[:,None,:] * norms[None,:,:]).mean(axis=2).clip(1e-12)
            coeff = numerator/denom
            coeff[:,k] = 1 - 1/np.sqrt(denom[:,k])
            y -= coeff[:,:,None] * sample[:,None,:].copy()
            w -= coeff[:,:,None] * w[:,k:k+1,:].copy()
    mixing = np.linalg.pinv(w, rcond=1e-6)
    output = mixing[:,None,:,0] * y[:,0,:,None]
    stem = project_back(output, len(x))
    return np.stack((stem, x-stem))


@lru_cache(maxsize=1)
def params():
    with np.load(HERE/'weights.npz',allow_pickle=False) as data:
        return dict(data)


def band_map():
    hz=np.fft.rfftfreq(NFFT,1/RATE)
    edges=np.geomspace(80,20000,94)
    return np.searchsorted(edges,hz)


def nmf_components(audio, model):
    """Sparse nonnegative fitting of class-tagged spectrotemporal spectra."""
    x=_samples(audio,stereo=True)
    spec=stft(x,NFFT,HOP)
    bands=band_map()
    power=np.sum(abs(spec)**2,axis=2)*.5
    binned=np.zeros((int(bands.max())+1,power.shape[1]),dtype='f4')
    np.add.at(binned,bands,power)
    mag=np.sqrt(binned).astype('f4')
    mag *= model['scale'][:,None]
    dictionary=model['dictionary']
    sample=dictionary.T @ mag
    gram=np.einsum('ij,ik->jk',dictionary,dictionary,optimize=False)
    activation=sample.clip(1e-12)
    floor=.015*sample.max(axis=0,keepdims=True)+1e-9
    for _ in range(32):
        activation *= sample / ((gram @ activation) + floor).clip(1e-10)
    index=model['class_index']
    images=np.stack([dictionary[:,index==i] @ activation[index==i] for i in range(len(CLASSES))])
    confidence=np.quantile(np.sum(images**2,axis=1)/(np.sum(images**2,axis=(0,1))[None,:]+1e-12),.78,axis=1)
    return images, confidence


def dictionary_stem(audio, images, k):
    powers=images**2
    mask=(powers[k] / (powers.sum(axis=0)+1e-12)).clip(0,1)
    return extract_stereo(audio,mask[band_map()],n_fft=NFFT,hop=HOP)[0]


def mono_classify(audio):
    powers=np.mean(audio**2,axis=0)
    mono=audio[:,int(np.argmax(powers))]
    model=old.load_model()
    feats=old.feature_batch(mono)
    return np.mean([old.classify(f,model) for f in feats],axis=0)


def analyse(audio,sr):
    if sr!=RATE:raise ValueError("48 kHz input required")
    x=_samples(audio,stereo=True)
    try:
        stems=demix(x)
        scores=np.array([mono_classify(s) for s in stems])
        return ("iva",stems,scores.max(axis=0),scores.argmax(axis=0))
    except ValueError:
        images,score=nmf_components(x,params())
        return ("dict",images,score,None)


def dictionary_baseline_scene(audio,sr):
    method,result,score,index=analyse(audio,sr)
    config=params(); threshold=config['threshold_'+method]
    stems={}
    for i,c in enumerate(CLASSES):
        if score[i]>=threshold:
            stems[c]=result[index[i]] if method=='iva' else dictionary_stem(audio,result,i)
    return stems


def dictionary_baseline(audio,sr,target):
    if target not in CLASSES:raise ValueError('Unknown semantic class')
    stems=dictionary_baseline_scene(audio,sr)
    if target not in stems:raise ValueError('Requested category below validation threshold')
    return stems[target]

def predict_scene(audio,sr,unused=None):
    """Frozen contender; the dictionary-only path is retained as an ablation."""
    from round3.mixture.gateway import predict_scene as predict
    return predict(audio,sr)


def separate(audio,sr,target):
    from round3.mixture.gateway import separate as predict
    return predict(audio,sr,target)
