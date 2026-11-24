"""Class-conditioned two-source extraction for separated, stationary stereo images."""
from functools import lru_cache
from pathlib import Path
import numpy as np

try:
    from .spatial import stft, _samples
except ImportError:  # research package before promotion
    from voicehunter.spatial import stft, _samples

CLASSES=("guitar","piano","speech","train","wind")
RATE=48000
BINS=np.geomspace(40,20000,33)
LOC=Path(__file__).resolve().parent
WEIGHTS=LOC/"semantic_profiles.npz" if LOC.name=="semantic" else LOC/"models/semantic_profiles.npz"


def descriptor(mono):
    """Describe up to 5 seconds of active spectra in a single dry/estimated signal."""
    mono=np.asarray(mono,dtype='f4')
    spec=np.abs(stft(mono,2048,480)[:,:,0]).astype('f8')
    rms=np.sqrt(np.sum(spec**2,axis=0))
    if len(rms)<10 or np.max(rms)<1e-6:
        raise ValueError("可分析的聲音過短或過弱。")
    select=rms>max(1e-6,np.quantile(rms,.85)*.12)
    if select.sum()<6: raise ValueError("聲源有效時長不足。")
    hz=np.fft.rfftfreq(2048,1/RATE)
    measures=[]
    bins=np.minimum(np.searchsorted(BINS,hz,side='right')-1,len(BINS)-2).clip(0)
    power=spec**2
    for b in range(len(BINS)-1):
        mask=bins==b
        measures.append(power[mask].sum(axis=0)/max(1,mask.sum()))
    z=np.log(np.maximum(np.stack(measures),1e-10))
    shape=z-z.max(axis=0)
    shape=np.maximum(shape,-13)
    active=shape[:,select]
    centroid=np.sum(hz[:,None]*power,axis=0)/(np.sum(power,axis=0)+1e-12)
    tonal_slice=spec[(hz>=300)&(hz<=6000)]
    tonal=np.exp(np.mean(np.log(tonal_slice+1e-10),axis=0))/(np.mean(tonal_slice,axis=0)+1e-10)
    # Adjacent FFT-bin contrast and envelope quantiles separate sustained
    # broadband noise, voiced activity and pitched instruments.
    contrast=(np.mean(abs(np.diff(tonal_slice,axis=0)),axis=0) / (np.mean(tonal_slice,axis=0)+1e-10))
    envelope=rms/(np.quantile(rms[select],.8)+1e-10)
    f=np.concatenate([np.mean(active,axis=1),np.std(active,axis=1),
                      np.quantile(active,.8,axis=1),np.mean(np.abs(np.diff(active,axis=1)),axis=1),
                      np.quantile(centroid[select]/1000,[.1,.5,.9]),np.quantile(tonal[select],[.1,.5,.9]),
                      np.quantile(contrast[select],[.1,.5,.9]),np.quantile(envelope,[.05,.25,.5,.75,.95])])
    return f


def fit_descriptors(x,y):
    center=x.mean(axis=0)
    scale=x.std(axis=0).clip(.035)
    z=np.clip((x-center)/scale,-5,5)
    responses=np.eye(len(CLASSES))[y]
    ridge=np.linalg.solve(z.T@z+len(z)*.65*np.eye(z.shape[1]), z.T@(responses-responses.mean(axis=0)))
    return dict(center=center,scale=scale,embedding=z,labels=np.asarray(y),ridge=ridge,prior=responses.mean(axis=0))


def classify(features,model):
    z=np.clip((features-model['center'])/model['scale'],-5,5)
    dist=np.sqrt(np.mean((model['embedding']-z)**2,axis=1))
    nearest=np.array([np.partition(dist[model['labels']==i],2)[:3].mean() for i in range(len(CLASSES))])
    scores=(z@model['ridge']+model['prior'])*3.5-2.25*nearest
    e=np.exp(scores-scores.max())
    return e/e.sum()


def feature_batch(signal):
    # Rise/fall and voice pauses are informative over a multi-second window.
    if len(signal)<=RATE*5:
        return [descriptor(signal)]
    starts=np.linspace(0,len(signal)-RATE*5,min(12,1+(len(signal)-1)//(RATE*3)),dtype=int)
    outputs=[]
    for i in starts:
        try: outputs.append(descriptor(signal[i:i+RATE*5]))
        except ValueError: continue
    if not outputs: raise ValueError("這個聲源缺少可用的音訊。")
    return outputs


def candidates(audio):
    probe=audio[::3]  # channel ratios only, full-rate audio used to render
    spec=stft(probe,512,128)
    mag=np.abs(spec)
    total=(mag[:,:,0]**2+mag[:,:,1]**2)
    phi=np.degrees(np.arctan2(mag[:,:,1],mag[:,:,0]+1e-15))
    h,_=np.histogram(phi,bins=np.arange(-.5,91.5,1),weights=np.sqrt(total))
    smooth=np.convolve(h,[.12,.22,.32,.22,.12],mode='same')
    peaks=[i for i in range(91) if smooth[i]==smooth[max(0,i-3):min(91,i+4)].max()]
    ordered=sorted(peaks,key=lambda i:-smooth[i])
    if not ordered or smooth[ordered[0]]<=0: raise ValueError("找不到足夠清楚的立體聲聲源。")
    a=ordered[0]
    b=next((i for i in ordered if abs(a-i)>=15 and smooth[i]>=.12*smooth[a]),None)
    if b is None: raise ValueError("缺少兩個分開的穩定立體聲方向。")
    angles=sorted([a,b])
    # Refine the centre of either panning cluster using its local energy.
    for k,i in enumerate(angles):
        select=abs(phi-i)<2.5
        wt=np.sqrt(total[select])
        if wt.sum()>0:angles[k]=float(np.sum(phi[select]*wt)/wt.sum())
    dist=np.minimum(abs(phi-angles[0]),abs(phi-angles[1]))
    focus=float(np.sqrt(total[dist<5]).sum()/(np.sqrt(total).sum()+1e-12))
    if focus<.36: raise ValueError("環境聲較分散，不能可靠推算兩個水平方向。")
    r=np.radians(angles)
    mixing=np.stack((np.cos(r),np.sin(r)))
    separated=audio@np.linalg.inv(mixing).T
    return separated.astype('f4'),mixing.astype('f4'),angles,focus


def analyse(audio,sr,model):
    x=_samples(audio,stereo=True)
    if sr!=RATE: raise ValueError("聲音類別模型的工作取樣率為 48 kHz。")
    sources,mix,angles,focus=candidates(x)
    probabilities=[]
    for mono in sources.T:
        features=feature_batch(mono)
        probabilities.append(np.mean([classify(f,model) for f in features],axis=0))
    return sources,mix,np.array(probabilities),angles,focus


@lru_cache(maxsize=1)
def load_model():
    with np.load(WEIGHTS,allow_pickle=False) as f:
        return dict(f)


def separate_semantic(audio,sr,target):
    if target not in CLASSES: raise ValueError("請選 guitar、piano、speech、train 或 wind。")
    model=load_model()
    sources,mix,prob,angles,focus=analyse(audio,sr,model)
    i=CLASSES.index(target);loc=int(np.argmax(prob[:,i]));score=float(prob[loc,i])
    if score<float(model['threshold']):
        raise ValueError("錄音中的指定聲音不夠明確。請提供一段示範，或改用其他分離模式。")
    stem=sources[:,loc,None]*mix[:,loc]
    return stem, audio-stem, dict(target=target, method="semantic", model="FSD50K stereo image classifier",
               candidate_confidence=round(score,4),panning_vector_degrees=angles,focus=round(focus,4),
               note="適用兩個聲像穩定且分開的來源。這些值描述左右聲道比例，不能解讀成實際現場角度。")
