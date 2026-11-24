"""Frozen Audio Spectrogram Transformer with a five-class transfer head."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from round2.semantic.semantic_candidate import candidates, CLASSES

HERE=Path(__file__).resolve().parent
FAMILIES=[range(140,147),(153,154),range(0,7),range(328,335),range(283,286)]


@lru_cache(maxsize=1)
def session():
    import onnxruntime as ort
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=3
    opt.inter_op_num_threads=1
    return ort.InferenceSession(str(HERE/'ast-int8.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])


@lru_cache(maxsize=1)
def filters():
    m=lambda hz:1127*np.log1p(hz/700)
    hz=m(np.fft.rfftfreq(512,1/16000))
    mel=np.linspace(m(20),m(8000),130)
    slope=(hz[:,None]-mel[:-1])/np.diff(mel)
    fb=np.clip(np.minimum(slope[:,:-1],1-slope[:,1:]),0,None)
    nl=np.arange(-63,64)
    low=np.sinc(nl*.31)*.31*np.kaiser(127,7.5)
    return fb.astype('f4'),(low/low.sum()).astype('f4')


def encode(mono,sr=48000):
    if sr!=48000: raise ValueError('Input sampling rate must be 48 kHz')
    fb,low=filters()
    x=np.convolve(mono,low,mode='same')[::3].astype('f4')
    rms=float(np.sqrt(np.mean(x.astype('f8')**2)))
    if rms>1e-8:x*=min(.08/rms,1/max(float(abs(x).max()),1e-8))
    if len(x)<400:x=np.pad(x,(0,400-len(x)))
    f=np.lib.stride_tricks.sliding_window_view(x,400)[::160][:1024].copy()
    f-=f.mean(axis=1,keepdims=True)
    f[:,1:]-=.97*f[:,:-1].copy()
    f[:,0]*=.03
    pow=np.abs(np.fft.rfft(f*np.hanning(400).astype('f4'),n=512,axis=1))**2
    z=np.zeros((1024,128),dtype='f4')
    z[:len(f)]=np.log((pow@fb).clip(1.192092955078125e-7))
    z=(z+4.2677393)/(2*4.5689974)
    sess=session()
    return sess.run(None,{sess.get_inputs()[0].name:z[None]})[0][0].astype('f4')


def fit(train,y):
    # Profile the broad AudioSet score pattern; the pretrained model is frozen.
    center=train.mean(axis=0)
    mu=train-center
    unit=mu/(np.linalg.norm(mu,axis=1,keepdims=True)+.001)
    k=(.5+.5*unit@unit.T)**2
    coef=np.linalg.solve(k+.16*np.eye(len(y)),np.eye(5)[y]-.2)
    return dict(center=center,basis=unit,coef=coef.astype('f4'))


def scores(features,w,blend):
    f=np.atleast_2d(features)
    unit=f-w['center']
    unit=unit/(np.linalg.norm(unit,axis=1,keepdims=True)+.001)
    learned=((.5+.5*unit@w['basis'].T)**2@w['coef']+.2)*6.
    prior=np.stack([f[:,list(cols)].max(axis=1) for cols in FAMILIES],axis=1)
    # Relative category evidence; blend affects the decision gap, not the waveform.
    logit=blend*prior+(1-blend)*learned
    exp=np.exp(logit-logit.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True)


@lru_cache(maxsize=1)
def load_model():
    with np.load(HERE/'weights.npz',allow_pickle=False) as f:return dict(f)


def predict_scene(audio,sr=48000,model=None):
    if model is None:model=load_model()
    out=dict.fromkeys(CLASSES)
    try: src,mix,_,_=candidates(audio)
    except ValueError:return out
    features=np.stack([encode(s,sr) for s in src.T])
    posterior=scores(features,model,float(model['blend']))
    for j,c in enumerate(CLASSES):
        i=int(posterior[:,j].argmax())
        if posterior[i,j]>=float(model['threshold']):out[c]=src[:,i,None]*mix[:,i]
    return out
