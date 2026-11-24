"""Stereo demixing by approximate simultaneous diagonalization of real local covariances."""
import numpy as np
from round3.mixture import model


def image_pair(stereo,return_quality=False):
    x=model._samples(stereo,stereo=True)
    mean=x.mean(axis=0)
    sample=x-mean
    cov=(sample.T.astype('f8') @ sample)/len(x)
    val,vec=np.linalg.eigh(cov)
    if val[1]<1e-12 or val[0]<1e-5*val[1]:raise ValueError('Two independent spatial waveforms required')
    white=(vec/np.sqrt(val)).T.astype('f4')
    s=model.stft(sample,1024,384)@white.T
    # Local cross-covariance over 8 frequency bins and 9 time frames.
    f=(s.shape[0]//8)*8;t=(s.shape[1]//9)*9
    if t<9:raise ValueError('Duration too short for stereo statistics')
    s=s[:f,:t,:].reshape(f//8,8,t//9,9,2)
    def pool(z):return z.mean(axis=(1,3)).ravel()
    a=pool(abs(s[...,0])**2);b=pool(abs(s[...,1])**2)
    c=pool(s[...,0]*np.conj(s[...,1]))
    d=a+b; threshold=np.quantile(d,.65)*.001+1e-11;keep=d>threshold
    # Unit-trace matrices suppress the dominance of a single loud band.
    d=d[keep];u=(a[keep]-b[keep])/d;v=2*c[keep]/d
    m=np.stack((u,v.real))
    gram=m@m.T;_,axes=np.linalg.eigh(gram);q=axes[:,-1]
    phi=.5*np.arctan2(q[1],q[0]);rot=np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    w=(rot@white).astype('f4');mix=np.linalg.inv(w)
    mono=x@w.T
    images=np.stack([mono[:,i,None]*mix[None,:,i] for i in (0,1)])
    off_fraction=float(np.mean(np.sqrt((v.real*q[0]-u*q[1])**2+v.imag**2)))
    panning_contrast=float(np.min(mix[0]*mix[1]/(np.sum(mix**2,axis=0)+1e-10)))
    if return_quality:return images,(off_fraction,panning_contrast)
    return images
