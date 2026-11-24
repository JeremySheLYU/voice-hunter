"""Target-conditioned full-rank stereo mask from learned class dictionaries."""
import numpy as np
from round3.mixture import model


def output(x, images, k, presence):
    """Alternate class-temporal prior with per-frequency spatial clustering."""
    # A second class estimate controls interference. A small diffuse floor keeps
    # weak unmodelled classes from being removed entirely when scoring is uncertain.
    strengths=np.array(presence,copy=True);strengths[k]=-np.inf
    second=int(strengths.argmax())
    a=images[k];b=images[second]+.16*(images.sum(axis=0)-images[k]-images[second])
    soft=a**1.4 / (a**1.4+b**1.4+1e-12)
    prior=np.clip(soft[model.band_map()],.01,.99)
    s=model.stft(x,model.NFFT,model.HOP)
    left,right=s[:,:,0],s[:,:,1]
    total=(abs(left)**2+abs(right)**2).clip(1e-12)
    l=abs(left)**2/total;r=abs(right)**2/total;c=left*right.conj()/total
    p=prior
    floor=.015
    rel=total**.25
    def angular_model(w):
        den=w.sum(axis=1).clip(1e-10)
        a=(w*l).sum(axis=1)/den+floor
        b=(w*r).sum(axis=1)/den+floor
        z=(w*c).sum(axis=1)/den
        det=(a*b-abs(z)**2).clip(1e-8)
        quad=(b[:,None]*l+a[:,None]*r-2*(np.conj(z[:,None])*c).real).clip(1e-10)
        return det[:,None]/quad**2
    for _ in range(3):
        u=angular_model(p*rel);v=angular_model((1-p)*rel)
        odds=prior*u/((1-prior)*v+1e-12)
        p=odds/(1+odds)
    return model.extract_stereo(x,p.astype('f4').clip(0,1),n_fft=model.NFFT,hop=model.HOP)[0]
