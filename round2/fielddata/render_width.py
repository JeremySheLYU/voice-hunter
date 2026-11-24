"""Audition the learned ambient width correction for a stereo virtual ±20° XY pickup."""
import argparse
import json
from pathlib import Path
import wave

import numpy as np

from fit_width import BANDS, FFT, HERE, decode


def render(x, rate, gains_db):
    mid=(x[:,0]+x[:,1])*.5
    side=(x[:,0]-x[:,1])*.5
    hop=FFT//2
    frequency=np.fft.rfftfreq(FFT,d=1/rate)
    band=np.clip(np.searchsorted(BANDS,frequency,side='right')-1,0,len(BANDS)-2)
    gains=10**(np.asarray(gains_db)[band]/20)
    win=np.sin(np.pi*(np.arange(FFT)+.5)/FFT)
    padded=np.pad(side,(hop,FFT))
    output=np.zeros(padded.shape,dtype=np.float64)
    for start in range(0,len(padded)-FFT+1,hop):
        frame=win*padded[start:start+FFT]
        output[start:start+FFT]+=win*np.fft.irfft(gains*np.fft.rfft(frame))
    side_out=output[hop:hop+len(x)]
    return np.column_stack((mid+side_out,mid-side_out)).astype('f4')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--condition',choices=['fitted_bands','fitted_constant','fixed_1.35','unchanged'],default='fitted_bands')
    args=p.parse_args()
    if args.output.exists():p.error('output path already exists')
    import subprocess
    num=int(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
                'stream=channels','-of','default=nw=1:nk=1',str(args.input)]))
    if num!=2:p.error('input must have exactly two channels')
    x,rate=decode(args.input,2)
    weights=json.loads((HERE/'width_results.json').read_text())['conditions'][args.condition]['gain_db']
    y=render(x,rate,weights)
    if not np.isfinite(y).all():p.error('invalid samples')
    peak=max(float(np.max(abs(y))),1)
    with wave.open(str(args.output),'wb') as w:
        w.setparams((2,3,rate,len(y),'NONE','not compressed'))
        w.writeframes(np.round(y/peak*8388607).astype('<i4').view('u1').reshape(-1,4)[:,:3].tobytes())
    print('Linked output gain:',1/peak)
if __name__=='__main__':main()
