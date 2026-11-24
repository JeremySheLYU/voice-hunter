"""Offline calibration of narrow XY stereo ambience against a published binaural teacher."""
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'round2_data' / 'field'
BANDS = np.array([100,400,1600,6400,15000])
FFT = 2048

def decode(path, channels):
    # Explicit pipe input also avoids file-path autodetection for a header-only zero sample rate.
    payload=path.read_bytes()
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
        'stream=sample_rate,channels','-of','json','pipe:0'],input=payload))['streams'][0]
    rate=int(info['sample_rate'])
    if rate<=0 or info['channels']!=channels:
        raise ValueError(f'Invalid rate or channels in {path}: {info}')
    data=subprocess.check_output(['ffmpeg','-nostdin','-loglevel','error','-i','pipe:0','-map','0:a:0',
        '-f','f32le','-c:a','pcm_f32le','pipe:1'],input=payload)
    return np.frombuffer(data,dtype='<f4').reshape(-1,channels),rate

def ratios(channels,rate):
    # Calculate side/mid power in band. Remove boundaries around time cut and startup.
    values=channels[round(.35*rate):round(3.8*rate)]
    mid=(values[:,0]+values[:,1])*.5; side=(values[:,0]-values[:,1])*.5
    window=np.hanning(FFT)
    def power(x):
        block=np.lib.stride_tricks.sliding_window_view(x, FFT)[::FFT//2]*window
        return np.mean(abs(np.fft.rfft(block))**2,axis=0)
    m,s=power(mid),power(side)
    f=np.fft.rfftfreq(FFT,d=1/rate)
    return np.array([10*np.log10((s[(f>=lo)&(f<hi)].sum()+1e-18)/(m[(f>=lo)&(f<hi)].sum()+1e-18))
                     for lo,hi in zip(BANDS,BANDS[1:])])

def input_stereo(foa):
    w,y,_,x=foa.T
    middle=.5*(w+np.cos(np.deg2rad(20))*x)
    side=.5*np.sin(np.deg2rad(20))*y
    return np.column_stack([middle+side,middle-side])

def fit():
    ls={}
    for p in sorted((DATA/'ihta_foa').glob('*.flac')):
        name=p.name.removeprefix('FirstOrderAmbisonics_')
        ears=DATA/'ihta_ears'/('Binaural_'+name)
        if not ears.exists():continue
        foa,a=decode(p,4); teacher,b=decode(ears,2)
        assert len(foa)/a==len(teacher)/b==4.
        raw=ratios(input_stereo(foa),a);target=ratios(teacher,b)
        ls[name]=dict(date=name.split('_')[0], input_db=raw.tolist(),teacher_db=target.tolist())
    dates=sorted(set(v['date'] for v in ls.values()))
    testdates=set(dates[3::4])
    assert len(ls)==102 and len(testdates)>=2
    def dif(rows):
        return np.array([np.array(v['teacher_db'])-np.array(v['input_db']) for v in rows])
    train=[x for x in ls.values() if x['date'] not in testdates]
    test=[x for x in ls.values() if x['date'] in testdates]
    tr=dif(train);te=dif(test)
    curves={'unchanged':np.zeros(4),'fixed_1.35':np.full(4,20*np.log10(1.35)),
            'fitted_constant':np.full(4,np.clip(np.median(tr),-12,12)),
            'fitted_bands':np.clip(np.median(tr,axis=0),-12,12)}
    details={}
    for name,g in curves.items():
        error=np.abs(te-g)
        details[name]=dict(gain_db=g.tolist(),heldout_median_band_error_db=float(np.median(error)),
                           median_absolute_error_by_band_db=np.median(error,axis=0).tolist(),
                           improved_captures_vs_unchanged=int(np.sum(np.mean(error,axis=1)<np.mean(abs(te),axis=1))))
    r=dict(teacher='published IHTApark binaural decoder (derived from the same FOA)',
        metric='absolute side/mid ratio error, dB; objective scene statistics only',
        training_dates=[d for d in dates if d not in testdates],test_dates=sorted(testdates),
        training_captures=len(train),heldout_captures=len(test),bands_hz=BANDS.tolist(),conditions=details,
        met_prespecified_gate=all(details['fitted_bands']['heldout_median_band_error_db']<details[k]['heldout_median_band_error_db'] for k in ['unchanged','fixed_1.35']),measurements=ls)
    (HERE/'width_results.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['training_captures'],r['heldout_captures'],details)
if __name__=='__main__':fit()
