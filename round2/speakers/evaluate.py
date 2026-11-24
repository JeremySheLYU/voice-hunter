"""Evaluate anonymous voice rank and path-reassignment separately on unseen readers."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZipFile

import numpy as np

from train import HERE,ROOT,decode,sha
from voicehunter.speech import separate_moving_speakers

FS=16000
AS=(.12,1.20)
BS=(1.52,2.6)


def original_method():
    with ZipFile(ROOT/'outputs'/'VoiceHunter_v0.1.zip') as z:
        txt=z.read('VoiceHunter/voicehunter/speech.py')
    module=ModuleType('voicehunter.baseline_speech')
    module.__package__='voicehunter'
    exec(compile(txt,'frozen_voicehunter/speech.py','exec'),module.__dict__)
    return module.separate_speakers


def excerpt(path,seconds):
    x=decode(path)
    # Use the highest-energy full interval; no lookup of speaker identity,
    # acoustic features, classifier predictions or product output is involved.
    n=round(seconds*8000)
    if len(x)<n:
        x=np.pad(x,(0,n-len(x)))
    starts=np.arange(0,len(x)-n+1,400)
    energy=[np.mean(x[i:i+n]**2) for i in starts]
    start=starts[np.argmax(energy)]
    y=np.interp(np.arange(2*n)*.5,np.arange(n),x[start:start+n]).astype('f4')
    y[:160]*=np.linspace(0,1,160)
    y[-160:]*=np.linspace(1,0,160)
    return .07*y/np.sqrt(np.mean(y*y)+1e-12)


def position(signal,right):
    gains=[.4,.9] if right else [.92,.36]
    out=np.stack([signal*gains[0],signal*gains[1]],axis=-1)
    out[:,0 if right else 1]=np.roll(out[:,0 if right else 1],5)
    for c in [0,1]:
        out[:,c]+=.035*np.roll(out[:,c],112+17*c)
    return out


def duration(path):
    with path.open('rb') as src:
        header=src.read(26)
    if header[:4]!=b'fLaC' or header[4]&127:raise ValueError('Unexpected FLAC header')
    packed=int.from_bytes(header[18:26],'big')
    return (packed & ((1<<36)-1))/(packed>>44)


def convolution_path(h):
    def convolve(signal,right):
        ir=h[int(right)]
        length=1<<(len(signal)+len(ir)-2).bit_length()
        return np.fft.irfft(np.fft.rfft(signal,n=length)[:,None]*np.fft.rfft(ir,n=length,axis=0),n=length,axis=0)[:len(signal)].astype('f4')
    return convolve


def make(a,b,cross,render=position):
    a1,a2=a;b1,b2=b
    length=round(7.5*FS)
    sa,sb=np.zeros(length,dtype='f4'),np.zeros(length,dtype='f4')
    sa[:len(a1)]=a1
    start=int(FS*1.4)
    sb[start:start+len(b1)]=b1
    beg=round(2.9*FS)
    sa[beg:beg+len(a2)]=a2;sb[beg:beg+len(b2)]=b2
    weight=np.clip((np.arange(length)/FS-4.02)/.5,0,1).astype('f4')[:,None] if cross else 0.
    s1=(1-weight)*render(sa,False)+weight*render(sa,True)
    s2=(1-weight)*render(sb,True)+weight*render(sb,False)
    x=s1+s2
    return x, s1,s2


def score(pred,ref):
    return float(10*np.log10(np.sum(ref*ref,dtype='f8')/max(1e-16,np.sum((pred-ref)**2,dtype='f8'))))


def outcome(f,scene):
    x,a,b=scene
    try:
        u,v,info=f(x,FS,target_span=AS,background_span=BS)
    except ValueError as ex:
        return {'status':'refused','reason':str(ex),'pass':False}
    ix=slice(round(4.68*FS),round(7.24*FS))
    power=[(s[ix]**2).mean(axis=1).reshape(-1,320).mean(axis=1) for s in [a,b]]
    keep=np.repeat((power[0]>power[0].max()*.001)&(power[1]>power[1].max()*.001),320)
    if keep.sum() < .10*FS:raise ValueError('Too little valid overlap')
    gains=[score(est[ix][keep],s[ix][keep])-score(x[ix][keep],s[ix][keep]) for est,s in [(u,a),(v,b)]]
    return {'status':'returned','gain_db':gains,'minimum_db':min(gains),'pass':min(gains)>=3.,
            'sample_sum_error':float(abs(u+v-x).max()),'reassigned_seconds':info.get('reassigned_seconds',0.)}


def summarize(rows):
    ret=[r for r in rows if r['status']=='returned']
    return {'n':len(rows),'returned':len(ret),'passed':sum(r['pass'] for r in rows),
            'harmful':sum(r['minimum_db']<0 for r in ret),'median_min_gain':float(np.median([r['minimum_db'] for r in ret])) if ret else None,
            'worst_min_gain':min((r['minimum_db'] for r in ret),default=None),
            'peak_reconstruction_fs':max((r['sample_sum_error'] for r in ret),default=None)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--measured-dir',type=Path)
    ap.add_argument('--final',action='store_true',help='Use the official LibriSpeech test-clean holdout.')
    args=ap.parse_args()
    renderer=position
    if args.measured_dir:
        import subprocess
        paths=[args.measured_dir/('stairway_az'+angle+'.wav') for angle in ['-45','+45']]
        paths_ir=paths
        h=[np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(p),'-ar',str(FS),'-f','f32le','pipe:1']),dtype='<f4').reshape(-1,2) for p in paths]
        renderer=convolution_path(h)
    train=json.loads((HERE/'training.json').read_text())
    split='test-clean' if args.final else 'dev-clean-2'
    test_root=args.cache/'LibriSpeech'/split
    files=sorted(test_root.rglob('*.flac'))
    assert files, f'Missing {split}'
    by=defaultdict(list)
    for f in files: by[f.stem.split('-')[0]].append(f)
    if args.final:
        assert not set(by)&set(train['assets'][1]['readers'])
    else:
        assert set(by)==set(train['assets'][1]['readers'])
    assert not set(by)&set(train['used_readers'])
    rng=np.random.default_rng(19011)
    talkers=sorted(by)
    rng.shuffle(talkers)
    rows=[];baseline=original_method()
    for ia,ib in zip(talkers[::2],talkers[1::2]):
        for rep in range(2):
            exs=[]; names=[]
            for who in (ia,ib):
                paths=[p for p in by[who] if duration(p)>=4.6]
                if len(paths)<4:raise ValueError(f'Insufficient qualifying independent utterances for {who}')
                enroll=paths[rep];score_path=paths[-rep-1]
                assert enroll!=score_path
                exs.append([excerpt(enroll,1.30),excerpt(score_path,4.40)])
                names.extend([enroll.stem,score_path.stem])
            for moving in [False,True]:
                scene=make(*exs,moving,renderer)
                rows.append({'pair':[ia,ib],'utterances':names,'replicate':rep,'cross':moving,
                             'v01':outcome(baseline,scene),'candidate':outcome(separate_moving_speakers,scene)})
    trial_summary={name:{m:summarize([r[m] for r in rows if r['cross']==moving]) for m in ('v01','candidate')}
                   for name,moving in [('fixed',False),('crossing',True)]}
    report={'partition':split,'files_catalogued':len(files),'reader_count':len(by),'speaker_overlap_with_train':0,
            'input_type':'real mono speech with synthetic two-microphone paths','groups':trial_summary,'scenes':rows,
            'hashes':{name:sha(ROOT/'voicehunter'/name) for name in ('speech.py','speaker_identity.py','models/speaker-distance.json')}}
    if args.measured_dir:
        report['input_type']='real public speech convolved with independent measured Aachen stairway 3 m BRIRs, ±45°; a temporal mixture of the two measured paths simulates movement'
        report['room_sources']=[{'file':p.name,'sha256':sha(p)} for p in paths_ir]
    suffix='-final.json' if args.final else '.json'
    (HERE/(('measured-heldout' if args.measured_dir else 'heldout')+suffix)).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(trial_summary),flush=True)


if __name__=='__main__':main()
