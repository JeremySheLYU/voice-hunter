"""Compare archived v0.1 and current product on an independently selected native-stereo pair."""
import json
import math
from pathlib import Path
import tempfile
from zipfile import ZipFile
import numpy as np
from run import ROOT,HERE,audit_export,cli,pcm,sha
import sys
sys.path.insert(0,str(ROOT))
from voicehunter.media import write_wav

def sdr(x,y):
    a=x.astype('float64')-x.mean(axis=0,dtype='float64')
    b=y.astype('float64')-y.mean(axis=0,dtype='float64')
    p=b*np.sum(a*b)/np.sum(b*b)
    return 10*math.log10(max(1e-25,np.sum(p*p))/max(1e-25,np.sum((a-p)**2)))

def main():
    sr=48000
    rows=json.loads((ROOT/'round2/envdata/all_sources.json').read_text())
    fresh=[next(t for t in rows if t['id']==k) for k in ['330660','386515']]
    frozen={r['id'] for r in json.loads((ROOT/'round2/environment/cohort.json').read_text())}
    xs=[]
    for i,r in enumerate(fresh):
        assert r['split']=='test' and r['id'] not in frozen and r['signal_layout'].startswith('two-channel with')
        p=ROOT/r['path'];assert sha(p.read_bytes())==r['sha256']
        sound=pcm(p).astype('f8')
        h=sound[3*sr:6*sr]
        assert h.shape==(sr*3,2)
        gain=.095/np.sqrt(np.mean(h*h))
        a=np.zeros((sr*6,2));start=int(sr*1.45*i)
        a[start:start+int(sr*1.25)]=sound[:int(sr*1.25)]*gain
        a[sr*3:]=h*gain;xs.append(a)
    scale=max(1,float(np.max(np.abs(sum(xs))))/.9)
    xs=[(x/scale).astype('f4') for x in xs]; mix=sum(xs)
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp); wav=temp/'sum.wav';write_wav(wav,mix,stem=True)
        with ZipFile(ROOT/'outputs/VoiceHunter_v0.1.zip') as z:z.extractall(temp/'release')
        old=temp/'release/VoiceHunter'
        for k,app,mode in [('v0.1',old,'example'),('v0.2',ROOT,'spatial-example')]:
            out=temp/k;r=cli(wav,out,mode,app=app,span=(.06,1.18),bg=(1.51,2.63),measured=False)
            if r.returncode:
                results.append(dict(version=k,returned=False,passed_quality=False,error=r.stderr.strip()));continue
            valid,info=audit_export(out,wav)
            scores={}
            for i,(label,name) in enumerate([('train','target.wav'),('wind','remaining.wav')]):
                ref=xs[i][sr*3:];base=sdr(mix[sr*3:],ref);final=sdr(pcm(out/name)[sr*3:],ref)
                scores[label]=dict(in_db=base,out_db=final,improvement_db=final-base)
            results.append(dict(version=k,returned=True,valid_output=valid,passed_quality=valid and all(a['improvement_db']>=3 for a in scores.values()),scores=scores,**info))
    doc=dict(input_ids=[r['id'] for r in fresh],source_hashes=[r['sha256'] for r in fresh],settings='native-pair-protocol.md',cases=results)
    (HERE/'native-pair-results.json').write_text(json.dumps(doc,indent=2)+'\n')
    print([(r['version'],r['passed_quality'],r.get('scores')) for r in results])

if __name__=='__main__':main()
