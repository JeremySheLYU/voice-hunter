"""Audit gain/polarity-invariant, time-aligned source duplicates across heldout and fit sets."""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RATE=1000
TIMES=[.13,.57,1.04,1.56]
def fingerprints(row):
    with wave.open(str(ROOT/row['path']),'rb') as w:
        if w.getsampwidth()!=2 or w.getnchannels()!=1:raise ValueError(row['id'])
        rate=w.getframerate(); audio=np.frombuffer(w.readframes(min(w.getnframes(),2*rate)),'<i2').astype('f8')
    starts=np.floor((np.arange(257)/RATE)*rate).astype(int)
    cs=np.r_[0.,np.cumsum(audio)]
    blocks=[]
    for t in TIMES:
        idx=starts+round(t*rate);x=np.diff(cs[idx])/np.diff(idx);x-=x.mean()
        n=np.linalg.norm(x)
        blocks.append(x/n if n>2 else np.zeros_like(x))
    return np.asarray(blocks,'f4')
def main():
    source={'history':json.loads((ROOT/'round2/semantic/cohort.json').read_text()),'train':json.loads((HERE/'train.json').read_text()),'test':json.loads((HERE/'test.json').read_text()),'oov':json.loads((ROOT/'round3/unknowns/manifest.json').read_text())}
    features={k:np.stack([fingerprints(r) for r in rows]) for k,rows in source.items()}
    findings=[]
    for fit,test in [('history','test'),('history','train'),('history','oov'),('train','test'),('train','oov'),('test','oov')]:
        a,b=features[fit],features[test]
        scores=np.stack([np.abs(a[:,j]@b[:,j].T) for j in range(len(TIMES))])
        match=scores.min(axis=0)>.996
        for i,j in zip(*match.nonzero()):
            findings.append({'source_a':source[fit][i]['id'],'source_b':source[test][j]['id'],'groups':[fit,test],'correlation':scores[:,i,j].tolist()})
    result={'method':'Four nonoverlapping 256 ms blocks at predeclared offsets, mono 1 kHz block averages, demeaned and gain-normalized. Candidate if all four absolute Pearson correlations exceed 0.996. Sensitivity limited to time-aligned material and detectable signal in all windows. No arbitrary time shift or cover-version search.','subsets':{k:len(v) for k,v in source.items()},'matches':findings}
    (HERE/'aligned_audio_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':main()
