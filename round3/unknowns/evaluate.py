"""Inspect false returns on two unrelated sounds; never use this cohort to select a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from round3.evaluation.evaluate import predictor
from round2.semantic.semantic_candidate import CLASSES
from .prepare import HERE, LABELS, ROOT
sys.path.insert(0,str(ROOT/'round2/semantic'))
from round2.semantic.evaluate import sound, SR

def main(models):
    lock=json.loads((ROOT/'round3/evaluation/selection.json').read_text())
    raw=json.loads((HERE/'manifest.json').read_text())
    groups={c:sorted((r for r in raw if r['category']==c),key=lambda r:r['id']) for c in LABELS}
    if not all(len(rows)==6 for rows in groups.values()):raise ValueError('Expected six clips per category.')
    pairs=[]
    for cycle,offset in enumerate((1,2,1)):
        for i,c in enumerate(LABELS):
            pairs.append([groups[c][2*cycle],groups[LABELS[(i+offset)%5]][2*cycle+1]])
    assert len({r['id'] for pair in pairs for r in pair})==30
    output={}
    for name in models:
        pred=predictor(name)
        counts=Counter(); all_rows=[]
        for i,pair in enumerate(pairs):
            waves=[sound(r['path']) for r in pair]
            n=min(int(4.5*SR),*[len(w) for w in waves])
            angles=np.radians([20,70] if i%2 else [68,18])
            mix=np.zeros((n,2),dtype='f4')
            for angle,w in zip(angles,waves):
                s=w[(len(w)-n)//2:(len(w)-n)//2+n]
                s=s*.08/max(np.sqrt(np.mean(s.astype('f8')**2)),1e-6)
                mix+=s[:,None]*np.array([np.cos(angle),np.sin(angle)],'f4')
            mix*=min(1.,.9/(np.max(abs(mix))+1e-8))
            out=pred(mix,CLASSES)
            called=[k for k in CLASSES if out.get(k) is not None]
            counts.update(called)
            all_rows.append(dict(ids=[r['id'] for r in pair],categories=[r['category'] for r in pair],returned=called))
        output[name]=dict(requests=len(pairs)*len(CLASSES),false_accepts=sum(counts.values()),
                          by_requested=dict(counts),rows=all_rows)
        (HERE/'results.json').write_text(json.dumps(output,indent=2))
        print(name,counts,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('models',nargs='+'); main(p.parse_args().models)
