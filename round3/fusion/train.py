"""Fit fusion/decision parameters solely on historical validation; its two heads use training-only audio."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from round3.apple_features import train as apple_train
from round3.fusion import model as m

ROOT=m.HERE.parents[1]
v2=apple_train.v2
TRANSFER_CACHE=Path('/tmp/voicehunter-round3-transfer-features')
CACHE=Path(tempfile.gettempdir())/'voicehunter-r3-fusion'
CACHE.mkdir(exist_ok=True)

def vectors_for_mix(scene):
    audio,t,meta=scene
    suffix=f"{meta['kind']}-{meta['source_id']}-{meta['background_id']}"
    a=np.load(apple_train.CACHE/('validation-'+suffix+'.npz'))
    b=np.load(TRANSFER_CACHE/('val-'+suffix+'.npz'))
    if len(a['stat']) and len(b['vectors']):
        return a['stat'],b['vectors'],b['priors'],a['deltas']
    out=CACHE/(suffix+'.npz')
    if not out.exists():
        try:
            stems,matrix=m.demix(audio)
            stats=np.stack([m.apple.describe(y) for y in stems.T])
            reps=[m.encoder().encode(y) for y in stems.T]
            vectors,priors=map(np.stack,zip(*reps))
            sl=slice(int(m.RATE*(2.45 if meta['kind']=='intermittent' else .15)),int(m.RATE*4.35))
            base=v2.sisdr(audio[sl],t[sl])
            dg=[v2.sisdr(stems[sl,i,None]*matrix[:,i],t[sl])-base for i in range(2)]
        except ValueError:stats=vectors=priors=dg=[]
        np.savez_compressed(out,stats=stats,vectors=vectors,priors=priors,gain=dg)
    with np.load(out) as f:return f['stats'],f['vectors'],f['priors'],f['gain']


def results(rows,thresh):
    metrics={}
    for name in apple_train.KINDS:
        r=[x for x in rows if x[0]==name]
        tp=sum(hi>=thresh[k] for _,k,hi,_,_,_ in r)
        useful=sum(hi>=thresh[k] and g>=3 for _,k,hi,_,_,g in r)
        fp=sum(lo>=thresh[j] for _,_,_,j,lo,_ in r)
        n=len(r)
        metrics[name]=dict(accuracy=float((tp+n-fp)/(2*n)),f1=float(2*tp/(n+tp+fp)),useful=float(useful/n),
                           tp=int(tp),fp=int(fp),tn=int(n-fp),fn=int(n-tp),n_useful=int(useful))
    return metrics

def run():
    train=json.loads((ROOT/'round2/semantic/cohort.json').read_text())
    head=m.tf_head(); apple=m.apple.load()
    records=[]
    for kind in apple_train.KINDS:
        for scene in v2.sources(train,'validation',kind):
            meta=scene[2]
            stat,vec,prior,gain=vectors_for_mix(scene)
            if len(stat):
                apples=np.stack([m.apple.estimate(s,apple) for s in stat])
                tf=m.transfer.stem_logits(vec,prior,head,float(head['prior_weight']))
            else:apples=tf=np.empty((0,5))
            records.append((kind,m.CLASSES.index(meta['target']),m.CLASSES.index(meta['absent']),apples,tf,gain))
    candidates=[]
    # Class-wise adjustments are regularized towards a shared threshold. The grid is deliberately small.
    thresholds=np.arange(.15,.56,.05)
    for blend in (.2,.4,.6,.8):
        raw=[]
        for kind,k,j,apples,tf,gain in records:
            if len(apples):
                prob=m.probability(apples,tf,blend)
                idx=np.argmax(prob[:,k]); high,low,d=prob[idx,k],max(prob[:,j]),gain[idx]
            else:high=low=d=-1e8
            raw.append((kind,k,high,j,low,d))
        for shared in thresholds:
            per=[]
            for k in range(len(m.CLASSES)):
                def objective(h):
                    hits=[hi>=h for _,c,hi,_,_,_ in raw if c==k]
                    fps=[lo>=h for _,_,_,j,lo,_ in raw if j==k]
                    return np.mean(hits)-np.mean(fps)-.75*abs(h-shared)
                per.append(max(thresholds,key=objective))
            for strength in (0.,.5):
                th=np.full(5,shared)*(1-strength)+np.array(per)*strength
                metrics=results(raw,th)
                quality=(min(v[k] for v in metrics.values() for k in ('accuracy','f1')),
                         min(v['useful'] for v in metrics.values()),-strength)
                candidates.append((quality,blend,th,metrics))
    _,blend,th,report=max(candidates,key=lambda r:r[0])
    np.savez_compressed(m.HERE/'calibration.npz',blend=blend,threshold=th)
    files=[m.HERE/'calibration.npz',m.HERE/'model.py',m.apple.HERE/'weights.npz',m.transfer.HERE/'head.npz',m.transfer.HERE/'yamnet.onnx']
    info=dict(method='fusion',metrics=report,params=dict(blend=blend,threshold=th.tolist()),
              sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (m.HERE/'freeze.json').write_text(json.dumps(info,indent=2))
    print(report,blend,th,flush=True)
if __name__=='__main__':run()
