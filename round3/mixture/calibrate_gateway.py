"""Freeze the predefined directness gate and combine the two trained heads."""
from pathlib import Path
import sys,json,tempfile,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3.transfer.candidate import Encoder,CLASSES,HERE as TRANSFER_DIR,stem_logits
from round3.mixture.mix_model import predict_raw,confidence
import evaluate as prior

def run():
    enc=Encoder()
    with np.load(TRANSFER_DIR/'head.npz') as f:stem_head=dict(f)
    with np.load(HERE/'mix_head.npz') as f:mix_head=dict(f)
    if 'cohort_hash' not in mix_head:raise ValueError('Expected the 625-source frozen mixture head')
    cohort=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    cache=Path(tempfile.gettempdir())/'vh-r3-jointdiagonalizer-v1'
    kinds=['persistent','intermittent'];bundle={}
    for kind in kinds:
        records=[]
        for x,t,m in prior.sources(cohort,'validation',kind):
            with np.load(cache/(kind+m['source_id']+'.npz')) as f:record=dict(f)
            raw,p=predict_raw(enc,mix_head,x)
            record.update(clip=confidence(raw,p,mix_head),label=CLASSES.index(m['target']),absent=CLASSES.index(m['absent']))
            records.append(record)
        bundle[kind]={k:np.stack([r[k] for r in records]) for k in records[0]}
    best=None
    for head_weight in (0,.25,.5,.75,1):
      for mix_weight in (0,.15,.30,.45):
        runs={}
        for kind,d in bundle.items():
            logits=stem_logits(d['z'],d['p'],stem_head,head_weight)
            prob=np.exp(logits-logits.max(axis=2,keepdims=True));prob/=prob.sum(axis=2,keepdims=True)
            scores=(1-mix_weight)*prob.max(axis=1)+mix_weight*d['clip']
            idx=np.arange(len(d['label']));pos_id=prob[idx,:,d['label']].argmax(axis=1)
            gain=d['d'][idx,pos_id];allowed=(d['q'][:,0]<=.16)&(d['q'][:,1]>=.05)
            runs[kind]=(scores[idx,d['label']],scores[idx,d['absent']],gain,allowed)
        for th in np.arange(.15,.701,.025):
            metric={}
            for kind,(a,b,g,ok) in runs.items():
                n=len(a);tp=int(((a>=th)&ok).sum());fp=int(((b>=th)&ok).sum());u=int(((a>=th)&ok&(g>=3)).sum())
                metric[kind]=dict(n=n,tp=tp,fp=fp,accuracy=(tp+n-fp)/(2*n),f1=2*tp/(tp+fp+n),useful=u,harmful=int(((a>=th)&ok&(g<0)).sum()))
            key=(min(min(r['accuracy'],r['f1']) for r in metric.values()),min(r['useful']for r in metric.values()),-mix_weight,-head_weight)
            if best is None or key>best[0]:best=(key,dict(stem_weight=head_weight,mixture_weight=mix_weight,threshold=round(float(th),5),validation=metric))
    (HERE/'gateway_params.json').write_text(json.dumps(best[1],indent=2)+'\n')
    print(json.dumps(best[1]))
if __name__=='__main__':run()
