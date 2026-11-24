"""Development-only spatial contrast, recognition and image oracle audit."""
import sys,json,time,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3.mixture.jointdiag import image_pair
from round3.transfer.candidate import Encoder,CLASSES,HERE as HEAD_DIR,select_stem,stem_logits
import evaluate as prior

def run():
    rows=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    enc=Encoder();cache=Path(tempfile.gettempdir())/'vh-r3-jointdiagonalizer-v1';cache.mkdir(exist_ok=True)
    with np.load(HEAD_DIR/'head.npz') as f:head=dict(f)
    summary={}
    for kind in ('persistent','intermittent','room'):
        n=tp=fp=usable=harm=0;o=[];q=[];rows_out=[]
        for mix,t,m in prior.sources(rows,'validation',kind):
            path=cache/(kind+m['source_id']+'.npz')
            if not path.exists():
                imgs,quality=image_pair(mix,True)
                fts=[enc.encode(s[:,np.argmax(np.mean(s**2,axis=0))]) for s in imgs];z,p=[np.stack(v) for v in zip(*fts)]
                cut=slice(int((2.45 if kind=='intermittent' else .15)*48000),int(4.35*48000))
                base=prior.sisdr(mix[cut],t[cut]);d=[prior.sisdr(s[cut],t[cut])-base for s in imgs]
                np.savez(path,z=z,p=p,q=quality,d=d)
            with np.load(path) as f:result=dict(f)
            d=result['d'];quality=result['q'];logits=stem_logits(result['z'],result['p'],head,float(head['prior_weight']))
            k=select_stem(logits,m['target'],float(head['threshold']))
            fals=select_stem(logits,m['absent'],float(head['threshold'])) is not None
            n+=1;tp+=int(k is not None);fp+=int(fals)
            if k is not None:usable+=int(d[k]>=3);harm+=int(d[k]<0)
            o.append(max(d));q.append(quality)
            rows_out.append(dict(**m,delta=None if k is None else float(d[k]),absent_returned=bool(fals),quality=quality.tolist()))
        summary[kind]=dict(n=n,tp=tp,fp=fp,f1=2*tp/(tp+n+fp),acc=(tp+n-fp)/(2*n),useful=usable,harmful=harm,
                           oracle_useful=sum(int(v>=3) for v in o),oracle_median=float(np.median(o)),
                           quality_percentiles=np.quantile(q,[0,.1,.5,.9,1],axis=0).tolist(),rows=rows_out)
        print(kind,{k:v for k,v in summary[kind].items() if k!='rows'},flush=True)
        (HERE/'jointdiag_validation.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':run()
