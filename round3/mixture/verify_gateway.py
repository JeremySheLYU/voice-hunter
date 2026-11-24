"""Fresh inference from the frozen model on each validation scene; no lookup features."""
from pathlib import Path
import json,time,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3.mixture.gateway import Predictor
import evaluate as prior


def run():
    t0=time.perf_counter();machine=Predictor();cold=time.perf_counter()-t0
    rows=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    expect=json.loads((HERE/'gateway_params.json').read_text())['validation']
    output={}
    for kind in ('persistent','intermittent','room','colocated'):
        n=tp=fp=useful=harmful=0;cost=[]
        for x,t,m in prior.sources(rows,'validation',kind):
            cut=slice(int((2.45 if kind=='intermittent' else .15)*48000),int(4.35*48000))
            start=time.perf_counter()
            info=machine.prepare(x,48000);a=info.predict(m['target']);b=info.predict(m['absent'])
            cost.append(time.perf_counter()-start);n+=1;tp+=int(a is not None);fp+=int(b is not None)
            if a is not None:
                if a.shape!=x.shape or not np.isfinite(a).all():raise ValueError('Invalid sound output')
                delta=prior.sisdr(a[cut],t[cut])-prior.sisdr(x[cut],t[cut]);useful+=int(delta>=3);harmful+=int(delta<0)
        result=dict(n=n,tp=tp,fp=fp,accuracy=(tp+n-fp)/(2*n),f1=2*tp/(n+tp+fp),useful=useful,harmful=harmful)
        if kind in expect and result!=expect[kind]:raise RuntimeError(f'Frozen mismatch: {result} vs {expect[kind]}')
        result.update(median_seconds=float(np.median(cost)),p95_seconds=float(np.quantile(cost,.95)))
        output[kind]=result;print(kind,result,flush=True)
    (HERE/'replay.json').write_text(json.dumps(dict(cold_model_load_seconds=cold,seconds_per_input=4.5,metrics=output),indent=2)+'\n')

if __name__=='__main__':run()
