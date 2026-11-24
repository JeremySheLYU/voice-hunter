"""Audit all validation scores and replay ordinary entry-point on cached/uncached geometry routes."""
from pathlib import Path
import sys,json,time,hashlib,tempfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3b.model import model as m
from round3b.model.encode_train import CACHE
from round3b.model.dev_features import run as prepare
from round2.semantic.evaluate import sources,sisdr
from round2.semantic.semantic_candidate import candidates


def run():
    weights=m.load_model();locked=json.loads((HERE/'freeze.json').read_text());times=[];checks=[]
    by={};allrows=[]
    cohort=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    for kind in ('persistent','intermittent'):
        seen={'basic':0,'fallback':0}
        for x,t,meta in sources(cohort,'validation',kind):
            path=CACHE/(kind+meta['source_id']+'.npz')
            with np.load(path) as f:d=dict(f)
            e=m.stem_evidence(d['ast'],d['apple'],d['yam'],d['prior'],weights)
            probs=m.posterior(e,weights,float(weights['smoothing']))
            emitted=[probs[:,k].max()>=weights['thresholds'][k] for k in range(len(m.CLASSES))]
            try:candidates(x);route='basic'
            except ValueError:route='fallback'
            if seen[route]<2:
                tic=time.perf_counter();out=m.predict_scene(x,48000,weights);times.append(time.perf_counter()-tic)
                for k,label in enumerate(m.CLASSES):
                    if emitted[k]!=(out[label] is not None):raise AssertionError('feature and waveform replay differ')
                checks.append(dict(kind=kind,source=meta['source_id'],route=route));seen[route]+=1
            for idfield,pr in [('target',True),('absent',False)]:
                req=meta[idfield];k=m.CLASSES.index(req);i=np.argmax(probs[:,k])
                delta=float(d['gain'][i]) if pr and emitted[k] else None
                allrows.append(dict(kind=kind,source=meta['source_id'],request=req,present=pr,returned=bool(emitted[k]),gain=delta))
        for c in m.CLASSES:
            group=[r for r in allrows if r['kind']==kind and r['request']==c];pos=[r for r in group if r['present']];neg=[r for r in group if not r['present']]
            by.setdefault(kind,{})[c]=dict(positive=len(pos),negative=len(neg),tp=sum(r['returned'] for r in pos),fp=sum(r['returned'] for r in neg),
                                            useful=sum(r['returned'] and r['gain']>=3 for r in pos))
    rec=dict(validation_classes=by,full_waveform_checked=checks,
             warm_and_first_inference_seconds=times,median_seconds=float(np.median(times)),rows=allrows)
    (HERE/'replay.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(by,checks,times)
if __name__=='__main__':run()
