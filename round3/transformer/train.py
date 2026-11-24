"""Fit the 75-example AST head and select its mixture score and delivery threshold."""
import os,time,sys,json,hashlib,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'round2/semantic'))
from round2.semantic import evaluate as old
from round3.classifier.train import score
from round3.transformer import model
from round3.transformer.model import CLASSES as C

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def run():
    base=json.loads((ROOT/'round2/semantic/cohort.json').read_text())
    val=[r for r in base if r['split']=='validation']
    rows=[r for r in base if r['split']=='train']+json.loads((ROOT/'round3/data/train.json').read_text())
    assert all(r['split']=='train' for r in rows)
    key=lambda r:hashlib.sha256(('vh3-ast-train|'+str(r['id'])).encode()).hexdigest()
    selected=[r for c in C for r in sorted([v for v in rows if v['category']==c],key=key)[:15]]
    write(HERE/'train_sources.json',[{k:r[k] for k in ['id','category','author','sha256','path','license']} for r in selected])
    cache=Path(tempfile.gettempdir())/'voicehunter-r3-ast-logits'
    cache.mkdir(exist_ok=True)
    tag=digest(HERE/'model.py')[:12]
    path=cache/f'training-{tag}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as f:train=f['features']
    else:
        features=[];t0=time.monotonic()
        for i,r in enumerate(selected):
            assert digest(ROOT/r['path'])==r['sha256']
            wave=old.sound(r['path'])
            n=min(len(wave),216000);a=(len(wave)-n)//2
            features.append(model.encode(wave[a:a+n]))
            if (i+1)%5==0:print('train',i+1,round(time.monotonic()-t0,1),'s',flush=True)
        train=np.stack(features);np.savez_compressed(path,features=train)
    weights=model.fit(train,[C.index(r['category']) for r in selected])
    results=[]
    for kind in ('persistent','intermittent'):
        for i,(x,t,ids) in enumerate(old.sources(val,'validation',kind)):
            meta=dict(id=ids['source_id'],label=ids['target'],absent=ids['absent'],kind=kind,deltas=[])
            try:src,mix,_,_=model.candidates(x)
            except ValueError:results.append((meta,np.zeros((0,527))));continue
            slot=cache/f"{tag}-{kind}-{ids['source_id']}.npz"
            if slot.exists():
                with np.load(slot) as f:feats=f['f']
            else:
                feats=np.stack([model.encode(s) for s in src.T]);np.savez_compressed(slot,f=feats)
            sl=slice(int(48000*(2.45 if kind=='intermittent' else .15)),int(4.35*48000))
            for j in (0,1):meta['deltas'].append(old.sisdr((src[:,j,None]*mix[:,j])[sl],t[sl])-old.sisdr(x[sl],t[sl]))
            results.append((meta,feats))
            if (i+1)%5==0:print(kind,i+1,flush=True)
    best=(-1.,-1.,-1.)
    for blend in (0.,.25,.5,.75,1.):
        pp=[(meta,model.scores(f,weights,blend) if len(f) else np.array([])) for meta,f in results]
        for th in np.arange(.20,.851,.01):
            metrics=score(pp,float(th))
            accuracy=min(min(r['accuracy'],r['f1']) for r in metrics.values())
            rank=(accuracy,min(r['useful'] for r in metrics.values()),np.mean([r['useful'] for r in metrics.values()]))
            if rank>best:
                best=rank;chosen=dict(blend=blend,threshold=float(th),metrics=metrics)
    np.savez_compressed(HERE/'weights.npz',**weights,blend=chosen['blend'],threshold=chosen['threshold'])
    write(HERE/'validation.json',chosen)
    predictions=[dict(meta=meta,source_scores=model.scores(f,weights,chosen['blend']).tolist() if len(f) else []) for meta,f in results]
    write(HERE/'validation_predictions.json',predictions)
    write(HERE/'freeze.json',dict(hashes={n:digest(HERE/n) for n in ('ast-int8.onnx','model.py','train.py','weights.npz','PROTOCOL.md','train_sources.json')},validation=chosen))
    print('FROZEN',chosen,flush=True)

if __name__=='__main__':run()
