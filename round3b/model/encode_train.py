"""Only materialize development training predictors; no final-set material."""
from pathlib import Path
import sys,hashlib,json,tempfile,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3.transformer import model as ast
from round3.apple_features import train as apple
from round2.semantic.evaluate import sound
C=ast.CLASSES
CACHE=Path(tempfile.gettempdir())/'voicehunter-r3b-calibration'
CACHE.mkdir(exist_ok=True)
def corpus():
    return [r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='train']+json.loads((ROOT/'round3/data/train.json').read_text())
def selected():
    rows=corpus()
    digest=lambda r:hashlib.sha256(('vh3-ast-train|'+str(r['id'])).encode()).hexdigest()
    return [r for c in C for r in sorted([v for v in rows if v['category']==c],key=digest)[:45]]
def run():
    rows=selected()
    orig=json.loads((ast.HERE/'train_sources.json').read_text())
    schema=hashlib.sha256((ast.HERE/'model.py').read_bytes()).hexdigest()[:12]
    old=Path(tempfile.gettempdir())/'voicehunter-r3-ast-logits'/f'training-{schema}.npz'
    original={}
    if old.is_file():
        with np.load(old) as f:original={r['sha256']:z for r,z in zip(orig,f['features'])}
    t0=time.perf_counter()
    for i,r in enumerate(rows):
        p=CACHE/(r['sha256']+'-ast.npy')
        if p.exists():continue
        if r['sha256'] in original:z=original[r['sha256']]
        else:
            x=sound(r['path']);n=min(len(x),216000);k=(len(x)-n)//2
            z=ast.encode(x[k:k+n])
        np.save(p,z)
        if i%15==0:print('training cached',i,round(time.perf_counter()-t0,1),flush=True)
    (HERE/'training_sources.json').write_text(json.dumps([{k:r[k] for k in ('id','sha256','author','category','license','path')} for r in rows],indent=1)+'\n')
if __name__=='__main__':run()
