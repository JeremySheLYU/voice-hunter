"""Use matching cached round-3 training/validation features when available."""
from pathlib import Path
import tempfile,json,hashlib,time,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3b.model.encode_train import CACHE
from round3.transformer import model as ast
from round3.apple_features import train as apple
from round3.fusion import train as fusion
from round3.mixture.jointdiag import image_pair
from round2.semantic.evaluate import sources,sisdr
from round2.semantic.semantic_candidate import candidates

def run():
    tag=hashlib.sha256((ast.HERE/'model.py').read_bytes()).hexdigest()[:12]
    original=Path(tempfile.gettempdir())/'voicehunter-r3-ast-logits'
    cohort=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    for kind in ('persistent','intermittent'):
        for x,t,m in sources(cohort,'validation',kind):
            assert kind in ('persistent','intermittent')
            slot=CACHE/(kind+m['source_id']+'.npz')
            if slot.exists():continue
            sign=original/(tag+'-'+kind+'-'+m['source_id']+'.npz')
            pair,quality=image_pair(x,True)
            try:
                sm,mix=fusion.m.demix(x)
                images=[sm[:,i,None]*mix[None,:,i] for i in (0,1)]
            except ValueError:
                images=pair
                sm=np.stack([s[:,np.argmax(np.mean(s**2,axis=0))] for s in pair],axis=1)
            if sign.exists():
                with np.load(sign) as f:z=f['f']
            else:z=np.stack([ast.encode(a) for a in sm.T])
            try:stats,vec,prior,gains=fusion.vectors_for_mix((x,t,m))
            except FileNotFoundError:stats=[]
            if len(stats)==0:
                stats=np.stack([fusion.m.apple.describe(a) for a in sm.T])
                rep=[fusion.m.encoder().encode(a) for a in sm.T]
                vec,prior=[np.stack(v) for v in zip(*rep)]
                cut=slice(int((2.45 if kind=='intermittent' else .15)*48000),int(4.35*48000))
                gains=[sisdr(a[cut],t[cut])-sisdr(x[cut],t[cut]) for a in images]
            np.savez_compressed(slot,ast=z,apple=stats,yam=vec,prior=prior,gain=gains,quality=quality)
        print('ready',kind,flush=True)
if __name__=='__main__':run()
