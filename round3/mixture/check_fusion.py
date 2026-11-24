"""Score only validation recordings after blind separation and transfer recognition."""
from pathlib import Path
import sys,json,time,tempfile,hashlib
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
import evaluate as prior
from round3.mixture.transfer_fusion import estimate_images
from round3.transfer.candidate import Encoder, CLASSES, HERE as HEAD_DIR,stem_logits,select_stem


def cached_examples(kinds):
    encoder=Encoder();tic=time.perf_counter()
    cohort=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    signature=hashlib.sha256((HERE/'model.py').read_bytes()+(HERE/'transfer_fusion.py').read_bytes()).hexdigest()[:12]
    cache=Path(tempfile.gettempdir())/('voicehunter-mixture-fusion-'+signature)
    cache.mkdir(exist_ok=True)
    for kind in kinds:
        for mix,t,meta in prior.sources(cohort,'validation',kind):
            path=cache/(kind+'-'+meta['source_id']+'.npz')
            if not path.exists():
                try:
                    images,method=estimate_images(mix,48000)
                except ValueError:
                    vectors=priors=deltas=np.array([]);method='no contrast'
                else:
                    area=slice(int((2.45 if kind=='intermittent' else .15)*48000),int(4.35*48000))
                    base=prior.sisdr(mix[area],t[area])
                    z=[encoder.encode(x[:,np.argmax(np.mean(x**2,axis=0))]) for x in images]
                    vectors,priors=[np.stack(v) for v in zip(*z)]
                    deltas=[prior.sisdr(x[area],t[area])-base for x in images]
                np.savez(path,z=vectors,p=priors,deltas=deltas,method=method)
            with np.load(path,allow_pickle=False) as data:
                yield meta,dict(data)
        print('encoded',kind,round(time.perf_counter()-tic,2),flush=True)


def summarize(rows):
    results={}
    for kind in sorted(set(r['kind'] for r in rows)):
        group=[r for r in rows if r['kind']==kind]
        n=len(group); tp=sum(r['returned'] for r in group);fp=sum(r['absent_returned'] for r in group)
        useful=sum(r['improvement'] is not None and r['improvement']>=3 for r in group)
        harms=sum(r['improvement'] is not None and r['improvement']<0 for r in group)
        deltas=[r['improvement'] for r in group if r['improvement'] is not None]
        results[kind]=dict(n=n,tp=tp,fp=fp,accuracy=(tp+n-fp)/(2*n),f1=2*tp/(tp+n+fp),useful=useful,
                 harmful=harms,median_improvement=float(np.median(deltas)) if deltas else None,
                 oracle_useful=sum(r['oracle_useful'] for r in group),paths=dict(Counter(r['method'] for r in group)))
    return results


def run():
    samples=list(cached_examples(('room','persistent','intermittent','colocated')))
    with np.load(HEAD_DIR/'head.npz',allow_pickle=False) as f:head=dict(f)
    rows=[]
    for meta,data in samples:
        true_ret=neg_ret=oracle=False;delta=None
        if len(data['z']):
            logits=stem_logits(data['z'],data['p'],head,float(head['prior_weight']))
            pos=select_stem(logits,meta['target'],float(head['threshold']))
            neg=select_stem(logits,meta['absent'],float(head['threshold']))
            true_ret=pos is not None;neg_ret=neg is not None
            if true_ret:delta=float(data['deltas'][pos])
            oracle=bool(np.max(data['deltas'])>=3)
        rows.append(dict(**meta,method=str(data['method']),returned=true_ret,absent_returned=neg_ret,
                         improvement=delta,oracle_useful=oracle))
    results=summarize(rows)
    (HERE/'fusion_validation.json').write_text(json.dumps(dict(summary=results,rows=rows),indent=2)+'\n')
    print(json.dumps(results),flush=True)

if __name__=='__main__':run()
