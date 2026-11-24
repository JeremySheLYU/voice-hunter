"""Fit on training recordings only. Select thresholds on the two original validation strata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from round3.apple_features import model as m

ROOT = m.HERE.parents[1]
sys.path.insert(0,str(ROOT/'round2/semantic'))
from round2.semantic import evaluate as v2

CACHE = Path(tempfile.gettempdir())/'voicehunter-r3-applefeatures'
CACHE.mkdir(exist_ok=True)
KINDS = ('persistent','intermittent')


def get_record(row):
    key = CACHE/f"source-{row['sha256']}.npy"
    if not key.exists():
        wave = v2.sound(row['path'])
        if len(wave)>int(m.RATE*4.5):
            start=(len(wave)-int(m.RATE*4.5))//2
            wave=wave[start:start+int(m.RATE*4.5)]
        feat=m.describe(wave)
        np.save(key,feat)
    return np.load(key), m.CLASSES.index(row['category'])


def get_scene(scene):
    mix, ref, meta = scene
    key=CACHE/f"validation-{meta['kind']}-{meta['source_id']}-{meta['background_id']}.npz"
    if key.exists():
        with np.load(key) as d:
            return meta, d['stat'], d['deltas']
    try:
        sources, mixing, stats = m.prepare(mix,m.RATE)
    except ValueError:
        stats, deltas = np.empty((0,2,len(m.label_order()))), np.empty(0)
    else:
        sl=slice(int(m.RATE*(2.45 if meta['kind']=='intermittent' else .15)),int(m.RATE*4.35))
        base=v2.sisdr(mix[sl],ref[sl])
        deltas=np.array([v2.sisdr(sources[sl,i,None]*mixing[:,i],ref[sl])-base for i in range(2)])
    np.savez_compressed(key,stat=stats,deltas=deltas)
    return meta,stats,deltas


def summarise(records,threshold):
    metrics={}
    for name in KINDS:
        rows=[r for r in records if r[0]==name]
        positive=np.array([r[1]>=threshold for r in rows])
        absent=np.array([r[2]>=threshold for r in rows])
        tp,fp=int(positive.sum()),int(absent.sum())
        useful=sum(hit and row[3]>=3. for hit,row in zip(positive,rows))
        n=len(rows)
        metrics[name]=dict(accuracy=(tp+n-fp)/(2*n),f1=2*tp/(n+tp+fp),useful=useful/n,
                           tp=tp,fp=fp,tn=n-fp,fn=n-tp,n_useful=int(useful))
    return metrics


def run(extra,workers,prep):
    rows=json.loads((ROOT/'round2/semantic/cohort.json').read_text())
    train=[r for r in rows if r['split']=='train']
    if extra:
        new=json.loads(Path(extra).read_text())
        if any(r['split']!='train' for r in new):raise ValueError('New input must be training-only.')
        orig_author={r['author'].strip().casefold() for r in rows if r['split']!='train'}
        if orig_author & {r['author'].strip().casefold() for r in new}:raise ValueError('Uploader collision.')
        train+=new
    with ThreadPoolExecutor(max_workers=workers) as pool:
        examples=list(pool.map(get_record,train))
        print('Training sources processed:',len(examples),flush=True)
        # Iterating each stratum separately bounds wave buffers with threaded execution.
        scenes=[]
        for kind in KINDS:
            for batch in iter_batches(v2.sources(rows,'validation',kind),max(workers*2,4)):
                scenes.extend(pool.map(get_scene,batch))
        print('Validation scenes:',len(scenes),flush=True)
    if prep:return
    labels=[lab for _,lab in examples]
    x=np.stack([s for s,_ in examples])
    label_order=np.array(m.label_order())
    results=[]
    for reg in (.08,.4,1.5):
        basis,weights=m.fit(x,labels,reg)
        for blend in (0.,.25,.5,.75,1.):
            model=dict(basis=basis,weights=weights,blend=blend,labels=label_order)
            rec=[]
            for meta,stats,deltas in scenes:
                ks=[m.CLASSES.index(meta[l]) for l in ('target','absent')]
                if len(stats):
                    scores=np.array([m.estimate(stat,model) for stat in stats])
                    best=scores[:,ks].max(axis=0)
                    d=deltas[int(np.argmax(scores[:,ks[0]]))]
                else:best=[-1e9,-1e9];d=-1e9
                rec.append((meta['kind'],*best,d))
            for threshold in (.08,.12,.16,.20,.24,.28,.32,.36,.40,.45,.50,.55,.60):
                report=summarise(rec,threshold)
                quality=(min(v[k] for v in report.values() for k in ('accuracy','f1')),
                         min(v['useful'] for v in report.values()),min(v['accuracy'] for v in report.values()))
                results.append((quality,reg,blend,threshold,report))
    _,reg,blend,threshold,report=max(results,key=lambda r:r[0])
    basis,weights=m.fit(x,labels,reg)
    np.savez_compressed(m.HERE/'weights.npz',basis=basis,weights=weights,labels=label_order,
                        blend=np.array(blend),threshold=np.array(threshold))
    rows=[(r['id'],r['author'],r['category'],r['sha256']) for r in train]
    freeze=dict(method='apple_features', native_classifier='SNClassifierIdentifierVersion1',
                metrics=report,parameters=dict(reg=reg,blend=blend,threshold=threshold),training_ids=rows,
                weights_sha256=hashlib.sha256((m.HERE/'weights.npz').read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256((m.HERE/'model.py').read_bytes()).hexdigest())
    (m.HERE/'freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
    print(json.dumps(report),reg,blend,threshold,flush=True)


def iter_batches(rows, size):
    b=[]
    for r in rows:
        b.append(r)
        if len(b)==size:
            yield b
            b=[]
    if b:yield b


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--extra');p.add_argument('--workers',type=int,default=3);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    run(args.extra,args.workers,args.prepare_only)
