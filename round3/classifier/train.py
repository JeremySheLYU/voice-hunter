"""Fit on training files and choose one delivery threshold on two validation strata."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
# The original evaluation module's local import is unqualified.
sys.path.append(str(ROOT/'round2/semantic'))
from round2.semantic import evaluate as protocol
from round3.classifier import model
from round3.classifier.model import CLASSES

CACHE=Path(tempfile.gettempdir())/'voicehunter-r3-spectral-cache'
FEATURE_TAG=hashlib.sha256((HERE/'model.py').read_bytes()).hexdigest()


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def training_feature(row):
    path=ROOT/row['path']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=row['sha256']: raise ValueError(f"Checksum mismatch for {row['id']}")
    cache=CACHE/f"{digest}-{FEATURE_TAG[:12]}.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as a: return a['features']
    p=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vn','-t','10',
                      '-ac','1','-ar','48000','-f','f32le','pipe:1'],check=True,capture_output=True)
    audio=np.frombuffer(p.stdout, dtype='<f4')
    features=model.describe(audio,48000)
    # The evaluation uses clipped excerpts. Include descriptors of a centred
    # excerpt for longer files, without allowing long recordings extra vote.
    if len(audio)>240000:
        n=216000
        a=(len(audio)-n)//2
        features=np.vstack([features,model.describe(audio[a:a+n],48000)[0]])
    with tempfile.NamedTemporaryFile(dir=CACHE, suffix='.npz', delete=False) as temporary:
        np.savez_compressed(temporary,features=features)
    os.replace(temporary.name, cache)
    return features


def validation_rows(rows,weights):
    valcache=CACHE/f"mixtures-{FEATURE_TAG[:12]}-{hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()[:16]}-v1.npz"
    if valcache.exists():
        with np.load(valcache,allow_pickle=False) as a:
            features=a['features']; counts=a['counts']; meta=json.loads(str(a['rows']))
    else:
        packed=[];counts=[];meta=[]
        for kind in ('persistent','intermittent'):
            for x,t,ids in protocol.sources(rows,'validation',kind):
                event=dict(kind=kind, label=ids['target'], absent=ids['absent'], id=ids['source_id'],deltas=[])
                try:
                    sources,mixing,_,_=model.candidates(x)
                    feats=[model.describe(s,48000) for s in sources.T]
                except ValueError:
                    counts.extend((0,0));meta.append(event);continue
                sl=slice(int(48000*(2.45 if kind=='intermittent' else .15)),int(48000*4.35))
                for i,f in enumerate(feats):
                    counts.append(len(f)); packed.append(f)
                    stem=sources[:,i,None]*mixing[:,i]
                    event['deltas'].append(float(protocol.sisdr(stem[sl],t[sl])-protocol.sisdr(x[sl],t[sl])))
                meta.append(event)
        features=np.vstack(packed)
        np.savez_compressed(valcache,features=features,counts=counts,rows=json.dumps(meta))
    ix=0; out=[]
    for j,row in enumerate(meta):
        scores=[]
        for n in counts[2*j:2*j+2]:
            n=int(n)
            if n: scores.append(model.scores_from_features(features[ix:ix+n],weights))
            ix+=n
        out.append((row,np.asarray(scores)))
    return out


def score(rows,threshold):
    out={}
    for kind in ('persistent','intermittent'):
        tp=fp=fn=tn=useful=0
        for row,p in rows:
            if row['kind']!=kind:continue
            label=CLASSES.index(row['label']);absent=CLASSES.index(row['absent'])
            if len(p) and p[:,label].max()>=threshold:
                tp+=1
                if row['deltas'][int(np.argmax(p[:,label]))]>=3: useful+=1
            else: fn+=1
            if len(p) and p[:,absent].max()>=threshold: fp+=1
            else: tn+=1
        out[kind]=dict(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=(tp+tn)/(tp+fn+fp+tn),f1=2*tp/max(1,2*tp+fp+fn),
                       useful=useful/(tp+fn),false_positive_rate=fp/(fp+tn))
    return out


def run(extra,final=False):
    CACHE.mkdir(parents=True,exist_ok=True)
    with open(ROOT/'round2/semantic/cohort.json') as f:
        rows=[r for r in json.load(f) if r['split'] in ('train','validation')]
    train=[r for r in rows if r['split']=='train']; validation=[r for r in rows if r['split']=='validation']
    if extra:
        with open(extra) as f: more=json.load(f)
        if any(r.get('split','train')!='train' for r in more): raise ValueError("Expected training-only manifest")
        train.extend(more)
    if len(set(r['sha256'] for r in train+validation))!=len(train+validation): raise ValueError('Repeated source hash')
    if len(set(str(r['id']) for r in train+validation))!=len(train+validation): raise ValueError('Repeated FSD id')
    if {r['author'].strip().lower() for r in train} & {r['author'].strip().lower() for r in validation}:
        raise ValueError('Training / validation uploader overlap')
    with ThreadPoolExecutor(max_workers=3) as pool:
        features=list(pool.map(training_feature,train))
    labels=np.array([CLASSES.index(r['category']) for r in train])
    weights=model.fit(features,labels)
    print(f"Fitted {len(train)} files with {sum(map(len,features))} feature windows, starting validation",flush=True)
    results=validation_rows(validation,weights)
    grid=[]
    for t in np.linspace(.08,.74,67):
        res=score(results,float(t))
        grid.append(dict(threshold=float(t),metrics=res))
    def criterion(v):
        strata=list(v['metrics'].values())
        return (min(min(r['accuracy'],r['f1']) for r in strata),np.mean([r['useful'] for r in strata]),
                -np.mean([r['false_positive_rate'] for r in strata]))
    best=max(grid,key=criterion)
    tally=dict(training=len(train),validation_ids=len(validation),class_categories=list(CLASSES),
               training_per_class=np.bincount(labels).tolist(),descriptors=sum(map(len,features)),
               train_file_ids=[r['id'] for r in train],threshold=best['threshold'],chosen=best['metrics'],
               selection='maximize worst across both strata and accuracy/F1, breaking ties by useful fraction then false return rate')
    dest=HERE/'weights.npz' if final else CACHE/'ablation-weights.npz'
    np.savez_compressed(dest,**weights,threshold=best['threshold'])
    json_write(HERE/('validation.json' if final else 'baseline_validation.json'),tally)
    if final:
        json_write(HERE/'validation_predictions.json',[dict(meta=meta,source_scores=scores.tolist()) for meta,scores in results])
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'model.py',HERE/'train.py',dest)}
        json_write(HERE/'freeze.json',dict(hashes=hashes,validation=best,weights='weights.npz',
                   train_source_manifests=['round2/semantic/cohort.json']+([str(extra)] if extra else [])))
    print(json.dumps(best),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extra',type=Path)
    p.add_argument('--freeze',action='store_true')
    a=p.parse_args();run(a.extra,a.freeze)
