"""Train spectral dictionaries and calibrate the two paths on validation only."""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3.mixture import model
import evaluate as prior

DATA = ROOT/'round2/semantic/cohort.json'
CLASSES=model.CLASSES


def dictionary(rows):
    rng=np.random.default_rng(8523)
    by_class={}
    bins=model.band_map()
    for c in CLASSES:
        snippets=[]
        for r in rows:
            if r['category']!=c or r['split']!='train':continue
            x=prior.sound(r['path'])
            spec=abs(model.stft(x,model.NFFT,model.HOP)[:,:,0])**2
            mag=np.zeros((int(bins.max())+1,spec.shape[1]),dtype='f4')
            np.add.at(mag,bins,spec)
            mag=np.sqrt(mag).T
            mag=mag[(mag**2).sum(axis=1)>.01*(mag**2).sum(axis=1).max()]
            if len(mag):
                snippets.extend(mag[np.linspace(0,len(mag)-1,min(70,len(mag)),dtype=int)])
        by_class[c]=np.stack(snippets)
    combined=np.concatenate(list(by_class.values()))
    scale=np.sqrt(combined.mean(axis=0).clip(.03))**-1
    templates=[];classes=[]
    for i,c in enumerate(CLASSES):
        x=by_class[c]*scale
        x /= np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8)
        centers=x[rng.choice(len(x),48,replace=False)].copy()
        for _ in range(20):
            labels=(x@centers.T).argmax(axis=1)
            for j in range(len(centers)):
                chosen=x[labels==j]
                if len(chosen): centers[j]=chosen.mean(axis=0)
            centers/=np.linalg.norm(centers,axis=1,keepdims=True).clip(1e-8)
        templates.append(centers.T);classes.extend([i]*len(centers))
    return dict(dictionary=np.concatenate(templates,axis=1),scale=scale,class_index=np.array(classes),
                threshold_dict=np.float64(.3),threshold_iva=np.float64(.30))


def metrics(rows,threshold):
    pos=[r for r in rows if r['present']];neg=[r for r in rows if not r['present']]
    tp=int(sum(r['score']>=threshold for r in pos));fp=int(sum(r['score']>=threshold for r in neg))
    return dict(accuracy=(tp+len(neg)-fp)/(len(pos)+len(neg)),f1=2*tp/max(1,tp+len(pos)+fp),
                useful=int(sum(r['score']>=threshold and r['improvement']>=3 for r in pos)),
                n_pos=len(pos),tp=tp,fp=fp,threshold=round(float(threshold),4))


def run():
    cohort=[r for r in json.loads(DATA.read_text()) if r['split'] in ('train','validation')]
    checkpoint=HERE/'validation_rows.json'
    print('training',flush=True)
    weights=dict(np.load(HERE/'weights.npz')) if '--resume' in sys.argv else dictionary(cohort)
    np.savez_compressed(HERE/'weights.npz',**weights)
    audits=json.loads(checkpoint.read_text()) if '--resume' in sys.argv and checkpoint.exists() else [];tic=time.perf_counter()
    for kind in ('room','colocated','persistent'):
        if sum(r['kind']==kind for r in audits)==110:continue
        for j,(mix,t,meta) in enumerate(prior.sources(cohort,'validation',kind)):
            span=slice(int(.15*48000),int(4.35*48000));base=prior.sisdr(mix[span],t[span])
            method,state,prob,indices=model.analyse(mix,48000)
            for present,label in [(True,meta['target']),(False,meta['absent'])]:
                k=CLASSES.index(label)
                r=dict(kind=kind,method=method,source=meta['source_id'],target=label,present=present,score=float(prob[k]))
                if present:
                    y=state[indices[k]] if method=='iva' else model.dictionary_stem(mix,state,k)
                    r['improvement']=prior.sisdr(y[span],t[span])-base
                audits.append(r)
            if j%12==0:print(kind,j,round(time.perf_counter()-tic,1),flush=True)
        checkpoint.write_text(json.dumps(audits,indent=2)+'\n')
    optimal={}
    for method in ('iva','dict'):
        subset=[r for r in audits if r['method']==method]
        if not subset:continue
        grid=[metrics(subset,q) for q in np.arange(.05,.81,.025)]
        # Prefer useful stems; within near-ties, choose better event F1.
        eligible=[g for g in grid if g['fp']<=.22*sum(not r['present'] for r in subset)]
        best=max(eligible or grid,key=lambda g:(g['useful']+.25*g['f1']*g['n_pos'],g['accuracy']))
        weights['threshold_'+method]=np.float64(best['threshold']);optimal[method]=best
    np.savez_compressed(HERE/'weights.npz',**weights)
    result={}
    for kind in ('room','colocated','persistent'):
        group=[dict(r,score=float(r['score']>=weights['threshold_'+r['method']])) for r in audits if r['kind']==kind]
        result[kind]=metrics(group,.5)
    manifest=dict(validation=result,calibration=optimal,seconds=round(time.perf_counter()-tic,2),
                  train_sources=sum(r['split']=='train' for r in cohort),classes=CLASSES)
    (HERE/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':run()
