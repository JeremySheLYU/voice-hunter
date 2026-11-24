"""Fit on train-identity synthetic mixtures, choose a calibration on validation."""
from pathlib import Path
import sys,time,json,hashlib,tempfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
import evaluate as prior
from round3.mixture import model,masker
from round3.mixture.mix_model import predict_raw,confidence
from round3.transfer.candidate import Encoder,CLASSES

def run():
    t0=time.perf_counter()
    rng=np.random.default_rng(517051)
    rows=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split'] in ('train','validation')]
    extra=ROOT/'round3/data/train.json'
    if extra.exists():rows.extend(json.loads(extra.read_text()))
    pool={i:[r for r in rows if r['split']=='train' and r['category']==c] for i,c in enumerate(CLASSES)}
    enc=Encoder();fs=48000;n=3*fs
    signature=hashlib.sha256(''.join(r['sha256'] for r in rows if r['split']=='train').encode()).hexdigest()[:10]
    path=Path(tempfile.gettempdir())/('vh-mix-model-train-'+signature+'-v2.npz')
    if path.exists():
        with np.load(path) as d:v,target=np.array(d['v']),np.array(d['y'])
    else:
        v=[];target=[]
        all_train=[r for r in rows if r['split']=='train']
        rng.shuffle(all_train)
        from functools import lru_cache
        @lru_cache(maxsize=len(all_train))
        def compact(path):return prior.sound(path).astype('f2')
        for i in range(1600):
            count=2 if i%10<7 else (1 if i%10<9 else 3)
            first=CLASSES.index(all_train[i]['category']) if i<len(all_train) else int(rng.integers(5))
            cats=[first,*rng.choice([j for j in range(5) if j!=first],count-1,replace=False)]
            label=np.zeros(5);label[cats]=1
            mono=np.zeros(n,dtype='f4')
            for j,idx in enumerate(cats):
                r=all_train[i] if j==0 and i<len(all_train) else pool[idx][rng.integers(len(pool[idx]))]
                x=compact(r['path']);k=int(rng.integers(0,max(1,len(x)-n+1)))
                clip=x[k:k+n].astype('f4')
                if len(clip)<n:clip=np.pad(clip,(0,n-len(clip)))
                clip*=.08*10**(rng.uniform(-4,4)/20)/(np.sqrt((clip**2).mean())+1e-8)
                # Artificial early and late reflections; no measured test room in fitting.
                if i%2==0:
                    base=clip.copy()
                    for lag,g in zip(rng.integers(250,5500,5),rng.uniform(-.42,.42,5)):
                        clip[lag:]+=base[:-lag]*g
                mono+=clip
            z,_=enc.encode(mono)
            v.append(z);target.append(label)
            if i%150==0:print('train',i,round(time.perf_counter()-t0,1),flush=True)
        v=np.stack(v);target=np.stack(target);np.savez(path,v=v,y=target)
    d=v.astype('f8');kernel=np.clip(d@d.T,0,1)**3
    coef=np.linalg.solve(kernel+np.eye(len(d))*.25,target)
    head=dict(anchor=v.astype('f4'),alpha=coef.astype('f4'),cohort_hash=signature)
    examples=[]
    for kind in ('room','colocated','persistent','intermittent'):
        for mix,t,meta in prior.sources(rows,'validation',kind):
            raw,p=predict_raw(enc,head,mix)
            examples.append(dict(kind=kind,raw=raw,p=p,target=CLASSES.index(meta['target']),absent=CLASSES.index(meta['absent'])))
    # Calibrate detection only, weighting all four development conditions equally.
    def eval_gate(coef,th):
        tp=fp=0
        for e in examples:
            score=e['raw']+coef*e['p']
            tp+=int(score[e['target']]>=th);fp+=int(score[e['absent']]>=th)
        n=len(examples);return (2*tp/(tp+fp+n)+(tp+n-fp)/(2*n))/2,tp,fp
    metric,weight,th=max((eval_gate(coef,th)[0],coef,th) for coef in (0,.015,.03,.05,.075) for th in np.arange(.12,.81,.025))
    head.update(prior_weight=float(weight),threshold=float(th))
    np.savez_compressed(HERE/'mix_head.npz',**head)
    audit={};counter=0
    for kind in ('room','colocated','persistent','intermittent'):
        results=[];tic=time.perf_counter()
        for mix,t,m in prior.sources(rows,'validation',kind):
            entry=examples[counter];counter+=1;scores=confidence(entry['raw'],entry['p'],head)
            k=CLASSES.index(m['target']);returned=bool(scores[k]>=th);fa=bool(scores[CLASSES.index(m['absent'])]>=th)
            area=slice(int((2.45 if kind=='intermittent' else .15)*fs),int(4.35*fs))
            base=prior.sisdr(mix[area],t[area])
            images,_=model.nmf_components(mix,model.params())
            y=masker.output(mix,images,k,scores)
            imp=prior.sisdr(y[area],t[area])-base
            results.append(dict(**m,returned=returned,false_accept=fa,delta=imp))
        n=len(results);tp=sum(x['returned'] for x in results);fp=sum(x['false_accept'] for x in results)
        audit[kind]=dict(n=n,tp=tp,fp=fp,accuracy=(tp+n-fp)/(2*n),f1=2*tp/(n+tp+fp),
                         useful=sum(x['returned'] and x['delta']>=3 for x in results),harmful=sum(x['returned'] and x['delta']<0 for x in results),
                         sec_per_scene=round((time.perf_counter()-tic)/n,3),rows=results)
        print(kind,{k:v for k,v in audit[kind].items() if k!='rows'},flush=True)
        (HERE/'mix_validation.json').write_text(json.dumps(dict(conditions=audit,training_mixtures=len(d),training_ids=sum(len(x) for x in pool.values()),
            threshold=th,prior_weight=weight),indent=2)+'\n')

if __name__=='__main__':run()
