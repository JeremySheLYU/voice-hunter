"""Group-disjoint out-of-fold stacking before independent confirmation."""
from pathlib import Path
import hashlib,tempfile,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3b.model import model as m
from round3b.model.encode_train import selected,corpus,CACHE
from round3.apple_features import train as aptrain
from round2.semantic.evaluate import sources
from round3b.model import dev_features

C=m.CLASSES
YCACHE=Path('/tmp/voicehunter-round3-transfer-features')

def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fold(row):return int(hashlib.sha256(('vh-r3b-fold|'+row['author'].strip().casefold()).encode()).hexdigest(),16)%5


def base(allrows,sel,ast_features,apfeatures,yamfeatures,train_s,train_a):
    yi=np.array([C.index(r['category']) for r in allrows])[train_a]
    targets=np.eye(5)[yi]
    reg=json.loads((m.apple.HERE/'freeze.json').read_text())['parameters']['reg']
    a=apfeatures[train_a].astype('f8')
    coef=np.linalg.solve((a@a.T)**2+reg*np.eye(len(a)),targets)
    y=yamfeatures[train_a].astype('f8')
    ycoef=np.linalg.solve(np.clip(y@y.T,0,1)**3+.16*np.eye(len(y)),targets)
    label=np.array([C.index(r['category']) for r in sel])[train_s]
    ast=m.ast.fit(ast_features[train_s],label)
    return dict(ast_center=ast['center'],ast_basis=ast['basis'],ast_coef=ast['coef'],
                app_basis=a.astype('f4'),app_coef=coef.astype('f4'),yam_basis=y.astype('f4'),
                yam_coef=ycoef.astype('f4'),label_order=np.array(m.apple.label_order()))


def meta_fit(ev,targets):
    cent=ev-ev.mean(axis=2,keepdims=True)
    scale=cent.std(axis=(0,2)).clip(.035)
    feat=(cent/scale[None,:,None]).clip(-6,6)
    shared=np.zeros(6);ins=np.zeros((6,5));bias=np.zeros(5)
    for i in range(800):
        logits=(feat*(shared[:,None]+ins)).sum(axis=1)+bias
        error=(m.softmax(logits)-targets)/len(targets)
        dg=(error[:,None,:]*feat).sum(axis=0)
        step=.16/(1+i/400)
        shared-=step*(dg.sum(axis=1)+.045*shared)
        ins-=step*(dg+.095*ins)
        bias-=step*(error.sum(axis=0)+.06*bias)
    return dict(scale=scale,shared=shared,individual=ins,intercept=bias)


def pair_thresholds(prob,sel,y):
    rng=np.random.default_rng(4621)
    scores=[];true=[]
    for i,r in enumerate(sel):
        eligible=[j for j,s in enumerate(sel) if y[i]!=y[j] and r['author'].strip().casefold()!=s['author'].strip().casefold()]
        for j in rng.choice(eligible,7,replace=False):
            scores.append(np.maximum(prob[i],prob[j]));true.append(np.eye(5,dtype=bool)[y[i]]|np.eye(5,dtype=bool)[y[j]])
    z=np.stack(scores);truth=np.stack(true);grid=np.arange(.1,.701,.02)
    thresholds=[]
    for k in range(5):
        def quality(h):
            tpr=(z[truth[:,k],k]>=h).mean();tnr=(z[~truth[:,k],k]<h).mean()
            return (tpr+tnr)/2-.14*abs(h-.32)
        thresholds.append(max(grid,key=quality))
    return np.array(thresholds)


def score_dev(z,t):
    idx=np.arange(len(z['good']))
    prob=z['score']
    best=prob[idx,:,z['label']].argmax(axis=1)
    hi=prob[idx,best,z['label']];lo=prob[idx,:,z['neg']].max(axis=1)
    pos=(hi>=t[z['label']])&z['good'];fp=((lo>=t[z['neg']])&z['good']).sum()
    tp=int(pos.sum());fp=int(fp);n=len(pos);d=z['gain'][idx,best]
    return dict(n=n,tp=tp,fp=fp,accuracy=(tp+n-fp)/(2*n),f1=2*tp/(n+tp+fp),useful=int((pos&(d>=3)).sum()),harmful=int((pos&(d<0)).sum()))


def run():
    allrows=corpus();sel=selected();y=np.array([C.index(r['category']) for r in sel])
    sha_index={r['sha256']:i for i,r in enumerate(allrows)}
    loc=np.array([sha_index[r['sha256']] for r in sel])
    af=np.stack([np.load(CACHE/(r['sha256']+'-ast.npy')) for r in sel])
    ap=np.stack([aptrain.get_record(r)[0] for r in allrows]);apn=m.apple.normalized_features(ap)
    def yvec(r):
        p=YCACHE/f"train-{r['sha256']}.npz"
        if p.exists():
            with np.load(p) as f:return f['vectors'].mean(axis=0)
        from round3.transfer.train import representation
        YCACHE.mkdir(exist_ok=True)
        return representation(r,m.fusion.encoder(),YCACHE).mean(axis=0)
    yf=np.stack([yvec(r) for r in allrows]); yf/=np.linalg.norm(yf,axis=1,keepdims=True).clip(1e-7)
    data=[dict(np.load(CACHE/(r['sha256']+'-prior.npz'))) for r in sel]
    yp=np.stack([d['p'] for d in data]);ym=np.stack([d['y'] for d in data])
    allfold=np.array([fold(r) for r in allrows]);selfold=allfold[loc]
    evidence=np.zeros((len(sel),6,5))
    for k in range(5):
        test=selfold==k
        head=base(allrows,sel,af,apn,yf,~test,allfold!=k)
        evidence[test]=m.stem_evidence(af[test],ap[loc[test]],ym[test],yp[test],head)
    meta=meta_fit(evidence,np.eye(5)[y])
    # A second out-of-fold layer keeps per-class score thresholds separate from the
    # same source's label when estimating the calibration operating point.
    fold_models=[meta_fit(evidence[selfold!=k],np.eye(5)[y[selfold!=k]]) for k in range(5)]
    full=base(allrows,sel,af,apn,yf,np.ones(len(sel),bool),np.ones(len(allrows),bool));full.update(meta)
    vrows=[r for r in json.loads((ROOT/'round2/semantic/cohort.json').read_text()) if r['split']=='validation']
    dev={}
    for kind in ('persistent','intermittent'):
        scene=[]
        for _,_,rec in sources(vrows,'validation',kind):
            with np.load(CACHE/(kind+rec['source_id']+'.npz')) as f:d=dict(f)
            e=m.stem_evidence(d['ast'],d['apple'],d['yam'],d['prior'],full)
            q=d['quality'];scene.append((e,C.index(rec['target']),C.index(rec['absent']),d['gain'],q[0]<=.16 and q[1]>=.05))
        dev[kind]=dict(zip(('ev','label','neg','gain','good'),map(np.stack,zip(*scene))))
    best=None
    for smoothing in (0,.25,.5):
        cross=np.zeros((len(sel),5))
        for k,w in enumerate(fold_models):
            cross[selfold==k]=m.posterior(evidence[selfold==k],w,smoothing)
        per=pair_thresholds(cross,sel,y)
        for d in dev.values():d['score']=np.stack([m.posterior(e,full,smoothing) for e in d['ev']])
        for adaptation in (0,.5,1.):
            for cut in np.arange(.12,.551,.02):
                th=np.clip(cut+adaptation*(per-np.mean(per)),.07,.72)
                perf={kind:score_dev(z,th) for kind,z in dev.items()}
                key=(min(min(v['accuracy'],v['f1']) for v in perf.values()),min(v['useful'] for v in perf.values()),-adaptation,-smoothing)
                if best is None or key>best[0]:best=(key,dict(thresholds=th.tolist(),smoothing=smoothing,training_thresholds=per.tolist(),validation=perf))
    config=best[1]
    np.savez_compressed(HERE/'weights.npz',**full,thresholds=config['thresholds'],smoothing=config['smoothing'])
    touched=[HERE/k for k in ('weights.npz','model.py','train.py','encode_train.py','encode_prior.py','dev_features.py','PROTOCOL.md','training_sources.json')]
    touched+=[ROOT/k for k in ('round3/transformer/ast-int8.onnx','round3/transformer/model.py','round3/transfer/yamnet.onnx','round3/transfer/candidate.py','round3/fusion/model.py','round3/apple_features/model.py','round3/mixture/jointdiag.py','round3/data/train.json')]
    lock=dict(name='r3b_ensemble',inference='round3b.model.model.predict_scene(audio, sr=48000)',config=config,
        training_count=len(allrows),ast_count=len(sel),oof_folds=5,fold_counts=np.bincount(selfold).tolist(),
        hashes={str(p.relative_to(ROOT)):hashfile(p) for p in touched})
    (HERE/'freeze.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(config),flush=True)

if __name__=='__main__':run()
