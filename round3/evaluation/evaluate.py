"""Reproducible, held-out, requested-class presence and extraction evaluation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / "round2/semantic"
sys.path[:0] = [str(ROOT), str(LEGACY)]
import evaluate as legacy
import semantic_candidate as old_model
from voicehunter.separation import auto_available, separate as apple_separate

CLASSES = tuple(legacy.C)
CONDITIONS = ("persistent", "intermittent")
SEED = "VH3-evaluation-20261121"
SR = 48000
ARTIFACT = {"classifier": "round3.classifier.model", "mixture": "round3.mixture.model",
            "apple_features": "round3.apple_features.model", "fusion": "round3.fusion.model", "transfer": "round3.transfer.candidate"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(s):
    return hashlib.sha256(f"{SEED} {s}".encode()).digest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def idkey(row):
    return str(row["id"])


def audit(test, others):
    for key, getter in (("id", idkey), ("author", lambda x: x["author"].casefold().strip()),
                        ("sha256", lambda x: x["sha256"].lower())):
        testset = {getter(r) for r in test}
        overlap = testset.intersection(getter(r) for r in others)
        if overlap:
            raise ValueError(f"Fresh sources overlap a prior/training set by {key}: {len(overlap)}")
        if key != "author" and len(testset) != len(test):
            raise ValueError(f"Duplicate fresh recordings by {key}")
    for row in test:
        if not row["author"].strip() or digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Missing author or mismatched SHA256 for {idkey(row)}")


def make_plan(args):
    test = read(args.test)
    sources = {idkey(r): r for r in test}
    against = [LEGACY / "cohort.json", *args.train]
    audit(test, [r for path in against for r in read(path)])
    if args.planned_train:
        planned = [r for r in read(args.planned_train) if r["split"]=="train"]
        for fn in (idkey, lambda r:r["author"].strip().casefold()):
            if {fn(r) for r in test}.intersection(fn(r) for r in planned):
                raise ValueError("Fresh test overlaps frozen train candidate pool")
    pools = {}
    for name in CLASSES:
        eligible = [r for r in test if r["category"] == name and float(r["duration"]) >= 4.5]
        pools[name] = sorted(eligible, key=lambda r: stable("clip " + idkey(r)))
    quota = min(20, min(map(len, pools.values()))) // 4
    if not quota:
        raise ValueError("Fewer than four qualified recordings in one class")
    groups = list(itertools.combinations(CLASSES, 2))
    unused = {c: list(reversed(pools[c])) for c in CLASSES}
    pairs = [(unused[a].pop(), unused[b].pop()) for _ in range(quota) for a,b in groups]
    pairs.sort(key=lambda pair: stable("pair " + ",".join(idkey(r) for r in pair)))
    requests = []
    negatives = Counter()
    for j, pair in enumerate(pairs):
        order = list(pair if j % 2 == 0 else reversed(pair))
        for target, other in (order, list(reversed(order))):
            t, b = target["category"], other["category"]
            index = len(requests)
            absent = min((c for c in CLASSES if c not in (t,b)),
                         key=lambda c:(negatives[c], stable(f"absent {index} {c}")))
            negatives[absent] += 1
            requests.append(dict(pair=j, index=index, target=t, distractor=b, absent=absent,
                                 source_id=idkey(target), background_id=idkey(other)))
    # Collapse pair clusters with any shared contributor.
    components = {n: n for n in range(len(pairs))}
    def head(n):
        while n != components[n]:
            n = components[n]
        return n
    owners = {}
    for i,pair in enumerate(pairs):
        for r in pair:
            owner = r['author'].strip().casefold()
            if owner in owners:
                components[head(i)] = head(owners[owner])
            owners[owner] = i
    for r in requests:
        r["cluster"] = head(r["pair"])
    used = {r["source_id"] for r in requests}
    info = dict(protocol_sha256=digest(HERE/"PROTOCOL.md"), test_sha256=digest(args.test),
                source_manifests={str(p.relative_to(ROOT)) if p.is_absolute() else str(p):digest(p) for p in against},
                planned_train_sha256=digest(args.planned_train) if args.planned_train else None,
                sources=[sources[i] for i in sorted(used)], excluded_ids=sorted(set(sources)-used),
                pairs=len(pairs), uploader_clusters=len({head(n) for n in components}),
                absent_counts=dict(negatives), requests=requests)
    out=HERE / "fresh-plan.json"
    if out.exists() and read(out) != info:
        raise ValueError("Fresh plan already exists with different inputs")
    write(out,info)


def fresh_scenes(plan, condition):
    if digest(HERE / "PROTOCOL.md") != plan["protocol_sha256"]:
        raise ValueError("Protocol changed since freeze")
    source = {idkey(r):r for r in plan["sources"]}
    for r in source.values():
        if digest(ROOT / r["path"]) != r["sha256"]:
            raise ValueError(f"Source changed since freeze: {idkey(r)}")
    n=legacy.N
    for row in plan["requests"]:
        mono=[]
        for key in ("source_id","background_id"):
            signal = legacy.sound(source[row[key]]["path"])
            if len(signal) < n:
                raise ValueError("Decoded input too short: " + row[key])
            start = (len(signal)-n)//2
            y = signal[start:start+n].copy()
            y *= .08/(np.sqrt(np.mean(y.astype('f8')**2))+.000001)
            mono.append(y)
        if condition=="intermittent":
            mono[0][int(1.1*SR):int(2.15*SR)]=0
            mono[1][:int(1.1*SR)]=0
        i=row["index"]
        angles=[[12,62],[24,70],[17,81],[8,44],[38,76]][i%5]
        if i%2:
            angles=angles[::-1]
        stems=[x[:,None]*np.array([np.cos(np.radians(a)),np.sin(np.radians(a))],dtype='f4')
               for x,a in zip(mono,angles)]
        scale=min(1.,.8/(np.max(np.abs(stems[0]+stems[1]))+1e-9))
        target,background=[x*scale for x in stems]
        yield target+background,target,dict(**row,kind=condition)


def predictor(name):
    if name=="apple":
        if not auto_available():
            raise RuntimeError("Apple SoundAnalysis unavailable on this host")
        def apple(x,labels):
            out={}
            for label in labels:
                try:
                    out[label]=apple_separate(x,SR,method="auto",target=label)[0]
                except ValueError:
                    out[label]=None
            return out
        return apple
    if name=="v02":
        weights=old_model.load_model()
        def old(x,labels):
            try:
                sources,pan,probs,_,_=old_model.analyse(x,SR,weights)
            except ValueError:
                return dict.fromkeys(labels,None)
            out={}
            for label in labels:
                col=CLASSES.index(label)
                idx=int(np.argmax(probs[:,col]))
                out[label]=(sources[:,idx,None]*pan[:,idx] if probs[idx,col]>=weights["threshold"] else None)
            return out
        return old
    module=importlib.import_module(ARTIFACT.get(name,name))
    if name in ("apple_features","fusion"):
        return lambda x,labels:module.predict_scene(x,SR,CLASSES)
    if name=="mixture":
        def mixture(x,labels):
            outputs=module.predict_scene(x,SR)
            return {label: outputs.get(label) for label in labels}
        return mixture
    if hasattr(module,"predict_scene"):
        model=module.load_model() if hasattr(module,"load_model") else None
        return lambda x,labels:module.predict_scene(x,SR,model)
    if hasattr(module,"Predictor"):
        model=module.Predictor()
        def prepare(x,labels):
            prepared=model.prepare(x,SR)
            return {label:prepared.predict(label) for label in labels}
        return prepare
    def single(x,labels):
        out={}
        for label in labels:
            try:
                out[label]=module.separate(x,SR,label)
            except ValueError:
                out[label]=None
        return out
    return single


def score_scene(predict, item):
    mix,truth,meta=item
    labels=(meta["target"],meta["absent"])
    # The scorer tracks positive/absent order privately. Every joint candidate gets a fixed,
    # unconditional label list, and query-specific baselines receive alphabetic requests.
    output=predict(mix,tuple(sorted(labels)))
    interval=slice(int((2.45 if meta['kind']=="intermittent" else .15)*SR),int(4.35*SR))
    result=[]
    defined=bool(np.sum((truth[interval].astype('f8')-truth[interval].mean(axis=0))**2)>1e-12)
    base=legacy.sisdr(mix[interval],truth[interval]) if defined else None
    for role,label in enumerate(labels):
        stem=output[label]
        row=dict(meta,present=role==0,requested=label,returned=stem is not None)
        if role==0:
            row['scorable']=defined
        if stem is not None:
            if not isinstance(stem,np.ndarray) or stem.shape!=mix.shape or not np.isfinite(stem).all():
                raise ValueError(f"Invalid waveform for {meta}, requested {label}")
            if role==0 and defined:
                row['delta']=legacy.sisdr(stem[interval],truth[interval])-base
                row['useful']=row['delta']>=3.
        result.append(row)
    return result


def measures(rows):
    pos=[r for r in rows if r['present']]
    neg=[r for r in rows if not r['present']]
    tp=sum(r['returned'] for r in pos)
    fp=sum(r['returned'] for r in neg)
    tn,fn=len(neg)-fp,len(pos)-tp
    good=sum(r['returned'] and r.get('useful',False) for r in pos)
    harmful=sum(r['returned'] and r.get('delta',0)<0 for r in pos)
    return dict(TP=tp,FP=fp,TN=tn,FN=fn,
                accuracy=(tp+tn)/len(rows),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/len(pos) if pos else None,
                useful=good,useful_fraction=good/len(pos) if pos else None,harmful=harmful,
                undefined_truth=sum(not r.get('scorable',True) for r in pos))


def summary(rows):
    overall=measures(rows)
    cats={label:measures([r for r in rows if r['requested']==label]) for label in CLASSES}
    overall["macro_f1"]=float(np.mean([c['f1'] for c in cats.values() if c['f1'] is not None]))
    out={"summary":overall,"requested_class":cats}
    if all('cluster' in r for r in rows):
        clusters=defaultdict(list)
        for r in rows:
            clusters[r['cluster']].append(r)
        counts=[measures(v) for v in clusters.values()]
        agg=np.asarray([[r[k] for k in ('TP','FP','TN','FN','useful')] for r in counts])
        idx=np.random.default_rng(131121).integers(0,len(counts),(2000,len(counts)))
        tp,fp,tn,fn,use=agg[idx].sum(axis=1).T
        ci=lambda a:np.quantile(a,[.025,.975]).tolist()
        out["cluster_bootstrap_95"]={"accuracy":ci((tp+tn)/(tp+fp+tn+fn)),
                                    "f1":ci(2*tp/(2*tp+fp+fn)),
                                    "useful_fraction":ci(use/(tp+fn)),"clusters":len(counts)}
    return out


def evaluate(args):
    out=HERE/args.dataset/args.method
    out.mkdir(parents=True,exist_ok=True)
    lock=HERE/"selection.json"
    if args.dataset=="fresh" and args.method not in ("apple","v02") and not lock.exists():
        raise ValueError("Primary selection must be frozen before any fresh candidate test")
    predict=predictor(args.method)
    plan=read(HERE/"fresh-plan.json") if args.dataset=="fresh" else None
    cohort=read(LEGACY/"cohort.json") if args.dataset=="historical" else None
    origin={"protocol":digest(HERE/"PROTOCOL.md"),"runner":digest(Path(__file__)),
            "scene_plan":digest(HERE/"fresh-plan.json") if plan is not None else digest(LEGACY/"cohort.json")}
    if lock.exists() and args.method not in ("apple","v02"):
        origin["selection_sha256"]=digest(lock)
    if args.fingerprint:
        origin["model_lock"]={str(p):digest(p) for p in args.fingerprint}
    for cond in CONDITIONS:
        target=out/(cond+".json")
        if target.exists():
            recorded=read(target)
            if recorded.get('origin')!=origin:
                raise ValueError(f"Result with different provenance exists at {target}")
            continue
        scenes=fresh_scenes(plan,cond) if plan is not None else legacy.sources(cohort,"test",cond)
        with ThreadPoolExecutor(args.workers) if args.workers>1 else nullcontext() as pool:
            task=lambda s:score_scene(predict,s)
            rows=[]
            while chunk := list(itertools.islice(scenes, max(1,2*args.workers))):
                batches=pool.map(task,chunk) if pool else map(task,chunk)
                rows.extend(r for batch in batches for r in batch)
        info=dict(dataset=args.dataset,method=args.method,condition=cond,origin=origin,**summary(rows),rows=rows)
        write(target,info)
        print(args.dataset,args.method,cond,info['summary'],flush=True)


def historical_reference():
    out=HERE/"historical"
    for name,file in (("archived_v02","test_results.json"),("archived_apple","apple_results.json")):
        archive=read(LEGACY/file)
        here=out/name
        here.mkdir(parents=True,exist_ok=True)
        for cond in CONDITIONS:
            group=archive[cond]
            s=summary(group['rows']);m=s['summary']
            expected=group['summary']
            assert [m[k] for k in ('TP','FP','useful','harmful')]==[expected[k] for k in ('returned','false_accept','useful','harmful')]
            write(here/(cond+'.json'),dict(source=str(LEGACY.relative_to(ROOT)/file),
                                          archive_sha256=digest(LEGACY/file),**s))
            print(name,cond,m,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='cmd',required=True)
    plan=subs.add_parser('plan');plan.add_argument('--test',type=Path,required=True)
    plan.add_argument('--train',type=Path,nargs='*',default=[])
    plan.add_argument('--planned-train',type=Path)
    run=subs.add_parser('run');run.add_argument('dataset',choices=['historical','fresh']);run.add_argument('method')
    run.add_argument('--workers',type=int,default=1); run.add_argument('--fingerprint',type=Path,nargs='*')
    subs.add_parser('archives')
    a=p.parse_args()
    if a.cmd=='plan': make_plan(a)
    elif a.cmd=='run': evaluate(a)
    else: historical_reference()
