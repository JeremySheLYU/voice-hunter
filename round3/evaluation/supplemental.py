"""Evaluate all above-quota holdout sources descriptively, separate from the balanced primary endpoint."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import itertools
from pathlib import Path

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("primary_eval",BASE/"evaluate.py")
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)


def plan():
    primary=ev.read(BASE/'fresh-plan.json')
    data=ev.read(BASE.parents[1]/'round3/data/test.json')
    groups=defaultdict(list)
    extra=set(primary['excluded_ids'])
    if not extra: return None
    used={}
    for r in data:
        if r['id'] in extra:
            groups[r['category']].append(r)
    for v in groups.values():
        v.sort(key=lambda r:ev.stable('extra '+r['id']))
    if len(extra)%2:
        donor=min(primary['sources'],key=lambda r:(len(groups[r['category']]),ev.stable('donor '+r['id'])))
        groups[donor['category']].append(donor)
    planrows=[];pairs=[]
    while any(groups.values()):
        names=sorted([c for c in groups if groups[c]],key=lambda c:(-len(groups[c]),ev.stable(c)))
        if len(names)<2:
            raise RuntimeError("Could not form disjoint cross-class secondary pairs")
        pair=[groups[c].pop() for c in names[:2]]
        pairs.append(pair)
    neg=Counter()
    edges=[{r['author'].strip().casefold() for r in p} for p in pairs]
    cluster={i:i for i in range(len(pairs))}
    for i in range(len(pairs)):
        for j in range(i):
            if edges[i]&edges[j]:
                old=cluster[i]
                for k in cluster:
                    if cluster[k]==old:cluster[k]=cluster[j]
    for j,p in enumerate(pairs):
        for target,bg in (p,list(reversed(p))):
            used[target['id']]=target
            if target['id'] not in extra:continue
            t,other=target['category'],bg['category']
            idx=len(planrows)
            absent=min((c for c in ev.CLASSES if c not in (t,other)),
                       key=lambda c:(neg[c],ev.stable(f'extra-negative {idx} {c}')))
            neg[absent]+=1
            planrows.append(dict(pair=j,index=idx,target=t,distractor=other,absent=absent,
                                 source_id=target['id'],background_id=bg['id'],cluster=cluster[j]))
    assert {r['source_id'] for r in planrows}==extra
    out=dict(protocol_sha256=ev.digest(BASE/'PROTOCOL.md'),primary_plan_sha256=ev.digest(BASE/'fresh-plan.json'),
             exploratory=True,sources=list(used.values()),requests=planrows,
             donor_ids=sorted(set(used)-extra),runtime_note='Only excess source IDs are scored as targets. Shared positive and negative requests by pair.')
    path=BASE/'supplemental-plan.json'
    if path.exists():
        if ev.read(path)!=out:raise ValueError("Supplemental plan changed after freeze")
    else:
        ev.write(path,out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('method',nargs='?',default=None)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--fingerprint',type=Path,nargs='*')
    a=p.parse_args()
    frozen=plan()
    if frozen is None or a.method is None:return
    if a.method not in ('apple','v02') and not (BASE/'selection.json').exists():
        raise ValueError("Primary model selection still pending")
    predict=ev.predictor(a.method)
    directory=BASE/"supplemental"/a.method
    directory.mkdir(parents=True,exist_ok=True)
    provenance={'plan':ev.digest(BASE/'supplemental-plan.json'),'judge':ev.digest(BASE/'evaluate.py'),
                'wrapper':ev.digest(Path(__file__)),
                'model_lock':{str(f):ev.digest(f) for f in (a.fingerprint or [])}}
    if a.method not in ('apple','v02'):provenance['selection']=ev.digest(BASE/'selection.json')
    for condition in ev.CONDITIONS:
        dest=directory/(condition+'.json')
        if dest.exists():
            if ev.read(dest)['origin']!=provenance:raise ValueError("Previous supplemental score mismatch")
            continue
        rows=[]
        scenes=ev.fresh_scenes(frozen,condition)
        with ThreadPoolExecutor(a.workers) as pool:
            while chunk:=list(itertools.islice(scenes,max(1,2*a.workers))):
                for batch in pool.map(lambda item:ev.score_scene(predict,item),chunk):rows+=batch
        # A supplemental class without positives and negatives has no estimable score.
        summary=ev.summary(rows)
        ev.write(dest,dict(dataset='supplemental',method=a.method,condition=condition,
                           origin=provenance,**summary,rows=rows))
        print(a.method,condition,summary['summary'],flush=True)


if __name__=='__main__':main()
