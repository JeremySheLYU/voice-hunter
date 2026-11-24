"""Second locked, uploader-independent comparison with the v0.3 primary."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
ev=load(ROOT/'round3/evaluation/evaluate.py','sibling_round3_evaluation')
paired=load(ROOT/'round3/evaluation/paired.py','sibling_round3_paired')
C=ev.CLASSES
OLD=('round2/semantic/cohort.json','round2/envdata/all_sources.json','round2/environment/native_inventory.json',
     'round3/data/train.json','round3/data/test.json','round3/data/native.json','round3/unknowns/manifest.json')
PREFIX='VH3b-frozen-external'
def h(x):
    return hashlib.sha256(f'{PREFIX} {x}'.encode()).digest()
def owner(r):
    return r['author'].strip().casefold()
def ident(r):
    return str(r['id'])
def keep_or_write(path,content):
    if path.exists():
        if ev.read(path)!=content:raise ValueError(f'Frozen {path.name} would change')
        return
    ev.write(path,content)


def plan(manifest):
    baseline=[r for fn in OLD for r in ev.read(ROOT/fn)]
    earlier_ids={ident(r) for r in baseline}
    earlier_owners={owner(r) for r in baseline}
    earlier_bytes={r.get('sha256') for r in baseline if r.get('sha256')}
    earlier_pcm=set(ev.read(ROOT/'round3/data/historic_pcm.json'))
    earlier_pcm.update(r['pcm_sha256'] for r in baseline if r.get('pcm_sha256'))
    rows=ev.read(manifest)
    ids=set();files=set();waves=set();pool={c:[] for c in C};excluded=[]
    for r in sorted(rows,key=lambda row:h('file '+ident(row))):
        ident_r=ident(r);ow=owner(r);raw=r.get('sha256');pcm=r.get('pcm_sha256')
        if (r['category'] not in C or float(r['duration'])<4.5 or not ow or not raw or
            ow in earlier_owners or ident_r in earlier_ids or raw in earlier_bytes or pcm in earlier_pcm
            or ident_r in ids or raw in files or (pcm is not None and pcm in waves)):
            excluded.append(ident_r);continue
        if ev.digest(ROOT/r['path'])!=raw:raise ValueError(f'File hash mismatch {ident_r}')
        ids.add(ident_r);files.add(raw);waves.add(pcm)
        pool[r['category']].append(r)
    groups={c:[] for c in C}
    for c in C:
        seen=set()
        for r in pool[c]:
            if owner(r) not in seen:
                seen.add(owner(r));groups[c].append(r)
    counts={c:len(v) for c,v in groups.items()}
    smallest=min(counts.values())
    quota=12 if smallest>=12 else (min(8,smallest)//4)*4
    mode='primary' if quota==12 else ('descriptive_only' if quota else 'insufficient')
    proof=dict(protocol_sha256=ev.digest(HERE/'PROTOCOL.md'),test_manifest_sha256=ev.digest(manifest),
               earlier_sha256={f:ev.digest(ROOT/f) for f in OLD},mode=mode,counts_distinct_uploaders=counts,
               distinct_target_recordings_per_class=quota,excluded_invalid_or_overlapping_ids=excluded)
    if not quota:
        keep_or_write(HERE/'plan.json',proof);return proof
    used={c:list(reversed(v[:quota])) for c,v in groups.items()}
    pairs=[(used[a].pop(),used[b].pop()) for i in range(quota//4) for a,b in itertools.combinations(C,2)]
    pairs.sort(key=lambda pair:h('pair '+' '.join(ident(r) for r in pair)))
    # Connect clusters where a listed uploader contributed across different class pairs.
    memberships=[{owner(r) for r in pair} for pair in pairs]
    clusters=list(range(len(pairs)))
    for i,m in enumerate(memberships):
        for j in range(i):
            if m & memberships[j]:
                old=clusters[i];clusters=[clusters[j] if k==old else k for k in clusters]
    neg=Counter();requests=[];selected=[]
    for i,pair in enumerate(pairs):
        ls=(pair if i%2==0 else tuple(reversed(pair)))
        selected.extend(ls)
        for t,b in (ls,tuple(reversed(ls))):
            n=len(requests)
            absent=min((c for c in C if c not in (t['category'],b['category'])),
                       key=lambda c:(neg[c],h(f'absent {n} {c}')))
            neg[absent]+=1
            requests.append(dict(index=n,pair=i,cluster=clusters[i],target=t['category'],distractor=b['category'],
                                 source_id=ident(t),background_id=ident(b),absent=absent))
    proof.update(sources=selected,requests=requests,pairs=len(pairs),uploader_clusters=len(set(clusters)),
                 not_selected_ids=sorted({ident(r) for r in rows}-{ident(r) for r in selected}),
                 absent_counts=dict(neg))
    keep_or_write(HERE/'plan.json',proof);return proof


def lock():
    model_paths=['round3/fusion/freeze.json','round3b/model/freeze.json']
    group=dict(plan_sha256=ev.digest(HERE/'plan.json'), model_files={p:ev.digest(ROOT/p) for p in model_paths},
               module='round3b.model.model',baseline='round3.fusion.model')
    keep_or_write(HERE/'model-selection.json',group)


def evaluate(method,workers):
    verified=ev.read(HERE/'plan.json')
    if verified['mode']=='insufficient':raise ValueError('Insufficient untouched material.')
    expected=ev.read(HERE/'model-selection.json')
    if expected['plan_sha256']!=ev.digest(HERE/'plan.json'):raise ValueError('Plan changed')
    for f,value in expected['model_files'].items():
        if ev.digest(ROOT/f)!=value:raise ValueError('Frozen model changed: '+f)
    p=verified.copy()
    p['protocol_sha256']=ev.digest(ROOT/'round3/evaluation/PROTOCOL.md')
    if method=='fusion':pred=ev.predictor('fusion')
    elif method=='challenger':pred=ev.predictor('round3b.model.model')
    else:raise ValueError('Unknown method')
    out=HERE/verified['mode']/method;out.mkdir(parents=True,exist_ok=True)
    origin=dict(model_lock_sha256=ev.digest(HERE/'model-selection.json'), runner=ev.digest(Path(__file__)))
    for cond in ev.CONDITIONS:
        dest=out/(cond+'.json')
        if dest.exists():
            if ev.read(dest)['origin']!=origin:raise ValueError('Score from different version')
            continue
        scenes=ev.fresh_scenes(p,cond)
        rows=[]
        with ThreadPoolExecutor(workers) as pool:
            while chunk:=list(itertools.islice(scenes,2*workers)):
                for batch in pool.map(lambda item:ev.score_scene(pred,item),chunk):rows+=batch
        result=dict(method=method,mode=verified['mode'],condition=cond,
                    origin=origin,**ev.summary(rows),rows=rows)
        ev.write(dest,result)
        print(method,cond,result['summary'],flush=True)


def summarize():
    verified=ev.read(HERE/'plan.json')
    report=dict(mode=verified['mode'],eligible=verified['counts_distinct_uploaders'],
                evaluated_per_class=verified['distinct_target_recordings_per_class'])
    if verified['mode']!='insufficient':
        out=HERE/verified['mode']
        gate=True
        for cond in ev.CONDITIONS:
            a,b=[ev.read(out/model/(cond+'.json')) for model in ('challenger','fusion')]
            ref=paired.compare(a['rows'],b['rows'])
            gate=gate and a['summary']['accuracy']>=.9 and a['summary']['f1']>=.9
            report[cond]=dict(challenger=a['summary'],fusion=b['summary'],
                              cluster_95_challenger=a['cluster_bootstrap_95'],paired_difference=ref)
        report['raw_point_estimates_all_ge_90']=bool(gate)
        report['predeclared_primary_gate_met']=bool(gate and verified['mode']=='primary')
    else:
        report['predeclared_primary_gate_met']=False
    ev.write(HERE/'summary.json',report)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    opts=cli.add_subparsers(dest='cmd',required=True)
    p=opts.add_parser('plan');p.add_argument('manifest',type=Path)
    opts.add_parser('lock')
    p=opts.add_parser('run');p.add_argument('method',choices=['fusion','challenger']);p.add_argument('--workers',type=int,default=1)
    opts.add_parser('summary');a=cli.parse_args()
    if a.cmd=='plan': plan(a.manifest)
    elif a.cmd=='lock':lock()
    elif a.cmd=='run':evaluate(a.method,a.workers)
    else:summarize()
