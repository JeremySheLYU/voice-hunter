"""Independently check all denominators, IDs, and model-blind test metadata."""
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
read=lambda p: json.loads(p.read_text())


def validate_result(path, plan=None):
    d=read(path)
    rows=d['rows']
    n=len(rows)//2
    assert len(rows)==2*n
    assert len({(r['source_id'],r['requested']) for r in rows})==2*n
    assert sum(r['present'] for r in rows)==n
    if plan:
        by_source={r['source_id']:r for r in plan['requests']}
        assert len(by_source)==n
        for r in rows:
            key=by_source[r['source_id']]
            assert r['requested']==key['target' if r['present'] else 'absent']
            assert r['target']!=r['distractor'] and r['absent'] not in (r['target'],r['distractor'])
    tp=sum(r['returned'] and r['present'] for r in rows)
    tn=sum(not r['returned'] and not r['present'] for r in rows)
    fp,fn=n-tn,n-tp
    useful=sum(r['returned'] and r['present'] and r.get('delta',float('-inf'))>=3 for r in rows)
    assert (tp,fp,tn,fn,useful)==tuple(d['summary'][k] for k in ('TP','FP','TN','FN','useful'))
    assert d['summary']['accuracy']==(tp+tn)/(2*n)
    assert d['summary']['f1']==2*tp/(2*tp+fp+fn)
    return n


def run():
    fresh=HERE/"fresh-plan.json"
    plan=read(fresh) if fresh.exists() else None
    if plan:
        assert len({r['source_id'] for r in plan['requests']})==len(plan['requests'])
        assert len({r['pair'] for r in plan['requests']})*2==len(plan['requests'])
        assert all(v==2 for v in Counter(r['pair'] for r in plan['requests']).values())
        assert len(set(Counter(r['target'] for r in plan['requests']).values()))==1
        prepared=ROOT/"round3/data/train.json"
        if prepared.exists():
            train=read(prepared)+read(ROOT/"round2/semantic/cohort.json")
            for field in ('author','sha256','id'):
                seen={str(r[field]).strip().casefold() for r in train}
                assert all(str(r[field]).strip().casefold() not in seen for r in plan['sources'])
    files=[p for p in HERE.glob('*/*/*.json') if not p.parent.name.startswith('archived') and p.stem in ('persistent','intermittent')]
    for p in files:
        n=validate_result(p,plan if p.parents[1].name=='fresh' else None)
        if p.parent.name=='v02' and p.parents[1].name=='historical':
            archived=read(ROOT/'round2/semantic/test_results.json')[p.stem]['rows']
            live=read(p)['rows']
            for a,b in zip(archived,live,strict=True):
                assert (a['source_id'],a['requested'],a['returned'])==(b['source_id'],b['requested'],b['returned'])
                if a['returned'] and a['present']:
                    assert abs(a['delta']-b['delta'])<1e-4
        print(p.relative_to(HERE),n,'paired positive/absent requests; passed')


if __name__=='__main__': run()
