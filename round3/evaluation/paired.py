"""Paired source/uploader-cluster bootstrap change versus frozen reference methods."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def clustered(rows):
    out=defaultdict(lambda:np.zeros(5,dtype='f8'))
    for r in rows:
        pos=r['present']; returned=r['returned']
        cell=(0 if returned else 3) if pos else (1 if returned else 2)
        out[r['cluster']][cell]+=1
        if pos and returned and r.get('useful'):
            out[r['cluster']][4]+=1
    return out


def scores(total):
    tp,fp,tn,fn,good=total.T
    return np.column_stack([(tp+tn)/(tp+fp+tn+fn),2*tp/(2*tp+fp+fn),good/(tp+fn)])


def key(r):
    return (r['cluster'],r['pair'],r['source_id'],r['background_id'],r['target'],r['requested'],r['present'])


def compare(candidate,baseline):
    if sorted(map(key,candidate))!=sorted(map(key,baseline)):
        raise ValueError('Cannot pair the two sets of evaluated queries')
    ca,ba=clustered(candidate),clustered(baseline)
    clusters=sorted(ca)
    arrays=[np.stack([d[i] for i in clusters]) for d in (ca,ba)]
    idx=np.random.default_rng(131121).integers(0,len(clusters),(2000,len(clusters)))
    delta=scores(arrays[0].sum(axis=0,keepdims=True))-scores(arrays[1].sum(axis=0,keepdims=True))
    samples=scores(arrays[0][idx].sum(axis=1))-scores(arrays[1][idx].sum(axis=1))
    return {k:{'estimate':float(delta[0,j]),'percentile95':np.quantile(samples[:,j],[.025,.975]).tolist()}
            for j,k in enumerate(('accuracy','f1','useful_fraction'))}


def main():
    out={}
    for candid in sorted((HERE/'fresh').iterdir()):
        if candid.name in ('apple','v02'):continue
        for cond in ('persistent','intermittent'):
            src=candid/(cond+'.json')
            if not src.exists():continue
            rows=json.loads(src.read_text())['rows']
            for base in ('apple','v02'):
                ref=json.loads((HERE/'fresh'/base/(cond+'.json')).read_text())['rows']
                out.setdefault(candid.name,{}).setdefault(cond,{})[base]=compare(rows,ref)
    if out:
        (HERE/'fresh_paired_differences.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
