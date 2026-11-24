"""Summarise frozen per-request outputs (never used for model selection)."""
import csv
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    table=[]
    by_class=[]
    methods={}
    for d in sorted(HERE.glob('*/*/*.json')):
        if d.stem not in ('intermittent','persistent') or d.parents[1].name not in ('fresh','historical'):
            continue
        result=json.loads(d.read_text())
        domain,method,condition=d.parents[1].name,d.parent.name,d.stem
        if domain=='historical' and method=='archived_v02':
            continue
        m=result['summary']
        rows=result.get('rows',[])
        success=[r['delta'] for r in rows if r['returned'] and r['present'] and r.get('delta') is not None]
        if not rows and method=='archived_apple':
            rows=json.loads((HERE.parents[1]/'round2/semantic/apple_results.json').read_text())[condition]['rows']
            success=[r['delta'] for r in rows if r['returned'] and r['present'] and r.get('delta') is not None]
        gates={k:bool(m[k]>=.90) for k in ('accuracy','f1','useful_fraction')}
        out=dict(dataset=domain,method=method,condition=condition,**m,
                 median_delta_returned=float(np.median(success)) if success else '',
                 **{k+'_at_least_90':v for k,v in gates.items()})
        cis=result.get('cluster_bootstrap_95')
        for k in ('accuracy','f1','useful_fraction'):
            ci=cis[k] if cis else ['','']
            out[k+'_95_lo'],out[k+'_95_hi']=ci
        table.append(out)
        by_class.extend(dict(dataset=domain,method=method,condition=condition,requested_class=c,**row)
                        for c,row in result['requested_class'].items())
        if domain=='fresh':
            methods.setdefault(method,{})[condition]=gates
    for name,rows in (('scorecard.csv',table),('scorecard_by_requested_class.csv',by_class)):
        if rows:
            with (HERE/name).open('w', newline='') as f:
                cw=csv.DictWriter(f,fieldnames=list(rows[0]));cw.writeheader();cw.writerows(rows)
    if methods:
        out={}
        for method,conditions in methods.items():
            if set(conditions)!= {'persistent','intermittent'}:
                continue
            out[method]={'detection_gate_met':all(row['accuracy'] and row['f1'] for row in conditions.values()),
                         'useful_90_met':all(row['useful_fraction'] for row in conditions.values()),
                         'by_condition':conditions}
        (HERE/'fresh_target_90.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__': main()
