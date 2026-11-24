"""Confirm width comparisons on actual rendered PCM, including numerical waveform invariants."""
import json
import numpy as np
from fit_width import HERE, DATA, decode, input_stereo, ratios
from render_width import render
path=HERE/'width_results.json'
r=json.loads(path.read_text());rows={}
for name,d in r['measurements'].items():
    if d['date'] not in r['test_dates']:continue
    foa,rate=decode(DATA/'ihta_foa'/('FirstOrderAmbisonics_'+name),4)
    xy=input_stereo(foa);raw=ratios(xy,rate);y=np.array(d['teacher_db'])
    rows[name]={}
    for key,config in r['conditions'].items():
        out=render(xy,rate,config['gain_db'])
        rows[name][key]=dict(absolute_error_db=abs(ratios(out,rate)-y).tolist(),
             max_mono_error=float(abs(xy.mean(axis=1)-out.mean(axis=1)).max()),
             neutral_null=float(abs(xy-out).max()) if key=='unchanged' else None)
scores={}
for key in r['conditions']:
    a=np.array([x[key]['absolute_error_db'] for x in rows.values()])
    scores[key]=float(np.median(a))
out=dict(heldout_captures=len(rows),actual_render_median_band_error_db=scores,
         max_mono_error=max(v['max_mono_error'] for row in rows.values() for v in row.values()),
         max_neutral_null=max(row['unchanged']['neutral_null'] for row in rows.values()),
         cases=rows)
(HERE/'render_eval.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['heldout_captures'],scores)
