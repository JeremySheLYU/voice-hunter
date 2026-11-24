"""Upper bound for choosing one of the two fixed-pan source images on fresh native stereo."""
import json

import numpy as np
from evaluate_stress import HERE, ROOT, WINDOW, digest, legacy_model, legacy_test, native_clip
from evaluate_fresh_native import PLAN

plan=json.loads(PLAN.read_text())
lookup={r['id']:r for r in plan['sources']}
counts={'positive':len(plan['scenes']),'directions_detected':0,'ideal_label_useful':0}
rows=[]
for meta in plan['scenes']:
    target,bg=[native_clip(str(ROOT/lookup[meta[k]]['path'])) for k in ('source_id','background_id')]
    audio=target+bg;gain=min(1.,.8/(float(abs(audio).max())+1e-9));audio=audio*gain;truth=target*gain
    try:
        s,m,_,_=legacy_model.candidates(audio)
    except ValueError:
        rows.append({**meta,'best_db':None})
        continue
    counts['directions_detected']+=1
    base=legacy_test.sisdr(audio[WINDOW],truth[WINDOW])
    best=max(legacy_test.sisdr((s[:,i,None]*m[:,i])[WINDOW],truth[WINDOW])-base for i in (0,1))
    counts['ideal_label_useful']+=int(best>=3)
    rows.append({**meta,'best_db':float(best)})
(HERE/'fresh_oracle_v02_demix.json').write_text(json.dumps(dict(plan_sha256=digest(PLAN),summary=counts,rows=rows),indent=2)+'\n')
print(counts)
