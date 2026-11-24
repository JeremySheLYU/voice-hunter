"""Hash actual waveforms from both controlled strata to check equality across inference environments."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evaluation',HERE/'evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
plan=ev.read(HERE/'fresh-plan.json')
out={}
for cond in ev.CONDITIONS:
    h=hashlib.sha256()
    for mixture,target,meta in ev.fresh_scenes(plan,cond):
        h.update(mixture.astype('<f4',copy=False).tobytes())
        h.update(target.astype('<f4',copy=False).tobytes())
    out[cond]=h.hexdigest()
print(json.dumps(out))
