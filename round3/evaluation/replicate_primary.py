"""Rerun selected joint model serially and verify every field needed for the primary claim."""
from pathlib import Path
import importlib.util
import json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evaluation',HERE/'evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)


def main():
    selection=ev.read(HERE/'selection.json')
    method=selection.get('primary') or selection.get('selected')
    if not isinstance(method,str):
        raise ValueError('Selection needs its named primary model')
    predict=ev.predictor(method)
    plan=ev.read(HERE/'fresh-plan.json')
    checks={}
    for condition in ev.CONDITIONS:
        official=ev.read(HERE/'fresh'/method/(condition+'.json'))['rows']
        seen=[]
        for scene in ev.fresh_scenes(plan,condition):
            seen.extend(ev.score_scene(predict,scene))
        decisions,quality,maximum=0,0,0.
        for a,b in zip(official,seen,strict=True):
            if (a['source_id'],a['requested']) != (b['source_id'],b['requested']):
                raise ValueError('Query sequence changed')
            decisions+=int(a['returned']!=b['returned'])
            quality+=int(bool(a.get('useful'))!=bool(b.get('useful')))
            if 'delta' in a and 'delta' in b:maximum=max(maximum,abs(a['delta']-b['delta']))
        checks[condition]=dict(requests=len(seen),decision_flips=decisions,
                              useful_delivery_flips=quality,maximum_abs_sisdri_change=maximum,
                              serial_summary=ev.measures(seen))
        print(condition,decisions,quality,maximum,flush=True)
    ev.write(HERE/'primary_reproducibility.json',checks)


if __name__=='__main__': main()
