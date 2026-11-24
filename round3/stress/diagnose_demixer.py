"""Oracle diagnostic only. The clean references here are never available in inference."""
import json
from pathlib import Path
import numpy as np
from evaluate_stress import HERE, SR, WINDOW, digest, legacy_model, legacy_test, scene_iter


def main():
    manifest=json.loads((HERE/'scenes.json').read_text())
    report={}
    for kind in ('room','native'):
        rows=[]
        for audio,truth,meta in scene_iter(kind,manifest):
            row=dict(meta)
            try:
                sources,mix,_,_=legacy_model.candidates(audio)
            except ValueError:
                row.update(demixer_accepted=False,oracle_best_db=None,useful=False)
            else:
                deltas=[legacy_test.sisdr((sources[:,i,None]*mix[:,i])[WINDOW],truth[WINDOW])-
                        legacy_test.sisdr(audio[WINDOW],truth[WINDOW]) for i in (0,1)]
                best=float(max(deltas))
                row.update(demixer_accepted=True,oracle_best_db=best,useful=best>=3)
            rows.append(row)
        report[kind]=dict(intended=len(rows),directions_detected=sum(r['demixer_accepted'] for r in rows),
                          ideal_label_useful=sum(r['useful'] for r in rows), rows=rows)
        print(kind,{k:v for k,v in report[kind].items() if k!='rows'},flush=True)
    (HERE/'oracle_v02_demix.json').write_text(json.dumps(dict(scene_sha256=digest(HERE/'scenes.json'),
                  oracle_definition="best reference-based choice of two old demixed images after existing direction gate; includes all intended positives",conditions=report),indent=2)+'\n')


if __name__=='__main__':
    main()
