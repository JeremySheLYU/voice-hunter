"""Cross-lab test of frozen profile against measured HUTUBS. Test-only dependency: h5py."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from train_evaluate import HERE, ROOT, RATE, cue, render_cues, summarize


def run(directory):
    frozen=json.loads((HERE/"results.json").read_text())["weight_sha256"]
    assert hashlib.sha256((ROOT/"voicehunter/models/cipic_horizontal_v1.json").read_bytes()).hexdigest()==frozen
    manifest=json.loads((HERE/"hutubs_manifest.json").read_text())
    hashes={r["file"]:r["sha256"] for r in manifest["files"]}
    rendered={}
    results=defaultdict(list)
    files=sorted(directory.glob("*_measured.sofa"))
    assert len(files)==96
    total=0
    for path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[path.name]
        subject=int(path.name.split("_")[0][2:])
        role="mannequin_repeat" if subject in (1,96) else "repeat_person22" if subject==88 else "independent_humans"
        with h5py.File(path) as f:
            assert f["SourcePosition"].attrs["Type"]==b"spherical"
            assert f["SourcePosition"].attrs["Units"]==b"degree, degree, metre"
            assert f["Data.SamplingRate"][0]==RATE
            assert np.allclose(f["Data.Delay"][:],0)
            assert f["ReceiverPosition"][0,1,0]>0 and f["ReceiverPosition"][1,1,0]<0
            ir=f["Data.IR"]
            total+=ir.shape[0]
            for idx,(az,el,r) in enumerate(f["SourcePosition"][:]):
                angle=round(float(-(az+180.)%360.-180.),4)
                if not (abs(el)<=0.01 and 0.01<abs(angle)<=80.01):continue
                l,r=ir[idx]
                ild,itd=cue(r,l) if angle>0 else cue(l,r)
                for model in ("analytic","measured"):
                    key=model,angle
                    if key not in rendered: rendered[key]=render_cues(model,angle)
                    pred,delay=rendered[key]
                    results[role].append(dict(subject=subject,angle=angle,model=model,
                                              ILD_MAE_dB=float(np.abs(ild-pred).mean()),ITD_abs_error_us=abs(itd-delay)))
    summary={}
    for group,vals in results.items():
        summary[group]={"n_subject_records":len({r['subject'] for r in vals})}
        for mod in ("analytic","measured"):
            rows=[r for r in vals if r["model"]==mod]
            summary[group][mod]=summarize(rows)
        summary[group]["by_subject"]={str(s):{m:summarize([r for r in vals if r["model"]==m and r["subject"]==s]) for m in ("analytic","measured")}
                                      for s in sorted({r["subject"] for r in vals})}
    payload={"model_sha256":frozen,"loaded_measured_poses":total,"results":summary,
             "credit":manifest["credit"],"license":manifest["license"],"assessment_only":True}
    (HERE/"external_results.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({g:{k:v for k,v in s.items() if k!="by_subject"} for g,s in summary.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"round2_data/spatial/hutubs")
    run(p.parse_args().data)
