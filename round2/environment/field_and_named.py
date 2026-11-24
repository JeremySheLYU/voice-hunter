"""Held-out registered speech/train/wind with genuine measured AmbiX soundscapes."""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import train

SITES=["TrainStation","Woodland","Beach","Park","ShoppingCentre","BusyStreet"]


def foa_bed(receipt, n):
    path=train.ROOT/receipt["path"]
    if train.digest(path)!=receipt["sha256"]:raise ValueError("Changed FOA source")
    raw=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",str(path),"-map","0:a:0",
                                 "-ar",str(train.SR),"-f","f32le","pipe:1"])
    ambix=np.frombuffer(raw,"<f4").reshape(-1,4)
    if len(ambix)<n:raise ValueError("Truncated FOA recording")
    w,y,_,x=ambix[:n].T
    scale=np.sqrt(.5)
    return np.stack((w+scale*x+scale*y,w+scale*x-scale*y),axis=1)*.5


def run_one(items, i, receipt, level):
    audio,refs,config,description=train.scene(items,17839+i,"delay" if i%2 else "pan",False)
    offset=round(train.START*train.SR)
    bed=np.zeros_like(audio)
    if level is not None:
        direct=foa_bed(receipt,len(audio)-offset)
        rms=float(np.sqrt(np.mean(direct**2)))
        bed[offset:]=direct*(.045 * 10**(level/20)/rms)
        audio+=bed
    scale=max(1,float(np.max(np.abs(audio)))/.92)
    audio/=scale;refs=[r/scale for r in refs];bed/=scale
    for t,it in zip(config["tracks"],items):t["id"]=it["category"]
    old_masks,_=train.multitrack._joint_masks(audio,config["tracks"])
    old=[train.spatial.extract_stereo(audio,m)[0] for m in old_masks]
    case=dict(trial=i,source=description,ambient_level_db=level,field_id=receipt["id"] if level is not None else None,
              measured_format="ACN SN3D FOA, ±45 degree cardioids",categories=[r["category"] for r in items],
              baseline=train.scores(old,refs,audio))
    with tempfile.TemporaryDirectory() as td:
        base=Path(td);train.write_wav(base/"input.wav",audio,stem=True)
        meta=train.multitrack.export_tracks(base/"input.wav",config,base/"out")
        stems=[train.load_media(base/"out"/(t["id"]+".wav"))[0] for t in config["tracks"]]
        rem=train.load_media(base/"out/remaining.wav")[0]
        case.update(projection=meta["projection"],estimate=train.scores(stems,refs,audio),
                    additive_error=float(np.max(abs(sum((s.astype("f8") for s in stems),rem.astype("f8"))-audio))))
        if level is not None:
            case["ambient_baseline"]=train.measure(audio-sum(old),bed,audio)
            case["ambient_estimate"]=train.measure(rem,bed,audio)
    return case


def main():
    parts,_=train.load_index(train.HERE/"cohort.json")
    seen=set();cases=[];raw=[]
    for i,site in enumerate(SITES):
        slots=[];authors=set()
        for category in ["speech","train","wind"]:
            choices=[r for r in parts["test"] if r["category"]==category and r["id"] not in seen and r["author"] not in authors]
            if not choices:break
            chosen=choices[0];slots.append(chosen);authors.add(chosen["author"])
        if len(slots)!=3:break
        seen.update(r["id"] for r in slots)
        receipt_path=train.ROOT/"round2/fielddata/receipts/eigenscape"/(site+".8.flac.json")
        receipt=json.loads(receipt_path.read_text());raw.append(receipt)
        for level in [None,-6,0]:
            cases.append(run_one(slots,i+1,receipt,level))
    out=dict(model_sha=train.digest(train.CHECKPOINT),FOA_sources=raw,source_manifest=train.digest(train.HERE/"cohort.json"),
             cases=cases,levels={})
    for level in [None,-6,0]:
        rows=[c for c in cases if c["ambient_level_db"]==level]
        section=dict(baseline=train.summary(rows,"baseline"),estimate=train.summary(rows,"estimate"))
        if level is not None:
            section["ambient_median_sisdri_baseline"]=float(np.median([r["ambient_baseline"]["improvement"] for r in rows]))
            section["ambient_median_sisdri_estimate"]=float(np.median([r["ambient_estimate"]["improvement"] for r in rows]))
        out["levels"][str(level)]=section
    (train.HERE/"field_and_named.json").write_text(json.dumps(out,indent=2)+"\n")
    print(out["levels"])
if __name__=="__main__":main()
