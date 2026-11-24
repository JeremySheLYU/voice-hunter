"""Fit horizontal interaural cue table on train subjects; evaluate rendered filters."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from voicehunter.binaural import render_headphones
from voicehunter.hrtf_cues import load_profile, relative_phase_delay_us

RATE = 44100
ANGLES = [-80,-65,-55,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,55,65,80]
BANDS = [250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12000]
HELD_ANGLES = [10,35,55]


def mat_element(blob, offset):
    tag, size = struct.unpack_from("<II", blob, offset)
    if tag >> 16:
        return tag & 65535, blob[offset+4:offset+4+(tag>>16)], offset+8
    return tag, blob[offset+8:offset+8+size], offset+8+(size+7)//8*8


def ears(path):
    # CIPIC Release 1.1 uses uncompressed little-endian Matlab v5 real doubles.
    blob = memoryview(path.read_bytes())
    if blob[126:128] != b"IM":
        raise ValueError(f"Unknown MAT endian/layout in {path}")
    offset = 128
    matrices = {}
    while offset < len(blob):
        kind, payload, offset = mat_element(blob,offset)
        if kind != 14:
            raise ValueError("Expected uncompressed miMATRIX")
        _, _, j = mat_element(payload,0)
        _, dimensions, j = mat_element(payload,j)
        _, name, j = mat_element(payload,j)
        key = bytes(name).decode()
        if key in ("hrir_l","hrir_r"):
            kind, values, _ = mat_element(payload,j)
            shape = tuple(np.frombuffer(dimensions,dtype="<i4"))
            if kind != 9 or shape != (25,50,200):
                raise ValueError(f"Unexpected {key} matrix type or shape")
            matrices[key] = np.frombuffer(values,dtype="<f8").reshape(shape,order="F")[:,8,:].copy()
    return matrices["hrir_l"],matrices["hrir_r"]


def cue(near, far):
    f = np.fft.rfftfreq(4096,1/RATE)
    near_power, far_power = abs(np.fft.rfft(near,4096))**2, abs(np.fft.rfft(far,4096))**2
    ild = []
    for center in BANDS:
        band = (f >= center * 2**(-1/3)) & (f <= center * 2**(1/3))
        ild.append(10*np.log10(max(near_power[band].sum(),1e-14)/max(far_power[band].sum(),1e-14)))
    return np.array(ild), relative_phase_delay_us(near,far,RATE)


def response_cues(directory, ids):
    cues = {}
    for key in ids:
        l,r = ears(directory/f"subject_{key}.mat")
        for i,angle in enumerate(ANGLES):
            if angle == 0:
                continue
            near,far = (l[i],r[i]) if angle<0 else (r[i],l[i])
            cues[key,angle] = cue(near,far)
    return cues


def fit(cues, training):
    profile = {"description":"CIPIC generic symmetric interaural cue calibration, horizontal, 1 m.",
               "source_notice":"CIPIC-NOTICE.txt","bands_hz":BANDS,"angles_deg":[0],
               "lowband_delay_us":[0.0],"near_far_ild_db":[[0.0]*len(BANDS)]}
    for a in sorted(set(ANGLES) - set(HELD_ANGLES)):
        if a<=0: continue
        rows = [cues[key,signed] for key in training for signed in (a,-a)]
        db = np.median(np.stack([row[0] for row in rows]),axis=0).clip(-24,24)
        profile["angles_deg"].append(a)
        profile["near_far_ild_db"].append(db.round(6).tolist())
        profile["lowband_delay_us"].append(round(float(np.median([row[1] for row in rows])),6))
    return profile


def render_cues(profile_name, angle, rate=RATE):
    impulse = np.zeros((4096,2),dtype="f4")
    impulse[0] = 1
    wave = render_headphones(impulse,np.zeros_like(impulse),rate,azimuth_deg=angle,profile=profile_name)
    near,far = (wave[:,0],wave[:,1]) if angle<0 else (wave[:,1],wave[:,0])
    return cue(near,far)


def summarize(rows):
    return {"directions":len(rows),
            "ILD_MAE_dB":float(np.mean([r["ILD_MAE_dB"] for r in rows])),
            "ITD_MAE_us":float(np.mean([r["ITD_abs_error_us"] for r in rows]))}


def evaluate(split, ids, cues, rendered):
    rows = []
    for key in ids:
        for angle in ANGLES:
            if not angle: continue
            measured_ild, measured_itd = cues[key,angle]
            for model in ("analytic","measured"):
                estimate_ild, estimate_itd = rendered[model,angle]
                rows.append(dict(id=key,angle=angle,model=model,
                                 ILD_MAE_dB=float(np.abs(measured_ild-estimate_ild).mean()),
                                 ITD_abs_error_us=abs(measured_itd-estimate_itd)))
    grouped = {}
    for model in ("analytic","measured"):
        subset = [row for row in rows if row["model"]==model]
        grouped[model] = dict(all=summarize(subset),
                              unseen_angles=summarize([r for r in subset if abs(r["angle"]) in HELD_ANGLES]),
                              by_subject={key:summarize([r for r in subset if r["id"]==key]) for key in ids})
    return grouped, rows


def controls():
    rng=np.random.default_rng(313); target=rng.normal(0,0.08,(12000,2)).astype("f4"); zeros=np.zeros_like(target)
    residual=np.column_stack((rng.uniform(-.04,.04,len(target)),rng.uniform(-.04,.04,len(target)))).astype("f4")
    # Compare mirror and residual at unity gain, independent of the acoustic reference.
    tests=[]
    for rate in (8000,16000,44100,48000,96000,192000):
        centered=render_headphones(target,zeros,rate,profile="measured")
        null=render_headphones(zeros,residual,rate,azimuth_deg=-45,profile="measured")
        sym=0.
        for angle in (5.0,10.5,35.0,55.0,80.0,90.0):
            pos=render_headphones(target,zeros,rate,azimuth_deg=angle,profile="measured")
            neg=render_headphones(target,zeros,rate,azimuth_deg=-angle,profile="measured")
            assert pos.shape==target.shape and np.isfinite(neg).all() and np.isfinite(pos).all()
            sym=max(sym,float(np.max(abs(pos-neg[:,::-1]))))
        tests.append(dict(rate=rate,mid_max_error=float(abs(centered-target.mean(axis=1,keepdims=True)).max()),
                          residual_max_error=float(abs(null-residual).max()),mirror_max_error=sym))
    return tests


def run(directory):
    manifest=json.loads((HERE/"manifest.json").read_text())
    all_ids=sorted(row["file"][8:11] for row in manifest if row["file"].endswith(".mat"))
    for row in manifest:
        if hashlib.sha256((directory/row["file"]).read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError(f"Input hash mismatch in {row['file']}.")
    human=sorted((k for k in all_ids if k not in ("021","165")),
                 key=lambda s:hashlib.sha256(("vh2-cipic-2026/"+s).encode()).hexdigest())
    groups=dict(train=human[:27],validation=human[27:33],test=human[33:],mannequin=["021","165"])
    assert (len(all_ids),len(human),len(groups["test"]))==(45,43,10)
    cues=response_cues(directory,groups["train"])
    profile=fit(cues,groups["train"])
    path=ROOT/"voicehunter/models/cipic_horizontal_v1.json"
    path.write_text(json.dumps(profile,indent=2)+"\n")
    load_profile.cache_clear()
    rendered={(model,a):render_cues(model,a) for model in ("analytic","measured") for a in ANGLES if a}
    reports, details={},{}
    for split, ids in groups.items():
        if split=="train": continue
        new=response_cues(directory,ids)
        reports[split],details[split]=evaluate(split,ids,new,rendered)
    ctl=controls()
    result=dict(groups=groups,angles_deg=ANGLES,unseen_abs_angles=HELD_ANGLES,
                train_measurements=27*2*(len(ANGLES[13:])-len(HELD_ANGLES)),
                raw_pose_count=45*25*50,horizontal_poses_in_raw=45*25,reports=reports, controls=ctl,
                weight_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rate=RATE)
    test=reports["test"]
    result["passes_frozen_gate"]=all(test["measured"][s][m]<test["analytic"][s][m]
                                      for s in ("all","unseen_angles") for m in ("ILD_MAE_dB","ITD_MAE_us"))
    (HERE/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    (HERE/"individual_direction_errors.json").write_text(json.dumps(details,indent=2)+"\n")
    print(json.dumps({p:{m:g["all"] for m,g in v.items()} for p,v in reports.items()},indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"round2_data/spatial/cipic")
    args=p.parse_args()
    run(args.data)
