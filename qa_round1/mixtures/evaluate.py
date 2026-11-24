"""Held-out three-way speech, train and wind separation using the shipped CLI."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
# Reuse identical data, summing and SI-SDR conventions as the RNNoise evaluation.
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("rnnoise_eval", HERE.parent/"rnnoise"/"evaluate.py")
shared = module_from_spec(spec)
spec.loader.exec_module(shared)
RATE = 48000
SOURCE_NAMES=("speech","train","wind")
TRIALS=(
    ("198-209-0000.ogg","1-119125-A-45.wav","1-137296-A-16.wav"),
    ("3436-172162-0000.ogg","2-122066-A-45.wav","2-104952-A-16.wav"),
    ("5703-47212-0000.ogg","3-135650-A-45.wav","3-117504-A-16.wav"),
)
PANS=((-0.55,0.0,0.55),(0.0,0.55,-0.55),(0.55,-0.55,0.0))
SOLO=1.25
SLOT=1.5
MIX_START=4.5
HELD_START=2.0
HELD_END=4.75


def wav_in(path):
    data=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",str(path),
                                  "-map","0:a:0","-f","f32le","-acodec","pcm_f32le","pipe:1"])
    return np.frombuffer(data,"<f4").reshape(-1,2)


def write_mix(path,y):
    subprocess.run(["ffmpeg","-nostdin","-y","-v","error","-f","f32le","-ar",str(RATE),
                    "-ac","2","-i","pipe:0","-c:a","pcm_f32le",str(path)],
                   input=np.asarray(y,dtype="<f4").tobytes(),check=True)


def build(waves,names,pans):
    nt=int((MIX_START+HELD_END-HELD_START)*RATE)
    truth={}
    config={"tracks":[]}
    scored=slice(int(MIX_START*RATE),nt)
    for i, (key,filepath,position) in enumerate(zip(SOURCE_NAMES,names,pans)):
        original=shared.pan(waves[filepath],position)
        held=original[int(HELD_START*RATE):int(HELD_END*RATE)]
        gain=.045 * 10**(-shared.db_rms(held)/20)
        source=np.zeros((nt,2),dtype=np.float32)
        start=int(SLOT*i*RATE)
        source[start:start+int(SOLO*RATE)]=original[:int(SOLO*RATE)]*gain
        source[scored]=held*gain
        truth[key]=source
        config["tracks"].append({"id":key,"mode":"example","span":[SLOT*i+.025,SLOT*i+1.225]})
    mix=sum(truth.values())
    headroom=max(1.0,float(np.max(np.abs(mix)))/.95)
    return {k:v/headroom for k,v in truth.items()}, mix/headroom, config,scored


def run_trial(work,project,trial,names,pans,waves):
    truth,x,config,window=build(waves,names,pans)
    write_mix(work/"input.wav",x)
    (work/"config.json").write_text(json.dumps(config))
    subprocess.run([sys.executable,"-m","voicehunter.multitrack",str(work/"input.wav"),
                    "--config",str(work/"config.json"),"--out",str(work/"out")],
                   cwd=project,check=True,capture_output=True,timeout=180)
    stems={k:wav_in(work/"out"/(k+".wav")) for k in [*SOURCE_NAMES,"remaining"]}
    if any(y.shape!=x.shape or not np.isfinite(y).all() for y in stems.values()):
        raise ValueError("Invalid output length, channels or non-finite samples")
    recon=float(np.max(np.abs(sum(v.astype(np.float64) for v in stems.values()) - x)))
    rows=[]
    for k,source in zip(SOURCE_NAMES,names):
        before=shared.sisdr(x[window],truth[k][window])
        after=shared.sisdr(stems[k][window],truth[k][window])
        rows.append({"trial":trial,"target":k,"filepath":source,
                     "input_sisdr_db":before,"output_sisdr_db":after,"gain_db":after-before,
                     "rms_gain_db":float(shared.db_rms(stems[k][window])-shared.db_rms(truth[k][window])),
                     "target_gate":"PASS" if after-before>=3 else "FAIL",
                     "max_reconstruction_error":recon,"status":"OK"})
    return rows, {"trial":trial,"status":"PASS" if recon<=1e-6 and all(
        row["target_gate"]=="PASS" for row in rows) else "FAIL","n_targets_passed":sum(
            row["target_gate"]=="PASS" for row in rows),"max_reconstruction_error":recon}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product-root",type=Path,default=HERE.parents[1])
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE)
    p.add_argument("--fetch",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/"sources.json").read_text())
    waves,missing=shared.read_sources(args.cache,manifest,args.fetch)
    rows,scenes=[],[]
    for i,(names,pans) in enumerate(zip(TRIALS,PANS),start=1):
        try:
            if any(name not in waves for name in names):
                raise FileNotFoundError("Missing or unavailable fixture, see missing_sources")
            with tempfile.TemporaryDirectory(prefix="voicehunter-overlap-") as tmp:
                result,scene=run_trial(Path(tmp),args.product_root.resolve(),i,names,pans,waves)
        except Exception as exc:
            status="SKIP" if isinstance(exc,FileNotFoundError) else "ERROR"
            detail=str(exc)
            if isinstance(exc,subprocess.CalledProcessError) and exc.stderr:
                detail+=" "+exc.stderr.decode(errors="replace")[-600:]
            result=[dict(trial=i,target=k,status=status,detail=detail) for k in SOURCE_NAMES]
            scene={"trial":i,"status":status,"detail":detail}
        rows.extend(result);scenes.append(scene)
    files=["voicehunter/"+f+".py" for f in ("multitrack","media","separation","spatial")]
    summary={"planned_scenes":3,"scenes":scenes,"scene_pass_count":sum(x["status"]=="PASS" for x in scenes),
             "target_pass_count":sum(x.get("target_gate")=="PASS" for x in rows),
             "source_count":len(waves),"missing_sources":missing,"threshold_db":3,
             "by_target":{k:[round(x["gain_db"],5) for x in rows if x["target"]==k and "gain_db" in x] for k in SOURCE_NAMES},
             "product_sha256":{s:hashlib.sha256((args.product_root/s).read_bytes()).hexdigest() for s in files}}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    keys=list(dict.fromkeys(k for x in rows for k in x))
    with (args.out/"sources-scored.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":
    main()
