"""Deterministic paired evaluation. No fixture download unless --fetch is supplied."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
RATE = 48000
N = RATE * 5
VOICE_FILES = dict(reader_198="198-209-0000.ogg", reader_3436="3436-172162-0000.ogg",
                   reader_5703="5703-47212-0000.ogg")
NOISE_FILES = dict(wind="1-137296-A-16.wav", train="1-119125-A-45.wav",
                   vacuum="1-100210-A-36.wav", music="Kevin_MacLeod_-_Vibe_Ace.ogg")


def db_rms(x):
    return 10 * np.log10(max(np.mean(np.square(x, dtype=np.float64)), 1e-30))


def normalize(x):
    return x * (0.045 / 10**(db_rms(x) / 20))


def pan(x, angle):
    theta = np.pi / 4 * (1 + angle)
    return np.column_stack((x * np.cos(theta), x * np.sin(theta)))


def sisdr(y, s):
    e = np.array(y, dtype=np.float64, copy=True)
    t = np.array(s, dtype=np.float64, copy=True)
    e -= np.mean(e, axis=0)
    t -= np.mean(t, axis=0)
    scale = np.sum(e*t) / np.sum(t*t)
    p = scale*t
    return float(10*np.log10(max(np.sum(p*p), 1e-24) / max(np.sum((e-p)**2), 1e-24)))


def peak_lag(x, y):
    a = x.mean(axis=1)
    b = y.mean(axis=1)
    n_fft = 1 << (2*len(a)-1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(b,n_fft)*np.conj(np.fft.rfft(a,n_fft)), n_fft)
    indices = np.arange(-960,961)
    return int(indices[np.argmax(corr[indices])])


def read_sources(cache, manifest, download):
    cache.mkdir(parents=True, exist_ok=True)
    waves, missing = {}, {}
    for name, item in manifest.items():
        path = cache / name
        if not path.exists() and download:
            try:
                with urllib.request.urlopen(item["source_url"], timeout=30) as data:
                    path.write_bytes(data.read())
            except OSError as e:
                missing[name] = "download error: "+str(e)
                continue
        if not path.exists():
            missing[name] = "fixture absent"
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Hash mismatch: "+name)
        wav = subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",str(path),
                                       "-map","0:a:0","-ac","1","-ar",str(RATE),
                                       "-t","5","-f","f32le","pipe:1"])
        x = np.frombuffer(wav, dtype="<f4").copy()
        if len(x) < N:
            raise ValueError(f"{name} has fewer than 5 seconds of samples")
        waves[name] = x[:N] - x[:N].mean()
    return waves, missing


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for x in rows for k in x))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product-root", type=Path, default=HERE.parents[1])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--fetch", action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.product_root))
    from voicehunter.engine import separate_clip
    from voicehunter.pretrained import capabilities, separate_pretrained
    manifest=json.loads((HERE/"sources.json").read_text())
    waves,missing=read_sources(args.cache,manifest,args.fetch)
    voices={key:normalize(pan(waves[name],az)) for (key,name),az in
            zip(VOICE_FILES.items(),[-0.55,0.55,0.0]) if name in waves}
    if "reader_198" in voices and "reader_3436" in voices:
        delay=int(.30 * RATE)
        voices["overlap_198_3436"]=normalize(voices["reader_198"] +
                   np.pad(voices["reader_3436"][:-delay], ((delay,0),(0,0))))
    noises={key:normalize(pan(waves[name],az)) for (key,name),az in
            zip(NOISE_FILES.items(),[0.15,-0.70,0.45,-0.25]) if name in waves}
    rng=np.random.default_rng(1801)
    noises["white"]=normalize(pan(rng.normal(size=N).astype(np.float32),0.35))
    voices={k:v.astype(np.float32) for k,v in voices.items()}
    noises={k:v.astype(np.float32) for k,v in noises.items()}
    rows=[]
    def evaluate(kind,speaker,noise,snr,x,s):
        x=np.asarray(x,dtype=np.float32)
        row={"condition":kind,"speaker":speaker,"noise":noise,"snr_db":snr,"status":"PASS"}
        rows.append(row)
        try:
            t,b,_=separate_clip(x,"rnnoise","voice")
            finite=np.isfinite(t).all() and np.isfinite(b).all()
            if (t.shape!=x.shape or b.shape!=x.shape or not finite):
                raise ValueError(f"Invalid outputs target={t.shape}, background={b.shape}, finite={finite}")
            err=float(np.max(np.abs(t+b-x)))
            row["max_reconstruction_error"]=err
            passed=err<=1e-6
            if kind=="mixture":
                before=sisdr(x,s)
                after=sisdr(t,s)
                row.update(input_sisdr_db=before,output_sisdr_db=after,sisdri_db=after-before)
                passed &= after-before>=1
            elif kind=="clean":
                after=sisdr(t,s); gain=float(db_rms(t)-db_rms(s));lag=peak_lag(s,t)
                row.update(output_sisdr_db=after,target_gain_db=gain,target_lag_samples=lag)
                passed &= after>=15 and abs(gain)<=3 and abs(lag)<=2
            elif kind=="no_speech":
                attenuation=float(db_rms(x)-db_rms(t))
                row.update(target_suppression_db=attenuation)
                passed &= attenuation>=10
            elif kind=="silence":
                peak=float(np.max(np.abs(t)))
                row["target_peak_fs"]=peak
                passed &= peak<=1e-7
            row["status"]="PASS" if passed else "FAIL"
        except Exception as e:
            row.update(status="ERROR",detail=str(e))

    for speaker in [*VOICE_FILES,"overlap_198_3436"]:
        for noise in [*NOISE_FILES,"white"]:
            for snr in (-5,0,10):
                if speaker not in voices or noise not in noises:
                    rows.append({"condition":"mixture","speaker":speaker,"noise":noise,"snr_db":snr,
                                 "status":"SKIP","detail":"missing source data"})
                    continue
                s=voices[speaker];v=noises[noise]
                mix=s+(10**(-snr/20))*v
                # Preserve ratio and exact stereo reference if peak needs headroom.
                headroom=max(1.0,float(np.max(np.abs(mix))) / 0.95)
                evaluate("mixture",speaker,noise,snr,mix/headroom,s/headroom)
    for speaker in [*VOICE_FILES,"overlap_198_3436"]:
        if speaker in voices: evaluate("clean",speaker,"","",voices[speaker],voices[speaker])
        else: rows.append(dict(condition="clean",speaker=speaker,status="SKIP",detail="missing fixture"))
    for noise in [*NOISE_FILES,"white"]:
        if noise in noises: evaluate("no_speech","",noise,"",noises[noise],None)
        else: rows.append(dict(condition="no_speech",noise=noise,status="SKIP",detail="missing fixture"))
    evaluate("silence","","","",np.zeros((N,2),dtype=np.float32),None)
    # Optional adapters are probed without downloading. Restore local env at exit.
    k="VOICEHUNTER_DEMUCS_CHECKPOINT";saved=os.environ.pop(k,None)
    try:
        unset={m["id"]:m for m in capabilities()}
        os.environ[k]=str(args.cache / "definitely-missing-demucs-checkpoint.th")
        invalid={m["id"]:m for m in capabilities()}
    finally:
        if saved is None:os.environ.pop(k,None)
        else:os.environ[k]=saved
    gates={"rnnoise_ready":unset["rnnoise"]["available"],
           "demucs_unset_rejected":not unset["demucs"]["available"],
           "missing_checkpoint_rejected":not invalid["demucs"]["available"]}
    probe=np.zeros((960,2),dtype=np.float32)
    for model,target,label in [("rnnoise","train","bad_rnnoise_target_rejected"),
                              ("no-such-model","voice","unknown_model_rejected")]:
        try:
            separate_pretrained(probe,RATE,model,target)
            gates[label]=False
        except ValueError:
            gates[label]=True
    optional=[{"adapter":"demucs","status":"SKIP","reason":"No configured, verified four-stem weights for this benchmark."}]
    m=[r for r in rows if r["condition"]=="mixture" and "sisdri_db" in r]
    gains=np.array([r["sisdri_db"] for r in m])
    def stats(vals):
        vals=np.asarray(vals)
        if not len(vals):return None
        return {"n":len(vals),"median":float(np.median(vals)),"min":float(min(vals)),"max":float(max(vals)),
                "positive":int(sum(vals>0)),"at_least_1db":int(sum(vals>=1))}
    case_stats=stats(gains)
    all_tested=len(m)==60
    files=("voicehunter/pretrained.py", "voicehunter/engine.py", "voicehunter/spatial.py",
           "voicehunter/models/std.rnnn")
    product_hashes={name:hashlib.sha256((args.product_root/name).read_bytes()).hexdigest()
                    for name in files}
    result={"product_sha256":product_hashes,"environment":{"python":sys.version.split()[0],"numpy":np.__version__,
              "platform":platform.platform(),"ffmpeg":subprocess.check_output(["ffmpeg","-version"],
              text=True).splitlines()[0]},"planned_cases":70,"n":len(rows),"missing_sources":missing,
            "statuses":{k:sum(r["status"]==k for r in rows) for k in ("PASS","FAIL","ERROR","SKIP")},
            "mixtures":case_stats,"by_noise":{k:stats([r["sisdri_db"] for r in m if r["noise"]==k]) for k in noises},
            "by_snr":{str(k):stats([r["sisdri_db"] for r in m if r["snr_db"]==k]) for k in (-5,0,10)},
            "by_speaker":{k:stats([r["sisdri_db"] for r in m if r["speaker"]==k]) for k in voices},
            "mixture_gate_pass":bool(all_tested and case_stats["at_least_1db"]>=45 and case_stats["median"]>=3),
            "controls":[r for r in rows if r["condition"]!="mixture"],"adapter_gates":gates,"optional":optional,
            "largest_reconstruction_error":max((r.get("max_reconstruction_error",0) for r in rows),default=None),
            "worst":min(m,key=lambda r:r["sisdri_db"]) if m else None}
    (args.out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    write_csv(args.out/"cases.csv",rows)
    print(json.dumps({"rows":len(rows),"status":result["statuses"],"mixtures":case_stats,"mixed_pass":result["mixture_gate_pass"]}))
if __name__=="__main__":
    main()
