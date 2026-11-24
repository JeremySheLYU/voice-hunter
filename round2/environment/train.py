"""Fit two fixed-location spatial filter parameters and evaluate on disjoint recordings."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from voicehunter import joint_spatial, multitrack, separation, spatial
from voicehunter.media import SAMPLE_RATE, load_media, write_wav

SR = SAMPLE_RATE
SOLO = 1.0
OVERLAP = 1.65
START = 3.45
GRID = [(tau, power) for tau in (.025,.075,.20) for power in (.45,1.25)]
CHECKPOINT = ROOT/"voicehunter/models/joint_spatial.json"
SCORED = slice(round((START+.04)*SR),round((START+OVERLAP-.04)*SR))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(y, ref, source):
    def sisdr(x):
        e = x[SCORED].astype("f8")
        r = ref[SCORED].astype("f8")
        e -= np.mean(e,axis=0); r -= np.mean(r,axis=0)
        p = r*(np.sum(r*e)/np.sum(r*r))
        return float(10*np.log10(max(1e-20,np.sum(p*p))/max(1e-20,np.sum((e-p)**2))))
    baseline = sisdr(source)
    actual = sisdr(y)
    return dict(input_sisdr=baseline,estimated_sisdr=actual,improvement=actual-baseline)


@lru_cache(maxsize=192)
def excerpt(path, channels=1):
    raw=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",str(path),"-map","0:a:0","-t","18",
                                 "-ar",str(SR),"-ac",str(channels),"-f","f32le","pipe:1"])
    audio=np.frombuffer(raw,"<f4").copy()
    if channels==2: audio=audio.reshape(-1,2)
    L=round(SOLO*SR); H=round(OVERLAP*SR)
    if len(audio)<L+H+round(.08*SR):
        raise ValueError("too short for disjoint source intervals")
    # For transient recordings a literal leading zero would be an invalid
    # reference. Choose energetic non-overlapping cuts using only the raw audio.
    power=audio.astype("f8")**2
    if channels==2: power=power.mean(axis=1)
    sums=np.r_[0,np.cumsum(power)]
    step=round(SR*.08)
    pivot=min(round(4*SR), max(L,len(audio)//2))
    pivot=min(len(audio)-H-round(.08*SR),pivot)
    starts=np.arange(0,pivot-L+1,step,dtype=int)
    finish=np.arange(pivot+round(.08*SR),len(audio)-H+1,step,dtype=int)
    first=starts[np.argmax(sums[starts+L]-sums[starts])]
    second=finish[np.argmax(sums[finish+H]-sums[finish])]
    ref=audio[first:first+L].copy(); held=audio[second:second+H].copy()
    ref-=ref.mean(axis=0); held-=held.mean(axis=0)
    if min(float(np.sqrt(np.mean(ref**2))),float(np.sqrt(np.mean(held**2))))<1e-5:
        raise ValueError("insufficient signal on the two disjoint intervals")
    return ref,held,[int(first),int(second)]


def load_index(path):
    records=json.loads(path.read_text())
    partitions={key:[] for key in ["train","validation","test"]}
    speakers={}; content={}; rejected=[]
    forbidden={v["sha256"] for v in json.loads((ROOT/"qa_round1/mixtures/sources.json").read_text()).values()}
    for r in sorted(records,key=lambda item:str(item["id"])):
        split={"val":"validation","dev":"train","valid":"validation","eval":"test"}.get(r["split"],r["split"])
        if split not in partitions:
            continue
        filepath=ROOT/r["path"]
        if r["sha256"] in forbidden: raise ValueError("Legacy test file appeared in new cohort")
        if content.setdefault(r["sha256"],split)!=split: raise ValueError("Recording split overlap")
        if speakers.setdefault(r["author"].casefold().strip(),split)!=split: raise ValueError("Uploader split overlap")
        # The collection process has independently screened recording permissions.
        license_text=r["license"].lower()
        if ("cc0" not in license_text and "publicdomain/zero" not in license_text and
                "cc by" not in license_text and "cc-by" not in license_text and "/by/" not in license_text):
            raise ValueError("Unsupported source permission: "+r["license"])
        if any(term in license_text for term in ["-nc","-nd","-sa","/by-nc","/by-nd","/by-sa"]):
            raise ValueError("Restricted source permission: "+r["license"])
        if r.get("duration",0)<SOLO+OVERLAP+.08: continue
        try:
            if digest(filepath)!=r["sha256"]: raise ValueError("Source hash mismatch")
            excerpt(filepath)
        except subprocess.CalledProcessError:
            raise
        except ValueError as e:
            if "mismatch" in str(e): raise
            rejected.append(dict(id=r["id"],reason=str(e)))
            continue
        partitions[split].append(r)
    return partitions,rejected


def images(r,rng,position,delayed,level):
    ref,held,offsets=excerpt(ROOT/r["path"])
    gain=.045*10**(level/20)/max(1e-9,np.sqrt(np.mean(held**2)))
    a=position*np.pi/2
    direction=np.array([np.cos(a),np.sin(a)],np.float32)
    delay=int(round(abs(position-.5)*24)) if delayed else 0
    far=1 if position<.5 else 0
    def render(mono):
        stereo=mono[:,None]*direction
        if delay: stereo[delay:,far],stereo[:delay,far]=stereo[:-delay,far].copy(),0.
        return stereo*gain
    return render(ref),render(held),offsets


def scene(records, seed, settings, include_extra):
    rng=np.random.default_rng(seed)
    places=list(rng.permutation(np.array([.19,.5,.81])+rng.uniform(-.045,.045,size=3)))
    truth=[]; config={"tracks":[],"separator":"adaptive"}; details=[]
    length=round((START+OVERLAP)*SR)
    mix=np.zeros((length,2),np.float32)
    for k,r in enumerate(records[:3]):
        name=f"source{k+1}"
        profile=float(rng.uniform(-2,2))
        ref,held,offsets=images(r,rng,places[k],settings=="delay",profile)
        source=np.zeros_like(mix); start=round(k*1.15*SR)
        source[start:start+len(ref)]=ref
        source[round(START*SR):]=held
        truth.append(source);mix+=source
        config["tracks"].append(dict(id=name,span=[k*1.15+.04,k*1.15+.96]))
        details.append(dict(track=name,asset=r["id"],author=r["author"],category=r["category"],
                            path=r["path"],sha256=r["sha256"],ref_held_start_samples=offsets,
                            pan_fraction=places[k],relative_db=profile))
    if include_extra and len(records)>3:
        extra=records[3]
        if extra["channels"] == 2:
            _,held,offsets=excerpt(ROOT/extra["path"],2)
            noise=held*(.045*10**(-12/20)/max(1e-9,float(np.sqrt(np.mean(held**2)))))
            bed="captured stereo retained"
        else:
            _,noise,offsets=images(extra,rng,.36,settings=="delay",-12)
            bed="panned mono"
        mix[round(START*SR):]+=noise
        details.append(dict(track="unassigned",asset=extra["id"],author=extra["author"],category=extra["category"],
                            path=extra["path"],sha256=extra["sha256"],ref_held_start_samples=offsets,relative_db=-12,bed=bed))
    gain=max(1.,float(np.max(np.abs(mix)))/.92)
    return mix/gain,[s/gain for s in truth],config,details


def scores(stems,truth,mix):
    return [measure(stem,ref,mix) for stem,ref in zip(stems,truth)]


def groups(records, max_scenes, seed, group_size=4):
    # Category-balanced shuffled pools. Roll past used authors within each scene.
    rng=np.random.default_rng(seed)
    bycat={}
    for r in records: bycat.setdefault(r["category"],[]).append(r)
    pools={k:list(rng.permutation(v)) for k,v in bycat.items()}
    counts=Counter()
    for index in range(max_scenes):
        used=set(); picked=[]
        for k in range(group_size):
            pool=[r for r in records if r["author"] not in used and r["id"] not in [p["id"] for p in picked]]
            if not pool: break
            cats={p["category"] for p in picked}
            def cost(r):
                return (int(k==3 and index%4==3 and r["channels"]!=2),counts[r["id"]],int(r["category"] in cats), list(x["id"] for x in pools[r["category"]]).index(r["id"]))
            choice=min(pool,key=cost)
            picked.append(choice); used.add(choice["author"]);counts[choice["id"]]+=1
        if len(picked)>=3: yield picked


@lru_cache(maxsize=16)
def real_ambience(receipt_path):
    receipt=json.loads(receipt_path.read_text())
    path=ROOT/receipt["path"]
    if digest(path)!=receipt["sha256"]: raise ValueError("Ambisonics hash mismatch")
    raw=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-i",str(path),"-map","0:a:0",
                                 "-ar",str(SR),"-f","f32le","pipe:1"])
    ambi=np.frombuffer(raw,"<f4").reshape(-1,4)
    w,y,_,x=ambi[:round(OVERLAP*SR)].T
    c=np.sqrt(.5)
    stereo=.5*np.stack((w+c*x+c*y,w+c*x-c*y),axis=1)
    return receipt,stereo


def collect(records, split, candidates, limit, seed, production):
    out=[]
    for i,picked in enumerate(groups(records,limit,seed)):
        use_foa=split=="train" and i%8==7
        x,truth,config,details=scene(picked,seed+i,"delay" if i%2 else "pan",i%4==3 and not use_foa)
        if use_foa:
            site_position=1+(i//8)%5
            filename=f"FirstOrderAmbisonics_20220202_Wed_12h00_Pos{site_position}_30sec.wav.json"
            receipt,bed=real_ambience(ROOT/"round2/fielddata/receipts/ihta_foa"/filename)
            gain=.045*10**(-12/20)/float(np.sqrt(np.mean(bed**2)))
            x[round(START*SR):]+=bed*gain
            details.append(dict(track="unassigned",asset=receipt["id"],sha256=receipt["sha256"],path=receipt["path"],
                                bed="real FOA, ACN SN3D virtual ±45 degree cardioids",relative_db=-12))
            scale=max(1.,float(np.max(np.abs(x)))/.92)
            x/=scale;truth=[t/scale for t in truth]
        tracks=config["tracks"]
        masks,_=multitrack._joint_masks(x,tracks)
        old=[spatial.extract_stereo(x,m)[0] for m in masks]
        case=dict(scene=i,split=split,capture="ITD + ILD" if i%2 else "ILD",
                  recordings=details,baseline=scores(old,truth,x),candidates={})
        if production:
            with tempfile.TemporaryDirectory() as td:
                base=Path(td);write_wav(base/"in.wav",x,stem=True)
                meta=multitrack.export_tracks(base/"in.wav",config,base/"out")
                stems=[load_media(base/"out"/(t["id"]+".wav"))[0] for t in tracks]
                rem=load_media(base/"out/remaining.wav")[0]
                case.update(projection=meta["projection"],estimate=scores(stems,truth,x),
                            additive_error=float(abs(sum((y.astype("f8") for y in stems),rem.astype("f8"))-x).max()))
        else:
            spect=spatial.stft(x)
            refs=joint_spatial.reference_templates(spect,[t["span"] for t in tracks],SR,separation.HOP)
            for tau,alpha in candidates:
                preds=old if refs is None else joint_spatial.reconstruct(spect,masks,refs,len(x),
                                    separation.N_FFT,separation.HOP,tau,alpha)
                case["candidates"][f"{tau}:{alpha}"]=scores(preds,truth,x)
            case["eligible_spatial"]=refs is not None
        out.append(case)
    return out


def summary(cases,label):
    rows=[c[label] if label in c else c["candidates"][label] for c in cases]
    array=np.asarray([[r["improvement"] for r in v] for v in rows])
    return dict(n=len(rows),scene_pass=int((array.min(axis=1)>=3).sum()),track_pass=int((array>=3).sum()),
                mean_worst=float(np.mean(array.min(axis=1))),median=float(np.median(array)),
                worst=float(np.min(array)),mean=float(np.mean(array)))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=["fit","test"])
    cli.add_argument("--manifest",type=Path,default=HERE/"cohort.json")
    args=cli.parse_args()
    parts,rejected=load_index(args.manifest)
    splits={k:len(v) for k,v in parts.items()}
    if any(n<3 for n in splits.values()): raise ValueError("Require >=3 distinct sources in each split: "+repr(splits))
    if args.phase=="fit":
        train=collect(parts["train"],"train",GRID,min(36,max(6,(len(parts["train"])+2)//3)),3321,False)
        baseline=summary(train,"baseline")
        ranking=sorted(((summary(train,f"{t}:{p}")["mean_worst"],t,p) for t,p in GRID),reverse=True)
        _,tau,alpha=ranking[0]
        val=collect(parts["validation"],"validation",[(tau,alpha)],min(18,max(6,(len(parts["validation"])+2)//3)),5729,False)
        v_new,v_old=summary(val,f"{tau}:{alpha}"),summary(val,"baseline")
        fit=dict(criterion="mean weakest named-image paired SI-SDR improvement",baseline=baseline,
                 candidates={f"{t}:{p}":summary(train,f"{t}:{p}") for t,p in GRID},
                 train=train,validation=val,validation_result=v_new,validation_baseline=v_old,
                 manifest_sha=digest(args.manifest),eligible=splits,rejected=rejected)
        (HERE/"fit.json").write_text(json.dumps(fit,indent=2)+"\n")
        if v_new["mean_worst"]<v_old["mean_worst"] or ranking[0][0]<baseline["mean_worst"]:
            raise RuntimeError("Spatial candidate did not clear validation veto")
        CHECKPOINT.write_text(json.dumps(dict(model="reference spatial Wiener v1",temperature=tau,spectral_exponent=alpha,
            supervised_objective=fit["criterion"],training_scenes=len(train),validation_scenes=len(val),
            source_manifest_sha256=digest(args.manifest)),indent=2)+"\n")
        print("training",baseline,ranking,"validation",v_old,v_new,flush=True)
    else:
        fit=json.loads((HERE/"fit.json").read_text())
        if fit["manifest_sha"]!=digest(args.manifest):raise ValueError("Test cohort changed after calibration")
        cases=collect(parts["test"],"test",[],min(24,max(8,(len(parts["test"])+2)//3)),9107,True)
        unseen=dict(baseline=summary(cases,"baseline"),estimate=summary(cases,"estimate"),
                    model_sha=digest(CHECKPOINT),manifest_sha=digest(args.manifest),cases=cases)
        (HERE/"test.json").write_text(json.dumps(unseen,indent=2)+"\n")
        print("test",unseen["baseline"],unseen["estimate"])
if __name__=="__main__": main()
