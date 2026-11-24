"""Replay two out-of-condition acoustic strata with one positive and one negative request."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from functools import lru_cache

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R2 = ROOT / "round2" / "semantic"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(R2))
import evaluate as legacy_test
import semantic_candidate as legacy_model

LABELS = tuple(legacy_test.C)
SR = legacy_test.SR
WINDOW = slice(round(.15 * SR), round(4.35 * SR))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(parts):
    return hashlib.sha256(("voicehunter-stress-v3|" + "|".join(parts)).encode()).hexdigest()


def create_manifest():
    semantic_path = R2 / "cohort.json"
    native_path = ROOT / "round2/environment/native_inventory.json"
    previous = ROOT / "round2/environment/native_stereo.json"
    excluded = {row["id"] for row in json.loads(previous.read_text())["excluded"]}
    source_rows = [r for r in json.loads(native_path.read_text()) if r["id"] not in excluded]
    assert len(source_rows) == len({r["id"] for r in source_rows}) == 16
    trainval = [r for r in json.loads(semantic_path.read_text()) if r["split"] in ("train", "validation")]
    assert not ({r['id'] for r in trainval} & {r['id'] for r in source_rows})
    assert not ({r['author'].casefold() for r in trainval} & {r['author'].casefold() for r in source_rows})
    for row in source_rows:
        if row["channels"] != 2 or row["duration"] < 4.5 or digest(ROOT / row['path']) != row['sha256']:
            raise ValueError("Frozen native source failed verification: " + row['id'])
    rows = []
    uses = collections.Counter()
    neg_count = collections.Counter()
    for src in sorted(source_rows, key=lambda r: int(r['id'])):
        for distractor in LABELS:
            if distractor == src["category"]:
                continue
            others = [r for r in source_rows if r["category"] == distractor and r["author"].casefold() != src["author"].casefold()]
            b = min(others, key=lambda r: (uses[r["id"]], stable_rank([src["id"], r["id"]])))
            absent = min((c for c in LABELS if c not in (src['category'], distractor)),
                         key=lambda c: (neg_count[c], stable_rank([src["id"], b["id"], c])))
            uses[b["id"]] += 1
            neg_count[absent] += 1
            rows.append(dict(kind="native", source_id=src["id"], background_id=b["id"], target=src["category"],
                             distractor=distractor, absent=absent))
    held = json.loads((R2 / 'test_results.json').read_text())['room']['rows']
    rooms = [{k: r[k] for k in ('kind','source_id','background_id','target','distractor','absent')}
             for r in held if r["present"]]
    return dict(native=rows, room=rooms, native_manifest_sha256=digest(native_path),
                cohort_sha256=digest(semantic_path), semantic_scene_builder_sha256=digest(R2 / "evaluate.py"),
                native_source_ids=[r["id"] for r in source_rows], native_source_sha256={r["id"]:r["sha256"] for r in source_rows},
                room_ir_sha256={str(s):digest(ROOT / f"round2_data/spatial/aachen/stairway_az{s:+d}.wav") for s in [-45,45]})


@lru_cache(maxsize=24)
def native_clip(pathstr):
    out = subprocess.run(["ffmpeg","-nostdin","-threads","1","-v","error","-i","pipe:0","-vn","-ac","2","-ar",str(SR),
                         "-f","f32le","pipe:1"],input=Path(pathstr).read_bytes(),capture_output=True,check=True)
    samples = np.frombuffer(out.stdout,dtype="<f4").reshape(-1,2)
    length = round(4.5 * SR)
    if len(samples) < length:
        raise ValueError("Insufficient native audio")
    start=(len(samples)-length)//2
    samples=samples[start:start+length].copy()
    samples*=.08 / max(1e-6, np.sqrt(np.mean(samples.astype('f8')**2)))
    return samples


def scene_iter(kind, manifest):
    if kind == "room":
        cohort=json.loads((R2 / "cohort.json").read_text())
        for row in (r for r in cohort if r["split"] == "test"):
            if digest(ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError("Changed source " + row["id"])
        for (audio, truth, meta), spec in zip(legacy_test.sources(cohort, "test", "room"), manifest["room"], strict=True):
            if meta != spec:
                raise RuntimeError(f"Scene drift: {meta} vs {spec}")
            yield audio, truth, meta
    elif kind == "native":
        lib={r["id"]:r for r in json.loads((ROOT / "round2/environment/native_inventory.json").read_text())}
        for row in manifest["native"]:
            stems=[]
            for key in ("source_id","background_id"):
                entry=lib[row[key]]
                stems.append(native_clip(str(ROOT / entry['path'])))
            mixed = stems[0]+stems[1]
            gain = min(1.,.8/(float(abs(mixed).max())+1e-9))
            yield mixed*gain, stems[0]*gain, row


def predict(baseline, module, model, audio, requests):
    if baseline == "r2":
        try:
            sources, mix, probs, _, _ = legacy_model.analyse(audio,SR,model)
        except ValueError:
            return {}, {}
        outputs = {}
        for name in requests:
            col = LABELS.index(name)
            idx = int(np.argmax(probs[:,col]))
            if probs[idx,col] >= float(model['threshold']):
                outputs[name] = sources[:,idx,None]*mix[:,idx]
        return outputs, {}
    if baseline == "apple":
        from voicehunter.separation import separate
        outputs = {}
        for name in requests:
            try:
                outputs[name] = separate(audio, SR, method="auto", target=name)[0]
            except ValueError:
                pass
        return outputs, {}
    outputs = module.predict_scene(audio,SR,model)
    if not isinstance(outputs, dict) or not set(requests).issubset(outputs):
        raise RuntimeError("A candidate must explicitly provide a stem or None for every requested class")
    return outputs, {}


def summary(data):
    errors=sum(r['error'] is not None for r in data)
    valid=[r for r in data if r["error"] is None]
    pos=[r for r in valid if r['present']]
    neg=[r for r in valid if not r['present']]
    tp=sum(r['returned'] for r in pos);fp=sum(r['returned'] for r in neg)
    fn=len(pos)-tp;tn=len(neg)-fp
    useful=sum(r['useful'] for r in data if r['present'])
    intended=sum(r['present'] for r in data)
    division=lambda a,b:a/b if b else None
    return dict(requests=len(data),errors=errors,status='complete' if not errors else 'incomplete',TP=tp,FP=fp,FN=fn,TN=tn,
                accuracy=division(tp+tn,len(data)) if not errors else None,
                precision=division(tp,tp+fp),recall=division(tp,tp+fn),
                F1=division(2*tp,2*tp+fp+fn) if not errors else None,
                useful=useful,useful_fraction=division(useful,intended), intended_positive=intended,
                harmful=sum(r.get('delta_db',0)<0 for r in pos if r['returned']),
                median_returned_delta_db=float(np.median([r['delta_db'] for r in pos if r['returned']])) if tp else None)


def run(args):
    manifest_path=HERE/'scenes.json'
    if args.freeze:
        if manifest_path.exists():raise FileExistsError("This stress suite is already frozen")
        manifest_path.write_text(json.dumps(create_manifest(),indent=2)+"\n")
        return
    manifest=json.loads(manifest_path.read_text())
    if manifest["semantic_scene_builder_sha256"]!=digest(R2/'evaluate.py'):
        raise RuntimeError("Round 2 scene builder changed")
    mod=None
    if args.method=="r2": model=legacy_model.load_model()
    elif args.method=="apple":
        from voicehunter.separation import auto_available
        if not auto_available():raise RuntimeError("Apple baseline unavailable")
        model=None
    else:
        mod=importlib.import_module(args.method)
        model=mod.load_model()
    inputs={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):digest(p) for p in args.model_file}
    if args.method not in ('r2','apple') and not inputs:
        raise ValueError("Specify frozen model file(s)")
    if args.method=='r2':
        inputs[str(legacy_model.WEIGHTS.relative_to(ROOT))]=digest(legacy_model.WEIGHTS)
        inputs[str(Path(legacy_model.__file__).relative_to(ROOT))]=digest(Path(legacy_model.__file__))
    result=dict(method=args.method, freeze_sha256=digest(manifest_path),protocol_sha256=digest(HERE/'PROTOCOL.md'),
                evaluator_sha256=digest(Path(__file__)),model_sha256=inputs, conditions={})
    for kind in (args.condition or ('room','native')):
        rows=[]
        started=time.monotonic()
        for index,(audio,truth,meta) in enumerate(scene_iter(kind,manifest)):
            base=legacy_test.sisdr(audio[WINDOW],truth[WINDOW])
            requests=(meta['target'],meta['absent'])
            try:
                outputs,errors=predict(args.method,mod,model,audio,requests)
            except Exception as error:
                outputs={};errors={k:f"{type(error).__name__}: {error}" for k in requests}
            for i,name in enumerate(requests):
                row=dict(**meta,requested=name,present=i==0,returned=False,useful=False,error=errors.get(name))
                stem=outputs.get(name)
                if stem is not None:
                    stem=np.asarray(stem)
                    if stem.shape != audio.shape or not np.issubdtype(stem.dtype,np.floating) or not np.isfinite(stem).all():
                        row['error']="Malformed, nonfinite, or misaligned output"
                    else:
                        row['returned']=True
                        if i==0:
                            row['delta_db']=legacy_test.sisdr(stem[WINDOW],truth[WINDOW])-base
                            row['useful']=row['delta_db']>=3
                rows.append(row)
        result['conditions'][kind]=dict(summary=summary(rows),by_requested_class={c:summary([r for r in rows if r['requested']==c]) for c in LABELS},
                                       wall_seconds=time.monotonic()-started,rows=rows)
        print(kind,result['conditions'][kind]['summary'],flush=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--method',default='r2')
    parser.add_argument('--model-file',type=Path,action='append',default=[])
    parser.add_argument('--condition', choices=('native','room'),action='append')
    parser.add_argument('--output',type=Path,default=HERE/'r2.json')
    run(parser.parse_args())
