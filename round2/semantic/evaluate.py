"""Fit and evaluate no-reference classification and extraction with uploader-held-out audio."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import semantic_candidate as candidate
from voicehunter.separation import separate, auto_available
from voicehunter.media import load_media

C=candidate.CLASSES
SR=48000
SECONDS=4.5
N=int(SR*SECONDS)


@lru_cache(maxsize=300)
def sound(path):
    for attempt in range(4):
        blob=(ROOT/path).read_bytes()
        out=subprocess.run(["ffmpeg","-nostdin","-v","error","-threads","1","-i","pipe:0","-t","10","-vn",
                            "-ac","1","-ar",str(SR),"-f","f32le","pipe:1"],input=blob,capture_output=True)
        if out.returncode==0 and len(out.stdout)>SR*4:
            return np.frombuffer(out.stdout,dtype='<f4')
        if attempt==3: raise RuntimeError(f"Input decoder failed for {path}: "+out.stderr.decode())
        time.sleep(.5)


def sources(rows,split,kind):
    r={c:sorted([x for x in rows if x['category']==c and x['split']==split],key=lambda x:int(x['id'])) for c in C}
    for g,c in enumerate(C):
        for i,target in enumerate(r[c]):
            other=C[(g+1+i%4)%len(C)]
            b=r[other][(i+g*2)%len(r[other])]
            absent=next(C[(g+k+i)%len(C)] for k in range(1,6) if C[(g+k+i)%len(C)] not in (c,other))
            voices=[sound(x['path']) for x in [target,b]]
            mono=[]
            for s in voices:
                if len(s)<N: raise ValueError("Scored sources must exceed 4.5 seconds.")
                start=(len(s)-N)//2
                signal=s[start:start+N].copy()
                signal*=.08/(np.sqrt(np.mean(signal.astype('f8')**2))+.000001)
                mono.append(signal)
            if kind=="intermittent":
                mono[0][int(1.1*SR):int(2.15*SR)]=0
                mono[1][:int(1.1*SR)]=0
            angles=[[12,62],[24,70],[17,81],[8,44],[38,76]][(g+i)%5]
            if (g+i)%2:angles=angles[::-1]
            if kind=="colocated":angles=[45,45]
            if kind=="room":
                irs=[room_ir(sign) for sign in ((-45,45) if (g+i)%2 else (45,-45))]
                truth=[]
                for s,ir in zip(mono,irs):
                    nfft=1<<(len(s)+len(ir)-2).bit_length()
                    truth.append(np.fft.irfft(np.fft.rfft(s,nfft)[:,None]*np.fft.rfft(ir,nfft,axis=0),axis=0)[:len(s)].astype('f4'))
            else:
                truth=[s[:,None]*np.array([np.cos(np.radians(a)),np.sin(np.radians(a))],dtype='f4') for s,a in zip(mono,angles)]
            scale=min(1.,.8/(np.max(np.abs(truth[0]+truth[1]))+1e-9))
            t,bg=[x*scale for x in truth]
            meta=dict(target=c,source_id=target['id'],distractor=other,background_id=b['id'],absent=absent,kind=kind)
            yield t+bg,t,meta


@lru_cache(maxsize=2)
def room_ir(sign):
    path=ROOT/f"round2_data/spatial/aachen/stairway_az{sign:+d}.wav"
    ir=load_media(path)[0][0:int(SR*.255)]
    return ir / np.sqrt(np.mean(np.sum(ir.astype('f8')**2,axis=0)))


def sisdr(est,ref):
    a=est.astype('f8');b=ref.astype('f8')
    a-=a.mean(axis=0);b-=b.mean(axis=0);v=np.sum(a*b)/(np.sum(b*b)+1e-12)*b
    return float(10*np.log10((np.sum(v*v)+1e-12)/(np.sum((a-v)**2)+1e-12)))


def merge_metrics(rows):
    pos=[r for r in rows if r['present']]
    neg=[r for r in rows if not r['present']]
    returned=[r for r in pos if r['returned']]
    return dict(intended=len(pos),returned=len(returned),useful=sum(bool(r.get('useful',False)) for r in returned),
                harmful=sum(r.get('delta',0)<0 for r in returned),negative=len(neg),
                false_accept=sum(r['returned'] for r in neg),
                median_returned_delta=float(np.median([r['delta'] for r in returned])) if returned else None)


def gather(cohort,split,kind,model=None,apple=False,resume=False):
    checkpoint=HERE/("apple_"+kind+"_checkpoint.jsonl")
    if apple and not resume:
        checkpoint.write_text("")
    rows=[json.loads(line) for line in checkpoint.read_text().splitlines()] if apple and checkpoint.exists() else []
    finished={(r['source_id'],r['requested']) for r in rows}
    def record(row):
        rows.append(row)
        if apple:
            with checkpoint.open("a") as fh:
                fh.write(json.dumps(row)+"\n")
    for x,t,meta in sources(cohort,split,kind):
        sl=slice(int(SR*(2.45 if kind=="intermittent" else .15)),int(SR*4.35))
        base=sisdr(x[sl],t[sl]);stem_meta=None;why=None
        if not apple:
            try:stem_meta=candidate.analyse(x,SR,model)
            except ValueError as e:why=str(e)
        requests={"target":meta['target'],"absent":meta['absent']}
        for role,label in requests.items():
            if (meta['source_id'],label) in finished:continue
            row=dict(**meta,present=role=="target",requested=label,returned=False)
            if apple:
                try:
                    s,_,_=separate(x,SR,method="auto",target=label)
                except (RuntimeError,ValueError) as e:
                    row["refusal"]=str(e);record(row);continue
            else:
                if stem_meta is None:
                    row["refusal"]=why;record(row);continue
                strokes,mix,prob,angles,focus=stem_meta
                k=C.index(label);idx=int(np.argmax(prob[:,k]));score=float(prob[idx,k])
                row.update(score=score,angles=angles,focus=focus)
                s=strokes[:,idx,None]*mix[:,idx]
            row['returned']=True
            if row['present']:
                row['delta']=sisdr(s[sl],t[sl])-base
                row['useful']=row['delta']>=3
            record(row)
    return rows


def gate(rows,threshold):
    output=[]
    for row in rows:
        new=row.copy()
        if row.get('returned',False) and row['score']<threshold:
            new['returned']=False;new['refusal']='trained confidence threshold'
        output.append(new)
    return output


def protocol(rows):
    classes_by_split={s:{r['category'] for r in rows if r["split"]==s} for s in ("train","validation","test")}
    assert all(v==set(C) for v in classes_by_split.values())
    train=[r for r in rows if r["split"]=="train"]
    groups={s:{r['author'].strip().casefold() for r in rows if r['split']==s} for s in classes_by_split}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not groups[a]&groups[b]
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    return train


def fit(rows):
    training=protocol(rows)
    def features(r):
        x=sound(r['path'])
        return candidate.feature_batch(x)
    with ThreadPoolExecutor(max_workers=4) as executor:
        data=list(executor.map(features,training))
    labels=[C.index(r['category']) for r,parts in zip(training,data) for _ in parts]
    examples=np.stack([f for parts in data for f in parts])
    model=candidate.fit_descriptors(examples,labels)
    val={s:gather(rows,"validation",s,model=model) for s in ("persistent","intermittent")}
    # Fixed grid, validation only; maximize useful positives subject to false accepts.
    grid=[]
    for x in np.linspace(.10,.90,81):
        allrows=[w for v in val.values() for w in gate(v,x)]
        m=merge_metrics(allrows)
        grid.append(dict(threshold=float(x),**m))
    ok=[r for r in grid if r['false_accept']<=.15*r['negative']]
    chosen=max(ok,key=lambda r:(r['useful'],-r['false_accept'],-r['threshold']))
    model['threshold']=chosen['threshold']
    np.savez_compressed(HERE/'semantic_profiles.npz',**model)
    (HERE/"fit.json").write_text(json.dumps(dict(training_clips=len(training),local_descriptors=len(labels),validation=chosen,
               grid=grid,cohort_sha256=hashlib.sha256((HERE/'cohort.json').read_bytes()).hexdigest()),indent=2)+"\n")
    print('FIT',chosen,flush=True)
    return model


def run_evaluation(rows,model,apple=False,resume=False):
    if apple and not auto_available():raise RuntimeError("Apple SoundAnalysis unavailable")
    reports={}
    kinds=("persistent","intermittent") if apple else ("persistent","intermittent","colocated","room")
    for kind in kinds:
        out=gather(rows,"test",kind,model,apple,resume)
        if not apple:out=gate(out,float(model['threshold']))
        reports[kind]=dict(summary=merge_metrics(out),by_target={c:merge_metrics([r for r in out if r['target']==c]) for c in C},rows=out)
        print(kind,reports[kind]['summary'],flush=True)
        (HERE/("apple_results.json" if apple else "test_results.json")).write_text(json.dumps(reports,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--apple",action="store_true");p.add_argument("--evaluate-only",action="store_true")
    p.add_argument("--resume-apple",action="store_true",help="Resume a previously interrupted Apple baseline run.")
    args=p.parse_args();cohort=json.loads((HERE/"cohort.json").read_text())
    if args.apple:run_evaluation(cohort,None,True,args.resume_apple)
    else:
        model=dict(np.load(HERE/"semantic_profiles.npz")) if args.evaluate_only else fit(cohort)
        run_evaluation(cohort,model)
