"""Frozen-label five-class stress with genuinely new unpanned stereo source captures."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from evaluate_stress import HERE, LABELS, ROOT, SR, WINDOW, digest, legacy_test, native_clip, stable_rank, summary
from round3.evaluation.evaluate import predictor

DATA=ROOT/'round3/data'
PLAN=HERE/'fresh_native_scenes.json'


def author(row):
    return row["author"].strip().casefold()


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    partial=path.with_suffix('.partial')
    partial.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    partial.replace(path)


def freeze():
    candidates=read(DATA/'native.json')
    development=[r for r in read(ROOT/'round2/semantic/cohort.json') if r['split']!='test']+read(DATA/'train.json')
    reserved=[r for r in read(DATA/'plan.json') if r['split']=='train']
    fresh_test=read(DATA/'test.json')
    previous=read(ROOT/'round2/envdata/manifest.json')+read(ROOT/'round2/environment/native_inventory.json')
    seen=set()
    bad_ids={str(r['id']) for r in development+reserved}
    bad_authors={author(r) for r in development+reserved}
    bad_hash={r['sha256'] for r in development}
    allowed=[]
    excluded=[]
    for r in sorted(candidates,key=lambda x:int(x['id'])):
        id_=str(r['id'])
        if id_ in seen or id_ in bad_ids or author(r) in bad_authors or r['sha256'] in bad_hash:
            raise ValueError('Training overlap or duplicate source '+id_)
        seen.add(id_)
        path=ROOT/r['path']
        if digest(path)!=r['sha256']:
            raise ValueError('Native recording checksum mismatch: '+id_)
        if r['channels']!=2 or r['duration']<4.5:
            excluded.append([id_,'header does not meet two-channel/length criteria']); continue
        audio=native_clip(str(path))
        signal=np.mean(audio.astype('f8')**2)
        side=np.mean(((audio[:,0].astype('f8')-audio[:,1])*.5)**2)
        # Native clip loader applies an RMS normalizer floored at 1e-6. A
        # separate unscaled centered RMS is obtained with FFmpeg rather than
        # inferring original level from the normalized array.
        import subprocess
        decoded=subprocess.run(['ffmpeg','-nostdin','-threads','1','-v','error','-i','pipe:0','-vn','-ac','2','-ar',str(SR),'-f','f32le','pipe:1'],
                               input=path.read_bytes(),capture_output=True,check=True)
        raw=np.frombuffer(decoded.stdout,dtype='<f4').reshape(-1,2)
        n=len(audio); start=(len(raw)-n)//2
        rms=float(np.sqrt(np.mean(raw[start:start+n].astype("f8")**2)))
        ratio=float(10*np.log10(max(side,1e-15)/max(signal,1e-15)))
        if rms<=1e-5 or ratio<=-50:
            excluded.append([id_,'quiet or near-duplicate channels',rms,ratio]);continue
        allowed.append({**r,'id':id_,'native_midwindow_rms':rms,'native_side_db':ratio})
    assigned=Counter();absents=Counter();scenes=[]
    for r in allowed:
        for other in LABELS:
            if r['category']==other:continue
            opt=[b for b in allowed if b['category']==other and author(r)!=author(b)]
            if not opt:continue
            b=min(opt,key=lambda b:(assigned[b['id']],stable_rank([r['id'],b['id']])))
            absent=min([c for c in LABELS if c not in (r['category'],other)],
                       key=lambda c:(absents[c],stable_rank([r['id'],b['id'],c])))
            assigned[b['id']]+=1; absents[absent]+=1
            scenes.append(dict(source_id=r['id'],background_id=b['id'],target=r['category'],distractor=other,absent=absent,kind='fresh_native'))
    overlap=lambda aa,bb,key:sorted({key(r) for r in aa}.intersection(key(r) for r in bb))
    if not scenes: raise ValueError("No eligible source pair")
    data=dict(sources=allowed,excluded=excluded,scenes=scenes,source_sha256=digest(DATA/'native.json'),
              protocol_sha256=digest(HERE/'FRESH_NATIVE_PROTOCOL.md'), prior_protocol_sha256=digest(HERE/'PROTOCOL.md'),
              unique_pairs=len({tuple(sorted([s['source_id'],s['background_id']])) for s in scenes}),
              uploaders=len({author(r) for r in allowed}), source_category_counts=dict(Counter(r['category'] for r in allowed)),
              fresh_test_shared_ids=overlap(allowed,fresh_test,lambda r:str(r['id'])),fresh_test_shared_authors=overlap(allowed,fresh_test,author),
              old_shared_ids=overlap(allowed,previous,lambda r:str(r['id'])),old_shared_authors=overlap(allowed,previous,author))
    if PLAN.exists() and data!=read(PLAN):raise FileExistsError('Fresh native plan changed after freeze')
    save(PLAN,data)
    print({k:v for k,v in data.items() if k not in ('scenes','sources')},flush=True)


def evaluate(args):
    plan=read(PLAN)
    if digest(HERE/'FRESH_NATIVE_PROTOCOL.md')!=plan['protocol_sha256'] or digest(DATA/'native.json')!=plan['source_sha256']:
        raise RuntimeError("Frozen inputs have changed")
    sources={r['id']:r for r in plan['sources']}
    get=predictor(args.model)
    rows=[]
    for s in plan['scenes']:
        a,b=[native_clip(str(ROOT/sources[s[k]]['path'])) for k in ('source_id','background_id')]
        mix=a+b; g=min(1.,.8/(float(abs(mix).max())+1e-9))
        audio,truth=mix*g,a*g
        ref=legacy_test.sisdr(audio[WINDOW],truth[WINDOW])
        prompts=(s['target'],s['absent'])
        out=get(audio,prompts)
        for i,label in enumerate(prompts):
            wave=out[label]
            rec=dict(**s,present=i==0,requested=label,returned=wave is not None,useful=False,error=None)
            if wave is not None and (wave.shape!=audio.shape or not np.isfinite(wave).all()):
                raise ValueError('Invalid stem')
            if wave is not None and rec['present']:
                rec['delta_db']=legacy_test.sisdr(wave[WINDOW],truth[WINDOW])-ref
                rec['useful']=bool(rec['delta_db']>=3)
            rows.append(rec)
    result=dict(model=args.model,plan_sha256=digest(PLAN),evaluator_sha256=digest(Path(__file__)),
                model_sha256={str(p):digest(p) for p in args.model_file},
                conditions={'fresh_native':dict(summary=summary(rows),by_requested_class={c:summary([r for r in rows if r['requested']==c]) for c in LABELS},rows=rows)})
    save(HERE/f'fresh_native_{args.model}.json',result)
    print(args.model,result['conditions']['fresh_native']['summary'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--model',default='v02')
    p.add_argument('--model-file',type=Path,action='append',default=[])
    a=p.parse_args()
    if a.freeze:freeze()
    else:evaluate(a)
