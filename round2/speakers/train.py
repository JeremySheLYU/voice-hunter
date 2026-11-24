"""Fit speech similarity on OpenSLR 31 training speakers, using only NumPy/FFmpeg."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from voicehunter.speaker_identity import frames, summarize

ORIGIN="https://www.openslr.org/resources/31/"
ASSETS=["train-clean-5","dev-clean-2"]


def sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def prepare(cache):
    import urllib.request
    corpus=[]
    for split in ASSETS:
        f=cache/(split+'.tar.gz')
        cache.mkdir(parents=True,exist_ok=True)
        if not f.exists():
            urllib.request.urlretrieve(ORIGIN+f.name,f.with_suffix('.part'))
            f.with_suffix('.part').replace(f)
        target=cache/'LibriSpeech'/split
        if not (target/'.complete').exists():
            with tarfile.open(f) as archive:
                archive.extractall(cache,filter='data',members=(m for m in archive if m.name.startswith('LibriSpeech/'+split+'/')))
            (target/'.complete').touch()
        files=sorted(target.rglob('*.flac'))
        if not files: raise ValueError('Empty corpus')
        corpus.append((target,files,{'name':f.name,'url':ORIGIN+f.name,'bytes':f.stat().st_size,
                                    'sha256':sha(f),'clips':len(files),
                                    'readers':sorted(set(p.stem.split('-')[0] for p in files))}))
    assert not set(corpus[0][2]['readers']) & set(corpus[1][2]['readers'])
    return corpus


def decode(path):
    out=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-ar','8000','-ac','1','-f','f32le','pipe:1'])
    return np.frombuffer(out,dtype='<f4')


def collect_file(path):
    signal=decode(path)
    # Select evenly spaced 1.0 s windows over each utterance, including
    # later content. Six excerpts limit runtime and oversampling of long files.
    starts=np.unique(np.linspace(0,max(0,len(signal)-8000),min(6,max(1,len(signal)//8000)),dtype=int))
    result=[]
    for j in starts:
        f=summarize(frames(signal[j:j+8000],8000))
        if f is not None:result.append(f)
    return result,len(signal)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,required=True)
    args=ap.parse_args()
    corpus=prepare(args.cache)
    files=corpus[0][1]
    x,labels,clip_ids=[],[],[]
    samples=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(p,(features,n)) in enumerate(zip(files,pool.map(collect_file,files))):
            x.extend(features);labels.extend([p.stem.split('-')[0]]*len(features));clip_ids.extend([i]*len(features));samples+=n
    x=np.vstack(x);labels=np.array(labels);clip_ids=np.array(clip_ids)
    rng=np.random.default_rng(20261121)
    scale=np.maximum(np.std(x,axis=0),.04)
    xx=x/scale
    by={v:np.flatnonzero(labels==v) for v in np.unique(labels)}
    rows=[]
    for reader,ix in by.items():
        # Each reader has equal weight in the training objective.
        away=np.flatnonzero(labels!=reader)
        anchor=rng.choice(ix,2000)
        same=rng.choice(ix,2000)
        diff=rng.choice(away,2000)
        ok=clip_ids[anchor]!=clip_ids[same]
        base=xx[anchor[ok]]
        rows.append(abs(base-xx[diff[ok]])-abs(base-xx[same[ok]]))
    contrasts=np.concatenate(rows)
    weights=np.zeros(x.shape[1])
    for epoch in range(70):
        z=np.clip(contrasts@weights,-40,40)
        grad=-(contrasts.T@(1/(1+np.exp(z))))/len(z)+.018*weights
        weights=np.maximum(0.,weights-.35*grad)
    weights=weights/scale
    model={"schema":1,"method":"Supervised nonnegative logit distance over 30 voice features",
           "training":"Mini LibriSpeech train-clean-5, OpenSLR 31. CC BY 4.0; Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. https://www.openslr.org/31/",
           "training_speakers":len(by),"train_segments":len(x),"absolute_feature_weights":weights.round(8).tolist()}
    model_path=ROOT/'voicehunter/models/speaker-distance.json'
    model_path.write_text(json.dumps(model,indent=2)+'\n')
    provenance=dict(assets=[entry[2] for entry in corpus],license="CC BY 4.0",rights="https://creativecommons.org/licenses/by/4.0/",source="https://www.openslr.org/31/",
                    modifications="Decoded mono excerpts and derived supervised metric coefficients; raw recordings excluded from product distribution.",
                    input_audio_seconds=samples/8000,feature_segments=len(x),triplets=len(contrasts),
                    used_readers=sorted(by),samples_per_reader=dict(Counter(labels)),model_sha256=sha(model_path))
    (HERE/'training.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:provenance[k] for k in ['input_audio_seconds','feature_segments','triplets','model_sha256']}),flush=True)


if __name__=='__main__':
    main()
