"""Fit the transfer head using original train and optional new training-only rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'round2/semantic'))
from round2.semantic.evaluate import sources, sound, sisdr
from round3.transfer.candidate import CLASSES, RATE, Encoder, fit_head, select_stem, stem_logits, candidates


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def representation(row, encoder, cache):
    path = ROOT / row['path']
    if not path.is_file() or digest(path) != row['sha256']:
        raise ValueError(f"Source hash mismatch in training: {path}")
    destination = cache / f"train-{row['sha256']}.npz"
    if destination.is_file():
        with np.load(destination) as d:
            return d['vectors']
    full = sound(row['path'])
    width = min(len(full), int(4.5 * RATE))
    # Different crops increase feature variation without overweighting long sources.
    starts = np.unique(np.linspace(0, len(full) - width, 2, dtype=int))
    vectors = [encoder.encode(full[start:start+width])[0] for start in starts]
    np.savez_compressed(destination, vectors=vectors)
    return vectors


def validation_features(rows, encoder, cache):
    output = []
    for kind in ('persistent', 'intermittent'):
        for audio, target, meta in sources(rows, 'validation', kind):
            path = cache / f"val-{kind}-{meta['source_id']}-{meta['background_id']}.npz"
            if path.is_file():
                with np.load(path) as d:
                    out = dict(d)
            else:
                try:
                    mono, matrix, _, _ = candidates(audio)
                    feat = [encoder.encode(s) for s in mono.T]
                    vectors, priors = [np.stack(v) for v in zip(*feat)]
                    interval = slice(int(RATE*(2.45 if kind == 'intermittent' else .15)),int(RATE*4.35))
                    baseline = sisdr(audio[interval], target[interval])
                    gain = [sisdr(mono[interval,i,None]*matrix[:,i],target[interval])-baseline for i in range(2)]
                except ValueError:
                    vectors = np.zeros((0,1024)); priors=np.zeros((0,5)); gain=[]
                out = dict(vectors=vectors, priors=priors, gain=np.asarray(gain))
                np.savez_compressed(path, **out)
            output.append((meta, out))
    return output


def measure(output, head, prior_weight, threshold):
    counts = {kind: dict(tp=0, fp=0, fn=0, tn=0, useful=0, positive=0, negative=0) for kind in ('persistent','intermittent')}
    for meta, values in output:
        row = counts[meta['kind']]
        logits = stem_logits(values['vectors'], values['priors'], head, prior_weight) if len(values['gain']) else None
        for role, label in (('positive', meta['target']),('negative',meta['absent'])):
            row[role] += 1
            index = None if logits is None else select_stem(logits, label, threshold)
            if role == 'negative':
                row['tn' if index is None else 'fp'] += 1
            else:
                row['fn' if index is None else 'tp'] += 1
                if index is not None and values['gain'][index] >= 3:
                    row['useful'] += 1
    for v in counts.values():
        v['f1'] = 2*v['tp']/max(1,2*v['tp']+v['fn']+v['fp'])
        v['accuracy'] = (v['tp']+v['tn'])/(v['positive']+v['negative'])
        v['useful_fraction'] = v['useful']/v['positive']
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extra-train', type=Path, help='JSON list with id/category/author/path/sha256; only rows marked train are used')
    p.add_argument('--cache', type=Path, default=Path('/tmp/voicehunter-round3-transfer-features'))
    args=p.parse_args()
    args.cache.mkdir(exist_ok=True, parents=True)
    cohort_path=ROOT/'round2/semantic/cohort.json'
    original=[r for r in read_json(cohort_path) if r['split'] in ('train','validation')]
    train=[r for r in original if r['split']=='train']
    if args.extra_train:
        extra=[r for r in read_json(args.extra_train) if r['split']=='train']
        val_authors={r['author'].strip().casefold() for r in original if r['split']=='validation'}
        if any(r['author'].strip().casefold() in val_authors for r in extra):
            raise ValueError('Additional training sources must not share original validation uploaders.')
        train.extend(extra)
    unique_rows={r['sha256']:r for r in train}
    if len(train)!=len(unique_rows):
        raise ValueError('Repeated training waveforms.')
    encoder=Encoder()
    x=[]; y=[]
    for r in train:
        vectors=representation(r,encoder,args.cache)
        x.extend(vectors); y.extend([CLASSES.index(r['category'])]*len(vectors))
    head=fit_head(x,np.asarray(y))
    val=validation_features(original,encoder,args.cache)
    grid=[]
    for prior_weight in (0., .30, .75, 1.50):
        for threshold in (.3,.4,.5,.6,.7):
            metrics=measure(val,head,prior_weight,threshold)
            grid.append(dict(prior_weight=prior_weight,threshold=threshold,metrics=metrics))
    def rank(candidate):
        primary=min(v[stat] for v in candidate['metrics'].values() for stat in ('f1','accuracy'))
        return (primary, sum(v['useful_fraction'] for v in candidate['metrics'].values()),
                -sum(v['fp'] for v in candidate['metrics'].values()),-candidate['prior_weight'])
    chosen=max(grid,key=rank)
    head.update(threshold=chosen['threshold'],prior_weight=chosen['prior_weight'])
    np.savez_compressed(HERE/'head.npz',**head)
    manifest=[{k:r[k] for k in ('category','id','author','path','sha256')} for r in train]
    (HERE/'train_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1)+'\n')
    report=dict(protocol='fit on training only; choose fixed grid by worst of presence Accuracy and F1 across two validation conditions, followed by useful fraction, fewer false accepts, smaller prior', count=len(train), by_class=dict(Counter(r['category'] for r in train)),train_feature_vectors=len(x), encoder_sha256=digest(HERE/'yamnet.onnx'), chosen=chosen, grid=grid)
    (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'model':'transfer','ntrain':len(train),**chosen}),flush=True)


if __name__=='__main__':
    main()
