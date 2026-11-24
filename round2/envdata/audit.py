"""Unify the collected sources; audit rights, source groups, bytes and stereo information."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from collect import ROOT,HERE,split

def h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def measure(row):
    p=ROOT/row['path']
    if h(p)!=row['sha256']: raise ValueError(str(p))
    pcm=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(p),'-ar','12000','-f','f32le','pipe:1'])
    a=np.frombuffer(pcm,'<f4').reshape(-1,row['channels']).astype(np.float64)
    if row['channels']==2:
        power=float(np.mean(a*a))
        separation=float(np.mean(((a[:,0]-a[:,1])/2)**2))
        ratio_db=10*np.log10(max(separation,1e-30)/max(power,1e-30))
        with np.errstate(invalid='ignore',divide='ignore'):
            corr=float(np.corrcoef(a.T)[0,1])
        row['channel_check_12khz']={'side_fraction_power_db':round(float(ratio_db),3),'lr_correlation':round(corr,6) if np.isfinite(corr) else None}
        row['signal_layout']='two-channel with measured differences' if ratio_db>-50 else 'near-dual-mono'
    else: row['signal_layout']='mono'
    mono=np.mean(a,axis=1)
    vectors=[]
    # Five independent short blocks from the beginning. A near-identical waveform
    # in all five is a conservative copy flag. Avoid silence-only matches.
    for at in (.25,1.0,1.75,2.5,3.25):
        x=mono[int(at*12000):int(at*12000)+6000:6]
        x=x-np.mean(x)
        vectors.append(x/(np.linalg.norm(x)+1e-9))
    return np.array(vectors,dtype='f4')

def main():
    primary=json.loads((HERE/'manifest.json').read_text())
    stereo=json.loads((HERE/'stereo_candidates.json').read_text())
    # Prefer the recording with original stereo cues, still count one origin.
    rows={r['id']:r for r in primary}
    rows.update({r['id']:r for r in stereo})
    canonical=sorted(rows.values(),key=lambda r:(r['category'], r['split'], int(r['id'])))
    hashes=set(); authors={}; vectors=[]; unique=[]; excluded=[]
    for r in canonical:
        if r['sha256'] in hashes:
            excluded.append(dict(id=r['id'],reason='identical file hash'));continue
        if r['split']!=split(r['author']):raise ValueError(r)
        a=r['author'].strip().casefold()
        if a in authors and authors[a]!=r['split']:raise ValueError('split leakage')
        authors[a]=r['split']; hashes.add(r['sha256'])
        vectors.append(measure(r));unique.append(r)
    m=np.stack(vectors)
    similar=np.ones((len(m),len(m)),dtype=bool)
    for j in range(m.shape[1]):
        x=m[:,j,:]; similar &= abs(x@x.T)>.997
    blocked=set()
    for i,j in zip(*np.triu(similar,1).nonzero()):
        r,k=unique[i],unique[j]
        if r['split']!=k['split']:
            blocked.add(k['id']);excluded.append(dict(id=k['id'],compared_to=r['id'],reason='cross-split aligned audio fingerprint'))
    kept=[r for r in unique if r['id'] not in blocked]
    (HERE/'all_sources.json').write_text(json.dumps(kept,indent=1,ensure_ascii=False)+'\n')
    with (HERE/'attribution.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['id','title','author','url','license','representation','category','split','path','sha256']
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(kept)
    def count(k):return dict(Counter(r[k] for r in kept))
    s=dict(sources=len(kept),unique_freesound_ids=len(set(r['id'] for r in kept)),authors=len(set(r['author'].strip().casefold() for r in kept)),authorgroups_per_split={k:len(set(r['author'].casefold() for r in kept if r['split']==k)) for k in ('train','validation','test')},
        seconds=round(sum(r['duration'] for r in kept),2),bytes=sum(r['bytes'] for r in kept),split=count('split'),license=count('license'),representation=count('representation'),signal_layout=count('signal_layout'),category=count('category'),stereo_by_category=dict(Counter(r['category'] for r in kept if r['signal_layout']=='two-channel with measured differences')),category_splits={k:dict(Counter(r['split'] for r in kept if r['category']==k)) for k in sorted(set(r['category'] for r in kept))},duplicate_ids_removed=len(primary)+len(stereo)-len(canonical),exclusions=excluded,
        metadata_sha256={p.name:h(p) for p in sorted((ROOT/'round2_data/environment/metadata').glob('*')) if p.is_file()},attribution='attribution.csv')
    (HERE/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(s,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
