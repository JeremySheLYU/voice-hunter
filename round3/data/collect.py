"""Fetch author-disjoint CC0/BY FSD50K examples on a predeclared priority list."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from math import isfinite
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from round2.envdata.collect import candidates, HF

ROLES=('test','train')
CATEGORIES=('guitar','piano','speech','train','wind')
QUOTAS={'train':90,'test':20}
DATA=ROOT/'round3_data/environment'
HISTORY=['round2/envdata/all_sources.json','round2/environment/native_inventory.json','round2/semantic/cohort.json']
def digest(b):
    return hashlib.sha256(b).hexdigest()
def group(r):
    return r['author'].strip().casefold()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.part')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=1)+'\n')
    tmp.replace(p)
def frozen():
    if (HERE/'plan.json').exists():
        return json.loads((HERE/'plan.json').read_text())
    seen=[x for f in HISTORY for x in json.loads((ROOT/f).read_text())]
    ids={x['id'] for x in seen}
    authors={group(x) for x in seen}
    buckets=defaultdict(list)
    for r in candidates():
        author=group(r)
        if author in authors or r['id'] in ids: continue
        role='test' if int(digest(('VH3-new-uploader:'+author).encode())[:12],16)%100<30 else 'train'
        record=dict(r,split=role)
        record['download_url']=f"{HF}/clips/{r['collection_partition']}/{r['id']}.wav"
        buckets[role,r['category']].append(record)
    plan=[]
    for role in ROLES:
        for c in CATEGORIES:
            seq=sorted(buckets[role,c],key=lambda r:(r['priority'],digest(('VH3-order:'+r['id']).encode())))
            rank=Counter()
            for r in seq:
                a=group(r)
                r['author_rank']=rank[a]
                rank[a]+=1
            plan.extend(sorted(seq,key=lambda r:(r['author_rank'],r['priority'],digest(('VH3-order:'+r['id']).encode()))))
    write(HERE/'plan.json',plan)
    write(HERE/'freeze.json',dict(sha256=digest((HERE/'plan.json').read_bytes()),
          quotas=QUOTAS,minimum_seconds={'test':4.9,'train':2.0},excluded_historic=HISTORY,
          author_partition='SHA256(VH3-new-uploader: + stripped casefolded uploader) first 12 hex mod 100; below 30=test; else train',
          order='Within each class/role prioritize diverse uploaders, few extraneous labels, prioritized matching descriptions, SHA tie breaker. Select in frozen order, skip invalid/unavailable/duplicate bytes.',
          population='Public mono FSD50K standardized audio with exactly one of five target class groups; inherited provenance/selection in round2/envdata/collect.py. Official weak class labels.'))
    return plan
def fetch(r):
    p=DATA/r['category']/(r['id']+'.wav')
    byte_limit=4_500_000
    for attempt in range(2):
        try:
            if p.is_file(): blob=p.read_bytes()
            else:
                req=urllib.request.Request(r['download_url'],headers={'User-Agent':'VoiceHunter-public-research/0.3'})
                with urllib.request.urlopen(req,timeout=18) as resp:
                    blob=resp.read(byte_limit+1)
            if len(blob)>byte_limit:return None,'size ceiling'
            try:
                with wave.open(io.BytesIO(blob),'rb') as w:
                    ch=w.getnchannels(); rate=w.getframerate(); n=w.getnframes(); width=w.getsampwidth(); pcm=w.readframes(n)
                if ch!=1 or rate<=0 or n==0 or len(pcm)!=n*width*ch:return None,'invalid PCM, channel count or length'
                minimum=4.9 if r['split']=='test' else 2.0
                if n<minimum*rate:return None,'below minimum duration'
            except (EOFError,wave.Error):return None,'unreadable waveform'
            r={k:v for k,v in r.items() if k not in ('priority','author_rank')}
            r.update(path=p.relative_to(ROOT).as_posix(),duration=n/rate,channels=ch,sample_rate=rate,
                frames=n,bytes=len(blob),sha256=digest(blob),pcm_sha256=digest(pcm),corpus='FSD50K',
                representation='standardized mono WAV',license_metadata='FSD50K.metadata.zip at Zenodo 4060432')
            if not p.is_file():
                p.parent.mkdir(parents=True,exist_ok=True)
                tmp=p.with_suffix('.part');tmp.write_bytes(blob);tmp.replace(p)
            return r,None
        except (OSError,ValueError) as e:
            if attempt==1: return None,str(e)[:180]
            time.sleep(.4)
def run():
    plan=frozen()
    old=[r for name in HISTORY for r in json.loads((ROOT/name).read_text())]
    byte_hash={r['sha256'] for r in old}
    pcm_hash=set()
    for row in old:
        p=ROOT/row['path']
        if p.suffix=='.wav' and p.is_file():
            with wave.open(str(p),'rb') as w:
                pcm_hash.add(digest(w.readframes(w.getnframes())))
    stats={}
    for role in ROLES:
        buckets={c:[r for r in plan if r['split']==role and r['category']==c] for c in CATEGORIES}
        cursors=Counter()
        counts=Counter()
        why=Counter()
        out=[]
        with ThreadPoolExecutor(max_workers=28) as pool:
            while True:
                batch=[]
                for c in CATEGORIES:
                    missing=QUOTAS[role]-counts[c]
                    if missing>0:
                        sample=buckets[c][cursors[c]:cursors[c]+min(18,missing+1)]
                        cursors[c]+=len(sample)
                        batch+=sample
                if not batch:break
                futures=[pool.submit(fetch,r) for r in batch]
                for r,job in zip(batch,futures):
                    value,err=job.result()
                    if value is None:
                        why[err]+=1;continue
                    if counts[r['category']]>=QUOTAS[role]:
                        (ROOT/value['path']).unlink(missing_ok=True)
                        continue
                    if value['sha256'] in byte_hash or value['pcm_sha256'] in pcm_hash:
                        why['duplicate waveform']+=1
                        (ROOT/value['path']).unlink(missing_ok=True);continue
                    byte_hash.add(value['sha256']);pcm_hash.add(value['pcm_sha256']);out.append(value);counts[r['category']]+=1
                write(HERE/(role+'.partial.json'),out)
                print(role,dict(counts),'examined',sum(cursors.values()),flush=True)
        write(HERE/(role+'.json'),out)
        stats[role]=dict(n=len(out),counts=counts,examined=sum(cursors.values()),rejected=why,uploaders=len({group(r) for r in out}),seconds=sum(r['duration'] for r in out))
        write(HERE/'summary.json',stats)
        (HERE/(role+'.ready')).write_text(digest((HERE/(role+'.json')).read_bytes())+'\n')
        print('FINISHED',role,stats[role],flush=True)
if __name__=='__main__':run()
