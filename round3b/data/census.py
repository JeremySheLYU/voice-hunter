"""Freeze the still-unused author census for a strict FSD50K-only follow-up."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,urllib.request,wave
from pathlib import Path
from itertools import zip_longest
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def load(f):return json.loads((ROOT/f).read_text())
def save(name,x):
 p=HERE/name;t=p.with_suffix('.part');t.write_text(json.dumps(x,indent=1,ensure_ascii=False)+'\n');t.replace(p)
def author(r):return r['author'].strip().casefold()
def snapshot():
 files=['round2/envdata/all_sources.json','round2/envdata/manifest.json','round2/envdata/stereo_candidates.json','round2/environment/native_inventory.json','round2/environment/cohort.json','round2/semantic/cohort.json','round3/data/train.json','round3/data/test.json','round3/data/native.json','round3/unknowns/manifest.json','round3/transformer/train_sources.json','round3/transfer/train_sources.json']
 hist=[r for f in files for r in load(f) if isinstance(r,dict) and 'author' in r]
 candidates=load('round3/data/plan.json')
 ids={r['id'] for r in hist};authors={author(r) for r in hist}
 rows=sorted([r for r in candidates if author(r) not in authors and r['id'] not in ids],key=lambda r:sha(('VH3B-primary-blind:'+r['id']).encode()))
 plan=HERE/'plan.json'
 if plan.exists():assert json.loads(plan.read_text())==rows
 else:save('plan.json',rows)
 save('freeze.json',{'plan_sha256':sha(plan.read_bytes()),'locked_exclusion_files':files,'minimum_seconds':4.5,'minimum_unique_sources_per_class':12,'minimum_unique_uploaders_per_class':12,'priority':'SHA256(VH3B-primary-blind: + source ID), candidate order per class. Frozen FSD50K only. Must count actual duration on mono standardized PCM; CC BY/CC0 and single target class checked via original corpus metadata. Do not pool external sources.','historical_authors':len(authors),'historical_ids':len(ids)})
 return rows
def header(r):
 req=urllib.request.Request(r['download_url'],headers={'Range':'bytes=0-1023','User-Agent':'VoiceHunter-audio-research/0.3b'})
 for attempt in range(2):
  try:
   with urllib.request.urlopen(req,timeout=12) as resp:blob=resp.read(1024)
   with wave.open(io.BytesIO(blob),'rb') as w:
    dur=w.getnframes()/w.getframerate();ch=w.getnchannels();rate=w.getframerate()
   return dict(id=r['id'],category=r['category'],author=r['author'],duration=dur,channels=ch,sample_rate=rate,eligible=(dur>=4.5 and ch==1))
  except (OSError,EOFError,wave.Error) as e:
   err=str(e)[:120]
 return dict(id=r['id'],category=r['category'],author=r['author'],error=err,eligible=False)
def main():
 rows=snapshot()
 sets={c:[r for r in rows if r['category']==c] for c in ['train','guitar','piano','wind','speech']}
 print('upper bounds', {c:(len(v),len({author(r) for r in v})) for c,v in sets.items()},flush=True)
 order=[r for group in zip_longest(*sets.values()) for r in group if r]
 with ThreadPoolExecutor(max_workers=32) as pool:
  read=list(pool.map(header,order))
 save('durations.json',read)
 summary={}
 for c,s in sets.items():
  good=[r for r in read if r['category']==c and r['eligible']]
  summary[c]=dict(candidates=len(s),candidate_uploaders=len({author(r) for r in s}),eligible=len(good),eligible_uploaders=len({author(r) for r in good}),retrieval_errors=sum('error' in r for r in read if r['category']==c))
 save('census.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
