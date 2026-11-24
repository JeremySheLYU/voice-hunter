"""Bind residual eligibility to independently fetched official Git LFS object digests."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import urllib.request,urllib.parse
from census import *
rows=load('round3b/data/plan.json')
ex=set(load('round3b/data/source_exclusions.json')['excluded_candidate_ids'])
rows=[r for r in rows if r['id'] not in ex]
chunk=65
u='https://huggingface.co/api/datasets/Fhrozen/FSD50k/paths-info/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b'
def get(batch):
 fields=[('paths[]','clips/'+r['download_url'].split('/clips/')[1]) for r in batch]
 q=urllib.request.Request(u,data=urllib.parse.urlencode(fields).encode(),headers={'User-Agent':'VoiceHunter-audio-research/0.3b'})
 with urllib.request.urlopen(q,timeout=20) as z:
  objects=json.load(z)
 return {Path(r['path']).stem:{'remote_file_bytes':r['size'],'remote_lfs_sha256':r['lfs']['oid']} for r in objects}
catalog={}
with ThreadPoolExecutor(max_workers=8) as pool:
 for part in pool.map(get,[rows[i:i+chunk] for i in range(0,len(rows),chunk)]):catalog.update(part)
assert set(catalog)=={r['id'] for r in rows}
save('lfs_provenance.json',catalog)
hdr={r['id']:r for r in load('round3b/data/durations.json')}
selected=[dict(r,split='unused_candidate',duration=hdr[r['id']]['duration'],sample_rate=hdr[r['id']]['sample_rate'],channels=hdr[r['id']]['channels'],**catalog[r['id']]) for r in rows if hdr[r['id']]['eligible']]
save('eligible_sources.json',selected)
save('sealed_primary.json',[])
save('evidence.json',{'gate_passed':False,'eligible_candidates':len(selected),'sealed_test_audio_files':0,'waveforms_fully_downloaded_for_this_census':0,'method':'Pinned Hugging Face LFS SHA-256 and byte length from the official mirror paths-info API and separately decoded RIFF PCM format/data headers. Only headers were read, not full PCM waveforms. No primary test audio or score produced.','sha256':{n:sha((HERE/n).read_bytes()) for n in ('feasibility.json','freeze.json','plan.json','durations.json','source_exclusions.json','lfs_provenance.json','eligible_sources.json','sealed_primary.json')}})
print(load('round3b/data/evidence.json'))
