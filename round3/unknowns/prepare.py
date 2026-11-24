"""Collect a frozen negative-control set of five entirely different event labels."""
import csv
import hashlib
import io
import json
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
from round2.envdata.collect import META, HF, req, GROUPS
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LABELS=('Bark','Meow','Church_bell','Clock','Drip')
REQUIRE=6

def sha(b):
    return hashlib.sha256(b).hexdigest()

def write(name, content):
    (HERE/name).write_text(json.dumps(content,indent=2,ensure_ascii=False))

def collect():
    seen_auth=set()
    seen_ids=set()
    for filename in ('round2/envdata/all_sources.json','round2/environment/native_inventory.json','round3/data/plan.json'):
        for row in json.loads((ROOT/filename).read_text()):
            seen_auth.add(row['author'].strip().casefold())
            seen_ids.add(row['id'])
    forbidden=set.union(*GROUPS.values(),{'Speech','Guitar','Keyboard_(musical)','Music','Rail_transport','Wind_noise_(microphone)'})
    candidates=[]
    with ZipFile(META/'FSD50K.ground_truth.zip') as z, ZipFile(META/'FSD50K.metadata.zip') as m:
        for split in ('dev','eval'):
            info=json.loads(m.read(f'FSD50K.metadata/{split}_clips_info_FSD50K.json'))
            for r in csv.DictReader(io.StringIO(z.read(f'FSD50K.ground_truth/{split}.csv').decode())):
                sound_id=r['fname']; meta=info[sound_id]; tags=set(r['labels'].split(','))
                labs=tags.intersection(LABELS); au=meta['uploader'].strip().casefold()
                if len(labs)!=1 or au in seen_auth or sound_id in seen_ids or tags & forbidden:continue
                lic=meta['license']
                if not re.fullmatch(r'https?://creativecommons.org/(?:licenses/by/[\d.]+|publicdomain/zero/1.0)/?',lic):continue
                candidates.append(dict(category=labs.pop(),author=meta['uploader'],id=sound_id,
                       license=lic,url=f'https://freesound.org/s/{sound_id}/',title=meta['title'],labels=sorted(tags),
                       download_url=f'{HF}/clips/{split}/{sound_id}.wav',path=f'round3_data/unknown/{sound_id}.wav'))
    queue=sorted(candidates,key=lambda r:(len(r['labels']),sha(('VH3-extra-negative:'+r['id']).encode())))
    for lab in LABELS:
        if sum(r['category']==lab for r in queue)<REQUIRE:raise RuntimeError(lab)
    write('freeze.json',dict(labels=LABELS,target_per_class=REQUIRE,
          protocol='Order by annotation count then SHA of ID. Minimum 3 seconds. One clip per uploader. Mix pairs once at two fixed pan positions, asking for all five target categories. All are declared absent by dataset labels. No tuning from these results.',
          ordered_ids={lab:[r['id'] for r in queue if r['category']==lab] for lab in LABELS}))
    return queue

def fetch(r):
    path=ROOT/r['path']
    try:
        blob=path.read_bytes() if path.is_file() else req(r['download_url'],4_500_000,18)
        with wave.open(io.BytesIO(blob)) as wav:
            if wav.getnchannels()!=1 or wav.getnframes()<3*wav.getframerate():return
            rate=wav.getframerate(); frames=wav.getnframes()
            if len(wav.readframes(frames))!=frames*wav.getsampwidth():return
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_bytes(blob)
        return dict(r,sha256=sha(blob),sample_rate=rate,frames=frames)
    except (ValueError,EOFError,OSError,wave.Error):return

if __name__=='__main__':
    queue=collect()
    nums={l:0 for l in LABELS}; authors=set(); chosen=[]
    for start in range(0,len(queue),16):
        batch=[r for r in queue[start:start+16] if nums[r['category']]<REQUIRE and r['author'].strip().casefold() not in authors]
        if not batch:continue
        with ThreadPoolExecutor(max_workers=5) as pool:
            for r in pool.map(fetch,batch):
                if not r:continue
                author=r['author'].strip().casefold()
                if nums[r['category']]>=REQUIRE or author in authors:continue
                nums[r['category']]+=1;authors.add(author);chosen.append(r)
        write('manifest.json',chosen)
        print(nums,flush=True)
        if all(v>=REQUIRE for v in nums.values()):break
