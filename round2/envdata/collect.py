"""Acquire attributed environmental source recordings, with uploader-disjoint assignment."""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request as net
import wave
from zipfile import ZipFile
from collections import Counter,defaultdict
from urllib.parse import urlsplit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=ROOT/'round2_data/environment'
META=RAW/'metadata'
HF='https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b'
Z='https://zenodo.org/api/records/4060432/files'
GROUPS={'train':{'Train'},'wind':{'Wind'},'piano':{'Piano'},'guitar':{'Acoustic_guitar','Electric_guitar'},'speech':{'Female_speech_and_woman_speaking','Male_speech_and_man_speaking'}}
TITLES={'train':r'train|rail|metro|subway|zug|steam|locomotiv|tram','wind':r'wind|storm|breeze|gale|vent|blow|gust|hurricane','piano':r'piano|key|chord|note','guitar':r'guit|strum|chord|riff','speech':r'voice|speak|talk|speech|phrase|word|read|dialog|sentence|narrat|vocal'}
ANCESTORS={'train':{'Rail_transport','Vehicle','Mechanisms'},'wind':{'Natural_sounds'},'piano':{'Keyboard_(musical)','Musical_instrument','Music'},'guitar':{'Guitar','Plucked_string_instrument','Musical_instrument','Music'},'speech':{'Human_voice','Speech'}}
ESC_URL='https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/meta/esc50.csv'

def req(url,limit,seconds=20):
    request=net.Request(url, headers={'User-Agent':'VoiceHunter-audio-research/0.2'})
    with net.urlopen(request,timeout=seconds) as response:
        data=response.read(limit+1)
    if len(data)>limit: raise ValueError('Size budget')
    return data

def sha(data):
    return hashlib.sha256(data).hexdigest()

def split(author):
    b=int(sha(('VoiceHunter-env-v2-uploader:'+author.strip().casefold()).encode())[:8],16)%100
    return 'train' if b<62 else 'validation' if b<80 else 'test'

def candidates():
    META.mkdir(parents=True,exist_ok=True)
    for name in ['FSD50K.metadata.zip','FSD50K.ground_truth.zip','FSD50K.doc.zip']:
        p=META/name
        if not p.is_file():p.write_bytes(req(f'{Z}/{name}/content',8*1024**2))
    esc=META/'esc50.csv'
    if not esc.exists():esc.write_bytes(req(ESC_URL,1024**2))
    excluded={r['src_file'] for r in csv.DictReader(io.StringIO(esc.read_text()))}
    out=[]
    with ZipFile(META/'FSD50K.ground_truth.zip') as g, ZipFile(META/'FSD50K.metadata.zip') as z:
        for part in ['dev','eval']:
            info=json.loads(z.read(f'FSD50K.metadata/{part}_clips_info_FSD50K.json'))
            for r in csv.DictReader(io.StringIO(g.read(f'FSD50K.ground_truth/{part}.csv').decode())):
                id=r['fname']; x=info[id]; lic=x['license']
                if id in excluded or not re.fullmatch(r'https?://creativecommons.org/(?:licenses/by/(?:1\.0|2\.0|2\.5|3\.0|4\.0)|publicdomain/zero/1\.0)/?',lic):continue
                labels=set(r['labels'].split(',')); cats=[c for c,k in GROUPS.items() if k & labels]
                if len(cats)!=1:continue
                c=cats[0];author=x['uploader'];title=x['title']
                if not author:continue
                noise=len(labels-GROUPS[c]-ANCESTORS[c])
                priority=(0 if re.search(TITLES[c],title,re.I) else 2)+min(6,noise)
                out.append(dict(id=id,category=c,author=author,license=lic,title=title,
                                url=f'https://freesound.org/s/{id}/',split=split(author),
                                collection_partition=part,labels=sorted(labels),priority=priority))
    return out

def save_catalogue(rows,catalog):
    sorted_rows=sorted(rows,key=lambda r:(r['category'],r['split'],int(r['id'])))
    tmp=catalog.with_suffix('.part')
    tmp.write_text(json.dumps(sorted_rows,ensure_ascii=False,indent=1)+'\n')
    tmp.replace(catalog)

def label_split(rows,target):
    buckets=defaultdict(list)
    for x in rows:buckets[x['category'],x['split']].append(x)
    for key,group in buckets.items():
        group.sort(key=lambda r:(r['priority'],sha(('holdout-env-'+r['id']).encode())))
        seen=Counter()
        # Within each label and split prefer distinct uploaders and events.
        for row in group:
            row['author_rank']=seen[row['author'].casefold()]
            seen[row['author'].casefold()]+=1
        group.sort(key=lambda r:(r['author_rank'],r['priority'],sha(('holdout-env-'+r['id']).encode())))
    sizes=dict(train=target,validation=max(4,target//3),test=max(4,target//3))
    return [(key,group,sizes[key[1]]) for key,group in buckets.items()]

def audio(x,kind):
    id=x['id']; d=RAW/x['category'];d.mkdir(exist_ok=True)
    file=d/(id+('.wav' if kind=='standardized' else '.flac'))
    official=f"{HF}/clips/{x['collection_partition']}/{id}.wav"
    if kind=='standardized':
        data=req(official,3_000_000,20)
        with wave.open(io.BytesIO(data),'rb') as h:
            rate=h.getframerate();ch=h.getnchannels();n=h.getnframes()
        if n/rate<4.9: raise ValueError('Short reference')
        file.write_bytes(data); url=official; original_hash=sha(data)
    else:
        html=req(x['url'],360000,15).decode('utf-8','replace')
        # Require current Freesound page and dataset metadata to agree on a permissible license type.
        page_l=sorted(set(re.findall(r'https?://creativecommons\.org/(?:licenses|publicdomain)/[^"<> ]+',html)))
        normalized=lambda l:urlsplit(l).path.strip('/').casefold()
        if not any(normalized(l)==normalized(x['license']) for l in page_l):raise ValueError('Page attribution mismatch')
        url=re.search(r'https?://cdn\.freesound\.org/previews/\d+/\d+_\d+-hq\.mp3',html)
        if not url:raise ValueError('Missing preview')
        url=url.group()
        # Record a bounded first-15-second excerpt. A partial MP3 transfer is decoded and re-encoded to FLAC without channel conversion.
        request=net.Request(url,headers={'Range':'bytes=0-819199','User-Agent':'VoiceHunter-audio-research/0.2'})
        with net.urlopen(request,timeout=18) as response:
            blob=response.read(819200)
        original_hash=sha(blob)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'input.mp3'; source.write_bytes(blob)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=sample_rate,channels','-of','json',str(source)]))['streams'][0]
            ch=probe['channels'];rate=int(probe['sample_rate'])
            if ch!=2:raise ValueError('Not stereo preview')
            temp_flac=Path(temp)/'clip.flac'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(source),'-t','15','-map_metadata','-1','-c:a','flac',str(temp_flac)],check=True,capture_output=True)
            ss=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=duration_ts,time_base','-of','json',str(temp_flac)]))['streams'][0]
            from fractions import Fraction
            n=round(int(ss['duration_ts'])*Fraction(ss['time_base'])*rate)
            if n/rate<4.9:raise ValueError('Short stereo reference')
            file.write_bytes(temp_flac.read_bytes())
    data=file.read_bytes()
    return {k:v for k,v in x.items() if k not in {'priority','author_rank'}}|dict(
        path=str(file.relative_to(ROOT)),duration=round(n/rate,4),channels=ch,sample_rate=rate,frames=n,
        bytes=len(data),sha256=sha(data),download_url=url,download_sha256=original_hash,representation=kind)

def collect(args):
    candidates_all=candidates()
    catalog=HERE/args.catalog
    prior=json.loads(catalog.read_text()) if catalog.exists() else []
    # Canonical IDs appear once across mono and original-stereo paths. Restore revision from pinned download URLs.
    recorded={x['id'] for x in prior}; known_hashes={x['sha256'] for x in prior}
    counts=Counter((x['category'],x['split']) for x in prior if x['representation']==args.kind)
    trials=Counter()
    buckets=label_split(candidates_all,args.per_class)
    work=[]
    for key,group,wanted in sorted(buckets):
        remain=max(0,wanted-counts[key])
        eligible=[x for x in group if x['id'] not in recorded]
        if args.kind=='stereo_preview':
            # Take independent IDs in a fixed order far from the front of the standardized queue.
            eligible=list(reversed(eligible))
            eligible=sorted([x for x in eligible if x['priority']<=3],key=lambda x:(x['priority'],sha(('stereo-preview-native-v2:'+x['id']).encode())))
        if remain:work.append([key,iter(eligible),wanted,0])
    failures=Counter(); byid={r['id']:r for r in prior}
    pending={}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for key,it,wanted,_ in work:
                outstanding=sum(k==key for k,row in pending.values())
                if counts[key]+outstanding >=wanted:continue
                if trials[key]>=wanted*args.attempt_factor:continue
                for _ in range(min(4,wanted-counts[key]-outstanding)):
                    x=next(it,None)
                    if x is None:break
                    if x['id'] in recorded:continue
                    trials[key]+=1
                    pending[pool.submit(audio,x,args.kind)]=(key,x)
                if len(pending)>=args.workers*2:break
            if not pending:break
            done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in done:
                key,x=pending.pop(f)
                try:r=f.result()
                except Exception as e:
                    failures[type(e).__name__+':'+str(e)[:48]]+=1;continue
                if r['sha256'] in known_hashes:continue
                known_hashes.add(r['sha256']);recorded.add(r['id']); counts[key]+=1;byid[r['id']]=r
            save_catalogue(list(byid.values()),catalog)
            if sum(counts.values()) and sum(counts.values())%20==0:print('indexed',len(byid),flush=True)
    print('representation',args.kind, 'counts',counts, 'errors',failures,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--kind',choices=['standardized','stereo_preview'],default='standardized')
    a.add_argument('--per-class',type=int,default=12,help='train target for each class; separate val/test one-third of this')
    a.add_argument('--catalog',default='manifest.json')
    a.add_argument('--workers',type=int,default=16)
    a.add_argument('--attempt-factor',type=int,default=6)
    collect(a.parse_args())
