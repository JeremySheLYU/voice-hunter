"""Select public CC0 / CC BY music by artist before examining audio; mirror exact MP3 members from FMA-small."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE='https://os.unil.cloud.switch.ch/fma/'
SOURCE='https://github.com/mdeff/fma'

def fetch(url, start, end):
    for attempt in range(4):
        try:
            headers={'Range':f'bytes={start}-{end}','User-Agent':'VoiceHunter-Dataset/0.2 (range request)'}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=50) as resp:
                content_range=resp.headers.get('Content-Range','')
                if resp.status!=206 or not content_range.startswith(f'bytes {start}-{end}/'):
                    raise ValueError(f'Expected bounded response, received {content_range}')
                result=resp.read(end-start+2)
                if len(result)!=end-start+1:raise ValueError('Incomplete HTTP range')
                return result,int(content_range.rsplit('/',1)[1])
        except (OSError,ValueError) as exc:
            if attempt==3:raise
            time.sleep(1+attempt)

class RangeFile:
    def __init__(self,name):
        self.url=BASE+name
        _,self.length=fetch(self.url,0,0)
        self.pos=0
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=([0,self.pos,self.length][whence])+offset
        if self.pos<0:raise OSError('negative seek')
        return self.pos
    def read(self,size=-1):
        n=self.length-self.pos if size<0 else min(size,self.length-self.pos)
        if n<=0:return b''
        data,_=fetch(self.url,self.pos,self.pos+n-1)
        self.pos+=n
        return data

def digest(txt):
    return hashlib.sha256(txt.encode()).hexdigest()

def rights(link):
    u=urllib.parse.urlsplit(link)
    return ((u.hostname or '').lower() in {'creativecommons.org','www.creativecommons.org'}
            and bool(re.fullmatch(r'/(publicdomain/zero/1\.0|licenses/by/[1-4]\.0(?:/us|/de|/uk|/fr|/es)?)/?',u.path)))

def pick_candidates():
    with zipfile.ZipFile(RangeFile('fma_small.zip')) as audio:
        infos={int(Path(i.filename).stem):i for i in audio.infolist() if i.filename.endswith('.mp3')}
    import bz2
    cache=ROOT/'round2_data/fma_raw_tracks.csv.bz2'
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(RangeFile('fma_metadata.zip')) as meta:
            info=meta.getinfo('fma_metadata/raw_tracks.csv')
            stream=member_bytes(meta.fp.url,info)
            cache.write_bytes(bz2.compress(stream,compresslevel=1))
    csv.field_size_limit(16*1024*1024)
    by_artist={}
    with bz2.open(cache,'rt',encoding='utf-8') as csv_file:
        for raw in csv.DictReader(csv_file):
            trackid=int(raw['track_id'])
            lic=raw['license_url']
            if trackid not in infos or not rights(lic):continue
            artist=raw['artist_id']
            if not artist or not raw['artist_name'].strip():continue
            row=dict(id=trackid,artist_id=artist,artist=raw['artist_name'],track=raw['track_title'],
                     album=raw['album_title'],license=lic,license_title=raw['license_title'],
                     track_url=raw['track_url'],artist_url=raw['artist_url'],copyright=raw['track_copyright_c'],
                     archive_member=infos[trackid].filename,source=SOURCE)
            by_artist.setdefault(artist,[]).append(row)
    candidates=[]
    for artist,items in by_artist.items():
        row=min(items,key=lambda r:digest('vh-r2-track/'+str(r['id'])))
        candidates.append(row)
    return sorted(candidates,key=lambda r:digest('vh-r2-artist/'+r['artist_id'])),infos

def member_bytes(url,info):
    # Fetch an entire member plus its local header in one range (capped at the archive end).
    import bz2,struct,zlib
    start=info.header_offset
    length=30+len(info.filename.encode())+len(info.extra)+info.compress_size
    blob,_=fetch(url,start,start+length+127)
    assert blob[:4]==b'PK\x03\x04'
    fn,extra=struct.unpack_from('<HH',blob,26); offset=30+fn+extra
    if len(blob)<offset+info.compress_size:
        blob,_=fetch(url,start,start+offset+info.compress_size-1)
    raw=blob[offset:offset+info.compress_size]
    if info.compress_type==zipfile.ZIP_BZIP2:payload=bz2.decompress(raw)
    elif info.compress_type==zipfile.ZIP_DEFLATED:payload=zlib.decompress(raw,-15)
    elif info.compress_type==zipfile.ZIP_STORED:payload=raw
    else:raise ValueError('Unsupported archive compression')
    if len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC:raise ValueError('ZIP integrity failure')
    return payload

def metadata(path):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
      'stream=sample_rate,channels,duration,channel_layout','-of','json',str(path)]))['streams'][0]
    return dict(rate=int(info['sample_rate']),channels=info['channels'],seconds=float(info['duration']))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count',type=int,default=220);ap.add_argument('--jobs',type=int,default=8)
    ap.add_argument('--data',type=Path,default=ROOT/'round2_data/stereo');args=ap.parse_args()
    args.data.mkdir(parents=True,exist_ok=True)
    cand,infos=pick_candidates()
    print('eligible unique artists',len(cand),'selected',min(args.count,len(cand)),flush=True)
    def download(pair):
        rank,r=pair
        out=args.data/f"fma_{r['id']:06d}.mp3"
        i=infos[r['id']]
        if not out.is_file():
            blob=member_bytes(BASE+'fma_small.zip',i)
            temp=out.with_suffix('.part')
            temp.write_bytes(blob);temp.replace(out)
        payload=out.read_bytes()
        import zlib
        if len(payload)!=i.file_size or zlib.crc32(payload)!=i.CRC:raise ValueError(f'Bad cache {out.name}')
        bucket=int(digest('vh-r2-split/'+r['artist_id'])[:12],16)%10
        return dict(**r,**metadata(out),filename=out.name,split='test' if bucket>=8 else 'validation' if bucket==7 else 'train',
                    bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        rows=list(pool.map(download,enumerate(cand[:args.count])))
    (HERE/'manifest.json').write_text(json.dumps(dict(dataset='FMA small',archive=BASE+'fma_small.zip',
        metadata_archive=BASE+'fma_metadata.zip',selection='One track per credited artist ID, hash-ranked before probing audio. Exact per-track CC0 or CC BY. Artist-hash split.',
        eligible_unique_artists=len(cand),records=rows),indent=2,ensure_ascii=False)+'\n')
    print('done',len(rows),sum(r['seconds'] for r in rows),sum(r['bytes'] for r in rows))

if __name__=='__main__':main()
