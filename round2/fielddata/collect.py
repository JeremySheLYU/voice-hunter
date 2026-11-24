"""Fetch deterministic short windows of permissively licensed spatial recordings from pinned Zenodo ZIPs."""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import time
import urllib.request
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAT = HERE / 'catalogue'
DATA = ROOT / 'round2_data' / 'field'
PLANS = {
 'eigenscape': ('2628463','Lite-EigenScape.zip',8),
 'ihta_foa': ('11184517','FirstOrderAmbisonics.zip',4),
 'ihta_ears': ('11184517','Binaural.zip',4),
 'london': ('10687363','BFormat_Ambix.zip',8),
}
def sha(b):return hashlib.sha256(b).hexdigest()

def fetch(url,start=None,end=None):
    h={'User-Agent':'VoiceHunter-Research/0.2 (+open-data archive excerpt benchmark)'}
    if start is not None: h['Range']=f'bytes={start}-{end}'
    for attempt in range(5):
        try:
            req=urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req,timeout=45) as r:
                if start is not None and (r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')):
                    raise ValueError(f'Invalid range response {r.status} {r.headers.get("Content-Range")}')
                b=r.read()
                if start is not None and len(b)!=end-start+1:raise ValueError('Short response')
                return b
        except Exception as exc:
            if attempt==4:raise
            time.sleep((1+attempt)*2)

class RemoteZipTail(io.BytesIO):
    """A sparse reader over the final ZIP directory; local offsets come from verified central entries."""
    def __init__(self,tail,total): super().__init__(tail);self.total=total;self.start=total-len(tail)
    def seek(self,n,whence=0):
        if whence==0: n-=self.start
        return super().seek(n,whence)+self.start
    def tell(self):return super().tell()+self.start

def metadata(id):
    p=CAT/(id+'.json')
    if not p.exists():p.write_bytes(fetch('https://zenodo.org/api/records/'+id))
    return json.loads(p.read_text())

def entries(key):
    record,archive,secs=PLANS[key]
    data=metadata(record)
    assert data['metadata']['license']['id']=='cc-by-4.0'
    a=next(f for f in data['files'] if f['key']==archive)
    url=a['links']['self'];size=a['size']
    tail=fetch(url,max(0,size-262144),size-1)
    with zipfile.ZipFile(RemoteZipTail(tail,size)) as z:
        items=[dict(name=i.filename,offset=i.header_offset,compressed=i.compress_size,method=i.compress_type,crc32=i.CRC) for i in z.infolist()
              if i.filename.lower().endswith(('.flac','.wav')) and not any(c.startswith('.') or c=='__MACOSX' for c in Path(i.filename).parts)]
    return dict(dataset=key,record=record,url=url,archive=archive,archive_checksum=a['checksum'],
                full_zip_size=size,seconds=secs,files=sorted(items,key=lambda f:f['name']))

def decode(key,sec,i,cat):
    name=i['name'];safe=re.sub('[^a-zA-Z0-9._-]','_',name)
    out=DATA/key/(safe+'.flac')
    receipt=HERE/'receipts'/key/(safe+'.json')
    if receipt.exists() and out.exists() and sha(out.read_bytes())==json.loads(receipt.read_text())['sha256']:return
    out.parent.mkdir(parents=True,exist_ok=True);receipt.parent.mkdir(parents=True,exist_ok=True)
    # One range per recording normally suffices even for 24-bit incompressible material.
    channels=2 if key=='ihta_ears' else 4
    length=min(i['compressed']+4096,(sec+1)*48000*3*channels+65536)
    while True:
        end=min(i['offset']+length-1,cat['full_zip_size']-1)
        b=fetch(cat['url'],i['offset'],end)
        assert b[:4]==b'PK\x03\x04'
        nl,xl=struct.unpack_from('<HH',b,26)
        start=30+nl+xl
        payload=b[start:start+i['compressed']]
        if i['method']==8: audio=zlib.decompressobj(-15).decompress(payload)
        elif i['method']==0: audio=payload
        else:raise ValueError('unsupported compression')
        # Select channels by index, preserve channel order. Do not mix with ffmpeg's speaker-layout heuristics.
        src=audio[:4];assert src in [b'RIFF',b'fLaC',b'RF64']
        args=['ffmpeg','-nostdin','-loglevel','error','-i','pipe:0','-t',str(sec),'-map_metadata','-1','-c:a','flac','-compression_level','2','-f','flac','pipe:1']
        proc=subprocess.run(args,input=audio,capture_output=True)
        probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=channels,sample_rate','-of','json','pipe:0'],input=proc.stdout,capture_output=True)
        streams=json.loads(probe.stdout or '{}').get('streams',[])
        if streams:
            st=streams[0];rate=int(st['sample_rate'])
            pcm=subprocess.run(['ffmpeg','-nostdin','-loglevel','error','-i','pipe:0','-f','s32le','-c:a','pcm_s32le','pipe:1'],input=proc.stdout,capture_output=True)
            count=len(pcm.stdout)//(4*channels)
            if st['channels']==channels and count==rate*sec:break
        if len(payload)==i['compressed']:raise ValueError(f'{name}: unsupported format/duration {streams}')
        length=min(length*2,i['compressed']+4096)
    out.write_bytes(proc.stdout)
    row=dict(id=key+'/'+safe,dataset=key,record=cat['record'],member=name,source_offset_s=0,
             duration_s=count/rate,rate=rate,channels=channels,bytes=out.stat().st_size,sha256=sha(proc.stdout),
             pcm_s32le_sha256=sha(pcm.stdout),archive_crc32=hex(i['crc32']),archive_range=[i['offset'],end],
             range_sha256=sha(b),path=out.relative_to(ROOT).as_posix(),license='CC-BY-4.0')
    receipt.write_text(json.dumps(row,indent=2)+'\n')
    print(key,name,count,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('sets',nargs='*',choices=list(PLANS));ap.add_argument('--catalogue',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    CAT.mkdir(exist_ok=True);tasks=[]
    for key in args.sets:
        catalog=CAT/(key+'.json')
        if not catalog.exists():catalog.write_text(json.dumps(entries(key),indent=2)+'\n')
        c=json.loads(catalog.read_text())
        print(key,len(c['files']),[f['name'] for f in c['files'][:7]],flush=True)
        if not args.catalogue:
            tasks.extend((key,c['seconds'],f,c) for f in c['files'])
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(decode,*args):args[2]['name'] for args in tasks}
        for job in concurrent.futures.as_completed(jobs):
            try:job.result()
            except Exception as e:failures.append([jobs[job],str(e)]);print('FAILED',jobs[job],e,flush=True)
    if failures:raise SystemExit(str(failures))

if __name__=='__main__':main()
