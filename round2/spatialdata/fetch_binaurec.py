"""Fetch selected paired BinauRec room recordings by immutable Zenodo ZIP offsets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import binascii
import hashlib
import io
import json
import os
import struct
import time
import urllib.request
import wave
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URL = 'https://zenodo.org/api/records/7256984/files/BinauRec.zip/content'


def fetch_scene(row, path):
    folder = path / row['scene']
    folder.mkdir(parents=True, exist_ok=True)
    members = row['members']
    start = min(x[3] for x in members)
    end = max(x[3]+x[2]+512 for x in members)
    checks = {}
    local = {key: folder/(key+'.wav') for key in ['target','noise_ssn','mixture_ssn']}
    # Always verify against stored manifest if preexisting.
    if (folder/'hashes.json').exists():
        expected=json.loads((folder/'hashes.json').read_text())
        if all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected[k] for k,p in local.items()):
            return row['scene']
    for attempt in range(4):
        try:
            req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=40) as res:
                if res.status!=206 or not res.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise IOError('The server did not return the requested archive range.')
                buf=res.read(end-start+1)
            break
        except (OSError, TimeoutError):
            if attempt==3:raise
            time.sleep(1+attempt*2)
    channels={}
    for name,rawsize,comp,offset,codec,crc in members:
        pos=offset-start
        if buf[pos:pos+4]!=b'PK\x03\x04':raise ValueError(name+' ZIP header')
        l,e=struct.unpack_from('<HH',buf,pos+26)
        data=buf[pos+30+l+e:pos+30+l+e+comp]
        if codec==8:data=zlib.decompress(data,-15)
        assert len(data)==rawsize and binascii.crc32(data)==crc,name
        key,side,_=Path(name).stem.rsplit('_',2)
        with wave.open(io.BytesIO(data)) as w:
            assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000)
            audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
        channels[key,side]=audio
    for key,p in local.items():
        tmp=p.with_suffix('.part')
        lr=np.column_stack([channels[key,'left'],channels[key,'right']])
        with wave.open(str(tmp),'wb') as w:
            w.setparams((2,2,16000,len(lr),'NONE','not compressed'));w.writeframes(lr.tobytes())
        os.replace(tmp,p);checks[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    (folder/'hashes.json').write_text(json.dumps(checks,indent=2))
    return row['scene']


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'round2_data/spatial/binaurec')
    ap.add_argument('--workers',type=int,default=5)
    args=ap.parse_args()
    plan=json.loads((HERE/'binaurec_plan.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for scene in pool.map(lambda row: fetch_scene(row,args.out),plan['scenes']):
            print(scene,flush=True)
