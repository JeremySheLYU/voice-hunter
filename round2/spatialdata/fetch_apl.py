"""Fetch complete AmbiX set and one 100-pose KU100 SOFA from the APL concert hall."""
import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
URL='https://zenodo.org/api/records/2641166/files/360-BRIR-FOAIR-database.zip/content'


def getbyte(span):
    start,end=span
    for n in range(4):
        try:
            req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req,timeout=60) as r:
                if r.status!=206 or not r.headers['Content-Range'].startswith(f'bytes {start}-'):
                    raise IOError('Unexpected ZIP range.')
                result=r.read(end-start+1)
            assert len(result)==end-start+1
            return result
        except (OSError,TimeoutError):
            if n==3:raise
            time.sleep(n*2+1)


def retrieve(row,out):
    name,length,compressed,position,codec,crc=row
    path=out/Path(name).name
    if path.exists() and (path.stat().st_size,binascii.crc32(path.read_bytes()))==(length,crc):return
    raw=getbyte((position,position+255))
    assert raw[:4]==b'PK\x03\x04'
    n,e=struct.unpack_from('<HH',raw,26)
    start=position+30+n+e
    spans=[(a,min(start+compressed-1,a+4*1024*1024-1)) for a in range(start,start+compressed,4*1024*1024)]
    crc_check=0
    decoder=zlib.decompressobj(-15) if codec==8 else None
    with ThreadPoolExecutor(max_workers=4) as pool,path.with_suffix('.part').open('wb') as f:
        for b in pool.map(getbyte,spans):
            block=decoder.decompress(b) if decoder else b
            crc_check=binascii.crc32(block,crc_check);f.write(block)
        if decoder:
            block=decoder.flush();f.write(block);crc_check=binascii.crc32(block,crc_check)
    tmp=path.with_suffix('.part')
    assert tmp.stat().st_size==length and crc==crc_check
    tmp.replace(path)
    print(name,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'round2_data/spatial/apl')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((HERE/'apl_plan.json').read_text())
    for row in plan['members']:retrieve(row,a.out)
