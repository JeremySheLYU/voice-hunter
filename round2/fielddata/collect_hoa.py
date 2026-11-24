"""Collect two seconds in all 25 original ACN/SN3D channels for one site per EigenScape class."""
import concurrent.futures
import json
import re
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path
import numpy as np
from collect import CAT, DATA, HERE, ROOT, RemoteZipTail, fetch, metadata, sha

def asset(a):
    key=a['key'].removesuffix('.zip')
    n=a['size'];url=a['links']['self']
    out=DATA/'eigenscape_hoa'/(key+'-01.wav')
    receipt=HERE/'receipts'/'eigenscape_hoa'/(key+'-01.json')
    if receipt.exists() and out.exists() and sha(out.read_bytes())==json.loads(receipt.read_text())['sha256']:return
    tail=fetch(url,n-262144,n-1)
    with zipfile.ZipFile(RemoteZipTail(tail,n)) as z:
        candidates=[i for i in z.infolist() if not any(c.startswith('.') or c=='__MACOSX' for c in Path(i.filename).parts)
                    and re.search(r'(?<!\d)0?1\.wav$',i.filename,re.I)]
        assert len(candidates)==1, [x.filename for x in z.infolist()]
        i=candidates[0]
    o=i.header_offset
    b=fetch(url,o,o+8_200_000-1)
    nl,xl=struct.unpack_from('<HH',b,26)
    compressed=b[30+nl+xl:]
    audio=zlib.decompressobj(-15).decompress(compressed) if i.compress_type==8 else compressed
    assert audio[:4] in [b'RIFF',b'RF64']
    out.parent.mkdir(parents=True,exist_ok=True)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-y','-loglevel','error','-i','pipe:0','-t','2','-map_metadata','-1','-c:a','pcm_s24le',str(out)],input=audio,check=True)
    def pcm(path):
        p=subprocess.check_output(['ffmpeg','-nostdin','-loglevel','error','-i',str(path),'-t','2','-f','s32le','pipe:1'])
        return np.frombuffer(p,dtype='<i4')
    h=pcm(out).reshape(-1,25)
    assert h.shape==(96000,25),h.shape
    f=pcm(DATA/'eigenscape'/(key+'.1.flac.flac')).reshape(-1,4)
    cor=[float(np.corrcoef(f[:,i],h[:,i])[0,1]) for i in range(4)]
    row=dict(id='eigenscape_hoa/'+key+'.1',dataset='eigenscape_hoa',record='2628463',member=i.filename,
             archive=a['key'],archive_checksum=a['checksum'],url=url,archive_range=[o,o+len(b)-1],
             range_sha256=sha(b),source_offset_s=0,duration_s=2,rate=48000,channels=25,
             format='ACN/SN3D order 4',bytes=out.stat().st_size,sha256=sha(out.read_bytes()),
             pcm_s32le_sha256=sha(h.tobytes()),first_four_vs_lite_correlation=cor,
             path=out.relative_to(ROOT).as_posix(),license='CC-BY-4.0')
    receipt.write_text(json.dumps(row,indent=2)+'\n');print(key,cor,flush=True)
if __name__=='__main__':
    z=metadata('2628463')
    blobs=[f for f in z['files'] if f['key'].endswith('.zip') and 'Lite' not in f['key']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(asset,blobs))
