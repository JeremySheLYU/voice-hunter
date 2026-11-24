"""Losslessly acquire short stereo and 5.1 master excerpts via real FLAC seek tables."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import urllib.request

import numpy as np

HERE=Path(__file__).resolve().parent
RAW=HERE.parents[1]/'round2_data/spatial/films'
BASE='https://media.xiph.org/'
FILES={'bbb':{'st':'BBB/BigBuckBunny-stereo.flac','51':'BBB/BigBuckBunny-surround.flac'},
       'sintel':{'st':'sintel/sintel-master-st.flac','51':'sintel/sintel-master-51.flac'}}
STARTS=(60,180,300,420)


def get(url,start,end):
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    for i in range(3):
        try:
            with urllib.request.urlopen(req,timeout=35) as f:
                if f.status!=206 or not f.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise IOError('Requested FLAC range unavailable')
                return f.read(end-start+1)
        except OSError:
            if i==2:raise
            time.sleep(1+i)


def excerpt(job):
    film,kind,path=job
    url=BASE+path
    header=get(url,0,32767)
    assert header[:4]==b'fLaC'
    pos=4
    points=[]
    while True:
        flag=header[pos]
        size=int.from_bytes(header[pos+1:pos+4],'big')
        if flag&127==3:
            points=[struct.unpack_from('>QQH',header,p)[:2] for p in range(pos+4,pos+size+4,18)]
        pos+=4+size
        if flag&128:break
    assert points and pos<len(header)
    vals=int.from_bytes(header[18:26],'big'); sr=vals>>44; ch=((vals>>41)&7)+1
    assert ch=={'51':6,'st':2}[kind]
    stream_info=b'fLaC'+bytes([128])+header[5:42]
    sha=lambda b:hashlib.sha256(b).hexdigest()
    rows=[]
    for start in STARTS:
        n=start*sr
        src_sample,offset=max(p for p in points if p[0]<=n)
        _,stop=min(p for p in points if p[0]>n+12*sr)
        begin,end=pos+offset,pos+stop-1
        payload=get(url,begin,end)
        samples=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','fatal','-f','flac','-i','pipe:0','-f','s32le','pipe:1'],input=stream_info+payload)
        a=np.frombuffer(samples,'<i4').reshape(-1,ch)
        segment=a[n-src_sample:n-src_sample+12*sr]
        assert segment.shape==(12*sr,ch)
        output=RAW/f'{film}_{kind}_{start}s.flac'
        subprocess.run(['ffmpeg','-loglevel','error','-y','-f','s32le','-ar',str(sr),'-ac',str(ch),'-channel_layout','5.1' if ch==6 else 'stereo','-i','pipe:0','-c:a','flac',str(output)],check=True,input=segment.tobytes())
        rr=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-f','s32le','pipe:1'])
        assert rr==segment.tobytes()
        rows.append(dict(film=film,kind=kind,file=output.name,url=url,start_seconds=start,duration=12,sr=sr,channels=ch,
                         layout='FL FR FC LFE BL BR' if ch==6 else 'FL FR',source_header_sha256=sha(header),source_range=[begin,end],
                         source_range_sha256=sha(payload),source_seekpoint_sample=src_sample,decoded_pcm_sha256=sha(rr),local_sha256=sha(output.read_bytes())))
        print(output.name,flush=True)
    return rows


if __name__=='__main__':
    RAW.mkdir(parents=True,exist_ok=True)
    jobs=[(film,k,path) for film,files in FILES.items() for k,path in files.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        scenes=sum(list(pool.map(excerpt,jobs)),[])
    (HERE/'films_index.json').write_text(json.dumps(scenes,indent=2)+'\n')
