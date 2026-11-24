"""Verify packaged user-facing command and HTTP routes on known interoperability fixtures."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import build_opener, Request, ProxyHandler
import wave
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from voicehunter.media import load_media, write_wav


def cli(*args, project=ROOT):
    return subprocess.run([sys.executable,'-m','voicehunter',*map(str,args)],cwd=project,
                          text=True,capture_output=True,timeout=150)


def process(path, dest, *args, project=ROOT):
    return cli('process',path,'--out',dest,*args,project=project)


def get_modes(project=ROOT):
    proc=cli('modes',project=project)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1000:])
    return {m['id']:m for m in json.loads(proc.stdout)}


def pipeline(output, original):
    files = {key:load_media(output/filename)[0] for key,filename in (
        ('target','target.wav'),('remainder','remaining.wav'),('mix','remix.wav'))}
    shapes = {key:list(w.shape) for key,w in files.items()}
    finite = all(np.isfinite(w).all() for w in files.values())
    aligned = all(w.shape == original.shape for w in files.values())
    if not aligned:
        raise AssertionError(f'Mismatched output shapes: {shapes}, expected {original.shape}')
    peak_error = float(np.max(abs(files['target'].astype('f8')+files['remainder']-original)))
    return bool(aligned and finite and peak_error <= 2e-6), dict(
        samples=len(original),channels=2,finite=bool(finite),peak_sum_error=peak_error,shapes=shapes)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


@contextmanager
def server():
    with tempfile.TemporaryFile() as err:
        proc=subprocess.Popen([sys.executable,'-m','voicehunter','serve','--port','0'],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=err,text=True)
        try:
            if not select.select([proc.stdout],[],[],30)[0]:
                raise RuntimeError('Local server did not announce its port')
            status=proc.stdout.readline().strip()
            if 'http://127.0.0.1:' not in status:
                err.seek(0);raise RuntimeError(f'Local server could not start: {status}; '+err.read().decode()[-500:])
            yield status.split('http://')[-1].rstrip('/')
        finally:
            proc.terminate()
            try: proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()


def request(host, route, payload=None, file_name=None):
    headers={}
    if file_name is not None:
        headers={'Content-Type':'application/octet-stream','X-Filename':file_name}
    elif payload is not None:
        payload=json.dumps(payload).encode()
        headers={'Content-Type':'application/json'}
    obj=Request('http://'+host+route,data=payload,headers=headers)
    opener=build_opener(ProxyHandler({}))
    try:
        with opener.open(obj,timeout=150) as rep:
            return rep.status,rep.read()
    except HTTPError as e:
        return e.code,e.read()


def api(host,route,payload=None,file_name=None):
    status,body=request(host,route,payload,file_name)
    return status,json.loads(body)


class Checks:
    def __init__(self,output):
        self.records=[]
        self.path=HERE/output

    def add(self,name,passed,**observations):
        self.records.append(dict(check=name,passed=bool(passed),**observations))
        self.path.write_text(json.dumps(self.records,indent=2,ensure_ascii=False)+'\n')
        print(('PASS' if passed else 'FAIL'),name,flush=True)

    def finish(self):
        return int(not self.records or any(not r['passed'] for r in self.records))


def preflight(tmp,checks):
    archive=ROOT/'outputs/VoiceHunter_v0.2_PublicAudio.zip'
    if not archive.exists():
        archive=ROOT/'outputs/VoiceHunter_v0.2_Runtime.zip'
    old=tmp/'v02'
    with ZipFile(archive) as src:
        for member in src.namelist():
            if '/voicehunter/' in member and not member.endswith('/'):
                dest=old/member.split('/',1)[1];dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(src.read(member))
    old_modes=get_modes(old)
    now_modes=get_modes()
    checks.add('existing-mode-discovery',all(now_modes.get(k)==v for k,v in old_modes.items()),
               archive_sha256=file_hash(archive),published_modes=sorted(old_modes))
    tests=(('example','studio.wav',('--span',.05,2.05,'--background',2.35,4.30)),
           ('speakers','duet.wav',('--span',.2,1.65,'--background',2.2,3.65)))
    for mode,filename,extra in tests:
        src=ROOT/'examples'/filename
        folders=[tmp/(mode+'-old'),tmp/(mode+'-now')]
        procs=[process(src,path,'--mode',mode,*extra,project=project) for path,project in zip(folders,[old,ROOT])]
        ok=all(p.returncode==0 for p in procs)
        matches=[]
        if ok:
            for name in ('target.wav','remaining.wav','remix.wav','settings.json'):
                if file_hash(folders[0]/name)==file_hash(folders[1]/name):
                    matches.append(name)
        checks.add('legacy-'+mode+'-byte-stability',ok and len(matches)==4,matched_files=matches,
                   codes=[p.returncode for p in procs])
    src=HERE/'guitar_wind.flac'
    paths=[tmp/'apple-old',tmp/'apple-now']
    procs=[process(src,d,'--mode','auto','--target','guitar',project=project)
           for d,project in zip(paths,[old,ROOT])]
    ok=procs[0].returncode==procs[1].returncode
    if procs[0].returncode==0 and ok:
        ok=all(file_hash(paths[0]/n)==file_hash(paths[1]/n) for n in ('target.wav','remaining.wav','remix.wav','settings.json'))
    elif ok:
        ok=procs[0].stderr.strip().splitlines()[-1:]==procs[1].stderr.strip().splitlines()[-1:] and not any(d.exists() for d in paths)
    checks.add('legacy-apple-route',ok,codes=[p.returncode for p in procs],
               outcome='returned' if procs[0].returncode==0 else 'refused')


def full(tmp,checks):
    lock=json.loads((ROOT/'round3/evaluation/selection.json').read_text())
    name=lock['primary']
    modes=get_modes()
    required={'speech','guitar','piano','train','wind'}
    checks.add('five-events-advertised', {o['value'] for o in modes.get('five-events',{}).get('options',[])}==required,primary=name)
    fixture=HERE/'guitar_wind.flac'
    info=json.loads((HERE/'fixture.json').read_text())
    checks.add('fixture-provenance',file_hash(fixture)==info['sha256'],sha256=file_hash(fixture),source_ids=[r['id'] for r in info['sources']])
    x,_=load_media(fixture)
    silent=tmp/'silence.wav';write_wav(silent,np.zeros_like(x),stem=True)
    mono=tmp/'mono.wav'
    with wave.open(str(mono),'wb') as f:
        f.setparams((1,2,48000,len(x),'NONE','none'));f.writeframes(np.zeros(len(x),dtype='<i2').tobytes())
    arguments=('--mode','five-events','--target','guitar')
    dest=tmp/'cli'
    out=process(fixture,dest,*arguments)
    ok=out.returncode==0
    meta={}
    if ok:
        settings=json.loads((dest/'settings.json').read_text())
        ok,meta=pipeline(dest,x)
        ok=ok and settings.get('model')==name and settings.get('method')=='five-events' and settings.get('target')=='guitar'
    checks.add('published-cli-accepts-stereo',ok,primary=name,returncode=out.returncode,
               detail=out.stderr.strip().splitlines()[-1:] if out.returncode else [],**meta)
    for label,sound,args in [('silence',silent,arguments),('mono',mono,arguments),
                             ('unknown-label',fixture,('--mode','five-events','--target','trombone'))]:
        output=tmp/('reject-'+label)
        result=process(sound,output,*args)
        checks.add('cli-refuses-'+label,result.returncode!=0 and bool(result.stderr.strip()) and not output.exists(),
                   returncode=result.returncode,message=result.stderr.strip().splitlines()[-1:],
                   wrote_folder=output.exists())
    with server() as host:
        status,cap=api(host,'/api/capabilities')
        found=next((m for m in cap['modes'] if m['id']=='five-events'),None)
        checks.add('http-capabilities',status==200 and found==modes.get('five-events') and found is not None,
                   status=status)
        status,clip=api(host,'/api/import',fixture.read_bytes(),fixture.name)
        checks.add('http-stereo-upload',status==200 and clip['channels']==2 and clip['duration']==len(x)/48000,status=status)
        status,sep=api(host,'/api/separate',dict(id=clip['id'],mode='five-events',target='guitar'))
        ok=status==200 and sep.get('metadata',{}).get('model')==name
        checks.add('http-accepts-request',ok,status=status,message=sep.get('error'))
        if ok:
            status,mix=api(host,'/api/render',dict(id=clip['id'],separation_id=sep['separation_id']))
            checks.add('http-neutral-render',status==200,status=status)
            downloads=tmp/'downloaded';downloads.mkdir()
            matches={}
            for url_attr,filename,row in (('target_url','target.wav',sep),('residual_url','remaining.wav',sep),('mix_url','remix.wav',mix)):
                status,raw=request(host,row[url_attr]);file=downloads/filename;file.write_bytes(raw)
                matches[filename]=status==200 and (dest/filename).exists() and file_hash(file)==file_hash(dest/filename)
            ok,meta=pipeline(downloads,x)
            checks.add('http-alignment-and-cli-parity',ok and all(matches.values()),bytes_match=matches,**meta)
        status,error=api(host,'/api/separate',dict(id=clip['id'],mode='five-events',target='trombone'))
        checks.add('http-rejects-unknown-label',status==400 and 'error' in error,status=status,message=error.get('error'))
        status,error=api(host,'/api/import',mono.read_bytes(),mono.name)
        checks.add('http-rejects-mono',status==400 and 'error' in error,status=status,message=error.get('error'))
        status,silentclip=api(host,'/api/import',silent.read_bytes(),silent.name)
        status,error=api(host,'/api/separate',dict(id=silentclip['id'],mode='five-events',target='guitar'))
        checks.add('http-refuses-silence',status==400 and 'error' in error and 'target_url' not in error,status=status,message=error.get('error'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','full'])
    stage=parser.parse_args().stage
    checks=Checks(stage+'.json')
    with tempfile.TemporaryDirectory(prefix='voicehunter-r3-cli-http-') as d:
        (preflight if stage=='preflight' else full)(Path(d),checks)
    raise SystemExit(checks.finish())
