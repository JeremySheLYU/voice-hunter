"""Independent functional qualification through public CLI, HTTP and archived v0.1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from zipfile import ZipFile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
rows=[]
def check(name,passed,**data):
    rows.append(dict(test=name,passed=bool(passed),**data))
    print(('PASS ' if passed else 'FAIL ')+name,flush=True)
def sha(data):return hashlib.sha256(data).hexdigest()
def cli(src,out,mode,angle=0,app=ROOT,span=(.2,1.65),bg=(2.2,3.65),extra=(),measured=True):
    args=[sys.executable,'-m','voicehunter','process',str(src),'--out',str(out),'--mode',mode,
        '--span',*map(str,span),'--background',*map(str,bg),'--output-mode','headphones-measured' if measured else 'stereo',
        '--azimuth',str(angle),*extra]
    return subprocess.run(args,cwd=app,capture_output=True,text=True,timeout=95)
def pcm(path):
    data=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:a:0','-ar','48000','-f','f32le','pipe:1'])
    return np.frombuffer(data,'<f4').reshape(-1,2)
def audit_export(d,src):
    names=sorted(p.name for p in d.glob('*'))
    x,t,b,y=[pcm(p) for p in [src,d/'target.wav',d/'remaining.wav',d/'remix.wav']]
    if any(v.shape!=x.shape for v in [t,b,y]):return False,{'names':names}
    err=float(np.max(abs(t.astype(np.float64)+b-x))); finite=bool(np.isfinite(t).all() and np.isfinite(b).all() and np.isfinite(y).all())
    return err<=2e-6 and finite and names==['remaining.wav','remix.wav','settings.json','target.wav'],dict(names=names,sum_error=err,frames=len(x),finite=finite)
def api(base,url,data=None,headers=None):
    req=Request(base+url.lstrip('/'),data=data,headers=headers or {})
    try:
        with urlopen(req,timeout=90) as f:return f.status,f.read()
    except HTTPError as e:return e.code,e.read()
def post(base,url,data):
    status,body=api(base,url,json.dumps(data).encode(),{'Content-Type':'application/json'})
    return status,json.loads(body)
def main():
    corpus=json.loads((ROOT/'round2/envdata/all_sources.json').read_text())
    public=next(r for r in corpus if r['id']=='386515')
    src=ROOT/public['path']
    cohort={r['id'] for r in json.loads((ROOT/'round2/environment/cohort.json').read_text())}
    check('field-source-rights-and-holdout',sha(src.read_bytes())==public['sha256'] and public['split']=='test' and public['id'] not in cohort and ('/by/' in public['license'] or '/zero/' in public['license']) and public['signal_layout'].startswith('two-channel with'),id=public['id'],title=public['title'],author=public['author'],license=public['license'],source=public['url'],hash=public['sha256'],signal=public['channel_check_12khz'])
    models={name:sha((ROOT/'voicehunter/models'/name).read_bytes()) for name in ['joint_spatial.json','speaker-distance.json','cipic_horizontal_v1.json','phase_retention.json']}
    train=json.loads((ROOT/'round2/environment/fit.json').read_text())
    test=json.loads((ROOT/'round2/environment/test.json').read_text())['cases']
    rawcells=dict(train=train['train'],validation=train['validation'],test=test)
    audit_groups={};receipt={r['id']:r for r in json.loads((ROOT/'round2/environment/cohort.json').read_text())}
    valid=True
    verify_raw='--verify-raw' in sys.argv
    raw_digests={}
    def disk_matches(path,expected):
        if not verify_raw:return True
        if path not in raw_digests:raw_digests[path]=sha((ROOT/path).read_bytes())
        return raw_digests[path]==expected
    for split,cases in rawcells.items():
        cells=[r for c in cases for r in c['recordings']]
        ids={r['asset'] for r in cells}
        auth={r['author'].strip().casefold() for r in cells if 'author' in r}
        for r in cells:
            if r['asset'] not in receipt:
                key=Path(r['asset']); provenance=ROOT/'round2/fielddata/receipts'/key.parent/(key.name+'.json')
                bed=json.loads(provenance.read_text())
                valid &= (bool(r.get('bed')) and bed['sha256']==r['sha256'] and bed['license'].startswith('CC-BY') and disk_matches(r['path'],r['sha256']))
                continue
            record=receipt[r['asset']]
            valid &= (record['split']==split and record['sha256']==r['sha256'] and
                      disk_matches(record['path'],r['sha256']) and
                      ('/zero/' in record['license'] or '/by/' in record['license']))
        audit_groups[split]=(ids,auth)
    overlap={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        overlap[a+'_'+b]={'ids':len(audit_groups[a][0]&audit_groups[b][0]),'authors':len(audit_groups[a][1]&audit_groups[b][1])}
        valid &= max(overlap[a+'_'+b].values())==0
    check('actual-fitted-and-scored-source-disjointness',valid,receipts=True,rehash_all_media=verify_raw,overlaps=overlap,unique_sources={k:len(v[0]) for k,v in audit_groups.items()})
    sys.path.insert(0,str(ROOT))
    from voicehunter.engine import mix_stems
    rate=48000; tt=np.arange(rate)/rate;mono=.15*np.sin(2*np.pi*779*tt)
    target=np.stack((mono,mono),axis=1).astype('f4');target[15360:32640,1]*=-1
    placed=mix_stems(target,np.zeros_like(target),0,0,1,output_mode='headphones-measured',azimuth_deg=90)
    keep=slice(20000,27200)
    retention=float(np.sqrt(np.mean(placed[keep]**2)/np.mean(target[keep]**2)))
    check('measured-phase-guard-retains-pocket',retention>=.95 and np.isfinite(placed).all(),rms_ratio=retention)
    with tempfile.TemporaryDirectory(prefix='vh-independent-',dir='/tmp') as temp:
        tmp=Path(temp); duet=tmp/'duet.wav'
        seed=(ROOT/'examples/duet.wav').read_bytes()
        assert sha(seed)=='15b6a51604a75ff221499b92f3eb747e5e10260bfe96551bd33e6e855ee8d9f8'
        duet.write_bytes(seed)
        # Normal producer interface on known two-reference speaker demo.
        for mode in ['speakers','spatial-example']:
            q=tmp/mode;cmd=cli(duet,q,mode,0)
            ok,info=audit_export(q,duet) if not cmd.returncode else (False,{'stderr':cmd.stderr[:400]})
            check('cli-bundled-'+mode,ok,**info)
        movingdir=tmp/'moving';movecmd=cli(duet,movingdir,'speakers_moving')
        moveok=(movecmd.returncode==0)
        ok,info=audit_export(movingdir,duet) if moveok else (not movingdir.exists() and bool(movecmd.stderr.strip()),{'reason':movecmd.stderr.strip()[:320]})
        check('cli-moving-speakers',ok,returned=moveok,**info)
        plus=tmp/'right';r=cli(duet,plus,'spatial-example',90)
        d=tmp/'spatial-example'
        if r.returncode==0 and d.is_dir():
            ok,info=audit_export(plus,duet)
            stem_same=all((d/n).read_bytes()==(plus/n).read_bytes() for n in ['target.wav','remaining.wav'])
            delta=float(np.max(abs(pcm(d/'remix.wav')-pcm(plus/'remix.wav'))))
            check('cli-measured-0-to-90',ok and stem_same and delta>1e-3,stems_identical=stem_same,max_render_delta=delta,**info)
        else:check('cli-measured-0-to-90',False,stderr=r.stderr[:400])
        fdir=tmp/'actual';f=cli(src,fdir,'spatial-example',90,span=(.2,1.25),bg=(2,3.2))
        accepted=f.returncode==0
        if accepted: good,info=audit_export(fdir,src)
        else:good,info=not fdir.exists() and bool(f.stderr.strip()),dict(refusal=f.stderr.strip()[:350])
        check('field-cli-functional',good,returned=accepted,**info)
        wrongdir=tmp/'overlap';neg=cli(duet,wrongdir,'spatial-example',span=(0,2),bg=(1,2.5))
        check('cli-overlap-no-partial',neg.returncode!=0 and not wrongdir.exists(),error=neg.stderr.strip()[:260])
        # Normal CLI archived edition processes identical source with the old example method.
        archive=ROOT/'outputs/VoiceHunter_v0.1.zip'
        if archive.is_file():
            with ZipFile(archive) as z:z.extractall(tmp/'v01')
            old=tmp/'v01/VoiceHunter';od=tmp/'earlier';new=tmp/'newref'
            a=cli(duet,od,'example',app=old,measured=False);b=cli(duet,new,'example',measured=False)
            hashes=all((od/n).read_bytes()==(new/n).read_bytes() for n in ['target.wav','remaining.wav','remix.wav']) if a.returncode==b.returncode==0 else False
            check('same-default-example-waveforms-v01',hashes,archive_sha=sha(archive.read_bytes()),old_status=a.returncode,new_status=b.returncode)
        # Known v0.1 unsupported video remux: preserve and report any fresh partial files.
        vp=tmp/'camera.webm'
        subprocess.run(['ffmpeg','-y','-nostdin','-v','error','-f','lavfi','-i','color=black:s=120x80:r=10','-i',str(duet),'-c:v','libvpx','-c:a','libopus','-t','5',str(vp)],check=True)
        bad=tmp/'vp8';cmd=cli(vp,bad,'spatial-example',extra=['--video'])
        partial={p.name:p.stat().st_size for p in bad.glob('*')}
        if cmd.returncode==0:
            check('vp8-webm-video-or-clean-failure','remix.mov' in partial and partial['remix.mov']>0,returncode=cmd.returncode,files=partial)
        else:
            check('vp8-webm-video-or-clean-failure',not partial,returncode=cmd.returncode,files=partial,error=cmd.stderr.strip()[:180])
        server_temp=tmp/'server-temp';server_temp.mkdir()
        proc=subprocess.Popen([sys.executable,'-u','-m','voicehunter','serve','--port','0'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env=os.environ|{'TMPDIR':str(server_temp)})
        try:
            line=proc.stdout.readline().strip();base=line[line.index('http://'):]
            status,body=api(base,'/api/capabilities');caps=json.loads(body)
            ids=[r['id'] for r in caps.get('modes',[])]
            check('http-enumerate-new-mode',status==200 and 'spatial-example' in ids,modes=ids)
            status,body=api(base,'/api/import',src.read_bytes(),{'X-Filename':src.name,'Content-Type':'application/octet-stream'})
            clip=json.loads(body);check('http-import-original-stereo',status==200 and clip.get('channels')==2,status=status)
            params=dict(id=clip['id'],mode='spatial-example',start=.2,end=1.25,background_start=2,background_end=3.2)
            code,sp=post(base,'/api/separate',params)
            check('http-field-cli-agreement',(code==200)==accepted and code in [200,400],status=code,returned=accepted)
            if code==200:
                stem_ok=True
                for role,key in [('target','target_url'),('remaining','residual_url')]:
                    code2,body=api(base,sp[key]);stem_ok&=code2==200 and body==(fdir/(role+'.wav')).read_bytes()
                check('http-actual-downloaded-stems',stem_ok)
                code2,mix=post(base,'/api/render',dict(id=clip['id'],separation_id=sp['separation_id'],output_mode='headphones-measured',azimuth_deg=90))
                dl,body=api(base,mix['mix_url'])
                check('http-measured-render-download-matches-cli',code2==dl==200 and body==(fdir/'remix.wav').read_bytes(),bytes=len(body))
            err,j=post(base,'/api/separate',params|{'background_start':.3})
            check('http-reject-overlap',err==400 and bool(j.get('error')) and 'separation_id' not in j,status=err)
            code,demo=api(base,'/api/demo?clip=duet');demo=json.loads(demo)
            if 'id' in demo:
                code,sp=post(base,'/api/separate',dict(id=demo['id'],mode='speakers',start=.2,end=1.65,background_start=2.2,background_end=3.65))
                check('http-speakers-demo',code==200,status=code)
                code,j=post(base,'/api/separate',dict(id=demo['id'],mode='speakers_moving',start=.2,end=1.65,background_start=2.2,background_end=3.65))
                check('http-moving-speakers-agrees-cli',(code==200)==moveok and code in [200,400] and 'speakers_moving' in ids,status=code)
            else:check('http-speakers-demo',False,error=demo)
            vpstatus,body=api(base,'/api/import',vp.read_bytes(),{'X-Filename':'camera.webm','Content-Type':'application/octet-stream'})
            vc=json.loads(body)
            spstatus,vsp=post(base,'/api/separate',dict(id=vc['id'],mode='spatial-example',start=.2,end=1.65,background_start=2.2,background_end=3.65))
            mrstatus,vmix=post(base,'/api/render',dict(id=vc['id'],separation_id=vsp['separation_id'],output_mode='headphones-measured',azimuth_deg=90))
            exc,vres=post(base,'/api/video',dict(id=vc['id'],render_id=vmix['render_id']))
            leftover=sorted(x.suffix for x in server_temp.rglob('*') if x.is_file() and x.suffix in ['.part','.mov'])
            check('http-vp8-export-clean-failure',vpstatus==spstatus==mrstatus==200 and exc==400 and not leftover and 'video_url' not in vres,
                  statuses=[vpstatus,spstatus,mrstatus,exc],partial_suffixes=leftover)
        finally:
            proc.terminate();proc.wait(timeout=6)
    summary={'checks':len(rows),'pass':sum(r['passed'] for r in rows),'models':models,'results':rows}
    (HERE/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(summary['pass'],'/',summary['checks'])
    sys.exit(summary['checks']!=summary['pass'])
if __name__=='__main__':main()
