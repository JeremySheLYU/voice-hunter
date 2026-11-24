"""Fetch and index the official SATP / AIR archives. No model or site is contacted at runtime."""
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import urllib.request
import wave
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1]/'round2_data/spatial'
AIR = 'https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip'
SATP = 'https://zenodo.org/api/records/18362685/files/SATP%20WAV.zip/content'


def sha(b):return hashlib.sha256(b).hexdigest()


def archive(url,p,expected,algo):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(url,timeout=90) as response,p.with_suffix('.part').open('wb') as out:
            shutil.copyfileobj(response,out,1024*1024)
        tmp=p.with_suffix('.part')
        with tmp.open('rb') as f:assert hashlib.file_digest(f,algo).hexdigest()==expected
        tmp.replace(p)
    with p.open('rb') as f:assert hashlib.file_digest(f,algo).hexdigest()==expected
    return p


def wavinfo(b):
    with wave.open(io.BytesIO(b)) as w:
        return dict(channels=w.getnchannels(),sr=w.getframerate(),frames=w.getnframes(),bits=w.getsampwidth()*8,
                    sha256=sha(b))


def air():
    p=RAW/'aachen'
    zpath=archive(AIR,p/'original.zip',json.loads((HERE/'sources.json').read_text())['aachen']['archive_sha256'],'sha256')
    rows=[]; unpaired=[]
    with zipfile.ZipFile(zpath) as z,zipfile.ZipFile(io.BytesIO(z.read('AIR_1_4/AIR_wav_files.zip'))) as inner:
        (HERE/'AIR_LICENSE.txt').write_bytes(z.read('AIR_1_4/license.txt'))
        pairs=defaultdict(dict)
        for name in sorted(inner.namelist()):
            if not name.lower().endswith('.wav'):continue
            if name.startswith('air_phone_'):
                info=wavinfo(inner.read(name))
                unpaired.append(dict(member=name,**info));continue
            room,channel,condition=Path(name).stem.removeprefix('air_').split('_',2)
            pairs[room,condition][channel]=name
        for (room,condition),channels in sorted(pairs.items()):
            assert set(channels)=={'0','1'}
            in_names=[channels['1'],channels['0']]
            inputs=[inner.read(n) for n in in_names]
            arrays=[]; infos=[]
            for b in inputs:
                info=wavinfo(b);infos.append(info)
                assert (info['channels'],info['sr'],info['bits'])==(1,48000,24)
                with wave.open(io.BytesIO(b)) as w:
                    arrays.append(np.frombuffer(w.readframes(w.getnframes()),dtype='u1').reshape(-1,3))
            stem=f'{room}_{condition}'
            dest=p/'stereo'/(stem+'.wav')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with wave.open(str(dest),'wb') as w:
                w.setparams((2,3,48000,len(arrays[0]),'NONE','not compressed'))
                w.writeframes(np.stack(arrays,axis=1).tobytes())
            info=wavinfo(dest.read_bytes())
            row=dict(id=stem,room=room,source_members_LR=in_names,source_sha256_LR=[i['sha256'] for i in infos],
                     file=str(dest.relative_to(RAW)),head=condition.startswith('1_'),split='external-room-eval',**info)
            if room=='stairway':
                head,dist,angle,method=condition.split('_')
                row.update(distance_m=int(dist),az_deg_right=int(angle)-90,measurement=method)
                if head=='1' and dist=='3' and int(angle)-90 in [-45,0,45] and method=='logsweep':
                    shutil.copyfile(dest,p/f'stairway_az{int(angle)-90:+d}.wav')
            rows.append(row)
    result=dict(source=AIR,paired_binaural_or_headless=rows,other_phone_single_channel_wavs=unpaired)
    (HERE/'aachen_index.json').write_text(json.dumps(result,indent=2)+'\n')


def satp():
    p=RAW/'satp'
    zpath=archive(SATP,p/'original.zip','1592b4ac667b2944ad3d412212dc669a','md5')
    source=json.loads((HERE/'sources.json').read_text())['satp']
    rows=[]
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if not n.endswith('.wav'):continue
            name=Path(n).name
            b=z.read(n)
            target=p/name
            target.write_bytes(b)
            code=target.stem
            item=source['locations'].get(code)
            info=wavinfo(b)
            assert (info['channels'],info['sr'],info['bits'])==(2,48000,24 if item else 16)
            rows.append(dict(id=code,type='binaural_field' if item else 'calibration_tone',
                       location=item,split='display-demo-only' if code=='W06' else 'external-field-reserve',
                       file=str(target.relative_to(RAW)),archive_member=n,**info))
    assert sum(r['type']=='binaural_field' for r in rows)==27
    (HERE/'satp_index.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    air()
    satp()
