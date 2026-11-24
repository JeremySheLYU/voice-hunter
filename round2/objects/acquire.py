"""Retrieve author-generated ADM jump-position test files and check AXML, CHNA and PCM."""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
from xml.etree import ElementTree as ET

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'round2_data/objects'
SHA='ed42d4ede70172d80cd4402c1674062350377f68'
REPO='ebu/ear-production-suite'
BASE=f'https://raw.githubusercontent.com/{REPO}/{SHA}/'
DIR='reaper-adm-extension/test/reaper_adm/data/'
NAMES=['jp_1_no_jump_position','jp_2_jump_position_no_interpolate','jp_3_jump_position_interpolate','jp_4_zero_length']


def fetch(path,tree):
    dest=OUT/path.split('/')[-1]
    if not dest.exists():
        data=urllib.request.urlopen(BASE+path,timeout=22).read()
    else:
        data=dest.read_bytes()
    check=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert tree[path]==check,path
    if not dest.exists():
        dest.write_bytes(data)
    return dest,data


def probe(data):
    assert data[:4]==b'RIFF' and data[8:12]==b'WAVE'
    pos=12
    chunks={}
    while pos+8<=len(data):
        tag,n=struct.unpack_from('<4sI',data,pos)
        pos+=8;chunks[tag]=data[pos:pos+n];pos+=n+(n%2)
    code,channels,rate,_,_,bits=struct.unpack_from('<HHIIHH',chunks[b'fmt '])
    assert (code,channels,rate,bits)==(1,1,48000,16)
    pcm=chunks[b'data']
    y=np.frombuffer(pcm,'<i2').astype(float)/32768
    tree=ET.fromstring(chunks[b'axml'].rstrip(b'\0'))
    for node in tree.iter():node.tag=node.tag.rsplit('}',1)[-1]
    audio_objects=tree.findall('.//audioObject')
    ids={o.text.strip() for node in audio_objects for o in node.findall('audioTrackUIDRef')}
    tracks,n_uids=struct.unpack_from('<HH',chunks[b'chna'])
    chna_uid=chunks[b'chna'][6:18].rstrip(b'\0').decode()
    assert tracks==n_uids==1 and chna_uid in ids
    positions=[]
    for channel in tree.iter('audioChannelFormat'):
        if channel.get('typeDefinition')!='Objects':continue
        for block in channel.iter('audioBlockFormat'):
            jump=block.find('jumpPosition')
            positions.append({'id':block.get('audioBlockFormatID'),'start':block.get('rtime'),'duration':block.get('duration'),
                              'coordinates':{n.get('coordinate'):float(n.text) for n in block.findall('position')},
                              'jump':None if jump is None else {'value':jump.text,'interpolationLength':jump.get('interpolationLength')}})
    assert len(positions)>=2 and all(p['coordinates'] for p in positions)
    peakbin=int(np.argmax(abs(np.fft.rfft(y))))
    hz=peakbin*rate/len(y)
    assert np.sqrt(np.mean(y*y))>.1 and abs(hz-440)<1
    return dict(container='RIFF WAVE with axml and chna',channels=channels,rate=rate,seconds=len(y)/rate,
                pcm_sha256=hashlib.sha256(pcm).hexdigest(),pcm_rms=float(np.sqrt(np.mean(y*y))),dominant_hz=hz,
                chna_uid=chna_uid,objects=[n.get('audioObjectID') for n in audio_objects],position_blocks=positions)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    index=json.load(urllib.request.urlopen(f'https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1',timeout=20))
    tree={r['path']:r['sha'] for r in index['tree']}
    files=[]
    for path in ['COPYRIGHT.TXT','LICENSE.TXT',DIR+'README-jump_position_test_files.txt']+[DIR+n+ext for n in NAMES for ext in ['.xml','.wav']]:
        f,v=fetch(path,tree)
        row={'path':str(f.relative_to(HERE.parents[1])),'original':BASE+path,'sha256':hashlib.sha256(v).hexdigest(),
             'git_blob':tree[path],'bytes':len(v)}
        if f.suffix=='.wav':row['audio']=probe(v)
        files.append(row)
    report={'source_repository':'https://github.com/'+REPO,'commit':SHA,
            'license':'GPL-3.0-or-later; independently licensed technical fixtures from upstream, preserved with original WAV/XML and license for standalone conformance use only; excluded from training and listening masters.',
            'copyright':'2019–2026 British Broadcasting Corporation, 2020–2021 European Broadcasting Union, 2019–2021 Institut für Rundfunktechnik GmbH.',
            'category':'authored ADM object automation / 440 Hz technical test, no Dolby claim','usage':'format conformance probe only, excluded from training and held-out quality scores','files':files}
    (HERE/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
