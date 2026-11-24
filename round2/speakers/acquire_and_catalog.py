"""Fetch and inventory the three public LibriSpeech archives used for this revision."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from evaluate import duration
from train import HERE,sha

ASSETS=[
    ('train-clean-5',31,'5df7d4e78065366204ca6845bb08f490'),
    ('dev-clean-2',31,'6d7ab67ac6a1d2c993d050e16d61080d'),
    ('test-clean',12,'32fa31d27d2e1cad72775fee3f4849a9')]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',type=Path,required=True)
    cache=ap.parse_args().cache;cache.mkdir(parents=True,exist_ok=True)
    report=[]; rows=[]
    for split,namespace,official_md5 in ASSETS:
        archive=cache/(split+'.tar.gz')
        url=f'https://www.openslr.org/resources/{namespace}/{archive.name}'
        if not archive.exists():
            part=archive.with_suffix('.part');urllib.request.urlretrieve(url,part);part.replace(archive)
        with archive.open('rb') as src:
            digest=hashlib.file_digest(src,'md5').hexdigest()
        if digest!=official_md5:
            raise ValueError(f'Download checksum mismatch: {split}, expected {official_md5} vs {digest}')
        target=cache/'LibriSpeech'/split
        nstart=len(rows)
        with tarfile.open(archive, mode='r|gz') as bundle:
            for member in bundle:
                if not member.isfile() or not member.name.startswith('LibriSpeech/'+split+'/') or not member.name.endswith('.flac'):
                    continue
                data=bundle.extractfile(member).read()
                packed=int.from_bytes(data[18:26],'big')
                if data[:4]!=b'fLaC' or packed>>44==0:raise ValueError('Bad FLAC header')
                seconds=(packed & ((1<<36)-1))/(packed>>44)
                reader=Path(member.name).stem.split('-')[0]
                rows.append((split,member.name,reader,round(seconds,5),len(data),hashlib.sha256(data).hexdigest()))
        print(f'{split} verified, {len(rows)-nstart} real FLAC entries.',flush=True)
        new=rows[nstart:]
        report.append({'partition':split,'clips':len(new),'readers':len(set(item[2] for item in new)),
                       'seconds':sum(r[3] for r in new),'url':url,'bytes':archive.stat().st_size,'archive_md5':digest,
                       'archive_sha256':sha(archive),'source_md5_index':f'https://www.openslr.org/resources/{namespace}/md5sum.txt',
                       'license':'CC BY 4.0','channels':'mono (originals)','official_page':f'https://www.openslr.org/{namespace}/'})
    with (HERE/'librispeech-files.csv').open('w',newline='') as f:
        out=csv.writer(f);out.writerow(['partition','path','reader','duration_s','bytes','sha256']);out.writerows(rows)
    (HERE/'corpus.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['partition','clips','readers','seconds','bytes']} for r in report]))


if __name__=='__main__':main()
