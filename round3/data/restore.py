"""Restore the frozen, rights-attributed audio from pinned FSD50K URLs."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE=Path(__file__).resolve().parent
def download(row,root):
    directory=root/'round3_data'/'environment'/row['category']
    directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(row['id']+'.wav')
    if destination.is_file():
        if hashlib.sha256(destination.read_bytes()).hexdigest()==row['sha256']:
            return
        raise ValueError(f'Unexpected existing file at {destination}. Preserve or remove it first.')
    for attempt in range(3):
        try:
            rq=urllib.request.Request(row['download_url'],headers={'User-Agent':'VoiceHunter-public-research/0.3'})
            with urllib.request.urlopen(rq,timeout=30) as r: data=r.read(row['bytes']+1)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError('Remote length or SHA-256 mismatch for '+row['id'])
            temp=destination.with_suffix('.part')
            temp.write_bytes(data);temp.replace(destination)
            return
        except OSError:
            if attempt==2:raise
            time.sleep(1+attempt)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=HERE.parents[1],help='Destination project root')
    p.add_argument('--split',choices=['train','test','both'],default='both')
    p.add_argument('--workers',type=int,default=12)
    args=p.parse_args()
    rows=[]
    for role in (('test','train') if args.split=='both' else (args.split,)):
        rows.extend(json.loads((HERE/(role+'.json')).read_text()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda row:download(row,args.root),rows))
    print('Verified',len(rows),'original audio files.')
if __name__=='__main__':main()
