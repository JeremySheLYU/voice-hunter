"""Restore pinned primary mono WAVs and native-stereo preview excerpts from the manifest."""
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def restore(r):
    rel=Path(r['path']); dest=(ROOT/rel).resolve()
    if not dest.is_relative_to((ROOT/'round2_data/environment').resolve()): raise ValueError('Unexpected destination')
    if dest.exists() and sha(dest.read_bytes())==r['sha256']:return 'cached'
    headers={'User-Agent':'VoiceHunter-audio-research/0.2'}
    stereo=r['representation']=='stereo_preview'
    if stereo:headers['Range']='bytes=0-819199'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(r['download_url'],headers=headers),timeout=35) as res:
                data=res.read(819200 if stereo else 3_000_001)
            if sha(data)!=r['download_sha256']:raise ValueError('Upstream SHA-256 changed')
            if stereo:
                with tempfile.TemporaryDirectory() as tmp:
                    original=Path(tmp)/'input.mp3';target=Path(tmp)/'clip.flac'
                    original.write_bytes(data)
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(original),'-t','15','-map_metadata','-1','-c:a','flac',str(target)],check=True)
                    data=target.read_bytes()
            if sha(data)!=r['sha256']:raise ValueError('Local asset checksum differs; see reference FFmpeg version')
            dest.parent.mkdir(exist_ok=True,parents=True)
            stage=dest.with_suffix(dest.suffix+'.part');stage.write_bytes(data);stage.replace(dest)
            return 'downloaded'
        except (OSError,ValueError) as e:
            if attempt==2:raise
            time.sleep(1.5*(attempt+1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--all-versions',action='store_true',help='restore both representations for any original ID selected twice')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--manifest',type=Path,help='exact JSON list of original selected FSD source/representation rows')
    a=ap.parse_args(); names=['manifest.json','stereo_candidates.json'] if a.all_versions else ['all_sources.json']
    rows=json.loads(a.manifest.read_text()) if a.manifest else [row for name in names for row in json.loads((HERE/name).read_text())]
    failures=[];counts={'cached':0,'downloaded':0}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(restore,row):row['id'] for row in rows}
        for future in cf.as_completed(futures):
            try: counts[future.result()]+=1
            except Exception as e:failures.append([futures[future],str(e)])
    print(json.dumps({'assets':len(rows),**counts,'failures':failures},indent=2))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
