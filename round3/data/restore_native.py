"""Recreate native FLAC excerpts from the frozen preview URLs using FFmpeg."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(blob):return hashlib.sha256(blob).hexdigest()
def main():
    rows=json.loads((HERE/'native.json').read_text())
    for r in rows:
        p=ROOT/r['path']
        if p.exists():
            if sha(p.read_bytes())!=r['sha256']:raise ValueError('Changed existing output: '+str(p))
            continue
        req=urllib.request.Request(r['download_url'],headers={'Range':'bytes=0-819199','User-Agent':'VoiceHunter-public-research/0.3'})
        with urllib.request.urlopen(req,timeout=30) as response:blob=response.read(819200)
        if sha(blob)!=r['download_sha256']:raise ValueError('Changed preview: '+r['id'])
        with tempfile.TemporaryDirectory() as d:
            i=Path(d)/'in.mp3';o=Path(d)/'out.flac';i.write_bytes(blob)
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(i),'-t','15','-map_metadata','-1','-c:a','flac',str(o)],check=True)
            audio=o.read_bytes()
        if sha(audio)!=r['sha256']:raise ValueError('Different FLAC bytes from current FFmpeg: '+r['id'])
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(audio)
    print('Verified',len(rows),'native excerpts.')
if __name__=='__main__':main()
