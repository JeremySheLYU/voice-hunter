"""Verify independent groups, file hashes, licenses and decoded frame lengths."""
import csv
import hashlib
import json
from pathlib import Path
import wave
from collections import Counter
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def author(r):return r['author'].strip().casefold()
def main():
    freeze=json.loads((HERE/'freeze.json').read_text())
    assert sha((HERE/'plan.json').read_bytes())==freeze['sha256']
    historic=[r for p in freeze['excluded_historic'] for r in json.loads((ROOT/p).read_text())]
    seen_ids={r['id'] for r in historic}
    seen_authors={author(r) for r in historic}
    seen_sha={r['sha256'] for r in historic}
    seen_pcm=set()
    for p in {ROOT/r['path'] for r in historic if r['path'].endswith('.wav')}:
        with wave.open(str(p),'rb') as w:seen_pcm.add(sha(w.readframes(w.getnframes())))
    rows=[]
    summary={}
    for role in ('test','train'):
        p=HERE/(role+'.json'); cohort=json.loads(p.read_text())
        assert sha(p.read_bytes())==(HERE/(role+'.ready')).read_text().strip()
        authors={author(r) for r in cohort}
        assert not authors & seen_authors,(authors&seen_authors)
        seen_authors|=authors
        for r in cohort:
            file=ROOT/r['path']
            assert r['id'] not in seen_ids and r['sha256'] not in seen_sha and r['pcm_sha256'] not in seen_pcm
            assert r['split']==role and file.is_file() and file.stat().st_size==r['bytes'] and sha(file.read_bytes())==r['sha256']
            assert any(v in r['license'] for v in ('/licenses/by/','/publicdomain/zero/'))
            with wave.open(str(file),'rb') as wav:
                assert (wav.getnchannels(),wav.getnframes(),wav.getframerate())==(r['channels'],r['frames'],r['sample_rate'])
                assert sha(wav.readframes(wav.getnframes()))==r['pcm_sha256']
            seen_ids.add(r['id']);seen_sha.add(r['sha256']);seen_pcm.add(r['pcm_sha256'])
        rows.extend(cohort)
        summary[role]={'files':len(cohort),'uploaders':len(authors),'classes':dict(Counter(r['category'] for r in cohort)),
          'seconds':sum(r['duration'] for r in cohort),'bytes':sum(r['bytes'] for r in cohort),
          'licenses':dict(Counter(r['license'] for r in cohort)),'manifest_sha256':sha(p.read_bytes())}
    (HERE/'audit.json').write_text(json.dumps({'checks':'passed','reference_unique_uploaders':len({author(r) for r in historic}),'reference_unique_ids':len({r['id'] for r in historic}),'cohorts':summary},indent=2)+'\n')
    with (HERE/'attribution.csv').open('w') as f:
        cols=['id','category','author','title','url','license','split','duration','path','download_url','sha256','pcm_sha256']
        out=csv.DictWriter(f,cols,extrasaction='ignore');out.writeheader();out.writerows(rows)
    print(summary)
if __name__=='__main__':main()
