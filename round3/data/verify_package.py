"""Verify delivered audio bytes against the three fixed, attributed manifests."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    total=0
    files=[]
    for source in ('train','test','native'):
        manifest=HERE/(source+'.json')
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()==(HERE/(source+'.ready')).read_text().strip(),source
        for r in json.loads(manifest.read_text()):
            path=ROOT/r['path']
            assert path.is_file() and path.stat().st_size==r['bytes'],r['id']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'],r['id']
            files.append(r['path'])
            total+=r['bytes']
    control=ROOT/'round3/unknowns/manifest.json'
    assert hashlib.sha256(control.read_bytes()).hexdigest()==json.loads((HERE/'delivery.json').read_text())['negative_control_manifest_sha256']
    for r in json.loads(control.read_text()):
        path=ROOT/r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'],r['id']
        files.append(r['path']);total+=path.stat().st_size
    assert len(files)==len(set(files))
    print('Verified',len(files),'audio files;',total,'bytes.')
if __name__=='__main__':main()
