"""Replay unchanged v0.3 acoustic-stress scorers with the frozen v0.3b model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STRESS = ROOT / 'round3/stress'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(STRESS))
import evaluate_fresh_native as native
import evaluate_stress as measured

MODULE = 'round3b.model.model'
LOCK = ROOT / 'round3b/model/freeze.json'


def verified_files():
    record=json.loads(LOCK.read_text())
    files=[]
    for relative,expected in record['hashes'].items():
        path=ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('A frozen model artifact changed: '+relative)
        files.append(path)
    return [LOCK,*files]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('condition',choices=('native','room'))
    args=p.parse_args()
    files=verified_files()
    if args.condition=='room':
        measured.run(SimpleNamespace(freeze=False,condition=['room'],method=MODULE,model_file=files,output=HERE/'room.json'))
    else:
        # The original stress driver has a hardcoded output name. Change only
        # its final save destination, leaving the locked protocols and scorer intact.
        original_save=native.save
        native.save=lambda _, report: original_save(HERE/'native.json',report)
        native.evaluate(SimpleNamespace(model=MODULE,model_file=files))
