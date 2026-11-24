"""Lock the final inference parameters and implementation before independent evaluation."""
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
manifest = here / 'freeze.json'
if manifest.exists():
    raise SystemExit('A freeze manifest already exists.')
files = ('round3/transfer/candidate.py', 'round3/transfer/train.py', 'round3/transfer/head.npz',
         'round3/transfer/yamnet.onnx','round3/transfer/yamnet_class_map.csv',
         'round3/transfer/validation.json','round3/transfer/train_sources.json',
         'round2/semantic/semantic_candidate.py','voicehunter/spatial.py')
val = json.loads((here / 'validation.json').read_text())
manifest.write_text(json.dumps(dict(name='transfer',predictor='round3.transfer.candidate:Predictor',
     final_training_count=val['count'], selection=val['chosen'],
     hashes={p:sha256((root/p).read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
