from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
root=here.parents[1]
files=['round3/mixture/'+x for x in ('model.py','gateway.py','jointdiag.py','mix_model.py','masker.py','gateway_params.json','mix_head.npz','train_mix.py','calibrate_gateway.py')]
files+=['round3/transfer/'+x for x in ('candidate.py','head.npz','yamnet.onnx')]
files+=['round2/semantic/cohort.json','round3/data/train.json']
config=json.loads((here/'gateway_params.json').read_text())
out=dict(name='mixture',predictor='round3.mixture.gateway:Predictor',module='round3.mixture.model:predict_scene',
         deployment='Research candidate; geometry gate is part of all reported results',
         training_sources=625,training_mix_examples=1600,validation=config['validation'],
         files={key:hashlib.sha256((root/key).read_bytes()).hexdigest() for key in files})
(here/'freeze.json').write_text(json.dumps(out,indent=2)+'\n')
