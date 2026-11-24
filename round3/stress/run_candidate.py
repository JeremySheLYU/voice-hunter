"""Run one frozen candidate on the two preregistered out-of-condition tests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARTIFACTS={
    'classifier':['classifier/model.py','classifier/weights.npz'],
    'apple_features':['apple_features/model.py','apple_features/weights.npz'],
    'transfer':['transfer/candidate.py','transfer/yamnet.onnx','transfer/head.npz'],
    'mixture':['mixture/model.py','mixture/weights.npz'],
    'fusion':['fusion/model.py','fusion/calibration.npz','transfer/head.npz','transfer/yamnet.onnx','apple_features/model.py','apple_features/weights.npz'],
    'room_transfer':['mixture/transfer_fusion.py','mixture/model.py','transfer/candidate.py','transfer/yamnet.onnx','transfer/head.npz'],
}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('candidate',help='Local candidate name, also used in output filename')
p.add_argument('--module',help='Predictor alias or module import path if different from candidate name')
p.add_argument('--file',type=Path,action='append',help='Override list with the actual version-locked source/weights paths')
p.add_argument('--fresh',action='store_true',help='Run on additional dry stereo corpus')
a=p.parse_args()
module_alias=a.module or ('round3.mixture.transfer_fusion' if a.candidate=='room_transfer' else a.candidate)
model_files=a.file or [ROOT/'round3'/s for s in ARTIFACTS.get(a.candidate,[])]
if not model_files:p.error('Supply --file for every frozen artifact used by this candidate')
flags=[v for f in model_files for v in ['--model-file',str(f)]]
env={**os.environ,'VOICEHUNTER_STRESS_CANDIDATE':module_alias,'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
if a.fresh:
    subprocess.run([sys.executable,str(HERE/'evaluate_fresh_native.py'),'--model',module_alias,*flags],env=env,check=True)
else:
    subprocess.run([sys.executable,str(HERE/'evaluate_stress.py'),'--method','round3.stress.candidate_adapter',
                    '--output',str(HERE/f'{a.candidate}.json'),*flags],env=env,check=True)
