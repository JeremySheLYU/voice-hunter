"""Run the original 54-case regression harness through the optional candidate."""
import importlib.util
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from voicehunter.speech import separate_moving_speakers

spec=importlib.util.spec_from_file_location('original_suite',root/'qa_round1/speakers/evaluate.py')
suite=importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.separate_speakers=separate_moving_speakers
if __name__=='__main__':suite.main()
