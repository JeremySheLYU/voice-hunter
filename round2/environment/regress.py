"""Re-run the locked v0.1 register/format suite, preserving its original results."""
from pathlib import Path
import importlib.util
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
script=ROOT/"qa_round1/multitrack/evaluate.py"
spec=importlib.util.spec_from_file_location("vh_reg",script)
suite=importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
dest=HERE/"legacy_registers"
dest.mkdir(exist_ok=True)
suite.__file__=str(dest/script.name)
original_execute=suite.execute
def explicit(folder, config, name):
    return original_execute(folder, dict(config,separator="adaptive"), name)
suite.execute=explicit
suite.run()
old=json.loads((script.parent/"results.json").read_text())
new=json.loads((dest/"results.json").read_text())
def summary(data):
    out={}
    for c in data["cases"]:
        key=c["kind"]
        out.setdefault(key,[]).append([round(v["improvement"],3) for v in c["measures"]])
    return dict(groups=out,order=data["order"],bounds=data["bounds"],bad=data["bad_config"])
(dest/"comparison.json").write_text(json.dumps(dict(baseline=summary(old),candidate=summary(new)),indent=2)+"\n")
