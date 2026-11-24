"""Repeat all frozen v0.1 cases under unchanged config and explicit adaptive mode."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENTRY=ROOT/"qa_round1/mixtures/evaluate.py"
spec=importlib.util.spec_from_file_location("legacy_practical",ENTRY)
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
original_build=prior.build


def run(cache,fetch=False):
    dest=HERE/"legacy_practical"
    for adaptive in [False,True]:
        def build(*args):
            stems,mix,config,section=original_build(*args)
            if adaptive:config["separator"]="adaptive"
            return stems,mix,config,section
        prior.build=build
        sys.argv=[str(ENTRY),"--product-root",str(ROOT),"--out",str(dest if adaptive else dest/"default"),
                  "--cache",str(cache)]+(["--fetch"] if fetch else [])
        prior.main()
    for name in ["default","adaptive"]:
        path=dest/"default" if name=="default" else dest
        print(name,json.loads((path/"summary.json").read_text())["scene_pass_count"])
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--fetch",action="store_true")
    args=parser.parse_args();run(args.cache,args.fetch)
