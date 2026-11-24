"""Descriptive replay of previously reported v0.3 scenes by the frozen v0.3b model."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import itertools
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('round3b_runner',HERE/'run.py')
rb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rb)
ev=rb.ev


def main(workers):
    root=HERE.parents[1]
    model_freeze=root/'round3b/model/freeze.json'
    if not model_freeze.is_file():
        raise ValueError('The challenger has not been frozen.')
    old_plan=root/'round3/evaluation/fresh-plan.json'
    source_plan=ev.read(old_plan)
    proof=dict(classification='reused v0.3 test; regression only',model_sha256=ev.digest(model_freeze),
             original_plan=ev.digest(old_plan),runner=ev.digest(Path(__file__)))
    pred=ev.predictor('round3b.model.model')
    out=HERE/'regression';out.mkdir(exist_ok=True)
    summaries={}
    for condition in ev.CONDITIONS:
        path=out/(condition+'.json')
        if not path.exists():
            scenes=ev.fresh_scenes(source_plan,condition)
            rows=[]
            with ThreadPoolExecutor(workers) as pool:
                while chunk:=list(itertools.islice(scenes,2*workers)):
                    for batch in pool.map(lambda item:ev.score_scene(pred,item),chunk):rows+=batch
            ev.write(path,dict(origin=proof,condition=condition,**ev.summary(rows),rows=rows))
        result=ev.read(path)
        if result['origin']!=proof:raise ValueError('A different regression already exists')
        reference=ev.read(root/'round3/evaluation/fresh/fusion'/(condition+'.json'))
        summaries[condition]=dict(challenger=result['summary'],previous_primary=reference['summary'],
                                  paired_difference=rb.paired.compare(result['rows'],reference['rows']))
        print(condition,result['summary'],flush=True)
    ev.write(out/'summary.json',dict(origin=proof,conditions=summaries,
                                    eligible_as_fresh_confirmation=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=1)
    main(p.parse_args().workers)
