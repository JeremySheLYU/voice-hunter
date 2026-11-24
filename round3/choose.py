"""Rank independently frozen models using *only* the predeclared historical validation scores."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
ORDER=['classifier','apple_features','transfer','mixture','transformer','fusion']
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def choose():
    ranking=[]
    for name in ORDER:
        path=ROOT/name/'freeze.json'
        s=read(path)
        if name=='classifier':v=s['validation']['metrics']
        elif name=='transfer':v=s['selection']['metrics']
        elif name=='mixture':v=s['validation']
        elif name=='transformer':
            v=s.get('metrics') or s.get('validation') or s.get('selection')
            if 'metrics' in v:v=v['metrics']
        else:v=s['metrics']
        m=[v[k] for k in ('persistent','intermittent')]
        floors=[min(x['accuracy'],x['f1']) for x in m]
        def useful(x):
            if 'useful_fraction' in x:return x['useful_fraction']
            return x['useful']/x['n'] if name=='mixture' else x['useful']
        ranking.append(dict(name=name,worst_detection=min(floors),worst_useful=min(useful(x) for x in m),
                           validation=v,freeze_sha256=digest(path)))
    ranking.sort(key=lambda e:(-e['worst_detection'],-e['worst_useful'],ORDER.index(e['name'])))
    lock=dict(primary=ranking[0]['name'],ranking=ranking,
              rule='Use largest minimum of accuracy and F1 over both validation conditions, then largest minimum useful fraction; simpler backend breaks ties.',
              fresh_source_plan=digest(ROOT/'evaluation/fresh-plan.json'))
    dest=ROOT/'evaluation/selection.json'
    if dest.exists() and read(dest)!=lock:raise RuntimeError('Selection already locked')
    dest.write_text(json.dumps(lock,indent=2,ensure_ascii=False))
    print([(r['name'],r['worst_detection'],r['worst_useful']) for r in ranking])
if __name__=='__main__':choose()
