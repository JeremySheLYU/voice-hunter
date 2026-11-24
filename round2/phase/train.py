"""Frozen threshold search/cross-artist test on CC0/CC BY FMA. Outputs compact numeric model."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0, str(ROOT))
from voicehunter.phase_guard import window_power, fallback_curve

RATE=16000
GRID=[.03,.06,.10,.16,.24]


def features(row):
    path=ROOT/'round2_data/stereo'/row['filename']
    x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-ss','5','-t','7','-map','0:a:0',
                    '-ar',str(RATE),'-f','f32le','pipe:1']),dtype='<f4').reshape(-1,2)
    m=x.mean(axis=1)
    if len(m)<6*RATE or np.mean(m.astype('f8')**2)<1e-7:return None
    centers, en, rr=window_power(x,RATE)
    s=(x[:,0]-x[:,1])*.5
    a=1+(int(row['sha256'][:2],16)/255-.5)*.7
    # Small independent side signal creates a spread of genuine fold severities.
    s *= min(.42, .3*np.std(m)/max(np.std(s),1e-8))
    y=np.column_stack((m+s, a*(m-s)))
    segment=slice(2*RATE, round(3.3*RATE))
    y[segment,int(row['sha256'][-1],16)%2]*=-1
    _,ce,cr=window_power(y,RATE)
    crop=(centers>2.06*RATE)&(centers<3.24*RATE)&(ce>1e-7)
    if not crop.any() or not (en>1e-7).any():return None
    ordinary=y.mean(axis=1)
    baseline=np.column_stack((ordinary,ordinary))
    scored=slice(round(2.06*RATE),round(3.24*RATE))
    expected=np.mean(y[scored].astype('f8')**2)
    def err(z):
        return abs(10*np.log10(max(np.mean(z[scored].astype('f8')**2),1e-16)/expected))
    measures={}
    for th in GRID:
        w=fallback_curve(y,RATE,th)[:,None]
        measures[str(th)]=dict(false_fraction=float(np.mean(rr[en>1e-7]<th)),recall=float(np.mean(cr[crop]<th)),
                               baseline_energy_error_db=err(baseline),guarded_energy_error_db=err(baseline*(1-w)+y*w))
    return dict(id=row['id'],artist_id=row['artist_id'],split=row['split'],measure=measures)

def main():
    manifest=json.loads((HERE.parent/'stereo/manifest.json').read_text())
    cases=[]
    for row in manifest['records']:
        if row['channels']==2:
            case=features(row)
            if case:cases.append(case)
    def group(split,threshold):
        rows=[c['measure'][str(threshold)] for c in cases if c['split']==split]
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]} | {'recordings':len(rows)}
    eligible=[t for t in GRID if group('train',t)['false_fraction']<=.01]
    if not eligible:raise SystemExit('No feasible training threshold')
    th=max(eligible,key=lambda t:(group('train',t)['recall'],-t))
    sums={s:group(s,th) for s in ['train','validation','test']}
    ok=sums['validation']['false_fraction']<=.02 and sums['validation']['recall']>=.80
    model=dict(threshold=th,validation_gate=ok,training_source='FMA small',training_records=sums['train']['recordings'],
               provenance="round2/phase/ and round2/stereo/manifest.json")
    (HERE/'result.json').write_text(json.dumps(dict(model=model,summary=sums,grid=GRID,per_recording=cases),indent=2)+'\n')
    if ok:(ROOT/'voicehunter/models/phase_retention.json').write_text(json.dumps(model,indent=2)+'\n')
    print(json.dumps(sums,indent=2),th,ok)

if __name__=='__main__':main()
