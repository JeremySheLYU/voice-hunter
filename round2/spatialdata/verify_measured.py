"""Score Voice Hunter's existing speech mode on unseen physically recorded BinauRec tests."""
import argparse
import json
from pathlib import Path
import sys
import hashlib

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from voicehunter.engine import separate_clip
from voicehunter.media import load_media


def energy(x):
    return np.mean(x.astype('f8')**2)


def sdri(ref, estimate):
    ref=ref.astype('f8')-ref.mean(axis=0)
    estimate=estimate.astype('f8')-estimate.mean(axis=0)
    projection=ref*np.sum(estimate*ref)/np.sum(ref*ref)
    return float(10*np.log10(max(1e-18,energy(projection))/max(1e-18,energy(projection-estimate))))


def test(folder, scene):
    ref, noise, recorded = [load_media(folder/(k+'.wav'))[0] for k in ['target','noise_ssn','mixture_ssn']]
    assert ref.shape==noise.shape==recorded.shape
    summed=ref+noise
    out=dict(scene=scene['scene'],utterance_id=scene['utterance_id'],scenario=scene['scenario'],
             nominal_db=scene['nominal_snr_db'],duration=len(ref)/48000,
             closure_db=float(10*np.log10(energy(recorded)/max(1e-18,energy(recorded-summed)))))
    for key,mixture in [('recorded',recorded),('software_added',summed)]:
        target,residual,_=separate_clip(mixture,'rnnoise','voice')
        assert target.shape==residual.shape==mixture.shape
        inp=sdri(ref,mixture); score=sdri(ref,target)
        out[key]=dict(input_db=inp,output_db=score,improvement_db=score-inp,
                      additive_max=float(np.max(abs(target+residual-mixture))))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'round2_data/spatial/binaurec')
    args=parser.parse_args()
    plan=json.loads((HERE/'binaurec_plan.json').read_text())['scenes']
    rows=[test(args.cache/s['scene'],s) for s in plan]
    def group(rows,key):
        val=[r[key]['improvement_db'] for r in rows]
        return dict(n=len(rows),median_db=float(np.median(val)),range_db=[min(val),max(val)],
                    positive=sum(x>0 for x in val),at_least_1db=sum(x>=1 for x in val),
                    max_additive_error=max(r[key]['additive_max'] for r in rows))
    detail={k:group(rows,k) for k in ['recorded','software_added']}
    detail['by_snr']={str(db):group([r for r in rows if r['nominal_db']==db],'recorded') for db in [-5,0,5]}
    detail['closure_range_db']=[min(r['closure_db'] for r in rows),max(r['closure_db'] for r in rows)]
    detail['closure_median_db']=float(np.median([r['closure_db'] for r in rows]))
    detail['model_sha256']=hashlib.sha256((ROOT/'voicehunter/models/std.rnnn').read_bytes()).hexdigest()
    (HERE/'binaurec_result.json').write_text(json.dumps(dict(summary=detail,rows=rows),indent=2)+'\n')
    print(json.dumps(detail,indent=2))
