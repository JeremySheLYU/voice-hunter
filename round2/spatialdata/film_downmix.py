"""Offline BS.775-style fold-down baseline vs fit, train-by-film rather than excerpt."""
from pathlib import Path
import json
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
RAW=HERE.parents[1]/'round2_data/spatial/films'
BASE=np.array([1,2**-.5,2**-.5,0],dtype='f8')


def load(path,channels):
    x=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','f32le','pipe:1'])
    return np.frombuffer(x,'<f4').reshape(-1,channels).astype('f8')


def design(y):
    # Symmetric common coefficients applied to both ears, keeping local polarity.
    return np.array([[y[:,0],y[:,2],y[:,4],y[:,5]],
                     [y[:,1],y[:,2],y[:,5],y[:,4]]]).transpose(2,0,1)


def sisdr(a,r):
    x=a-a.mean(axis=0);ref=r-r.mean(axis=0)
    target=ref*np.sum(x*ref)/np.sum(ref**2)
    return float(10*np.log10(np.sum(target**2)/max(1e-14,np.sum((x-target)**2))))


index=json.loads((HERE/'films_index.json').read_text())
rows=[]; covariance=np.zeros((4,4));cross=np.zeros(4)
for film in ('bbb','sintel'):
    for start in (60,180,300,420):
        ss={r['kind']:r for r in index if r['film']==film and r['start_seconds']==start}
        X=design(load(RAW/ss['51']['file'],6));y=load(RAW/ss['st']['file'],2)
        assert X.shape[:2]==y.shape
        if film=='bbb':
            xx=X.reshape(-1,4); yy=y.ravel()
            covariance+=xx.T@xx;cross+=xx.T@yy
        rows.append((film,start,X,y))
weights=np.linalg.solve(covariance+1e-9*np.trace(covariance)*np.eye(4),cross)
metrics=[]
for film,start,X,y in rows:
    baseline=X@BASE
    fitted=X@weights
    a,b=sisdr(baseline,y),sisdr(fitted,y)
    metrics.append(dict(film=film,start=start,split='train' if film=='bbb' else 'test',
                        fixed_sisdr_db=a,learned_sisdr_db=b,improvement_db=b-a,
                        fixed_zero_lag_cosine=float(np.sum(baseline*y)/np.sqrt(np.sum(y*y)*np.sum(baseline*baseline)))))
summary={fold:float(np.median([r['improvement_db'] for r in metrics if r['split']==fold])) for fold in ('train','test')}
output=dict(rows=metrics,weights=weights.tolist(),baseline=BASE.tolist(),median_improvement_db=summary,
            weight_names=['front','centre','ipsilateral surround','contralateral surround'],lfe_weight=0)
(HERE/'film_downmix_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
