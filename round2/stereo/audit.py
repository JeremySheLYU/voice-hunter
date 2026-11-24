"""Probe the downloaded, individually licensed FMA excerpts for actual stereo energy."""
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
records=json.loads((HERE/'manifest.json').read_text())['records']


def check(row):
    p=ROOT/'round2_data/stereo'/row['filename']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-ar','16000','-f','f32le','pipe:1'])
    x=np.frombuffer(raw,'<f4').astype('f8').reshape(-1,row['channels'])
    power=float(np.mean(x*x))
    side_ratio=None
    if x.shape[1]==2:
        side=(x[:,0]-x[:,1])*.5
        side_ratio=float(10*np.log10(max(float(np.mean(side*side))/max(power,1e-28),1e-28)))
    return {'id':row['id'],'split':row['split'],'channels':x.shape[1],'decoded_s':len(x)/16000,
            'has_distinct_stereo':bool(side_ratio is not None and side_ratio > -60 and power>1e-10),
            'side_to_mean_channel_power_db':side_ratio}

with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(check,records))
summary={'artists':len(set(row['artist_id'] for row in records)), 'sources':len(records),
         'channel_counts':dict(collections.Counter(row['channels'] for row in rows)),
         'files_with_measured_stereo_difference':sum(r['has_distinct_stereo'] for r in rows),
         'split_counts':dict(collections.Counter(row['split'] for row in records)),
         'decoded_seconds':sum(row['decoded_s'] for row in rows),'bytes':sum(row['bytes'] for row in records)}
(HERE/'audit.json').write_text(json.dumps(dict(summary=summary,records=rows),indent=2)+'\n')
print(summary)
