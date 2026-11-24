"""Render one predetermined held-out capture into three time-aligned, nominally level-matched audition files."""
import json
import subprocess
import numpy as np
from fit_width import HERE, DATA, decode, input_stereo
from render_width import render

name='20220205_Sat_08h30_Pos1_30sec.wav.flac'
x,rate=decode(DATA/'ihta_foa'/('FirstOrderAmbisonics_'+name),4)
y,ear_rate=decode(DATA/'ihta_ears'/('Binaural_'+name),2)
assert rate==ear_rate==48000
xy=input_stereo(x)
cfg=json.loads((HERE/'width_results.json').read_text())['conditions']['fitted_bands']
variants={'virtualXY':xy,'learned':render(xy,rate,cfg['gain_db']),'published_decode':y}
def loudness(sig):
    out=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-f','f32le','-ar',str(rate),'-ac','2','-i','pipe:0','-af','loudnorm=print_format=json','-f','null','-'],input=sig.astype('<f4').tobytes(),capture_output=True,check=True)
    log=out.stderr.decode()
    return json.loads(log[log.rindex('{'):log.rindex('}')+1])
target=min(-24,*(float(loudness(x)['input_i']) - float(loudness(x)['input_tp']) - 1 for x in variants.values()))
folder=HERE/'demo';folder.mkdir(exist_ok=True)
results={}
for k,v in variants.items():
    gain=10**((target-float(loudness(v)['input_i']))/20)
    dst=folder/(k+'.flac')
    subprocess.run(['ffmpeg','-y','-nostdin','-loglevel','error','-f','f32le','-ar',str(rate),'-ac','2','-i','pipe:0','-c:a','flac',str(dst)],input=(v*gain).astype('<f4').tobytes(),check=True)
    final,_=decode(dst,2)
    results[k]={'gain_db':20*np.log10(gain),'LUFS':float(loudness(final)['input_i']),'dBTP':float(loudness(final)['input_tp'])}
meta={'source':name,'original_creators':'Josep Llorca-Bofí, Jonas Heck, Christian Dreier, Michael Vorländer','source_url':'https://zenodo.org/records/11184517','license':'CC BY 4.0','modifications':'Opening 4s, virtual ±20° stereo, derived learned side-gain render, and original published binaural decode, each scaled by a frame-invariant gain to equal integrated loudness.','measurements':results}
(folder/'attribution-and-levels.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
print(results)
