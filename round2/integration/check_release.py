"""Run packaged regression entry points and profiles without using private inputs."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from voicehunter.engine import mix_stems

checks=[]
def record(name,ok,details):
    checks.append(dict(test=name,passed=bool(ok),details=details))

for entry in ['tests/validate_audio.py','tests/check_binaural.py','tests/test_multitrack.py',
              'tests/validate_av_sync.py','scripts/speaker_demo.py','round2/integration/check_ambix.py']:
    proc=subprocess.run([sys.executable,entry],cwd=ROOT,text=True,capture_output=True)
    record(entry,proc.returncode==0,proc.stdout[-350:]+proc.stderr[-350:])

sr=48000
t=np.arange(sr*3)/sr
a=(.2*np.sin(2*np.pi*330*t)).astype('f4')
stereo=np.column_stack((a,a))
stereo[sr:2*sr,1]*=-1
zero=np.zeros_like(stereo)
new=mix_stems(stereo,zero,0,0,1,output_mode='headphones-measured',azimuth_deg=0)
old=mix_stems(stereo,zero,0,0,1,output_mode='headphones',azimuth_deg=0)
inside=slice(sr+sr//8,2*sr-sr//8)
outside=slice(sr//8,sr-sr//8)
record('local-polarity-cancellation',np.max(np.abs(old[inside]))<1e-6 and np.max(np.abs(new[inside]-stereo[inside]))<1e-6 and np.max(np.abs(new[outside]-old[outside]))<1e-6,
       {'baseline_rms':float(np.sqrt(np.mean(old[inside]**2))),'measured_rms':float(np.sqrt(np.mean(new[inside]**2)))})
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    f=p/'dom.json'
    proc=subprocess.run([sys.executable,'qa_round1/workbench/run.py','--out',str(f),'--strict'],cwd=ROOT,text=True,capture_output=True)
    if f.exists():
        d=json.loads(f.read_text()); rows=d['dom_http']['checks']
        record('desktop-workbench-http-dom',all(r['status']=='pass' for r in rows) and d['exports']['pass'],
               {'passed':sum(r['status']=='pass' for r in rows),'total':len(rows),'actual_browser':d['browser']['execution'],'individual':rows})
    else:record('desktop-workbench-http-dom',False,proc.stderr[-1500:])
out=Path(__file__).with_name('release.json')
out.write_text(json.dumps(dict(checks=checks),indent=2,ensure_ascii=False))
print([(r['test'],r['passed']) for r in checks])
sys.exit(not all(r['passed'] for r in checks))
