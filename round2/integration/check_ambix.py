"""Probe known cardinal plane waves and a real field file through the release interface."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from voicehunter.ambix import decode_ambix

def main():
    size=48000
    source=.12*np.sin(2*np.pi*231*np.arange(size)/48000)
    results={}
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)
        for name,vec in dict(front=(1,0,0,1),left=(1,1,0,0),right=(1,-1,0,0),back=(1,0,0,-1),up=(1,0,1,0)).items():
            raw=(source[:,None]*np.asarray(vec)).astype('<f4').tobytes()
            inp=path/(name+'.wav')
            subprocess.run(['ffmpeg','-v','error','-f','f32le','-ar','48000','-ac','4','-i','pipe:',
                            '-c:a','pcm_f32le',str(inp)],input=raw,check=True)
            actual=decode_ambix(inp,half_angle=45)
            l=.5*(vec[0]+np.cos(np.pi/4)*vec[3]+np.sin(np.pi/4)*vec[1])
            r=.5*(vec[0]+np.cos(np.pi/4)*vec[3]-np.sin(np.pi/4)*vec[1])
            err=float(np.max(np.abs(actual-source[:,None]*[l,r])))
            results[name]=dict(max_error=err,pass_gate=actual.shape==(size,2) and err<1e-6)
        files=sorted((ROOT/'round2_data/field/eigenscape').glob('*.flac'))
        if files:
            a=decode_ambix(files[0]);results['field']=dict(source=files[0].name,frames=len(a),channels=a.shape[1],
                       pass_gate=bool(a.shape[1]==2 and np.max(np.abs(a))>0 and np.isfinite(a).all()))
    print(json.dumps(results,indent=2))
    sys.exit(not all(c['pass_gate'] for c in results.values()))

if __name__=='__main__':main()
