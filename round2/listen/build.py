"""Create an offline level-matched comparison on two attributed CC BY excerpts."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import wave

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from voicehunter.binaural import render_headphones

RATE=48000
SECONDS=4.75
VOICE_HASH="0473ba9613befa1026db5e254546f105aeffba8366b5468fc5e8d474d844ca1d"


def decode(path,channels,start):
    data=subprocess.check_output(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(start),
              "-i",str(path),"-t",str(SECONDS),"-vn","-ar",str(RATE),"-ac",str(channels),"-f","f32le","-"])
    x=np.frombuffer(data,dtype="<f4").reshape((-1,channels)).copy()
    assert len(x)==int(RATE*SECONDS)
    return x


def db(x):
    return 20*np.log10(np.sqrt(np.mean(x.astype("f8")**2)))


def wav(x):
    with io.BytesIO() as b:
        with wave.open(b,"wb") as f:
            f.setparams((2,2,RATE,len(x),"NONE","not compressed"))
            f.writeframes(np.round(np.clip(x,-1,1)*32767).astype("<i2").tobytes())
        return b.getvalue()


def meter(data):
    proc=subprocess.run(["ffmpeg","-nostdin","-hide_banner","-i","pipe:0","-af",
              "loudnorm=I=-23:TP=-1:LRA=7:print_format=json","-f","null","-"],
               input=data,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
    text=proc.stderr.decode()
    return json.loads(re.findall(r'\{\s*"input_i".*?\}',text,re.S)[-1])


def build(voice,bed):
    assert hashlib.sha256(voice.read_bytes()).hexdigest()==VOICE_HASH
    v=decode(voice,1,1.0)
    b=decode(bed,2,3.0)
    v*=10**((-21-db(v))/20)
    b*=10**((-30-db(b))/20)
    nfade=int(RATE*.035)
    for x in (v,b):
        x[:nfade]*=np.linspace(0,1,nfade)[:,None]
        x[-nfade:]*=np.linspace(1,0,nfade)[:,None]
    target=np.repeat(v,2,axis=1)
    clips={"ref":target+b}
    for mode in ("analytic","measured"):
        for side,angle in (("L",-45),("R",45)):
            clips[mode+side]=render_headphones(target,b,RATE,profile=mode,azimuth_deg=angle)
    embed={}
    results={}
    for key, signal in clips.items():
        m=meter(wav(signal))
        gain=-23.-float(m["input_i"])
        signal=signal*10**(gain/20)
        assert np.max(np.abs(signal))<1
        data=wav(signal)
        check=meter(data)
        results[key]=dict(wav_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
              adjustment_db=gain,integrated_lufs=float(check["input_i"]),true_peak_dbtp=float(check["input_tp"]))
        embed[key]=base64.b64encode(data).decode("ascii")
    values=[v['integrated_lufs'] for v in results.values()]
    assert max(values)-min(values)<=.1
    html=(HERE/"template.html").read_text().replace("__MATERIAL__",json.dumps(embed,separators=(",",":")))
    (HERE/"index.html").write_text(html)
    assert (HERE/"index.html").stat().st_size<8_000_000
    report=dict(sampling_hz=RATE,seconds=SECONDS,voice_excerpt=[1,1+SECONDS],bed_excerpt=[3,3+SECONDS],
                dry_voice_sha256=VOICE_HASH,bed_sha256=hashlib.sha256(bed.read_bytes()).hexdigest(),
                asset=results,within_spread_lu=max(values)-min(values),human_ratings=0,
                voice_origin="https://raw.githubusercontent.com/librosa/data/5285c94e33c74bbe97501f5c9fa95e5e2ce55d48/audio/198-209-0000.ogg",
                bed_origin="https://doi.org/10.5281/zenodo.18362685",bed_member="W06.wav")
    (HERE/"calibration.json").write_text(json.dumps(report,indent=2)+"\n")
    print((HERE/"index.html").stat().st_size,report["within_spread_lu"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--voice",type=Path,required=True)
    p.add_argument("--bed",type=Path,default=ROOT/"round2_data/spatial/satp/W06.wav")
    a=p.parse_args();build(a.voice,a.bed)
