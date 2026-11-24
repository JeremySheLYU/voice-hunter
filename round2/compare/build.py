"""One registered held-out speech/train/wind mixture, old/new CLI, matched loudness."""
import argparse
import base64
from html import escape
import io
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import wave
from zipfile import ZipFile

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"environment"))
import train


def measure_loudness(x):
    r=subprocess.run(["ffmpeg","-hide_banner","-nostdin","-f","f32le","-ar",str(train.SR),"-ac","2","-i","pipe:0",
                      "-af","loudnorm=I=-23:TP=-1:LRA=11:print_format=json","-f","null","-"],
                     input=np.asarray(x,"<f4").tobytes(),capture_output=True,check=True)
    info=json.loads(re.findall(rb'\{[^{}]+\}',r.stderr)[-1])
    return float(info['input_i']),float(info['input_tp'])


def encode(x, target):
    x=x.copy()
    length=round(train.SR*.006)
    ramp=np.sin(np.linspace(0,np.pi/2,length))**2
    x[:length]*=ramp[:,None]
    x[-length:]*=ramp[::-1,None]
    before,peak=measure_loudness(x)
    gain=target-before
    if peak+gain>-.98: raise ValueError("Matching would exceed headroom")
    samples=np.rint(x*10**(gain/20)*32767).astype("<i2")
    calibrated=samples.astype("f4")/32768
    after,peak=measure_loudness(calibrated)
    if abs(after-target)>.1:raise ValueError("LUFS tolerance exceeded")
    file=io.BytesIO()
    with wave.open(file,"wb") as wav:
        wav.setparams((2,2,train.SR,len(x),"NONE","not compressed"));wav.writeframes(samples.tobytes())
    data=file.getvalue()
    import hashlib
    return base64.b64encode(data).decode("ascii"),dict(input_lufs=before,gain_db=gain,delivered_lufs=after,
                            true_peak_dbtp=peak,frames=len(x),wav_sha256=hashlib.sha256(data).hexdigest())


def main():
    options=argparse.ArgumentParser(description=__doc__)
    options.add_argument("--legacy-zip",type=Path,default=train.ROOT/"outputs/VoiceHunter_v0.1.zip")
    args=options.parse_args()
    frozen=json.loads((train.HERE/"field_and_named.json").read_text())
    audit=next(c for c in frozen['cases'] if c["trial"]==1 and c["ambient_level_db"] is None)
    by_id={r["id"]:r for r in json.loads((train.HERE/"cohort.json").read_text())}
    items=[by_id[r['asset']] for r in audit['source']]
    if [r["category"] for r in items]!=["speech","train","wind"]:raise ValueError("Unexpected scene")
    assert train.digest(args.legacy_zip)=="eff4673286a17556e8e1d6ebe8633459c786285881aafa036632c037ee1b97c7"
    if train.digest(train.CHECKPOINT)!=frozen["model_sha"]:raise ValueError("Checkpoint changed")
    x,refs,cfg,_=train.scene(items,17840,"delay",False)
    for t,it in zip(cfg["tracks"],items):t["id"]=it["category"]
    cfg.pop("separator",None)
    signals={"原始混音":x,"列車真值":refs[1]}
    scores={}
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        with ZipFile(args.legacy_zip) as z:z.extractall(td/"old")
        train.write_wav(td/"in.wav",x,stem=True)
        (td/"config.json").write_text(json.dumps(cfg))
        for name,root,extra,key in [("v0.1 分軌",td/"old"/"VoiceHunter",[],"baseline"),
                                     ("v0.2 固定方位",train.ROOT,["--separator","adaptive"],"estimate")]:
            output=td/key
            subprocess.run([sys.executable,"-m","voicehunter.multitrack",str(td/"in.wav"),"--config",str(td/"config.json"),
                            "--out",str(output),*extra],cwd=root,check=True,capture_output=True)
            y,_=train.load_media(output/"train.wav")
            signals[name]=y
            score=train.measure(y,refs[1],x);scores[name]=score
            if abs(score["estimated_sisdr"]-audit[key][1]["estimated_sisdr"])>.001:
                raise ValueError("Published scene not reproduced")
    audio={}; levels={}
    for label,sig in signals.items():
        audio[label],levels[label]=encode(sig[train.SCORED],-23)
    out=dict(trial=1,source_order=[r['id'] for r in items],selection="first frozen fresh named test scene, no external bed",
             scored_section_s=[train.SCORED.start/train.SR,train.SCORED.stop/train.SR],sample_rate=train.SR,edge_fade_each_ms=6,
             release_sha256=train.digest(args.legacy_zip),calibration_sha256=frozen["model_sha"],scores=scores,loudness=levels)
    (HERE/"metrics.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    def safe_url(url): return escape(url.replace("http://","https://"),quote=True)
    source_lines=[f'<li>{escape(r["category"])} · <a href="{safe_url(r["url"])}">{escape(r["title"])}</a> · {escape(r["author"])} · <a href="{safe_url(r["license"])}">CC BY 3.0</a></li>' for r in items]
    note=("# 素材出處\n\n三段 CC BY 3.0 錄音均已擷取、混合、調整聲像及響度，部分版本經分軌。權利標示適用於頁面內嵌的聲音及其衍生試聽片段。\n\n"+
          "\n".join(f'- {r["category"]} — {r["title"]}, {r["author"]}. {r["url"]} • {r["license"]}' for r in items)+
          "\n\n取自 FSD50K（Eduardo Fonseca 等人），資料集 CC BY 4.0，https://zenodo.org/records/4060432 。\n")
    (HERE/"CREDITS.md").write_text(note)
    html=(HERE/"template.html").read_text().replace("__AUDIO__",json.dumps(audio,ensure_ascii=False)).replace("__CREDITS__","".join(source_lines))
    html=html.replace("__OLD__",f'{scores["v0.1 分軌"]["improvement"]:+.2f}').replace("__NEW__",f'{scores["v0.2 固定方位"]["improvement"]:+.2f}')
    html=html.replace("__RANGE__",f'{min(v["delivered_lufs"] for v in levels.values()):.2f} 至 {max(v["delivered_lufs"] for v in levels.values()):.2f}')
    (HERE/"index.html").write_text(html)
    print(out["scores"],out["loudness"])
if __name__=="__main__":main()
