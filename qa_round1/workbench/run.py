"""Run round-1 workbench QA against a new temporary local server. Requires node, ffmpeg, ffprobe, numpy."""
from contextlib import contextmanager
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def run(args):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True).stdout

@contextmanager
def isolated_server(fixture=None):
    args = [sys.executable, "-u", "-m", "voicehunter", "serve", "--port", "0"]
    if fixture:
        args.extend(["--demo-file", str(fixture)])
    p = subprocess.Popen(args, stdout=subprocess.PIPE, cwd=ROOT, text=True)
    try:
        line = p.stdout.readline().strip()
        if "http://127.0.0.1:" not in line:
            raise RuntimeError("Isolated server failed to start: " + line)
        yield line[line.index("http://127.0.0.1:"):]
    finally:
        p.terminate()
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: p.kill(); p.wait()

def decode(path):
    raw = run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "2",
               "-ar", 48000, "-f", "f32le", "pipe:1"])
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)

def probe(path):
    return json.loads(run(["ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,channels,sample_rate,duration", "-of", "json", path]))["streams"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--strict", action="store_true", help="exit nonzero on failed checks")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="voicehunter-qa-") as d:
        tmp = Path(d)
        mov = tmp / "訪談 QA.mov"
        # Known WAV mapped into an original moving-image MOV, with no private content.
        run(["ffmpeg","-y","-nostdin","-v","error","-f","lavfi","-i","testsrc2=s=160x96:r=12:d=9.6",
             "-i",ROOT/"examples"/"studio.wav","-map","0:v:0","-map","1:a:0","-c:v","mpeg4","-q:v","6",
             "-c:a","pcm_s16le","-shortest",mov])
        mono=tmp/"mono.wav"
        run(["ffmpeg","-y","-v","error","-i",ROOT/"examples"/"studio.wav","-t","1","-ac","1",mono])
        with isolated_server(mov) as base:
            env={**os.environ,"VH_QA_BASE":base,"VH_QA_DIR":str(tmp),"VH_QA_VIDEO":str(mov),"VH_QA_MONO":str(mono)}
            for name in ("browser","dom"):
                subprocess.run(["node",str(HERE/(name+".cjs"))], env=env, cwd=ROOT, check=True)
        with isolated_server() as clean_base:
            no_phone_env={**env,"VH_QA_BASE":clean_base}
            subprocess.run(["node",str(HERE/"dom.cjs"),"--no-phone"], env=no_phone_env, cwd=ROOT, check=True)
        browser=json.loads((tmp/"browser.json").read_text())
        dom=json.loads((tmp/"dom.json").read_text())
        dom["checks"].extend(json.loads((tmp/"no-phone.json").read_text())["checks"])
        raw, fore, back, mixed, muxed = (decode(tmp/n) for n in ("original.wav","target.wav","residual.wav","mix.wav","video.mov"))
        n = min(len(mixed),len(muxed)); corr=float(np.corrcoef(mixed[:n],muxed[:n])[0,1]); origcorr=float(np.corrcoef(raw[:n],muxed[:n])[0,1])
        err=float(np.max(np.abs(raw-fore-back)))
        report={"ran":datetime.datetime.now(datetime.timezone.utc).isoformat(),"fixture":"Synthetic stereo studio.wav + generated 160x96 color pattern MOV; the optional iPhone demo route was configured with the same synthetic MOV","isolated_server":"ephemeral localhost port and temporary clip storage","browser":browser,"dom_http":dom,
                "exports":{"original_samples_per_channel":len(raw)//2,"target_samples_per_channel":len(fore)//2,"residual_samples_per_channel":len(back)//2,"remix_samples_per_channel":len(mixed)//2,"muxed_decoded_samples_per_channel":len(muxed)//2,"max_abs_additive_error":err,"remix_vs_original_rms_difference":float(np.sqrt(np.mean((mixed-raw)**2))),"decoded_mov_vs_downloaded_mix_correlation":corr,"decoded_mov_vs_original_correlation":origcorr,"mov_streams":probe(tmp/"video.mov"),"wav_streams":probe(tmp/"mix.wav")}}
        report["exports"]["pass"]=(len(raw)==len(fore)==len(back)==len(mixed) and err<1e-5 and corr>.97 and corr>origcorr+.0001 and report["exports"]["remix_vs_original_rms_difference"]>.001 and {s["codec_type"] for s in report["exports"]["mov_streams"]}=={"video","audio"})
        args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps({"browser":browser.get("execution"),"dom_http": [(r["id"],r["status"]) for r in dom["checks"]], "exports":report["exports"]["pass"]},ensure_ascii=False))

        if args.strict and (not report["exports"]["pass"] or any(r["status"] == "fail" for r in dom["checks"]) or any(not r["passed"] for r in browser["checks"])):
            sys.exit(1)

if __name__=="__main__": main()
