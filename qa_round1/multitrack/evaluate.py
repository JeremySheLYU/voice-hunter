"""Generate known stereo stems, run the released CLI and score only scored overlap."""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from voicehunter.media import SAMPLE_RATE as SR, write_wav

GATE = 2e-6


def read_float(path):
    with open(path, "rb") as handle:
        hdr = handle.read(12)
        assert hdr[:4] == b"RIFF" and hdr[8:] == b"WAVE"
        while True:
            tag, size = struct.unpack("<4sI", handle.read(8))
            if tag == b"fmt ":
                fmt, channels, rate = struct.unpack_from("<HHI", handle.read(size))
                assert (fmt, channels, rate) == (3, 2, SR)
                handle.seek(size % 2, 1)
            elif tag == b"data":
                return np.frombuffer(handle.read(size), dtype="<f4").reshape(-1, 2)
            else:
                handle.seek(size + size % 2, 1)


def scene(kind, seed, n=3, seconds=7):
    rng = np.random.default_rng(seed)
    t = np.arange(SR*seconds) / SR
    sources, tracks = [], []
    overlap = n + .15
    for i in range(n):
        freq = (171 + i*297)*(1 + rng.uniform(-.015, .015))
        if kind == "similar":
            freq = 255 + 2*i
        phase = rng.uniform(0, 2*np.pi, 3)
        inst = np.full(len(t), freq)
        if kind == "changed":
            # Reference identities rotate their note register only at overlap.
            inst[t > overlap] = 171 + ((i+1) % n)*297
        angle = np.cumsum(2*np.pi*inst/SR)
        tonal = sum(amp*np.sin((k+1)*angle+p) for k,(amp,p) in enumerate(zip([.78,.31,.17], phase)))
        active = ((t >= i) & (t < i+.94)) | (t > overlap)
        envelope = .6+.38*np.sin(2*np.pi*(1.13+i*.51)*t+phase[0])
        mono = active*tonal*envelope*.12
        pos = np.pi/4 if kind == "similar" else .23 + 1.08*i/max(1,n-1)
        stereo = np.stack((np.cos(pos)*mono,np.sin(pos)*mono), axis=1)
        sources.append((np.rint(stereo*2**17)/2**17).astype('f4'))
        tracks.append(dict(id=f"object{i+1}", span=[i+.1,i+.84]))
    return sources, {"tracks": tracks}


def sisdr(estimate, reference):
    # Use per-channel DC removal and a single scalar shared by both channels.
    ref = reference.astype("f8") - reference.mean(axis=0, dtype="f8")
    est = estimate.astype("f8") - estimate.mean(axis=0, dtype="f8")
    proj = ref * (np.sum(ref*est) / np.sum(ref*ref))
    return 10*math.log10(max(np.sum(proj*proj),1e-20) / max(np.sum((est-proj)**2),1e-20))


def execute(folder, config, name):
    path = folder / (name+".json")
    path.write_text(json.dumps(config))
    dest = folder/name
    start = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "voicehunter.multitrack", str(folder/"input.wav"),
                             "--config", str(path), "--out", str(dest)],
                            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    error = proc.stderr.read().decode()
    proc.stderr.close()
    rss = usage.ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024)
    return dest, dict(exit=proc.returncode, stderr=error, seconds=round(time.perf_counter()-start,3),
                      peak_cli_rss_MiB=round(rss,1))


def inspect(folder, mix, config, label):
    dest, info = execute(folder, config, label)
    if info["exit"]:
        return {}, dict(**info, valid=False)
    stems = {t["id"]:read_float(dest/(t["id"]+".wav")) for t in config["tracks"]}
    total = read_float(dest/"remaining.wav").astype('f8')
    for a in stems.values():
        total += a
    err = float(np.max(np.abs(total-mix)))
    finite = bool(np.isfinite(total).all())
    remix = subprocess.check_output(["ffprobe","-v","error","-show_streams","-of","json",str(dest/"remix.wav")])
    stream = json.loads(remix)["streams"][0]
    shape_ok = all(a.shape == mix.shape for a in stems.values()) and int(stream["channels"])==2 and int(stream["duration_ts"])==len(mix)
    return stems, dict(**info, valid=finite and shape_ok and err<=GATE,
                       frames=len(mix), sum_error=err, shape_ok=shape_ok, finite=finite)


def run():
    out = dict(cases=[],bounds=[],criteria="criteria.md")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for kind in ["distinct","similar","changed"]:
            for seed in [11,23,47]:
                folder = root/f"{kind}-{seed}"
                folder.mkdir()
                sources, config = scene(kind,seed)
                mix = sum(sources)
                write_wav(folder/"input.wav",mix,stem=True)
                stems, stats = inspect(folder,mix,config,"out")
                rows = []
                s = slice(round(SR*3.5),round(SR*6.5))
                for track,source in zip(config["tracks"],sources):
                    if not stems:
                        break
                    inp = sisdr(mix[s],source[s])
                    est = sisdr(stems[track["id"]][s],source[s])
                    rows.append(dict(id=track["id"],input=inp,output=est,improvement=est-inp))
                out["cases"].append(dict(kind=kind,seed=seed,engineering=stats,measures=rows,
                    all_three_pass=len(rows)==3 and all(r["improvement"]>=3 for r in rows)))
                if (kind,seed)==("distinct",11):
                    rev, second = inspect(folder,mix,dict(tracks=config["tracks"][::-1]),"reverse")
                    delta=max(float(np.max(np.abs(a-rev[k]))) for k,a in stems.items())
                    out["order"]=dict(max_diff=delta,passed=delta<=GATE and second["valid"])
                    bad = dict(tracks=[dict(id="remaining",span=[0.1,0.84])])
                    path, neg = execute(folder,bad,"bad")
                    out["bad_config"]=dict(**neg,passed=neg['exit']!=0 and bool(neg['stderr'].strip()) and not path.exists())
        for n,seconds in [(1,4),(8,60)]:
            folder = root/f"bounds{n}"
            folder.mkdir()
            sources, config = scene("distinct",11,n,seconds)
            mix=sum(sources)
            del sources
            write_wav(folder/"input.wav",mix,stem=True)
            _, meta=inspect(folder,mix,config,"out")
            out["bounds"].append(dict(track_count=n,duration=seconds,**meta))
    path = Path(__file__).with_name("results.json")
    path.write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    run()
