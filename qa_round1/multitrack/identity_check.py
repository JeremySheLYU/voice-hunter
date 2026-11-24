"""Post hoc 3x3 correlation matrices for diagnosing register changes, never relabelled."""
import json
from pathlib import Path
import tempfile

import numpy as np
from evaluate import SR, inspect, scene, write_wav

rows = []
for kind in ["distinct", "changed"]:
    sources, config = scene(kind, 11)
    mix = sum(sources)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_wav(tmp/"input.wav", mix, stem=True)
        stems, _ = inspect(tmp, mix, config, "out")
        crop = slice(round(SR*3.5), round(SR*6.5))
        for track in config["tracks"]:
            a = stems[track["id"]][crop].astype('f8').ravel()
            rms = float(np.sqrt(np.mean(a*a)))
            dot = []
            for source in sources:
                b = source[crop].astype('f8').ravel()
                dot.append(float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))))
            rows.append(dict(scene=kind,seed=11,emitted_id=track["id"],rms=rms,
                             stereo_waveform_cosine_by_true_object=dot))
out=Path(__file__).with_name("identity.json")
out.write_text(json.dumps(rows,indent=2)+"\n")
