"""Create an engineering fixture from two explicitly credited downstream training clips."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def create():
    cohort = json.loads((ROOT / 'round2/semantic/cohort.json').read_text())
    source_ids = ('15753', '5997')
    records = []
    parts = []
    for source_id, angle in zip(source_ids, (12, 72)):
        row = next(r for r in cohort if r['id'] == source_id and r['split'] == 'train')
        file = ROOT / row['path']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256']
        wave = np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(file),
            '-ss','0.4','-t','4.5','-ac','1','-ar','48000','-f','f32le','pipe:1']), dtype='<f4')
        assert len(wave) == 216000
        gain = .09 / max(np.sqrt(np.mean(wave.astype('f8')**2)), 1e-8)
        parts.append(wave[:,None] * gain * np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))]))
        records.append({**{k:row[k] for k in ('id','title','author','url','license','sha256','category')},
                        'start_seconds':.4, 'duration_seconds':4.5,'equal_power_pan_degrees':angle,
                        'split':'historical training'})
    mix = np.asarray(sum(parts), dtype='<f4')
    mix *= min(1.,.9/np.max(np.abs(mix)))
    path = HERE/'guitar_wind.flac'
    subprocess.run(['ffmpeg','-v','error','-y','-nostdin','-f','f32le','-ar','48000','-ac','2','-i',
                    'pipe:0','-sample_fmt','s32','-compression_level','8',str(path)],input=mix.tobytes(), check=True)
    (HERE/'fixture.json').write_text(json.dumps(dict(sources=records,samples=216000,sample_rate=48000,
        channels=2,requested='guitar',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        description='Non-evaluation plumbing fixture; cropped, RMS-matched, panned, mixed and FLAC encoded.'),indent=2)+'\n')


if __name__ == '__main__':
    create()
