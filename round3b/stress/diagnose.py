"""Count decisions at the already-frozen geometry gate. No threshold is modified."""
import json
from pathlib import Path
from run import HERE, STRESS, ROOT, native, measured, verified_files
from round3.mixture.jointdiag import image_pair


def frozen_native():
    plan=native.read(native.PLAN)
    lookup={r['id']:r for r in plan['sources']}
    for record in plan['scenes']:
        a,b=[native.native_clip(str(ROOT/lookup[record[k]]['path'])) for k in ('source_id','background_id')]
        total=a+b
        yield total*min(1.,.8/(float(abs(total).max())+1e-9))


def audit(audio_stream):
    passed=0
    no_pair=0
    rejected=0
    for audio in audio_stream:
        try:
            _,quality=image_pair(audio,True)
        except ValueError:
            no_pair+=1
            continue
        if quality[0]>.16 or quality[1]<.05:
            rejected+=1
        else:
            passed+=1
    return dict(total=passed+no_pair+rejected,gate_passed=passed,no_image_pair=no_pair,geometry_quality_rejected=rejected)


if __name__=='__main__':
    verified_files()
    scene_file=STRESS/'scenes.json'
    manifest=json.loads(scene_file.read_text())
    report={'native':audit(frozen_native()),'room':audit(x for x,_,_ in measured.scene_iter('room',manifest))}
    (HERE/'geometry.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
