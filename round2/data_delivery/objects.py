"""Redistribute unchanged source-format ADM conformance examples with author XML, original notices and pinned provenance."""
import json
from pathlib import Path
import tempfile
import build

def object_pack():
    meta=build.read('objects/manifest')
    rows=[]
    license_id='GPL-3.0-or-later, full terms in round2_data/objects/LICENSE.TXT'
    for r in meta['files']:
        desc=('Unchanged source-format PCM or authored XML with ADM time metadata. Shares one 3 s tone across four file variants; no ML use or acoustic-performance inference.' if Path(r['path']).suffix in {'.wav','.xml'} else 'Unchanged upstream notice/documentation.')
        record=dict(collection='EBU_object_conformance',path=r['path'],bytes=r['bytes'],sha256=r['sha256'],credit=meta['copyright'],license=license_id,source=r['original'],description=desc)
        rows.append((record,build.ROOT/r['path']))
    build.EXTRA['EBU_object_conformance']=['round2/objects/manifest.json','round2/objects/README.md']
    return build.pack('ADM_Object_Conformance.zip',rows)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as stage:
        build.STAGE=Path(stage)
        outcome=object_pack()
        (build.META/'objects_pack.json').write_text(json.dumps(outcome,indent=2)+'\n')
        print(outcome)
