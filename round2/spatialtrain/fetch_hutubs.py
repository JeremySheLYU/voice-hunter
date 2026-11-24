"""Download CC BY 4.0 measured HUTUBS HRIRs from the SOFA repository."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
BASE = "https://sofacoustics.org/data/database/hutubs/"
CREDIT = "F. Brinkmann, M. Dinakaran, R. Pelzer, J. J. Wohlgemuth, F. Seipel, S. Weinzierl, et al.; TU Berlin, 2019. DOI 10.14279/depositonce-8487."


def retrieve(request):
    dest,url,expected = request
    for attempt in range(3):
        try:
            if not dest.exists():
                with urllib.request.urlopen(url,timeout=30) as req:
                    data=req.read()
                if not data.startswith(b"\x89HDF") and not data.startswith(b"%PDF"):
                    raise ValueError(f"Unexpected response {dest.name}")
                dest.with_suffix(".part").write_bytes(data)
                dest.with_suffix(".part").replace(dest)
            blob=dest.read_bytes()
            digest=hashlib.sha256(blob).hexdigest()
            if expected and expected!=digest:
                raise ValueError(f"Hash mismatch in {dest}")
            return dict(file=dest.name,url=url,sha256=digest,bytes=len(blob))
        except OSError:
            if attempt==2:raise
            time.sleep(2+attempt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=HERE.parents[1]/"round2_data/spatial/hutubs")
    args=p.parse_args()
    args.data.mkdir(parents=True,exist_ok=True)
    manifest=HERE/"hutubs_manifest.json"
    prior={r["file"]:r["sha256"] for r in json.loads(manifest.read_text())["files"]} if manifest.exists() else {}
    names=["Documentation.pdf"]+[f"pp{i}_HRIRs_measured.sofa" for i in range(1,97)]
    requests=[(args.data/n,BASE+n,prior.get(n)) for n in names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(retrieve,requests))
    manifest.write_text(json.dumps(dict(credit=CREDIT,license="CC BY 4.0",doi="10.14279/depositonce-8487",
                                       primary_terms=BASE+"Documentation.pdf",files=rows),indent=2)+"\n")
    print(len(rows),sum(r["bytes"] for r in rows))
