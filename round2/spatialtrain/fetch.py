"""Retrieve the 45 original CIPIC HRIR matrices and its usage conditions."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]


def retrieve(row, directory):
    dest = directory / row["file"]
    def checked(blob):
        return len(blob) == row["size"] and hashlib.sha1(b"blob %d\x00" % len(blob) + blob).hexdigest() == row["git_blob"]
    if dest.is_file() and checked(dest.read_bytes()):
        return row
    for retry in range(4):
        try:
            with urllib.request.urlopen(row["url"], timeout=30) as response:
                blob = response.read()
            if not checked(blob):
                raise ValueError("Git blob checksum or size differs: "+row["file"])
            dest.with_suffix(".part").write_bytes(blob)
            dest.with_suffix(".part").replace(dest)
            return row
        except OSError:
            if retry == 3:
                raise
            time.sleep(retry + 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=BASE/"round2_data/spatial/cipic")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads((HERE/"upstream.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row: retrieve(row, args.output), index))
    for row in index:
        row["sha256"] = hashlib.sha256((args.output/row["file"]).read_bytes()).hexdigest()
    (HERE/"manifest.json").write_text(json.dumps(index, indent=2)+"\n")
    print(f"Verified {len(index)} files ({sum(row['size'] for row in index)} bytes).")
