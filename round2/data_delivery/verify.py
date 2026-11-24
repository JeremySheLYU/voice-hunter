"""Verify the selected downloaded/locally saved ZIPs and their member-level source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('names',nargs='*',help='archive names, default all parts plus sampler')
    args=ap.parse_args()
    catalog=json.loads((args.directory/'manifest.json').read_text())
    records={item['name']:item for item in [*catalog['archives'],catalog['sampler'],catalog['optional_format_test']]}
    for name in (args.names or records):
        if name not in records:raise SystemExit('Unknown archive '+name)
        item=records[name]
        with (args.directory/name).open('rb') as stream:
            if sha(stream)!=item['sha256']:raise SystemExit('Archive checksum failed: '+name)
        with ZipFile(args.directory/name) as z:
            assets=json.loads(z.read('notices/ATTRIBUTION.json'))
            if len(assets)!=item['assets']:raise SystemExit('Asset count mismatch: '+name)
            for asset in assets:
                with z.open(asset['path']) as stream:
                    if sha(stream)!=asset['sha256']:raise SystemExit('Source checksum failed: '+asset['path'])
        print(f"{name}: {item['assets']} verified")


if __name__=='__main__':
    main()
