"""Restore the published AST checkpoint, validating its original byte count and SHA-256."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

HERE=Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def restore():
    spec=json.loads((HERE/'upstream_hash.json').read_text())
    output=HERE/'ast-int8.onnx'
    if output.exists():
        if output.stat().st_size==spec['bytes'] and sha(output)==spec['sha256']:
            print(f"Verified {output.name}");return
        raise ValueError(f"Existing {output} has unexpected contents")
    url=f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=HERE,suffix='.download',delete=False) as f:
            temp=Path(f.name)
            with urllib.request.urlopen(url,timeout=90) as response:
                size=0
                while block:=response.read(1024*1024):
                    size+=len(block)
                    if size>spec['bytes']: raise ValueError("Unexpected checkpoint size")
                    f.write(block)
        if size!=spec['bytes'] or sha(temp)!=spec['sha256']:
            raise ValueError('Checkpoint verification failed')
        temp.replace(output)
        print(f"Restored and verified {output.name}")
    finally:
        if temp:temp.unlink(missing_ok=True)


if __name__=='__main__':restore()
