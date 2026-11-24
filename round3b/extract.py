"""Extract with the follow-up research model after restoring its AST checkpoint."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from voicehunter.media import load_media, write_wav, SAMPLE_RATE
from round3b.model.model import predict_scene
from round2.semantic.semantic_candidate import CLASSES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--target',required=True,choices=CLASSES)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.exit(2,'請指定新的輸出資料夾。\n')
    audio,info=load_media(a.input)
    try:stem=predict_scene(audio,SAMPLE_RATE)[a.target]
    except (ValueError,RuntimeError) as e:p.exit(2,str(e)+'\n')
    if stem is None:p.exit(2,'此錄音沒有可可靠交付的指定目標。\n')
    if stem.shape!=audio.shape or not np.isfinite(stem).all():p.exit(2,'音軌未通過完整性檢查。\n')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as stage:
        folder=Path(stage)/'output';folder.mkdir()
        write_wav(folder/'target.wav',stem,stem=True)
        write_wav(folder/'remaining.wav',audio-stem,stem=True)
        folder.rename(a.out)

if __name__=='__main__':main()
