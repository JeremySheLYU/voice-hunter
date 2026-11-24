from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'round2/semantic'))
from round3b.model.encode_train import selected,CACHE
from round3.transfer.candidate import Encoder
from round2.semantic.evaluate import sound
def run():
    enc=Encoder()
    for r in selected():
        slot=CACHE/(r['sha256']+'-prior.npz')
        if slot.exists():continue
        x=sound(r['path']);n=min(len(x),216000);k=(len(x)-n)//2
        y,p=enc.encode(x[k:k+n]);np.savez(slot,y=y,p=p)
if __name__=='__main__':run()
