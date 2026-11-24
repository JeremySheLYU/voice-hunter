"""No synthetic panning: mix held-out native stereo captures exactly as recorded."""
import json
from pathlib import Path
import tempfile
import numpy as np
import train


def main():
    source_list=json.loads((train.HERE/"native_inventory.json").read_text())
    frozen=json.loads((train.HERE/"cohort.json").read_text())
    dev=[r for r in frozen if r["split"]!="test"]
    authors={r["author"].casefold().strip() for r in dev}
    ids={r["id"] for r in dev}
    allowed=[];not_used=[]
    for r in source_list:
        if r["author"].casefold().strip() in authors or r["id"] in ids: raise ValueError("Split conflict")
        if train.digest(train.ROOT/r["path"])!=r["sha256"]: raise ValueError("changed input")
        _,h,_=train.excerpt(train.ROOT/r["path"],2)
        diff=10*np.log10(max(1e-15,np.mean(((h[:,0]-h[:,1])*.5)**2)) / np.mean(h**2))
        if diff<=-50:
            not_used.append(dict(id=r["id"],reason="nearly duplicate channels",side_ratio_db=float(diff)));continue
        allowed.append(r)
    cases=[]
    for i,picked in enumerate(train.groups(allowed,min(12,len(allowed)),90231,group_size=3)):
        x=np.zeros((round((train.START+train.OVERLAP)*train.SR),2),np.float32)
        truth=[]; desc=[];config={"tracks":[]}
        for j,r in enumerate(picked[:3]):
            solo,held,offsets=train.excerpt(train.ROOT/r["path"],2)
            gain=.045/np.sqrt(np.mean(held**2))
            source=np.zeros_like(x);start=round(j*1.15*train.SR)
            source[start:start+len(solo)]=solo*gain;source[round(train.START*train.SR):]=held*gain
            truth.append(source);x+=source;desc.append(dict(id=r["id"],author=r["author"],category=r["category"],offsets=offsets))
            config["tracks"].append(dict(id=f"s{j}",span=[j*1.15+.04,j*1.15+.96]))
        scale=max(1.,float(np.max(np.abs(x))/.91));x/=scale;truth=[t/scale for t in truth]
        row=dict(sources=desc)
        with tempfile.TemporaryDirectory() as tmp:
            td=Path(tmp);train.write_wav(td/"in.wav",x,stem=True)
            for key in ["baseline","estimate"]:
                if key=="estimate":config["separator"]="adaptive"
                dst=td/key;settings=train.multitrack.export_tracks(td/"in.wav",config,dst)
                est=[train.load_media(dst/(t["id"]+".wav"))[0] for t in config["tracks"]]
                row[key]=train.scores(est,truth,x)
                if key=="estimate":
                    rem=train.load_media(dst/"remaining.wav")[0]
                    row["spatial_used"]=settings["spatial_filter_applied"]
                    row["max_sum_error"]=float(np.max(abs(sum((v.astype("f8") for v in est),rem.astype("f8"))-x)))
        cases.append(row)
    output=dict(test_inventory=len(source_list),eligible_native=len(allowed),excluded=not_used,
                unique_test_sources=len({s["id"] for c in cases for s in c["sources"]}),
                adaptive_used=sum(c["spatial_used"] for c in cases),baseline=train.summary(cases,"baseline"),
                estimate=train.summary(cases,"estimate"),model_sha=train.digest(train.CHECKPOINT),cases=cases)
    (train.HERE/"native_stereo.json").write_text(json.dumps(output,indent=2)+"\n")
    print({k:v for k,v in output.items() if k not in ["cases","excluded"]})
if __name__=="__main__": main()
