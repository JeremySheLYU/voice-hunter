"""Score the two-manual-example public entry point under fixed and changed positions."""
import json
import numpy as np
import train


def main():
    groups=train.load_index(train.HERE/"cohort.json")[0]
    records=[]
    for i,g in enumerate(train.groups(groups["test"],8,55601)):
        mixture,truth,config,assets=train.scene(g[:2],7421+i,"pan",False)
        a,b=config["tracks"]
        for condition in ["stationary","colocated","moved"]:
            src=[v.copy() for v in truth]
            if condition=="colocated":
                src=[np.repeat(s.mean(axis=1,keepdims=True),2,axis=1) for s in src]
            if condition=="moved":
                for s in src: s[round(train.START*train.SR):] = s[round(train.START*train.SR):,::-1].copy()
            mix=sum(src)
            old=train.separation.separate(mix,train.SR,method="example",target_span=a["span"],background_span=b["span"])[0]
            target,rest,meta=train.multitrack.separate_spatial_examples(mix,a["span"],b["span"])
            records.append(dict(scene=i,condition=condition,source=[r['asset'] for r in assets],baseline=train.measure(old,src[0],mix),
                   estimate=train.measure(target,src[0],mix),spatial_used=meta["spatial_filter_applied"],
                   reconstruct=float(np.max(np.abs(target.astype("f8")+rest-mix))),
                   max_change_fullscale=float(np.max(abs(target-old)))))
    summary={}
    for cond in ["stationary","colocated","moved"]:
        cases=[r for r in records if r["condition"]==cond]
        row={}
        for method in ["baseline","estimate"]:
            gains=np.array([r[method]["improvement"] for r in cases])
            row[method]=dict(pass_count=int(np.sum(gains>=3)),cases=len(cases),median=float(np.median(gains)),worst=float(np.min(gains)))
        row['spatial_used']=sum(r["spatial_used"] for r in cases)
        row['worst_additive_error']=max(r['reconstruct'] for r in cases)
        summary[cond]=row
    (train.HERE/"two_targets.json").write_text(json.dumps(dict(summary=summary,cases=records),indent=2)+"\n")
    print(summary)
if __name__=="__main__": main()
