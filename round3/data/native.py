"""Freeze and acquire native-channel Freesound stereo excerpts for a separate stress cohort."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from round3.data.collect import *
from round2.envdata import collect as v2
import numpy as np
import subprocess
v2.RAW=ROOT/'round3_data/native'
v2.RAW.mkdir(exist_ok=True)
def take(r):
    try:
        rec=v2.audio(r,'stereo_preview')
        p=ROOT/rec['path']
        pcm=np.frombuffer(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(p),'-ar','12000','-f','f32le','pipe:1']),'<f4').reshape(-1,2).astype('f8')
        ratio=10*np.log10(max(np.mean(((pcm[:,0]-pcm[:,1])*.5)**2),1e-30)/max(np.mean(pcm**2),1e-30))
        if ratio<=-50:
            p.unlink(missing_ok=True)
            return None,'near dual mono'
        rec['side_fraction_power_db']=float(ratio)
        rec['signal_layout']='two-channel with measured differences'
        return rec,None
    except Exception as e:return None,str(e)[:120]
def main():
    test=json.loads((HERE/'test.json').read_text())
    primary_ids={r['id'] for r in test}; authors={group(r) for r in test}
    plan=sorted([r for r in frozen() if r['split']=='test'],key=lambda r:(r['id'] in primary_ids,group(r) in authors,r['priority'],digest(('vh3-native-only:'+r['id']).encode())))
    path=HERE/'native_plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:write(path,plan)
    lock={'sha256':digest(path.read_bytes()),'budget':100,'target_files_per_class':4,'minimum_seconds':4.9,'max_source_excerpt_seconds':15,'minimum_side_fraction_db':-50,
        'prefer_distinct_new_ids':True,'policy':'Frozen within-category queue from fresh holdout authors only, sort previously unused IDs then prefer uploaders absent from primary test. Attempt interleaved groups until per-class target, fixed cap 100/class, or eligible queue exhausted. Skip unavailable/mismatched Creative Commons pages, non-stereo, clips shorter than 4.9s, near-dual-mono or repeated source audio. Disclose overlap of all shared recording IDs and uploaders with primary holdout.'}
    write(HERE/'native_freeze.json',lock)
    used=set();counts=Counter();cursor=Counter();fail=Counter();out=[]
    buckets={c:[r for r in plan if r['category']==c][:100] for c in CATEGORIES}
    with ThreadPoolExecutor(max_workers=26) as pool:
        while True:
            batch=[]
            for c in CATEGORIES:
                if counts[c]>=4:continue
                took=buckets[c][cursor[c]:cursor[c]+12];cursor[c]+=len(took);batch.extend(took)
            if not batch:break
            for (row,err),original in zip(pool.map(take,batch),batch):
                if row is None:fail[err]+=1;continue
                if counts[row['category']]>=4 or row['sha256'] in used:
                    (ROOT/row['path']).unlink(missing_ok=True)
                    continue
                used.add(row['sha256']);counts[row['category']]+=1
                row['shares_primary_test_id']=row['id'] in primary_ids
                row['shares_primary_test_author']=group(row) in authors
                out.append(row)
            write(HERE/'native.partial.json',out)
            print('classes',counts,'attempts',cursor,'failures',fail,flush=True)
    write(HERE/'native.json',out)
    write(HERE/'native_audit.json',{'count':len(out),'classes':counts,'authors':len({group(r) for r in out}),'shared_primary_test_ids':[r['id'] for r in out if r['shares_primary_test_id']],'shared_primary_test_uploaders':len({group(r) for r in out if r['shares_primary_test_author']}),'attempts':cursor,'skipped':fail,'total_seconds':sum(r['duration'] for r in out),'sha256':digest((HERE/'native.json').read_bytes())})
    (HERE/'native.ready').write_text(digest((HERE/'native.json').read_bytes())+'\n')
if __name__=='__main__':main()
