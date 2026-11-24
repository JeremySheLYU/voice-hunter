"""Preregister and collect a separate duration/layout-only reserve for scarce native speech."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from round3.data.native import *
def reserve():
    primary=json.loads((HERE/'test.json').read_text())
    ids={r['id'] for r in primary};authors={group(r) for r in primary}
    frozen_plan=json.loads((HERE/'native_plan.json').read_text())
    queue=[r for r in frozen_plan if r['category']=='speech'][100:]
    addendum={'purpose':'Secondary native speech cohort if fewer than two eligible speech excerpts among initial 100; predeclared before any model was scored on new native clips. Initial attempts found zero qualifying speech in first 60; eligibility checks only. No class assessment or model score used to choose this acquisition extension.','activation':'Only append if primary initial cohort has fewer than two speech files.','ids':[r['id'] for r in queue],'cap_total_speech':4,'selection':'Same frozen re-use priority, page/attribution match, ≥4.9s true stereo, side fraction >−50 dB, existing author split; scan remaining reserve queue in order.'}
    lock=HERE/'native_reserve_freeze.json'
    if lock.exists():assert json.loads(lock.read_text())==addendum
    else:write(lock,addendum)
    while not (HERE/'native.ready').exists():time.sleep(1)
    rows=json.loads((HERE/'native.json').read_text())
    current=sum(r['category']=='speech' for r in rows)
    if current>=2:
        print('Reserve not needed.',flush=True)
        (HERE/'native.final.ready').write_text(digest((HERE/'native.json').read_bytes())+'\n')
        return
    used={r['sha256'] for r in rows};fail=Counter();n=0
    with ThreadPoolExecutor(max_workers=26) as pool:
        for start in range(0,len(queue),30):
            batch=queue[start:start+30];n+=len(batch)
            for r,e in pool.map(take,batch):
                if r is None:fail[e]+=1;continue
                if r['sha256'] in used or current>=4:
                    (ROOT/r['path']).unlink(missing_ok=True);continue
                used.add(r['sha256']);r['shares_primary_test_id']=r['id'] in ids;r['shares_primary_test_author']=group(r) in authors;r['native_speech_reserve']=True;current+=1;rows.append(r)
            print('eligible speech',current,'reserve examined',n,flush=True)
            if current>=4:break
    audit=json.loads((HERE/'native_audit.json').read_text())
    audit['initial_native_count']=audit['count'];audit['reserve_examined']=n;audit['reserve_eligibility_rejections']=fail
    audit.update(count=len(rows),classes=Counter(r['category'] for r in rows),authors=len({group(r) for r in rows}),shared_primary_test_ids=[r['id'] for r in rows if r['shares_primary_test_id']],shared_primary_test_uploaders=len({group(r) for r in rows if r['shares_primary_test_author']}),total_seconds=sum(r['duration'] for r in rows))
    write(HERE/'native.json',rows);audit['sha256']=digest((HERE/'native.json').read_bytes());write(HERE/'native_audit.json',audit)
    (HERE/'native.ready').write_text(audit['sha256']+'\n');(HERE/'native.final.ready').write_text(audit['sha256']+'\n')
if __name__=='__main__':reserve()
