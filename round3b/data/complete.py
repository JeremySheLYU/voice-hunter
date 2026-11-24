"""Apply all inherited ESC-50 original-uploader exclusions to the frozen-order length census."""
import json
import time
from pathlib import Path
from collections import Counter
from census import ROOT,HERE,author,sha,save,load
first=['qa_round1/events/public_sources.json','qa_round1/mixtures/sources.json']
prior_authors=set();prior_ids=set()
for f in first:
 for name,r in load(f).items():
  if 'ESC-50/' in r.get('source_url',''):
   prior_authors.add(r['attribution'].split(',')[0].strip().casefold())
   prior_ids.add(name.split('-')[1])
plan=load('round3b/data/plan.json')
removed=[r for r in plan if r['id'] in prior_ids or author(r) in prior_authors]
save('source_exclusions.json',{'purpose':'Apply original full-history exclusion to two additional prior ESC-50 benchmark manifests. Stable order retained. Census-only, no model trained or scored.','old_fsd_benchmark_manifests':first,'authors':sorted(prior_authors),'all_old_esc50_original_ids_including_already_excluded':sorted(prior_ids),'excluded_candidate_ids':[r['id'] for r in removed]})
while not (HERE/'durations.json').exists():time.sleep(1)
exclude={r['id'] for r in removed}
det=load('round3b/data/durations.json')
s={}
for c in ['guitar','piano','speech','train','wind']:
 group=[r for r in det if r['category']==c and r['id'] not in exclude]
 good=[r for r in group if r['eligible']]
 s[c]={'candidates':len(group),'candidate_uploaders':len({author(r) for r in group}),'eligible_sources':len(good),'eligible_uploaders':len({author(r) for r in good}),'unreadable_headers':sum('error' in r for r in group)}
out=dict(fsd50k_primary_gate_met=all(r['eligible_uploaders']>=12 for r in s.values()),source='FSD50K residual pool, files and authors unused by the project, CC BY or CC0, ≥4.5 s, mono PCM. At least 12 distinct eligible uploaders per class required.',classes=s)
save('feasibility.json',out);print(out)
