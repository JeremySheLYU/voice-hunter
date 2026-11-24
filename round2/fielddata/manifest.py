"""Build a complete source-grouped inventory from verified extraction receipts."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
from collect import CAT, HERE, ROOT

def assemble():
    sites={}
    with (CAT/'eigenscape_sites.csv').open(encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            kind,idx=row['Filename'][:-4].split('-')
            sites[kind,int(idx)]=row['Location']
    grouped={}
    rows=[json.loads(p.read_text()) for p in sorted((HERE/'receipts').glob('*/*.json'))]
    dates=sorted({Path(i['member']).name.split('_')[0] for i in rows if i['dataset']=='ihta_foa'})
    test_dates=set(dates[3::4])
    seen=set()
    for item in rows:
        data =(ROOT/item['path']).read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        assert digest==item['sha256'] and len(data)==item['bytes'] and digest not in seen
        seen.add(digest)
        k=item['dataset'];member=Path(item['member']).name
        if k.startswith('eigenscape'):
            m=re.search(r'([A-Za-z]+)[.-]0?(\d+)\.',member)
            family,idx=m.group(1),int(m.group(2))
            group=f'eigenscape/{family}/{idx}';location=sites[family,idx]
            split='test' if idx>=7 else ('validation' if idx==6 else 'train')
        elif k.startswith('ihta'):
            date=member.split('_')[0];group='ihta/'+member;location='IHTApark Aachen'
            split='test' if date in test_dates else 'train'
        else:
            group='london/'+member;location='London audio stimuli'
            split='reference_only' if ('Music' in member or 'TV' in member) else 'external_test'
        key=item['record']
        item.update(source_group=group,location=location,split=split,source='https://zenodo.org/records/'+key)
        if 'archive' not in item: item['archive']=json.loads((CAT/(k+'.json')).read_text())['archive']
        item['credit']=[v['name'] for v in json.loads((CAT/(key+'.json')).read_text())['metadata']['creators']]
        if k=='ihta_ears':item['format']='binaural teacher decode from paired AmbiX capture'
        elif k!='eigenscape_hoa':item['format']='FOA AmbiX ACN/SN3D W,Y,Z,X'
        grouped[group]=max(grouped.get(group,0),item['duration_s'])
    counts=collections.Counter(i['dataset'] for i in rows)
    assert counts=={'eigenscape':64,'eigenscape_hoa':8,'ihta_foa':102,'ihta_ears':102,'london':7},counts
    summary=dict(files=len(rows),unique_originals=len(grouped),unique_timeline_s=sum(grouped.values()),file_timeline_s=sum(i['duration_s'] for i in rows),by_collection=counts,
        bytes=sum(i['bytes'] for i in rows),rates=dict(collections.Counter(i['rate'] for i in rows)),channels=dict(collections.Counter(i['channels'] for i in rows)),
        duplicate_sha256_count=len(rows)-len(seen),split_by_asset=dict(collections.Counter(i['split'] for i in rows)),
        note='Eight fourth-order excerpts and 102 binaural versions share source times with first-order files. Counts refer to local excerpts.')
    (HERE/'manifest.json').write_text(json.dumps({'summary':summary,'files':rows},indent=2,ensure_ascii=False)+'\n')
    print(summary)
if __name__=='__main__':assemble()
