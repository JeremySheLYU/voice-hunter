"""Audit independent official labels and original Freesound licensing without touching model results."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv, hashlib, json, re, time, urllib.request
from urllib.parse import urlsplit,unquote
from html import unescape
DIR=Path(__file__).resolve().parent
ROOT=DIR.parents[2]
def norm(a):return unquote(a).strip().casefold()
def load(p):return json.loads(p.read_text())
def digest(v):return hashlib.sha256(v).hexdigest()
def read(u,n=380000,timeout=8):
    req=urllib.request.Request(u,headers={'User-Agent':'VoiceHunter-audio-evaluation/0.3b'})
    with urllib.request.urlopen(req,timeout=timeout) as res:
        return res.read(n),res.geturl()
historic=[r for p in ['round2/envdata/all_sources.json','round3/data/train.json','round3/data/test.json','round3/data/native.json','round3/unknowns/manifest.json'] for r in (load(ROOT/p) if isinstance(load(ROOT/p),list) else [])]
ids={str(r['id']) for r in historic if 'id' in r}
authors={norm(r['author']) for r in historic if 'author' in r}
shas={r['sha256'] for r in historic if 'sha256' in r}
for p in [ROOT/'qa_round1/events/public_sources.json', ROOT/'qa_round1/rnnoise/sources.json',ROOT/'qa_round1/mixtures/sources.json',ROOT/'qa_round1/speakers/source_manifest.json']:
    s=p.read_text()
    for name,id_ in re.findall(r'freesound.org/people/([^/ ]+)/sounds/(\d+)',s):
        authors.add(norm(name));ids.add(id_)
    for id_ in re.findall(r'[1-5]-(\d+)-[A-Z]-\d+\.wav',s):ids.add(id_)
    for h in re.findall(r'\b[a-f0-9]{64}\b',s):shas.add(h)
old_esc=load(ROOT/'round3b/data/source_exclusions.json')
authors.update(old_esc['authors'])
ids.update(old_esc['all_old_esc50_original_ids_including_already_excluded'])
def lookup(r):
    if r['id'] in ids:return {**r,'error':'recording previously used'}
    try:
        page,link=read('https://freesound.org/s/'+r['id']+'/')
        html=page.decode('utf8','replace'); match=re.search(r'/people/([^/]+?)/sounds/'+r['id']+r'/|/people/([^/]+?)/sounds/'+r['id']+r'$',link)
        if not match:raise ValueError('source author unknown')
        author=unquote(match.group(1) or match.group(2))
        if norm(author) in authors:return {**r,'author':author,'error':'contributor previously used'}
        lic=re.search(r'https?://creativecommons\.org/(publicdomain/zero/1.0|licenses/by/(?:1.0|2.0|2.5|3.0|4.0))/',html)
        if not lic:raise ValueError('missing CC0/BY individual source permission')
        if len(set(re.findall(r'https?://creativecommons\.org/(?:licenses|publicdomain)/[^"<> ]+',html)))>3:raise ValueError('ambiguous license')
        preview=re.search(r'https?://cdn\.freesound\.org/previews/\d+/'+r['id']+r'_\d+-hq\.mp3',html)
        if not preview:raise ValueError('preview unavailable')
        title=re.search(r'<meta[^>]*property="og:title"[^>]*content="(.*?)"',html)
        return {**r,'url':link,'author':author,'license':lic.group(0),'title':unescape(title.group(1)) if title else None,'download_url':preview.group(0)}
    except Exception as e:return {**r,'error':str(e)}
def queues():
    cat={}
    for x in csv.DictReader((DIR/'esc50.csv').open()):
        if x['category'] in ('train','wind'):
            k=x['src_file'];cat[k]=dict(id=k,category=x['category'],corpus="ESC-50 source",official=x['filename'])
    for p in ['test_post_competition_scoring_clips.csv','train_post_competition.csv']:
        for x in csv.DictReader((DIR/p).open()):
            if x['label'] not in ['Electric_piano','Acoustic_guitar'] or x.get('manually_verified','1')!='1':continue
            k=x['freesound_id'];cat[k]=dict(id=k,category='guitar' if x['label']=='Acoustic_guitar' else 'piano',corpus='FSDKaggle2018 source',official=x['fname'])
    return sorted(cat.values(),key=lambda x:digest(('VH3b-external-20261121:'+x['id']).encode()))
if __name__=='__main__':
    pool=queues()
    for categories in [('wind','train'),('guitar','piano')]:
        entries=[r for c in categories for r in [x for x in pool if x['category']==c][:100]]
        with ThreadPoolExecutor(max_workers=26) as ex: rows=list(ex.map(lookup,entries))
        fresh=[r for r in rows if 'error' not in r]
        (DIR/('audit_'+categories[0]+'.json')).write_text(json.dumps(rows,indent=1))
        print(categories,'eligible',Counter(r['category'] for r in fresh),'reasons',Counter(r['error'] for r in rows if 'error'in r),flush=True)
