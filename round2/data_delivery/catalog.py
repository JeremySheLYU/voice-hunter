"""Verify the physical delivery from the archive bytes and regenerate its top-level catalogue."""
import concurrent.futures
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
from zipfile import ZipFile

import build
import finalize

def inspect_zip(path):
    blob=build.read_bytes(path)
    with ZipFile(io.BytesIO(blob)) as z:
        manifest=json.loads(z.read('notices/ATTRIBUTION.json'))
        for r in manifest:
            if len(r['credit'])<2 or not r['source'] or not r['license']:raise ValueError('Missing credit '+r['path'])
            payload=z.read(r['path'])
            if len(payload)!=r['bytes'] or hashlib.sha256(payload).hexdigest()!=r['sha256']:raise ValueError('Invalid '+r['path'])
        return dict(name=path.name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),
                    assets=len(manifest),logical_bytes=sum(r['bytes'] for r in manifest),
                    collections=sorted({r['collection'] for r in manifest})),{r['path'] for r in manifest}

def main():
    paths=sorted(build.OUT.glob('*.zip'))
    with concurrent.futures.ThreadPoolExecutor(5) as workers:
        entries=list(workers.map(inspect_zip,paths))
    records,sets=zip(*entries)
    sampler=next(r for r in records if r['name']=='Public_Spatial_Sampler.zip')
    obj=next(r for r in records if r['name']=='ADM_Object_Conformance.zip')
    packs=[];allpaths=set();cnt=Counter()
    for record,names in entries:
        if record['name'] in {sampler['name'],obj['name']}:continue
        if allpaths & names:raise ValueError('Unexpected repeated source pathname in data packs')
        allpaths|=names
        if len(record['collections'])!=1:raise ValueError(record)
        cnt[record['collections'][0]]+=len(names)
        packs.append(record)
    envrows=build.read('envdata/manifest')+build.read('envdata/all_sources')
    fields=build.read('fielddata/manifest')['files']
    expected=Counter({'Freesound':len({x['path'] for x in envrows}),'FMA':194,'CIPIC':45,'HUTUBS':96,'SATP':28,
                      'BinauRec':108,'Huddersfield':13,'Huddersfield_SOFA':1,'AIR':232,'Blender_films':16})
    expected.update('Field_'+f['dataset'] for f in fields if f['split']!='reference_only')
    for c in build.read('speakers/corpus'): expected['LibriSpeech_'+c['partition']]=c['clips']
    if cnt!=expected:raise ValueError(f'Missing/extra collection items: {expected-cnt}, {cnt-expected}')
    summary=dict(parts=len(packs),asset_files=len(allpaths),logical_asset_bytes=sum(r['logical_bytes'] for r in packs),
                 stored_archive_bytes=sum(r['bytes'] for r in packs))
    excluded=[dict(path=r['path'],reason='Underlying third-party music or TV rights unverified; held locally as reference.',source=r['source'],retrieval='round2/fielddata/collect.py') for r in fields if r['split']=='reference_only']
    m=dict(summary=summary,sampler=sampler,archives=packs,excluded=excluded,by_collection=dict(cnt),
           note='Audio asset files are counted including alternate representations and repeated measurements; they are not counts of independent scenes. Sampler and optional ADM test material are excluded from the bulk archive totals.')
    (build.META/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    (build.META/'objects_pack.json').write_text(json.dumps(obj,indent=2)+'\n')
    (build.META/'INDEX.md').write_text('# Voice Hunter 公開音訊資料包\n\n'+
        f"共 {len(packs)} 個獨立 ZIP，內含 {len(allpaths):,} 個來源音訊或量測資料檔。解壓資料為 {summary['logical_asset_bytes']:,} bytes，ZIP 合計 {summary['stored_archive_bytes']:,} bytes。每包內 notices/ATTRIBUTION.json 逐檔記錄作者、來源、授權、轉換及 SHA-256。解壓到同一目錄可還原專案相對路徑。額外的 Public_Spatial_Sampler.zip 收錄其中 11 個檔案（31,175,495 bytes），適合先試聽。\n\n"+
        '| 資料包 | 檔案數 | ZIP MiB |\n|---|---:|---:|\n'+''.join(f"| {r['name']} | {r['assets']} | {r['bytes']/1024**2:.1f} |\n" for r in packs)+
        '\n每個 ZIP 均已逐檔重新解碼 ZIP 壓縮、核驗音訊／量測位元組及對應 SHA-256。不同表示會重用同一時間軸，獨立錄音與訓練樣本數見各訓練組報告。C1m.sofa 以原始大小保留於獨立 ZIP，為 140 MiB 常規分包的單一例外。兩段含第三方電視／音樂的倫敦參考刺激只保留在本機，重取來源見 manifest.json。已收錄逐檔內容的原始 tar.gz 和 ZIP 不重複打包。\n')
    finalize.main()
    print(summary,flush=True)

if __name__=='__main__':main()
