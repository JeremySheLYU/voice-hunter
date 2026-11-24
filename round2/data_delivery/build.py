"""Make self-contained, source-credited, checksummed archive parts. Run from the project root."""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import shutil
import time
import os
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/VoiceHunter_Public_Data'
META=ROOT/'round2/data_delivery'
LIMIT=138*1024**2
CC='https://creativecommons.org/licenses/by/'
LIB='LibriSpeech ASR corpus; Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, with LibriVox volunteer readers. See the recorded reader ID and the original SPEAKERS.TXT.'
SOURCES=json.loads((ROOT/'round2/spatialdata/sources.json').read_text())
EXCLUSIONS=[]
INVENTORY=[]
EXTRA=defaultdict(list)


def read(name):
    return json.loads((ROOT/'round2'/(name+'.json')).read_text())


def read_bytes(path):
    for attempt in range(4):
        try:return path.read_bytes()
        except TimeoutError:
            if attempt==3:raise
            time.sleep(2**attempt)

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def add(group,path,credit,license,source,note,expected=None,local=None,size=None):
    p=Path(path)
    if p.is_absolute():p=p.relative_to(ROOT)
    f=Path(local) if local else ROOT/p
    if size is None:size=f.stat().st_size
    r=dict(collection=group,path=p.as_posix(),bytes=size,sha256=expected,credit=credit,license=license,source=source,description=note)
    INVENTORY.append((r,f))


def sources(key):
    d=SOURCES[key]
    return ', '.join(c['name'] for c in d['creators']),d['license_url'],d['record_url']


def collect_basic():
    env={x['path']:x for x in read('envdata/manifest')+read('envdata/all_sources')}
    for r in env.values():
        assert r['license'].startswith(('http://creativecommons.org/licenses/by/','http://creativecommons.org/publicdomain/zero/','https://creativecommons.org/licenses/by/','https://creativecommons.org/publicdomain/zero/'))
        add('Freesound',r['path'],r['author'],r['license'],r['url'],f"{r['id']} {r['title']}; {r['category']}; {r['split']}; {r['representation']} (standard FSD50K WAV, or Freesound MP3 preview decoded as FLAC as identified).",r['sha256'])
    EXTRA['Freesound']+=['round2/envdata/LICENSE-FSD50K.txt','round2/envdata/README.md']
    for r in read('fielddata/manifest')['files']:
        if r['split']=='reference_only':
            EXCLUSIONS.append({'path':r['path'],'reason':'Reference-only third-party music/television stimulus; rights for underlying programme have not been verified.','source':r['source'],'retrieval':'round2/fielddata/collect.py'}); continue
        key='Field_'+r['dataset']
        add(key,r['path'],', '.join(r['credit']),CC+'4.0/',r['source'],f"{r['member']}; {r['duration_s']} s excerpt from {r['source_offset_s']} s; {r['format']}; {r['split']}",r['sha256'])
        if not EXTRA[key]:EXTRA[key].append('round2/fielddata/ATTRIBUTION.md')
    for r in read('stereo/manifest')['records']:
        add('FMA',Path('round2_data/stereo')/r['filename'],r['artist'],r['license'],r['track_url'],f"{r['track']}; {r['album']}; FMA-small {r['archive_member']}; artist {r['artist_id']}; {r['split']}; unmodified FMA MP3. {r['copyright']}",r['sha256'])
    for r in read('spatialtrain/manifest'):
        if r['file'].endswith('.mat'):
            add('CIPIC',Path('round2_data/spatial/cipic')/r['file'],'The Regents of the University of California, CIPIC; V. R. Algazi, R. O. Duda, D. M. Thompson and C. Avendano.', 'University of California permission & disclaimer in CIPIC_NOTICE.txt',r['url'],'Intact measured responses for one participant or mannequin condition.',r['sha256'])
    EXTRA['CIPIC']+=['round2_data/spatial/cipic/CIPIC_NOTICE.txt','round2_data/spatial/cipic/HRIR_README.txt']
    h=read('spatialtrain/hutubs_manifest')
    for r in h['files']:
        if r['file'].endswith('.sofa'):
            add('HUTUBS',Path('round2_data/spatial/hutubs')/r['file'],h['credit'],CC+'4.0/',f"https://doi.org/{h['doi']}",f"Original SOFA. Direct source {r['url']}",r['sha256'])
    for r in read('spatialdata/satp_index'):
        add('SATP',Path('round2_data/spatial')/r['file'],*sources('satp'),r['archive_member']+'; unmodified original WAV',r['sha256'])
    bdir=ROOT/'round2_data/spatial/binaurec'
    for p in sorted(bdir.rglob('*.wav')):
        m=json.loads(next(p.parent.glob('*.json')).read_text())
        c,l,u=sources('binaurec')
        add('BinauRec',p,c+'; underlying speech '+LIB,l,u,p.parent.name+' / '+p.stem+'; matched measured left/right front channels interleaved as stereo',m[p.stem])
    for p in sorted((ROOT/'round2_data/spatial/apl').glob('*')):
        if p.suffix not in ('.wav','.sofa') or p.name.startswith('.'):continue
        if p.suffix=='.sofa':
            add('Huddersfield_SOFA',p,*sources('apl'),'Complete measured 360 degree binaural responses in SOFA, one intact source, above the nominal part size.')
        else:
            add('Huddersfield',p,*sources('apl'),'Original AmbiX WAV for position '+p.stem)
    for r in read('spatialdata/aachen_index')['paired_binaural_or_headless']:
        d=SOURCES['aachen']
        add('AIR',Path('round2_data/spatial')/r['file'],d['attribution'],'MIT, AIR_LICENSE.txt',d['url'],f"Interleaved original measured WAV pair {r['source_members_LR']}",r['sha256'])
    EXTRA['AIR'].append('round2/spatialdata/AIR_LICENSE.txt')
    for r in read('spatialdata/films_index'):
        film=r['film']; credit=('© 2008 Blender Foundation / www.bigbuckbunny.org' if film=='bbb' else '© Blender Foundation | www.sintel.org')+'; music by Jan Morgenstern. Master mirrored at Xiph.'
        official='https://peach.blender.org/about/' if film=='bbb' else 'https://durian.blender.org/sharing/'
        add('Blender_films',Path('round2_data/spatial/films')/r['file'],credit,CC+'3.0/',r['url'],f"12-second film-master excerpt starting {r['start_seconds']}s, {r['channels']} channels; film terms {official}",r['local_sha256'])


def collect_speech(stage):
    with (ROOT/'round2/speakers/librispeech-files.csv').open() as f:rows=list(csv.DictReader(f))
    corpus={r['partition']:r for r in read('speakers/corpus')}
    cache=ROOT/'work/round2-corpus/librispeech'
    def extract_partition(item):
        part,cor=item
        source=cache/(part+'.tar.gz')
        cached_dirs=[Path(d) for d in os.environ.get('VH_CORPUS_BUILD_CACHE','').split(os.pathsep) if d]
        for cache_dir in cached_dirs:
            all_rows=[r for r in rows if r['partition']==part]
            if all((cache_dir/r['path']).is_file() for r in all_rows):
                for r in all_rows:
                    add('LibriSpeech_'+part,cache/r['path'],LIB,CC+'4.0/',cor['official_page'],f"Original FLAC, reader {r['reader']}, {part}. Original tar.gz {cor['url']}",r['sha256'],cache_dir/r['path'],int(r['bytes']))
                for src in sorted((cache_dir/part).glob('*.TXT')):
                    EXTRA['LibriSpeech_'+part].append((src,'notices/LibriSpeech/'+src.name))
                print('Recovered staged partition',part,flush=True)
                return
        whole=read_bytes(source)
        assert hashlib.sha256(whole).hexdigest()==cor['archive_sha256']
        wanted={r['path']:r for r in rows if r['partition']==part}
        with tarfile.open(fileobj=io.BytesIO(whole),mode='r|gz') as tar:
            for member in tar:
                if not member.isfile():continue
                name=member.name
                if name in wanted:
                    r=wanted.pop(name)
                    src=stage/name
                    src.parent.mkdir(parents=True,exist_ok=True)
                    src.write_bytes(tar.extractfile(member).read())
                    add('LibriSpeech_'+part,cache/name,LIB,CC+'4.0/',cor['official_page'],f"Original FLAC, reader {r['reader']}, {part}. Original tar.gz {cor['url']}",r['sha256'],src,int(r['bytes']))
                elif name in {'LibriSpeech/SPEAKERS.TXT','LibriSpeech/LICENSE.TXT','LibriSpeech/README.TXT'}:
                    src=stage/part/Path(name).name
                    src.parent.mkdir(parents=True,exist_ok=True)
                    src.write_bytes(tar.extractfile(member).read())
                    EXTRA['LibriSpeech_'+part].append((src,'notices/LibriSpeech/'+src.name))
        assert not wanted,part
        print('Extracted verified FLAC partition',part,flush=True)
    with concurrent.futures.ThreadPoolExecutor(3) as jobs:list(jobs.map(extract_partition,corpus.items()))


def collect_air_singletons(stage):
    rows={r['member']:r for r in read('spatialdata/aachen_index')['other_phone_single_channel_wavs']}
    p=ROOT/'round2_data/spatial/aachen/original.zip'
    with ZipFile(io.BytesIO(read_bytes(p))) as z:
        nested=next(k for k in z.namelist() if k.endswith('/AIR_wav_files.zip'))
        with ZipFile(io.BytesIO(z.read(nested))) as zz:
            for info in zz.infolist():
                name=Path(info.filename).name
                if name not in rows:continue
                r=rows.pop(name)
                s=stage/'AIR_phone'/name
                s.parent.mkdir(exist_ok=True)
                s.write_bytes(zz.read(info))
                add('AIR',Path('round2_data/spatial/aachen/phone_original')/name,SOURCES['aachen']['attribution'],'MIT, AIR_LICENSE.txt',SOURCES['aachen']['url'],'Original single-channel phone IR expanded from internal WAV archive; '+info.filename,r['sha256'],s)
    assert not rows


def description():
    return ('Voice Hunter – public research corpus. Each sound, measurement and recording remains subject to its source terms. '
            'ATTRIBUTION.json lists authors, licenses, origins and changes for every asset in this part. '
            'Content-specific notices may be in the notices folder. The FSD50K collection is credited to Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font and Xavier Serra; https://doi.org/10.5281/zenodo.4060432 (CC BY 4.0). '
            'FMA dataset curated by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier Bresson; https://github.com/mdeff/fma. '
            'Audio for each FMA track is credited and licensed individually. Original source authors have not endorsed Voice Hunter.\n'
            'Extract parts into a shared directory to restore project-relative paths. Each zip is independently readable.\n')


def record_zip(out,items):
    return dict(name=out.name,sha256=sha(out),bytes=out.stat().st_size,assets=len(items),
                logical_bytes=sum(r['bytes'] for r,p in items),collections=sorted(set(r['collection'] for r,p in items)))

def pack(filename,items):
    out=OUT/filename
    if out.is_file():
        with ZipFile(out) as archived:
            present=json.loads(archived.read('notices/ATTRIBUTION.json'))
            assert {r['path']:r['bytes'] for r in present}=={r['path']:r['bytes'] for r,p in items}
            known={r['path']:r['sha256'] for r in present}
            assert all(not r['sha256'] or r['sha256']==known[r['path']] for r,p in items)
        return record_zip(out,items)
    tmp=STAGE/filename
    checked=[]
    with ZipFile(tmp,'w',compression=ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        keys=sorted(set(r['collection'] for r,p in items))
        z.writestr('notices/README.txt',description())
        emitted=set()
        for k in keys:
            for extra in EXTRA[k]:
                p,n=(ROOT/extra,'notices/'+Path(extra).name) if isinstance(extra,str) else extra
                if n not in emitted:z.write(p,n);emitted.add(n)
        def read_asset(pair):
            r,p=pair
            payload=read_bytes(p)
            actual=hashlib.sha256(payload).hexdigest()
            if r.get('sha256') and r['sha256']!=actual:raise ValueError('Checksum mismatch: '+str(p))
            r['sha256']=actual
            return r,p,payload
        with concurrent.futures.ThreadPoolExecutor(18) as workers:
            for r,p,payload in workers.map(read_asset,items,buffersize=32):
                z.writestr(r['path'],payload,compress_type=ZIP_STORED if p.suffix in ('.flac','.mp3','.sofa') else ZIP_DEFLATED,compresslevel=1)
                checked.append(r)
        z.writestr('notices/ATTRIBUTION.json',json.dumps(checked,indent=1,ensure_ascii=False))
    with ZipFile(tmp) as z:
        for r in checked:
            with z.open(r['path']) as fp: actual=hashlib.file_digest(fp,'sha256').hexdigest()
            if actual!=r['sha256']:raise ValueError('Post-write verification failed: '+r['path'])
    v=record_zip(tmp,items)
    shutil.copyfile(tmp,out.with_suffix('.part'))
    out.with_suffix('.part').replace(out)
    assert sha(out)==v['sha256']
    tmp.unlink()
    print(f"PACK {filename} {v['assets']} {v['bytes']}",flush=True)
    return v


def sample():
    scenes=[('Field_eigenscape','TrainStation.1'),('Field_eigenscape_hoa','TrainStation-01'),('Field_ihta_ears','20220202_Wed_12h00_Pos1'),('Field_ihta_foa','20220202_Wed_12h00_Pos1'),('Field_london','Fan'),('SATP','W06'),('Huddersfield','C1m.wav'),('AIR','stairway'),('Blender_films','bbb_51_180'),('Freesound','/train/'),('Freesound','/wind/'),('FMA','')]
    selected=[]
    for key,match in scenes:
        options=sorted((pair for pair in INVENTORY if pair[0]['collection']==key and match in pair[0]['path']),key=lambda pair:pair[0]['path'])
        if options:selected.append(options[0])
    while sum(r['bytes'] for r,p in selected)>34*1024**2:selected.pop(-2)
    info=pack('Public_Spatial_Sampler.zip',selected)
    assert info['bytes']<35*1024**2
    return info


def run():
    global STAGE
    STAGE=Path(tempfile.mkdtemp(prefix='vh-fast-packs-'))
    OUT.mkdir(parents=True,exist_ok=True);META.mkdir(parents=True,exist_ok=True)
    collect_basic()
    representative=sample()
    def bundle_group(pair):
        grp,items=pair
        batch=[];used=0;part=1
        collected=[]
        for r,p in sorted(items,key=lambda it:it[0]['path']):
            if batch and used+r['bytes']>LIMIT:
                collected.append(pack(f'{grp}_{part:02d}.zip',batch));batch=[];used=0;part+=1
            batch.append((r,p));used+=r['bytes']
        if batch:collected.append(pack(f'{grp}_{part:02d}.zip',batch))
        return collected
    groups=defaultdict(list)
    reports=[]
    for r,p in INVENTORY:groups[r['collection']].append((r,p))
    # Build file groups in parallel with the three sequential-access FLAC archives.
    with concurrent.futures.ThreadPoolExecutor(4) as workers:
        def add_more():
            collect_speech(STAGE);collect_air_singletons(STAGE)
        acquired=workers.submit(add_more)
        ready=[workers.submit(bundle_group,item) for item in groups.items() if item[0]!='AIR']
        acquired.result()
        extra=defaultdict(list)
        for r,p in INVENTORY:
            if r['collection']=='AIR' or r['collection'].startswith('LibriSpeech_'):extra[r['collection']].append((r,p))
        ready.extend(workers.submit(bundle_group,item) for item in extra.items())
        for future in ready:reports.extend(future.result())
    reports.sort(key=lambda r:r['name'])
    adm=ROOT/'round2/objects/manifest.json'
    if adm.is_file():
        record=json.loads(adm.read_text())
        EXCLUSIONS.append(dict(path='round2_data/objects/',source=record['source_repository'],commit=record['commit'],
                               reason='EBU ADM GPL-3.0-or-later technical conformance fixtures. No commercial weights or audio quality tests used them. Four WAVs have identical audio with differing metadata.',retrieval='See round2/objects/manifest.json for the pinned original and SHA-256 of each individual file.'))
    paths=[r['path'] for r,p in INVENTORY]; assert len(set(paths))==len(paths)
    assert all(r['bytes']<=140*1024**2 or r['name']=='Huddersfield_SOFA_01.zip' for r in reports)
    overview=dict(parts=len(reports),asset_files=len(paths),logical_asset_bytes=sum(r['bytes'] for r,p in INVENTORY),stored_archive_bytes=sum(r['bytes'] for r in reports))
    manifest=dict(summary=overview,sampler=representative,archives=reports,excluded=EXCLUSIONS,
                  note='Distinct source files are counted including alternate representations; they are not counts of independent scenes. The sampler is a deliberate subset and is excluded from totals.')
    (META/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (META/'INDEX.md').write_text('# Voice Hunter 公開音訊資料包\n\n'+
        f"共 {len(reports)} 個獨立 ZIP，內有 {len(paths):,} 個資料檔。解壓後音訊與量測資料共 {overview['logical_asset_bytes']:,} bytes，壓縮檔共 {overview['stored_archive_bytes']:,} bytes。每包內 notices/ATTRIBUTION.json 記錄每一檔的來源、作者、授權與 SHA-256。為便利直接試聽，Public_Spatial_Sampler.zip 另收錄其中若干檔案，獨立計數。\n\n"+
        '| 資料包 | 檔案 | ZIP MiB |\n|---|---:|---:|\n'+''.join(f"| {r['name']} | {r['assets']} | {r['bytes']/1024**2:.1f} |\n" for r in reports)+
        '\n兩段含第三方電視或音樂的倫敦錄音及四個 EBU GPL 技術 WAV（PCM 完全相同，物件時間資料不同）只保留在本機。完整 Huddersfield C1m.sofa 以一個獨立 ZIP 保留，此單檔是 140 MiB 常規分包的唯一例外。manifest.json 列有來源及重取方法。原始下載壓縮包及其重複內容未重複收錄。LibriSpeech 的原版 tar.gz 由 OpenSLR 完整校驗流程另行下載。\n')
    shutil.copy2(META/'INDEX.md',OUT/'INDEX.md');shutil.copy2(META/'manifest.json',OUT/'manifest.json')
    import subprocess,sys
    subprocess.run([sys.executable,str(META/'objects.py')],check=True)
    subprocess.run([sys.executable,str(META/'finalize.py')],check=True)
    shutil.rmtree(STAGE)
    print(json.dumps(overview),flush=True)


if __name__=='__main__':run()
