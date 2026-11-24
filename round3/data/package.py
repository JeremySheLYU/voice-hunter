"""Assemble and independently verify the new source-audio delivery."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    assert (HERE/'native.final.ready').read_text()==(HERE/'native.ready').read_text()
    old=json.loads((ROOT/'round2/envdata/all_sources.json').read_text())
    src={k:json.loads((HERE/(k+'.json')).read_text()) for k in ('train','test','native')}
    oldauthors={r['author'].strip().casefold() for r in old}
    trainauthors={r['author'].strip().casefold() for r in src['train']}
    nativeauthors={r['author'].strip().casefold() for r in src['native']}
    assert not nativeauthors&(oldauthors|trainauthors)
    unk_path=ROOT/'round3/unknowns/manifest.json'
    unknowns=json.loads(unk_path.read_text())
    unknown_author={r['author'].strip().casefold() for r in unknowns}
    assert len(unknowns)==30 and not unknown_author&(oldauthors|trainauthors|nativeauthors|{r['author'].strip().casefold() for r in src['test']})
    src['unknown']=[dict(r,channels=1,duration=r['frames']/r['sample_rate'],bytes=(ROOT/r['path']).stat().st_size,split='test_negative_control',representation='standardized mono WAV') for r in unknowns]
    all_rows=sum(src.values(),[])
    unique_ids={r['id'] for r in all_rows}
    native_sha={r['sha256'] for r in src['native']}
    assert len(native_sha)==len(src['native'])
    with (HERE/'all_attribution.csv').open('w') as f:
        cols=['id','category','author','title','url','license','split','representation','duration','channels','path','download_url','sha256']
        out=csv.DictWriter(f,cols,extrasaction='ignore');out.writeheader();out.writerows(all_rows)
    status={k:len(v) for k,v in src.items()}
    total=sum(r['bytes'] for r in all_rows)
    shared=sorted({r['id'] for r in src['test']}&{r['id'] for r in src['native']})
    summary={'files':status,'unique_freesound_sources':len(unique_ids),'raw_bytes':total,'shared_primary_native_test_ids':shared,
             'native_training_author_overlap':0,'old_new_author_overlap':0,'negative_control_manifest_sha256':hashlib.sha256(unk_path.read_bytes()).hexdigest(),
             'license_file_counts':{}}
    for r in all_rows:summary['license_file_counts'][r['license']]=summary['license_file_counts'].get(r['license'],0)+1
    (HERE/'delivery.json').write_text(json.dumps(summary,indent=2)+'\n')
    description=f'''# Voice Hunter v0.3 公開音訊

## 內容

本包收錄 {status['train']} 份新訓練錄音、{status['test']} 份受控混音保留測試錄音，以及 {status['native']} 份保留原有左右聲道的立體聲壓力測試片段。另有 {status['unknown']} 份不同類別的負對照（犬吠、貓叫、鐘鈴、時鐘、滴水），用來檢查在目標缺席時是否誤交音軌。總計 {len(all_rows)} 個檔案，對應 {len(unique_ids)} 個不同 Freesound 錄音 ID。每項錄音的作者、標題、原始頁面、授權、原始下載網址與成品 SHA-256 收於 `round3/data/all_attribution.csv`。所有音檔放在 `round3_data/`。

最初兩組使用 FSD50K 公開的標準化單聲道 WAV。每個訓練聲類 90 份，每份長於或等於兩秒；保留測試每份長於或等於 4.9 秒。兩個聲源混合時的方位另行指定，原始 WAV 只有一個聲道。獨立立體聲壓力組擷取 Freesound 高音質預聽前十五秒內、最少 4.9 秒的部分，解碼為 FLAC；左右兩聲道均保留，通過側聲道能量檢查。與單聲道保留組重複的錄音 ID 清單見 `round3/data/delivery.json`。測試組不參與本輪模型擬合。

## 檔案與重現

- 使用 Python 3，在解壓目錄執行 `python3 round3/data/verify_package.py` 可逐一核對所有音檔 SHA-256。
- `round3/data/train.json`、`test.json`、`native.json` 分別提供訓練、保留測試和原生立體聲的音檔詳情。
- `round3/data/restore.py` 可由 FSD50K 鏡像的固定版本重新下載兩組單聲道 WAV，並核對 SHA-256。
- `round3/data/restore_native.py` 可依每項公開預聽網址擷取及解碼立體聲片段。網上原檔或 FFmpeg 版本可能改變；SHA-256 以交付的 FLAC 為準。
- `round3/data/freeze.json` 和 `native_freeze.json` 記錄取得音訊前固定的條件。各 `plan.json` 是包含合格候選的隊列，不代表實際使用了其中所有錄音。實際收錄以三份音檔清單為準。
- `round3/data/audit.json` 與 `native_audit.json` 提供實際數量、授權與稽核結果。

## 授權與來源

FSD50K 資料集由 Eduardo Fonseca、Xavier Favory、Jordi Pons、Frederic Font、Xavier Serra 等人建立，整體依 CC BY 4.0 授權。本包採用官方弱標籤作為分類依據，取用可依 CC0 或 CC BY 再利用的 Freesound 錄音；各檔的原始創作者與適用版本以 CSV 逐檔標示。修改包含部分錄音採取前段、音訊解碼／重新封裝，研究混音和模型輸出不包含在這份原音包。

FSD50K 官方資料與文件 https://zenodo.org/records/4060432

FSD50K 固定鏡像 https://huggingface.co/datasets/Fhrozen/FSD50k/tree/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b

授權說明 https://creativecommons.org/licenses/by/4.0/
'''
    rd=HERE/'README_Audio_zh.md';rd.write_text(description)
    out=ROOT/'outputs/VoiceHunter_R3_Public_Audio.zip'
    tmp=out.with_suffix('.zip.part')
    selected=['all_attribution.csv','train.json','test.json','native.json','train.ready','test.ready','native.ready','audit.json','native_audit.json','freeze.json','native_freeze.json','native_reserve_freeze.json','plan.json','native_plan.json','delivery.json','aligned_audio_audit.json','verify_package.py','restore.py','restore_native.py']
    with ZipFile(tmp,'w',compression=ZIP_DEFLATED,compresslevel=1) as z:
        z.write(rd,'README_Audio_zh.md')
        z.write(ROOT/'round2/envdata/LICENSE-FSD50K.txt','LICENSE-FSD50K.txt')
        for f in ('manifest.json','freeze.json'):
            q=ROOT/'round3/unknowns'/f
            z.write(q,q.relative_to(ROOT).as_posix())
        for f in selected:z.write(HERE/f,(HERE/f).relative_to(ROOT).as_posix())
        for r in all_rows:z.write(ROOT/r['path'],r['path'])
    with ZipFile(tmp) as z:
        assert not z.testzip()
        for r in all_rows:
            assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    tmp.replace(out)
    h=hashlib.sha256()
    with out.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    print(json.dumps(summary,indent=1));print(out,out.stat().st_size,h.hexdigest())
if __name__=='__main__':main()
