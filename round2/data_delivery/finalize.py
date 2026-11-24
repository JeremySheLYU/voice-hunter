"""Register the independent GPL technical pack alongside (outside) the ML data packs."""
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/VoiceHunter_Public_Data'


def main():
    path=HERE/'manifest.json'
    m=json.loads(path.read_text())
    o=json.loads((HERE/'objects_pack.json').read_text())
    if not (OUT/o['name']).is_file():raise ValueError('Missing technical pack')
    m['optional_format_test']=o|dict(unique_tones=1,wav_variants=4,authored_xml=4,
                                     note='Complete unmodified upstream technical fixtures and source XML with original notices; optional, excluded from all training and audio corpus counts.')
    m['excluded']=[r for r in m['excluded'] if r.get('path')!='round2_data/objects/']
    m['part_size_note']='Nominal maximum 140 MiB. Huddersfield_SOFA_01.zip preserves a single 163,170,474-byte measured SOFA and is larger.'
    path.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    d=HERE/'INDEX.md'
    line='\n## 獨立物件格式測試包\n\nADM_Object_Conformance.zip 包含 EBU 提供的四個原始物件式 ADM WAV、四份作者的 XML 及完整原始 GPL-3.0-or-later 版權和授權文件。四個例子共用同一段 3 秒 440 Hz 音訊，各包含不同位置時間資料。此獨立測試包用於格式驗證。抽樣包和格式包不列入上述訓練／評估來源檔案總數。\n'
    txt=d.read_text().replace('兩段含第三方電視或音樂的倫敦錄音及四個 EBU GPL 技術 WAV（PCM 完全相同，物件時間資料不同）只保留在本機。','兩段涉及第三方電視或音樂的倫敦錄音只保留在本機。')
    if '## 獨立物件格式測試包' not in txt:txt+=line
    d.write_text(txt)
    shutil.copyfile(path,OUT/'manifest.json');shutil.copyfile(d,OUT/'INDEX.md')
    print(m['summary'],o['name'],o['bytes'])


if __name__=='__main__':main()
