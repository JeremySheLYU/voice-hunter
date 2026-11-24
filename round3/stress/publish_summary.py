"""Build three verified primary-model rows and an accompanying readable comparison."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
g=lambda name:json.loads((HERE/name).read_text())
hist, fresh, label=sys.argv[1:]
native_plan=g('fresh_native_scenes.json')
nativeg=g('scenes.json')
primary_hist=g(hist)['conditions']
primary_fresh=g(fresh)['conditions']
methods=[
    ('v0.2 研究候選',g('r2.json')['conditions'],g('fresh_native_v02.json')['conditions']),
    ('原有 Apple 自動模式',g('apple.json')['conditions'],g('fresh_native_apple.json')['conditions']),
    (label,primary_hist,primary_fresh),
]
for extra, history, new in [
    ("AST Transformer 探索比較","transformer.json","fresh_native_round3.transformer.model.json"),
    ("幾何檢查候選探索比較","mixture.json","fresh_native_mixture.json"),
]:
    if (HERE/history).exists() and (HERE/new).exists():
        methods.append((extra,g(history)['conditions'],g(new)['conditions']))
conditions=[('room','實測樓梯間響應'),('native','既有天然立體聲素材'),('fresh_native','新抓取天然立體聲素材')]
rows=[]
report=[
    '# 五類聲源分軌的聲場壓力測試', '',
    '對每組混音分別請求一種標記存在和一種標記缺席的聲音。根據是否交軌計算準確率和 F1。交出的目標軌若在共同發聲部分的雙聲道 SI-SDR 比原音改善至少 3 dB，此次才計為有用分軌。低於原混音的輸出另列。', '',
    '| 條件 | 方法 | 準確率 | F1 | 有用分軌／應交 | 誤交／缺席 | 劣於原音 |','|---|---|---:|---:|---:|---:|---:|']
details=['','各類聲源有用分軌數如下。','', '| 條件 | 方法 | 吉他 | 鋼琴 | 人聲 | 列車 | 風 |','|---|---|---:|---:|---:|---:|---:|']
for key,cn in conditions:
    for name,h,f in methods:
        d=next(ds[key] for ds in [h,f] if key in ds)
        s=d['summary']
        if s['errors'] or s['accuracy'] is None or s['F1'] is None:
            raise ValueError('Incomplete scores: '+key+' '+name)
        report.append(f"| {cn} | {name} | {s['accuracy']:.1%} | {s['F1']:.3f} | {s['useful']}/{s['intended_positive']} | {s['FP']}/{s['FP']+s['TN']} | {s['harmful']} |")
        sections=[cn,name]+[f"{d['by_requested_class'][c]['useful']}/{d['by_requested_class'][c]['intended_positive']}" for c in ['guitar','piano','speech','train','wind']]
        details.append('| '+' | '.join(sections)+' |')
        if name==label:
            rows.append(dict(label=cn,accuracy=s['accuracy'],f1=s['F1'],useful=s['useful'],positive=s['intended_positive'],harmful=s['harmful']))
report+=details
report+=['', '### 如何解讀', '',
         '樓梯間使用 55 組已公開的舊版保留測試素材，各自與同一間實際樓梯間的左右脈衝響應卷積，以評估強混響的影響。輸入並非手機直接錄下的現場對話。','',
         '既有天然立體聲的 16 個錄音來自 15 位上傳者，保留每筆素材原有的左右聲道，再將兩種素材等強度相加。64 個目標條件對應 53 個獨特混音。當兩個聲源輪流作為目標時，混音可以完全相同。','',
         f"新抓取立體聲使用 {len(native_plan['sources'])} 個具有實際左右差異的錄音、{native_plan['uploaders']} 位上傳者，配成 {len(native_plan['scenes'])} 個目標條件、{native_plan['unique_pairs']} 個不重複的混音。新組與過去素材、兩輪訓練的來源 ID 和上傳者均沒有重疊。當中 {len(native_plan['fresh_test_shared_ids'])} 個錄音 ID、{len(native_plan['fresh_test_shared_authors'])} 位上傳者與新一輪受控測試素材有交集；同一 Freesound 錄音的不同聲道版不宜相加充作獨立錄音。",'',
         '「缺席」依原始目錄標籤和混音組合推定。兩個各自錄下的立體聲檔相加，不能完全代表一部手機在同一現場的收音。實驗中的目標、條件和底層音檔重複使用，對照應逐條件或逐組解讀。','']
oracle=g('oracle_v02_demix.json')['conditions']
report+=['### 分離器的幾何限制','']
for k,cn in conditions[:2]:
    o=oracle[k]
    report.append(f"{cn}中，v0.2 固定方向分離器在 {o['directions_detected']}/{o['intended']} 組估出兩個輸出。即使在測試後用乾淨參考聲軌替它挑出表現較好的一軌，分軌達標也只有 {o['ideal_label_useful']}/{o['intended']}。此項診斷只適用於原有固定方向分離器。")
    report.append('')
fresh_oracle=g('fresh_oracle_v02_demix.json')['summary']
report.append(f"在新抓取的原始立體聲條件上，v0.2 的固定方向分離器在 {fresh_oracle['directions_detected']}/{fresh_oracle['positive']} 組估出兩條候選軌。以真值替它選出最好的候選軌後，共 {fresh_oracle['ideal_label_useful']}/{fresh_oracle['positive']} 組達到 3 dB。")
report.append('')
(HERE/'user_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(HERE/'README.md').write_text('\n'.join(report)+'\n')
print(rows)
