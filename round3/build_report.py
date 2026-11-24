"""Build Chinese scorecard from measured outputs; no fit or evaluation in this script."""
import csv
import html
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def h(value):return html.escape(str(value))
def pc(x):return f'{float(x)*100:.1f}%'
def table(head,rows):
    return '<div class="wrap"><table><thead><tr>'+''.join('<th>'+h(a)+'</th>' for a in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+h(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
NAMES={'apple':'原有 Apple 自動模式','v02':'v0.2 研究候選','apple_features':'Apple 分軌後分類','classifier':'頻譜分類器','mixture':'混合與空間模型','transfer':'YAMNet 遷移學習','fusion':'Apple＋YAMNet 融合','transformer':'Audio Spectrogram Transformer'}
LAB={'guitar':'吉他','piano':'鋼琴','speech':'人聲','train':'列車','wind':'環境風聲'}
COND={'persistent':'兩種聲音持續重疊','intermittent':'目標聲間歇出現'}
def make():
    sel=read('evaluation/selection.json'); name=sel['primary']
    sec=[]; data={}
    for cand in ('apple','v02',name):
        for cond in COND:data[(cand,cond)]=read(f'evaluation/fresh/{cand}/{cond}.json')
    met=table(('條件','方法','Accuracy','F1','精確率','召回率','分軌改善 ≥3 dB'),[
        (COND[c],NAMES[m],pc(row['accuracy']),f"{row['f1']:.3f}",pc(row['precision']),pc(row['recall']),f"{row['useful']}/{row['TP']+row['FN']}") for c in COND for m in ('apple','v02',name) for row in (data[m,c]['summary'],)])
    sec.append('<h2>新錄音上的結果</h2>'+met)
    passing=all(data[name,c]['summary'][k]>=.9 for c in COND for k in ('accuracy','f1'))
    sec.append('<p>這裡的正例指音訊確有指定類別，負例則要求一個依資料標記缺席的類別。系統只要交軌就記為「有」。另用乾淨參考音軌評量，分離後 SI-SDR 比混音提高至少 3 dB 才算有用。未交付仍計入分母。</p>')
    rows=[]
    for c in COND:
        item=data[name,c]; by=item['requested_class']
        for category in LAB:
            s=by[category]
            rows.append((COND[c],LAB[category],f"{s['f1']:.3f}",pc(s['accuracy']),f"{s['useful']}/{s['TP']+s['FN']}",s['FP']))
    sec.append('<h2>哪些聲音容易混淆？</h2>'+table(('條件','要求的聲音','F1','準確率','有用的正例','誤交軌'),rows))
    ci=[]
    for c in COND:
        info=data[name,c]['cluster_bootstrap_95']; ci.append([COND[c]]+[' – '.join(pc(v) for v in info[k]) for k in ('accuracy','f1','useful_fraction')])
    sec.append('<h2>樣本與不確定性</h2><p>主測試使用 60 段本輪新取得的錄音，五類各 12 段，形成 30 個不共用來源的配對。每個條件有 60 次目標存在與 60 次目標缺席的請求。同一對素材會重複用於不同條件。下表按共用上傳者的配對分組，重抽 2,000 次估計 95% 區間。</p>'+table(('條件','準確率','F1（百分制）','有用分軌率'),ci))
    sec.append('<h2>訓練與選模</h2><p>原有訓練集為 175 段，本輪實際加入 450 段未使用過的錄音。不同候選使用的數量見各模型訓練資料。驗證集與本次訓練集按上傳者分開。依事先設定的規則，從兩種驗證條件的準確率與 F1 取四個分數的最低值，選出主模型，再測封存的新錄音。公司內建或第三方模型的原始預訓練資料可能含網路上重傳的聲音。</p>')
    if 'ranking' in sel:
        sec.append(table(('候選','最弱的驗證指標','最弱的有用分軌率'),[(NAMES.get(r['name'],r['name']),pc(r['worst_detection']),pc(r['worst_useful'])) for r in sel['ranking']]))
    if (HERE/'stress/user_summary.json').is_file():
        ss=read('stress/user_summary.json')
        sec.append('<h2>換成天然左右聲道與強混響</h2>'+table(('條件','準確率','F1','有用分軌','輸出後比混音差'),[(r['label'],pc(r['accuracy']),f"{r['f1']:.3f}",f"{r['useful']}/{r['positive']}",r['harmful']) for r in ss]))
    if (HERE/'unknowns/results.json').is_file():
        unknown=read('unknowns/results.json')[name]
        sec.append(f"<p>額外的類別外負對照取自 30 個獨立上傳者的狗吠、貓叫、鐘聲、時鐘與滴水錄音。按標注這些混音都沒有五個目標類別，模型在 {unknown['requests']} 次請求中仍交出了 {unknown['false_accepts']} 條軌。標籤來自資料集，並非逐秒人工核驗。</p>")
    sec.append('<h2>適用範圍</h2><p>主測試把真實錄音混成兩個穩定、可分開的左右聲像。手機單鏡頭現場、任意數量聲源、真實移動與任意新類別的效果，須用相應資料另外量測。這輪的準確率與 F1 涵蓋人聲、鋼琴、吉他、列車、環境風聲五類。「人聲」代表是否有人說話，沒有衡量說話者身分。</p>')
    style=':root{color:#193136;background:#eef1ed;font:16px/1.7 -apple-system,BlinkMacSystemFont,sans-serif}body{margin:0}main{max-width:980px;margin:36px auto;padding:clamp(18px,5vw,58px);background:#fff}h1{font-size:clamp(32px,6vw,54px);margin:.28em 0;line-height:1.2}h2{font-size:24px;margin:1.6em 0 .45em}p{max-width:78ch}.intro{font-size:19px}.wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;border-bottom:1px solid #d2ddd8;padding:8px 11px}th{background:#eef5f1;font-size:14px}header small{color:#37675e}footer{margin-top:36px;border-top:1px solid #d2ddd8;padding-top:12px}'
    result='達到' if passing else '未達到'
    title='五類聲音的 90% 測試'
    doc=f'<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voice Hunter · {title}</title><style>{style}</style><main><header><small>Sound Cocoon · Voice Hunter 0.3 研究版 · 2026-11-21</small><h1>{title}</h1><p class="intro">本輪主模型為 {NAMES[name]}。在全新受控測試的兩種條件中，{result}準確率及 F1 均至少 90% 的預設門檻。</p></header>'+''.join(sec)+'<footer>逐筆數值、選模紀錄及可重跑的評分程式收錄於下載版 round3/evaluation。資料採 FSD50K 與個別錄音的公開授權，逐檔來源與作者見資料清單。</footer></main></html>'
    (HERE/'REPORT.html').write_text(doc)
if __name__=='__main__':make()
