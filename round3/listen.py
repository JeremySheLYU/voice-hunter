"""Export the first predeclared train and piano targets, including refusals."""
from pathlib import Path
import json
import html
import subprocess
import numpy as np
from round3.evaluation.evaluate import fresh_scenes,predictor
from round3.build_report import LAB
HERE=Path(__file__).resolve().parent
def main():
    method=json.loads((HERE/'evaluation/selection.json').read_text())['primary']
    plan=json.loads((HERE/'evaluation/fresh-plan.json').read_text())
    by_id={r['id']:r for r in plan['sources']}
    predict=predictor(method);old=predictor('apple')
    out=HERE/'listen';out.mkdir(exist_ok=True)
    sections=[]
    wanted={'train','piano'}
    for mix,t,meta in fresh_scenes(plan,'persistent'):
        c=meta['target']
        if c not in wanted:continue
        wanted.remove(c)
        items=[('原始混音',mix),('獨立目標參考',t),('原有自動模式',old(mix,[c])[c]),('本輪主模型',predict(mix,[c])[c])]
        scale=min(1.,.95/max(float(np.max(np.abs(y))) for _,y in items if y is not None))
        snippets=[]
        for i,(label,w) in enumerate(items):
            if w is None:
                snippets.append(f'<div><p>{label}</p><em>模型未交出此軌。</em></div>');continue
            path=out/f'{c}-{i}.flac'
            subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar','48000','-ac','2','-i','pipe:0','-sample_fmt','s32',str(path)],input=(w*scale).astype('<f4').tobytes(),check=True)
            snippets.append(f'<div><p>{label}</p><audio controls preload=metadata src="{path.name}"></audio></div>')
        credits=[]
        for i in ('source_id','background_id'):
            r=by_id[meta[i]]
            credits.append(f'<a href="{html.escape(r["url"])}">{html.escape(r["title"])}</a> · {html.escape(r["author"])} · <a href="{html.escape(r["license"])}">原作授權</a>')
        sections.append(f'<section><h2>{LAB[c]}與{LAB[meta["distractor"]]} · 提取{LAB[c]}</h2><article>'+''.join(snippets)+'</article><small>剪輯、設定左右位置並混合。原作 '+ ' ｜ '.join(credits)+'</small></section>')
        if not wanted:break
    (out/'index.html').write_text('<!doctype html><html lang=zh-Hant><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1"><title>Voice Hunter 0.3 試聽</title><style>body{background:#eff4f1;color:#183133;font:16px/1.65 system-ui}main{max-width:880px;margin:2em auto;background:white;padding:clamp(16px,5vw,54px)}article{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}audio{width:100%}section{padding-bottom:2em;border-bottom:1px solid #c8d4cd}small{display:block;margin-top:1.4em}p{margin-bottom:.3em}a{color:#08615d}</style><main><h1>五類分軌 · 耳機試聽</h1><p>新測試集內首個列車與首個鋼琴目標，依排定次序展示。每組音檔套用相同增益。</p>'+''.join(sections)+'</main></html>')
if __name__=='__main__':main()
