# 耳機方位比較

雙擊 `index.html`。中央原聲和兩種耳機渲染共用同一段公開朗讀與倫敦運河雙耳實錄。選左／右 45° 之後，在播放中按 A 或 B 即可從相同位置比較。按揭曉顯示當次隨機對照。所有音訊皆嵌在頁面；音訊時間 4.75 秒，短暫頭尾淡化。±45° 是後製指定的人聲方位，原本在運河錄製的背景保留自己的左右聲道。旁白不是現場人物。

五個音檔皆為 48 kHz 雙聲道 PCM，各自校準至 −23 LUFS，最高與最低相差 0.02 LU。最高真峰值 −2.36 dBTP。逐檔雜湊與測量在 `calibration.json`。已檢查嵌入檔、程式語法、兩種隨機排序、播放同步、方位切換、拖動、暫停與延後揭曉。程式控制以模擬 Web Audio 時鐘驗證。仍無真人聆聽結果。

前景為 Heather Barnett 朗讀 Jane Austen 的 LibriSpeech/librosa 聲音例子 `198-209-0000.ogg`，CC BY 4.0，截取第 1.00–5.75 秒。背景為 Tin Oberman、Andrew Mitchell、Francesco Aletta、José Antonio Almagro Pastor、Kristian Jambrošić、Jian Kang 的 Soundscape Attributes Translation Project，W06，Regent’s Canal at Little Venice，2019-08-29，CC BY 4.0，截取第 3.00–7.75 秒。改編包含裁切、淡入淡出、音量校準、立體聲混音與前景空間化。正式引用與條款也在頁內。 [LibriSpeech 來源](https://github.com/librosa/data/tree/5285c94e33c74bbe97501f5c9fa95e5e2ce55d48) · [SATP](https://doi.org/10.5281/zenodo.18362685) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

重新建立時，使用校驗符合 calibration.json 的兩個原始聲音，並備妥 NumPy、FFmpeg。

~~~sh
python3 round2/listen/build.py --voice /path/to/198-209-0000.ogg --bed /path/to/W06.wav
node round2/listen/verify.cjs
~~~
