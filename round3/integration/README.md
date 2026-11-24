# 命令列與本機工作台驗收

使用發佈版本的兩個入口測試。命令列呼叫 python -m voicehunter process。本機工作台以 python -m voicehunter serve --port 0 啟動獨立程序，由作業系統分配臨時回環連接埠，再經實際的 HTTP 上傳、分軌、渲染及 WAV 下載。結果記錄於 preflight.json 和 full.json。這組工程檢查評估可用性、時間對齊及音軌完整性，辨識準確率另見 evaluation。

## 驗收條件

* 對照 v0.2 原始發佈壓縮檔，比對公開選項和原有示範分軌、固定雙人及 Apple 自動模式。
* 已鎖定的研究模型必須透過 CLI 和 HTTP 選項出現，提供五種聲音類別。
* 同一個立體聲片段必須有完整輸出。要求兩條分軌與混音長度一致、各樣本為有限值，兩條分軌相加和解碼原音的最大振幅差在 0.000002 以內。兩個入口下載的 WAV 應逐個位元相同。
* 對靜音、單聲道來源及不支援的類別，CLI 回傳錯誤且不建立輸出目錄。工作台傳回可讀的錯誤訊息。
* 驗收錄音取自已知訓練來源，僅用於測試功能。使用原始素材切取、等化 RMS、設定固定聲像、混音及輸出 FLAC，完整署名、授權與原作連結見 fixture.json。音源為 eriatarka 的 clean riff.mp3 與 dobroide 的 wind.poplar.02.wav，兩份均採 CC BY 3.0。[吉他原作](https://freesound.org/s/15753/) · [風聲原作](https://freesound.org/s/5997/) · [授權](https://creativecommons.org/licenses/by/3.0/)

## 重跑

從專案根目錄用所選研究模型的 Python 環境執行，需已安裝 FFmpeg。preflight 同時需要初版發佈檔案 outputs/VoiceHunter_v0.2_PublicAudio.zip。full 使用隨附的吉他與風聲混音、五類模型與 selection.json。若要重建測試混音，執行 python round3/integration/fixture.py 並事先還原列出的兩份原音。

~~~sh
python round3/integration/run.py preflight
python round3/integration/run.py full
~~~

Git 儲存庫在 `outputs/VoiceHunter_v0.2_Runtime.zip` 保留 v0.2 發佈版的執行程式與模型。所有 `voicehunter/` 成員逐一保留原始位元組；此較小的 ZIP 與完整歷史發佈包具有不同的外層 SHA-256。若另行提供完整發佈包，回歸程式優先核對該包。
