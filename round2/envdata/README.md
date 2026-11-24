# 公開環境聲、樂器與人聲素材

本組收集 330 份獨立投稿，來自 286 位作者，合計 64.95 分鐘。其中有 66 份火車、64 份風、71 份鋼琴、67 份吉他、62 份人聲。206 個來源分到訓練、61 個驗證、63 個測試；各組投稿者完全分離。授權為 CC0 173 份及 CC BY 3.0 157 份。按最終採用的表示，65 份可量到明顯左右差異，18 份雙聲道編碼接近雙單聲道，其餘 247 份採資料集的單聲道版本。每個來源以 Freesound 原始錄音 ID 為單位。完整取得和分析的數量見 summary.json，各類別、資料切分、錄音作者、聲道配置及累計時長均有明列。round2/environment/cohort.json 是演算法組開始試驗時凍結的較小子集。後續下載的錄音保留作未來實驗，並沒有全數用於這次模型擬合。

## 兩種來源表示

- FSD50K 的標準音檔是 44.1 kHz、16-bit 單聲道 WAV，從固定 revision 的資料集鏡像取回。這些素材可以提供聲音類別及獨立波形，模擬混音時才會獲得已知方位。它們自身沒有立體聲方位。
- Freesound 的原始投稿頁另提供 HQ MP3 預覽。本組在原頁重新核對授權，將最多頭 15 秒的真正雙聲道預覽解碼為 FLAC，不作聲道擴充或方位模擬。預覽來源經過有損編碼，把它存成 FLAC 是為了保存解碼結果。獨立量度 12 kHz 的左右訊號差；若差聲功率相對平均聲道功率小於或等於 −50 dB，標示為 near-dual-mono。雙聲道編碼不保證有實測物件角度。若同一 Freesound ID 有兩種表示，最終獨立錄音清單只計一次，優先保留原投稿的雙聲道版本。
- FSD50K 是弱標註的個別事件／音樂音訊。標為列車的錄音可以包含車站背景，標為風的來源可能是音效或現場收音。沒有從標籤推斷每個檔案都是在戶外實錄，也沒有把數個不同投稿的軌道當作同一次活動中同期錄下的多軌。
- 每份收錄音訊的 Freesound 投稿授權是 CC0 或 CC BY。授權網址、作者、標題、投稿連結、下載位址、表示方式與雜湊分別在 all_sources.json 及 attribution.csv。FSD50K 資料集本身採 CC BY 4.0，原授權摘錄保存在 LICENSE-FSD50K.txt。原作者為 Eduardo Fonseca、Xavier Favory、Jordi Pons、Frederic Font 與 Xavier Serra，資料集論文為 *FSD50K: An Open Dataset of Human-Labeled Sound Events*。來源包括 [官方版本](https://doi.org/10.5281/zenodo.4060432)、[官方探索介面](https://fsannotator.upf.edu/fsd/release/FSD50K/) 及[固定版本的 WAV 鏡像](https://huggingface.co/datasets/Fhrozen/FSD50k/tree/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b)。

## 切分與重現

資料擷取前以正規化的投稿者名稱雜湊決定 train、validation 或 test，同名投稿者不會跨組。選取時優先有更多不同作者、主類別清楚及不含其他目標類別的錄音。排除 ESC-50 清單的全部 Freesound 源 ID，避免上一輪保留題的來源又進入訓練。最終 audit 對檔案完整性、重複 ID 和雜湊、跨組的五個時段波形近似重複進行檢查。僅憑公開投稿資訊無法斷言沒有來自同一場地或不同帳號的其他錄製。

在專案根目錄執行以下命令。已收錄音檔放在 round2_data/environment/，portable 程式包以清單及還原程式取代數百 MB 的原始資料。預設只還原最終獨立來源。要重跑已凍結的訓練，再加 --all-versions 即可同時還原某些來源的兩個表示。還原工具對下載內容及產物作 SHA-256 核對。相同的 FFmpeg 8.0.1 編碼器版本有助逐位相同地重建 FLAC。

~~~sh
python3 round2/envdata/restore.py --all-versions
~~~

如要重新選取一批可用投稿，可執行 collect.py（需要網路），再執行 audit.py。網站可用性會影響重新選取結果。重現本輪來源應使用已存清單及 restore.py。為避免更改既有結果，建議在單獨解壓的目錄進行重選。
