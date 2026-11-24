# 實測雙耳錄音、空間響應與原生 5.1 電影素材

本目錄的程式可以自正式發布網址重建實際取得的子集。兩種使用結果最值得注意。外部實錄 BinauRec 的 RNNoise 語音改善中位為 +4.69 dB，仍有一例嚴重漏聲。使用一部電影訓練的 5.1 降混係數則在另一部電影退步，因此沒有加入產品。

| 使用者想研究的素材 | 本次實際取得 | 用途及界限 |
| --- | --- | --- |
| SATP 倫敦城市雙耳錄音 | 27 段，合計 810.04 秒，另有一個 60 秒校準音。真人佩戴 HEAD acoustics BHS II 錄音，原生 48 kHz、24-bit、L/R。校準音為 16-bit。 | W06 用作示範底音，其餘作外部保留；多個地點在同片區，沒有聲源單軌或真值方位。 |
| Aachen AIR v1.4 | 完整 202,650,373-byte 原始發布檔。內附的 344 個單聲道 WAV 中，可明確對齊的 224 個合成 112 對 L/R，跨五個房間標籤，包含 95 對有假人頭、17 對無假人頭。其餘 120 個電話測試 WAV 保留原形式。 | 逐對保留原位元深度及 48 kHz；左為原編號 1、右為 0。真實樓梯間 ±45° 與正前方的脈衝響應可重現傳播、頭部和室內反射效果。另有原始 MAT 測量留在官方發行包內。 |
| Huddersfield St Paul’s Concert Hall | 完整 13 個 4 聲道 AmbiX FOA 測點響應，另取距中央 1 m 的 KU100 SOFA，100 個頭部轉角 × 2 耳 × 240,000 個 48 kHz 樣本。 | 所有測點只來自一個音樂廳。各角度代表同一量測系統，不能算作 100 個獨立場景。 |
| BinauRec | 36 個不同語音素材 ID，6 種揚聲器幾何及 3 個標稱 SNR。每個條件取得真實錄下的語音、穩態語音形噪音及同時播放的混音，共 108 個雙聲道檔。實際混音共 332.77 秒。 | 法國 Nancy 同一房間，假人頭兩側助聽器的前麥克風，取原始 16 kHz PCM；原素材與噪音來自 LibriSpeech。固定測試集，沒有用於調參。 |
| Blender 電影音軌 | 《Big Buck Bunny》及《Sintel》分別取 60、180、300、420 秒開始的 12 秒，保留兩套原生 master，構成 8 個時間窗與 16 個音檔。原生 stereo 與 5.1 的位元率可不同，均為 48 kHz。 | 5.1 為 FL、FR、FC、LFE、BL、BR 聲道，不含 Dolby Atmos 物件或高度資訊。分割依整部電影；同段不同格式屬於同一節目內容。 |

## 兩項外部檢查

在 BinauRec 上測試既有人聲分離器。36 例中，35 例提升 SI-SDR，33 例提升至少 1 dB。中位改善 +4.69 dB，範圍 −50.45 至 +7.72 dB。最差的一例為 scenarioF_snr-5_11604，輸出目標軌 RMS 為 −48.89 dBFS，真實的語音參考為 −26.54 dBFS。物理上另行播放語音及噪音，可能存在環境變化、時序及電平差異；兩個獨奏波形加總對真實混音的還原信噪比中位為 15.46 dB。另用獨奏聲軌在軟體內準確相加再計分，中位 +4.87 dB，同一最差案例下降 51.17 dB。此檢查量到了現有模型在新環境的行為，未比較訓練前後改進。

以 Big Buck Bunny 的 4 段資料擬合 4 個鏡像對稱的 5.1 → stereo 線性係數，將對應的原生立體聲 master 作參考。對照固定權重為前方 1、中央 0.707、同側環繞 0.707，兩者均捨棄 LFE。對訓練電影改善中位 +2.69 dB，換成整部未參與訓練的 Sintel，四段中三段退步，中位 −4.53 dB。未進行時間平移或因測試片重新調整模型。兩種 master 可有獨立混音及母帶處理，時間對齊誤差也會影響波形比較。這個 5.1 降混候選只留作負面基準。

## 來源與重跑

來源、作者、權利及地點在 sources.json，Aachen 的授權全文在 AIR_LICENSE.txt。SATP、BinauRec 及 Huddersfield 錄音採 CC BY 4.0。BinauRec 此子集只取 LibriSpeech 朗讀和以其製作的語音形噪音。乾淨語料由 Vassil Panayotov、Guoguo Chen、Daniel Povey、Sanjeev Khudanpur 等整理（LibriSpeech ASR corpus，OpenSLR 12，CC BY 4.0）。影片音訊來自 Xiph 提供的原生電影 master。© 2008 Blender Foundation | www.bigbuckbunny.org。© Blender Foundation | www.sintel.org。兩部電影採 CC BY 3.0。裁切及重新 FLAC 封裝為本次修改，並已比對輸出逐樣本與所取區間一致。相關官方來源可見下方。

- [SATP 官方音檔與條款](https://zenodo.org/records/18362685)
- [Aachen AIR 官方版本](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
- [Huddersfield FOA 與 BRIR](https://zenodo.org/records/2641166)
- [BinauRec 實錄](https://zenodo.org/records/7256984) · [LibriSpeech](https://www.openslr.org/12/)
- [Xiph 原生混音 master](https://media.xiph.org/) · [Big Buck Bunny 授權](https://peach.blender.org/about/) · [Sintel 授權](https://durian.blender.org/sharing/)

執行環境需要 Python、NumPy、FFmpeg 和 ffprobe。SOFA 格式檢查額外使用 h5py。各個下載程式預設寫入專案根目錄內的 round2_data/spatial。下載時驗證完整壓縮檔雜湊或每個選定成員的 CRC32 及檔案大小；切片來源至少保存各自的內容雜湊。個別下載和量測可獨立執行。

~~~sh
python3 round2/spatialdata/fetch_collections.py
python3 round2/spatialdata/fetch_binaurec.py
python3 round2/spatialdata/fetch_apl.py
python3 round2/spatialdata/index_verified.py
python3 round2/spatialdata/verify_measured.py
python3 round2/spatialdata/films.py
python3 round2/spatialdata/film_downmix.py
~~~

spatial_validation.json 包含 262 個本機檔案（不含後補的 16 個電影節錄）的驗證數量及資料形狀。satp_index.json、aachen_index.json、spatial_validation.json 和 films_index.json 分別收錄 SHA-256 與配對關係。binaurec_result.json 和 film_downmix_results.json 保留逐例成績。每位說話者是否獨立無法由 BinauRec 的素材 ID 本身確立。
