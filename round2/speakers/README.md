# 說話者換位追蹤

新版提供兩個可獨立選用的模式。固定位置使用原本的 `speakers`；移動測試採 `speakers_moving`。後者以兩段單獨人聲校準兩條空間路徑，再用學自公開朗讀錄音的聲音特徵距離與序列判斷追蹤身分。若示範段的兩條估計軌串音太多，會提示重新選段。模型保留 30 個特徵權重，沒有保留個人聲紋模板。

## 資料與授權

完整取得三個官方 OpenSLR 壓縮檔，合計 805,664,639 bytes，逐一符合官方 MD5；另外記錄 SHA-256。解讀完整 FLAC 標頭後，共 5,228 段、12.75 小時、94 位不重疊的朗讀者。原音為單聲道。方向性測試把獨立人聲製成立體聲混音，另在完全相同語音上套用真正量測的雙耳房間響應。

| 官方資料 | 原始段數 | 說話者 | 來源時長 | 本次使用 |
|---|---:|---:|---:|---|
| Mini LibriSpeech train-clean-5 | 1,519 | 28 | 5.31 小時 | 每段解碼，從全段均勻取樣，形成 8,650 個有聲的一秒片段、54,628 個配對比較，訓練辨別是否為同一說話者的距離權重。 |
| Mini LibriSpeech dev-clean-2 | 1,089 | 26 | 2.04 小時 | 全部建檔，以全部 26 位讀者的 104 段不同語句製作開發場景，用來改進跨時間追蹤與串音拒絕規則。 |
| LibriSpeech test-clean | 2,620 | 40 | 5.40 小時 | 全部建檔，以全部 40 位讀者的 160 段不同語句製作最終測試。評分全程固定模型。 |

LibriSpeech © 2014 Vassil Panayotov，CC BY 4.0。研究著作作者包括 Vassil Panayotov、Guoguo Chen、Daniel Povey、Sanjeev Khudanpur。從原音轉碼、裁切、取得聲學特徵並訓練權重。來源 [OpenSLR 31](https://www.openslr.org/31/)、[OpenSLR 12](https://www.openslr.org/12/)、[授權條款](https://creativecommons.org/licenses/by/4.0/)。

Aachen stairway 取自 RWTH Aachen AIR Database（© 2011 RWTH Aachen University，MIT），實際使用 3 m、左右 ±45° 兩個幾何位置，各有一對原生左右耳脈衝響應。將乾淨獨立語音各自與相應雙耳響應進行卷積，再用半秒交叉淡化模擬換位。兩個位置皆為同一間樓梯間，並非兩間房間或連續現場錄音。詳細授權及來源見 round2/spatialdata 或實際來源清單。[RWTH 資料頁](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)

## 最終盲測結果

使用官方 test-clean；40 位全新朗讀者構成 20 對，每對兩組不重複語句。兩人各需一段 1.08 秒校準，重疊期間各自唸另一段新內容。嚴格依原身分比對，兩軌在兩人均有聲的換位後時段都較未分離的原音改善至少 3 dB，才算一組通過。固定位置案例使用同一個計分時段。測量為保留增益及波形誤差的 SNR 改善，並非真人聽覺分數。

| 場景 | v0.1 通過 | 新移動模式通過 | 新模式拒絕 | 新模式交出劣於未處理的軌 |
|---|---:|---:|---:|---:|
| 簡單聲學，固定 | 40/40 | 38/40 | 0 | 1 |
| 簡單聲學，換位 | 0/40 | 35/40 | 0 | 3 |
| 實測樓梯間，固定 | 2/40 | 0/40 | 40 | 0 |
| 實測樓梯間，換位 | 0/40 | 0/40 | 40 | 0 |

簡單聲學的換位條件，取兩軌中改善較小者，40 組的中位數為 +25.81 dB，v0.1 為 −3.50 dB；兩版都使用相同錄音與混音。新的移動模式在固定位置少兩組通過，因此它沒有過「固定場景無退步」門檻，原本的固定位置模式繼續可用。實測樓梯間串音過高，新的移動模式沒有交付任何成功分軌。

原版 54 案凍結回歸中，固定預期用途 24/24 通過，位置交叉 0/6 提升到 5/6；一對真人讀者的身分仍未能跟上。拒絕由 10/54 增至 24/54，交出的有害結果由六個減至一個。FSDD 公開數字錄音僅在這項既有回歸測試中使用，沒參與學習或清楚度最終驗收。

## 重現

需要 Python、NumPy、FFmpeg。請先準備約 1.7 GB 可用空間。若 QA 資料與原版程式在同一工作目錄，從專案根目錄執行以下指令。Aachen 命令需要另行提供已準備的左右兩組響應。原 v0.1 ZIP 置於 `outputs/VoiceHunter_v0.1.zip`，用來作逐案對照。

~~~sh
python3 round2/speakers/acquire_and_catalog.py --cache work/round2-corpus/librispeech
python3 round2/speakers/train.py --cache work/round2-corpus/librispeech
tar -xzf work/round2-corpus/librispeech/test-clean.tar.gz -C work/round2-corpus/librispeech
python3 round2/speakers/evaluate.py --cache work/round2-corpus/librispeech --final
python3 round2/speakers/evaluate.py --cache work/round2-corpus/librispeech --final --measured-dir round2_data/spatial/aachen
~~~

`librispeech-files.csv` 收錄完整 5,228 段音訊的檔名、分區、讀者、長度及 SHA-256；`corpus.json` 是三個完整壓縮檔的指紋。`training.json` 記錄參與學習的組別、數量及權重雜湊。`heldout-final.json`、`measured-heldout-final.json` 列出最終測試所有語句、兩個版本的每案得分與拒絕原因。Mini 開發集數字在不帶 final 後綴的檔案。`final-lock.json` 鎖定獨立驗收前的候選碼；驗收後改為讓使用者選用，全部 160 個場景逐項重新計算，與鎖定版本完全相符（integration-verify.json）。沒有把 Mini 開發集結果冒充最終測試。
