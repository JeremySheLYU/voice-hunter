# 水平耳機聲像的實測校準

Voice Hunter 提供可選的 `headphones-measured`。近耳保留未染色的乾聲；遠耳用從真實耳部量測學出的頻率響應差和到達時間差建立 FIR。正中位置仍輸出等量左右耳，環境軌原有的位置和寬度控制繼續有效。個人耳形、仰角、頭部追蹤、後方定位和場景距離均未建模；80° 至 90° 使用 80° 的端點值。這是群體的水平雙耳線索模型，沒有從畫面或錄音反推出聲源座標。

## 使用了哪些資料

| 資料庫 | 下載及驗證 | 擬合或評估 |
| --- | --- | --- |
| UC Davis CIPIC | 45 個原始 MAT，181,373,400 bytes，56,250 個有左右耳脈衝響應的聲源位置；另存兩份使用說明 | 43 位真人依預定雜湊切分為 27 位訓練、6 位驗證、10 位最終測試。兩組 KEMAR 是同一假人不同耳廓，另列。 |
| TU Berlin HUTUBS | 全部 96 個實測 SOFA，159,182,969 bytes，42,240 個聲源位置；另存作者使用說明 | 全部只用於跨資料庫測試。93 個獨立真人、1 個人類重測、2 個同一假人重測，各自報告。 |

訓練實際用了 486 個 CIPIC 水平方位，左右互為鏡像合併。每個角度和頻帶以訓練者的中位數擬合，沒有使用未見受試者的值。18 個頻帶涵蓋 250 至 12,000 Hz，時差用 300 至 1200 Hz 的雙耳相位斜率估計。訓練完全排除 ±10°、±35°、±55°，檢驗方向插值。98,490 是下載的原始位置數量，含多數角度及仰角未用於這款水平模型；請勿將這個數字當作訓練樣本數。

## 相較於原版的誤差

結果以演算法真正輸出的波形和實測左右耳的差值計算；數值越低越接近該受試者的量測，沒有採集真人聆聽偏好。

| 評估群組 | 方位筆數 | 雙耳聲級差 MAE，舊→新 | 低頻雙耳相位斜率時差 MAE，舊→新 |
| --- | ---: | ---: | ---: |
| CIPIC 驗證 6 人 | 144 | 4.21 → 1.61 dB | 87.02 → 82.42 µs |
| CIPIC 最終測試 10 人 | 240 | 4.31 → 1.53 dB | 69.19 → 59.68 µs |
| 上列 10 人且角度也未見 | 60 | 4.22 → 1.59 dB | 69.45 → 64.50 µs |
| HUTUBS 外部測試 93 人 | 1,488 | 6.05 → 1.79 dB | 65.03 → 42.61 µs |

CIPIC 最終測試的聲級差誤差在 10/10 人下降，時差在 9/10 人下降；驗證組的時差則只有 3/6 人下降。HUTUBS 93 人的兩項平均誤差都下降。兩組使用不同量測系統、取樣方向與距離，其中 93 人 × 16 方向不能當作 1,488 位獨立聽者。聚合頻帶的誤差下降不能代替水平方位聆聽測試、頭外感或音質偏好結果。鏡像、中央、人聲空白時的底聲、長度和有限值等控制在 8、16、44.1、48、96、192 kHz 通過。原有 `tests/check_binaural.py` 亦通過。原始版與校準版都保留原版的中點聲源及反相限制。

## 重現

從專案根目錄執行。應用與 CIPIC 訓練僅需 Python 和 NumPy。額外檢查 HUTUBS 的標準 SOFA 檔需要 h5py。下載程式對 CIPIC 的不可變 Git 物件雜湊與所有已登錄的檔案雜湊作核對。

~~~sh
python3 round2/spatialtrain/fetch.py
python3 round2/spatialtrain/train_evaluate.py
python3 round2/spatialtrain/fetch_hutubs.py
python3 round2/spatialtrain/external_evaluate.py
~~~

`protocol.md`、`external_protocol.md` 描述評估前固定的門檻。`manifest.json`、`hutubs_manifest.json` 列出輸入、網址與 SHA-256。`results.json` 和 `external_results.json` 列出逐人誤差。`individual_direction_errors.json` 保存 CIPIC 逐方向結果。`voicehunter/models/cipic_horizontal_v1.json` 是實際部署的擬合參數。

## 權利與引文

CIPIC HRTF Database，V. R. Algazi、R. O. Duda、D. M. Thompson、C. Avendano，2001。Copyright (c) 2001 The Regents of the University of California. All Rights Reserved. 作者允許教育、研究與商業使用，每次複製須保留原權利告知。模型旁已附 `CIPIC-NOTICE.txt`；原文另請商業產品的開發者以書面致謝。原始碼鏡像僅負責轉載。 [原論文](https://escholarship.org/uc/item/3d10j9jw) · [原始檔與權利告知](https://github.com/amini-allight/cipic-hrtf-database/blob/7664a1a4fa1ab6b3e14502d1c28fa43a653e5a7f/read_me.txt)

HUTUBS，Fabian Brinkmann、Manoj Dinakaran、Robert Pelzer、Jan Joschka Wohlgemuth、Fabian Seipel、Stefan Weinzierl 等，TU Berlin，2019。原作者使用 CC BY 4.0 公開量測檔。本項擷取頻帶及時差，作外部評估。 [資料 DOI](https://doi.org/10.14279/depositonce-8487) · [作者文件與授權](https://sofacoustics.org/data/database/hutubs/Documentation.pdf) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
