# Voice Hunter 公開音訊資料包

共 29 個獨立 ZIP，內含 6,610 個來源音訊或量測資料檔。解壓資料為 2,928,205,003 bytes，ZIP 合計 2,706,208,802 bytes。每包內 notices/ATTRIBUTION.json 逐檔記錄作者、來源、授權、轉換及 SHA-256。解壓到同一目錄可還原專案相對路徑。額外的 Public_Spatial_Sampler.zip 收錄其中 11 個檔案（31,175,495 bytes），適合先試聽。

| 資料包 | 檔案數 | ZIP MiB |
|---|---:|---:|
| AIR_01.zip | 232 | 27.9 |
| BinauRec_01.zip | 108 | 50.6 |
| Blender_films_01.zip | 16 | 40.6 |
| CIPIC_01.zip | 35 | 125.3 |
| CIPIC_02.zip | 10 | 35.8 |
| FMA_01.zip | 143 | 137.6 |
| FMA_02.zip | 51 | 53.4 |
| Field_eigenscape_01.zip | 52 | 135.4 |
| Field_eigenscape_02.zip | 12 | 30.0 |
| Field_eigenscape_hoa_01.zip | 8 | 44.0 |
| Field_ihta_ears_01.zip | 102 | 61.7 |
| Field_ihta_foa_01.zip | 102 | 131.0 |
| Field_london_01.zip | 5 | 15.4 |
| Freesound_01.zip | 153 | 113.4 |
| Freesound_02.zip | 104 | 111.8 |
| Freesound_03.zip | 111 | 115.9 |
| HUTUBS_01.zip | 87 | 137.6 |
| HUTUBS_02.zip | 9 | 14.2 |
| Huddersfield_01.zip | 13 | 44.3 |
| Huddersfield_SOFA_01.zip | 1 | 155.6 |
| LibriSpeech_dev-clean-2_01.zip | 1089 | 128.7 |
| LibriSpeech_test-clean_01.zip | 1013 | 138.1 |
| LibriSpeech_test-clean_02.zip | 1103 | 138.1 |
| LibriSpeech_test-clean_03.zip | 504 | 74.9 |
| LibriSpeech_train-clean-5_01.zip | 644 | 138.1 |
| LibriSpeech_train-clean-5_02.zip | 629 | 138.1 |
| LibriSpeech_train-clean-5_03.zip | 246 | 53.1 |
| SATP_01.zip | 16 | 108.0 |
| SATP_02.zip | 12 | 82.1 |

每個 ZIP 均已逐檔重新解碼 ZIP 壓縮、核驗音訊／量測位元組及對應 SHA-256。不同表示會重用同一時間軸，獨立錄音與訓練樣本數見各訓練組報告。C1m.sofa 以原始大小保留於獨立 ZIP，為 140 MiB 常規分包的單一例外。兩段含第三方電視／音樂的倫敦參考刺激只保留在本機，重取來源見 manifest.json。已收錄逐檔內容的原始 tar.gz 和 ZIP 不重複打包。

## 獨立物件格式測試包

ADM_Object_Conformance.zip 包含 EBU 提供的四個原始物件式 ADM WAV、四份作者的 XML 及完整原始 GPL-3.0-or-later 版權和授權文件。四個例子共用同一段 3 秒 440 Hz 音訊，各包含不同位置時間資料。此獨立測試包用於格式驗證。抽樣包和格式包不列入上述訓練／評估來源檔案總數。
