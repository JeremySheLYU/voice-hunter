# Voice Hunter v0.3 公開音訊

## 內容

本包收錄 450 份新訓練錄音、87 份受控混音保留測試錄音，以及 19 份保留原有左右聲道的立體聲壓力測試片段。另有 30 份不同類別的負對照（犬吠、貓叫、鐘鈴、時鐘、滴水），用來檢查在目標缺席時是否誤交音軌。總計 586 個檔案，對應 579 個不同 Freesound 錄音 ID。每項錄音的作者、標題、原始頁面、授權、原始下載網址與成品 SHA-256 收於 `round3/data/all_attribution.csv`。所有音檔放在 `round3_data/`。

最初兩組使用 FSD50K 公開的標準化單聲道 WAV。每個訓練聲類 90 份，每份長於或等於兩秒；保留測試每份長於或等於 4.9 秒。兩個聲源混合時的方位另行指定，原始 WAV 只有一個聲道。獨立立體聲壓力組擷取 Freesound 高音質預聽前十五秒內、最少 4.9 秒的部分，解碼為 FLAC；左右兩聲道均保留，通過側聲道能量檢查。與單聲道保留組重複的錄音 ID 清單見 `round3/data/delivery.json`。測試組不參與本輪模型擬合。

## 檔案與重現

- 使用 Python 3，在解壓目錄執行 `python3 round3/data/verify_package.py` 可逐一核對所有音檔 SHA-256。
- `round3/data/train.json`、`test.json`、`native.json` 分別提供訓練、保留測試和原生立體聲的音檔詳情。
- `round3/data/restore.py` 可由 FSD50K 鏡像的固定版本重新下載兩組單聲道 WAV，並核對 SHA-256。
- `round3/data/restore_native.py` 可依每項公開預聽網址擷取及解碼立體聲片段。網上原檔或 FFmpeg 版本可能改變；SHA-256 以交付的 FLAC 為準。
- `round3/data/freeze.json` 和 `native_freeze.json` 記錄取得音訊前固定的條件。各 `plan.json` 是包含合格候選的隊列，不代表實際使用了其中所有錄音。實際收錄以三份音檔清單為準。
- `round3/data/audit.json` 與 `native_audit.json` 提供實際數量、授權與稽核結果。

## 授權與來源

FSD50K 資料集由 Eduardo Fonseca、Xavier Favory、Jordi Pons、Frederic Font、Xavier Serra 等人建立，整體依 CC BY 4.0 授權。本包採用官方弱標籤作為分類依據，取用可依 CC0 或 CC BY 再利用的 Freesound 錄音；各檔的原始創作者與適用版本以 CSV 逐檔標示。修改包含部分錄音採取前段、音訊解碼／重新封裝，研究混音和模型輸出不包含在這份原音包。

FSD50K 官方資料與文件 https://zenodo.org/records/4060432

FSD50K 固定鏡像 https://huggingface.co/datasets/Fhrozen/FSD50k/tree/ccf1acaff12f3f4a4c10052dddafbb6d22152c9b

授權說明 https://creativecommons.org/licenses/by/4.0/
