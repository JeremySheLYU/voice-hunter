# 五類聲音的聲學特徵基線

`model.py` 使用頻譜包絡、倒頻譜、時間變化、週期性、頻帶對比和音高類能量，為吉他、鋼琴、人聲、火車和風聲估計五個分數。各段錄音的全段摘要和局部摘要共同進入核函數嶺迴歸。訓練按原始錄音和類別加權，每一段長錄音或樣本較多的類別不會因視窗數目較多取得額外權重。推論使用 NumPy。

兩個左右位置穩定的聲源先以 v0.2 的空間估計分離，再逐軌評估上述五個聲音類別。輸出軌的左右聲道使用估計的原聲像。若無法找到兩個穩定方向或指定類別分數不足，就不交軌。分數用於排序與門檻判定，沒有經過機率校準。

從 Voice Hunter 專案根目錄執行：

```sh
python3 round3/classifier/train.py --extra round3/data/train.json --freeze
```

需要 NumPy、FFmpeg、v0.2 的 `round2/semantic` 評估工具與兩代來源清單所列的公開音訊。只擬合 train 來源；原有 validation 上的兩種場景用於選一個全域門檻，以兩組 Accuracy 和 F1 中的最低值最大為首要準則。訓練與驗證會檢查來源 ID、音檔 SHA-256 和上傳者分組。直接將整段驗證音軌分離與計分後，會儲存 `validation.json`、各案例的分數、`weights.npz` 和凍結碼的 SHA-256。外部測試採用相同權重及門檻。原有資料不加新資料的對照可用 `python3 round3/classifier/train.py` 重跑，結果寫入 `baseline_validation.json`。
