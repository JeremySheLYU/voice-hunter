# Voice Hunter · 第二個研究模型

這個補充包與 Voice Hunter 0.3 的獨立測試分開保存。將補充包解壓至 Voice Hunter 0.3 的專案根目錄，讓 round3b 與 voicehunter、round3 並列。既有工作台使用本輪依驗證規則預先選定的融合模型。

## 使用

在已安裝好 0.3 研究版依賴的 Mac 上，先下載 AST 的鎖定模型檔，再使用下列命令輸出目標軌與剩餘軌。需要 FFmpeg、Apple SoundAnalysis、Python 與 ONNX Runtime。

```sh
.venv/bin/python round3/transformer/restore.py
.venv/bin/python -m round3b.extract input.mov --target piano --out results/challenger-piano
```

## 資料與解讀

AST 分類頭從 75 份擴至 225 份訓練錄音，另外兩組分類頭仍使用 625 份。把同一位上傳者的訓練錄音放在同一折，再以各折留出的預測訓練類別權重。另有一道聲像穩定性檢查，會在混音不符合固定方向假設時拒絕交軌。完整模型、原始碼、鎖定資訊、依賴、重現指令見 model/README.md 與 model/freeze.json。

開發驗證集中，連續重疊的 Accuracy 為 88.2%、F1 為 0.883，間歇聲為 85.5%、0.857。這些數值參與調參；evaluation/regression 收錄重新使用 0.3 素材的逐筆重測，不能當成第二批新錄音。重新測試 76 次天然立體聲目標及 55 次樓梯間目標，這個模型全部拒絕分軌。0.3 工作台保留原先鎖定的融合方法。

嘗試建立另一個互不重疊的 FSD50K 對照組時，核對篩選條件下剩下的 551 個錄音標頭。每類要求 12 位新的上傳者，符合時長的鋼琴為 0 位、吉他和列車各 2 位。另一條以 ESC-50 補充列車、風聲的方案也欠足夠新來源。數據見 data/feasibility.json 與 round3/stress/external/feasibility.json。這批素材無法提供新的五類 90% 確認成績。

## 重新使用 0.3 測資的比較

| 條件 | 準確率 | F1 | 有用分軌 |
|---|---:|---:|---:|
| 兩種聲音持續重疊 | 85.8% | 0.862 | 52/60 |
| 目標間歇 | 85.0% | 0.850 | 49/60 |
