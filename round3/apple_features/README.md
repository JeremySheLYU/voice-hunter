# Apple 聲音特徵分軌實驗

每次推論先根據左右聲道比例估計兩個聲像，再各自在 macOS 本機上計算 SoundAnalysis 事件分數。把每個事件的平均與第 85 百分位作為特徵，平方根變換並正規化，以二次核嶺迴歸學到人聲、吉他、鋼琴、列車與風聲五類的分數。類別映射、核正則化與拒絕門檻使用驗證集選擇。Apple 數值反映聲音事件，不能視為分軌純度。

```sh
python3 -m round3.apple_features.train --extra round3/data/train.json
```

此命令使用 v0.2 的 175 個訓練來源與本輪 450 個新增的訓練來源。原始訓練子集的驗證結果在 old_training_validation.json，最終鎖定參數和來源在 freeze.json。只有訓練腳本會快取公開訓練音檔的特徵。
