# Audio Spectrogram Transformer 五類聲源候選

本方法先從兩個固定立體聲聲像取得兩條來源估計，接著為每條來源計算 527 個 AudioSet 類別的聲音特徵。來自 MIT 的 AST 保持凍結。Voice Hunter 用 75 份來源樣本（每類 15 段）訓練五類的嶺迴歸分類器，並以開發錄音選擇類別分數混合權重與交軌門檻。先前的 NumPy、YAMNet 與 Apple 分類器沒有參與這個候選模型的推論。

從專案根目錄重跑訓練：

```sh
python3 round3/transformer/restore.py
python3 -m venv .venv-ast
.venv-ast/bin/python -m pip install -r round3/transfer/requirements.txt
.venv-ast/bin/python round3/transformer/train.py
```

需要 FFmpeg 及來源清單所指的已下載聲音檔案。模型輸入為 16 kHz 單聲道、128 個 Kaldi 對數 Mel 頻帶，依上游的均值與標準差正規化。模型在 CPU 上運行。推論介面為 `round3.transformer.model.predict_scene(stereo_float32, 48000)`，傳回五類的立體聲波形或空值。上方的下載指令會取得鎖定版本並核驗原始檔案，如已經存在便只核驗內容。匯出檔的原始 LFS 校驗值保存在 `upstream_hash.json`，下游訓練可由 `train_sources.json` 重建。預訓練資料是 AudioSet，訓練影片與 Freesound 是否存在同源轉載無法全面查核。

[原始 AST 論文](https://arxiv.org/abs/2104.01778) · [MIT 原始權重](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/tree/f826b80d28226b62986cc218e5cec390b1096902) · [實際使用的量化 ONNX](https://huggingface.co/Xenova/ast-finetuned-audioset-10-10-0.4593/resolve/249a1fbf0286b40e7f1ed687a8ae396997bf7dc6/onnx/model_quantized.onnx)
