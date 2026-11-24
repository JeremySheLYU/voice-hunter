# Voice Hunter 0.3 研究版

在 Mac 上從立體聲錄音或影片提取聲源、獨聽每一軌，然後調整聲像。內建耳機模式可指定前方水平面的角度。檔案在本機處理，示範錄音可以直接開啟。

完整工程報告收錄系統架構、公開資料的計數口徑、基準比較、壓力測試及重現步驟。[閱讀報告](docs/Voice_Hunter_Engineering_Report_zh-Hant.md) · [下載 PDF](docs/Voice_Hunter_Engineering_Report_zh-Hant.pdf)

## 五類自動分軌

在 Mac 上安裝本版的 Python 依賴後，雙擊 `Start-Voice-Hunter.command`，匯入立體聲影片或錄音，在「分離方式」選「五類自動分軌 · 固定聲像（實驗）」，再選要提取的聲音。音色與空間的兩組判斷使用 Apple SoundAnalysis 與本機 YAMNet，輸出目標和剩餘兩軌。「人聲」涵蓋說話聲，人物身分追蹤請選下方雙人模式。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m voicehunter process input.mov --mode five-events --target guitar --out results/guitar
```

本輪新增 450 段訓練錄音並比較六個候選。依上一輪保留的驗證集先定下主模型，再在 60 段全新錄音的固定聲像測試中評分。連續重疊時準確率 86.7%、F1 0.873，間歇聲為 85.0%、0.850。保留天然左右聲道的另一組新錄音測得 61.8%、0.442。這個選項適合離線試聽，目標和剩餘軌可分別核對。

- `round3/REPORT.html` 收錄完整成績、置信區間與限制。`round3/listen/index.html` 可離線比較首個列車及鋼琴測試片段。
- `round3/evaluation/selection.json` 是按驗證成績鎖定的候選順序。`round3/evaluation/fresh`、`round3/stress` 與 `round3/unknowns/results.json` 收錄獨立測試逐筆數據。
- 重跑評估前，將另附的公開音檔包解壓在專案內，對照其 README 將音檔放到列明的相對位置，或執行 `python3 round3/data/restore.py`。舊資料的下載方式見 round2/envdata/README.md。主模型所需的推論權重已打包。比較用的 AST 編碼器可執行 `python3 round3/transformer/restore.py` 下載並驗證。

## 立體聲與耳機功能

- **固定位置或移動中的兩人。** 原有固定位置模式繼續可用。「雙人對話 · 移動追蹤（實驗）」加入用公開朗讀資料訓練的音色距離。受控換位盲測由 0/40 改善到 35/40；實測樓梯間響應下則全部拒絕。先獨聽各軌確認人物。
- **兩個固定方位的聲源。** 選「雙聲源 · 空間與音色分軌」，為目標和另一聲源各標清楚的示範段。在保存原始左右聲像的 12 組天然錄音組合，三軌 CLI 會退回頻譜模式，兩版同為 4/12。聲源移動、方位模糊或要保留環境底聲時，可選原本的示範模式。Apple 自動類別選取沿用前版。
- **耳機比較。** 保持音量、角度和背景寬度相同，切換「耳機 · 幾何模型」和「耳機 · 實測雙耳模型」。新模式校準左右耳的頻率能量差和時間差；嚴重反相的局部片段會平滑退回原來的立體聲。沒有個人化與頭部追蹤。
- **公開試聽與數據。** 打開 round2/REPORT.html 查看中文結果，round2/listen/index.html 可離線比較倫敦現場環境加上已知方位的人聲，含出處與響度校準。

~~~sh
python3 -m voicehunter process examples/duet.wav --mode speakers_moving --span 0.2 1.65 --background 2.2 3.65 --out results/moving
python3 -m voicehunter process examples/studio.wav --mode spatial-example --span 0.05 2.05 --background 2.35 4.30 --output-mode headphones-measured --azimuth 45 --out results/objects
python3 -m voicehunter.multitrack examples/studio.wav --config examples/multitrack.json --separator adaptive --out results/named-spatial
~~~

如已有確定聲道順序的 AmbiX，可在專案根目錄執行以下命令，輸出一對朝向 ±45° 的虛擬心形麥克風。適用 ACN/SN3D 的 4、9、16 或 25 聲道；高階檔在這個投影中使用一階分量 W、Y、X。原生 5.1、物件式聲軌及未標註的四聲道不能假定為 AmbiX。輸出可接續匯入工作台。

~~~sh
python3 -m voicehunter.ambix field.flac --layout acn-sn3d --half-angle 45 --out field-stereo.wav
~~~

## 開始試聽

解壓縮後，雙擊 `Start-Voice-Hunter.command`。瀏覽器會開啟混音台。保持終端機視窗開啟，結束時按 Control+C。若用終端機，請在解壓後的目錄執行：

```sh
python3 -m voicehunter serve --port 8845
```

然後開啟 <http://127.0.0.1:8845/?demo=studio>。點「提取聲源」，先獨聽目標和其餘聲音，再點「加寬環境」和「套用混音」。播放中切換 A 原音與 B 混音。雙人模式的內建例子在 <http://127.0.0.1:8845/?demo=duet>，裡面有兩個不同方位的合成人聲型音色。

在自己的錄音上使用時，匯入有兩聲道的 MOV、MP4、WAV 或其他支援格式。每段最長 60 秒，最多 160 MiB。程式取第一條音軌，轉為 48 kHz 工作取樣率。可下載兩條 32-bit float WAV 分軌、24-bit PCM 混音，以及沿用原影片畫面的 MOV。瀏覽器的匯入和輸出是臨時檔，請下載要保留的成果。此版本在桌面離線運行，瀏覽器負責操控與播放。

## 執行環境

本次已測試 Apple Silicon Mac、Python 3.14、NumPy 2.4 與 FFmpeg。建議 Python 3.10 以上，需有 `ffmpeg` 和 `ffprobe`。若另一台 Mac 已安裝 Homebrew，但缺少依賴，可執行：

```sh
brew install ffmpeg python
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Apple 自動分類還需 macOS 的 SoundAnalysis 和可用的命令列開發工具。可用 `xcode-select --install` 安裝系統工具。RNNoise 模式需要 FFmpeg 包含 `arnndn` 濾鏡，隨專案附上原始模型。程式啟動時檢查可用功能；模型啟用細節及授權列在 `voicehunter/models/NOTICE.md`。在終端機可用 `python3 -m voicehunter modes` 查看本機啟用的模式。

## 分軌與混音

| 模式 | 怎樣用 | 已知限制 |
|---|---|---|
| 示範片段 | 標一段主要是目標聲音的時間；可另標一段背景。以兩者的頻譜學習非負字典，將估計的遮罩套用到原始左右聲道。 | 示範內容會影響整段結果，兩種聲音音色相近時容易串音。 |
| Apple 自動辨識 | 選說話、鋼琴、吉他、音樂、列車、環境風或麥克風風噪。系統分類器找出代表性時段，再由頻譜模型實際提取聲音。 | 需要有足夠分數與時間變化；同一聲音全程持續、過短或太弱時，會提示改選示範段。辨識分數是事件線索，不能換算成軌道純度。 |
| RNNoise 人聲 | 直接估計整段說話聲，將單聲道人聲估計轉成遮罩，回到原立體聲重建目標和其餘聲音。 | 多位說話者會合在一起；這個降噪網路主要針對語音及干擾聲。 |
| 雙人位置分離 | 各標記 A、B 一段清楚的獨講，至少 0.35 秒。利用每個位置的左右訊號關係分出 A，另一軌含 B 與環境。 | 適合兩人位置穩定。兩人站太近、移動、由同一喇叭播放、或強混響會降低可分性。 |
| Demucs（選配） | 在獨立 Python 環境提供本地四軌模型，驗證後顯示歌聲、鼓、貝斯、其他伴奏。 | 封裝未附 Demucs 權重；本次已驗證未安裝時會隱藏此選項，未做此模型的推理驗證。 |

立體聲模式可調目標音量、相對原位的左右偏移、以及其餘聲音的寬度。預設值（0 dB、原位、100%）加總回原錄音。目標音量不變時，左右偏移和寬度調整保留合回單聲道的結果，最終輸出的防削波增益可能同時降低兩個聲道。

幾何耳機模式把目標軌渲染成位於 −90° 至 +90° 的虛擬點聲源，使用簡化頭部模型的雙耳時間差和頻率相關的遠側耳衰減，背景保留立體聲。角度是你設定的混音參數。0° 是正前方；此版本未建模個人耳廓、上下、後方、頭部追蹤或公尺距離。演算法直接分析音訊，目前畫面用於保留原片及輸出，沒有自動辨認嘴形或追蹤畫面中的發聲者。

## 命令列

在專案目錄執行。每次指定一個新的或空的輸出目錄。

```sh
python3 -m voicehunter process examples/studio.wav --mode example --span 0.05 2.05 --background 2.35 4.30 --pan 0.15 --width 1.35 --out results/studio
python3 -m voicehunter process examples/duet.wav --mode speakers --span 0.2 1.65 --background 2.2 3.65 --out results/duet
python3 -m voicehunter process input.mov --mode rnnoise --output-mode headphones --azimuth 25 --gain 1.5 --width 1.35 --video --out results/headphones
python3 -m voicehunter process input.mov --mode auto --target train --out results/train
```

需要同時輸出多個聲源時，使用多軌命令列。每次可指定 1–8 個目標，支援示範時段及 Apple 已知類別。各目標共同分配同一段立體聲的時頻成分，另有剩餘軌；高度重疊的聲音可能分到多條軌。參見隨附的 `examples/multitrack.json`。

```sh
python3 -m voicehunter.multitrack examples/studio.wav --config examples/multitrack.json --out results/multitrack
```

## 測試與解讀

- `python3 tests/validate_audio.py` 重建兩個已知位置的音色，檢查重疊段的分離、左右能量、單聲道合成及極端參數。此案例目標軌 SI-SDR 改善 26.45 dB。
- `python3 scripts/speaker_demo.py` 建立獨講加重疊的人聲型合成訊號。另以兩段真人錄音混成固定位置的測試，A 軌的重疊段 SNR 從 4.15 dB 提高至 23.10 dB。來源、樣本校驗值及重現方式在 `tests/voice_sources.md`。
- `python3 tests/check_binaural.py` 檢驗雙耳方向、到達時間與頻率衰減；`python3 tests/test_multitrack.py` 驗證重疊目標的聯合分配與可加性。
- `python3 tests/validate_av_sync.py` 驗證有影音起始時間差的影片仍保持同步。

上述是已知聲源或已知時間關係的受控測試。對一般 iPhone 現場錄音，請先獨聽分軌，檢查相互滲漏，再比較整體混音。兩軌相加能還原原音只說明可重建性；好聽程度和真實場景的分離準確度仍需試聽及更多錄音驗證。
