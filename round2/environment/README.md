# 固定方位聲源的立體聲分軌

v0.2 新增選用的「固定方位」投影。給每個物件一小段各自發聲的示範，演算法從示範逐頻估計左右聲道的空間協方差，再把原來的頻譜估計與雙通道 Wiener 濾波器結合。適用於 2–4 個目標、例段有穩定左右特徵的情況。兩聲源有明確參照時也可選用工作台的 spatial-example。多軌設定未指定 separator 時仍使用原有的 spectral 演算法。

## 實際校準的資料與參數

cohort.json 固定 184 個不同 Freesound 錄音 ID（112 訓練、35 驗證、37 測試），按原錄音者分組。實際用於 36 個校準混音的是 **89 個原錄音、78 位上傳者**，涵蓋人聲、風、火車、吉他、鋼琴。這 89 筆有 25 個來源檔為雙通道。給定真值的命名軌先投影為單聲道再設定聲像；另有五個訓練案例保留額外錄音的左右訊號，其中四個有明顯左右差異，一個近似雙單聲道。四個訓練案例另加入 IHTA 同一公園、同一天時段、四個擷取位置的真實四通道 AmbiX 環境聲，解碼成左右 ±45° 的心形指向。聲源資料、分組、授權與雜湊皆列於 cohort.json、fit.json 及各採集說明。

六個候選參數組合，依訓練場景中「三個命名軌裡表現最差的一軌」的平均 SI-SDR 改善選擇。最終空間溫度 0.075、頻譜先驗指數 0.45。實際檢查 12 個獨立錄音者驗證場景，原有方法 4/12、新法 12/12 三軌都改善至少 3 dB。固定這兩個係數後才進入測試。這是有監督的超參數校準，實際分軌仍需要本段示範，並未訓練自動辨認火車的神經網路。

## 結果

每條命名軌相對混音的 SI-SDR 至少提高 3 dB，一個三軌場景才算全部通過。

| 固定權重後的測試 | spectral | adaptive |
| --- | ---: | ---: |
| 上輪三個 LibriSpeech／ESC-50 真聲源固定聲像混音 | 0/3 | 3/3 |
| 另選的 Freesound 測試上傳者、13 個不同混音 | 3/13 | 13/13 |
| 新測試素材的六組人聲＋列車＋風聲，有各自示範 | 2/6 | 6/6 |
| 相同六組加實錄 EigenScape UK AmbiX 背景，低於每軌 6 dB | 1/6 | 6/6 |
| 相同六組加同等音量的 AmbiX 背景 | 1/6 | 4/6 |
| 16 個自然雙通道擷取保留原始 L/R，不加聲像，合成 12 個混音 | 4/12 | 4/12 |

13 個通用混音的 39 軌，原有流程 22/39 通過，新法 39/39，SI-SDR 改善中位 +3.93 → +14.43 dB。三個舊場景的火車軌分別由 +0.88、+2.29、−0.27 dB，提升為 +9.64、+9.12、+11.93 dB。實錄 AmbiX 背景越響，仍有更多未命名的聲音漏入命名軌。以真實背景為參考，剩餘聲軌的 SI-SDR 改善中位，在 −6 dB 條件由 +1.64 變為 −3.99 dB，同等音量由 +0.62 變為 −8.49 dB。採用原版演算法作預設即考慮此取捨。已知命名軌與剩餘軌總和仍可重建原音。上述實錄背景試驗的最差加總誤差為 2.80 × 10⁻⁹ 滿刻度。

自然雙通道套件原有 17 個候選，1 個近雙單聲道依事先門檻排除。其餘 16 個全部使用，12 組均因來源的方向靠近或太分散而退回 spectral，兩法皆為 23/36 條軌達標。這批測試沒有證實自然立體聲錄音的改善。每條真值代表整個原始錄音的擷取片段；裡面可能已經有多個事件。自然雙通道套件初次執行時排程器為每組預留了一條未使用聲源、因此只涵蓋 12 個原始錄音；已將排程修正為每組三個確實要評分的聲源，以上是重新執行的完整結果。

雙聲源模式的另一組測試中，八個固定方向案例的目標 SI-SDR 改善中位由 +7.62 提高到 +18.34 dB。八個同位置案例全部退回原方法。若示範後立刻左右對調，舊版仍有 8/8 通過，固定方位模式卻為 0/8，改善中位 −31.17 dB。辨識同一位移動人物應另選人物追蹤功能。不同套件會重用部分測試原始素材，不可合計為互相獨立的錄音數。

## 使用與重跑

使用既有設定檔時，命令列可直接切換模式。這條命令使用隨產品附送的樂器示例。

~~~sh
python3 -m voicehunter.multitrack examples/studio.wav --config examples/multitrack.json --separator adaptive --out example-spatial-export
~~~

也可在設定檔中指定 separator 為 adaptive 或 spectral。adaptive_example.json 的時間是供早段依次有三個獨奏示範的 7.25 秒場景使用；自己的素材需改成各物件真實的示範區間。settings.json 記錄 requested_separator 和 spatial_filter_applied，供確認當次實際使用的路徑。

從完整專案根目錄執行以下步驟。環境資料重建需要網路。訓練及測試需要 NumPy、FFmpeg/ffprobe，無須 GPU。

~~~sh
python3 round2/envdata/restore.py --manifest round2/environment/cohort.json
python3 round2/envdata/restore.py --manifest round2/environment/native_inventory.json
python3 round2/fielddata/collect.py ihta_foa eigenscape
python3 round2/environment/train.py fit
python3 round2/environment/train.py test
python3 round2/environment/field_and_named.py
python3 round2/environment/native_stereo.py
python3 round2/environment/two_targets.py
python3 round2/environment/legacy_practical.py --cache /tmp/voicehunter-r1-rnnoise-source --fetch
python3 round2/environment/regress.py
~~~

fit 會重寫兩個係數及 fit.json；重跑請使用另外解壓的副本。歷史資料由 qa_round1/mixtures/sources.json 提供並在使用前驗雜湊，沒有混入權重訓練。完整逐案分數在 fit.json、test.json、field_and_named.json、two_targets.json、native_stereo.json、legacy_practical 與 legacy_registers。研究及產品的其餘授權見資料目錄及來源作者標示。
