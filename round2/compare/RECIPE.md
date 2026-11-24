# 第一個受測場景的列車分軌比較

選曲規則先固定為 field_and_named.json 中 trial=1、無外加 AmbiX 背景的測試場景，沒有按分數選例。三份公開錄音在此受控混音獲得固定模擬聲像，前三段為單獨示範。試聽只播放之後三者同時發聲的同一個 1.57 秒計分區間。各檔首尾均加入 6 ms 防爆音漸變。每個版本以常數增益校準到 −23 LUFS，以 48 kHz PCM16 雙聲道嵌入 index.html。下游的響度校準只用於聆聽，數值評估使用原始匯出，不用試聽的增益。適合快速聽串音，也保留原始混音供比較。

可離線雙擊 index.html。網頁不傳送音訊或聽眾資料。要重建，先依 round2/environment/README.md 還原 cohort.json 中的公開錄音，並確保選用的 v0.1 原始 ZIP 和 voicehunter/models/joint_spatial.json 都有正確雜湊。在專案根目錄執行：

~~~sh
python3 round2/compare/build.py --legacy-zip outputs/VoiceHunter_v0.1.zip
~~~

建置程式從原版 ZIP 內執行 v0.1，從專案執行 v0.2 並傳入 --separator adaptive，使用同一份音訊及示範設定。數值及授權出處見 metrics.json、CREDITS.md。此為單一受控場景的開卷比較。
