# 按作者分組的真實立體聲音樂

從 FMA-small 的原始 ZIP 和 raw_tracks.csv 配對選取片段。僅採用發行資料中 license_url 明確為 CC0 或 CC BY 的條目，按固定雜湊每位作者選一首。共有 194 位作者、194 個 MP3，200,254,133 bytes；193 個檔原有雙聲道，183 個全曲側聲道能量大於平均通道能量的 −60 dB。另有 1 個原始單聲道。完整 ffmpeg 解碼共 5,817.45 秒。音樂沒有乾淨物件參考軌。

來源為 Michaël Defferrard 等人發表的 FMA，歌曲授權按個別作品記錄。manifest.json 保存曲名、作者、原始樂曲連結、授權連結、原始 ZIP member、位元組與 SHA-256。各作者只出現在一個分組。train.py 使用 140 首立體聲訓練，23 首驗證，30 首測試。另一首單聲道未用於防抵銷模型。

在專案根目錄用 python3 round2/stereo/fetch.py 恢復原始 MP3，再用 python3 round2/stereo/audit.py 重新核對聲道。需連線到 FMA 公開封存主機並能使用 HTTP Range；本機已有全部 MP3 時仍會核對官方中央目錄。學習腳本為 python3 round2/phase/train.py。評估素材依 CC0 或作品本身的 CC BY 條款使用與轉交。

來源 https://github.com/mdeff/fma
