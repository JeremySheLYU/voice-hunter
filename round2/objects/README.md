# ADM 物件格式校驗

EBU EAR Production Suite 的開發者在官方儲存庫附上四個依 ITU-R BS.2076 跳轉行為建立的測試 WAV，每個都有相對應的可編輯 XML。我們把這組素材連同來源與授權獨立保存，作為技術規格校驗素材。它們不參與演算法權重訓練、品質聆聽或主評測分數。這四個檔案也沒有作為商用內容示範或 Dolby Atmos 成品計數。

每檔皆為 RIFF WAVE、含 axml 與 chna、48 kHz、16-bit，帶一個 ADM Objects 聲道。讀取 PCM 確認有非零 440 Hz 音調。CHNA 的軌道 UID 與 XML 物件引用一致；每個時間區塊均有座標與時間，四種 jumpPosition 條件透過不同的中繼資料指定轉場。四檔使用同一三秒素材；報告應計為四種物件自動化測試、一份獨特音訊。沒有從這些音調推估真實環境定位。

來源鎖定 commit ed42d4ede70172d80cd4402c1674062350377f68。儲存庫 COPYRIGHT.TXT 明確列明整個儲存庫內容的版權人與 GPL-3.0-or-later 條款。未改動的四個原檔、四個對應 XML、完整 LICENSE.TXT、COPYRIGHT.TXT 及作者對此測試的說明存放在 round2_data/objects/；來源與每檔 Git blob 及 SHA-256 見 manifest.json。此處發行的報告含來源連結與重取程式。從專案根目錄執行 `python3 round2/objects/acquire.py` 可重取及解析。

上游 [專案、授權與公告](https://github.com/ebu/ear-production-suite/tree/ed42d4ede70172d80cd4402c1674062350377f68)、[測試 WAV 與原始 XML](https://github.com/ebu/ear-production-suite/tree/ed42d4ede70172d80cd4402c1674062350377f68/reaper-adm-extension/test/reaper_adm/data)。

另外查過 S3A 的三個 ADM 廣播劇場景。官方註冊頁將取用限於學術研究並禁止再分發，這次沒有納入。EBU QC 的 22.2 識別聲範例大小約 237 MB，亦超過這組的 40 MB 額度。Orbit Spatial 的公開頁目前對 ADM BWF 寫的是 coming soon。[S3A 取用條款](https://cvssp.org/data/s3a/public/radiodrama_register.php) · [EBU QC 條目](https://qc.ebu.io/testmaterials/74/) · [Orbit Spatial 目錄](https://orbitspatial.io/test-files/)
