# RNNoise model

Voice Hunter redistributes the original Xiph RNNoise v0.1 speech-enhancement
network in the plain-text weight format accepted by FFmpeg's `arnndn` filter.
The numeric arrays are from `xiph/rnnoise`, tag `v0.1`, file
`src/rnn_data.c` (Git blob `8f6c99bb8ee6770fba29820bc5d05fadb1aba6f3`).
The FFmpeg-readable serialization is the `std.rnnn` artifact maintained in
`richardpl/arnndn-models` (Git blob `2c940d14b54486e50cdd188232a3f1cb75ec6af7`).
See `COPYING.rnnoise.txt` for the upstream BSD 3-Clause license, copyright
notices, and disclaimer. No other model from that repository is bundled.

Verified 2026-11-21 by comparing all 15 numeric weight and bias arrays with the tagged Xiph source. `std.rnnn` SHA-256:
`6b8943dc4a9b6b24425873992a44f29c0577503276456af46a8854774faeb294`.

- [Original weight arrays](https://github.com/xiph/rnnoise/blob/v0.1/src/rnn_data.c)
- [Original license](https://github.com/xiph/rnnoise/blob/v0.1/COPYING)
- [FFmpeg model provenance](https://github.com/richardpl/arnndn-models/blob/master/README.md)
- [FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html#arnndn)

Optional Demucs weights are not included. If you already have the official
four-stem `htdemucs` individual checkpoint (`955717e8-8726e21a.th`) and
its working Python environment, set `VOICEHUNTER_DEMUCS_CHECKPOINT` to the
absolute trusted weight path and `VOICEHUNTER_DEMUCS_PYTHON` to that
environment's Python executable, then restart Voice Hunter. The app validates
the local checkpoint and model package before showing the option. Only load
publisher-supplied checkpoints you trust because Demucs uses Python pickle.

- [Demucs project and setup](https://github.com/facebookresearch/demucs)
- [Model names and checkpoint URLs](https://github.com/facebookresearch/demucs/tree/main/demucs/remote)

## Public-data calibrations in 0.2

speaker-distance.json was fitted on Mini LibriSpeech by Vassil Panayotov and contributors under CC BY 4.0. The training file, speaker groups, archive checksums and credit are recorded in round2/speakers. The test-clean material is evaluation-only in this release.

joint_spatial.json contains two selected scalar parameters. Training uses attributed CC0 / CC BY Freesound files and CC BY field material. Consult round2/environment/cohort.json, round2/envdata/attribution.csv and round2/fielddata/ATTRIBUTION.md for the exact files, creators, licenses and transformations.

cipic_horizontal_v1.json derives from measured CIPIC data. The University of California Regents' permission and required notice are bundled in CIPIC-NOTICE.txt. HUTUBS was used only for independent evaluation.

phase_retention.json contains one threshold selected with FMA-small excerpts. Each recording, credited artist, source and license URL is identified in round2/stereo/manifest.json. The method creates controlled relative-polarity inversions for calibration. No third-party master recordings are encoded in the scalar parameter.

The earlier RNNoise and optional Demucs interfaces retain their original model provenance.
