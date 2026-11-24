# Five-class stereo extraction with YAMNet

The research candidate accepts a stereo WAV or the stereo audio stream of a video. Select a target class from guitar, piano, speech, train or wind. When the signal supports two well-separated positions and the requested class passes the calibrated threshold, the program writes stereo target and remainder WAV files. A refusal exits with code 2. The new directory supplied to --out must not exist. FFmpeg and ffprobe must be available.

From the project root:

~~~sh
python3 -m venv .venv-transfer
.venv-transfer/bin/python -m pip install -r round3/transfer/requirements.txt
.venv-transfer/bin/python -m round3.transfer.extract recording.mov --target piano --out piano-tracks
~~~

## Method

The original fixed-pan stereo demixer estimates two candidate source images. Each is low-pass filtered and converted to a 16 kHz mono waveform for Google's public YAMNet encoder. The frozen embedding vector has 1,024 dimensions. A separate, locally trained class-balanced kernel ridge head learns five classes. The inference stage combines its output with YAMNet's own AudioSet class families. The blend and single global presence threshold are selected on original uploader-disjoint validation only. Stereo panning and output length are retained in the source image that is returned. Each stem is classified separately without requiring that exactly one of each class occur. The program returns at most one stereo source image for the requested class.

These five classes are a closed vocabulary. The default spatial stage requires stationary, distinguishable left/right levels. Unknown source types, arbitrary source counts, diffuse stereo and real-room stereo recordings require separate evidence. The encoder was pretrained by Google on AudioSet clips from YouTube. Source IDs and uploader partitions are checked for the downstream Freesound corpus; possible cross-posts in upstream AudioSet pretraining cannot be excluded.

## Model and provenance

The unmodified ONNX export has SHA-256 d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e. It was obtained from the audiomagic/yamnet-onnx repository at commit f25b741c2f0bdc6d7e6db24b5fddda23347dbafd. The accompanying label map has SHA-256 cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2. The package includes the Apache 2.0 license and distributor's model card. Voice Hunter trained the downstream head and calibration; the embedded YAMNet network remains the public pretrained model. The original TensorFlow HDF5 weights were independently downloaded for provenance with SHA-256 13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744. The tests use the ONNX export.

[Google's YAMNet guide](https://www.tensorflow.org/hub/tutorials/yamnet) · [Google's transfer-learning guide](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio) · [Pinned ONNX release](https://huggingface.co/audiomagic/yamnet-onnx/tree/f25b741c2f0bdc6d7e6db24b5fddda23347dbafd)

## Calibration and reproduction

Per-recording ID, author and digest are in train_sources.json; full provenance and license records are in the original and additional data manifests. See validation.json for the complete tuning grid, and freeze.json for the hashes used by independent evaluation. For exact repeatability use NumPy 2.5.5 and ONNX Runtime 1.32.0 (macOS ARM), with the acquired corpora and FFmpeg. The additional training input accepts only rows designated train. The original calibration uses 55 source pairs in each of two conditions, based on the original validation recordings.

~~~sh
.venv-transfer/bin/python round3/transfer/train.py --extra-train round3/data/train.json
~~~
