# Controlled overlap check with two spoken-voice recordings

Two public example WAVs supplied the source signals. The test simulates a stereo recording with fixed talker and microphone positions. Run it separately from the included synthesized-phoneme demo. The third-party audio is downloaded only if you choose to reproduce this check.

| Source | File | SHA-256 |
|---|---|---|
| Speaker A | [ggml-org/whisper.cpp, samples/jfk.wav](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav) | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| Speaker B | [alphacep/vosk-api, python/example/test.wav](https://github.com/alphacep/vosk-api/blob/master/python/example/test.wav) | `dcfea5712c43a43ba7ae8083afb39d36993e5a69c46e88b68aaa72b65cb615bb` |

From the project's top-level folder with FFmpeg and NumPy available:

```sh
sample_dir="$(mktemp -d)"
curl -fsSL https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav -o "$sample_dir/jfk.wav"
curl -fsSL https://raw.githubusercontent.com/alphacep/vosk-api/master/python/example/test.wav -o "$sample_dir/test.wav"
shasum -a 256 "$sample_dir/jfk.wav" "$sample_dir/test.wav"
python3 scripts/speaker_demo.py --voice-a "$sample_dir/jfk.wav" --voice-b "$sample_dir/test.wav"
```

Compare both hashes with the table. Add `--export-dir "$sample_dir/result"` to listen to the mix, estimates and references.

Each original is mono at 16 kHz. FFmpeg resamples to 24 kHz. The generator trims leading and trailing low-level silence, keeps the first 45% of each file for calibration and uses the remaining 55% for the overlapping section. Each source is normalized and spatially mixed with distinct left/right gains, delays and early reflections. The source A pan is −0.50 and B pan is +0.42. Independent low-level Gaussian room noise is reproducible from seed 19. The estimator receives only the stereo mix, an A-only span of 0.20–1.65 s, and a B-only span of 2.20–3.65 s.

Scores use the **held-out overlap from 4.10 to 6.55 s**. For each estimate, waveform SNR is `10*log10(sum(reference_A**2)/sum((estimate-reference_A)**2))`, across both channels. This measures source fidelity on known synthetic mixing conditions. It does not measure intelligibility, perceived spatial quality, movement or performance in an actual meeting.

| Measurement | Value |
|---|---:|
| A reference vs. unprocessed stereo mixture | 4.15 dB |
| A reference vs. separated stereo A | 23.10 dB |
| SNR improvement (unrounded difference) | +18.95 dB |
| Maximum absolute error when adding both outputs back together | 1.49 × 10⁻⁸, normalized full scale |
