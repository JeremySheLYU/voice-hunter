# Three-way held-out chronological source test

Prespecified before testing. Use the shipped `python -m voicehunter.multitrack`
command with default gain/width. Three `example`-mode tracks: speech, train
and wind. Supply only a 1.20 s solo example of each source in separate
calibration slots; use the production automatic background-example selection.
Never explicitly supply a clean held-out reference to the separator.

Three independent trials: librosa LibriSpeech speakers 198 / 3436 / 5703 in
that order; the first lexicographic ESC-50 `train` and `wind` filename in folds
1, 2, and 3 respectively. Nine distinct original recordings. Source audio [0,
1.25) s is used only for solo references; [2.00, 4.75) s is used exclusively
for the full simultaneous overlap. A 0.75 s chronological gap separates source
examples from scoring audio. Source panning rotates -0.55/0/+0.55 across
trials. Normalize each held-out stem to equal broadband stereo RMS; apply
each resulting gain to that entire source including the solo. Avoid clipping
by jointly attenuating the mixture and references if needed.

Score original-mix and extracted-stem stereo SI-SDR on the overlap for each
of the three source names. Each stem passes at SI-SDR improvement >= +3 dB.
A scene passes only when all three stems pass, source+remaining reconstructs
within 1e-6 FS peak error, and all outputs have expected shape and finite
samples. Record all rows, errors, skips, and lowest improvements. Three
controlled trials with physically simulated fixed panning do not represent
an actual train-station microphone field recording. No product code changes.
