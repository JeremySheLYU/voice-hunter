# Round 1 protocol for the released RNNoise path

Fixed before running the evaluation matrix. Test Voice Hunter through
`voicehunter.engine.separate_clip(..., "rnnoise", "voice")`; all three model
modes (RNNoise available, Demucs unset, Demucs misconfigured) are also checked
without downloading anything. Treat the optional real Demucs inference as
skipped unless a checkpoint and runtime are available. No product code changes.

Sources: the three LibriSpeech example narrators in the canonical librosa/data
collection (198-209-0000, 3436-172162-0000, 5703-47212-0000). Use the first
5 s of each recording. Fourth voice condition overlaps the first two utterances
with delay 0.3 s, panned left and right; its target is their aggregate.
Environmental recordings: first lexicographic fold-1 ESC-50 filenames with
train, wind and vacuum_cleaner labels, as listed in the dataset CSV. Music:
first five seconds of 'Vibe Ace' by Kevin MacLeod from the same librosa
collection. Additional non-recorded control: Gaussian white noise seeded 1801.
Record hashes, selection, applicable dataset/license declarations, and skips.
Any unreachable source is reported rather than silently substituted.

Full crossing: 4 voice conditions × 5 interferers × nominal stereo-energy
SNR -5, 0, +10 dB. Whole-clip RMS determines SNR; include inactive speech
intervals. Three speakers use different static stereo pans; one interfering
source is panned to a different location. These are replayable manufactured
mixtures using recorded source material, not four in-situ meetings or outdoor
microphone captures. Secondary checks: each clean voice alone, each interferer
alone, digital silence. Evaluation captures gains and defects in just these
recordings. No perceptual listener panel.

Predeclared **engineering checks**, not scientific norms:
1. Each noisy scene passes if full stereo SI-SDR gain >= +1 dB. Overall gate
   is >= 75% cases passing and a median >= +3 dB. Report median, min, max,
   percentage with any gain, and each individual row; never drop negative gains.
2. Clean speech alone: absolute SI-SDR >= 15 dB and broadband target gain
   between -3 and +3 dB. An improvement score against an exact target is undefined.
3. Only an interferer: target RMS should be at least 10 dB below mixture
   (attenuation >= 10 dB). This measures false foreground energy, not detection
   precision; the product has no binary voice-present decision.
4. Exact silence should remain finite and silent (peak <= 1e-7).
5. Every finite output must retain length/channels and target+residual must
   reconstruct its input within 1e-6 FS max. Clean speech references should have
   no global delay (absolute maximum-correlation lag <= 2 samples, search ±960).
6. Ready RNNoise and disabled unconfigured Demucs must be advertised correctly;
   invalid model/target should be rejected. Missing real optional-model weights
   are an explicit coverage gap.

Rights: fixture media stays outside shipped code and results; keep source records,
hashes, links, exact windows and measured rows for independent replication.
