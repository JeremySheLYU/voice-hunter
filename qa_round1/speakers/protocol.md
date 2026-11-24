# Frozen speaker-separation QA: protocol defined before scoring

Test the v0.1 two-example, stereo-only speaker separation entry point without adjusting its code or thresholds. Input is a controlled mixture of independent known voices rather than an acoustic field recording. Simulate capture at 48 kHz; source bandwidth for spoken digits is limited to 4 kHz. No videos or voice identifiers are provided to the estimator.

For each scenario below use three disjoint pairs from FSDD (six human speakers), and three independent synthetic talker pairs, with fixed seeds. Identical voice content across each pair's scenarios enables within-pair comparison; repeated scenarios are not new independent recordings. Calibrate on four utterances and evaluate four different utterances for each speaker. Human calibration digits 0–3 use recording 10; held-out digits 4–7 use recording 2. No source WAV is repeated across the two partitions. The synthetic controls independently draw calibration and held-out pitches, envelopes and vowel sequences from the same talker distributions.

The evaluation section includes held-out A-only and B-only passages, followed by an overlap. In the 50% duty stress case, only portions of the overlap contain B. Compute separation scores on 20 ms frames where both true references are active (both mean-square levels above -30 dB relative to their active peaks). Also report the B-only leakage into A relative to untreated B, A-only gain error, and maximum additive reconstruction error across the clip. For contaminated enrollment, put the other talker's *enrollment* utterance into the example at -8 dB relative input RMS; do not leak scored tokens back into enrollment.

| Scenario | Expected before test |
|---|---|
| Opposite sides, stationary, weak reflections | return useful tracks |
| Reversed sides, stationary, weak reflections | return useful tracks |
| Opposite sides, A 8 dB below B | return useful tracks |
| Opposite sides, approximately half B duty | return useful tracks |
| Close positions | exploratory, rejection is possible |
| Identical positions and transfer functions | refuse ambiguous request |
| Strong diffuse and early reflections | exploratory |
| Sources cross to each other's positions after calibration | exploratory |
| Source A/B enrollment contaminated in both directions | exploratory |

Runtime gates, defined in advance: stationary expected-use cases must all return output, median **minimum** improvement across the two outputs at least +6 dB and worst at least +3 dB; no delivered result may reconstruct worse than 1e-5 of normalized full scale or contain nonfinite values. Identical-position cases must all be refused. For hard cases report every rejection and harmful accepted output (min track improvement <=0 dB), even when the aggregate is favorable. The primary measure is ordinary waveform SNR improvement against the unprocessed mixture for each target. This retains gain and stereo alignment errors; report the less-improved speaker per case to expose asymmetric outcomes. Pure B leakage rejection is measured by change in A-stem level over B-only audio; positive dB means stronger suppression. These engineering checks do not measure intelligibility or naturalness.

Sources: Free Spoken Digit Dataset, Jakobovski and contributors, Creative Commons Attribution-ShareAlike 4.0, commit 26eb9aaf76e81b692f806f9140c2d2777410d7a1. No corpus audio or derived listening clips are shipped. Source manifest contains exact filenames and hashes. Without the optional public corpus, the synthetic controls still run and every omitted human case is listed as skipped.
