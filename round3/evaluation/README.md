# Held-out automatic separation scores

Read `PROTOCOL.md` for the predeclared 90% criterion and `selection.json` for the validation-only primary choice. The two test conditions are controlled stereo mixtures of real, weakly labeled recordings. Original stereo geometry, new categories, and measured reverberation need separate tests. The prevalence of positive requests is 50% in this experiment.

## Result files

- `scorecard.csv` aggregates class-presence detection and reference-track quality; `scorecard_by_requested_class.csv` separates guitar, piano, speech, train and wind. Accuracy, recall, precision, binary F1, macro F1 and useful fraction are recorded as fractions.
- `fresh-plan.json` fixed 60 fresh recordings (12/class), 30 disjoint source pairs and 25 source-uploader clusters. Both roles are exercised, with one positive and one absent request per target turn per condition, giving 120 queries per condition. The exact same source pairs are used for persistent and intermittent conditions. The two conditions contain 60 unique recordings in total.
- `supplemental-plan.json` describes the remaining 27 test recordings as targets (3 guitars, 8 speech, 8 trains, 8 wind). One primary-set piano clip serves only as the unscored distractor for an odd remaining target. These secondary results and the balanced primary result should be quoted separately.
- Each `fresh/METHOD/CONDITION.json` contains exact per-request decisions, counts, audio-quality gains, aggregate confusion matrix, 95% confidence intervals, and per-class scores. `TP` is “requested class present and a finite stereo waveform returned,” including returns with poor quality. `useful` requires a returned positive waveform and >=3 dB stereo SI-SDR improvement over the input mixture on the shared scoring window. Refusals stay in the denominator.
- A digitally silent ground truth makes SI-SDR improvement undefined. Such positive requests remain in the useful-delivery denominator and never count as useful. The primary intermittent stratum has one such scoring window. Verify each result's `undefined_truth` field.
- `historical/` contains reruns and independently recomputed aggregate scores against the original round-2 Apple and v0.2 rows. Those materials were previously reported and should be described as regression evidence. Use `fresh/` for the headline.
- The Apple baseline was double-checked with independent repeated runs. Its per-request presence decisions were identical across serial and parallel runs. One positive persistent track switched sides of the +3 dB threshold. The headline reference comes from serial execution with BLAS limited to one thread. `apple_replicates.json` records the observed variability.
- Confidence intervals are 2,000 percentile bootstrap draws over disjoint paired source/uploader groups with a fixed seed. The grouping keeps outputs for reused audio together. This does not sample devices, environments, listeners or underlying recording populations. Primary selection and any demonstrative 90% result are scoped to this exact test design.

## Reproduce

Start at the project root after restoring the source manifests in `round2/semantic` and `round3/data`, supplying FFmpeg, NumPy, and each chosen model's runtime. Apple methods additionally require macOS and Xcode command-line tools. Transfer and fusion need ONNX Runtime. Old raw data files must reside at their manifest paths.

~~~sh
python3 round3/evaluation/evaluate.py archives
python3 round3/evaluation/evaluate.py plan --test round3/data/test.json --planned-train round3/data/plan.json
python3 round3/evaluation/evaluate.py run fresh v02 --workers 2
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 round3/evaluation/evaluate.py run fresh apple --workers 1
python3 round3/evaluation/evaluate.py run fresh classifier --fingerprint round3/classifier/freeze.json
python3 round3/evaluation/supplemental.py v02
python3 round3/evaluation/check.py
python3 round3/evaluation/summarize.py
python3 round3/evaluation/paired.py
~~~

The inference adapter accepts an audio mixture, sampling rate and requested labels only. Source identifiers, labels of the actual interfering sources, true time windows and oracle directions are unavailable to inference. The independent scorer sees the reference waveforms after the model returns. A clean rerun can use a fresh unpacked copy and new output directories; previously scored files retain their hashes and are reused only with identical provenance.
