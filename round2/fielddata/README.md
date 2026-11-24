# Field sound, first and fourth order

The source-of-truth inventory is `manifest.json`. Load `files[*].path` relative to the project. Format names describe the original coefficient order / authored binaural representation. FFmpeg may guess a four-speaker layout from a file header. Access channels by index. AmbiX first order is [W,Y,Z,X]. The same first four coefficients in the higher-order recordings were compared directly with their Lite counterparts. Field scene class names do not establish source-isolated reference audio.

Local acquisition uses validated byte ranges and partial decompression of published ZIP members, then decodes exactly the specified source window. This avoids downloading tens of gigabytes of unused time. Each receipt retains a checksum of the fetched compressed bytes and of the resulting PCM. The published complete-archive checksum is recorded as provenance; no claim is made that every byte of a multi-gigabyte remote archive was downloaded or locally hashed. `manifest.py` verifies every saved excerpt's local SHA and rejects duplicates. See ATTRIBUTION.md for credits and preprocessing.

Rebuild from the project root with Python 3, NumPy, ffmpeg and ffprobe:

~~~sh
python3 round2/fielddata/collect.py eigenscape ihta_foa ihta_ears london --workers 4
python3 round2/fielddata/collect_hoa.py
python3 round2/fielddata/manifest.py
python3 round2/fielddata/fit_width.py
python3 round2/fielddata/score_render.py
~~~

The original metadata and entry indexes are bundled in catalogue/. The site listing is from [the EigenScape author's deposited CSV](https://zenodo.org/api/records/2628463/files/Metadata-EigenScape.csv/content). Preserve credits when reusing excerpts.

## Ambient width experiment

A reproducible optional comparison converts field AmbiX into a known virtual ±20° stereo microphone pickup, then fits the side-to-mid power ratio in 4 bands against the publisher's binaural rendering. All takes from a held-out date belong to the evaluation group. Source classes, room labels, speaker identity and spatial object angles are not part of this training. Fixed baseline, learned global side gain and learned 4-band side gain are retained. The experimental renderer accepts standard stereo WAV, with results validated only for this acquisition pattern and field site. Use actual rendered results in render_eval.json; fit_width.py also records the ideal band-domain predictions in width_results.json. No human preference scores were collected.

~~~sh
python3 round2/fielddata/render_width.py narrow-xy.wav comparison.wav --condition fitted_bands
~~~

## Results on this acquisition

| Collection | File count | Independent new source timelines | Excerpt per file |
| --- | ---: | ---: | ---: |
| EigenScape Lite, AmbiX order one | 64 | 64 UK recording locations | 8 s |
| EigenScape, AmbiX order four | 8 | 0 additional; same first two seconds of site 1 in each class | 2 s |
| Aachen IHTApark, AmbiX order one | 102 | 102 source takes over 18 dates at a single park | 4 s |
| Aachen IHTApark, binaural studio decoder | 102 | 0 additional; alternative render of each capture | 4 s |
| London indoor study, edited AmbiX | 7 | 7 stimuli | 8 s |

283 distinct audio files, 455,248,508 bytes, comprising 173 different source recording/stimulus IDs and 976 s of unique source time. Total timeline across all encoded representations is 1,400 s. 243 files are sampled at 48 kHz and 40 at 44.1 kHz. The optional demo files are additional derived copies excluded from these counts.

The 78 fitting captures span 14 dates at the Aachen site. The 24 evaluation captures span Feb 5, Feb 12, Apr 21 and May 2, 2022. Median absolute band-ratio error on actual rendered waveforms was 14.72 dB for the virtual narrow XY baseline, 12.11 dB for 1.35× widening, 3.57 dB for the learned single gain, and 2.80 dB for the learned four-band gain. The latter gains were +7.38/+12/+12/+12 dB; the last three were limited by the predeclared cap. The largest mid-channel deviation was 7.9e−9 of full scale. These numbers quantify spectral width relative to this specific decoder at this site. They do not quantify subjective spaciousness or the behavior of arbitrary preexisting stereo mixes.

The demo/ directory contains one complete 4-second held-out example with virtual stereo, the learned correction, and the published decoder, matched to −24.01…−23.99 LUFS. Rebuild using `python3 round2/fielddata/demo.py`. Read demo/attribution-and-levels.json for source, changes, licensing and peaks. The headphones and listening environment were not standardized.
