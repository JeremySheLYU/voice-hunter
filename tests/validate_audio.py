"""Deterministic two-object fixture and end-to-end signal checks.

Run from any directory: python3 tests/validate_audio.py
Use --export-dir PATH for auditionable WAVs and numeric results.
No model checkpoint, network, or source recording is required.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE_RATE = 48000
TARGET_SPAN = (0.05, 2.05)
BACKGROUND_SPAN = (2.35, 4.30)
OVERLAP_SPAN = (4.65, 9.15)
EPS = 1e-20


def tone(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def deposit(track, note, time):
    start = round(time * SAMPLE_RATE)
    count = min(note.size, track.size - start)
    track[start:start + count] += note[:count]


def bell_note(midi, rng):
    """Struck metal bars with a short strike and inharmonic decaying overtones."""
    t = np.arange(round(0.50 * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for multiple, strength, decay in [(1, 1.0, 0.145), (2.76, 0.35, 0.070),
                                       (4.18, 0.19, 0.048), (5.41, 0.07, 0.034)]:
        signal += strength * np.exp(-t / decay) * np.sin(2*np.pi*tone(midi)*multiple*t)
    signal += 0.03 * rng.standard_normal(t.size) * np.exp(-t/.008)
    return signal * (1 - np.exp(-t/.001)) * np.minimum(1, (.5-t)/.015)


def pluck_note(midi, rng):
    """Plucked low strings with pitched harmonics that decay at different rates."""
    t = np.arange(round(0.53 * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for h in range(1, 9):
        amplitude = np.sin(np.pi*.23*h) / h**1.25
        phase = rng.uniform(-0.4, 0.4)
        signal += amplitude * np.exp(-t * (5 + .65*h)) * np.sin(2*np.pi*tone(midi)*h*t + phase)
    return signal * (1-np.exp(-t/.0025)) * np.minimum(1, (.53-t)/.025)


def delay(x, seconds):
    k = round(seconds*SAMPLE_RATE)
    return np.r_[np.zeros(k), x[:-k]] if k else x.copy()


def make_fixture():
    """Bars right, strings left. Isolated cues, then the same timbres in a duet."""
    rng = np.random.default_rng(31)
    length = round(SAMPLE_RATE*9.6)
    bells, guitar = np.zeros(length), np.zeros(length)
    for start, stop, pattern, interval, instrument, track in [
        (.08, 1.62, [76, 79, 81, 79], .36, bell_note, bells),
        (2.37, 3.85, [52, 55, 59, 55], .32, pluck_note, guitar),
        (4.46, 9.10, [76, 79, 81, 79, 76, 81], .37, bell_note, bells),
        (4.51, 9.00, [52, 55, 59, 55, 52, 59], .29, pluck_note, guitar),
    ]:
        for i, onset in enumerate(np.arange(start, stop, interval)):
            note = instrument(pattern[i % len(pattern)], rng)
            deposit(track, rng.uniform(.87, 1.0)*note, onset)

    target = np.column_stack((.58*delay(bells, .00027)+.075*delay(bells, .021),
                              bells+.085*delay(bells, .012)))
    background = np.column_stack((guitar+.07*delay(guitar, .017),
                                  .56*delay(guitar, .00032)+.09*delay(guitar, .026)))
    start, end = (round(t*SAMPLE_RATE) for t in OVERLAP_SPAN)
    background *= np.sqrt(np.mean(target[start:end]**2)/np.mean(background[start:end]**2))
    gain = .72/np.abs(target+background).max()
    target, background = (gain*x.astype(np.float32) for x in (target, background))
    return (target+background).astype(np.float32), target, background


def decibels(ratio):
    return 10*np.log10(max(ratio, EPS))


def si_sdr(estimate, reference):
    estimate = estimate.astype(np.float64).ravel()
    reference = reference.astype(np.float64).ravel()
    estimate -= estimate.mean()
    reference -= reference.mean()
    fitted = reference * (np.dot(estimate, reference)/max(np.dot(reference, reference), EPS))
    return decibels(np.sum(fitted**2)/max(np.sum((estimate-fitted)**2), EPS))


def projection_sir(estimate, target, interferer):
    """Fit both known source images, ignoring artifacts; SI-SDR checks artifacts separately."""
    basis = np.column_stack((target.ravel(), interferer.ravel())).astype(np.float64)
    coeffs = np.linalg.lstsq(basis, estimate.ravel(), rcond=None)[0]
    source_power = coeffs**2 * np.sum(basis**2, axis=0)
    return decibels(source_power[0]/max(source_power[1], EPS))


def right_minus_left_db(x):
    power = np.mean(x.astype(np.float64)**2, axis=0)
    return decibels(power[1]/max(power[0], EPS))


def write_wav(path, x):
    # One shared PCM scale for the original and isolated tracks; mirrors the mix balance.
    samples = np.rint(np.clip(x, -1, 1)*32767).astype('<i2')
    with wave.open(str(path), 'wb') as f:
        f.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        f.writeframes(samples.tobytes())


def validate(export_dir=None):
    from voicehunter.separation import separate
    from voicehunter.spatial import peak_safe, remix

    mix, target, background = make_fixture()
    found, rest, _ = separate(mix, SAMPLE_RATE, method='example',
                              target_span=TARGET_SPAN, background_span=BACKGROUND_SPAN)
    assert found.shape == rest.shape == mix.shape
    assert np.isfinite(found).all() and np.isfinite(rest).all()
    neutral = remix(found, rest)
    solo = np.zeros_like(found)
    moved = remix(found, solo, pan=.65)
    narrow = remix(solo, rest, width=0)
    original_bg = remix(solo, rest, width=1)
    wide = remix(solo, rest, width=1.5)
    enhanced = remix(found, rest, pan=.35, width=1.2)
    boosted = remix(found, rest, target_gain_db=24, pan=1, width=2)
    protected, peak_gain = peak_safe(boosted)
    assert 0 < peak_gain < 1, 'Extreme gain should engage stereo-linked export headroom'
    assert np.max(np.abs(protected)) <= 10**(-1/20)+1e-6
    assert np.allclose(protected, boosted*peak_gain), 'Headroom must preserve stereo balance'
    assert np.allclose(enhanced.mean(axis=1), mix.mean(axis=1), atol=2e-6), 'Fold-down changed'
    for invalid in [{'pan': 1.5}, {'width': -1}, {'target_gain_db': float('nan')}]:
        try:
            remix(found, rest, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid render parameters accepted: {invalid}')
    example = slice(round(OVERLAP_SPAN[0]*SAMPLE_RATE), round(OVERLAP_SPAN[1]*SAMPLE_RATE))
    separation_gain = si_sdr(found[example], target[example]) - si_sdr(mix[example], target[example])
    rest_gain = si_sdr(rest[example], background[example]) - si_sdr(mix[example], background[example])
    leak_gain = (projection_sir(found[example], target[example], background[example]) -
                 projection_sir(mix[example], target[example], background[example]))
    # Side-to-mid ratio accommodates energy-compensated width implementations.
    def side_ratio(x):
        left, right = x.astype(np.float64).T
        return np.mean((left-right)**2)/max(np.mean((left+right)**2), EPS)

    results = {
        'target_si_sdr_improvement_db': float(separation_gain),
        'residual_si_sdr_improvement_db': float(rest_gain),
        'target_interference_ratio_improvement_db': float(leak_gain),
        'target_original_right_minus_left_db': float(right_minus_left_db(target[example])),
        'target_separated_right_minus_left_db': float(right_minus_left_db(found[example])),
        'pan_right_change_db': float(right_minus_left_db(moved)-right_minus_left_db(found)),
        'mono_background_lr_max_difference': float(np.max(np.abs(narrow[:, 0]-narrow[:, 1]))),
        'wider_background_side_mid_change_db': float(decibels(side_ratio(wide)/side_ratio(original_bg))),
        'neutral_max_sample_error': float(np.max(np.abs(neutral-mix))),
        'estimated_stem_sum_max_error': float(np.max(np.abs(found+rest-mix))),
        'mono_fold_down_error_at_non_neutral_render': float(np.max(
            np.abs(enhanced.mean(axis=1)-mix.mean(axis=1)))),
    }
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        for name, signal in [('original', mix), ('reference-bells-right', target),
                             ('reference-strings-left', background), ('estimate-bells', found),
                             ('estimate-background', rest), ('spatial-remix', enhanced)]:
            write_wav(export_dir/(name+'.wav'), signal)
        (export_dir/'metrics.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    assert separation_gain > 3, 'Target quality must improve in the duet, not only its solo intro'
    assert rest_gain > 3, 'Residual track must also suppress the target during overlap'
    assert leak_gain > 4, 'Selected target must reduce interference from the other source'
    assert right_minus_left_db(found[example]) > 1, 'Original target is on the right'
    assert results['pan_right_change_db'] > 1, 'Positive pan must shift the target right'
    assert results['mono_background_lr_max_difference'] < 2e-6
    assert results['wider_background_side_mid_change_db'] > 1
    assert results['neutral_max_sample_error'] < 2e-6
    assert results['estimated_stem_sum_max_error'] < 2e-6
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--fixture-only', type=Path, metavar='WAV',
                        help='Write only the input stereo recording, without loading Voice Hunter')
    args = parser.parse_args()
    if args.fixture_only:
        args.fixture_only.parent.mkdir(parents=True, exist_ok=True)
        write_wav(args.fixture_only, make_fixture()[0])
    else:
        validate(args.export_dir)
