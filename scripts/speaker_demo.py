#!/usr/bin/env python3
"""Generate and score an overlapping two-voice calibration example, offline.

Default voices are synthetic vowel/consonant sequences. Supply --voice-a and
--voice-b WAV files to use recordings, or --try-say to try macOS speech. Each
source uses distinct temporal excerpts for calibration and overlap evaluation.

Real-speech repeatability (public test inputs, optional; supply local paths):
  --voice-a jfk.wav, from github.com/ggml-org/whisper.cpp, samples/jfk.wav,
    Git blob 3184d372cd2f8b804d3a540c70ec50d927b335d2.
  --voice-b test.wav, from github.com/alphacep/vosk-api, python/example/test.wav,
    Git blob c41144a21710590e568e4e612d2a40baf9a71223.

The speakers' positions, microphone transfer functions, reflections and overlaps
are simulated. The score is waveform SNR on the held-out overlap, not a
general-purpose subjective quality score or a real meeting recording result.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicehunter.speech import separate_speakers

SR = 24000
A_REF = (0.20, 1.65)
B_REF = (2.20, 3.65)
OVERLAP = (4.10, 6.55)
VOWELS = {
    'ah': (730, 1090, 2440), 'ee': (270, 2290, 3010), 'uh': (520, 1190, 2390),
    'oo': (300, 870, 2240), 'eh': (580, 1770, 2510), 'ih': (390, 1990, 2550),
    'aw': (570, 840, 2410), 'ae': (620, 1840, 2500), 'ur': (490, 1370, 1710),
}


def synthetic_solo(pitch, tract_scale, rng, sequence):
    """Harmonic/aspirate vowel synthesis, rendered by this script only."""
    pieces = []
    for index, (onset, vowel, duration) in enumerate(sequence):
        n = int(duration * SR)
        t = np.arange(n) / SR
        u = t / duration
        f0 = pitch * (1 + 0.06 * np.sin(np.pi * u) - 0.045 * u
                      + 0.04 * np.sin(2.0 * index) + 0.013 * np.sin(2 * np.pi * 5 * t))
        phase = np.cumsum(f0) * (2 * np.pi / SR)
        sound = np.zeros(n)
        centers = np.array(VOWELS[vowel]) * tract_scale
        for harmonic in range(1, int(6500 / pitch)):
            freq = harmonic * f0
            shape = sum(strength / (1 + ((freq - center) / width) ** 2)
                        for strength, center, width in zip([1., 0.63, 0.36], centers, [65, 90, 135]))
            sound += (shape + 0.015) / harmonic ** 0.45 * np.sin(harmonic * phase)
        env = np.minimum(t / 0.024, 1) * np.minimum((duration - t) / 0.055, 1)
        sound *= np.maximum(env, 0) * (0.88 + 0.12 * np.sin(np.pi * u))
        before = rng.standard_normal(int(SR * (0.065 if onset == 's' else 0.035)))
        before = np.diff(before, prepend=0) * np.sin(np.linspace(0, np.pi, len(before)))**2
        before *= {'s': 0.095, 'h': 0.035, 'd': 0.05, 'm': 0.006, '': 0.0}[onset]
        pieces.extend((before, sound, np.zeros(int(SR * 0.035))))
    return np.concatenate(pieces)


def generated_voice(which):
    text = [('h','eh',0.22),('','oo',0.28),('s','ee',0.27),('m','aw',0.26),
            ('d','ae',0.23),('s','uh',0.25),('h','ah',0.3),('m','ee',0.24),
            ('d','oo',0.27),('s','eh',0.28),('h','ih',0.26),('','ur',0.33),
            ('d','ah',0.29),('m','aw',0.27),('s','ee',0.28),('','oo',0.26),
            ('d','eh',0.29),('h','uh',0.28),('m','ae',0.27),('','ih',0.27)]
    if which == 'B':
        text = text[::2] + text[1::2]
    return synthetic_solo(135 if which == 'A' else 209, 1 if which == 'A' else 1.13,
                          np.random.default_rng(7 if which == 'A' else 12), text)


def load(path):
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-vn',
                                    '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'])
    mono = np.frombuffer(data, dtype='<f4').astype(float)
    if not mono.size or np.max(np.abs(mono)) < 1e-5:
        raise ValueError(f"No audible samples in {path.name}")
    active = np.flatnonzero(np.abs(mono) > np.max(np.abs(mono)) * 0.007)
    return mono[max(0, active[0] - SR // 64):active[-1]+1]


def voice_solos(args):
    if args.voice_a or args.voice_b:
        if not (args.voice_a and args.voice_b):
            raise ValueError("Supply --voice-a and --voice-b together.")
        return load(args.voice_a), load(args.voice_b), "separate user-provided source files"
    if args.try_say:
        with tempfile.TemporaryDirectory() as tmp:
            voices = []
            lines = [
                ('Samantha', 'The sample begins at the start of the sentence. We can hear each voice speaking very clearly today.'),
                ('Daniel', 'Hello Jeremy. This is the second speaker. The train is arriving at the station behind our table.'),
            ]
            for index, (name, sentence) in enumerate(lines):
                file = Path(tmp) / f"{index}.aiff"
                subprocess.run(['say', '-v', name, '-o', str(file), sentence], check=True)
                voices.append(load(file))
            return *voices, "macOS synthesized English speech"
    return generated_voice('A'), generated_voice('B'), "script-synthesized vowel/consonant sequences"


def prepare_source(mono, solo_start, pan, delay, *, length):
    # Reserve the first 45% for the example; only the later 55% may enter the
    # evaluation mix. Short source files are intentionally looped *within*
    # each partition so data used for calibration never leaks into evaluation.
    cut = int(len(mono) * 0.45)
    if cut < SR * 0.4 or len(mono) - cut < SR * 0.4:
        raise ValueError("Each source must contain at least 1 second of audible material.")
    original = np.zeros(length)
    for when, dur, part in ((solo_start, 1.75, mono[:cut]), (4.0, 2.7, mono[cut:])):
        start = int(round(when * SR))
        count = int(round(dur * SR))
        wavelet = np.resize(part, count)
        fade = min(SR // 100, count // 2)
        wavelet[:fade] *= np.linspace(0, 1, fade)
        wavelet[-fade:] *= np.linspace(1, 0, fade)
        original[start:start + count] = wavelet
    def delayed(frames):
        return np.pad(original, (frames, 0))[:len(original)]
    # A small delayed reflection in each channel; separate B and A have
    # different inter-channel levels, arrival times and reflection patterns.
    theta = (pan + 1) * np.pi / 4
    left = (np.cos(theta) * delayed(max(0, delay)) + 0.10 * delayed(121 + max(0, delay)))
    right = (np.sin(theta) * delayed(max(0, -delay)) + 0.08 * delayed(173 + max(0, -delay)))
    s = np.column_stack((left, right))
    return s * (0.22 / max(0.01, np.max(np.abs(s))))


def snr(estimated, truth):
    return 10 * np.log10(np.sum(truth ** 2) / max(np.sum((estimated - truth) ** 2), 1.e-12))


def write_wav(path, x):
    with wave.open(str(path), 'wb') as f:
        f.setparams((2, 2, SR, 0, 'NONE', 'not compressed'))
        f.writeframes(np.rint(np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice-a', type=Path)
    parser.add_argument('--voice-b', type=Path)
    parser.add_argument('--try-say', action='store_true')
    parser.add_argument('--export-dir', type=Path)
    args = parser.parse_args()
    va, vb, label = voice_solos(args)
    n = int(7 * SR)
    a = prepare_source(va, 0.10, -0.50, -7, length=n)
    b = prepare_source(vb, 2.10, 0.42, 5, length=n)
    room = np.random.default_rng(19).normal(0, 0.0008, (n, 2))
    mix = (a + b + room).astype(np.float32)
    target, residual, info = separate_speakers(mix, SR, target_span=A_REF, background_span=B_REF)
    section = slice(round(OVERLAP[0] * SR), round(OVERLAP[1] * SR))
    base = snr(mix[section], a[section])
    after = snr(target[section], a[section])
    report = {
        'fixture': label, 'duration_seconds': n/SR,
        'A_reference_seconds': A_REF, 'B_reference_seconds': B_REF,
        'evaluation_overlap_seconds': OVERLAP,
        'input_A_snr_db': round(base, 2), 'output_A_snr_db': round(after, 2),
        'improvement_A_snr_db': round(after - base, 2),
        'max_reconstruction_error': float(np.max(np.abs(target + residual - mix))), **info,
    }
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for name, sound in [('speaker-mix', mix), ('speaker-A-estimate', target),
                            ('speaker-B-and-room', residual), ('speaker-A-reference', a),
                            ('speaker-B-reference', b)]:
            write_wav(args.export_dir / f'{name}.wav', sound)
        (args.export_dir / 'speaker-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if after - base <= 3.0 or report['max_reconstruction_error'] > 1.e-6:
        raise SystemExit("Separation failed the controlled fixed-position check.")


if __name__ == '__main__':
    main()
