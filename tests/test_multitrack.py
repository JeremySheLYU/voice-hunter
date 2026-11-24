"""Verify three concurrent sources, written WAV reconstruction, and ID safety."""
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicehunter.media import SAMPLE_RATE, load_media, write_wav
from voicehunter.multitrack import export_tracks


class TestJointTracks(unittest.TestCase):
    def test_three_simultaneous_tones_and_additive_export(self):
        n = SAMPLE_RATE
        t = np.arange(n * 5) / n
        sources, tracks = [], []
        for i, (frequency, left, right) in enumerate([(385, 1, .3), (797, .63, .63), (1463, .24, 1)]):
            pilot = (t >= i) & (t < i+1)
            envelope = np.where(pilot | (t >= 3), .14, 0)
            s = envelope * (1+.24*np.sin(2*np.pi*(i+1)*t)) * np.sin(2*np.pi*frequency*t)
            sources.append(np.column_stack([s*left, s*right]).astype('f4'))
            tracks.append({"id": "source" + str(i), "span": [i + .1, i + .9]})
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            original = temp / "test.wav"
            mixture = sum(sources)
            write_wav(original, mixture, stem=True)
            export_tracks(original, {"tracks": tracks}, temp / "out")
            mixes = [load_media(temp / "out" / (track["id"] + ".wav"))[0] for track in tracks]
            remain = load_media(temp / "out" / "remaining.wav")[0]
            np.testing.assert_allclose(sum(mixes) + remain, mixture, atol=2e-7)
            for stem, expected in zip(mixes, sources):
                overlap = slice(n*3 + n//10, n*5 - n//10)
                error = np.mean((expected[overlap] - stem[overlap])**2)
                mixture_error = np.mean((expected[overlap] - mixture[overlap])**2)
                self.assertGreater(10*np.log10(mixture_error/error), 8)
            config = json.loads((temp / "out" / "settings.json").read_text())
            self.assertEqual(len(config["tracks"]), 3)
            self.assertEqual(config["remix_peak_protection_gain"], 1.)
            for unsafe in ["../mix", "remaining", "source0"]:
                tracks[1]["id"] = unsafe
                with self.assertRaises(ValueError):
                    export_tracks(original, {"tracks": tracks}, temp / "bad")
                self.assertFalse((temp / "bad").exists())


if __name__ == "__main__":
    unittest.main()
