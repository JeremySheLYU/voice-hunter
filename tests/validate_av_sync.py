"""Round-trip a movie with delayed audio and one with delayed video (FFmpeg, NumPy)."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicehunter.media import SAMPLE_RATE, load_media, mux_video, write_wav


def video_start(path):
    fields = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "V:0",
        "-show_entries", "stream=start_time:format=start_time", "-of", "json", path
    ])
    data = json.loads(fields)
    return float(data["streams"][0]["start_time"]) - float(data["format"]["start_time"])


def stream_hash(path):
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", path, "-map", "0:V:0", "-c:v", "copy", "-f", "hash", "-"
    ])


def run():
    with tempfile.TemporaryDirectory(prefix="voice-hunter-sync-") as temporary:
        directory = Path(temporary)
        source, audio, output = (directory / f for f in ("source.mov", "audio.wav", "output.mov"))
        for video_delay, audio_delay in ((0.0, 0.5), (0.5, 0.0), (0.7, 0.2)):
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-itsoffset", str(video_delay), "-f", "lavfi",
                "-i", "color=black:s=160x90:r=10:d=2", "-itsoffset", str(audio_delay),
                "-f", "lavfi", "-i", "sine=frequency=800:sample_rate=48000:duration=1.5",
                "-ac", "2", "-c:a", "aac", "-c:v", "mpeg4", source
            ], check=True)
            mix, metadata = load_media(source)
            write_wav(audio, mix)
            mux_video(source, audio, output)
            decoded, _ = load_media(output)
            onset = lambda x: np.flatnonzero(np.abs(x[:, 0]) > 0.04)[0] / SAMPLE_RATE
            original_onset = onset(mix)
            assert abs(onset(decoded) - original_onset) < .002
            assert abs(original_onset - video_start(source) - (audio_delay-video_delay)) < .002
            assert abs(video_start(output)-video_start(source)) < .002
            assert abs(len(decoded)-len(mix)) <= SAMPLE_RATE*.003
            assert stream_hash(source) == stream_hash(output)
            print(f"PASS v={video_delay:.1f} a={audio_delay:.1f} "
                  f"container={metadata['container_duration']:.3f}s "
                  f"audio={metadata['source_audio_duration']:.3f}s")


if __name__ == "__main__":
    run()
