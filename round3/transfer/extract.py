"""Extract a requested source from a supported stereo recording using the transfer model."""
import argparse
from pathlib import Path

from round3.transfer.candidate import CLASSES, Predictor
from voicehunter.media import load_media, write_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', type=Path)
    parser.add_argument('--target', required=True, choices=CLASSES)
    parser.add_argument('--out', required=True, type=Path,
                        help='Destination directory (must be new)')
    args = parser.parse_args()
    audio, _ = load_media(args.media)
    estimate = Predictor().prepare(audio).predict(args.target)
    if estimate is None:
        parser.exit(2, 'No track delivered: target confidence or stereo separation was insufficient.\n')
    args.out.mkdir(parents=True, exist_ok=False)
    write_wav(args.out / 'target.wav', estimate, stem=True)
    write_wav(args.out / 'remainder.wav', audio-estimate, stem=True)


if __name__ == '__main__':
    main()
