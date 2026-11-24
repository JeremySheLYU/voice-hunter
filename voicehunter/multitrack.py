"""Offline multi-target exports with a shared stereo mask for each named source.

Run: python3 -m voicehunter.multitrack INPUT --config CONFIG.json --out OUTPUT

Each example track takes a target span and optional background span in seconds.
Auto tracks select a class supported by the local Apple backend. The default
spectral mode keeps the original projection. With --separator adaptive or
config field "separator": "adaptive", two to four manual examples with distinct
coherent stereo images use calibrated Wiener filters; other cases retain the
spectral projection. This opt-in mix can route unlabeled ambience into the
named tracks. The remainder stays in remaining.wav.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

from . import joint_spatial
from . import separation as sep
from .media import SAMPLE_RATE, load_media, write_wav
from .spatial import extract_stereo, remix, stft


def _validate_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("tracks"), list):
        raise ValueError("Config must contain a 'tracks' list.")
    if not 1 <= len(config["tracks"]) <= 8:
        raise ValueError("Specify 1 to 8 tracks per export.")
    if set(config) - {"tracks", "background_width", "separator"}:
        raise ValueError("Unknown config field.")
    if config.get("separator", "spectral") not in ("adaptive", "spectral"):
        raise ValueError("The separator must be adaptive or spectral.")
    width = config.get("background_width", 1.0)
    if not isinstance(width, (int, float)) or not 0 <= width <= 2:
        raise ValueError("background_width must be 0 to 2.")
    names = {"remaining", "remix"}
    references = set()
    for track in config["tracks"]:
        if not isinstance(track, dict) or set(track) - {"id", "mode", "target", "span", "background", "gain_db", "pan"}:
            raise ValueError("Invalid track object or unknown track field.")
        name = track.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", name):
            raise ValueError("Use a short file-safe track id made of letters, digits, hyphens and underscores.")
        if name.lower() in names:
            raise ValueError(f"Duplicate or reserved track id: {name}")
        names.add(name.lower())
        mode = track.get("mode", "example")
        if mode == "auto":
            key = track.get("target")
            if not isinstance(key, str) or key not in sep.EVENT_GROUPS:
                raise ValueError(f"Unknown auto target: {key}. Try {', '.join(sep.EVENT_GROUPS)}.")
            if "span" in track or "background" in track:
                raise ValueError(f"{name}: select either an auto target or example spans.")
            reference = ("auto", key)
        elif mode == "example":
            if "target" in track:
                raise ValueError(f"{name}: use 'span' to select the target example.")
            for field in ("span", "background"):
                if field == "background" and field not in track:
                    continue
                span = track.get(field)
                if (not isinstance(span, list) or len(span) != 2 or not all(
                        isinstance(v, (int, float)) and math.isfinite(v) for v in span)
                        or not 0 <= span[0] < span[1]):
                    raise ValueError(f"{name}: invalid {field} times in seconds.")
            reference = ("example", tuple(track["span"]))
            if "background" in track:
                a, b = track["span"]
                c, d = track["background"]
                if max(a, c) < min(b, d):
                    raise ValueError(f"{name}: target and background examples overlap.")
        else:
            raise ValueError("Multitrack modes are 'auto' and 'example'.")
        if reference in references:
            raise ValueError("Two tracks refer to the same target. Choose distinct examples or labels.")
        references.add(reference)
        for field, minimum, maximum in [("gain_db", -12, 12), ("pan", -1, 1)]:
            v = track.get(field, 0)
            if not isinstance(v, (int, float)) or not minimum <= v <= maximum:
                raise ValueError(f"{name}: {field} must be between {minimum} and {maximum}.")


def _features(audio):
    spectrum = stft(audio, sep.N_FFT, sep.HOP)
    magnitude = np.sqrt((abs(spectrum[:, :, 0])**2 + abs(spectrum[:, :, 1])**2) * 0.5)
    scale = np.sqrt(np.maximum(magnitude.mean(axis=1, keepdims=True), max(sep.EPS, magnitude.max() * 1e-4)))
    return np.ascontiguousarray(magnitude / scale)


def _auto_windows(audio, tracks):
    keys = [t["target"] for t in tracks if t.get("mode") == "auto"]
    if not keys:
        return {}
    all_scores = sep._native_windows(audio, SAMPLE_RATE, keys)
    results = {}
    for key in keys:
        supported = set(all_scores["labels"]).intersection(sep.EVENT_GROUPS[key])
        if not supported:
            raise ValueError(f"The local event model has no labels for {key}.")
        windows = []
        for w in all_scores["windows"]:
            score = max((w["class_scores"].get(label, 0.) for label in supported), default=0.)
            windows.append({"start": w["start"], "duration": w["duration"], "score": score})
        results[key] = {"labels": sorted(supported), "windows": windows}
    return results


def _joint_masks(audio, tracks):
    features = _features(audio)
    duration = len(audio) / SAMPLE_RATE
    times = np.arange(features.shape[1]) * sep.HOP / SAMPLE_RATE
    auto_windows = _auto_windows(audio, tracks)
    masks, metadata = [], []
    for track in tracks:
        probability, bg = None, None
        info = dict(track)
        if track.get("mode", "example") == "auto":
            fg, bg, probability, info_auto = sep._selection_from_result(auto_windows[track["target"]], times)
            info.update(info_auto)
        else:
            fg = sep._span_frames(track["span"], times, duration, "目標")
            if "background" in track:
                bg = sep._span_frames(track["background"], times, duration, "背景")
        foreground_dict = sep._dictionary(features, fg)
        if bg is None:
            bg = sep._background_without_span(features, ~fg, foreground_dict)
        background_dict = sep._dictionary(features, bg)
        masks.append(sep._source_mask(features, foreground_dict, background_dict, probability))
        metadata.append(info)
    # In overlapping claims, divide the largest single claim proportionally.
    # A shared nonnegative leftover mask thus always remains, also in overlap.
    total = np.zeros_like(masks[0])
    strongest = np.zeros_like(total)
    for m in masks:
        total += m
        np.maximum(strongest, m, out=strongest)
    np.divide(strongest, total, out=strongest, where=total > 0)
    for m in masks:
        m *= strongest
    return masks, metadata


def _spatial_images(audio, tracks, masks):
    if not 2 <= len(tracks) <= 4 or any(t.get("mode", "example") != "example" for t in tracks):
        return None
    spectrum = stft(audio, sep.N_FFT, sep.HOP)
    templates = joint_spatial.reference_templates(spectrum, [t["span"] for t in tracks], SAMPLE_RATE, sep.HOP)
    if templates is None:
        return None
    return joint_spatial.reconstruct(spectrum, masks, templates, len(audio), sep.N_FFT,
                                     sep.HOP, *joint_spatial.parameters())


def separate_spatial_examples(audio, target_span, background_span):
    """Separate one named target using two spatial reference spans at 48 kHz."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("需要有效且非空的 48 kHz 雙聲道音訊。")
    if target_span is None or background_span is None:
        raise ValueError("請各選一段目標聲音和對照聲音。")
    target = list(target_span)
    other = list(background_span)
    tracks = [{"id": "target", "span": target, "background": other},
              {"id": "comparison", "span": other, "background": target}]
    _validate_config({"tracks": tracks})
    masks, _ = _joint_masks(audio, tracks)
    images = _spatial_images(audio, tracks, masks)
    if images is None:
        stem, residual, info = sep.separate(audio, SAMPLE_RATE, method="example", target_span=target, background_span=other)
        projection = "spectral fallback: example spatial images are too diffuse or too similar"
    else:
        stem = images[0]
        residual = audio - stem
        info = {"target_span": target, "background_span": other}
        projection = "reference-conditioned stereo Wiener projection"
    info.update(method="spatial-example", projection=projection, spatial_filter_applied=images is not None,
                model="reference spatial Wiener v1", residual_policy="All other and unassigned audio")
    return stem, residual, info


def export_tracks(input_path, config, directory):
    """Export additive IEEE float WAV stems and one remix, atomically as a set."""
    _validate_config(config)
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Select a new or empty output directory.")
    audio, source = load_media(input_path)
    if np.max(np.abs(audio)) < sep.EPS:
        raise ValueError("This recording is silent.")
    masks, metadata = _joint_masks(audio, config["tracks"])
    tracks = config["tracks"]
    images = None
    if config.get("separator", "spectral") == "adaptive":
        images = _spatial_images(audio, tracks, masks)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".voicehunter-", dir=directory.parent) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        # Accumulate in float64 to avoid progressive subtraction errors. Store
        # stems as float WAV at unity gain, including any over-unity samples.
        remaining = audio.astype(np.float64)
        rendered = np.zeros_like(remaining)
        zero = np.zeros_like(audio)
        for i, (mask, track) in enumerate(zip(masks, tracks)):
            if images is None:
                stem, _ = extract_stereo(audio, mask, n_fft=sep.N_FFT, hop=sep.HOP)
            else:
                stem = images[i]
            write_wav(stage / (track["id"] + ".wav"), stem, stem=True)
            remaining -= stem
            rendered += remix(stem, zero, target_gain_db=track.get("gain_db", 0), pan=track.get("pan", 0))
        remaining = remaining.astype(np.float32)
        write_wav(stage / "remaining.wav", remaining, stem=True)
        width = config.get("background_width", 1.0)
        rendered += remix(zero, remaining, width=width)
        gain = write_wav(stage / "remix.wav", rendered)
        manifest = {
            "source": Path(input_path).name, "source_media": source, "samples": len(audio),
            "sample_rate": SAMPLE_RATE, "channels": 2, "tracks": metadata,
            "requested_separator": config.get("separator", "spectral"),
            "spatial_filter_applied": images is not None,
            "remaining": "remaining.wav", "background_width": width, "remix_peak_protection_gain": gain,
            "projection": ("Reference-conditioned stereo Wiener projection." if images is not None else
                           "Shared spectral masks proportional to the strongest target claim."),
            "residual_policy": "The stereo residual retains unassigned content; unity float stems sum to the decoded source.",
        }
        (stage / "settings.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(directory)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    parser.add_argument("--separator", choices=("spectral", "adaptive"),
                        help="override the config separator; default spectral, adaptive for distinct stationary examples")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.separator is not None:
            if not isinstance(config, dict):
                raise ValueError("Config must contain a 'tracks' list.")
            config["separator"] = args.separator
        export_tracks(args.input, config, args.out)
    except (ValueError, OSError, RuntimeError, ArithmeticError) as exc:
        parser.exit(1, f"{exc}\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
