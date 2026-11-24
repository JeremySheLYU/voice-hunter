"""Isolated optional four-stem Demucs adapter; all weights are provided locally."""
from pathlib import Path
import sys


def main(argv):
    import numpy as np
    import torch
    from demucs.apply import apply_model
    from demucs.states import load_model
    from demucs.audio import convert_audio

    checkpoint = Path(argv[2])
    # Official HTDemucs weights can include pickled model-constructor metadata.
    # Load only an explicitly configured checkpoint obtained from the publisher.
    package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_model(package).cpu().eval()
    if set(model.sources) != {"vocals", "drums", "bass", "other"}:
        raise ValueError("This adapter requires a standard four-stem checkpoint.")
    if argv[1] == "--probe":
        return
    _, target, rate_arg, folder_arg = argv[2:6]
    rate = int(rate_arg)
    folder = Path(folder_arg)
    x = torch.from_numpy(np.load(folder / "input.npy", allow_pickle=False).T.copy())
    working = convert_audio(x, rate, model.samplerate, model.audio_channels)
    reference = working.mean(0)
    mean = reference.mean()
    std = reference.std(unbiased=False).clamp_min(1e-7)
    with torch.inference_mode():
        estimates = apply_model(model, ((working - mean) / std)[None], shifts=0,
                                split=True, device=torch.device("cpu"), progress=False)[0]
    est = estimates[model.sources.index(target)] * std + mean
    converted = convert_audio(est, model.samplerate, rate, 2)
    if converted.shape[-1] < x.shape[1]:
        converted = torch.nn.functional.pad(converted, (0, x.shape[1] - converted.shape[-1]))
    np.save(folder / "target.npy", converted[:, :x.shape[1]].T.contiguous().numpy(),
            allow_pickle=False)


if __name__ == "__main__":
    main(sys.argv)
