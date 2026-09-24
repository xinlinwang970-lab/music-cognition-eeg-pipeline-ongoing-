from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ExperimentConfig, expected_audio_windows, load_audio_mono, validate_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MuQ-MuLan embeddings from NMED-E original audio windows.")
    parser.add_argument("--audio", default="data/stimuli/nmed_e_original.wav")
    parser.add_argument("--output-csv", default="outputs/features/original_muq_mulan_embeddings.csv")
    parser.add_argument("--model", default="OpenMuQ/MuQ-MuLan-large")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def as_numpy_2d(embeds) -> np.ndarray:
    if isinstance(embeds, (tuple, list)):
        embeds = embeds[0]
    if hasattr(embeds, "last_hidden_state"):
        embeds = embeds.last_hidden_state
    if hasattr(embeds, "detach"):
        embeds = embeds.detach().cpu().numpy()
    embeds = np.asarray(embeds)
    if embeds.ndim == 3:
        embeds = embeds.mean(axis=1)
    if embeds.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeds.shape}")
    return embeds


def main() -> None:
    args = parse_args()
    try:
        import torch
        from muq import MuQMuLan
    except ImportError as exc:
        raise ImportError("Install optional MuQ dependencies first: python -m pip install -r requirements-muq.txt") from exc

    config = ExperimentConfig()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}. Place the NMED-E original audio in data/stimuli/."
        )

    y, sr = load_audio_mono(audio_path, args.sample_rate)
    required_samples = int(round(config.stimulus_duration_seconds * sr))
    if len(y) < required_samples:
        raise ValueError(f"Audio is shorter than {config.stimulus_duration_seconds} s after resampling: {audio_path}")
    if len(y) > required_samples:
        y = y[:required_samples]

    device = resolve_device(args.device)
    model = MuQMuLan.from_pretrained(args.model).to(device).eval()
    windows = expected_audio_windows(config)
    all_embeddings: list[np.ndarray] = []

    samples_per_window = int(round(config.window_seconds * sr))
    segments = []
    for window_id in range(config.expected_windows):
        start = window_id * samples_per_window
        end = start + samples_per_window
        segments.append(y[start:end])

    with torch.no_grad():
        for start in range(0, len(segments), args.batch_size):
            batch = np.stack(segments[start : start + args.batch_size], axis=0)
            wavs = torch.tensor(batch, dtype=torch.float32, device=device)
            embeds = model(wavs=wavs)
            all_embeddings.append(as_numpy_2d(embeds))

    embedding_matrix = np.concatenate(all_embeddings, axis=0)
    embedding_columns = [f"emb_{idx:04d}" for idx in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=embedding_columns)
    rows = pd.concat(
        [
            windows.reset_index(drop=True),
            pd.DataFrame({"embedding_norm": np.linalg.norm(embedding_matrix, axis=1)}),
            embedding_df,
        ],
        axis=1,
    )

    validate_feature_table(rows, config)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    print(f"Embedding shape: {embedding_matrix.shape}")


if __name__ == "__main__":
    main()
