from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct

from common import ExperimentConfig, expected_audio_windows, load_audio_mono, validate_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 5 s acoustic features from the NMED-E original audio.")
    parser.add_argument("--audio", default="data/stimuli/nmed_e_original.wav")
    parser.add_argument("--output-csv", default="outputs/features/original_acoustic_features.csv")
    parser.add_argument("--sample-rate", type=int, default=24000)
    return parser.parse_args()


def frame_signal(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    starts = np.arange(0, len(y) - frame_length + 1, hop_length)
    frames = np.stack([y[start : start + frame_length] for start in starts], axis=0)
    return frames * np.hanning(frame_length)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(freqs: np.ndarray, sr: int, n_mels: int = 32) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sr / 2]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.searchsorted(freqs, hz_points)
    filters = np.zeros((n_mels, len(freqs)), dtype=float)
    for mel_idx in range(n_mels):
        left, center, right = bins[mel_idx : mel_idx + 3]
        if center > left:
            filters[mel_idx, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            filters[mel_idx, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return filters


def band_energy(power: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(power[:, mask].mean())


def pitch_class_features(power: np.ndarray, freqs: np.ndarray) -> tuple[float, float, float]:
    mask = freqs >= 27.5
    safe_freqs = freqs[mask]
    if safe_freqs.size == 0:
        return 0.0, 0.0, 0.0
    midi = np.rint(69 + 12 * np.log2(safe_freqs / 440.0)).astype(int)
    pitch_classes = np.mod(midi, 12)
    chroma = np.zeros((power.shape[0], 12), dtype=float)
    for pc in range(12):
        chroma[:, pc] = power[:, mask][:, pitch_classes == pc].sum(axis=1)
    chroma = chroma / np.maximum(chroma.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(chroma * np.log2(np.maximum(chroma, 1e-12))).sum(axis=1)
    return float(chroma.mean()), float(chroma.std()), float(entropy.mean())


def summarize_window(y: np.ndarray, sr: int) -> dict[str, float]:
    frames = frame_signal(y)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    power = spectrum**2 + 1e-12
    power_sum = power.sum(axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.diff(np.signbit(frames), axis=1) != 0, axis=1)
    centroid = (power * freqs).sum(axis=1) / power_sum
    bandwidth = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / power_sum)
    cumulative = np.cumsum(power, axis=1)
    rolloff = freqs[np.argmax(cumulative >= 0.85 * power_sum[:, None], axis=1)]
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    norm_power = power / power_sum[:, None]
    spectral_flux = np.sqrt(np.sum(np.diff(norm_power, axis=0) ** 2, axis=1)) if len(norm_power) > 1 else np.array([0.0])
    chroma_mean, chroma_sd, chroma_entropy = pitch_class_features(power, freqs)
    filters = mel_filterbank(freqs, sr)
    log_mel = np.log(np.maximum(power @ filters.T, 1e-12))
    mfcc = dct(log_mel, type=2, norm="ortho", axis=1)[:, :13]

    features: dict[str, float] = {
        "rms_mean": float(np.mean(rms)),
        "rms_sd": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_sd": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flux_mean": float(np.mean(spectral_flux)),
        "spectral_flux_sd": float(np.std(spectral_flux)),
        "low_band_energy": band_energy(power, freqs, 20.0, 250.0),
        "mid_band_energy": band_energy(power, freqs, 250.0, 2000.0),
        "high_band_energy": band_energy(power, freqs, 2000.0, 6000.0),
        "upper_band_energy": band_energy(power, freqs, 6000.0, sr / 2),
        "chroma_mean": chroma_mean,
        "chroma_sd": chroma_sd,
        "chroma_entropy_mean": chroma_entropy,
    }
    for index, value in enumerate(np.mean(mfcc, axis=0), start=1):
        features[f"mfcc_{index:02d}_mean"] = float(value)
    return features


def main() -> None:
    args = parse_args()
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

    windows = expected_audio_windows(config)
    rows = []
    for row in windows.to_dict("records"):
        start = int(round(row["start_sec"] * sr))
        end = int(round(row["end_sec"] * sr))
        features = summarize_window(y[start:end], sr)
        rows.append({**row, **features})

    feature_df = pd.DataFrame(rows)
    validate_feature_table(feature_df, config)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
