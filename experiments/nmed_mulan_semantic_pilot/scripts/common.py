from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


REQUIRED_WINDOW_COLUMNS = {
    "subject_id",
    "condition",
    "window_id",
    "start_sec",
    "end_sec",
    "center_sec",
    "beta_log_power",
    "engagement_mean",
}

NON_FEATURE_COLUMNS = {
    "condition",
    "window_id",
    "start_sec",
    "end_sec",
    "center_sec",
    "source_path",
}


@dataclass(frozen=True)
class ExperimentConfig:
    condition: str = "original"
    window_seconds: float = 5.0
    stimulus_duration_seconds: float = 475.0
    fast_timescale_seconds: float = 60.0

    @property
    def expected_windows(self) -> int:
        return int(round(self.stimulus_duration_seconds / self.window_seconds))

    @property
    def fast_rolling_windows(self) -> int:
        return int(round(self.fast_timescale_seconds / self.window_seconds))


def read_nmed_window_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NMED window table not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_WINDOW_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"NMED window table is missing required columns: {missing}")
    return df


def centered_rolling_residual(values: Iterable[float], window_size: int) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    slow = series.rolling(window=window_size, center=True, min_periods=window_size).mean()
    return series - slow


def build_original_group_timeseries(
    window_df: pd.DataFrame,
    config: ExperimentConfig = ExperimentConfig(),
) -> pd.DataFrame:
    original = window_df.loc[window_df["condition"] == config.condition].copy()
    if original.empty:
        raise ValueError(f"No rows found for condition={config.condition!r}")

    grouped = (
        original.groupby("window_id", as_index=False)
        .agg(
            start_sec=("start_sec", "first"),
            end_sec=("end_sec", "first"),
            center_sec=("center_sec", "first"),
            n_subjects=("subject_id", "nunique"),
            beta_mean=("beta_log_power", "mean"),
            beta_sd=("beta_log_power", "std"),
            engagement_mean=("engagement_mean", "mean"),
            engagement_sd=("engagement_mean", "std"),
        )
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    expected = config.expected_windows
    if len(grouped) != expected:
        raise ValueError(
            f"Expected {expected} windows, found {len(grouped)}. "
            "If the source pipeline changes its valid EEG/engagement window span, update ExperimentConfig."
        )

    rolling = config.fast_rolling_windows
    grouped["beta_slow60"] = grouped["beta_mean"].rolling(rolling, center=True, min_periods=rolling).mean()
    grouped["engagement_slow60"] = grouped["engagement_mean"].rolling(
        rolling, center=True, min_periods=rolling
    ).mean()
    grouped["beta_fast60"] = grouped["beta_mean"] - grouped["beta_slow60"]
    grouped["engagement_fast60"] = grouped["engagement_mean"] - grouped["engagement_slow60"]
    return grouped


def expected_audio_windows(config: ExperimentConfig = ExperimentConfig()) -> pd.DataFrame:
    rows = []
    for window_id in range(config.expected_windows):
        start = window_id * config.window_seconds
        end = start + config.window_seconds
        rows.append(
            {
                "window_id": window_id,
                "start_sec": start,
                "end_sec": end,
                "center_sec": (start + end) / 2.0,
            }
        )
    return pd.DataFrame(rows)


def build_time_feature_table(config: ExperimentConfig = ExperimentConfig()) -> pd.DataFrame:
    windows = expected_audio_windows(config)
    fraction = windows["center_sec"] / config.stimulus_duration_seconds
    centered = windows["center_sec"] - windows["center_sec"].mean()
    scaled = centered / centered.std(ddof=0)
    windows["time_elapsed_fraction"] = fraction
    windows["time_centered"] = scaled
    windows["time_centered_sq"] = scaled**2
    windows["time_centered_cu"] = scaled**3
    return windows


def load_audio_mono(path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, native_sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if native_sr != sample_rate:
        factor = gcd(native_sr, sample_rate)
        mono = signal.resample_poly(mono, sample_rate // factor, native_sr // factor).astype("float32")
    return mono, sample_rate


def validate_feature_table(features: pd.DataFrame, config: ExperimentConfig = ExperimentConfig()) -> None:
    required = {"window_id", "start_sec", "end_sec", "center_sec"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")
    if len(features) != config.expected_windows:
        raise ValueError(f"Expected {config.expected_windows} feature rows, found {len(features)}")
    expected_ids = list(range(config.expected_windows))
    if features["window_id"].tolist() != expected_ids:
        raise ValueError("Feature table window_id values must run from 0 to expected_windows - 1")


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    columns = []
    for column in features.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            columns.append(column)
    return columns


def align_targets_and_features(
    targets: pd.DataFrame,
    features: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if target_column not in targets.columns:
        raise ValueError(f"Target column not found: {target_column}")
    validate_feature_table(features)
    if feature_columns is None:
        feature_columns = numeric_feature_columns(features)
    if not feature_columns:
        raise ValueError("No numeric feature columns found")

    merged = targets.merge(features[["window_id", *feature_columns]], on="window_id", how="inner")
    keep_columns = ["window_id", "start_sec", "end_sec", "center_sec", target_column, *feature_columns]
    merged = merged[keep_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if len(merged) < 20:
        raise ValueError(f"Too few complete rows after alignment: {len(merged)}")
    return merged, feature_columns
