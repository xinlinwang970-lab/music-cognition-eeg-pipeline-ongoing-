from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    ExperimentConfig,
    build_time_feature_table,
    build_original_group_timeseries,
    centered_rolling_residual,
    expected_audio_windows,
    numeric_feature_columns,
    validate_feature_table,
)


def make_window_df(n_windows: int = 24) -> pd.DataFrame:
    rows = []
    for subject_id in ["S01", "S02"]:
        for condition in ["original", "control"]:
            for window_id in range(n_windows):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": condition,
                        "window_id": window_id,
                        "start_sec": float(window_id * 5),
                        "end_sec": float(window_id * 5 + 5),
                        "center_sec": float(window_id * 5 + 2.5),
                        "beta_log_power": float(window_id + (subject_id == "S02")),
                        "engagement_mean": float(window_id * 2 + (subject_id == "S02")),
                    }
                )
    return pd.DataFrame(rows)


def test_centered_rolling_residual_has_expected_edges():
    residual = centered_rolling_residual([1, 2, 3, 4, 5], window_size=3)
    assert np.isnan(residual.iloc[0])
    assert residual.iloc[2] == 0
    assert np.isnan(residual.iloc[-1])


def test_build_original_group_timeseries_adds_fast60_columns():
    config = ExperimentConfig(stimulus_duration_seconds=120.0, fast_timescale_seconds=30.0)
    grouped = build_original_group_timeseries(make_window_df(), config=config)
    assert len(grouped) == 24
    assert grouped["n_subjects"].min() == 2
    assert "beta_fast60" in grouped.columns
    assert "engagement_fast60" in grouped.columns


def test_expected_audio_windows_and_feature_validation():
    config = ExperimentConfig(stimulus_duration_seconds=20.0)
    windows = expected_audio_windows(config)
    windows["rms_mean"] = [0.1, 0.2, 0.3, 0.4]
    validate_feature_table(windows, config)
    assert windows["window_id"].tolist() == [0, 1, 2, 3]


def test_numeric_feature_columns_excludes_identifiers():
    features = pd.DataFrame(
        {
            "window_id": [0, 1],
            "start_sec": [0.0, 5.0],
            "end_sec": [5.0, 10.0],
            "center_sec": [2.5, 7.5],
            "emb_0000": [0.1, 0.2],
            "embedding_norm": [1.0, 1.1],
            "source_path": ["a.wav", "a.wav"],
        }
    )
    assert numeric_feature_columns(features) == ["emb_0000", "embedding_norm"]


def test_build_time_feature_table_contains_only_chronological_features():
    config = ExperimentConfig(stimulus_duration_seconds=20.0)
    features = build_time_feature_table(config)
    validate_feature_table(features, config)
    assert numeric_feature_columns(features) == [
        "time_elapsed_fraction",
        "time_centered",
        "time_centered_sq",
        "time_centered_cu",
    ]
    assert features["time_elapsed_fraction"].between(0, 1).all()
    assert np.isclose(features["time_centered"].mean(), 0.0)
