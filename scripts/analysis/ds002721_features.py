from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

BETA_BAND = (15.0, 30.0)
FEATURE_SPEC_ID = "frontal_beta_v1"
PREPROCESSING_ID = "raw_edf_no_artifact_rejection_trial_level_v1"


def beta_log_power(signal: np.ndarray, fs: float, band: tuple[float, float] = BETA_BAND) -> float:
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if len(signal) < int(fs):
        return float("nan")
    nperseg = min(len(signal), int(round(fs * 2.0)))
    freqs, psd = welch(signal, fs=fs, window="hann", nperseg=nperseg, detrend="constant")
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 2:
        return float("nan")
    power = float(np.trapezoid(psd[mask], freqs[mask]))
    return float(np.log10(power + np.finfo(float).tiny))


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 5 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def subject_effects(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, df in trials.groupby("subject_id", sort=True):
        row: dict[str, object] = {"subject_id": subject_id, "n_trials": int(len(df))}
        for target in ["energy", "tension", "pleasantness"]:
            row[f"rho_{target}_primary"] = spearman(df["beta_log_power"], df[target])
            row[f"rho_{target}_roi"] = spearman(df["frontal_roi_beta_log_power"], df[target])
        rows.append(row)
    return pd.DataFrame(rows)
