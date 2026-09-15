from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import signal
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "toy1_1_outputs"

EEG_FILE = DATA_DIR / "CleanEEG_stim22.mat"
CB_FILE = DATA_DIR / "CleanCB_All.mat"
SFP_FILE = DATA_DIR / "GSN-HydroCel-129.sfp"

SUBJECT = "S01"
CONDITION = "original/intact stimulus 22"
CHANNEL = "E11"
CHANNEL_INDEX = 10
WINDOW_SEC = 5.0
MIN_SHIFT_SECONDS = 30.0
SEED = 20260817


def integrate_band(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_welch_psd(
    segment: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if nperseg is None:
        nperseg = min(int(round(2.0 * fs)), len(segment))
    nperseg = min(nperseg, len(segment))
    if noverlap is None:
        noverlap = nperseg // 2
    return signal.welch(
        segment,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )


def compute_band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        raise RuntimeError(f"Not enough frequency bins for band {low}-{high} Hz.")
    return integrate_band(psd[mask], freqs[mask])


def make_complete_windows(n_complete_windows: int, window_sec: float) -> pd.DataFrame:
    starts = np.arange(n_complete_windows, dtype=float) * window_sec
    return pd.DataFrame(
        {
            "window_id": np.arange(n_complete_windows, dtype=int),
            "start_sec": starts,
            "end_sec": starts + window_sec,
            "center_sec": starts + window_sec / 2.0,
        }
    )


def zscore_series(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    sd = np.std(arr)
    if sd == 0 or not np.isfinite(sd):
        raise RuntimeError("Cannot z-score constant or invalid series.")
    return (arr - np.mean(arr)) / sd


def read_sfp(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "label": parts[0],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def load_subject_series() -> dict[str, object]:
    eeg_meta = scipy_io.loadmat(EEG_FILE, squeeze_me=True, variable_names=["subs22", "fs"])
    cb = scipy_io.loadmat(CB_FILE, squeeze_me=True)

    eeg_subjects = [str(s).strip() for s in np.ravel(eeg_meta["subs22"])]
    cb_subjects = [str(s).strip() for s in np.ravel(cb["subIDs"])]
    if SUBJECT not in eeg_subjects:
        raise RuntimeError(f"{SUBJECT} is not present in {EEG_FILE.name}.")
    if SUBJECT not in cb_subjects:
        raise RuntimeError(f"{SUBJECT} is not present in {CB_FILE.name}.")

    eeg_trial_index = eeg_subjects.index(SUBJECT)
    cb_subject_index = cb_subjects.index(SUBJECT)
    fs_eeg = float(np.ravel(eeg_meta["fs"])[0])
    fs_cb = float(np.ravel(cb["fs"])[0])

    eeg_mat = scipy_io.loadmat(EEG_FILE, squeeze_me=True, variable_names=["eeg22"])
    eeg = eeg_mat["eeg22"]
    x = np.asarray(eeg[CHANNEL_INDEX, :, eeg_trial_index], dtype=float).copy()
    del eeg, eeg_mat
    gc.collect()

    all_cb = np.asarray(cb["allCB"], dtype=float)
    xax = np.asarray(cb["xax"], dtype=float).ravel()
    engagement = all_cb[:, cb_subject_index, 0].copy()

    if not np.isfinite(x).all():
        raise RuntimeError("Selected EEG contains NaN or Inf.")
    if not np.isfinite(engagement).all():
        raise RuntimeError("Selected engagement contains NaN or Inf.")

    return {
        "x": x,
        "engagement": engagement,
        "xax": xax,
        "fs_eeg": fs_eeg,
        "fs_cb": fs_cb,
    }


def synthetic_validation() -> dict[str, float]:
    fs_test = 125
    t = np.arange(0, 10, 1 / fs_test)
    x11 = np.sin(2 * np.pi * 11 * t)
    x20 = np.sin(2 * np.pi * 20 * t)
    f11, p11 = compute_welch_psd(x11, fs_test)
    f20, p20 = compute_welch_psd(x20, fs_test)
    ua11 = compute_band_power(f11, p11, 10, 12)
    b11 = compute_band_power(f11, p11, 15, 30)
    ua20 = compute_band_power(f20, p20, 10, 12)
    b20 = compute_band_power(f20, p20, 15, 30)
    assert ua11 > b11
    assert b20 > ua20
    return {
        "signal_11hz_upper_alpha": ua11,
        "signal_11hz_beta": b11,
        "signal_20hz_upper_alpha": ua20,
        "signal_20hz_beta": b20,
    }


def compute_features(
    x: np.ndarray,
    engagement: np.ndarray,
    engagement_time: np.ndarray,
    fs_eeg: float,
    fs_cb: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    eeg_duration = len(x) / fs_eeg
    engagement_duration = len(engagement) / fs_cb
    common_duration = min(eeg_duration, engagement_duration)
    n_complete_windows = int(np.floor(common_duration / WINDOW_SEC))
    usable_duration = n_complete_windows * WINDOW_SEC
    windows = make_complete_windows(n_complete_windows, WINDOW_SEC)

    feature_rows = []
    window_samples = int(round(WINDOW_SEC * fs_eeg))
    for row in windows.itertuples(index=False):
        start = int(round(row.start_sec * fs_eeg))
        end = start + window_samples
        if end > len(x):
            raise RuntimeError("EEG complete-window calculation exceeded available samples.")
        freqs, pxx = compute_welch_psd(x[start:end], fs_eeg)
        upper_alpha = compute_band_power(freqs, pxx, 10, 12)
        beta = compute_band_power(freqs, pxx, 15, 30)
        feature_rows.append(
            {
                "window_id": int(row.window_id),
                "start_sec": float(row.start_sec),
                "end_sec": float(row.end_sec),
                "center_sec": float(row.center_sec),
                "upper_alpha_log_power": float(np.log10(upper_alpha + 1e-20)),
                "beta_log_power": float(np.log10(beta + 1e-20)),
            }
        )

    engagement_rows = []
    for row in windows.itertuples(index=False):
        mask = (engagement_time >= row.start_sec) & (engagement_time < row.end_sec)
        if not mask.any():
            raise RuntimeError(f"No engagement samples in window {row.window_id}.")
        engagement_rows.append(
            {
                "window_id": int(row.window_id),
                "engagement_mean": float(np.mean(engagement[mask])),
                "n_engagement_samples": int(mask.sum()),
            }
        )

    analysis = pd.DataFrame(feature_rows).merge(pd.DataFrame(engagement_rows), on="window_id")
    if analysis["window_id"].duplicated().any():
        raise RuntimeError("Duplicate window IDs after merge.")
    if analysis[["upper_alpha_log_power", "beta_log_power", "engagement_mean"]].isna().any().any():
        raise RuntimeError("Missing feature or engagement value after merge.")

    timing = {
        "eeg_duration_sec": eeg_duration,
        "engagement_duration_sec": engagement_duration,
        "common_duration_sec": common_duration,
        "n_complete_windows": int(n_complete_windows),
        "usable_duration_sec": usable_duration,
        "discarded_eeg_tail_sec": eeg_duration - usable_duration,
        "discarded_engagement_tail_sec": engagement_duration - usable_duration,
    }
    return analysis, timing


def valid_circular_shifts(n_windows: int, min_shift_windows: int) -> np.ndarray:
    shifts = []
    for shift in range(n_windows):
        circular_distance = min(shift, n_windows - shift)
        if circular_distance >= min_shift_windows:
            shifts.append(shift)
    return np.asarray(shifts, dtype=int)


def exact_circular_shift_test(feature: np.ndarray, engagement: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    rho_obs = float(spearmanr(feature, engagement).statistic)
    min_shift_windows = int(np.ceil(MIN_SHIFT_SECONDS / WINDOW_SEC))
    shifts = valid_circular_shifts(len(feature), min_shift_windows)
    rows = []
    for shift in shifts:
        rho = float(spearmanr(feature, np.roll(engagement, shift)).statistic)
        rows.append(
            {
                "shift_windows": int(shift),
                "shift_seconds": float(shift * WINDOW_SEC),
                "rho": rho,
            }
        )
    null = pd.DataFrame(rows)
    p_empirical = float((1 + np.sum(np.abs(null["rho"]) >= abs(rho_obs))) / (len(null) + 1))
    return rho_obs, p_empirical, null


def write_e11_location(sfp: pd.DataFrame) -> dict[str, str]:
    e11 = sfp[sfp["label"].eq(CHANNEL)].iloc[0]
    fid_nz = sfp[sfp["label"].eq("FidNz")].iloc[0]
    location_text = f"""Dataset channel: {CHANNEL}
Coordinates: x={e11['x']}, y={e11['y']}, z={e11['z']}
Coordinate convention: inferred from {SFP_FILE.name}; FidNz has strongly positive y, so +y is anterior, x is left-right, and +z is superior.
Approximate scalp region: frontal/anterior midline.
Nearest standard electrode if identifiable: Fz, approximate. The NMED-E clean data use EGI numbered electrodes, not native 10-20 labels, so this mapping should be treated as a practical anatomical approximation rather than a publication-grade relabeling.
Confidence: medium.
Source metadata file: {SFP_FILE}
Dataset README note: the cleaned montage is EGI numbered electrodes 1-124 plus the vertex reference (129).
"""
    (OUTPUT_DIR / "e11_location.txt").write_text(location_text, encoding="utf-8")
    return {
        "channel_region": "frontal/anterior midline",
        "nearest_standard_electrode": "Fz (approximate)",
        "confidence": "medium",
        "coordinate_convention": "+y anterior, x left-right, +z superior",
    }


def plot_e11_location(sfp: pd.DataFrame) -> None:
    sensors = sfp[sfp["label"].str.match(r"^E\d+$", na=False)].copy()
    e11 = sensors[sensors["label"].eq(CHANNEL)].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sensors["x"], sensors["y"], s=18, color="#777777", alpha=0.75, label="EGI sensors")
    ax.scatter([e11["x"]], [e11["y"]], s=120, color="#d62728", edgecolor="black", label=CHANNEL, zorder=3)
    for label in ["FidNz", "FidT9", "FidT10"]:
        row = sfp[sfp["label"].eq(label)]
        if not row.empty:
            ax.scatter(row["x"], row["y"], marker="x", s=70, color="#1f77b4")
            ax.text(float(row["x"].iloc[0]), float(row["y"].iloc[0]), label, fontsize=8)
    ax.text(float(e11["x"]) + 0.2, float(e11["y"]) + 0.15, CHANNEL, fontsize=11, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate (+ anterior toward nasion)")
    ax.set_title("NMED-E HydroCel sensor map: E11 highlighted")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_e11_sensor_location.png", dpi=170)
    plt.close(fig)


def plot_timeseries(analysis: pd.DataFrame) -> None:
    plot_df = analysis.copy()
    plot_df["z_upper_alpha"] = zscore_series(plot_df["upper_alpha_log_power"])
    plot_df["z_beta"] = zscore_series(plot_df["beta_log_power"])
    plot_df["z_engagement"] = zscore_series(plot_df["engagement_mean"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["center_sec"], plot_df["z_upper_alpha"], label="z upper-alpha 10-12 Hz", linewidth=1.0)
    ax.plot(plot_df["center_sec"], plot_df["z_beta"], label="z beta 15-30 Hz", linewidth=1.0)
    ax.plot(plot_df["center_sec"], plot_df["z_engagement"], label="z engagement", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z-score")
    ax.set_title(f"{SUBJECT} {CHANNEL}: complete-window alpha/beta power and engagement")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_full_timeseries_95_windows.png", dpi=160)
    plt.close(fig)


def plot_permutation(null: pd.DataFrame, rho_obs: float, p_empirical: float, title: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null["rho"], bins=24, color="#9ecae1", edgecolor="#333333", alpha=0.9)
    ax.axvline(rho_obs, color="#d62728", linewidth=2, label=f"observed rho={rho_obs:.3f}")
    ax.axvline(-rho_obs, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Circular-shift Spearman rho")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}: observed rho={rho_obs:.3f}, empirical p={p_empirical:.3f}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=160)
    plt.close(fig)


def plot_running_correlation(analysis: pd.DataFrame) -> dict[str, list[float]]:
    local_windows = 12
    centers = []
    alpha_rhos = []
    beta_rhos = []
    for start in range(0, len(analysis) - local_windows + 1):
        chunk = analysis.iloc[start : start + local_windows]
        centers.append(float(chunk["center_sec"].mean()))
        alpha_rhos.append(float(spearmanr(chunk["upper_alpha_log_power"], chunk["engagement_mean"]).statistic))
        beta_rhos.append(float(spearmanr(chunk["beta_log_power"], chunk["engagement_mean"]).statistic))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(centers, alpha_rhos, label="upper-alpha running rho", linewidth=1.2)
    ax.plot(centers, beta_rhos, label="beta running rho", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("60-s local Spearman rho")
    ax.set_title("Exploratory running correlation; not inferential evidence")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_running_correlation_exploratory.png", dpi=160)
    plt.close(fig)
    return {"center_sec": centers, "alpha_rho": alpha_rhos, "beta_rho": beta_rhos}


def linear_trend_sensitivity(analysis: pd.DataFrame) -> dict[str, float]:
    time = analysis["center_sec"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    detrended = {}
    for column in ["upper_alpha_log_power", "beta_log_power", "engagement_mean"]:
        values = analysis[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(time, values, 1)
        out[f"{column}_linear_slope_per_sec"] = float(slope)
        detrended[column] = values - (slope * time + intercept)
    out["rho_alpha_detrended_descriptive"] = float(
        spearmanr(detrended["upper_alpha_log_power"], detrended["engagement_mean"]).statistic
    )
    out["rho_beta_detrended_descriptive"] = float(
        spearmanr(detrended["beta_log_power"], detrended["engagement_mean"]).statistic
    )
    return out


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.default_rng(SEED)

    for path in [EEG_FILE, CB_FILE, SFP_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    validation = synthetic_validation()
    series = load_subject_series()
    x = series["x"]
    engagement = series["engagement"]
    xax = series["xax"]
    fs_eeg = float(series["fs_eeg"])
    fs_cb = float(series["fs_cb"])
    engagement_time = xax if abs(float(np.nanmax(xax)) - len(engagement) / fs_cb) < 5 else np.arange(len(engagement)) / fs_cb

    analysis, timing = compute_features(x, engagement, engagement_time, fs_eeg, fs_cb)
    analysis.to_csv(OUTPUT_DIR / "analysis_table_95_windows.csv", index=False)

    if timing["n_complete_windows"] != 95:
        raise RuntimeError(f"Expected 95 complete windows, got {timing['n_complete_windows']}.")

    sfp = read_sfp(SFP_FILE)
    location = write_e11_location(sfp)
    plot_e11_location(sfp)

    rho_alpha, p_alpha, alpha_null = exact_circular_shift_test(
        analysis["upper_alpha_log_power"].to_numpy(dtype=float),
        analysis["engagement_mean"].to_numpy(dtype=float),
    )
    rho_beta, p_beta, beta_null = exact_circular_shift_test(
        analysis["beta_log_power"].to_numpy(dtype=float),
        analysis["engagement_mean"].to_numpy(dtype=float),
    )
    alpha_null.to_csv(OUTPUT_DIR / "alpha_permutation_null.csv", index=False)
    beta_null.to_csv(OUTPUT_DIR / "beta_permutation_null.csv", index=False)

    permutation_summary = {
        "method": "exact circular shift over all valid unique shifts",
        "seed": SEED,
        "n_windows": int(len(analysis)),
        "min_shift_windows": int(np.ceil(MIN_SHIFT_SECONDS / WINDOW_SEC)),
        "min_shift_seconds": MIN_SHIFT_SECONDS,
        "n_valid_shifts": int(len(alpha_null)),
        "excluded_shifts": "0 and circular-distance 1-5 windows",
        "upper_alpha_observed_rho": rho_alpha,
        "upper_alpha_empirical_p": p_alpha,
        "beta_observed_rho": rho_beta,
        "beta_empirical_p": p_beta,
    }
    (OUTPUT_DIR / "permutation_summary.json").write_text(
        json.dumps(permutation_summary, indent=2),
        encoding="utf-8",
    )

    plot_timeseries(analysis)
    plot_permutation(alpha_null, rho_alpha, p_alpha, "Upper-alpha circular-shift null", "03_upper_alpha_permutation.png")
    plot_permutation(beta_null, rho_beta, p_beta, "Beta circular-shift null", "04_beta_permutation.png")
    running = plot_running_correlation(analysis)
    pd.DataFrame(running).to_csv(OUTPUT_DIR / "running_correlation_exploratory.csv", index=False)
    trend = linear_trend_sensitivity(analysis)

    summary = {
        "subject": SUBJECT,
        "condition": CONDITION,
        "channel": CHANNEL,
        "channel_region": location["channel_region"],
        "nearest_standard_electrode": location["nearest_standard_electrode"],
        "eeg_sampling_rate_hz": fs_eeg,
        "engagement_sampling_rate_hz": fs_cb,
        "common_duration_sec": timing["common_duration_sec"],
        "usable_duration_sec": timing["usable_duration_sec"],
        "n_complete_windows": timing["n_complete_windows"],
        "upper_alpha_vs_engagement_rho": rho_alpha,
        "upper_alpha_empirical_p": p_alpha,
        "beta_vs_engagement_rho": rho_beta,
        "beta_empirical_p": p_beta,
        "interpretation_boundary": "Single-subject exploratory analysis; not a biomarker, population inference, or real-time decoding result.",
        "timing": timing,
        "synthetic_validation": validation,
        "permutation": permutation_summary,
        "linear_trend_sensitivity": trend,
    }
    (OUTPUT_DIR / "toy1_1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("NMED-E TOY1.1 COMPLETE")
    print()
    print("Subject:", SUBJECT)
    print("Condition:", CONDITION)
    print("Channel:", CHANNEL)
    print("Channel region:", location["channel_region"])
    print("Nearest standard electrode:", location["nearest_standard_electrode"])
    print()
    print("EEG duration:", timing["eeg_duration_sec"])
    print("Engagement duration:", timing["engagement_duration_sec"])
    print("Common duration:", timing["common_duration_sec"])
    print("Complete windows:", timing["n_complete_windows"])
    print("Usable duration:", timing["usable_duration_sec"])
    print("Discarded EEG tail:", timing["discarded_eeg_tail_sec"])
    print("Discarded engagement tail:", timing["discarded_engagement_tail_sec"])
    print()
    print("Original Toy1:")
    print("alpha rho = 0.2566495")
    print("beta rho  = 0.2919898")
    print()
    print("Toy1.1 complete windows:")
    print("alpha rho =", rho_alpha)
    print("beta rho  =", rho_beta)
    print()
    print("Upper-alpha:")
    print("Observed Spearman rho:", rho_alpha)
    print("Circular-shift empirical p:", p_alpha)
    print()
    print("Beta:")
    print("Observed Spearman rho:", rho_beta)
    print("Circular-shift empirical p:", p_beta)
    print()
    print("Interpretation:")
    print("Single-subject exploratory result only.")
    return summary


if __name__ == "__main__":
    main()
