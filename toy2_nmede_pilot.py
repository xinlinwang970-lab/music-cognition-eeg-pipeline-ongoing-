from __future__ import annotations

import gc
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy.stats import spearmanr

from toy1_1_nmede_diagnostics import (
    CHANNEL,
    CHANNEL_INDEX,
    MIN_SHIFT_SECONDS,
    SFP_FILE,
    WINDOW_SEC,
    compute_band_power,
    compute_welch_psd,
    read_sfp,
    synthetic_validation,
    valid_circular_shifts,
    zscore_series,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "toy2_outputs"

EEG_FILES = {
    "original": {
        "path": DATA_DIR / "CleanEEG_stim22.mat",
        "eeg_key": "eeg22",
        "subs_key": "subs22",
        "cb_stimulus_index": 0,
        "description": "original/intact stimulus 22",
    },
    "control": {
        "path": DATA_DIR / "CleanEEG_stim23.mat",
        "eeg_key": "eeg23",
        "subs_key": "subs23",
        "cb_stimulus_index": 1,
        "description": "control stimulus 23",
    },
}
CB_FILE = DATA_DIR / "CleanCB_All.mat"
N_SUBJECTS = 5


def subject_list(value: np.ndarray) -> list[str]:
    return [str(s).strip() for s in np.ravel(value)]


def select_subjects() -> tuple[list[str], dict[str, list[str]]]:
    subject_sets = {}
    for condition, spec in EEG_FILES.items():
        meta = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["subs_key"]])
        subject_sets[condition] = subject_list(meta[spec["subs_key"]])
    cb = scipy_io.loadmat(CB_FILE, squeeze_me=True, variable_names=["subIDs"])
    subject_sets["engagement"] = subject_list(cb["subIDs"])
    valid = sorted(set(subject_sets["original"]) & set(subject_sets["control"]) & set(subject_sets["engagement"]))
    if len(valid) < N_SUBJECTS:
        raise RuntimeError(f"Need {N_SUBJECTS} valid subjects, found {len(valid)}.")
    return valid[:N_SUBJECTS], subject_sets


def complete_windows(common_duration: float) -> pd.DataFrame:
    n_complete = int(np.floor(common_duration / WINDOW_SEC))
    starts = np.arange(n_complete, dtype=float) * WINDOW_SEC
    return pd.DataFrame(
        {
            "window_id": np.arange(n_complete, dtype=int),
            "start_sec": starts,
            "end_sec": starts + WINDOW_SEC,
            "center_sec": starts + WINDOW_SEC / 2.0,
        }
    )


def window_engagement(values: np.ndarray, time: np.ndarray, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in windows.itertuples(index=False):
        mask = (time >= row.start_sec) & (time < row.end_sec)
        if not mask.any():
            raise RuntimeError(f"No engagement samples in window {row.window_id}.")
        rows.append(
            {
                "window_id": int(row.window_id),
                "engagement_mean": float(np.mean(values[mask])),
                "n_engagement_samples": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def extract_features_for_subject(
    eeg_channel: np.ndarray,
    engagement: np.ndarray,
    engagement_time: np.ndarray,
    fs_eeg: float,
    fs_cb: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    eeg_duration = len(eeg_channel) / fs_eeg
    engagement_duration = len(engagement) / fs_cb
    common_duration = min(eeg_duration, engagement_duration)
    windows = complete_windows(common_duration)
    usable_duration = len(windows) * WINDOW_SEC
    window_samples = int(round(WINDOW_SEC * fs_eeg))

    feature_rows = []
    for row in windows.itertuples(index=False):
        start = int(round(row.start_sec * fs_eeg))
        end = start + window_samples
        if end > len(eeg_channel):
            raise RuntimeError("Complete EEG window exceeded available samples.")
        freqs, pxx = compute_welch_psd(eeg_channel[start:end], fs_eeg)
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

    eeg_features = pd.DataFrame(feature_rows)
    engagement_windows = window_engagement(engagement, engagement_time, windows)
    table = eeg_features.merge(
        engagement_windows[["window_id", "engagement_mean", "n_engagement_samples"]],
        on="window_id",
        how="inner",
    )
    if table[["upper_alpha_log_power", "beta_log_power", "engagement_mean"]].isna().any().any():
        raise RuntimeError("Missing values after EEG/engagement alignment.")
    timing = {
        "eeg_duration_sec": float(eeg_duration),
        "engagement_duration_sec": float(engagement_duration),
        "common_duration_sec": float(common_duration),
        "n_complete_windows": int(len(windows)),
        "usable_duration_sec": float(usable_duration),
        "discarded_eeg_tail_sec": float(eeg_duration - usable_duration),
        "discarded_engagement_tail_sec": float(engagement_duration - usable_duration),
    }
    return table, timing


def detrend(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(time, values, 1)
    return values - (slope * time + intercept)


def correlation_row(subject_id: str, condition: str, table: pd.DataFrame, detrended: bool) -> dict[str, object]:
    time = table["center_sec"].to_numpy(dtype=float)
    engagement = table["engagement_mean"].to_numpy(dtype=float)
    alpha = table["upper_alpha_log_power"].to_numpy(dtype=float)
    beta = table["beta_log_power"].to_numpy(dtype=float)
    if detrended:
        engagement = detrend(engagement, time)
        alpha = detrend(alpha, time)
        beta = detrend(beta, time)
    return {
        "subject_id": subject_id,
        "condition": condition,
        "rho_alpha_detrended" if detrended else "rho_alpha_raw": float(spearmanr(alpha, engagement).statistic),
        "rho_beta_detrended" if detrended else "rho_beta_raw": float(spearmanr(beta, engagement).statistic),
        "n_windows": int(len(table)),
    }


def circular_shift_result(subject_id: str, condition: str, feature_name: str, feature: np.ndarray, engagement: np.ndarray) -> dict[str, object]:
    rho_obs = float(spearmanr(feature, engagement).statistic)
    min_shift_windows = int(np.ceil(MIN_SHIFT_SECONDS / WINDOW_SEC))
    shifts = valid_circular_shifts(len(feature), min_shift_windows)
    null = [float(spearmanr(feature, np.roll(engagement, int(shift))).statistic) for shift in shifts]
    null_arr = np.asarray(null, dtype=float)
    p_empirical = float((1 + np.sum(np.abs(null_arr) >= abs(rho_obs))) / (len(null_arr) + 1))
    return {
        "subject_id": subject_id,
        "condition": condition,
        "feature": feature_name,
        "rho_observed": rho_obs,
        "empirical_two_sided_p": p_empirical,
        "n_valid_shifts": int(len(shifts)),
        "min_shift_windows": int(min_shift_windows),
        "min_shift_seconds": float(MIN_SHIFT_SECONDS),
    }


def compute_deltas(raw: pd.DataFrame, detrended: pd.DataFrame) -> pd.DataFrame:
    raw_wide = raw.pivot(index="subject_id", columns="condition", values=["rho_alpha_raw", "rho_beta_raw"])
    det_wide = detrended.pivot(index="subject_id", columns="condition", values=["rho_alpha_detrended", "rho_beta_detrended"])
    rows = []
    for subject_id in raw_wide.index:
        for feature, raw_key, det_key in [
            ("upper_alpha", "rho_alpha_raw", "rho_alpha_detrended"),
            ("beta", "rho_beta_raw", "rho_beta_detrended"),
        ]:
            rho_original_raw = float(raw_wide.loc[subject_id, (raw_key, "original")])
            rho_control_raw = float(raw_wide.loc[subject_id, (raw_key, "control")])
            rho_original_detrended = float(det_wide.loc[subject_id, (det_key, "original")])
            rho_control_detrended = float(det_wide.loc[subject_id, (det_key, "control")])
            rows.append(
                {
                    "subject_id": subject_id,
                    "feature": feature,
                    "rho_original_raw": rho_original_raw,
                    "rho_control_raw": rho_control_raw,
                    "delta_rho_raw": rho_original_raw - rho_control_raw,
                    "rho_original_detrended": rho_original_detrended,
                    "rho_control_detrended": rho_control_detrended,
                    "delta_rho_detrended": rho_original_detrended - rho_control_detrended,
                }
            )
    return pd.DataFrame(rows)


def describe_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, df in deltas.groupby("feature", sort=False):
        for column in ["delta_rho_raw", "delta_rho_detrended"]:
            values = df[column].to_numpy(dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "delta_type": column,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_positive": int(np.sum(values > 0)),
                    "n": int(len(values)),
                    "sign_flip_mean_p_exploratory": sign_flip_p(values),
                }
            )
    return pd.DataFrame(rows)


def sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    null = []
    for signs in itertools.product([-1, 1], repeat=len(values)):
        null.append(abs(float(np.mean(np.asarray(signs) * values))))
    return float((1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1))


def paired_plot(raw: pd.DataFrame, feature: str, output_name: str) -> None:
    value_col = "rho_alpha_raw" if feature == "upper_alpha" else "rho_beta_raw"
    fig, ax = plt.subplots(figsize=(6, 5))
    for subject_id, df in raw.groupby("subject_id"):
        ordered = df.set_index("condition").loc[["original", "control"]]
        y = ordered[value_col].to_numpy(dtype=float)
        ax.plot([0, 1], y, color="#999999", linewidth=1.0, zorder=1)
        ax.scatter([0, 1], y, s=50, label=subject_id, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1], ["Original", "Control"])
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"{feature.replace('_', ' ').title()}: Original vs Control")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=160)
    plt.close(fig)


def delta_raw_plot(deltas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = sorted(deltas["subject_id"].unique())
    x = np.arange(len(subjects))
    width = 0.36
    for offset, feature in [(-width / 2, "upper_alpha"), (width / 2, "beta")]:
        vals = deltas[deltas["feature"].eq(feature)].set_index("subject_id").loc[subjects, "delta_rho_raw"]
        ax.bar(x + offset, vals, width=width, label=feature.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, subjects)
    ax.set_ylabel("Delta rho raw (Original - Control)")
    ax.set_title("Raw delta rho by subject")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_delta_rho_raw.png", dpi=160)
    plt.close(fig)


def raw_vs_detrended_plot(deltas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"upper_alpha": "#1f77b4", "beta": "#ff7f0e"}
    for feature, df in deltas.groupby("feature"):
        ax.scatter(df["delta_rho_raw"], df["delta_rho_detrended"], s=60, label=feature.replace("_", " "), color=colors[feature])
        for row in df.itertuples(index=False):
            ax.text(row.delta_rho_raw + 0.01, row.delta_rho_detrended + 0.01, row.subject_id, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta rho raw")
    ax.set_ylabel("Delta rho detrended")
    ax.set_title("Original-Control delta: raw vs detrended")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_raw_vs_detrended_delta.png", dpi=160)
    plt.close(fig)


def recommendation(group_summary: pd.DataFrame) -> str:
    raw = group_summary[group_summary["delta_type"].eq("delta_rho_raw")]
    strong = raw[(raw["n_positive"] >= 4) & (raw["median"] >= 0.10)]
    if not strong.empty:
        return "STRONG GO"
    moderate = raw[(raw["n_positive"] >= 3) & (raw["median"] > 0)]
    if not moderate.empty:
        return "MODERATE GO"
    return "RECONSIDER"


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for spec in EEG_FILES.values():
        if not spec["path"].exists():
            raise FileNotFoundError(spec["path"])
    if not CB_FILE.exists():
        raise FileNotFoundError(CB_FILE)

    validation = synthetic_validation()
    selected_subjects, subject_sets = select_subjects()
    pd.DataFrame({"subject_id": selected_subjects, "selection_order": range(1, len(selected_subjects) + 1)}).to_csv(
        OUTPUT_DIR / "selected_subjects.csv",
        index=False,
    )

    sfp = read_sfp(SFP_FILE)
    e11 = sfp[sfp["label"].eq(CHANNEL)].iloc[0].to_dict()
    cb = scipy_io.loadmat(CB_FILE, squeeze_me=True)
    cb_subjects = subject_list(cb["subIDs"])
    all_cb = np.asarray(cb["allCB"], dtype=float)
    fs_cb = float(np.ravel(cb["fs"])[0])
    xax = np.asarray(cb["xax"], dtype=float).ravel()
    engagement_duration = all_cb.shape[0] / fs_cb
    engagement_time = xax if abs(float(np.nanmax(xax)) - engagement_duration) < 5 else np.arange(all_cb.shape[0]) / fs_cb

    window_tables = []
    timing_rows = []
    raw_rows = []
    detrended_rows = []
    circular_rows = []

    for condition, spec in EEG_FILES.items():
        meta = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["subs_key"], "fs"])
        eeg_subjects = subject_list(meta[spec["subs_key"]])
        fs_eeg = float(np.ravel(meta["fs"])[0])
        eeg_mat = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["eeg_key"]])
        eeg = np.asarray(eeg_mat[spec["eeg_key"]], dtype=float)
        for subject_id in selected_subjects:
            eeg_trial_index = eeg_subjects.index(subject_id)
            cb_subject_index = cb_subjects.index(subject_id)
            eeg_channel = eeg[CHANNEL_INDEX, :, eeg_trial_index].copy()
            engagement = all_cb[:, cb_subject_index, spec["cb_stimulus_index"]].copy()
            if not np.isfinite(eeg_channel).all() or not np.isfinite(engagement).all():
                raise RuntimeError(f"NaN/Inf found for {subject_id} {condition}.")
            table, timing = extract_features_for_subject(eeg_channel, engagement, engagement_time, fs_eeg, fs_cb)
            table.insert(0, "condition", condition)
            table.insert(0, "subject_id", subject_id)
            window_tables.append(table.drop(columns=["n_engagement_samples"]))
            timing_rows.append({"subject_id": subject_id, "condition": condition, **timing})
            raw_rows.append(correlation_row(subject_id, condition, table, detrended=False))
            detrended_rows.append(correlation_row(subject_id, condition, table, detrended=True))
            engagement_values = table["engagement_mean"].to_numpy(dtype=float)
            circular_rows.append(
                circular_shift_result(
                    subject_id,
                    condition,
                    "upper_alpha",
                    table["upper_alpha_log_power"].to_numpy(dtype=float),
                    engagement_values,
                )
            )
            circular_rows.append(
                circular_shift_result(
                    subject_id,
                    condition,
                    "beta",
                    table["beta_log_power"].to_numpy(dtype=float),
                    engagement_values,
                )
            )
        del eeg, eeg_mat
        gc.collect()

    window_level = pd.concat(window_tables, ignore_index=True)
    timing_qc = pd.DataFrame(timing_rows)
    raw = pd.DataFrame(raw_rows)
    detrended = pd.DataFrame(detrended_rows)
    circular = pd.DataFrame(circular_rows)
    deltas = compute_deltas(raw, detrended)
    group_summary = describe_deltas(deltas)
    rec = recommendation(group_summary)

    window_level.to_csv(OUTPUT_DIR / "toy2_window_level_data.csv", index=False)
    timing_qc.to_csv(OUTPUT_DIR / "timing_qc.csv", index=False)
    raw.to_csv(OUTPUT_DIR / "toy2_correlations_raw.csv", index=False)
    detrended.to_csv(OUTPUT_DIR / "toy2_correlations_detrended.csv", index=False)
    circular.to_csv(OUTPUT_DIR / "toy2_circular_shift_results.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "toy2_original_control_deltas.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "toy2_group_summary.csv", index=False)

    paired_plot(raw, "upper_alpha", "01_upper_alpha_original_vs_control.png")
    paired_plot(raw, "beta", "02_beta_original_vs_control.png")
    delta_raw_plot(deltas)
    raw_vs_detrended_plot(deltas)

    raw_summary = group_summary[group_summary["delta_type"].eq("delta_rho_raw")].set_index("feature")
    det_summary = group_summary[group_summary["delta_type"].eq("delta_rho_detrended")].set_index("feature")
    summary = {
        "n_subjects": len(selected_subjects),
        "subject_ids": selected_subjects,
        "channel": CHANNEL,
        "channel_region": "frontal/anterior midline",
        "nearest_standard_electrode": "Fz (approximate)",
        "channel_coordinates": e11,
        "window_size_sec": WINDOW_SEC,
        "features": ["upper_alpha_10_12_hz_log_power", "beta_15_30_hz_log_power"],
        "n_positive_delta_raw": {feature: int(raw_summary.loc[feature, "n_positive"]) for feature in raw_summary.index},
        "median_delta_raw": {feature: float(raw_summary.loc[feature, "median"]) for feature in raw_summary.index},
        "n_positive_delta_detrended": {feature: int(det_summary.loc[feature, "n_positive"]) for feature in det_summary.index},
        "median_delta_detrended": {feature: float(det_summary.loc[feature, "median"]) for feature in det_summary.index},
        "recommendation": rec,
        "recommendation_rule": "STRONG GO if any feature has >=4/5 positive raw deltas and median raw delta >=0.10; MODERATE GO if any feature has >=3/5 positive raw deltas and positive median raw delta; otherwise RECONSIDER.",
        "synthetic_validation": validation,
        "interpretation_boundary": "Five-subject exploratory pilot; not a biomarker, population inference, causal, BCI, or real-time decoding result.",
    }
    (OUTPUT_DIR / "toy2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("NMED-E TOY2 COMPLETE")
    print()
    print("Subjects:", ", ".join(selected_subjects))
    print("Channel: E11 (~Fz)")
    print("Window: 5 s non-overlap")
    print()
    print("UPPER ALPHA")
    print("Subject | rho Original | rho Control | Delta raw | Delta detrended")
    alpha_rows = deltas[deltas["feature"].eq("upper_alpha")].set_index("subject_id").loc[selected_subjects]
    for subject_id, row in alpha_rows.iterrows():
        print(
            f"{subject_id} | {row.rho_original_raw:.4f} | {row.rho_control_raw:.4f} | "
            f"{row.delta_rho_raw:.4f} | {row.delta_rho_detrended:.4f}"
        )
    print()
    print("BETA")
    print("Subject | rho Original | rho Control | Delta raw | Delta detrended")
    beta_rows = deltas[deltas["feature"].eq("beta")].set_index("subject_id").loc[selected_subjects]
    for subject_id, row in beta_rows.iterrows():
        print(
            f"{subject_id} | {row.rho_original_raw:.4f} | {row.rho_control_raw:.4f} | "
            f"{row.delta_rho_raw:.4f} | {row.delta_rho_detrended:.4f}"
        )
    print()
    print("Summary:")
    print(f"Upper-alpha positive Delta: {int(raw_summary.loc['upper_alpha', 'n_positive'])}/5")
    print(f"Upper-alpha median Delta raw: {float(raw_summary.loc['upper_alpha', 'median']):.4f}")
    print(f"Beta positive Delta: {int(raw_summary.loc['beta', 'n_positive'])}/5")
    print(f"Beta median Delta raw: {float(raw_summary.loc['beta', 'median']):.4f}")
    print()
    print("Recommendation:")
    print(rec)
    return summary


if __name__ == "__main__":
    main()
