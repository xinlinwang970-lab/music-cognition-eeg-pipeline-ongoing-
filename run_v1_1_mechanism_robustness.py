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
from scipy.stats import spearmanr

from run_full_v1 import bootstrap_ci, exact_sign_flip, json_ready
from toy1_1_nmede_diagnostics import compute_band_power, compute_welch_psd, read_sfp
from toy2_nmede_pilot import (
    CB_FILE,
    CHANNEL,
    CHANNEL_INDEX,
    DATA_DIR,
    EEG_FILES,
    PROJECT_DIR,
    SFP_FILE,
    WINDOW_SEC,
    complete_windows,
    subject_list,
    synthetic_validation,
)


FULL_V1_DIR = PROJECT_DIR / "full_v1_outputs"
OUTPUT_DIR = PROJECT_DIR / "v1_1_outputs"
REPORT_DIR = PROJECT_DIR / "reports"
SEED = 20260817
TIMESCALE_WINDOWS = {"raw": None, "slow30": 6, "fast30": 6, "slow60": 12, "fast60": 12}
ROI_Y_MIN = 7.5
ROI_Z_MIN = 0.0


def effect_summary(values: np.ndarray, endpoint: str, family: str) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed, p, null = exact_sign_flip(values)
    mean_ci = bootstrap_ci(values, "mean", rng)
    median_ci = bootstrap_ci(values, "median", rng)
    return {
        "endpoint": endpoint,
        "family": family,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_positive": int(np.sum(values > 0)),
        "sign_flip_observed_mean": observed,
        "sign_flip_two_sided_p": p,
        "n_sign_combinations": int(len(null)),
        "bootstrap_mean_ci_low": mean_ci[0],
        "bootstrap_mean_ci_high": mean_ci[1],
        "bootstrap_median_ci_low": median_ci[0],
        "bootstrap_median_ci_high": median_ci[1],
    }


def centered_slow(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=window).mean()


def timescale_components(df: pd.DataFrame, value_col: str) -> dict[str, pd.Series]:
    raw = df[value_col].astype(float)
    slow30 = centered_slow(raw, 6)
    slow60 = centered_slow(raw, 12)
    return {
        "raw": raw,
        "slow30": slow30,
        "fast30": raw - slow30,
        "slow60": slow60,
        "fast60": raw - slow60,
    }


def corr_pair(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 5:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def compute_timescale_correlations(window_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (subject_id, condition), df in window_level.groupby(["subject_id", "condition"], sort=True):
        df = df.sort_values("window_id").reset_index(drop=True)
        beta = timescale_components(df, "beta_log_power")
        engagement = timescale_components(df, "engagement_mean")
        for representation in ["raw", "slow30", "fast30", "slow60", "fast60"]:
            rows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "representation": representation,
                    "rho": corr_pair(beta[representation], engagement[representation]),
                    "n_valid_windows": int((beta[representation].notna() & engagement[representation].notna()).sum()),
                }
            )
    corr = pd.DataFrame(rows)
    wide = corr.pivot(index=["subject_id", "representation"], columns="condition", values="rho").reset_index()
    wide["delta_rho"] = wide["original"] - wide["control"]
    return corr, wide


def group_summary_from_delta_table(delta_table: pd.DataFrame, value_col: str, family: str) -> pd.DataFrame:
    rows = []
    for key, df in delta_table.groupby("representation" if "representation" in delta_table.columns else "endpoint", sort=False):
        rows.append(effect_summary(df[value_col].to_numpy(dtype=float), str(key), family))
    return pd.DataFrame(rows)


def find_audio_files() -> dict[str, Path]:
    candidates = {}
    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aif", ".aiff"}
    for path in DATA_DIR.rglob("*"):
        if path.suffix.lower() not in audio_exts:
            continue
        name = path.name.lower()
        if "22" in name or "original" in name or "intact" in name:
            candidates["original"] = path
        elif "23" in name or "control" in name:
            candidates["control"] = path
    return candidates


def write_audio_unavailable(subjects: list[str], raw_delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    note = (
        "Actual Original and Control stimulus audio files are not present locally. "
        "The NMED-E README states that users should contact the author for stimulus audio access. "
        "Audio envelope control was therefore not computed; no substitute audio was used."
    )
    envelope = pd.DataFrame(
        [
            {"condition": "original", "audio_available": False, "reason": note},
            {"condition": "control", "audio_available": False, "reason": note},
        ]
    )
    results = pd.DataFrame(
        {
            "subject_id": subjects,
            "partial_rho_original": np.nan,
            "partial_rho_control": np.nan,
            "delta_partial_rho": np.nan,
            "audio_control_status": "not_computed_audio_unavailable",
            "reason": note,
        }
    )
    envelope.to_csv(OUTPUT_DIR / "audio_envelope_windows.csv", index=False)
    results.to_csv(OUTPUT_DIR / "audio_control_results.csv", index=False)
    (OUTPUT_DIR / "audio_control_unavailable.txt").write_text(note, encoding="utf-8")
    return envelope, results


def define_frontal_roi() -> pd.DataFrame:
    sfp = read_sfp(SFP_FILE)
    sensors = sfp[sfp["label"].str.match(r"^E\d+$", na=False)].copy()
    roi = sensors[(sensors["y"] >= ROI_Y_MIN) & (sensors["z"] >= ROI_Z_MIN)].copy()
    roi = roi.sort_values(["y", "x"], ascending=[False, True]).reset_index(drop=True)
    if roi.empty:
        raise RuntimeError("Frontal ROI definition selected no channels.")
    roi["channel_index_zero_based"] = roi["label"].str.replace("E", "", regex=False).astype(int) - 1
    roi["approximate_region"] = "frontal/anterior"
    roi["reason_for_inclusion"] = f"EGI numeric sensor with y >= {ROI_Y_MIN} and z >= {ROI_Z_MIN}; defined before ROI results."
    roi.to_csv(OUTPUT_DIR / "frontal_roi_definition.csv", index=False)
    return roi


def window_beta_for_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    n_windows = int(np.floor(min(len(signal) / fs, 479.15) / WINDOW_SEC))
    windows = complete_windows(n_windows * WINDOW_SEC)
    samples = int(round(WINDOW_SEC * fs))
    values = []
    for row in windows.itertuples(index=False):
        start = int(round(row.start_sec * fs))
        segment = signal[start : start + samples]
        freqs, pxx = compute_welch_psd(segment, fs)
        beta = compute_band_power(freqs, pxx, 15, 30)
        values.append(float(np.log10(beta + 1e-20)))
    return np.asarray(values, dtype=float)


def compute_roi_results(subjects: list[str], window_level: pd.DataFrame, roi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cb = scipy_io.loadmat(CB_FILE, squeeze_me=True)
    cb_subjects = subject_list(cb["subIDs"])
    all_cb = np.asarray(cb["allCB"], dtype=float)
    fs_cb = float(np.ravel(cb["fs"])[0])
    xax = np.asarray(cb["xax"], dtype=float).ravel()
    engagement_time = xax if abs(float(np.nanmax(xax)) - all_cb.shape[0] / fs_cb) < 5 else np.arange(all_cb.shape[0]) / fs_cb
    roi_indices = roi["channel_index_zero_based"].to_numpy(dtype=int)

    rows = []
    for condition, spec in EEG_FILES.items():
        meta = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["subs_key"], "fs"])
        eeg_subjects = subject_list(meta[spec["subs_key"]])
        fs_eeg = float(np.ravel(meta["fs"])[0])
        eeg_mat = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["eeg_key"]])
        eeg = np.asarray(eeg_mat[spec["eeg_key"]], dtype=float)
        for subject_id in subjects:
            eeg_trial_index = eeg_subjects.index(subject_id)
            channel_features = []
            for idx in roi_indices:
                channel_features.append(window_beta_for_signal(eeg[idx, :, eeg_trial_index], fs_eeg))
            roi_beta = np.median(np.vstack(channel_features), axis=0)
            cb_subject_index = cb_subjects.index(subject_id)
            engagement = all_cb[:, cb_subject_index, spec["cb_stimulus_index"]]
            windows = complete_windows(len(roi_beta) * WINDOW_SEC)
            engagement_means = []
            for row in windows.itertuples(index=False):
                mask = (engagement_time >= row.start_sec) & (engagement_time < row.end_sec)
                engagement_means.append(float(np.mean(engagement[mask])))
            df = pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "window_id": np.arange(len(roi_beta), dtype=int),
                    "center_sec": windows["center_sec"],
                    "roi_beta_log_power": roi_beta,
                    "engagement_mean": engagement_means,
                }
            )
            beta = timescale_components(df.rename(columns={"roi_beta_log_power": "beta_log_power"}), "beta_log_power")
            engagement_comp = timescale_components(df, "engagement_mean")
            for representation in ["raw", "slow30", "fast30", "slow60", "fast60"]:
                rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": condition,
                        "representation": representation,
                        "rho_roi_beta": corr_pair(beta[representation], engagement_comp[representation]),
                    }
                )
        del eeg, eeg_mat
        gc.collect()

    roi_corr = pd.DataFrame(rows)
    wide = roi_corr.pivot(index=["subject_id", "representation"], columns="condition", values="rho_roi_beta").reset_index()
    wide["delta_roi_beta"] = wide["original"] - wide["control"]
    roi_corr.to_csv(OUTPUT_DIR / "frontal_roi_condition_correlations.csv", index=False)
    wide.to_csv(OUTPUT_DIR / "frontal_roi_results.csv", index=False)
    return roi_corr, wide


def compute_group_engagement(window_level: pd.DataFrame, subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (condition, window_id), dfw in window_level.groupby(["condition", "window_id"], sort=True):
        means = dfw.set_index("subject_id")["engagement_mean"]
        for subject_id in subjects:
            others = means.drop(index=subject_id)
            rows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "window_id": int(window_id),
                    "group_minus_subject_engagement": float(others.mean()),
                }
            )
    group_eng = pd.DataFrame(rows)
    merged = window_level.merge(group_eng, on=["subject_id", "condition", "window_id"], how="inner")
    corr_rows = []
    for (subject_id, condition), df in merged.groupby(["subject_id", "condition"], sort=True):
        corr_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "rho_group_engagement": float(
                    spearmanr(df["beta_log_power"], df["group_minus_subject_engagement"]).statistic
                ),
            }
        )
    corr = pd.DataFrame(corr_rows)
    wide = corr.pivot(index="subject_id", columns="condition", values="rho_group_engagement").reset_index()
    wide["delta_group_rho"] = wide["original"] - wide["control"]
    group_eng.to_csv(OUTPUT_DIR / "leave_one_subject_out_group_engagement_windows.csv", index=False)
    wide.to_csv(OUTPUT_DIR / "group_engagement_results.csv", index=False)
    return corr, wide


def compute_individual_differences(subjects: list[str], deltas: pd.DataFrame) -> pd.DataFrame:
    ratings_file = DATA_DIR / "CleanRatings_All.mat"
    if not ratings_file.exists():
        return pd.DataFrame({"status": ["not_computed_ratings_file_missing"]})
    mat = scipy_io.loadmat(ratings_file, squeeze_me=True)
    rating_subjects = subject_list(mat["subIDs"])
    ratings22 = np.asarray(mat["allRatings22"], dtype=float)
    questions = ["pleasant", "arousing", "interesting", "predictable", "familiar", "genre_listening_frequency"]
    rows = []
    delta_series = deltas.set_index("subject_id")["delta_beta_raw"]
    for col, question in enumerate(questions):
        values = []
        delta_values = []
        for subject_id in subjects:
            if subject_id in rating_subjects:
                values.append(float(ratings22[rating_subjects.index(subject_id), col]))
                delta_values.append(float(delta_series.loc[subject_id]))
        rho = float(spearmanr(values, delta_values).statistic) if len(values) >= 5 else np.nan
        rows.append(
            {
                "metadata_variable": question,
                "condition": "original",
                "n": len(values),
                "spearman_rho_with_delta_beta_raw": rho,
                "status": "exploratory_hypothesis_generating",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "individual_differences.csv", index=False)
    return out


def plot_timescale(summary: pd.DataFrame) -> None:
    order = ["raw", "slow30", "fast30", "slow60", "fast60"]
    df = summary.set_index("endpoint").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["endpoint"], df["mean"], color="#fb9a99", edgecolor="#333333")
    ax.errorbar(
        df["endpoint"],
        df["mean"],
        yerr=[df["mean"] - df["bootstrap_mean_ci_low"], df["bootstrap_mean_ci_high"] - df["mean"]],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Original-Control delta rho")
    ax.set_title("Beta timescale decomposition")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_beta_timescale_delta.png", dpi=160)
    plt.close(fig)


def plot_audio_control(raw_delta: pd.Series, audio_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.zeros(len(raw_delta)), raw_delta, label="raw delta beta-engagement", alpha=0.8)
    if audio_results["delta_partial_rho"].notna().any():
        ax.scatter(np.ones(len(audio_results)), audio_results["delta_partial_rho"], label="partial delta controlling envelope")
    else:
        ax.text(
            0.5,
            0.5,
            "Audio stimulus files unavailable;\npartial envelope control not computed.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1], ["Raw", "Envelope controlled"])
    ax.set_ylabel("Original-Control delta rho")
    ax.set_title("Audio envelope control")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_audio_envelope_control.png", dpi=160)
    plt.close(fig)


def plot_e11_vs_roi(subject_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(subject_summary["delta_raw"], subject_summary["delta_roi_raw"], s=45)
    for row in subject_summary.itertuples(index=False):
        ax.text(row.delta_raw + 0.006, row.delta_roi_raw + 0.006, row.subject_id, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("E11 delta beta raw")
    ax.set_ylabel("Frontal ROI delta beta raw")
    ax.set_title("E11 vs predefined frontal ROI")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_e11_vs_frontal_roi.png", dpi=160)
    plt.close(fig)


def plot_personal_vs_group(subject_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(subject_summary["delta_raw"], subject_summary["delta_group_rho"], s=45, color="#33a02c")
    for row in subject_summary.itertuples(index=False):
        ax.text(row.delta_raw + 0.006, row.delta_group_rho + 0.006, row.subject_id, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Own-engagement delta beta raw")
    ax.set_ylabel("Group-minus-subject engagement delta")
    ax.set_title("Personal vs leave-one-subject-out group engagement")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_personal_vs_group_engagement.png", dpi=160)
    plt.close(fig)


def plot_heatmap(subject_summary: pd.DataFrame) -> None:
    cols = [
        "delta_raw",
        "delta_slow30",
        "delta_fast30",
        "delta_slow60",
        "delta_fast60",
        "delta_partial_rho",
        "delta_roi_raw",
        "delta_group_rho",
    ]
    matrix = subject_summary.set_index("subject_id")[cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax.set_xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Subject-level Original-Control delta rho")
    fig.colorbar(im, ax=ax, label="delta rho")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_subject_effect_heatmap.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, object]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    text = f"""# V1.1 Mechanism and Robustness

## 1. Motivation From Full V1
Full V1 found a modest observed-sample beta raw Original-Control effect at E11. V1.1 asks what that pattern means, without changing the primary feature, channel, window size, PSD method, or frequency band.

## 2. Timescale Decomposition
The beta-engagement association was decomposed into raw, 30 s slow/fast, and 60 s slow/fast components using centered rolling means on the 5 s window trajectories. The strongest group mean delta among these was `{summary['timescale_strongest_endpoint']}`.

## 3. Acoustic Envelope Control
Audio envelope control was not computed because the actual Original and Control stimulus audio files were not present locally. The NMED-E README states that stimulus audio access requires contacting the author. No substitute audio was used.

## 4. Frontal ROI Robustness
A frontal/anterior ROI was defined before inspecting ROI results using montage coordinates: EGI sensors with y >= {ROI_Y_MIN} and z >= {ROI_Z_MIN}. ROI beta was aggregated by median across channels.

## 5. Group Engagement Analysis
Leave-one-subject-out group engagement was computed separately for each condition and window. This tests whether a participant's E11 beta follows moments that other trained listeners collectively rated as engaging.

## 6. Individual Variability
Retrospective original-stimulus ratings were inspected only as exploratory hypothesis-generating correlates of `delta_beta_raw`.

## 7. Limitations
- Engagement is not flow.
- EEG and engagement were collected in separate sessions.
- Acoustic confounding remains possible because stimulus audio was unavailable.
- The analysis remains exploratory and mechanism-oriented.
- No causal claim is supported.
- No BCI or real-time decoding claim is supported.

## 8. Decision for Next Stage
{summary['next_stage_recommendation']}
"""
    (REPORT_DIR / "v1_1_mechanism_robustness.md").write_text(text, encoding="utf-8")


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    validation = synthetic_validation()
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    window_level = pd.read_csv(FULL_V1_DIR / "window_level_data.csv")
    full_deltas = pd.read_csv(FULL_V1_DIR / "original_control_deltas.csv")
    full_summary = json.loads((FULL_V1_DIR / "full_v1_summary.json").read_text(encoding="utf-8"))
    subjects = full_summary["included_subject_ids"]

    # A. Timescale decomposition.
    timescale_corr, timescale_delta = compute_timescale_correlations(window_level)
    timescale_corr.to_csv(OUTPUT_DIR / "timescale_condition_correlations.csv", index=False)
    timescale_delta.to_csv(OUTPUT_DIR / "timescale_correlations.csv", index=False)
    timescale_summary = group_summary_from_delta_table(timescale_delta, "delta_rho", "timescale")
    timescale_summary.to_csv(OUTPUT_DIR / "timescale_group_summary.csv", index=False)

    raw_delta = timescale_delta[timescale_delta["representation"].eq("raw")].set_index("subject_id")["delta_rho"]
    full_raw = full_deltas.set_index("subject_id")["delta_beta_raw"]
    max_repro_diff = float(np.max(np.abs(raw_delta.loc[full_raw.index] - full_raw)))
    reproduction = {
        "full_v1_beta_raw_reproduced": bool(max_repro_diff < 1e-12),
        "max_abs_delta_beta_raw_difference": max_repro_diff,
    }
    (OUTPUT_DIR / "full_v1_reproduction_check.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")
    if not reproduction["full_v1_beta_raw_reproduced"]:
        raise RuntimeError("Full V1 raw beta deltas did not reproduce exactly.")

    # B. Audio envelope control.
    audio_files = find_audio_files()
    if {"original", "control"}.issubset(audio_files):
        raise NotImplementedError("Audio files were found, but this script currently expects unavailable NMED-E stimuli.")
    _, audio_results = write_audio_unavailable(subjects, full_deltas)
    audio_summary = pd.DataFrame([effect_summary(full_deltas["delta_beta_raw"].to_numpy(dtype=float), "raw_beta_reference", "audio_control")])
    audio_summary.to_csv(OUTPUT_DIR / "audio_control_group_summary.csv", index=False)

    # C. Frontal ROI robustness.
    roi = define_frontal_roi()
    _, roi_delta = compute_roi_results(subjects, window_level, roi)
    roi_raw = roi_delta[roi_delta["representation"].eq("raw")].rename(columns={"delta_roi_beta": "delta_roi_raw"})
    roi_summary = group_summary_from_delta_table(
        roi_delta.rename(columns={"delta_roi_beta": "delta_rho"}),
        "delta_rho",
        "frontal_roi",
    )
    roi_summary.to_csv(OUTPUT_DIR / "frontal_roi_group_summary.csv", index=False)

    # D. Group engagement.
    _, group_delta = compute_group_engagement(window_level, subjects)
    group_summary = pd.DataFrame([effect_summary(group_delta["delta_group_rho"].to_numpy(dtype=float), "group_engagement", "group_engagement")])
    group_summary.to_csv(OUTPUT_DIR / "group_engagement_summary.csv", index=False)

    # Optional E. Individual differences from available retrospective ratings.
    individual = compute_individual_differences(subjects, full_deltas)

    subject_summary = pd.DataFrame({"subject_id": subjects})
    for representation, column in [
        ("raw", "delta_raw"),
        ("slow30", "delta_slow30"),
        ("fast30", "delta_fast30"),
        ("slow60", "delta_slow60"),
        ("fast60", "delta_fast60"),
    ]:
        vals = timescale_delta[timescale_delta["representation"].eq(representation)].set_index("subject_id")["delta_rho"]
        subject_summary[column] = subject_summary["subject_id"].map(vals)
    partial = audio_results.set_index("subject_id")["delta_partial_rho"]
    subject_summary["delta_partial_rho"] = subject_summary["subject_id"].map(partial)
    roi_vals = roi_raw.set_index("subject_id")["delta_roi_raw"]
    subject_summary["delta_roi_raw"] = subject_summary["subject_id"].map(roi_vals)
    group_vals = group_delta.set_index("subject_id")["delta_group_rho"]
    subject_summary["delta_group_rho"] = subject_summary["subject_id"].map(group_vals)
    subject_summary["training_years"] = np.nan
    subject_summary.to_csv(OUTPUT_DIR / "v1_1_subject_summary.csv", index=False)

    plot_timescale(timescale_summary)
    plot_audio_control(full_deltas["delta_beta_raw"], audio_results)
    plot_e11_vs_roi(subject_summary)
    plot_personal_vs_group(subject_summary)
    plot_heatmap(subject_summary)

    all_group_summary = pd.concat(
        [
            timescale_summary,
            audio_summary,
            roi_summary.assign(endpoint=lambda d: "roi_" + d["endpoint"].astype(str)),
            group_summary,
        ],
        ignore_index=True,
    )
    all_group_summary.to_csv(OUTPUT_DIR / "v1_1_group_summary.csv", index=False)

    strongest = timescale_summary.sort_values("mean", ascending=False).iloc[0]
    raw = timescale_summary[timescale_summary["endpoint"].eq("raw")].iloc[0]
    slow_best = timescale_summary[timescale_summary["endpoint"].isin(["slow30", "slow60"])].sort_values("mean", ascending=False).iloc[0]
    fast_best = timescale_summary[timescale_summary["endpoint"].isin(["fast30", "fast60"])].sort_values("mean", ascending=False).iloc[0]
    roi_raw_summary = roi_summary[roi_summary["endpoint"].eq("raw")].iloc[0]
    group_row = group_summary.iloc[0]

    if slow_best["mean"] > 0 and fast_best["mean"] <= raw["mean"] * 0.5:
        next_stage = "Slow effects dominate; next stage should emphasize music-structure and engagement-arc annotation."
    elif fast_best["mean"] > 0:
        next_stage = "Fast component remains positive; next stage can examine local time-resolved structure without changing bands/channels."
    elif roi_raw_summary["mean"] > 0:
        next_stage = "ROI preserves a positive direction; spatial robustness is plausible but timescale interpretation remains central."
    else:
        next_stage = "Robustness analyses weaken the beta pattern; treat beta as a weak exploratory result."

    summary = {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "primary_feature": "E11 beta log power 15-30 Hz",
        "full_v1_reproduction": reproduction,
        "timescale_strongest_endpoint": str(strongest["endpoint"]),
        "timescale_raw_mean_delta": float(raw["mean"]),
        "timescale_slow_best_endpoint": str(slow_best["endpoint"]),
        "timescale_slow_best_mean_delta": float(slow_best["mean"]),
        "timescale_fast_best_endpoint": str(fast_best["endpoint"]),
        "timescale_fast_best_mean_delta": float(fast_best["mean"]),
        "audio_control_status": "not_computed_audio_unavailable",
        "frontal_roi_channels": roi["label"].tolist(),
        "frontal_roi_raw_mean_delta": float(roi_raw_summary["mean"]),
        "group_engagement_mean_delta": float(group_row["mean"]),
        "individual_differences_status": "computed_from_retrospective_ratings" if "metadata_variable" in individual.columns else "not_computed",
        "next_stage_recommendation": next_stage,
        "interpretation_boundary": "Mechanism/robustness analyses only; no flow, biomarker, causal, BCI, or real-time decoding claim.",
    }
    (OUTPUT_DIR / "v1_1_summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    write_report(summary)

    print("NMED-E V1.1 MECHANISM/ROBUSTNESS COMPLETE")
    print()
    print("Full V1 reproduction max abs diff:", max_repro_diff)
    print("Timescale strongest endpoint:", summary["timescale_strongest_endpoint"])
    print("Raw mean delta:", f"{summary['timescale_raw_mean_delta']:.4f}")
    print("Best slow mean delta:", summary["timescale_slow_best_endpoint"], f"{summary['timescale_slow_best_mean_delta']:.4f}")
    print("Best fast mean delta:", summary["timescale_fast_best_endpoint"], f"{summary['timescale_fast_best_mean_delta']:.4f}")
    print("Audio control:", summary["audio_control_status"])
    print("ROI channels:", ", ".join(summary["frontal_roi_channels"]))
    print("ROI raw mean delta:", f"{summary['frontal_roi_raw_mean_delta']:.4f}")
    print("Group engagement mean delta:", f"{summary['group_engagement_mean_delta']:.4f}")
    print("Next-stage recommendation:", next_stage)
    return summary


if __name__ == "__main__":
    main()
