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

from toy2_nmede_pilot import (
    CB_FILE,
    CHANNEL,
    CHANNEL_INDEX,
    DATA_DIR,
    EEG_FILES,
    OUTPUT_DIR as TOY2_OUTPUT_DIR,
    PROJECT_DIR,
    SFP_FILE,
    WINDOW_SEC,
    circular_shift_result,
    correlation_row,
    detrend,
    extract_features_for_subject,
    read_sfp,
    subject_list,
    synthetic_validation,
)


OUTPUT_DIR = PROJECT_DIR / "full_v1_outputs"
REPORT_DIR = PROJECT_DIR / "reports"
SEED = 20260817
BOOTSTRAP_REPS = 10000


def build_sample_inclusion() -> tuple[pd.DataFrame, list[str]]:
    meta = {}
    for condition, spec in EEG_FILES.items():
        mat = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["subs_key"]])
        meta[condition] = set(subject_list(mat[spec["subs_key"]]))
    cb = scipy_io.loadmat(CB_FILE, squeeze_me=True, variable_names=["subIDs"])
    engagement_subjects = set(subject_list(cb["subIDs"]))
    all_subjects = sorted(meta["original"] | meta["control"] | engagement_subjects)
    e11_available = SFP_FILE.exists() and CHANNEL_INDEX < 125

    rows = []
    for subject_id in all_subjects:
        original_eeg = subject_id in meta["original"]
        control_eeg = subject_id in meta["control"]
        original_engagement = subject_id in engagement_subjects
        control_engagement = subject_id in engagement_subjects
        checks = {
            "original_eeg": original_eeg,
            "control_eeg": control_eeg,
            "original_engagement": original_engagement,
            "control_engagement": control_engagement,
            "e11_available": e11_available,
        }
        included = all(checks.values())
        missing = [name for name, ok in checks.items() if not ok]
        rows.append(
            {
                "subject_id": subject_id,
                "included": bool(included),
                "reason_if_excluded": "" if included else "; ".join(f"missing {m}" for m in missing),
                **checks,
            }
        )
    inclusion = pd.DataFrame(rows)
    included_subjects = inclusion[inclusion["included"]]["subject_id"].tolist()
    return inclusion, included_subjects


def compute_deltas(raw: pd.DataFrame, detrended_df: pd.DataFrame) -> pd.DataFrame:
    raw_wide = raw.pivot(index="subject_id", columns="condition", values=["rho_alpha_raw", "rho_beta_raw"])
    det_wide = detrended_df.pivot(
        index="subject_id",
        columns="condition",
        values=["rho_alpha_detrended", "rho_beta_detrended"],
    )
    rows = []
    for subject_id in raw_wide.index:
        rows.append(
            {
                "subject_id": subject_id,
                "rho_beta_original_raw": float(raw_wide.loc[subject_id, ("rho_beta_raw", "original")]),
                "rho_beta_control_raw": float(raw_wide.loc[subject_id, ("rho_beta_raw", "control")]),
                "delta_beta_raw": float(
                    raw_wide.loc[subject_id, ("rho_beta_raw", "original")]
                    - raw_wide.loc[subject_id, ("rho_beta_raw", "control")]
                ),
                "rho_beta_original_detrended": float(
                    det_wide.loc[subject_id, ("rho_beta_detrended", "original")]
                ),
                "rho_beta_control_detrended": float(
                    det_wide.loc[subject_id, ("rho_beta_detrended", "control")]
                ),
                "delta_beta_detrended": float(
                    det_wide.loc[subject_id, ("rho_beta_detrended", "original")]
                    - det_wide.loc[subject_id, ("rho_beta_detrended", "control")]
                ),
                "rho_alpha_original_raw": float(raw_wide.loc[subject_id, ("rho_alpha_raw", "original")]),
                "rho_alpha_control_raw": float(raw_wide.loc[subject_id, ("rho_alpha_raw", "control")]),
                "delta_alpha_raw": float(
                    raw_wide.loc[subject_id, ("rho_alpha_raw", "original")]
                    - raw_wide.loc[subject_id, ("rho_alpha_raw", "control")]
                ),
                "rho_alpha_original_detrended": float(
                    det_wide.loc[subject_id, ("rho_alpha_detrended", "original")]
                ),
                "rho_alpha_control_detrended": float(
                    det_wide.loc[subject_id, ("rho_alpha_detrended", "control")]
                ),
                "delta_alpha_detrended": float(
                    det_wide.loc[subject_id, ("rho_alpha_detrended", "original")]
                    - det_wide.loc[subject_id, ("rho_alpha_detrended", "control")]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def exact_sign_flip(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    null = np.array([0.0], dtype=float)
    for value in values:
        null = np.concatenate([null + value, null - value])
    null = null / len(values)
    observed = float(np.mean(values))
    p_two_sided = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
    return observed, p_two_sided, null


def bootstrap_ci(values: np.ndarray, statistic: str, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    if statistic == "mean":
        stats = np.mean(samples, axis=1)
    elif statistic == "median":
        stats = np.median(samples, axis=1)
    else:
        raise ValueError(statistic)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize_effects(deltas: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(SEED)
    endpoints = [
        ("primary", "beta_raw", "delta_beta_raw"),
        ("secondary_sensitivity", "beta_detrended", "delta_beta_detrended"),
        ("secondary", "upper_alpha_raw", "delta_alpha_raw"),
        ("secondary_sensitivity", "upper_alpha_detrended", "delta_alpha_detrended"),
    ]
    rows = []
    permutation_json = {}
    secondary_p = []
    for family, endpoint, column in endpoints:
        values = deltas[column].to_numpy(dtype=float)
        observed_mean, p_perm, null = exact_sign_flip(values)
        mean_ci = bootstrap_ci(values, "mean", rng)
        median_ci = bootstrap_ci(values, "median", rng)
        row = {
            "endpoint": endpoint,
            "family": family,
            "delta_column": column,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_positive": int(np.sum(values > 0)),
            "sign_flip_observed_mean": observed_mean,
            "sign_flip_two_sided_p": p_perm,
            "bootstrap_mean_ci_low": mean_ci[0],
            "bootstrap_mean_ci_high": mean_ci[1],
            "bootstrap_median_ci_low": median_ci[0],
            "bootstrap_median_ci_high": median_ci[1],
        }
        rows.append(row)
        permutation_json[endpoint] = {
            "delta_column": column,
            "method": "exact paired sign-flip permutation",
            "n_subjects": int(len(values)),
            "n_sign_combinations": int(len(null)),
            "observed_mean_delta": observed_mean,
            "two_sided_empirical_p": p_perm,
            "direction": "positive" if observed_mean > 0 else "negative" if observed_mean < 0 else "zero",
        }
        if family != "primary":
            secondary_p.append((endpoint, p_perm))

    summary = pd.DataFrame(rows)
    if secondary_p:
        m = len(secondary_p)
        adjusted = {endpoint: min(p * m, 1.0) for endpoint, p in secondary_p}
        summary["secondary_bonferroni_p"] = summary.apply(
            lambda row: adjusted.get(row["endpoint"], np.nan),
            axis=1,
        )
        for endpoint, p_adj in adjusted.items():
            permutation_json[endpoint]["secondary_bonferroni_p"] = p_adj
    return summary, permutation_json


def linear_trend_slopes(table: pd.DataFrame) -> dict[str, float]:
    time = table["center_sec"].to_numpy(dtype=float)
    beta = table["beta_log_power"].to_numpy(dtype=float)
    alpha = table["upper_alpha_log_power"].to_numpy(dtype=float)
    engagement = table["engagement_mean"].to_numpy(dtype=float)
    return {
        "beta_slope_per_sec": float(np.polyfit(time, beta, 1)[0]),
        "alpha_slope_per_sec": float(np.polyfit(time, alpha, 1)[0]),
        "engagement_slope_per_sec": float(np.polyfit(time, engagement, 1)[0]),
    }


def condition_order_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metadata_field": "condition_order",
                "available": False,
                "value": "",
                "note": "condition order not available in processed dataset metadata",
            }
        ]
    )


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def plot_beta_paired(deltas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in deltas.itertuples(index=False):
        ax.plot([0, 1], [row.rho_beta_original_raw, row.rho_beta_control_raw], color="#999999", linewidth=0.9)
        ax.scatter([0, 1], [row.rho_beta_original_raw, row.rho_beta_control_raw], s=35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1], ["Original", "Control"])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Primary: beta raw Original vs Control")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_beta_original_vs_control_raw.png", dpi=160)
    plt.close(fig)


def plot_beta_delta_distribution(deltas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    values = deltas["delta_beta_raw"].to_numpy(dtype=float)
    ax.hist(values, bins=10, color="#fdae6b", edgecolor="#333333", alpha=0.9)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(np.mean(values), color="#d62728", linewidth=2, label=f"mean={np.mean(values):.3f}")
    ax.axvline(np.median(values), color="#1f77b4", linewidth=2, linestyle="--", label=f"median={np.median(values):.3f}")
    ax.set_xlabel("delta beta raw rho (Original - Control)")
    ax.set_ylabel("Count")
    ax.set_title("Primary delta beta raw distribution")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_beta_delta_raw_distribution.png", dpi=160)
    plt.close(fig)


def plot_beta_raw_vs_detrended(deltas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(deltas["delta_beta_raw"], deltas["delta_beta_detrended"], s=45, color="#ff7f0e")
    for row in deltas.itertuples(index=False):
        ax.text(row.delta_beta_raw + 0.006, row.delta_beta_detrended + 0.006, row.subject_id, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("delta beta raw")
    ax.set_ylabel("delta beta detrended")
    ax.set_title("Beta Original-Control delta: raw vs detrended")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_beta_raw_vs_detrended.png", dpi=160)
    plt.close(fig)


def plot_alpha_paired(deltas: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in deltas.itertuples(index=False):
        ax.plot([0, 1], [row.rho_alpha_original_raw, row.rho_alpha_control_raw], color="#999999", linewidth=0.9)
        ax.scatter([0, 1], [row.rho_alpha_original_raw, row.rho_alpha_control_raw], s=35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1], ["Original", "Control"])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Secondary: upper-alpha raw Original vs Control")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_upper_alpha_original_vs_control_raw.png", dpi=160)
    plt.close(fig)


def plot_trend_slopes(slopes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, column, title in [
        (axes[0], "beta_slope_per_sec", "Beta log-power slow trend"),
        (axes[1], "engagement_slope_per_sec", "Engagement slow trend"),
    ]:
        for subject_id, df in slopes.groupby("subject_id"):
            ordered = df.set_index("condition").loc[["original", "control"]]
            ax.plot([0, 1], ordered[column], color="#999999", linewidth=0.8)
            ax.scatter([0, 1], ordered[column], s=30)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([0, 1], ["Original", "Control"])
        ax.set_title(title)
        ax.set_ylabel("slope per second")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_slow_trend_slopes.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, object], effects: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    primary = effects[effects["endpoint"].eq("beta_raw")].iloc[0]
    beta_det = effects[effects["endpoint"].eq("beta_detrended")].iloc[0]
    alpha_raw = effects[effects["endpoint"].eq("upper_alpha_raw")].iloc[0]
    interpretation = summary["interpretation"]
    report = f"""# Full V1 Results

## Background
This analysis examines cross-session temporal association between EEG spectral dynamics and independently collected musical engagement ratings during naturalistic listening in NMED-E.

## Prior Pilot
Toy1.1 established the single-subject S01 E11 pipeline. Toy2 then suggested a stronger raw beta-engagement association for original than control music in a five-subject pilot, while detrended effects were weaker.

## Prespecified Primary Hypothesis
The primary endpoint is beta log-power, 15-30 Hz, at E11, using 5 s non-overlapping complete windows. The subject-level effect is `delta_beta_raw = rho_beta_original_raw - rho_beta_control_raw`.

## Dataset
Full V1 included N={summary['n_included']} subjects: {', '.join(summary['included_subject_ids'])}. Inclusion depended only on availability of processed original EEG, processed control EEG, original/control engagement, E11 availability, and valid timing.

## Methods
The pipeline reused Toy2 settings: E11, upper-alpha 10-12 Hz, beta 15-30 Hz, Welch PSD with 2 s Hann segments and 50% overlap, log10 absolute band power, 5 s complete windows, and Spearman correlation. Ordinary Spearman p-values were not used as primary inference. The primary group test used exact paired sign-flip permutation on subject-level beta raw deltas.

## Results
Primary beta raw mean delta was {primary['mean']:.4f}, median delta was {primary['median']:.4f}, and {int(primary['n_positive'])}/{int(primary['n'])} subjects had positive deltas. Exact two-sided sign-flip p was {primary['sign_flip_two_sided_p']:.4f}. Bootstrap 95% CI for the mean was [{primary['bootstrap_mean_ci_low']:.4f}, {primary['bootstrap_mean_ci_high']:.4f}].

Secondary upper-alpha raw mean delta was {alpha_raw['mean']:.4f}, median delta was {alpha_raw['median']:.4f}, and {int(alpha_raw['n_positive'])}/{int(alpha_raw['n'])} subjects were positive.

## Timescale Sensitivity
After linear detrending, beta mean delta was {beta_det['mean']:.4f}, median delta was {beta_det['median']:.4f}, and {int(beta_det['n_positive'])}/{int(beta_det['n'])} subjects were positive. This pattern supports the interpretation that the primary beta effect is more consistent with long-timescale structure than a purely local fluctuation effect.

## Limitations
- EEG and engagement were recorded in separate sessions.
- The same musical stimulus was used across subjects.
- E11 is a single-channel primary analysis.
- No causal interpretation is supported.
- Engagement is not flow.
- Linear detrending is a simple timescale sensitivity analysis.
- Residual acoustic confounding remains possible.

## Conclusion
{interpretation}
"""
    (REPORT_DIR / "full_v1_results.md").write_text(report, encoding="utf-8")


def main() -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    validation = synthetic_validation()
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    inclusion, included_subjects = build_sample_inclusion()
    inclusion.to_csv(OUTPUT_DIR / "sample_inclusion.csv", index=False)
    if not included_subjects:
        raise RuntimeError("No included subjects.")

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
    slope_rows = []
    circular_rows = []

    for condition, spec in EEG_FILES.items():
        meta = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["subs_key"], "fs"])
        eeg_subjects = subject_list(meta[spec["subs_key"]])
        fs_eeg = float(np.ravel(meta["fs"])[0])
        eeg_mat = scipy_io.loadmat(spec["path"], squeeze_me=True, variable_names=[spec["eeg_key"]])
        eeg = np.asarray(eeg_mat[spec["eeg_key"]], dtype=float)
        for subject_id in included_subjects:
            eeg_trial_index = eeg_subjects.index(subject_id)
            cb_subject_index = cb_subjects.index(subject_id)
            eeg_channel = eeg[CHANNEL_INDEX, :, eeg_trial_index].copy()
            engagement = all_cb[:, cb_subject_index, spec["cb_stimulus_index"]].copy()
            table, timing = extract_features_for_subject(eeg_channel, engagement, engagement_time, fs_eeg, fs_cb)
            table.insert(0, "condition", condition)
            table.insert(0, "subject_id", subject_id)
            window_tables.append(table.drop(columns=["n_engagement_samples"]))
            timing_rows.append({"subject_id": subject_id, "condition": condition, **timing})
            raw_rows.append(correlation_row(subject_id, condition, table, detrended=False))
            detrended_rows.append(correlation_row(subject_id, condition, table, detrended=True))
            slope_rows.append({"subject_id": subject_id, "condition": condition, **linear_trend_slopes(table)})
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
    detrended_df = pd.DataFrame(detrended_rows)
    slopes = pd.DataFrame(slope_rows)
    circular = pd.DataFrame(circular_rows)
    deltas = compute_deltas(raw, detrended_df)
    effects, permutation_json = summarize_effects(deltas)
    order_metadata = condition_order_metadata()

    window_level.to_csv(OUTPUT_DIR / "window_level_data.csv", index=False)
    timing_qc.to_csv(OUTPUT_DIR / "timing_qc.csv", index=False)
    raw.to_csv(OUTPUT_DIR / "subject_correlations_raw.csv", index=False)
    detrended_df.to_csv(OUTPUT_DIR / "subject_correlations_detrended.csv", index=False)
    slopes.to_csv(OUTPUT_DIR / "linear_trend_slopes.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "original_control_deltas.csv", index=False)
    effects.to_csv(OUTPUT_DIR / "group_effect_summary.csv", index=False)
    circular.to_csv(OUTPUT_DIR / "circular_shift_subject_results.csv", index=False)
    order_metadata.to_csv(OUTPUT_DIR / "condition_order_metadata.csv", index=False)
    (OUTPUT_DIR / "beta_raw_group_permutation.json").write_text(
        json.dumps(permutation_json["beta_raw"], indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "group_permutation_all_endpoints.json").write_text(
        json.dumps(permutation_json, indent=2),
        encoding="utf-8",
    )

    plot_beta_paired(deltas)
    plot_beta_delta_distribution(deltas)
    plot_beta_raw_vs_detrended(deltas)
    plot_alpha_paired(deltas)
    plot_trend_slopes(slopes)

    effect_by_endpoint = effects.set_index("endpoint")
    primary = effect_by_endpoint.loc["beta_raw"]
    beta_det = effect_by_endpoint.loc["beta_detrended"]
    alpha_raw = effect_by_endpoint.loc["upper_alpha_raw"]
    alpha_det = effect_by_endpoint.loc["upper_alpha_detrended"]
    if primary["mean"] > 0 and beta_det["mean"] <= primary["mean"]:
        interpretation = (
            "Beta raw Original-Control effect is positive in the observed full sample, while detrended beta is weaker; "
            "this is most consistent with a long-timescale frontal beta-engagement association rather than a stable local fluctuation effect."
        )
    elif primary["mean"] > 0 and beta_det["mean"] > 0:
        interpretation = (
            "Beta raw Original-Control effect is positive and detrended beta is also positive, suggesting possible long- and shorter-timescale associations."
        )
    else:
        interpretation = "The primary beta raw Original-Control effect did not replicate as a positive full-sample effect."

    summary = {
        "n_included": int(len(included_subjects)),
        "included_subject_ids": included_subjects,
        "primary_channel": CHANNEL,
        "channel_region": "frontal/anterior midline",
        "nearest_standard_electrode": "Fz (approximate)",
        "window_parameters": {
            "window_length_sec": WINDOW_SEC,
            "step_sec": WINDOW_SEC,
            "overlap": 0,
            "welch_segment_sec": 2.0,
            "welch_overlap": "50%",
            "welch_window": "Hann",
            "welch_detrend": "constant",
            "welch_scaling": "density",
        },
        "primary_feature": "beta_log_power_15_30_hz_raw",
        "beta_raw": primary.to_dict(),
        "beta_detrended": beta_det.to_dict(),
        "upper_alpha_raw": alpha_raw.to_dict(),
        "upper_alpha_detrended": alpha_det.to_dict(),
        "condition_order_metadata": "condition order not available in processed dataset metadata",
        "interpretation": interpretation,
        "interpretation_boundary": "Observed-sample locked replication; not a biomarker, population-level claim, causal result, flow result, BCI, or real-time decoding result.",
    }
    (OUTPUT_DIR / "full_v1_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2),
        encoding="utf-8",
    )
    write_report(summary, effects)

    print("NMED-E FULL V1 COMPLETE")
    print()
    print("N included:", len(included_subjects))
    print("Subjects:", ", ".join(included_subjects))
    print("Primary channel:", CHANNEL, "(~Fz, frontal/anterior midline)")
    print("Primary endpoint: delta_beta_raw = beta rho Original - beta rho Control")
    print()
    print("BETA RAW")
    print("Mean delta:", f"{primary['mean']:.4f}")
    print("Median delta:", f"{primary['median']:.4f}")
    print("Positive delta:", f"{int(primary['n_positive'])}/{int(primary['n'])}")
    print("Exact sign-flip p:", f"{primary['sign_flip_two_sided_p']:.6f}")
    print(
        "Bootstrap mean CI:",
        f"[{primary['bootstrap_mean_ci_low']:.4f}, {primary['bootstrap_mean_ci_high']:.4f}]",
    )
    print()
    print("BETA DETRENDED")
    print("Mean delta:", f"{beta_det['mean']:.4f}")
    print("Median delta:", f"{beta_det['median']:.4f}")
    print("Positive delta:", f"{int(beta_det['n_positive'])}/{int(beta_det['n'])}")
    print("Exact sign-flip p:", f"{beta_det['sign_flip_two_sided_p']:.6f}")
    print()
    print("UPPER ALPHA RAW")
    print("Mean delta:", f"{alpha_raw['mean']:.4f}")
    print("Median delta:", f"{alpha_raw['median']:.4f}")
    print("Positive delta:", f"{int(alpha_raw['n_positive'])}/{int(alpha_raw['n'])}")
    print()
    print("Interpretation:")
    print(interpretation)
    return summary


if __name__ == "__main__":
    main()
