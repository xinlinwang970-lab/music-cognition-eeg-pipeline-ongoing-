from __future__ import annotations

import json
import math
import platform
import re
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from run_full_v1 import bootstrap_ci, exact_sign_flip, json_ready
from scripts.analysis.ds002721_features import (
    BETA_BAND,
    FEATURE_SPEC_ID,
    PREPROCESSING_ID,
    beta_log_power,
    spearman,
    subject_effects,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "ds002721"
DERIVED_DIR = PROJECT_DIR / "derived"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ds002721_v2_0"
REPORT_DIR = PROJECT_DIR / "reports"

DATASET_ID = "ds002721"
SNAPSHOT_TAG = "1.0.3"
SEED = 20260819
MAX_EXACT_SIGN_FLIP_N = 24
MONTE_CARLO_SIGN_FLIP_REPS = 50000
PRIMARY_CHANNEL = "Fz"
FRONTAL_ROI = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"]
MUSIC_RUNS = {2, 3, 4, 5}
QUESTION_CODES = {
    "pleasantness": 800,
    "energy": 801,
    "tension": 802,
}
ALL_QUESTION_CODES = set(range(800, 808))
CONSTRUCT_FAMILIES = {
    "energy": "affective_activation",
    "tension": "affective_tension",
    "pleasantness": "affective_valence",
}


@dataclass(frozen=True)
class TrialSpec:
    subject_id: str
    run_id: str
    trial_id: str
    stimulus_id: str
    onset: float
    duration: float
    energy: float
    tension: float
    pleasantness: float


def ensure_dirs() -> None:
    for path in [DATA_DIR, DERIVED_DIR, OUTPUT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_openneuro_manifest() -> pd.DataFrame:
    manifest_path = DATA_DIR / "openneuro_file_manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    query = """
    query DatasetFiles($datasetId: ID!, $tag: String!) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        id
        tag
        size
        files(recursive: true) {
          filename
          size
          directory
          urls
        }
      }
    }
    """
    response = requests.post(
        "https://openneuro.org/crn/graphql",
        json={"query": query, "variables": {"datasetId": DATASET_ID, "tag": SNAPSHOT_TAG}},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])

    rows = []
    for file_info in payload["data"]["snapshot"]["files"]:
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "snapshot_tag": SNAPSHOT_TAG,
                "filename": file_info["filename"],
                "size": file_info["size"],
                "directory": bool(file_info["directory"]),
                "url": file_info["urls"][0] if file_info.get("urls") else "",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def download_file(manifest: pd.DataFrame, rel_path: str) -> Path:
    row = manifest[manifest["filename"].eq(rel_path)]
    if row.empty:
        raise FileNotFoundError(f"{rel_path} not found in OpenNeuro manifest")
    info = row.iloc[0]
    target = DATA_DIR / rel_path
    expected_size = int(info["size"]) if pd.notna(info["size"]) else None
    if target.exists() and expected_size is not None and target.stat().st_size == expected_size:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(str(info["url"]), stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if expected_size is not None and target.stat().st_size != expected_size:
        raise IOError(f"Size mismatch for {rel_path}: got {target.stat().st_size}, expected {expected_size}")
    return target


def run_number(filename: str) -> int | None:
    match = re.search(r"_task-run(\d+)_", filename)
    return int(match.group(1)) if match else None


def subject_ids(manifest: pd.DataFrame) -> list[str]:
    ids = sorted({Path(name).parts[0] for name in manifest["filename"] if str(name).startswith("sub-")})
    return ids


def subject_music_files(manifest: pd.DataFrame, subject_id: str, suffix: str) -> list[str]:
    files = []
    for name in manifest["filename"]:
        name = str(name)
        if not name.startswith(f"{subject_id}/eeg/") or not name.endswith(suffix):
            continue
        run = run_number(name)
        if run in MUSIC_RUNS:
            files.append(name)
    return sorted(files, key=lambda n: run_number(n) or 0)


def download_root_metadata(manifest: pd.DataFrame) -> None:
    for rel_path in ["README", "CHANGES", "dataset_description.json", "participants.tsv"]:
        download_file(manifest, rel_path)


def download_subject_metadata(manifest: pd.DataFrame, subject_id: str) -> None:
    suffixes = ["_channels.tsv", "_events.tsv", "_events.json", "_eeg.json"]
    for suffix in suffixes:
        for rel_path in subject_music_files(manifest, subject_id, suffix):
            download_file(manifest, rel_path)


def download_subject_eeg(manifest: pd.DataFrame, subject_id: str) -> None:
    download_subject_metadata(manifest, subject_id)
    for rel_path in subject_music_files(manifest, subject_id, "_eeg.edf"):
        download_file(manifest, rel_path)


def parse_events(events_path: Path, subject_id: str, run_id: str) -> list[TrialSpec]:
    events = pd.read_csv(events_path, sep="\t")
    events["trial_type"] = pd.to_numeric(events["trial_type"], errors="coerce").astype("Int64")
    events = events.dropna(subset=["trial_type"]).sort_values("onset").reset_index(drop=True)
    music_rows = events[events["trial_type"].eq(788)].reset_index(drop=True)
    trials: list[TrialSpec] = []

    for idx, music_row in music_rows.iterrows():
        onset = float(music_row["onset"])
        next_onset = float(music_rows.loc[idx + 1, "onset"]) if idx + 1 < len(music_rows) else math.inf
        segment = events[(events["onset"] >= onset) & (events["onset"] < next_onset)]
        same_onset = segment[np.isclose(segment["onset"].astype(float), onset, atol=0.005)]
        pre_music = events[(events["onset"] >= onset - 10.0) & (events["onset"] <= onset)]
        stim_rows = pd.concat(
            [
                same_onset[same_onset["trial_type"].between(300, 699, inclusive="both")],
                pre_music[pre_music["trial_type"].between(300, 699, inclusive="both")],
            ],
            ignore_index=True,
        ).sort_values("onset")
        stimulus_id = (
            str(int(stim_rows["trial_type"].iloc[-1])) if len(stim_rows) else f"{subject_id}_{run_id}_{idx + 1:02d}"
        )

        ratings: dict[str, float] = {}
        for rating_name, question_code in QUESTION_CODES.items():
            answer_value = np.nan
            q_positions = [pos for pos, code in enumerate(segment["trial_type"].astype(int).tolist()) if code == question_code]
            for q_pos in q_positions:
                end_pos = len(segment)
                codes_after = segment["trial_type"].astype(int).tolist()
                for later_pos in range(q_pos + 1, len(segment)):
                    later_code = codes_after[later_pos]
                    if later_code in ALL_QUESTION_CODES and later_code != question_code:
                        end_pos = later_pos
                        break
                block = segment.iloc[q_pos:end_pos]
                answer_rows = block[block["trial_type"].between(901, 909, inclusive="both")]
                if not answer_rows.empty:
                    answer_value = float(int(answer_rows["trial_type"].iloc[-1]) - 900)
                    break
                response_rows = block[block["trial_type"].between(833, 841, inclusive="both")]
                if not response_rows.empty:
                    answer_value = float(int(response_rows["trial_type"].iloc[-1]) - 832)
                    break
            ratings[rating_name] = answer_value

        duration = float(music_row.get("duration", 20.0))
        if not np.isfinite(duration) or duration <= 0:
            duration = 20.0
        trials.append(
            TrialSpec(
                subject_id=subject_id,
                run_id=run_id,
                trial_id=f"{subject_id}_{run_id}_{idx + 1:02d}",
                stimulus_id=stimulus_id,
                onset=onset,
                duration=duration,
                energy=ratings["energy"],
                tension=ratings["tension"],
                pleasantness=ratings["pleasantness"],
            )
        )
    return trials


def synthetic_validation() -> dict[str, object]:
    fs = 1000.0
    t = np.arange(0, 20.0, 1.0 / fs)
    beta20 = beta_log_power(np.sin(2 * np.pi * 20.0 * t), fs)
    alpha11 = beta_log_power(np.sin(2 * np.pi * 11.0 * t), fs)
    ok = bool(beta20 > alpha11)
    result = {"beta_20hz_log_power": beta20, "beta_11hz_log_power": alpha11, "passed": ok}
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    if not ok:
        raise AssertionError("Synthetic beta validation failed: 20 Hz must exceed 11 Hz in beta power")
    return result


def mne_version() -> str:
    import mne

    return str(mne.__version__)


def channels_for_subject(manifest: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    first_channel_file = subject_music_files(manifest, subject_id, "_channels.tsv")[0]
    path = download_file(manifest, first_channel_file)
    return pd.read_csv(path, sep="\t")


def freeze_channel_and_roi(manifest: pd.DataFrame, subject_id: str) -> tuple[str, list[str]]:
    channels = channels_for_subject(manifest, subject_id)
    labels = channels["name"].astype(str).tolist()
    if PRIMARY_CHANNEL not in labels:
        raise ValueError("Primary Fz channel is absent; nearest frontal-midline fallback must be specified manually")
    roi = [label for label in FRONTAL_ROI if label in labels]
    if PRIMARY_CHANNEL not in roi:
        raise ValueError("Frozen ROI must include Fz")
    pd.DataFrame(
        {
            "channel": roi,
            "role": ["primary_and_roi" if ch == PRIMARY_CHANNEL else "roi" for ch in roi],
            "rule": "Predeclared 10-20 frontal channels: FP1, FP2, F7, F3, Fz, F4, F8",
        }
    ).to_csv(OUTPUT_DIR / "frontal_roi_definition.csv", index=False)
    return PRIMARY_CHANNEL, roi


def inspect_metadata(manifest: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    download_root_metadata(manifest)
    subjects = subject_ids(manifest)
    first_subject = subjects[0]
    download_subject_metadata(manifest, first_subject)
    primary, roi = freeze_channel_and_roi(manifest, first_subject)

    channel_rows = []
    for rel_path in subject_music_files(manifest, first_subject, "_channels.tsv"):
        df = pd.read_csv(DATA_DIR / rel_path, sep="\t")
        run = f"run{run_number(rel_path)}"
        for _, row in df.iterrows():
            channel_rows.append({"subject_id": first_subject, "run_id": run, **row.to_dict()})
    pd.DataFrame(channel_rows).to_csv(OUTPUT_DIR / "channel_inventory.csv", index=False)

    rating_rows = [
        {
            "rating_name": "pleasantness",
            "event_code": 800,
            "construct_family": "affective_valence",
            "scale_min": 1,
            "scale_max": 9,
            "scale_direction": "higher means stronger agreement with the item",
            "role": "construct_contrast",
        },
        {
            "rating_name": "energy",
            "event_code": 801,
            "construct_family": "affective_activation",
            "scale_min": 1,
            "scale_max": 9,
            "scale_direction": "higher means stronger agreement with the item",
            "role": "primary",
        },
        {
            "rating_name": "tension",
            "event_code": 802,
            "construct_family": "affective_tension",
            "scale_min": 1,
            "scale_max": 9,
            "scale_direction": "higher means stronger agreement with the item",
            "role": "secondary",
        },
    ]
    pd.DataFrame(rating_rows).to_csv(OUTPUT_DIR / "rating_inventory.csv", index=False)

    participants = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")
    dataset_manifest = pd.DataFrame(
        [
            {
                "dataset": "DS002721",
                "openneuro_dataset_id": DATASET_ID,
                "snapshot_tag": SNAPSHOT_TAG,
                "n_participants": int(len(participants)),
                "n_openneuro_files": int(len(manifest)),
                "eeg_format": "EDF",
                "sampling_rate_hz": 1000,
                "primary_channel": primary,
                "frontal_roi": ",".join(roi),
                "stimulus_type": "film_score_music_clips",
                "population_type": "healthy_adults",
                "rating_granularity": "trial",
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    dataset_manifest.to_csv(DERIVED_DIR / "ds002721_dataset_manifest.csv", index=False)
    dataset_manifest.to_csv(DERIVED_DIR / "dataset_manifest.csv", index=False)
    return first_subject, primary, roi


def extract_subject_trials(manifest: pd.DataFrame, subject_id: str, primary: str, roi: list[str]) -> pd.DataFrame:
    import mne

    download_subject_eeg(manifest, subject_id)
    rows = []
    qc_rows = []
    for edf_rel in subject_music_files(manifest, subject_id, "_eeg.edf"):
        run = run_number(edf_rel)
        run_id = f"run{run}"
        events_rel = edf_rel.replace("_eeg.edf", "_events.tsv")
        trials = parse_events(DATA_DIR / events_rel, subject_id, run_id)
        channels_rel = edf_rel.replace("_eeg.edf", "_channels.tsv")
        channels = pd.read_csv(DATA_DIR / channels_rel, sep="\t")
        available = set(channels["name"].astype(str))
        picks = list(dict.fromkeys(ch for ch in [primary, *roi] if ch in available))
        if primary not in picks:
            raise ValueError(f"{subject_id} {run_id}: primary channel {primary} not in EDF")
        raw = mne.io.read_raw_edf(DATA_DIR / edf_rel, include=picks, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data_all = raw.get_data()
        pick_to_index = {ch: idx for idx, ch in enumerate(raw.ch_names)}
        raw_n_times = int(raw.n_times)
        edf_duration_sec = raw_n_times / sfreq

        for trial in trials:
            requested_start = int(round(trial.onset * sfreq))
            requested_stop = int(round((trial.onset + trial.duration) * sfreq))
            start = max(0, requested_start)
            stop = min(raw_n_times, requested_stop)
            trial_end = trial.onset + trial.duration
            trimmed = bool(start != requested_start or stop != requested_stop)
            inside_bounds = bool(requested_start >= 0 and requested_stop <= raw_n_times)
            n_samples = int(max(0, stop - start))
            skip_reason = ""
            if n_samples < int(round(sfreq * 10.0)):
                skip_reason = "less_than_10s_after_bounds_clipping"
                qc_rows.append(
                    {
                        "subject_id": subject_id,
                        "run_id": run_id,
                        "trial_id": trial.trial_id,
                        "stimulus_id": trial.stimulus_id,
                        "edf_file": edf_rel,
                        "edf_duration_sec": edf_duration_sec,
                        "onset": trial.onset,
                        "duration": trial.duration,
                        "trial_end": trial_end,
                        "requested_start_sample": requested_start,
                        "requested_stop_sample": requested_stop,
                        "start_sample": start,
                        "stop_sample": stop,
                        "raw_n_times": raw_n_times,
                        "n_samples": n_samples,
                        "inside_bounds": inside_bounds,
                        "trimmed": trimmed,
                        "skipped": True,
                        "skip_reason": skip_reason,
                    }
                )
                continue
            data = data_all[:, start:stop]
            channel_beta = {ch: beta_log_power(data[pick_to_index[ch]], sfreq) for ch in picks}
            roi_values = [channel_beta[ch] for ch in roi if ch in channel_beta and np.isfinite(channel_beta[ch])]
            qc_rows.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "stimulus_id": trial.stimulus_id,
                    "edf_file": edf_rel,
                    "edf_duration_sec": edf_duration_sec,
                    "onset": trial.onset,
                    "duration": trial.duration,
                    "trial_end": trial_end,
                    "requested_start_sample": requested_start,
                    "requested_stop_sample": requested_stop,
                    "start_sample": start,
                    "stop_sample": stop,
                    "raw_n_times": raw_n_times,
                    "n_samples": n_samples,
                    "inside_bounds": inside_bounds,
                    "trimmed": trimmed,
                    "skipped": False,
                    "skip_reason": skip_reason,
                }
            )
            rows.append(
                {
                    "dataset": "DS002721",
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "stimulus_id": trial.stimulus_id,
                    "condition": "music",
                    "aggregation_level": "trial",
                    "window_start": trial.onset,
                    "window_end": trial.onset + n_samples / sfreq,
                    "primary_channel": primary,
                    "channel_or_roi": primary,
                    "feature_name": "beta_log_power",
                    "feature_value": channel_beta[primary],
                    "beta_log_power": channel_beta[primary],
                    "frontal_roi_beta_log_power": float(np.median(roi_values)) if roi_values else np.nan,
                    "energy": trial.energy,
                    "tension": trial.tension,
                    "pleasantness": trial.pleasantness,
                    "rating_complete": bool(
                        np.isfinite(trial.energy) and np.isfinite(trial.tension) and np.isfinite(trial.pleasantness)
                    ),
                    "rating_granularity": "trial",
                    "stimulus_type": "film_score_music_clip",
                    "population_type": "healthy_adults",
                    "feature_spec_id": FEATURE_SPEC_ID,
                    "preprocessing_id": PREPROCESSING_ID,
                }
            )
        raw.close()
    qc = pd.DataFrame(qc_rows)
    qc_path = OUTPUT_DIR / "trial_bounds_qc.csv"
    if qc_path.exists():
        previous = pd.read_csv(qc_path)
        previous = previous[~previous["subject_id"].eq(subject_id)]
        qc = pd.concat([previous, qc], ignore_index=True)
    qc.to_csv(qc_path, index=False)
    return pd.DataFrame(rows)


def effect_summary(values: np.ndarray, endpoint: str, family: str) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(np.mean(values))
    if len(values) <= MAX_EXACT_SIGN_FLIP_N:
        observed, p, null = exact_sign_flip(values)
        sign_flip_method = "exact"
        n_sign_samples = int(len(null))
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(MONTE_CARLO_SIGN_FLIP_REPS, len(values)))
        null = (signs * values).mean(axis=1)
        p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))
        sign_flip_method = "monte_carlo"
        n_sign_samples = int(len(null))
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
        "sign_flip_method": sign_flip_method,
        "n_sign_samples": n_sign_samples,
        "n_sign_combinations": int(2 ** len(values)) if len(values) < 63 else "",
        "bootstrap_mean_ci_low": mean_ci[0],
        "bootstrap_mean_ci_high": mean_ci[1],
        "bootstrap_median_ci_low": median_ci[0],
        "bootstrap_median_ci_high": median_ci[1],
    }


def group_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spatial in ["primary", "roi"]:
        for target in ["energy", "tension", "pleasantness"]:
            col = f"rho_{target}_{spatial}"
            rows.append(effect_summary(effects[col].to_numpy(dtype=float), f"{target}_{spatial}", "DS002721"))
    return pd.DataFrame(rows)


def loso_group_rating(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rating_cols = ["energy", "tension", "pleasantness"]
    group_means = trials.groupby(["stimulus_id", "subject_id"], as_index=False)[rating_cols].mean()
    rows = []
    for _, trial in trials.iterrows():
        peers = group_means[
            group_means["stimulus_id"].eq(trial["stimulus_id"])
            & ~group_means["subject_id"].eq(trial["subject_id"])
        ]
        row = trial.to_dict()
        for target in rating_cols:
            row[f"group_{target}_minus_subject"] = float(peers[target].mean()) if not peers.empty else np.nan
        rows.append(row)
    augmented = pd.DataFrame(rows)

    effect_rows = []
    for subject_id, df in augmented.groupby("subject_id", sort=True):
        row: dict[str, object] = {"subject_id": subject_id, "n_trials": int(len(df))}
        for target in rating_cols:
            row[f"rho_group_{target}_primary"] = spearman(df["beta_log_power"], df[f"group_{target}_minus_subject"])
            row[f"rho_group_{target}_roi"] = spearman(
                df["frontal_roi_beta_log_power"], df[f"group_{target}_minus_subject"]
            )
        effect_rows.append(row)
    return augmented, pd.DataFrame(effect_rows)


def run_subjects(manifest: pd.DataFrame, subjects: list[str], primary: str, roi: list[str]) -> pd.DataFrame:
    frames = []
    for idx, subject_id in enumerate(subjects, start=1):
        cache_path = OUTPUT_DIR / f"{subject_id}_trial_features.csv"
        if cache_path.exists():
            print(f"[{idx}/{len(subjects)}] DS002721 using cache {subject_id}", flush=True)
            frame = pd.read_csv(cache_path)
        else:
            print(f"[{idx}/{len(subjects)}] DS002721 extracting {subject_id}", flush=True)
            frame = extract_subject_trials(manifest, subject_id, primary, roi)
            frame.to_csv(cache_path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_unified_tables(trials: pd.DataFrame, effects: pd.DataFrame, summary: pd.DataFrame) -> None:
    trials = trials.copy()
    string_columns = [
        "dataset",
        "subject_id",
        "run_id",
        "trial_id",
        "stimulus_id",
        "condition",
        "aggregation_level",
        "primary_channel",
        "channel_or_roi",
        "feature_name",
        "rating_granularity",
        "stimulus_type",
        "population_type",
        "feature_spec_id",
        "preprocessing_id",
    ]
    for column in string_columns:
        if column in trials.columns:
            trials[column] = trials[column].astype("string")
    trials.to_parquet(DERIVED_DIR / "ds002721_trial_level_features.parquet", index=False)
    trials.to_parquet(DERIVED_DIR / "trial_level_features.parquet", index=False)
    effects.to_csv(OUTPUT_DIR / "subject_level_effects.csv", index=False)
    effects.to_csv(DERIVED_DIR / "subject_level_effects.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "group_level_summary.csv", index=False)
    summary.to_csv(DERIVED_DIR / "group_level_summary.csv", index=False)


def plot_outputs(effects: pd.DataFrame, loso_effects: pd.DataFrame, cross: pd.DataFrame) -> None:
    energy = effects["rho_energy_primary"].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(energy, bins=10, color="#5da5da", edgecolor="black")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("DS002721 Energy subject effects")
    plt.xlabel("rho(beta, Energy)")
    plt.ylabel("subjects")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_energy_subject_effects.png", dpi=160)
    plt.close()

    labels = ["Energy", "Tension", "Pleasantness"]
    data = [
        effects["rho_energy_primary"].to_numpy(dtype=float),
        effects["rho_tension_primary"].to_numpy(dtype=float),
        effects["rho_pleasantness_primary"].to_numpy(dtype=float),
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Construct comparison")
    plt.ylabel("subject-level rho")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_construct_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(effects["rho_energy_primary"], effects["rho_energy_roi"], color="#60bd68")
    for _, row in effects.iterrows():
        plt.text(row["rho_energy_primary"], row["rho_energy_roi"], row["subject_id"], fontsize=8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Primary Fz vs frontal ROI Energy")
    plt.xlabel("Fz rho(beta, Energy)")
    plt.ylabel("ROI rho(beta, Energy)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_primary_vs_roi_energy.png", dpi=160)
    plt.close()

    merged = effects[["subject_id", "rho_energy_primary"]].merge(
        loso_effects[["subject_id", "rho_group_energy_primary"]], on="subject_id"
    )
    plt.figure(figsize=(7, 6))
    plt.scatter(merged["rho_energy_primary"], merged["rho_group_energy_primary"], color="#f17cb0")
    for _, row in merged.iterrows():
        plt.text(row["rho_energy_primary"], row["rho_group_energy_primary"], row["subject_id"], fontsize=8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Own Energy vs LOSO group Energy")
    plt.xlabel("own Energy rho")
    plt.ylabel("LOSO group Energy rho")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_personal_vs_group_energy.png", dpi=160)
    plt.close()

    matrix = cross.pivot(index="dataset", columns="target_name", values="mean_effect")
    plt.figure(figsize=(8, 4))
    im = plt.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-0.3, vmax=0.3)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, label="mean effect")
    plt.title("Cross-dataset construct matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_cross_dataset_construct_matrix.png", dpi=160)
    plt.close()


def smoke_one_subject(trials: pd.DataFrame) -> None:
    first_subject = sorted(trials["subject_id"].unique())[0]
    df = trials[trials["subject_id"].eq(first_subject)].copy()
    df.to_csv(OUTPUT_DIR / "smoke_one_subject_trial_features.csv", index=False)
    if df["trial_id"].duplicated().any():
        raise AssertionError("Duplicate trial IDs in one-subject smoke test")
    if df[["beta_log_power", "frontal_roi_beta_log_power"]].isna().any().any():
        raise AssertionError("Missing beta feature in one-subject smoke test")
    target_counts = df[["energy", "tension", "pleasantness"]].notna().sum()
    if (target_counts < 10).any():
        raise AssertionError(f"Too few valid target ratings in one-subject smoke test: {target_counts.to_dict()}")
    rating_qc = pd.DataFrame(
        [
            {
                "subject_id": first_subject,
                "n_trials": int(len(df)),
                "n_complete_all_three_targets": int(
                    df[["energy", "tension", "pleasantness"]].notna().all(axis=1).sum()
                ),
                "n_energy": int(target_counts["energy"]),
                "n_tension": int(target_counts["tension"]),
                "n_pleasantness": int(target_counts["pleasantness"]),
                "n_missing_energy": int(df["energy"].isna().sum()),
                "n_missing_tension": int(df["tension"].isna().sum()),
                "n_missing_pleasantness": int(df["pleasantness"].isna().sum()),
            }
        ]
    )
    rating_qc.to_csv(OUTPUT_DIR / "smoke_one_rating_qc.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(df)), df["beta_log_power"], marker="o")
    plt.title(f"{first_subject} beta by trial")
    plt.xlabel("trial")
    plt.ylabel("Fz beta log power")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "00_smoke_beta_by_trial.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(df)), df["energy"], marker="o", color="#f17cb0")
    plt.title(f"{first_subject} Energy by trial")
    plt.xlabel("trial")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "00_smoke_energy_by_trial.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(df["beta_log_power"], df["energy"])
    plt.title(f"{first_subject} beta vs Energy")
    plt.xlabel("Fz beta log power")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "00_smoke_beta_vs_energy.png", dpi=160)
    plt.close()


def cross_dataset_summary(ds_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nmede_summary_path = PROJECT_DIR / "v1_1_outputs" / "timescale_group_summary.csv"
    if nmede_summary_path.exists():
        nmede = pd.read_csv(nmede_summary_path)
        endpoint = "fast60" if "fast60" in set(nmede["endpoint"]) else "raw"
        row = nmede[nmede["endpoint"].eq(endpoint)].iloc[0]
        rows.append(
            {
                "dataset": "NMED-E",
                "n_subjects": int(row["n"]),
                "stimulus_type": "naturalistic_music_original_control",
                "target_name": "engagement",
                "construct_family": "engagement",
                "target_granularity": "continuous",
                "neural_feature": "frontal_beta_15_30_hz",
                "spatial_feature": "E11_near_Fz",
                "aggregation_level": endpoint,
                "mean_effect": float(row["mean"]),
                "median_effect": float(row["median"]),
                "n_positive": int(row["n_positive"]),
                "sign_flip_p": float(row["sign_flip_two_sided_p"]),
                "bootstrap_ci_low": float(row["bootstrap_mean_ci_low"]),
                "bootstrap_ci_high": float(row["bootstrap_mean_ci_high"]),
                "interpretation": "NMED-E discovery baseline; not a DS002721 replication target",
            }
        )

    for target in ["energy", "tension", "pleasantness"]:
        row = ds_summary[ds_summary["endpoint"].eq(f"{target}_primary")].iloc[0]
        rows.append(
            {
                "dataset": "DS002721",
                "n_subjects": int(row["n"]),
                "stimulus_type": "film_score_music_clips",
                "target_name": target,
                "construct_family": CONSTRUCT_FAMILIES[target],
                "target_granularity": "trial",
                "neural_feature": "frontal_beta_15_30_hz",
                "spatial_feature": "Fz",
                "aggregation_level": "trial",
                "mean_effect": float(row["mean"]),
                "median_effect": float(row["median"]),
                "n_positive": int(row["n_positive"]),
                "sign_flip_p": float(row["sign_flip_two_sided_p"]),
                "bootstrap_ci_low": float(row["bootstrap_mean_ci_low"]),
                "bootstrap_ci_high": float(row["bootstrap_mean_ci_high"]),
                "interpretation": "DS002721 transportability target; construct-specific, not pooled",
            }
        )
    cross = pd.DataFrame(rows)
    cross.to_csv(DERIVED_DIR / "cross_dataset_v2_0_summary.csv", index=False)
    return cross


def write_report(
    subjects: list[str],
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    loso_effects: pd.DataFrame,
    cross: pd.DataFrame,
) -> None:
    energy = summary[summary["endpoint"].eq("energy_primary")].iloc[0]
    tension = summary[summary["endpoint"].eq("tension_primary")].iloc[0]
    pleasantness = summary[summary["endpoint"].eq("pleasantness_primary")].iloc[0]
    roi_energy = summary[summary["endpoint"].eq("energy_roi")].iloc[0]
    loso_energy_summary = effect_summary(
        loso_effects["rho_group_energy_primary"].to_numpy(dtype=float), "group_energy_primary", "DS002721_LOSO"
    )

    go = (
        abs(float(energy["mean"])) >= 0.05
        or np.sign(float(energy["mean"])) == np.sign(float(roi_energy["mean"]))
        or abs(float(loso_energy_summary["mean"])) > abs(float(energy["mean"]))
    )
    decision = "GO to MUSIN-G" if go else "RECONSIDER before MUSIN-G"

    text = f"""# DS002721 V2.0 Results

## 1. Why This Dataset

DS002721 is an independent OpenNeuro affective music-listening EEG dataset. It is used here to evaluate transportability of the predeclared frontal beta feature, not to replicate NMED-E engagement.

## 2. Construct Distinctions

- Energy != Engagement.
- Tension != Engagement.
- Pleasantness != Engagement.
- The constructs are not pooled into one overall effect.

## 3. Dataset Structure

- OpenNeuro dataset: `{DATASET_ID}`
- Snapshot used: `{SNAPSHOT_TAG}`
- Subjects analyzed: {len(subjects)}
- EEG format: EDF
- Sampling rate from sidecar: 1000 Hz
- Music runs analyzed: run2-run5
- BIDS trial duration observed in events: 20 s

## 4. Frozen Beta Feature

The transported feature is frontal beta 15-30 Hz log10 absolute Welch band power. Primary channel was frozen as Fz because it exists in the first inspected channel file. Frontal ROI was frozen before outcome analysis as: {", ".join(FRONTAL_ROI)}.

## 5. Trial-Level Aggregation

Each music trial produces exactly one Fz beta value and one frontal ROI beta value. Ratings are parsed from question/answer event codes and retained at the trial level. No 5 s pseudo-replicated rating windows are created.

## 6. Energy Result

- mean rho: {energy['mean']:.4f}
- median rho: {energy['median']:.4f}
- positive subjects: {int(energy['n_positive'])}/{int(energy['n'])}
- sign-flip p: {energy['sign_flip_two_sided_p']:.4g}
- bootstrap mean CI: [{energy['bootstrap_mean_ci_low']:.4f}, {energy['bootstrap_mean_ci_high']:.4f}]

## 7. Tension Result

- mean rho: {tension['mean']:.4f}
- median rho: {tension['median']:.4f}
- positive subjects: {int(tension['n_positive'])}/{int(tension['n'])}
- sign-flip p: {tension['sign_flip_two_sided_p']:.4g}

## 8. Pleasantness Contrast

- mean rho: {pleasantness['mean']:.4f}
- median rho: {pleasantness['median']:.4f}
- positive subjects: {int(pleasantness['n_positive'])}/{int(pleasantness['n'])}
- sign-flip p: {pleasantness['sign_flip_two_sided_p']:.4g}

## 9. ROI Robustness

Energy ROI mean rho was {roi_energy['mean']:.4f}. This is compared against Fz without changing ROI membership after seeing outcomes.

## 10. LOSO Group-Rating Robustness

For each stimulus and subject, group ratings were computed from all other subjects. LOSO group Energy mean rho was {loso_energy_summary['mean']:.4f}. Group ratings are a shared-stimulus reference, not ground truth.

## 11. Comparison With NMED-E

NMED-E transports only the frontal beta feature, not the engagement construct or Original-Control endpoint. The cross-dataset summary is saved to `derived/cross_dataset_v2_0_summary.csv`.

## 12. Limitations

- Raw EEG was used without artifact rejection.
- DS002721 targets affective ratings, not engagement.
- Music trial duration differs from the plan expectation; BIDS events/README indicate 20 s clips.
- No flow claim is made.
- No causal claim is made.
- No biomarker claim is made.
- No BCI claim is made.

## 13. Decision for MUSIN-G

Decision: **{decision}**.

This decision does not require p < 0.05. It reflects whether a coherent construct-specific pattern, ROI consistency, or LOSO group-rating pattern exists.
"""
    (REPORT_DIR / "ds002721_v2_0_results.md").write_text(text, encoding="utf-8")


def finalize_outputs(
    validation: dict[str, object],
    subjects: list[str],
    primary: str,
    roi: list[str],
) -> dict[str, object]:
    all_trials = run_subjects_from_cache(subjects)
    all_trials.to_csv(OUTPUT_DIR / "trial_level_features.csv", index=False)
    effects = subject_effects(all_trials)
    summary = group_summary(effects)
    write_unified_tables(all_trials, effects, summary)

    loso_trials, loso_effects = loso_group_rating(all_trials)
    loso_trials.to_csv(OUTPUT_DIR / "group_rating_robustness_trial_table.csv", index=False)
    loso_effects.to_csv(OUTPUT_DIR / "group_rating_robustness.csv", index=False)
    loso_summary = group_summary(
        loso_effects.rename(
            columns={
                "rho_group_energy_primary": "rho_energy_primary",
                "rho_group_tension_primary": "rho_tension_primary",
                "rho_group_pleasantness_primary": "rho_pleasantness_primary",
                "rho_group_energy_roi": "rho_energy_roi",
                "rho_group_tension_roi": "rho_tension_roi",
                "rho_group_pleasantness_roi": "rho_pleasantness_roi",
            }
        )
    )
    loso_summary.to_csv(OUTPUT_DIR / "group_rating_robustness_summary.csv", index=False)

    cross = cross_dataset_summary(summary)
    plot_outputs(effects, loso_effects, cross)
    write_report(subjects, effects, summary, loso_effects, cross)

    summary_json = {
        "dataset": "DS002721",
        "snapshot_tag": SNAPSHOT_TAG,
        "n_subjects": len(subjects),
        "n_trials": int(len(all_trials)),
        "primary_channel": primary,
        "frontal_roi": roi,
        "validation": validation,
        "energy_primary_mean": float(summary[summary["endpoint"].eq("energy_primary")]["mean"].iloc[0]),
        "tension_primary_mean": float(summary[summary["endpoint"].eq("tension_primary")]["mean"].iloc[0]),
        "pleasantness_primary_mean": float(summary[summary["endpoint"].eq("pleasantness_primary")]["mean"].iloc[0]),
        "energy_roi_mean": float(summary[summary["endpoint"].eq("energy_roi")]["mean"].iloc[0]),
        "loso_group_energy_primary_mean": float(loso_summary[loso_summary["endpoint"].eq("energy_primary")]["mean"].iloc[0]),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mne": mne_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "interpretation_boundary": "Transportability analysis only; no flow, causal, biomarker, BCI, or real-time decoding claim.",
    }
    (OUTPUT_DIR / "ds002721_v2_0_summary.json").write_text(
        json.dumps(json_ready(summary_json), indent=2), encoding="utf-8"
    )

    print("DS002721 V2.0 finalize complete")
    print("Subjects:", len(subjects))
    print("Trials:", len(all_trials))
    print("Energy primary mean:", f"{summary_json['energy_primary_mean']:.4f}")
    print("Tension primary mean:", f"{summary_json['tension_primary_mean']:.4f}")
    print("Pleasantness primary mean:", f"{summary_json['pleasantness_primary_mean']:.4f}")
    print("Energy ROI mean:", f"{summary_json['energy_roi_mean']:.4f}")
    print("LOSO group Energy primary mean:", f"{summary_json['loso_group_energy_primary_mean']:.4f}")
    return summary_json


def prepare_context() -> tuple[dict[str, object], pd.DataFrame, str, str, list[str], list[str]]:
    ensure_dirs()
    validation = synthetic_validation()
    manifest = fetch_openneuro_manifest()
    first_subject, primary, roi = inspect_metadata(manifest)
    subjects = subject_ids(manifest)
    return validation, manifest, first_subject, primary, roi, subjects


def cached_subjects() -> set[str]:
    return {path.name.split("_trial_features.csv")[0] for path in OUTPUT_DIR.glob("sub-*_trial_features.csv")}


def missing_subjects(subjects: list[str]) -> list[str]:
    cached = cached_subjects()
    return [subject for subject in subjects if subject not in cached]


def run_subjects_from_cache(subjects: list[str]) -> pd.DataFrame:
    frames = []
    missing = []
    for idx, subject_id in enumerate(subjects, start=1):
        cache_path = OUTPUT_DIR / f"{subject_id}_trial_features.csv"
        if cache_path.exists():
            print(f"[{idx}/{len(subjects)}] DS002721 using cache {subject_id}", flush=True)
            frames.append(pd.read_csv(cache_path))
        else:
            missing.append(subject_id)
    if missing:
        raise FileNotFoundError(
            "Missing subject caches for finalize-cache/full: " + ", ".join(missing)
        )
    return pd.concat(frames, ignore_index=True)


def verify_existing_outputs() -> dict[str, object]:
    ensure_dirs()
    required_output_files = [
        OUTPUT_DIR / "trial_level_features.csv",
        OUTPUT_DIR / "subject_level_effects.csv",
        OUTPUT_DIR / "group_level_summary.csv",
        OUTPUT_DIR / "group_rating_robustness.csv",
        OUTPUT_DIR / "group_rating_robustness_summary.csv",
        OUTPUT_DIR / "ds002721_v2_0_summary.json",
        OUTPUT_DIR / "01_energy_subject_effects.png",
        OUTPUT_DIR / "02_construct_comparison.png",
        OUTPUT_DIR / "03_primary_vs_roi_energy.png",
        OUTPUT_DIR / "04_personal_vs_group_energy.png",
        OUTPUT_DIR / "05_cross_dataset_construct_matrix.png",
        REPORT_DIR / "ds002721_v2_0_results.md",
        DERIVED_DIR / "cross_dataset_v2_0_summary.csv",
        DERIVED_DIR / "ds002721_trial_level_features.parquet",
    ]
    cache_files = sorted(OUTPUT_DIR.glob("sub-*_trial_features.csv"))
    missing = [str(path.relative_to(PROJECT_DIR)) for path in required_output_files if not path.exists()]

    trial_rows = None
    subject_rows = None
    group_rows = None
    summary_json: dict[str, object] = {}
    refreshed_summary_json: dict[str, object] = {}
    if (OUTPUT_DIR / "trial_level_features.csv").exists():
        trial_rows = int(len(pd.read_csv(OUTPUT_DIR / "trial_level_features.csv", usecols=["subject_id"])))
    if (OUTPUT_DIR / "subject_level_effects.csv").exists():
        subject_rows = int(len(pd.read_csv(OUTPUT_DIR / "subject_level_effects.csv", usecols=["subject_id"])))
    if (OUTPUT_DIR / "group_level_summary.csv").exists():
        group_rows = int(len(pd.read_csv(OUTPUT_DIR / "group_level_summary.csv", usecols=["endpoint"])))
    if (OUTPUT_DIR / "ds002721_v2_0_summary.json").exists():
        summary_json = json.loads((OUTPUT_DIR / "ds002721_v2_0_summary.json").read_text(encoding="utf-8"))
    if (
        (OUTPUT_DIR / "group_level_summary.csv").exists()
        and (OUTPUT_DIR / "group_rating_robustness_summary.csv").exists()
        and (OUTPUT_DIR / "trial_level_features.csv").exists()
    ):
        group = pd.read_csv(OUTPUT_DIR / "group_level_summary.csv")
        loso_group = pd.read_csv(OUTPUT_DIR / "group_rating_robustness_summary.csv")
        refreshed_summary_json = dict(summary_json)
        refreshed_summary_json.update(
            {
                "dataset": "DS002721",
                "snapshot_tag": SNAPSHOT_TAG,
                "n_subjects": subject_rows,
                "n_trials": trial_rows,
                "primary_channel": PRIMARY_CHANNEL,
                "frontal_roi": FRONTAL_ROI,
                "energy_primary_mean": float(group[group["endpoint"].eq("energy_primary")]["mean"].iloc[0]),
                "tension_primary_mean": float(group[group["endpoint"].eq("tension_primary")]["mean"].iloc[0]),
                "pleasantness_primary_mean": float(group[group["endpoint"].eq("pleasantness_primary")]["mean"].iloc[0]),
                "energy_roi_mean": float(group[group["endpoint"].eq("energy_roi")]["mean"].iloc[0]),
                "loso_group_energy_primary_mean": float(
                    loso_group[loso_group["endpoint"].eq("energy_primary")]["mean"].iloc[0]
                ),
                "interpretation_boundary": (
                    "Transportability analysis only; no flow, causal, biomarker, BCI, or real-time decoding claim."
                ),
            }
        )
        (OUTPUT_DIR / "ds002721_v2_0_summary.json").write_text(
            json.dumps(json_ready(refreshed_summary_json), indent=2), encoding="utf-8"
        )

    figure_sizes = {}
    for path in required_output_files:
        if path.suffix.lower() == ".png" and path.exists():
            figure_sizes[path.name] = int(path.stat().st_size)

    check = {
        "phase": "verify-existing",
        "complete": not missing and len(cache_files) == 31,
        "missing_required_files": missing,
        "n_subject_caches": len(cache_files),
        "trial_level_rows": trial_rows,
        "subject_level_rows": subject_rows,
        "group_summary_rows": group_rows,
        "summary_json_n_subjects": summary_json.get("n_subjects"),
        "summary_json_n_trials": summary_json.get("n_trials"),
        "energy_primary_mean": summary_json.get("energy_primary_mean"),
        "tension_primary_mean": summary_json.get("tension_primary_mean"),
        "pleasantness_primary_mean": summary_json.get("pleasantness_primary_mean"),
        "loso_group_energy_primary_mean": (
            refreshed_summary_json or summary_json
        ).get("loso_group_energy_primary_mean"),
        "figure_sizes": figure_sizes,
    }
    (OUTPUT_DIR / "existing_outputs_check.json").write_text(json.dumps(json_ready(check), indent=2), encoding="utf-8")
    print("DS002721 existing-output verification complete")
    print("Complete:", check["complete"])
    print("Subject caches:", check["n_subject_caches"])
    print("Trial rows:", check["trial_level_rows"])
    print("Subject rows:", check["subject_level_rows"])
    print("Missing files:", ", ".join(missing) if missing else "none")
    return check


def main(phase: str = "full", max_new_subjects: int | None = None) -> dict[str, object]:
    validation, manifest, first_subject, primary, roi, subjects = prepare_context()

    if phase == "inspect":
        summary_json = {
            "phase": "inspect",
            "dataset": "DS002721",
            "snapshot_tag": SNAPSHOT_TAG,
            "n_subjects_available": len(subjects),
            "first_subject": first_subject,
            "primary_channel": primary,
            "frontal_roi": roi,
            "validation": validation,
            "outputs": [
                "outputs/ds002721_v2_0/channel_inventory.csv",
                "outputs/ds002721_v2_0/rating_inventory.csv",
                "outputs/ds002721_v2_0/frontal_roi_definition.csv",
                "derived/ds002721_dataset_manifest.csv",
            ],
        }
        (OUTPUT_DIR / "inspect_summary.json").write_text(
            json.dumps(json_ready(summary_json), indent=2), encoding="utf-8"
        )
        print("DS002721 inspect phase complete")
        print("Subjects available:", len(subjects))
        print("Primary channel:", primary)
        print("Frontal ROI:", ", ".join(roi))
        return summary_json

    if phase == "smoke-one":
        one_subject_trials = run_subjects(manifest, [first_subject], primary, roi)
        smoke_one_subject(one_subject_trials)
        summary_json = {
            "phase": "smoke-one",
            "dataset": "DS002721",
            "snapshot_tag": SNAPSHOT_TAG,
            "subject_id": first_subject,
            "n_trials": int(len(one_subject_trials)),
            "primary_channel": primary,
            "frontal_roi": roi,
            "validation": validation,
            "missing_values": one_subject_trials[
                ["beta_log_power", "frontal_roi_beta_log_power", "energy", "tension", "pleasantness"]
            ]
            .isna()
            .sum()
            .to_dict(),
            "outputs": [
                "outputs/ds002721_v2_0/smoke_one_subject_trial_features.csv",
                "outputs/ds002721_v2_0/00_smoke_beta_by_trial.png",
                "outputs/ds002721_v2_0/00_smoke_energy_by_trial.png",
                "outputs/ds002721_v2_0/00_smoke_beta_vs_energy.png",
            ],
        }
        (OUTPUT_DIR / "smoke_one_summary.json").write_text(
            json.dumps(json_ready(summary_json), indent=2), encoding="utf-8"
        )
        print("DS002721 one-subject smoke phase complete")
        print("Subject:", first_subject)
        print("Trials:", len(one_subject_trials))
        print("Missing values:", summary_json["missing_values"])
        return summary_json

    if phase == "smoke-three":
        smoke_subjects = subjects[:3]
        smoke_three = run_subjects(manifest, smoke_subjects, primary, roi)
        smoke_three_effects = subject_effects(smoke_three)
        smoke_three_effects.to_csv(OUTPUT_DIR / "smoke_three_subject_effects.csv", index=False)
        summary_json = {
            "phase": "smoke-three",
            "dataset": "DS002721",
            "snapshot_tag": SNAPSHOT_TAG,
            "subjects": smoke_subjects,
            "n_trials": int(len(smoke_three)),
            "primary_channel": primary,
            "frontal_roi": roi,
            "validation": validation,
            "effects": smoke_three_effects.to_dict(orient="records"),
            "outputs": [
                "outputs/ds002721_v2_0/smoke_three_subject_effects.csv",
            ],
        }
        (OUTPUT_DIR / "smoke_three_summary.json").write_text(
            json.dumps(json_ready(summary_json), indent=2), encoding="utf-8"
        )
        print("DS002721 three-subject smoke phase complete")
        print("Subjects:", ", ".join(smoke_subjects))
        print("Trials:", len(smoke_three))
        return summary_json

    if phase == "cache-batch":
        todo = missing_subjects(subjects)
        selected = todo[:max_new_subjects] if max_new_subjects is not None else todo
        if selected:
            batch_trials = run_subjects(manifest, selected, primary, roi)
            n_trials = int(len(batch_trials))
        else:
            n_trials = 0
        remaining = missing_subjects(subjects)
        summary_json = {
            "phase": "cache-batch",
            "dataset": "DS002721",
            "snapshot_tag": SNAPSHOT_TAG,
            "processed_subjects": selected,
            "n_processed_trials": n_trials,
            "n_cached_subjects": len(cached_subjects()),
            "remaining_subjects": remaining,
            "n_remaining_subjects": len(remaining),
        }
        (OUTPUT_DIR / "cache_batch_summary.json").write_text(
            json.dumps(json_ready(summary_json), indent=2), encoding="utf-8"
        )
        print("DS002721 cache-batch phase complete")
        print("Processed:", ", ".join(selected) if selected else "none")
        print("Cached subjects:", len(cached_subjects()))
        print("Remaining subjects:", len(remaining))
        return summary_json

    if phase == "finalize-cache":
        return finalize_outputs(validation, subjects, primary, roi)

    if phase != "full":
        raise ValueError(f"Unknown phase: {phase}")

    todo = missing_subjects(subjects)
    if todo:
        run_subjects(manifest, todo, primary, roi)
    return finalize_outputs(validation, subjects, primary, roi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DS002721 V2.0 in ordered phases.")
    parser.add_argument(
        "--phase",
        choices=["inspect", "smoke-one", "smoke-three", "cache-batch", "finalize-cache", "verify-existing", "full"],
        default="smoke-one",
        help="Execution phase. Default avoids full 31-subject download.",
    )
    parser.add_argument(
        "--max-new-subjects",
        type=int,
        default=None,
        help="For cache-batch, process at most this many subjects without an existing cache.",
    )
    args = parser.parse_args()
    if args.phase == "verify-existing":
        verify_existing_outputs()
    else:
        main(args.phase, max_new_subjects=args.max_new_subjects)
