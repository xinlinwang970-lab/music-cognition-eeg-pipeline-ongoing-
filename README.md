# Music Cognition EEG Pipeline V2.0

This repository contains a reproducible research pipeline for testing whether a frozen frontal beta EEG feature shows consistent relationships with music engagement and affective ratings across open music EEG datasets.

The repository is organized for methodological review. It preserves the current analysis scripts and documents the scientific boundary of the project: NMED-E is used as the discovery baseline for a frontal beta-engagement pattern, while DS002721 is used as a cross-dataset transportability test. The present evidence supports a paradigm-dependent candidate relationship, not a universal EEG marker for music emotion, flow, or BCI.

## Research Question

Can a prespecified frontal beta feature, defined as 15-30 Hz log power over frontal/anterior sensors, show reproducible engagement-related structure in NMED-E and transport to independent trial-level affective ratings in DS002721?

## Current Result

In NMED-E, the strongest current result is the fast60 frontal beta contrast: beta-engagement coupling is more positive for the original stimulus than for the control stimulus in most included participants. This pattern is supported by the frontal ROI and leave-one-subject-out group engagement checks.

In DS002721, the same transported frontal beta feature does not show a stable positive relationship with Energy, Tension, or Pleasantness. Energy is the primary transportability target and remains close to zero. This negative/null cross-dataset result is part of the main conclusion.

## Repository Map

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── nmede.yaml
│   └── ds002721.yaml
├── docs/
│   ├── METHODS.md
│   ├── DATASETS.md
│   ├── LIMITATIONS.md
│   └── RESULTS_SUMMARY.md
├── metadata/
│   ├── construct_map.csv
│   └── feature_specs.yaml
├── results/
│   ├── figures/
│   └── summary/
├── scripts/
│   └── analysis/
├── tests/
├── run_full_v1.py
├── run_v1_1_mechanism_robustness.py
└── run_ds002721_v2_0.py
```

Raw EEG data are intentionally not part of the public repository. See [docs/DATASETS.md](docs/DATASETS.md) and [data/README.md](data/README.md).

Two legacy helper scripts, `toy1_1_nmede_diagnostics.py` and `toy2_nmede_pilot.py`, are retained because the NMED-E V1/V1.1 scripts currently import shared helper functions from them. They are not presented as separate claims.

## Main Analyses

1. `run_full_v1.py`
   Runs the NMED-E discovery baseline using E11, approximate Fz, beta 15-30 Hz log power, 5 s windows, and original-minus-control subject-level engagement correlations.

2. `run_v1_1_mechanism_robustness.py`
   Reuses the NMED-E baseline outputs and evaluates timescale decomposition, predefined frontal ROI robustness, audio-control availability, and leave-one-subject-out group engagement.

3. `run_ds002721_v2_0.py`
   Transports the frozen frontal beta feature to OpenNeuro DS002721 trial-level music ratings. The default command is a one-subject smoke run to avoid unintentional full data downloads.

## Installation

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Verification

Run unit and repository smoke tests:

```powershell
python -m pytest tests -q
```

Check the existing DS002721 generated outputs, if the cached outputs are present locally:

```powershell
python run_ds002721_v2_0.py --phase verify-existing
```

Run a minimal DS002721 smoke analysis. This may download one OpenNeuro subject if the local cache is absent:

```powershell
python run_ds002721_v2_0.py --phase smoke-one
```

The NMED-E scripts require the NMED-E MATLAB files to be placed locally under `data/`. They are not redistributed here:

```powershell
python run_full_v1.py
python run_v1_1_mechanism_robustness.py
```

## Reproducibility Notes

- The transported neural feature is fixed as frontal beta 15-30 Hz log power.
- NMED-E uses window-level continuous engagement trajectories.
- DS002721 uses one beta value per music trial, matching trial-level affect ratings.
- The DS002721 command defaults to `smoke-one`, not a full 31-subject run, to avoid large automatic downloads.
- The public `results/` directory contains curated summaries and figures for review, while regenerated full outputs are ignored.

## Scientific Boundary

This repository does not claim a flow biomarker, clinical marker, causal mechanism, universal music-emotion feature, or real-time BCI-ready decoder. The present result is narrower: frontal beta in NMED-E is a candidate engagement-related feature whose transportability to DS002721 affect ratings is weak or absent.

## Before Public Release

Before pushing this repository to GitHub, confirm:

- GitHub account and target repository name.
- Final license choice.
- Whether any generated CSVs or figures should be withheld.
- Whether the repository should be public immediately or shared privately with Prof. Joydeep Bhattacharya first.
