# Music Cognition EEG Pipeline V2.0

This repository contains a reproducible research pipeline for testing whether a prespecified frontal beta EEG feature shows consistent relationships with music engagement and affective ratings across open music EEG datasets.

The repository is organized for methodological review. It preserves the current analysis scripts and documents the scientific boundary of the project: NMED-E is used as the discovery baseline for a frontal beta-engagement pattern, while DS002721 is used as a cross-dataset transportability test. The comparison evaluates construct specificity, dataset transportability, and methodological boundaries rather than presenting a universal EEG marker for music emotion, flow, or BCI.

NMED-E is a musically trained naturalistic-listening sample, whereas DS002721 is a healthy-adult affective-listening dataset without public musical expertise labels. This participant and task contrast is treated as an interpretive boundary and a candidate moderator for future work, not as evidence that musical expertise explains the cross-dataset difference.

## Research Question

Can a prespecified frontal beta feature, defined as 15-30 Hz log power over frontal/anterior sensors, show reproducible engagement-related structure in NMED-E and transport to independent trial-level affective ratings in DS002721?

The repository does not treat engagement, affect, expertise, and musical experience as interchangeable constructs. Instead, it uses the current open-data analyses to clarify where a candidate EEG relationship appears, where it does not transport, and which factors remain unresolved.

## Current Result

In NMED-E, the strongest current result is the fast60 frontal beta contrast: beta-engagement coupling is more positive for the original stimulus than for the control stimulus in most included participants. This pattern is supported by the frontal ROI and leave-one-subject-out group engagement checks.

In DS002721, the same transported frontal beta feature does not show a stable positive relationship with Energy, Tension, or Pleasantness. Energy is the primary transportability target and remains close to zero. This negative/null cross-dataset result is part of the main conclusion and highlights the need for designs that measure participant background, musical expertise, stimulus structure, and behavioral construct explicitly.

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
│   ├── RESULTS_SUMMARY.md
│   └── RESEARCH_EXTENSIONS.md
├── metadata/
│   ├── construct_map.csv
│   └── feature_specs.yaml
├── results/
│   ├── figures/
│   └── summary/
├── experiments/
│   └── nmed_mulan_semantic_pilot/
├── scripts/
│   └── analysis/
├── tests/
├── run_full_v1.py
├── run_v1_1_mechanism_robustness.py
└── run_ds002721_v2_0.py
```

Raw EEG data are intentionally not part of the public repository. See [docs/DATASETS.md](docs/DATASETS.md) and [data/README.md](data/README.md).

Two legacy helper scripts, `toy1_1_nmede_diagnostics.py` and `toy2_nmede_pilot.py`, are retained because the NMED-E V1/V1.1 scripts currently import shared helper functions from them. They are not presented as separate claims.

The `experiments/nmed_mulan_semantic_pilot/` directory contains an exploratory original-only extension that compares time-only, acoustic, and MuQ-MuLan feature models on NMED-E original-condition dynamics. It is included as a methodological extension and negative-control result, not as a claim that MuQ-MuLan currently explains EEG beyond the stimulus timeline.

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

See [docs/RESEARCH_EXTENSIONS.md](docs/RESEARCH_EXTENSIONS.md) for neutral research extensions suggested by the current dataset and construct boundaries.

## Repository Hygiene Checklist

Before publishing or updating the repository, confirm:

- The selected license is appropriate for the analysis code and documentation.
- Dataset-specific citation and access requirements are documented.
- Raw EEG, downloaded archives, restricted data, local caches, and private working exports are not staged.
- Curated CSV summaries and figures are small enough for GitHub and do not redistribute restricted source data.
- Public documentation states the construct and dataset boundaries of the current result.
