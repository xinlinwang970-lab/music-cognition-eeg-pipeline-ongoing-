# NMED-E MuQ-MuLan Semantic Pilot

This directory is an exploratory extension of the Music Cognition EEG Pipeline. It tests whether high-dimensional music embeddings from MuQ-MuLan explain NMED-E engagement and frontal beta dynamics beyond acoustic and time-only baselines.

The current experiment is original-only:

```text
Elgar original audio, 5 s windows aligned to the existing NMED-E window table
-> time-only, acoustic, and MuQ-MuLan features
-> NMED-E original-condition EEG beta and engagement windows
-> time-resolved encoding / prediction models
```

The public repository includes code, configuration, tests, and a curated result snapshot. It does not include restricted/copyrighted stimulus audio, MuQ-MuLan model weights, local dependency folders, raw EEG, derived window-level input tables, or full feature matrices.

## Scientific Boundary

This is not a strict reproduction of the Nature Communications fMRI + Google MuLan/MusicLM study. It is an EEG adaptation of the same method logic:

```text
music audio -> high-dimensional semantic representation -> brain / behavior signal
```

NMED-E stimulus audio files are not redistributed here. The NMED-E README states that users should contact the dataset author for stimulus audio access. For a pilot, you may place a manually aligned copy of the intact Elgar excerpt locally, but the exact NMED-E stimulus is preferred for any serious result. The analysis expects 95 complete 5 s windows from 0-475 s.

## Folder Layout

```text
.
├── configs/
│   └── experiment.yaml
├── data/
│   ├── external/
│   │   └── README.md
│   └── stimuli/
│       └── README.md
├── results/
│   ├── metrics/
│   └── figures/
├── docs/
│   └── RESULTS_SUMMARY.md
├── scripts/
│   ├── 00_prepare_nmed_targets.py
│   ├── 01_extract_acoustic_features.py
│   ├── 02_extract_muq_mulan_embeddings.py
│   ├── 03_run_original_semantic_pilot.py
│   ├── 04_make_time_features.py
│   └── common.py
└── tests/
```

## Setup

Base acoustic and analysis dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
```

Optional MuQ-MuLan dependencies:

```powershell
python -m pip install -r requirements-muq.txt
```

If you install the optional dependencies into a local vendor folder instead of an activated environment, set `PYTHONPATH` before running the MuQ script:

```powershell
python -m pip install --target vendor\muqdeps -r requirements-muq.txt
$env:PYTHONPATH = (Resolve-Path 'vendor\muqdeps').Path
$env:HF_HOME = (Join-Path (Resolve-Path '.').Path 'models\hf')
```

`MuQ-MuLan-large` weights are released under CC-BY-NC-4.0. Use them only in compatible non-commercial research settings.

## Data Inputs

The project expects the derived NMED-E window table at:

```text
data/external/nmed_window_level_data.csv
```

That table is not included in the public repository. It should be regenerated from the main NMED-E pipeline or copied from a local derived output.

The original stimulus audio should be placed manually at:

```text
data/stimuli/nmed_e_original.wav
```

Do not commit or redistribute the audio file.

## Curated Result Snapshot

The public snapshot keeps only compact review artifacts:

```text
results/metrics/model_comparison_summary.csv
results/metrics/*_metrics.json
results/figures/*_prediction.png
docs/RESULTS_SUMMARY.md
```

The strongest control result is that `beta_slow60` is already well explained by the time-only baseline (`R2 = 0.746`, Spearman rho = `0.869`). This means the beta slow trend is strongly coupled to stimulus time/course structure. MuQ-MuLan and acoustic features show high rank correlations for slow beta, but their cross-validated R2 is negative in the current setup, so they should not be interpreted as providing evidence beyond the time-only baseline.

## Run The First Experiment

Prepare the NMED-E original-condition group time series:

```powershell
python scripts/00_prepare_nmed_targets.py
```

Extract low-level acoustic features:

```powershell
python scripts/01_extract_acoustic_features.py --audio data/stimuli/nmed_e_original.wav
```

Extract MuQ-MuLan embeddings:

```powershell
python scripts/02_extract_muq_mulan_embeddings.py --audio data/stimuli/nmed_e_original.wav --device cpu
```

Create the time-only baseline. This feature table contains only chronological predictors: elapsed time fraction plus centered linear, quadratic, and cubic time terms. It tests how much of the EEG or engagement target can be explained by the stimulus timeline alone, without acoustic, MIR, or semantic information.

```powershell
python scripts/04_make_time_features.py
```

Run the time-only baseline:

```powershell
python scripts/03_run_original_semantic_pilot.py --feature-csv outputs/features/original_time_features.csv --feature-kind time_only --target engagement_fast60
```

Run the semantic pilot with MuQ-MuLan features:

```powershell
python scripts/03_run_original_semantic_pilot.py --feature-csv outputs/features/original_muq_mulan_embeddings.csv --feature-kind muq_mulan --target engagement_fast60 --pca-components 8
```

Run the acoustic baseline:

```powershell
python scripts/03_run_original_semantic_pilot.py --feature-csv outputs/features/original_acoustic_features.csv --feature-kind acoustic --target engagement_fast60
```

## Interpretation

Useful first questions:

- Do MuQ-MuLan embeddings predict group engagement better than low-level acoustic features?
- Do music semantics predict frontal beta dynamics?
- Do acoustic or MuQ-MuLan features explain frontal beta dynamics beyond the time-only baseline?

Any positive result should be treated as a pilot because this original-only experiment lacks the NMED-E control-stimulus audio and currently shows strong time-course confounding for slow beta dynamics.
