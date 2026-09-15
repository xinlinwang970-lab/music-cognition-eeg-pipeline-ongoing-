# Methods

## Overview

The pipeline uses a staged analysis design. NMED-E is treated as the discovery baseline for a frontal beta-engagement relationship. DS002721 is treated as an external transportability test of the same neural feature in a different music EEG dataset with different task structure and rating constructs.

The central feature is frozen before cross-dataset transport:

```text
frontal beta log power, 15-30 Hz
```

The feature is not retuned on DS002721 outcomes.

## Feature Extraction

Frequency-domain power is estimated with Welch PSD:

- Band: 15-30 Hz.
- Representation: log10 absolute band power.
- Window: Hann.
- Detrend: constant.
- Welch segment length: 2 s.

The scripts use the same beta-band definition in NMED-E and DS002721. Temporal aggregation differs by dataset because the behavioral measurements differ.

## NMED-E Discovery Baseline

NMED-E uses continuous engagement trajectories. EEG and engagement are aligned into 5 s non-overlapping windows. For each participant and condition, the analysis computes:

```text
rho = Spearman(beta_window, engagement_window)
delta_rho = rho_original - rho_control
```

Group-level summaries are then computed over participant-level `delta_rho` values:

- Mean and median.
- Standard deviation and IQR.
- Number of positive effects.
- Exact paired sign-flip test.
- Bootstrap confidence intervals.

The V1.1 robustness layer evaluates:

- Raw, slow30, fast30, slow60, and fast60 timescale components.
- A predefined frontal/anterior ROI.
- Leave-one-subject-out group engagement.
- Audio-control availability.

The audio-control analysis is explicitly marked unavailable because the actual original and control stimulus audio files are not present locally.

## DS002721 Cross-Dataset Transportability

DS002721 uses trial-level affective ratings. The pipeline therefore computes one beta value per music trial rather than creating repeated 5 s pseudo-observations for the same rating.

For each subject:

```text
rho(beta_trial, Energy_trial)
rho(beta_trial, Tension_trial)
rho(beta_trial, Pleasantness_trial)
```

Group summaries are computed over subject-level correlations. Energy is the primary transportability target because it is the closest affective-activation construct, but it is not treated as equivalent to NMED-E engagement.

For DS002721, large-N sign-flip inference uses a fixed-seed Monte Carlo procedure when exact enumeration would be too expensive.

## Reproducibility Commands

Run tests:

```powershell
python -m pytest tests -q
```

Verify existing DS002721 outputs:

```powershell
python run_ds002721_v2_0.py --phase verify-existing
```

Run the minimal DS002721 smoke phase:

```powershell
python run_ds002721_v2_0.py --phase smoke-one
```

Run NMED-E locally after placing the required files under `data/`:

```powershell
python run_full_v1.py
python run_v1_1_mechanism_robustness.py
```

## Implementation Boundary

The public-release cleanup does not change the core analysis logic. It organizes documentation, dependency metadata, ignored files, curated result summaries, and smoke tests around the existing scripts.
