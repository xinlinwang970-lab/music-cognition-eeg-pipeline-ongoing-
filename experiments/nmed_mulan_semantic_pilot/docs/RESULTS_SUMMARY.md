# NMED-E MuQ-MuLan Semantic Pilot Results

## Purpose

This exploratory extension tests whether MuQ-MuLan music embeddings explain NMED-E original-condition EEG and engagement dynamics beyond lower-level acoustic features and a time-only baseline.

The time-only baseline is important because slow EEG dynamics can track the stimulus timeline, section order, fatigue, arousal drift, or broad work-level structure even when a music feature model is not adding specific explanatory information.

## Feature Sets

| Feature set | Description |
|---|---|
| time_only | Elapsed time fraction plus centered linear, quadratic, and cubic time terms |
| acoustic | RMS, spectral, band-energy, chroma proxy, flux, and MFCC-style features |
| muq_mulan | 512-dimensional MuQ-MuLan audio embeddings plus embedding norm, reduced with PCA in the current encoding model |

## Key Results

| Target | Feature set | R2 | Spearman rho | Interpretation |
|---|---|---:|---:|---|
| engagement_fast60 | time_only | -7.330 | -0.015 | Timeline alone does not explain fast engagement residuals |
| engagement_fast60 | acoustic | -1.276 | -0.057 | Low-level acoustics do not explain fast engagement residuals |
| engagement_fast60 | muq_mulan | -0.089 | 0.120 | MuQ-MuLan is slightly positive but weak and not decisive |
| beta_fast60 | time_only | -0.008 | -0.086 | Timeline alone does not explain fast beta residuals |
| beta_fast60 | acoustic | -0.000 | -0.034 | Low-level acoustics do not explain fast beta residuals |
| beta_fast60 | muq_mulan | -0.072 | -0.056 | MuQ-MuLan does not explain fast beta residuals |
| beta_mean | time_only | 0.576 | 0.721 | Raw beta contains a strong time-course component |
| beta_mean | acoustic | -8.075 | 0.762 | High rank correlation but poor calibrated prediction |
| beta_mean | muq_mulan | -2.699 | 0.645 | High rank correlation but poor calibrated prediction |
| beta_slow60 | time_only | 0.746 | 0.869 | Slow beta is strongly explained by stimulus timeline |
| beta_slow60 | acoustic | -2.853 | 0.784 | High rank correlation, but no R2 gain over time-only |
| beta_slow60 | muq_mulan | -2.279 | 0.860 | High rank correlation, but no R2 gain over time-only |

## Interpretation

The most important result is the time-only baseline. It shows that `beta_slow60` and `beta_mean` are strongly aligned with the stimulus timeline. This makes the slow beta result scientifically interesting, but it also means that MuQ-MuLan or acoustic correlations should not be interpreted as specific evidence of music-semantic encoding unless they outperform time-only controls.

At the current stage, the results support a narrower conclusion:

```text
NMED-E frontal beta slow dynamics appear strongly coupled to the temporal course of the intact Elgar stimulus.
Current acoustic and MuQ-MuLan models do not yet demonstrate additional predictive value beyond the time-only baseline.
```

## Next Controls

The next step is to add interpretable MIR and musicological controls, especially loudness, spectral brightness, onset density, harmonic/tonal change, section-boundary novelty, and manually annotated phrase or form events. These controls can test whether the time-course effect reflects broad chronological drift or specific long-range musical structure.
