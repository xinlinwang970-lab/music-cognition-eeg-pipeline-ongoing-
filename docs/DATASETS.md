# Datasets

## NMED-E

NMED-E is used as the discovery baseline. The local analysis expects processed MATLAB files and channel coordinates under `data/`:

- `CleanEEG_stim22.mat`
- `CleanEEG_stim23.mat`
- `CleanCB_All.mat`
- `CleanRatings_All.mat`
- `GSN-HydroCel-129.sfp`

The included NMED-E sample in the current analysis has 22 participants with both original and control EEG plus continuous engagement data.

NMED-E contributes the continuous engagement construct. The primary NMED-E contrast is:

```text
Original beta-engagement rho - Control beta-engagement rho
```

The actual stimulus audio files are not present locally. Acoustic envelope control is therefore not computed, and no substitute audio is used.

## DS002721

DS002721 is an OpenNeuro music EEG dataset used as an external transportability test. The current script targets snapshot `1.0.3` and music runs 2-5.

The analysis uses:

- EDF EEG files.
- BIDS events files.
- Channels files.
- Trial-level ratings parsed from events.

The current full generated output contains:

- 31 subjects.
- 1240 trial rows.
- 31 subject-level effect rows.

DS002721 contributes trial-level affective targets:

- Energy: primary cross-dataset target.
- Tension: secondary target.
- Pleasantness: construct contrast.

These targets are not treated as continuous engagement. DS002721 is not a direct replication dataset for the NMED-E engagement construct.

## Data Availability Policy

Raw EEG, MATLAB data files, EDF files, and downloaded archives are not committed. Users should obtain the datasets from their original sources and follow the relevant licenses, access terms, and citation requirements.

The repository keeps only:

- Analysis scripts.
- Metadata and feature specifications.
- Documentation.
- Small curated result summaries.
- Selected generated figures.

## Local Data Layout

```text
data/
├── CleanEEG_stim22.mat
├── CleanEEG_stim23.mat
├── CleanCB_All.mat
├── CleanRatings_All.mat
├── GSN-HydroCel-129.sfp
└── ds002721/
    ├── dataset_description.json
    ├── participants.tsv
    └── sub-*/eeg/
```

The `.gitignore` file excludes the raw local contents of `data/`.
