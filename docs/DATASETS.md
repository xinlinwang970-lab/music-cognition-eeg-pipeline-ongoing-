# Datasets

## NMED-E

NMED-E is used as the discovery baseline. The local analysis expects processed MATLAB files and channel coordinates under `data/`:

- `CleanEEG_stim22.mat`
- `CleanEEG_stim23.mat`
- `CleanCB_All.mat`
- `CleanRatings_All.mat`
- `GSN-HydroCel-129.sfp`

The included NMED-E sample in the current analysis has 22 participants with both original and control EEG plus continuous engagement data.

The NMED-E source publication describes the participants as adult musicians with formal classical music training. The task used a complete naturalistic classical music excerpt and a temporally manipulated control stimulus. The current public pipeline treats NMED-E as a musically trained listening sample, but it does not model individual training years or use expertise as a within-dataset predictor.

NMED-E contributes the continuous engagement construct. The primary NMED-E contrast is:

```text
Original beta-engagement rho - Control beta-engagement rho
```

The actual stimulus audio files are not present locally. Acoustic envelope control is therefore not computed, and no substitute audio is used.

## DS002721

DS002721 is an OpenNeuro music EEG dataset used as an external transportability test. The current script targets snapshot `1.0.3` and music runs 2-5.

The DS002721 documentation describes 31 healthy adult participants listening to 40 short affective music clips from film-score excerpts and reporting induced emotional responses. The released participant table used here contains age and sex, but not a public musical expertise measure that can be directly compared with NMED-E.

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

## Participant and Task Context

The two datasets differ in participant background, task context, and available behavioral constructs:

- NMED-E: musically trained adult listeners, longer naturalistic classical music listening, original-versus-control stimulus contrast, and continuous engagement trajectories.
- DS002721: healthy adult listeners, short affective music clips, trial-level affect ratings, and demographic metadata without a public musical expertise variable.

This difference is scientifically useful but also limiting. The current cross-dataset result can motivate an expertise hypothesis, but it cannot determine whether musical expertise moderates the frontal beta-engagement relationship because task design, stimulus structure, behavioral construct, and participant metadata also differ. A future dataset or experiment should measure expertise, training history, musical motivation, and task context by design.

## Dataset Sources

- NMED-E source publication: <https://doi.org/10.1111/ejn.16324>
- DS002721 OpenNeuro dataset: <https://openneuro.org/datasets/ds002721>

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
