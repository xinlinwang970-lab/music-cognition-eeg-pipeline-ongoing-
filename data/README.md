# Local Data Directory

Place dataset files here when running the analyses locally. Raw EEG and restricted dataset files are intentionally excluded from Git.

Expected local NMED-E files:

- `CleanEEG_stim22.mat`
- `CleanEEG_stim23.mat`
- `CleanCB_All.mat`
- `CleanRatings_All.mat`
- `GSN-HydroCel-129.sfp`

Expected local DS002721 layout:

- `data/ds002721/openneuro_file_manifest.csv`
- `data/ds002721/sub-*/eeg/*_events.tsv`
- `data/ds002721/sub-*/eeg/*_channels.tsv`
- `data/ds002721/sub-*/eeg/*_eeg.edf`

The DS002721 script can fetch metadata and subject files from OpenNeuro when network access is available. Do not commit downloaded EEG data.
