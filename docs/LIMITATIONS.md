# Limitations

## Scientific Limits

This project does not currently support claims of:

- A universal music-emotion EEG marker.
- A flow biomarker.
- A causal neural mechanism.
- A clinical marker.
- A real-time BCI-ready feature.
- A generalized decoder for engagement or affect.

The strongest current statement is narrower: a prespecified frontal beta feature shows an internally consistent NMED-E engagement-related pattern, especially in fast60, frontal ROI, and leave-one-subject-out group engagement checks, but does not transport cleanly to DS002721 affective ratings.

## NMED-E Limits

- EEG and continuous engagement ratings are not sufficient to establish causality.
- The actual stimulus audio files are not available locally, so acoustic controls remain incomplete.
- E11 is treated as approximate frontal midline/Fz based on the available sensor montage, not as an exact 10-20 electrode.
- The V1.1 analyses are robustness checks around a staged exploratory-to-confirmatory pipeline, not an independently preregistered confirmatory study.

## DS002721 Limits

- DS002721 has trial-level affective ratings, not continuous engagement trajectories.
- Energy, Tension, and Pleasantness are not engagement synonyms.
- Raw EDF data are used without a full artifact-rejection pipeline in the current version.
- Some event and stimulus alignments require practical parsing choices from BIDS events.
- Monte Carlo sign-flip inference is used for large-N group tests where exact enumeration is not practical.

## Cross-Dataset Interpretation

The DS002721 negative/null result is central. It means the current evidence is better framed as task- and construct-dependent than as a universal frontal beta response to music.

The two datasets also differ in participant background, task context, and the availability of musical expertise metadata. NMED-E can be interpreted as a musically trained naturalistic-listening sample, whereas DS002721 is a healthy-adult affective-listening sample without a public expertise label. The present pipeline therefore cannot test expertise as a moderator. It can only motivate that question by showing that a feature discovered in one music-listening context does not automatically transport to another context with different stimuli and behavioral targets.

The next scientifically useful step is not to tune DS002721 until it becomes positive. A stronger next step would be to acquire or reconstruct valid acoustic controls for NMED-E and test whether the NMED-E fast60 beta-engagement relationship remains after controlling for stimulus structure.
