# Research Extensions

## Purpose

This document lists research extensions suggested by the current analysis. These are not claims made by the repository and are not required to reproduce the reported results. They identify variables that remain unresolved in the present open-data comparison.

## Dataset and Construct Boundaries

The current result is best interpreted as a dataset- and construct-boundary finding:

```text
NMED-E supports a candidate frontal beta-engagement pattern.
DS002721 does not support strong transport to trial-level affect ratings.
```

The datasets also differ in participant background, task context, stimulus structure, and available metadata. NMED-E is a musically trained naturalistic-listening sample, whereas DS002721 is a healthy-adult affective-listening dataset without public musical expertise labels. This contrast motivates further study, but the present repository does not isolate musical expertise, task context, or stimulus structure causally.

## Possible Extensions

### Stimulus and Acoustic Controls

The NMED-E audio stimuli are not redistributed in this repository. A useful extension would test whether the NMED-E beta-engagement relationship remains after controlling for acoustic features such as loudness, spectral centroid, onset density, harmonic change, and stimulus envelope.

### High-Dimensional Music Representations

Foundation-model audio embeddings could be used to test whether higher-level musical structure explains engagement and EEG dynamics beyond low-level acoustic features. Candidate models include MuLan-style music-language embeddings, CLAP-style audio-text embeddings, and audio-only music representation models.

### Participant Background and Expertise

Future datasets should measure musical expertise directly, including training history, current practice, listening experience, familiarity, and music-reward sensitivity. This would allow expertise to be modeled as a participant-level moderator rather than inferred indirectly from dataset identity.

### Task Context

The contrast between naturalistic engagement ratings and short-clip affect ratings suggests that task context should be modeled explicitly. Future work could compare naturalistic listening, short affective listening, active performance, and controlled stimulus-manipulation paradigms.

### Replication and Preregistration

A stronger confirmatory study would prespecify the EEG feature, behavioral construct, stimulus-control strategy, participant variables, and statistical tests before data collection or model tuning.

## Interpretation Rule

Extensions should avoid tuning the current feature until a positive result appears. The scientific value of the present repository is that it makes the current positive and null results visible together, with their construct and dataset limits stated explicitly.
