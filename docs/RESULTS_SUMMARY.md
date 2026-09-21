# Results Summary

## NMED-E Discovery Baseline

The NMED-E baseline finds a positive original-minus-control beta-engagement pattern in the observed sample, but the raw primary endpoint remains uncertain at the group level.

Full V1 primary beta raw result:

| Endpoint | N | Mean delta rho | Median | Positive | Sign-flip p | Bootstrap mean CI |
|---|---:|---:|---:|---:|---:|---|
| beta raw | 22 | 0.0937 | 0.0791 | 15/22 | 0.1821 | [-0.0389, 0.2214] |

V1.1 timescale robustness identifies fast60 as the clearest current NMED-E endpoint:

| Endpoint | N | Mean delta rho | Median | Positive | Sign-flip p | Bootstrap mean CI |
|---|---:|---:|---:|---:|---:|---|
| fast60 | 22 | 0.0992 | 0.1106 | 17/22 | 0.0024 | [0.0445, 0.1564] |

The frontal ROI and leave-one-subject-out group engagement checks point in the same direction:

| Endpoint | N | Mean delta rho | Median | Positive | Sign-flip p | Bootstrap mean CI |
|---|---:|---:|---:|---:|---:|---|
| ROI fast60 | 22 | 0.1136 | 0.1051 | 17/22 | 0.0005 | [0.0600, 0.1713] |
| LOSO group engagement | 22 | 0.1117 | 0.1295 | 17/22 | 0.0016 | [0.0536, 0.1698] |

## DS002721 Transportability Test

The same frontal beta feature does not transport as a positive Energy relationship in DS002721.

| Endpoint | N | Mean rho | Median | Positive | Sign-flip p | Bootstrap mean CI |
|---|---:|---:|---:|---:|---:|---|
| Energy, Fz | 31 | -0.0183 | -0.0228 | 14/31 | 0.5457 | [-0.0768, 0.0398] |
| Tension, Fz | 31 | -0.0456 | -0.0528 | 9/31 | 0.1028 | [-0.0984, 0.0052] |
| Pleasantness, Fz | 31 | -0.0027 | -0.0418 | 11/31 | 0.9250 | [-0.0556, 0.0545] |

ROI Energy remains weak:

| Endpoint | N | Mean rho | Median | Positive | Sign-flip p | Bootstrap mean CI |
|---|---:|---:|---:|---:|---:|---|
| Energy, ROI | 31 | -0.0131 | -0.0134 | 15/31 | 0.5968 | [-0.0613, 0.0317] |

## Interpretation

The most conservative interpretation is:

```text
NMED-E supports a candidate frontal beta-engagement pattern.
DS002721 does not support strong cross-dataset transport to trial-level affect ratings.
```

This boundary is important. The result is useful because it narrows the next research question toward expertise, task context, stimulus structure, and construct validity rather than overclaiming a universal EEG marker.

The dataset contrast is relevant for future work on participant background and task context: NMED-E is a musically trained naturalistic-listening sample, whereas DS002721 is a healthy-adult affective-listening dataset without public musical expertise labels. This contrast should be treated as an unresolved interpretive boundary, not as evidence that expertise alone explains the cross-dataset difference.
