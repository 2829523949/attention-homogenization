# RealBind DaH oracle pilot

Status: **DAH_STRONG_SIGNAL**

## Primary comparison

| Variant | Left | Right | Composition | Δ vs Original | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.460000 | 0.520000 | 0.490000 | +0.000000 | NA |
| Blur-only | 0.590000 | 0.600000 | 0.595000 | +0.105000 | 0.00106502 |
| Decompose-only | 0.530000 | 0.630000 | 0.580000 | +0.090000 | 0.024746 |
| Handoff+TargetLock | 0.570000 | 0.650000 | 0.610000 | +0.120000 | 0.00427241 |
| Full DaH | 0.590000 | 0.650000 | 0.620000 | +0.130000 | 0.00218624 |

Primary generation-quality counts (pooled):

| Variant | Invalid | Format violation | Truncation | Hit max tokens |
| --- | ---: | ---: | ---: | ---: |
| Original | 3 | 3 | 0 | 0 |
| Blur-only | 1 | 1 | 0 | 0 |
| Decompose-only | 8 | 8 | 0 | 0 |
| Handoff+TargetLock | 3 | 3 | 0 | 0 |
| Full DaH | 3 | 3 | 0 | 0 |

## Ablation comparisons

Positive Δ means the comparison variant is more accurate on the same 200 cases.

| Comparison | Baseline variant | Comparison variant | Baseline | Comparison | Δ accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| background/non-candidate defocus | Original | Blur-only | 0.490000 | 0.595000 | +0.105000 |
| two-stage decomposition and explicit handoff | Original | Decompose-only | 0.490000 | 0.580000 | +0.090000 |
| target visual restriction | Decompose-only | Handoff+TargetLock | 0.580000 | 0.610000 | +0.030000 |
| pair-defocus Stage 1 | Handoff+TargetLock | Full DaH | 0.610000 | 0.620000 | +0.010000 |
| full method versus pair blur | Blur-only | Full DaH | 0.595000 | 0.620000 | +0.025000 |

## Decomposed pipelines

| Variant | Bind Acc | Attr Acc given correct bind | Final Acc |
| --- | ---: | ---: | ---: |
| Decompose-only | 0.780000 | 0.711538 | 0.580000 |
| Handoff+TargetLock | 0.780000 | 0.743590 | 0.610000 |
| Full DaH | 0.820000 | 0.719512 | 0.620000 |

Wrong-binding final-correct diagnostics (not counted as binding success):

| Variant | Wrong bind but final correct | Of which same-color accidental |
| --- | ---: | ---: |
| Decompose-only | 5 | 0 |
| Handoff+TargetLock | 6 | 0 |
| Full DaH | 6 | 0 |

## Strong-path repair

| Variant | Strong Failure Rescue | Original Success Preservation | Failure on original strong-ready |
| --- | ---: | ---: | ---: |
| Original | 0.000000 (0/28; CI [0.000000,0.000000]) | 1.000000 (98/98) | 0.318182 (28/88) |
| Blur-only | 0.535714 (15/28; CI [0.346121,0.720000]) | 0.908163 (89/98) | 0.170455 (15/88) |
| Decompose-only | 1.000000 (28/28; CI [1.000000,1.000000]) | 0.795918 (78/98) | 0.000000 (0/88) |
| Handoff+TargetLock | 0.892857 (25/28; CI [0.760000,1.000000]) | 0.785714 (77/98) | 0.056818 (5/88) |
| Full DaH | 0.892857 (25/28; CI [0.760000,1.000000]) | 0.785714 (77/98) | 0.056818 (5/88) |

## Full DaH failure source

- Full DaH failures: 76/200.
- Failure due to binding: 30 (0.394737 of Full DaH failures).
- Failure due to attribute after correct binding: 46 (0.605263 of Full DaH failures).

## Visual-interference subset

- Source selected set: 56 cases; visual-interference rows: 11; wrong-target-color rows: 9; rescue-reversal rows: 6.
- Full DaH rescued among 9 wrong-target-color cases: 8/9.

## Historical baseline audit

- Raw output agreement: 200/200.
- Normalized prediction agreement: 200/200.
- Correctness agreement: 200/200.

## Paired bootstrap

All repair-vs-Original deltas use the same 200 cases, family-cluster bootstrap, seed 42, and 10,000 replicates. McNemar is exact two-sided.
- Blur-only: Δ=+0.105000, 95% CI=[+0.055000, +0.160000].
- Decompose-only: Δ=+0.090000, 95% CI=[+0.020000, +0.160000].
- Handoff+TargetLock: Δ=+0.120000, 95% CI=[+0.055000, +0.185000].
- Full DaH: Δ=+0.130000, 95% CI=[+0.065000, +0.195000].

## Fixed interpretation rule

Full DaH observed accuracy exceeds Original, family-bootstrap delta CI lower bound > 0, and strong-failure rescue >= .5

The label is descriptive and was applied after inference; it did not alter data, prompts, target selection, scoring, or tuning.
