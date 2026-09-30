# Historical implementation sanity references

These values detect gross implementation drift. They are not targets and must
never be used to tune RealBind2 prompts, scorer, data, or repair behavior.

Historical RealBind v1.1 / Qwen3.5-4B:

| Metric | Approximate value |
|---|---:|
| I2C pooled | 0.695 |
| PAIR_REL | 0.70 |
| REL2I pooled | 0.79 |
| REL2C pooled | 0.49 |

Historical DaH 200-case oracle-localization pilot:

| Variant | Accuracy |
|---|---:|
| Original | 0.49 |
| Blur-only | 0.595 |
| Decompose-only | 0.58 |
| Handoff + TargetLock | 0.61 |
| Full DaH | 0.62 |

R4 40-case audit: TargetCorrect 34/40, TargetRegionCorrect 25/40,
MinimalStateSufficient 16/40.

Strong Integrated 40-case audit: GroundedTargetCorrect 26/40,
MinimalStateSufficient 20/40, RawFinal 25/40, CleanFinal 20/40.

