# Qwen3.5 RealBind visual-interference mechanism test

Main statistics contain exactly the completed pilot's nine valid wrong-target-color cases. The two third-color cases are saved separately and excluded. Of the six total behavioral rescue-reversal cases, four are in this main sample and two are excluded third-color cases.

## Fixed protocol

The two clean runs use byte-identical text and differ only by the original image. The analyzed state is the final processed prompt token's post-block residual, immediately before the first answer token. Layers are the 32 true decoder blocks indexed 0–31. Logit-lens scores use the model-native final norm and tied lm_head. Multi-token candidates use summed full-sequence teacher-forced token log-probabilities.

Case labels were fixed before outcome inspection: selection-like requires a terminal wrong-identity sign onset no later than the wrong-color onset, with paired Facts-only target-positive margins; binding/readout-like requires the color to turn terminally wrong while identity remains target-positive at that onset; all other cases are unclear. These are diagnostic labels, not explicit-token or symbolic-pipeline claims.

## Results

- Cases: 9
- Main rescue-reversal cases: 4
- Best mean color-divergence layer: 27 (Facts−Image margin difference 9.065972)
- Best mean identity-divergence layer: 31 (Facts−Image margin difference 2.690987)
- Best rescue window: 28–31, max rate 1.000000
- Best reverse-corruption window: 31, max rate 1.000000
- Layers with both forward rescue and reverse corruption: [28, 29, 30, 31]
- Selection-like / binding-readout-like / unclear: 1 / 5 / 3
- No-op patch: PASS; max first-token-logit difference 0.0

## Layer-wise causal curves

| Layer | Rescue | Rescue rate | Reverse corruption | Corruption rate |
|---:|---:|---:|---:|---:|
| 0 | 0/9 | 0.000000 | 0/9 | 0.000000 |
| 1 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 2 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 3 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 4 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 5 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 6 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 7 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 8 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 9 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 10 | 1/9 | 0.111111 | 0/9 | 0.000000 |
| 11 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 12 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 13 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 14 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 15 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 16 | 3/9 | 0.333333 | 0/9 | 0.000000 |
| 17 | 3/9 | 0.333333 | 0/9 | 0.000000 |
| 18 | 3/9 | 0.333333 | 0/9 | 0.000000 |
| 19 | 2/9 | 0.222222 | 0/9 | 0.000000 |
| 20 | 3/9 | 0.333333 | 0/9 | 0.000000 |
| 21 | 3/9 | 0.333333 | 0/9 | 0.000000 |
| 22 | 3/9 | 0.333333 | 0/9 | 0.000000 |
| 23 | 5/9 | 0.555556 | 0/9 | 0.000000 |
| 24 | 7/9 | 0.777778 | 0/9 | 0.000000 |
| 25 | 7/9 | 0.777778 | 0/9 | 0.000000 |
| 26 | 7/9 | 0.777778 | 0/9 | 0.000000 |
| 27 | 8/9 | 0.888889 | 0/9 | 0.000000 |
| 28 | 9/9 | 1.000000 | 1/9 | 0.111111 |
| 29 | 9/9 | 1.000000 | 2/9 | 0.222222 |
| 30 | 9/9 | 1.000000 | 2/9 | 0.222222 |
| 31 | 9/9 | 1.000000 | 9/9 | 1.000000 |

## Interpretation boundary

Layer-wise margin divergence alone establishes only a different answer-aligned residual trajectory. A successful cross-condition patch establishes that the paired residual difference at that layer is causally relevant to the behavioral error. Identity/color ordering is interpreted only through the fixed diagnostic rule and does not establish an internal symbolic R→I→C pipeline.

FINAL STATUS: **CAUSAL_RESIDUAL_BRIDGE_FOUND**
