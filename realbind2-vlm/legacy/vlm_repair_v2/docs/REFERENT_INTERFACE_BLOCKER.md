# Scoped referent-interface blocker

Status: **SCOPED — raw-question-only I2C A/B role is underspecified; REL2C smoke interface is not blocked.**

Exact frozen example:

```text
What color is the racket?
Answer with only the color.
```

The visible question supplies the noun `racket`, but it contains no literal A/B symbol and no visible information that says whether the noun is the benchmark's logical A or B. The frozen QA row stores `qa_type=I2C_A` or `I2C_B`, and the corrected interface treats that explicit protocol task type as a legal role input. If a future runner supplies only the raw text and image, it must construct `TARGET`, not A/B; it must not read `family.A`, `family.B`, `left_is_A`, or hidden IDs.

## Missing information

- a user-visible A/B label or marker;
- a legally exposed protocol role for the atomic operation; or
- a previously certified entity state carrying the A/B role.

## Existing experiments at risk

Historical Stage-1/Stage-2 and baseline runners join QA/family rows for evaluator validation and role mapping. Those artifacts are preserved as historical evidence. They are not automatic inputs to the corrected pipeline.

## Legitimate redesigns

1. Expose the atomic protocol operation (`I2C_A` or `I2C_B`) as an explicit non-evaluator interface input and retain the current visible noun.
2. Change the prompt to expose a legal role marker such as “the A object” only if the benchmark protocol intends that marker to be user-visible.
3. Use `TARGET` internally for raw-question-only I2C and defer any A/B mapping until a later certified state supplies it.

The current corrected smoke is restricted to REL2C, where both nouns and the left/right selection are visible in the frozen prompt. No hidden metadata is used to bypass this scoped blocker.
