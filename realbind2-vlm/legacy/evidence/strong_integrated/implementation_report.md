# Strong Integrated Minimal Visual Compiler repair

## Exact final architecture

# Final architecture

QUESTION
-> minimal dependency compilation
-> persistent A/B entity grounding
   primary: frozen R4 MM-GDINO-L candidate pool + frozen Qwen3.5 candidate-grid selector
   fallback: SAM3-LiteText-L top-1 only for missing or low-support R4 regions
-> deterministic bbox-center target resolution
-> SAM3-LiteText-L target mask isolation (box prompt, mask threshold 0.5, fixed 16 px dilation)
-> frozen crop-only Qwen3.5-4B target color reading (unchanged generic prompt)
-> minimal executable state
   [TARGET_ENTITY] A/B
   [TARGET_ATTRIBUTE] color = <color>
-> visual exit
-> frozen text-only Qwen3.5-4B final reasoner


## Primary development table

| Method | Logical target | Grounded target | Attribute correct | Minimal state sufficient | Clean final | Raw final |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla VLM | - | - | - | - | - | 20/40 |
| Historical R4 | 34/40 | 25/40 | 21/40 | 16/40 | 16/40 | 21/40 |
| Strong Integrated Repair | 36/40 | 26/40 | 25/40 | 20/40 | 20/40 | 25/40 |

## Clean survival chain

40 -> logical target 36/40 -> grounded target 26/40 -> target isolated 23/40 -> attribute correct 25/40 -> minimal state sufficient 20/40 -> final correct 25/40; clean final 20/40.

## Component provenance

| Primitive | Selected processor | Why selected |
|---|---|---|
| Referent grounding | R4 MM-GDINO-L + frozen Qwen selector; SAM3-LiteText-L top-1 fallback only on missing/low-support | best measured selected-region quality; SAM3-LiteText-L candidate oracle was higher (31/40 top-16, 36/40 top-200) but local selection collapsed to 16/40, so direct SAM3 was not primary |
| Region binding | frozen Qwen3.5 candidate-grid selector | 0.897 selection efficiency on R4 pool; same matcher failed on SAM3 pool |
| Target resolution | deterministic bbox-center geometry | no LLM; reliable when A/B regions are correct |
| Target isolation | SAM3-LiteText-L box-prompt mask + 16 px dilation, bbox fallback | valid mask on 23/40 entities; improved target-local color reading |
| Target attribute | frozen crop-only Qwen3.5-4B | unchanged generic color prompt; strongest validated local color reader |
| Final execution | frozen text-only Qwen3.5-4B | no visual tokens; prior sufficient-state faithfulness preserved |

## Scientific questions

**Q1.** Clean minimal-state sufficiency improves from historical R4 16/40 to 20/40 (+4); raw final improves from 21/40 to 25/40 (+4).

**Q2.** Direct/stronger grounding recovers candidate recall at the oracle level (SAM3-LiteText-L 31/40 top-16 and 36/40 top-200 vs 29/40 for R4), but the available local matcher cannot exploit the larger SAM3 pool; direct SAM3 selection reached only 16/40. Stronger grounding alone is not sufficient without a stronger region-text matcher.

**Q3.** Mask-level target isolation changed target-local attribute correctness from 22/40 (bbox crop) to 25/40; valid SAM3 masks were available for 23/40 entities. The mask-isolated input is the frozen primary attribute input.

**Q4.** A stronger fine-grained VLM beyond crop-Qwen3.5 was not validated locally; the best validated reader remains crop-Qwen3.5-4B with SAM3 mask isolation.

**Q5.** Integrated image-free repair reaches clean final 20/40 and raw final 25/40 versus Vanilla VLM 20/40. Raw final cleanly exceeds Vanilla; clean final equals Vanilla, so the clean baseline-breaking criterion is not met.

**Q6.** 5/40 raw final correct answers are compensating errors (final correct despite an insufficient minimal state); they are excluded from clean repair success.

**Q7.** Dominant remaining bottleneck is grounding/binding: 4 grounding failures and 9 binding failures, then 3 isolation, 3 attribute, and 1 target-resolution failures.

## Failure taxonomy

{
  "ATTRIBUTE_FAILURE": 3,
  "BINDING_FAILURE": 9,
  "GROUNDING_FAILURE": 4,
  "ISOLATION_FAILURE": 3,
  "SUCCESS": 20,
  "TARGET_RESOLUTION_FAILURE": 1
}

## Freeze / replication decision

**Primary development recommendation: `RAW_GAIN_ONLY_COMPENSATING_ERRORS_REMAIN`.**

Raw image-free accuracy exceeds Vanilla (25/40 vs 20/40), but clean minimal-state final accuracy equals Vanilla (20/40); the gain is partly supported by 5/40 compensating errors. The method is deterministic, image-free at final execution, and leakage-audited, but it is not ready to freeze as a clean baseline-breaking result.

Exact next replication protocol: freeze these checkpoints, prompts, thresholds, mask dilation, fallback rules, and state format; evaluate on a larger held-out frozen 20-family/40-QA construction; report logical/grounded/attribute/sufficient/clean/raw metrics, paired LEFT/RIGHT region-state equality, and no-GT leakage; predeclare one stronger local region-text matcher or direct grounder as the next intervention while keeping the final reasoner frozen.
