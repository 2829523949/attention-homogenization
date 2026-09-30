# Processor interface contracts

All processor registrations are explicit `ProcessorCapability` records. A registration states the primitive operation, legal input types, output schema, score semantics, model revision/path, deterministic behavior, and the separated capability fields `can_acquire`, `supported_predicates`, `supports_same_claim_rescoring`, `supports_certification_score`, `score_normalization`, and `comparable_across_counterfactual_views`. Processor output is an `Observation` or a typed region/mask result; no processor may return a final compositional answer. A processor may be acquisition-only; the certifier refuses score-based certification for it rather than fabricating a value.

## Grounding

Legal interface:

```text
image + GroundingQuery
  -> region candidates
```

`GroundingQuery` must contain semantic text/constraints whose provenance is `QUESTION_TEXT`, `PREVIOUS_CERTIFIED_FACT`, `PREVIOUS_ENTITY_STATE`, or an explicitly legal tool observation. A raw logical symbol `"A"` or `"B"` is rejected. `GroundingQuery` serialization records token provenance.

The corrected adapters are:

- `FlorenceQueryGroundingAdapter`: calls Florence open-vocabulary grounding with the legal query text. The bundled postprocessor returns boxes/labels but no numeric referent-region confidence, so this backend is explicitly **acquisition-only** (`supports_certification_score=false`). It is distinct from the legacy generic `<OD>` route.
- A score-capable grounding backend (for example a future Grounding DINO adapter with native detection scores) must be registered separately and must implement the same-claim `score(image, query, region, view_id)` contract.
- `Siglip2RegionQueryScorer` is a separate semantic evidence stage, not a grounding acquisition processor. It scores a fixed persistent region against a fixed legal query using the same crop, template, model, and normalization on RAW/FOCUS/REMOVE. It emits `FIXED_REGION_QUERY_COMPATIBILITY` / `SEMANTIC_REGION_SUPPORT` evidence. This layer does not prove full logical identity, instance uniqueness, Florence confidence, or a calibrated probability.

## Category reader

```text
region/mask + candidate ontology
  -> category Observation
```

The candidate ontology must be supplied by a legal interface or configuration, not by hidden family labels.

## Color reader

```text
region/mask
  -> Observation(predicate="color", proposed_value in frozen RealBind color vocabulary)
```

The current adapters are `PixelColorReader` and `SiglipColorReader`. Their vocabulary is the frozen RealBind order: red, orange, yellow, green, blue, purple, pink, brown, black, white, gray. `PixelColorReader` is acquisition-only and explicitly unsupported for certification. `SiglipColorReader` is certification-capable: `read` returns the arg-max candidate; `score_claim` returns the fixed candidate's real logit margin, `margin = candidate_logit - best_alternative_logit`, normalized as `sigmoid(margin)` (`siglip2_color_margin_sigmoid_v1`). The same fixed candidate, processor, and normalizer are used on RAW/FOCUS/REMOVE.

## State/attribute reader

The registry supports a primitive `read_state` capability. No new state model was downloaded or enabled because the audited REL2C smoke interface asks only for color.

## Segmentation

SAM2 is an optional `region -> mask` refiner. It can refine an already legally acquired region and cannot resolve A/B. No SAM2 checkpoint was downloaded; the bbox path is sufficient for this task.

## Geometry

```text
EntityState.region(s)
  -> deterministic coordinates and relations
```

Horizontal ordering is computed from normalized region centers. The final reasoner is not asked to infer left/right when entity regions exist. Geometry emits `DerivedFact` records pointing to the source region-binding certificates.

## D-drive artifact rule

Existing model paths are reused:

- Florence-2: `/mnt/d/vlm_repair_v2/models/florence2/Florence-2-base-ft`
- SigLIP2: `/mnt/d/vlm_repair_v2/models/siglip2/siglip2-base-patch16-224`
- external processor dependencies: `/mnt/d/vlm_repair_v2/external/`

No new visual model weights, repository, harness dependency, Hugging Face cache, Torch cache, pip cache, or temporary model artifact was downloaded in this refactor. Environment helpers continue to redirect caches to `/mnt/d/vlm_repair_v2/`.
