# Certification protocol

## Raw observations versus certified facts

A specialist returns:

```text
Observation(
    entity_id,
    predicate,
    proposed_value,
    score,
    processor,
    provenance
)
```

An `Observation` is not a `CertifiedFact` and is not sufficient input to the final reasoner. It first passes the binding/evidence certifier.

## Counterfactual evidence

For an entity-fact claim `f(E_i)=v`, the certifier keeps the same entity, query, primitive, and claim while scoring:

```text
s_raw    = score(RAW,     same claim)
s_focus  = score(FOCUS(E_i), same claim)
s_remove = score(REMOVE(E_i), same claim)
```

It records:

```text
focus_delta     = s_focus - s_raw
necessity_delta = s_focus - s_remove
```

The expected pattern is target-preserving evidence that is retained/strengthened in `FOCUS` and materially reduced by `REMOVE`. The score normalizer is processor-specific and recorded in the certificate. Raw confidences from different processors are never silently treated as comparable.

The same protocol is used for `REFERENT_TO_REGION` where the grounding processor exposes a query/region compatibility score. If it does not, the certificate records `support=UNSUPPORTED` and status `UNCERTAIN` with failure code `CERTIFICATION_SCORE_UNSUPPORTED`; no fake score is created. Florence-2 OVD currently falls in this category: it is acquisition-only, and its certificates report `GROUNDING_ACQUIRED_BUT_NOT_SCORE_CERTIFIABLE`.

Fixed-candidate colour scores are implemented by `SiglipColorReader.score_claim`: the original candidate value is scored on RAW/FOCUS/REMOVE with the same processor, template set, and normalization. A view whose arg-max label changes is **not** converted to score zero; the fixed candidate's real logit margin is retained.

`SEMANTIC_REGION_SUPPORT` is a distinct evidence layer. A supported SigLIP2 fixed-region/fixed-query score can be recorded as `UNCALIBRATED` diagnostic evidence with raw model logit and `sigmoid` normalization metadata, but it does **not** satisfy the DAG's full `REFERENT_TO_REGION` identity obligation. Without a calibration artifact it can never become `CERTIFIED`, and the controller continues to abstain while full identity is unresolved.

The minimal proof state compiler traverses the required Predicate-DAG closure. A fact enters only when its source DAG node is required and all dependency IDs are already satisfied; deterministic derived facts follow the same rule. The controller, not the final reasoner, decides whether the strict CERTIFIED state is complete. Diagnostic states built with `include_uncalibrated=True` are trace-only and are rejected by the hard-handoff request.

## Gates and calibration

Configuration exposes:

- `epsilon_focus`: permitted focus loss;
- `tau_remove`: required focus-vs-remove separation;
- `minimum_base_score`: required raw normalized score.

The current default smoke configuration is `UNCALIBRATED_DIAGNOSTIC`. It reports deltas and pattern checks but does not tune thresholds on the 100 frozen families and does not claim scientific improvement. With no legitimate development/calibration set currently declared, a passing diagnostic pattern is stored as `UNCALIBRATED`, not as a scientific `CERTIFIED` result.

`CERTIFIED` is available only when a declared calibration protocol is active and the configured gate passes. `REJECTED` indicates a failed necessity/base gate; `UNCERTAIN` indicates missing/unsupported/ambiguous evidence; `UNCALIBRATED` indicates a diagnostic pattern without legitimate threshold calibration.

## View provenance

Every view records:

- view type (`RAW`, `FOCUS`, `REMOVE`);
- method and parameters;
- source entity ID;
- predicted region/mask path;
- source image SHA-256;
- output image SHA-256 when materialized.

Automatic views accept only predicted `RegionPrediction` objects. GT/oracle views, if retained, are separate evaluator functions and namespace/configuration.

## Minimal state rule

The compiler may include only the facts required by the current Predicate DAG. It must not include GT labels, evaluator-only fields, unused proposals, hidden annotations, raw image paths/bytes, or a precomputed final answer field.
