# Corrected architecture and novelty contract

This document is the contract for the primary research method. The historical generic Florence path is retained as `LEGACY_GENERIC_PROPOSAL_BASELINE` and is never selected by the main configuration.

## Frozen main interface

```text
DECOMPOSE
  -> ACQUIRE legal primitive evidence
  -> DEFOCUS / COUNTERFACTUAL VIEWS
  -> CERTIFY ENTITY-FACT BINDINGS
  -> COMPILE MINIMAL CERTIFIED STATE
  -> HARD HANDOFF
  -> REASON without the raw image
```

The intended contribution is an **entity-indexed, causally-certified, minimal visual proof state before compositional reasoning**, followed by removal of the raw image from final reasoning.

The main path is not `question -> generic scene graph -> LLM` and not `question -> visual tools -> image + facts -> VLM`. It is:

```text
Question
  -> Predicate DAG
  -> legal ReferentSpecs
  -> persistent Entity IDs
  -> specialist Observations
  -> binding/evidence certification
  -> minimal certified ProofState
  -> RAW IMAGE EXIT
  -> language-only reasoner
```

## Main-method invariants

1. **Entity-indexed.** Every acquired region and primitive fact is keyed by a persistent `E_i`, never by an anonymous proposal index.
2. **Persistent binding.** Once `A -> E_0` or `B -> E_1` is established, later operations reuse that entity. Re-grounding requires an explicit, justified, traceable re-resolution event.
3. **Specialist primitive acquisition.** Grounding, category, color, state, segmentation, and geometry have separate capability contracts. A processor returns primitive evidence, never a compositional answer.
4. **Counterfactual binding certification.** For an entity-fact claim, the same primitive evidence is scored on `RAW`, `FOCUS(E_i)`, and `REMOVE(E_i)` when the processor exposes a meaningful score interface.
5. **Minimal proof-state construction.** The compiler follows the current Predicate DAG and emits only dependency-required entities and facts, with certificate and provenance references.
6. **Hard handoff.** The main final reasoner accepts only the frozen question and the machine/text ProofState. Its API has no image argument, image tensor, embedding, caption dump, or raw proposal channel.
7. **Calibration safety.** Thresholds are configuration, not frozen-evaluation tuning. Without a legitimate development protocol, the smoke runner reports `UNCALIBRATED_DIAGNOSTIC` and does not claim a gain.

## Forbidden shortcuts

- `image + "A" -> detector -> bbox`.
- `A -> hidden family category -> detector`.
- Reusing `family.A.bbox`, `family.B.bbox`, `left_is_A`, canonical color, instance IDs, or hidden answer fields inside automatic acquisition.
- Mapping anonymous `<OD>`/`<REGION_PROPOSAL>` outputs to A/B as the main interface.
- Re-grounding a resolved logical reference for each attribute query.
- Asking the final reasoner to inspect the raw image after state compilation.
- Serializing all proposals, all captions, or an unrestricted scene graph into the ProofState.

## Legacy boundary

The old route is tagged `LEGACY_GENERIC_PROPOSAL_BASELINE`:

```text
image
  -> anonymous generic proposals
  -> heuristic A/B selection
  -> attribute readers
  -> reasoning
```

It is retained for engineering/history, but the main config contains an explicit route assertion that rejects this backend. Its smoke results are historical plumbing artifacts, not evidence for the corrected method.

## Separate ablations

The following are separate conditions, not aliases for the main method:

| ID | Condition | Raw image at final reasoner |
|---|---|---:|
| `A_ORIGINAL` | Original image -> Qwen | yes |
| `B_IMAGE_RAW_OBSERVATIONS` | Image + raw specialist observations | yes |
| `C_IMAGE_STRUCTURED_FACTS` | Image + structured facts | yes |
| `D_CERTIFIED_FACTS_IMAGE` | Certified facts + image | yes |
| `E_MAIN_HARD_HANDOFF` | Certified minimal ProofState only | no |
| `F_ORACLE_CERTIFIED_STATE` | Oracle certified state diagnostic upper bound | no; oracle namespace only |

Only `E_MAIN_HARD_HANDOFF` is selected by `main_method.yaml`.
