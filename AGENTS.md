# Agent Notes

## 2026-08-13 CST: Final Intrinsic-4 B/O-Readable Causal-Subspace Pilot — Outcome B Stop

- User authorized the final strictly frozen B/O causal-subspace discovery pilot using the intrinsic label geometry, with no rank/site/alpha/strength search, no retraining, no V2D automatic continuation, and a stop after Outcome A/B. Added `experiments/qwen35_bo_intrinsic4_causal.py`; independent output root is `outputs/qwen35_bo_intrinsic4_causal/`.
- Definition is fixed from the preceding pure label audit and frozen original ridge only: `B_O` is the saved 9×4 ground-truth delta-O basis; `W_intrinsic=B_O^T W_raw` (4×7680), no new Ridge fit. Its standard float64 numerical rank is 4 with singular values 1.3255065, 1.1586360, .8819290, .8259515. Held-out semantic gate passed: intrinsic-coordinate equivalence to the legal-output projection of 9D decoder relative error 1.99e-16; reconstructed delta MSE .00558316 / MAE .0594293; rank-delta MAE .0836316; slot and full three-object ranking recovery both 1.0, identical to original 9D ranking recovery.
- Offline intrinsic projector audit over the frozen 108 pairs passed: max E_BO=2.95712e-15, max E_perp=2.80996e-15, reconstruction 0, relative dot 1.45427e-15, energy-sum error 1.11e-15. bf16 runtime gate on four pre-outcome sanity pairs passed: max L_perp=.000297933 (<.001); BO actual/intended intrinsic displacement ratio [.9997577,.9999991] within [.999,1.001].
- CUDA bf16 causal run completed all 432 rows (108 frozen held-out REL pairs × NOOP/FULL/BO_READABLE_INTRINSIC/INTRINSIC_DECODER_ORTHOGONAL) at unchanged `OBJ_BUNDLE/stage_2`. FULL reproduced prior capacity: CE=.5590277, RR=.0342963, positive 100/108, NOOP=0. Intrinsic BO: mean CE=.0150463, median near 0, RR=.00100454, positive 56/108. Intrinsic ORTH: CE=.277778, RR=.0171461, positive 92/108. BO−ORTH RR=-.0161415, pair-bootstrap 95% [-.0206336,-.0121526], only 22/108 positive paired contrasts; no source copies.
- Verdict: **Outcome B** — exact intrinsic 4D B/O geometry remains strongly linearly decodable, but its linear-readable natural-difference component does not show disproportionate independent causal control under this interchange. This is a single-held-out-template discovery pilot, not cross-template causal generalization. Stop the entire B/O causal-subspace rescue route: do not change projector/rank/site/layer/alpha/strength, norm-match, refit decoder, run V2D linkage, or add rescue experiments without an explicitly new protocol.

## 2026-08-13 CST: B/O Intrinsic Label-Geometry Audit

- User requested a pure offline audit of the original complete-pair B/O difference labels only: no model, hidden states, causal results, decoder/projector/site/split edits, retraining, Phase V, or intervention. Added `experiments/qwen35_bo_intrinsic_rank_audit.py`; outputs are in `outputs/qwen35_bo_intrinsic_rank_audit/`.
- Original `O` is exactly a 9D identity-major flatten of a 3×3 binary assignment matrix: rows `(red, blue, green)`, columns `(leftmost, middle, rightmost)`. All 36 scene labels are legal one-hot permutation assignments; six unique labels cover all `3!=6` orderings. Original OBJ complete pairs are directed `source != target` pairs within `split × geometry_family`: train 120 pairs, held-out test 30; each split covers all 30 unique ordered delta-O vectors.
- Difference-label result: theoretical all ordered assignment differences, train, held-out, and train+held-out all have exact SymPy integer rank 4 and float64 numerical rank 4. Train+held-out spectrum is 13.4164 repeated four times, then 4.22e-15 and below; held-out is 6 repeated four times, then 1.17e-15 and below. No split undercoverage.
- Explicit 6×9 row/column-sum constraint matrix has exact rank 5 (one redundant constraint), hence kernel dimension 4. All actual delta-O satisfy `C delta-O=0` exactly. The saved 9×4 orthonormal intrinsic basis reconstructs all actual delta labels with max L2 error 9.97e-15.
- Frozen repaired `W_raw` was read only from its saved `.npz`: output dim/rank is 9, while label intrinsic rank is 4. Its first four singular values are 1.32551, 1.15864, .881929, .825951; sigma5=1.29591e-6 (`sigma5/sigma4=1.56899e-6`). Projection into forbidden output complement has Frobenius ratio 6.68574e-7 (1.42759e-6 versus total 2.13528); it is constraint-violating/redundant label output, not extra legal delta-O semantics. Verdict `IR-A: B/O difference labels lie exactly in a 4D intrinsic structural subspace.`
- This supports only a future separately authorized intrinsic-coordinate (`c_O in R4`) decoder refit; it does not retroactively choose a causal projector rank or authorize any intervention/V2D work.

## 2026-08-13 CST: B/O Complete-Rowspace Projector Repair and Corrected Causal Pilot

- User requested a strictly gated repair of `experiments/qwen35_bo_subspace_interchange.py`: replace the invalid rank-4 `1e-6*sigma_max` projector with the complete frozen `W_raw` numerical row space; only then rerun the frozen 108-pair B/O interchange; only run V2D linkage on Outcome A.
- Updated `experiments/qwen35_bo_subspace_interchange.py` and `experiments/qwen35_o2y/run.py`. The repair uses float64 SVD and NumPy-standard tolerance `sigma_max * max(W_raw.shape) * eps64`, retaining actual numerical rank 9 (not hard-coded). It represents the row space as d×r orthonormal `Q`, using `Q(Q^T delta)` without materializing a 7680² projector. Added complete-pair float64 projector audits, native-bf16 actual-hook-write capture, resumable corrected causal checkpointing, difference-decoder-only manipulation reporting, pair-level interaction residuals, and final stop-gate reports.
- Old rank-4 BO/ORTH outputs are preserved but quarantined in `outputs/qwen35_bo_subspace_interchange/raw_invalid/` with metadata status `IMPLEMENTATION_INVALID_PROJECTOR_TRUNCATION` and reason `excluded_nonzero_decoder_singular_direction_before_outcome_unsealing`. The repair workflow did not read/aggregate their CE, RR, contrast, or copy fields. Old FULL capacity `raw/capacity.parquet` remains unchanged.
- Phase R passed. Singular values include 1.3255065, 1.1586360, .8819290, .8259515, 1.2959106e-6, 3.36734e-7, 2.91234e-7, 2.62644e-7, 1.66023e-7; standard tolerance 2.2603897e-12; rank 9. Across all 108 frozen pairs, max E_BO=3.37579e-15 and max E_ORTH=2.18761e-15 (<1e-10); reconstruction max 1.38778e-17; relative dot max 1.03396e-15. Runtime bf16 sanity on four pairs passed: max L_ORTH=.000288357 (<.001) and BO actual/intended decoder displacement [.99988245,1.00001343] within [.999,1.001]. No downstream outcome was inspected before this gate.
- Phase C ran CUDA bf16 on the unchanged Qwen3.5-4B, frozen `OBJ_BUNDLE/stage_2`, all 108 held-out REL pairs and NOOP/FULL/BO_READABLE/DECODER_ORTHOGONAL. Outputs: `corrected_causal/{interventions.parquet,manipulation_checks.parquet,norm_summary.parquet,pair_effects.parquet,condition_summary.parquet,report.md}`. Corrected manipulation passed: BO fraction mean=1, max BO E=3.37579e-15; ORTH fraction max=2.18761e-15. FULL capacity reproduced exactly: mean CE=.5590277, mean RR=.0342963, source-directed 100/108, NOOP=0.
- Corrected causal verdict is **Outcome B**: BO mean CE=.0046296 / RR=.00040805, ORTH mean CE=.251157 / RR=.0155235, so BO−ORTH mean RR=−.0151155; BO source-directed rate=.601852 and BO−ORTH positive paired rate=.185185. No source-answer copies for either. It is only a single-held-out-template discovery pilot; 108 pairs are not independent geometries. Therefore Phase V V2D→B/O compatibility, structural alignment, component-only, and blockade experiments were not run. Do not search sites/layers/ranks/alphas/strengths or continue V2D linkage without a new explicit protocol.

## 2026-08-10 CST: Phase 3A Coordinate-Probe Data and Design Audit

- User asked whether spatial geometry enters the text-state coordinate-probe data and requested the complete image/prompt design. Read-only audit only; no model forward, state extraction, probe refit, image generation, prompt change, or intervention was run.
- Distinguished two similarly named artifacts. `outputs/phase3a_relation_answer_compression_pilot/coordinate_selectivity.csv` belongs to the failed legacy yes/no behavior gate and is implementation-only. The behavior-valid evidence is `outputs/phase3a_descriptor_carrier_pilot/`, with a stricter offline reanalysis in `evidence_audit/`.
- Valid D2 experiment: local Qwen3-VL-2B-Instruct, bf16/eager, 144 existing Grid-Exact 448x448 images (two non-antialiased 2x2-cell colored squares on a 14x14 merged visual grid; x positions exact, `dx` in -5/-4/-3/+3/+4/+5), 24 family-disjoint groups split 16/4/4 train/validation/test, and two RB/BR direct left/right queries per image (288 rows). Prompt is `Compare the horizontal positions of the <A-color> object and the <B-color> object.` followed by `Where is the <subject-color> object horizontally relative to the <reference-color> object? Answer with exactly one word, "left" or "right". Answer:`. Full behavior gate is 288/288 exact, every R×Q cell 1.0, output frequency .5/.5.
- Cached h0–h28 states are pooled text-token states for descriptor-A (`S_A`), descriptor-B (`S_B`), the final shared-prefix carrier token, and pre-answer prediction token. RB/BR prefix checks have 0 failures: vision, both descriptor states, and carrier are exact before the differing direct-question suffix. Coordinate labels are ground-truth `a_x=grid_x_A/13`, `b_x=grid_x_B/13`, and pair center, from the frozen scene manifest; no model-derived coordinates are used.
- In the offline coordinate audit, ridge probes standardize features, fit only 96 train images, select alpha from {0.01,.1,1,10,100} only on 24 validation images, and test on 24 images from 4 held-out families; 400 family-cluster bootstrap draws give CIs. At L18: S_A reads x_A R2=.962 and x_B=.915; S_B reads x_A=.919 and x_B=.985. Coordinate correlation is low on held-out data (.086), so this is not a simple A/B coordinate collinearity artifact. Therefore spatial coordinate information is linearly decodable from text residual states.
- Important limit from `evidence_audit`: after train-only residualization of the other object coordinate, B-slot own-minus-other selectivity has positive bootstrap CIs at L6/L8/L20/L23, but A-slot own-minus-other never has a wholly positive CI (L18 +.049, [-.697,.410]); only four test families make some CIs wide. Carrier transplants made no relation-answer flips. The defensible conclusion is spatially decodable text-state content with suggestive asymmetric B-slot selectivity, not proven two-slot object-specific binding, a reusable relation carrier, or causal transfer.

## Required Practice

## 2026-07-27 23:30:33 CST: Qwen3.5 D2L Execution Status Check

- User asked to inspect the "Distributed Spatial Evidence -> Localized Task Decision" experiment. Read-only audit except this AGENTS entry; no model forward, resume, analysis generation, script repair, prompt/data/model change, or output mutation was run.
- Relevant script/output root: `experiments/qwen35_d2l/run.py`, `outputs/qwen35_d2l/`. Full intended pipeline is `manifest -> extract -> patch -> analyze`.
- Manifest/extract completed: `task_rows.parquet` has 144 REL rows from 18 families and 4 queries (RB/BR/GR/GB each 36); `pair_manifest.parquet` has 144 directed pairs (48 changed, 96 invariant; A/B/C each 48); 144 `.npz` state files exist; `state_manifest.parquet` clean behavior is 1.0 overall and 1.0 for every query. Token partition audit shows V/T/P coverage OK, P at 262, `left/right` single-token candidates 2282/1246.
- Patch stage is incomplete, not currently running. Current valid partial files are `coalition_results_partial.csv` and `all_state_arbiter_partial.csv`; old `*.invalid_batch_expand.csv` files are stale failed-run artifacts and should be ignored. Completed partial coverage is 424/576 expected pair-stage keys = 3392/4608 coalition rows, with 424 all-state arbiter rows. Missing keys are 152 total: remaining `typeC_m0_t1` partial plus all `typeC_m0_t2`, `typeC_m1_t0`, `typeC_m1_t1`, `typeC_m1_t2`; missing split is 80 invariant and 72 changed.
- Partial VTP arbiter integrity is good on completed rows: max abs logit error 0.0, top-1 agreement 1.0, changed source-answer copy 1.0, bad count 0. Partial coalition trends are visible but not final evidence: changed rows currently n=30/stage/coalition over 14 families; stage_5 P mean v=1/copy=1, stage_2 T mean v=.858/copy=1, stage_2 VT mean v=1.001/copy=1, invariant controls have zero flips in completed rows.
- Final outputs are absent: no `coalition_results.parquet`, `all_state_arbiter.parquet`, `all_state_arbiter_summary.json`, `coalition_summary.parquet`, `shapley_results.parquet`, `interaction_results.parquet`, `invariant_controls.parquet`, figures, or `report.md`. To finish, resume only the fixed patch stage and then run analyze; do not interpret partial CSVs as the completed D2L result.

## 2026-07-28 12:39:06 CST: Qwen3.5 D2L Resumed From Checkpoint And Completed

- User requested continuing the previous D2L experiment from checkpoint. Ran `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_d2l/run.py --stage patch`; sandbox CUDA failed with missing NVIDIA driver, then approved external CUDA run completed from existing partial checkpoint. Ran `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_d2l/run.py --stage analyze`.
- No script, prompt, data, model, state-cache, or protocol changes were made. `patch` resumed from 424/576 pair-stage keys and completed all missing typeC keys. Current valid outputs: `coalition_results.parquet`, `all_state_arbiter.parquet`, `all_state_arbiter_summary.json`, `coalition_summary.parquet`, `shapley_results.parquet`, `interaction_results.parquet`, `invariant_controls.parquet`, `report.md`, and six figure/CSV artifacts under `outputs/qwen35_d2l/figures/`.
- Final patch coverage: 4608 coalition rows = 576 rows per coalition over NONE/V/T/P/VT/VP/TP/VTP; 576 pair-stage keys; 1536 changed rows and 3072 invariant rows; every stage has 1152 rows. Changed rows are balanced at 48 per stage×coalition; invariant rows 96 per stage×coalition.
- VTP all-state arbiter passed: max abs logit error 0.0, top-1 agreement 1.0, changed source-answer copy 1.0. Clean behavior remained 1.0 from the completed state manifest.
- Main final D2L results on changed pairs: stage_2 V/T/P/VT/VTP mean v = .129/.870/.001/.999/1.000; stage_3 = -.007/.595/.411/.589/1.000; stage_4 = -.002/.495/.505/.495/1.000; stage_5 = 0/0/1.000/0/1.000. Source-copy rates: stage_2 T and VT 1.0 while P 0; stage_5 P 1.0 while V/T/VT 0.
- Shapley means: stage_2 phi_V=.129, phi_T=.870, phi_P=.001; stage_3 phi_V=-.007, phi_T=.596, phi_P=.411; stage_4 phi_V=-.002, phi_T=.496, phi_P=.506; stage_5 phi_V=0, phi_T=0, phi_P=1. Interactions are near zero at all stages. Invariant controls had max erroneous flip 0.0 but nonzero margin drift up to about .93.
- Report conclusion: predefined pattern `F/weak: no clean D2L pattern`. The experiment supports a valid exact all-state patching pipeline and late localized prediction-token decision capacity, but not a clean distributed-spatial-evidence-to-localized-decision pattern. Stop after fixed D2L completion; no additional token splitting, scans, subspace fitting, data generation, model switch, or repair was run.

## 2026-07-28 15:39:49 CST: Phase VR-Audit Selective Visual Restoration Gate Stop

- User requested Phase VR-Audit for Qwen3.5 selective visual restoration: validate three-object bbox -> Qwen3.5 decoder vision-token mapping, validate source->target exact visual-region restoration implementation, freeze a pilot primary stage if gates pass, and stop without PRED mediation or visual mechanism conclusions.
- Added `experiments/qwen35_visual_restoration/run.py` and output root `outputs/qwen35_visual_restoration/`. Reused Qwen3.5-4B, official non-thinking template, current REL prompt, left/right candidate scoring, existing 18 paired families/36 images, Type A/B/C metadata, macro stages 2-5, D2L state caches, and residual-output patch hook. No new images/prompts/tasks/cache/attention masks/probes/subspaces/layer-head scans/processor edits/PRED mediation/second model were run.
- Built deterministic split in `family_split.json`: pilot families `typeA_m0_t0,typeA_m1_t2,typeB_m0_t0,typeB_m1_t2,typeC_m0_t0,typeC_m1_t2`; confirmatory has the remaining 12 families. Pilot is balanced by move type 2/2/2, mirror 3/3, and RB left/right 3/3. Manifest hashes recorded.
- Mapping audit passed using runtime `image_grid_thw`, processor `patch_size`/`merge_size`, and actual decoder image rows from `infer_vision_token_indices`, not hard-coded Qwen3 grids. All 36 images had 14x14=196 decoder image tokens; red/blue/green bboxes each mapped to exactly 4 tokens and matched the existing Grid-Exact merged token indices offset into decoder rows. Saved `token_mapping.parquet`, full JSON, `mapping_audit.json`, and 12 overlays (4 per type). Two overlays were visually inspected and showed correct grid/object/union alignment.
- Built `region_manifest.parquet` with 48 pilot directed rows (16 changed, 32 invariant) and 1152 condition rows. Main region controls all matched critical union size 8: CRITICAL, RED_MATCHED, BLUE_MATCHED, BACKGROUND_MATCHED, RANDOM_BACKGROUND x8, CRITICAL_SHUFFLED_DELTA x8. FULL_VISION has all 196 vision rows by design; NOOP has 0 rows; source/target bbox-only are 4-token diagnostics only. Region manifest audit passed.
- Started GPU pilot with `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_visual_restoration/run.py --stage pilot --patch-batch-size 8`. The first changed directed row (`typeA_m0_t0_I0__REL__GR -> typeA_m0_t0_I1__REL__GR`) completed all 4 stages and wrote 96 partial rows. Then the mandatory batch=1 vs small-batch consistency gate failed with `max_abs_logit_diff=0.4375`; this is not a tiny floating difference. Per protocol, stopped immediately.
- Because an implementation gate failed, no full `pilot_results.parquet`, no stage summary, no primary restoration stage, and no confirmatory experiment were produced. Saved stop artifacts: `pilot_results_partial.csv` (96 rows only), `batch_consistency_gate.json`, `pilot_summary.json`, `frozen_restoration_config.json` with `primary_stage=null`, and `audit_report.md`.
- Do not interpret partial restoration rows as visual causal evidence. Future work should first debug the batch-expanded hook implementation or run a preregistered batch-1-only implementation audit; do not proceed to confirmatory or stage freezing from this failed VR-Audit.

## 2026-07-30 CST: Phase VR-Batch-Arbiter Verdict B And Batch1 Pilot Resume Status

- User requested the fixed-sample batch/padding arbiter followed, only on Verdict A/B, by the original six-family VR-Audit pilot in exclusively batch=1 mode. Used the existing `experiments/qwen35_visual_restoration/batch_arbiter.py` and updated `experiments/qwen35_visual_restoration/run.py` with explicit `--batch-size 1` validation, `execution_mode=batch1_only`, forced condition chunk size 1, checkpoint mode-mixing rejection, and finalized `batch_size=1` metadata. No images, family split, mapping, regions, prompts, scoring, stages, dtype, attention implementation, weights, data, confirmatory run, or PRED mediation changed.
- Batch-arbiter artifacts in `outputs/qwen35_visual_restoration/batch_arbiter/` include the deterministic balanced eight-sample manifest, runtime config, batch=1 repeat/no-op, duplicate [x,x], same-length mixed-batch, padding-applicability, task-relevant-difference, minimal-restoration, report, and verdict. Verdict B: batch=1 repeats and no-op are exact (full-vocab and margin diff 0; prediction/top-1/sign stable); duplicate batch internal full-vocab diff 0; same-length batches preserve prediction/sign and metadata alignment; all formal REL rows have seq_len 263 so heterogenous padding is not applicable. Original 0.4375 was reproduced at token id 264 (`' a'`) for FULL_VISION, not a left/right candidate; delta_left=-1.4782e-05, delta_right=-1.46478e-05, delta_margin=-1.3411e-07, no prediction change. Minimal CRITICAL-minus-BACKGROUND effect direction matched between modes for both selected changed samples. Formal execution must remain `batch1_only`; small batches are not allowed for confirmatory main statistics.
- The permitted batch1 pilot was resumed from `pilot_results_batch1_partial.csv` only; these rows all declare `execution_mode=batch1_only` and contain no duplicate keys. At last audit it had 1,896/4,352 expected condition rows (84 pair-stage keys), no final `pilot_results.parquet` or implementation-gate/pilot-summary refresh. The external CUDA tool kills foreground runs after roughly 18 minutes and fails to preserve detached children, so the remaining 2,456 independent batch=1 forwards were not completed in this session. Do not analyze, stage-select, or interpret the partial result. Resume with `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_visual_restoration/run.py --stage pilot --execution-mode batch1_only --patch-batch-size 1 --batch-size 1` in a durable CUDA session, then run `--stage analyze`; no confirmatory work is authorized regardless.

For every future conversation or substantial work session in this repository, append the core result here.

Each entry should include:

- Date/time if available.
- User request or experiment goal.
- Files changed or created.
- Commands run when relevant.
- Key numerical results or conclusions.
- Known caveats, blockers, or follow-up work.

Keep entries concise but specific enough that a future agent can resume without rediscovering the same context.

## 2026-08-08 17:53:51 CST: Phase LLaVA-Behavior-Arbiter Scorer Invalid Stop

- User requested an independent LLaVA-NeXT semantic candidate-scoring and clean-behavior arbitration phase, with calibration, visual mapping, restoration, commitment, and mediation paused. Added `experiments/llava_next_crossarch/behavior_arbiter.py` and `tests/llava_behavior_arbiter/test_behavior_arbiter.py`; all 8 unit tests passed before model execution. No prompt, answer contract, few-shot/system instruction, model checkpoint, tokenizer, processor, chat template, data, family, or threshold changed.
- Frozen audit passed: checkpoint `llava-hf/llava-v1.6-mistral-7b-hf`, revision `2424fdd47412fccc66d91719126b420e9fbd7065`, model/processor/tokenizer/chat-template/data hashes matched the prior gate and the legacy 144-row parquet hash was unchanged. Fixed result-blind audit has 16 rows, 8 left/8 right, RB/BR/GR/GB=4 each, Type A/B/C covered. Prompt-only length is 2981, image placeholders 2928, actual next-token row is 2980 after `[/INST]`; prompt-only has no EOS or assistant closing control.
- CUDA command run: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/llava_next_crossarch/behavior_arbiter.py --stage arbiter`; sandbox CPU attempt correctly stopped by the formal CUDA guard, then the same command ran in approved CUDA fp16 offload mode. Offline report refresh and unit-test command also ran. Outputs are under `outputs/llava_next_crossarch/behavior_arbiter/`; old `outputs/llava_next_crossarch/behavior/` remains read-only.
- Exact full-template assistant serialization is `left=[1749]`, `right=[1103]`, one token each; terminal controls are `[2,259]` (`</s>` plus trailing template control) and are excluded. Bare `left`/`right` are the only exact template variants; leading-space/newline variants are diagnostic and were not selected by accuracy. Gold Scorer 1 (direct final-row log-probability), Gold Scorer 2 (incremental batch=1 teacher forcing from frozen multimodal inputs), and optional exact-prefix full-sequence scorer agree exactly on all 16×2 candidate rows; prediction row and batch=1 repeat checks pass.
- Concrete arbitration result: all 16 Gold1 predictions are lowercase-candidate `left`; greedy generation is semantically valid on 16/16 but produces capitalized `Left` token 18365 on 12 rows and `Right` token 9554 on 4 rows (strict lowercase format 0/16). Gold1 full-vocabulary argmax equals the actual greedy first token on all 16 rows, but candidate semantic label disagrees with greedy on 4 audit samples because lowercase candidate IDs differ from uppercase generated IDs. All mismatches are explained; the required one-token scorer↔greedy compatibility gate nevertheless fails.
- The old final semantic fields match Gold1 exactly on all audit rows: same candidate IDs, same prediction row, zero score difference; final old v3 score did not include EOS/control. Therefore old all-left is not a hidden logit-shift/EOS bug; it scores lowercase `left/right`, whose lowercase `left` score wins. Since scorer validity failed, no 144-row rerun, full behavior metrics, changed-pair viability, mechanism row manifest, or behavior claim was produced.
- Final verdict is `SCORER_INVALID`, `mechanism_allowed=false`. This is measurement invalidity, not a model behavior failure. Stop here; do not resume calibration/mapping/restoration/commitment/mediation, prompt search, few-shot, checkpoint search, or threshold changes without a new explicit protocol.

## 2026-07-26 21:02:29 CST: Qwen3.5 Unified Structure Stage A Repair And Phase O2Y Pilot Stop

- User first corrected the Qwen3.5 non-thinking gate: official `enable_thinking=False` renders an empty closed `<think>\n\n</think>\n\n` assistant prefix, which is valid non-thinking mode. Updated `experiments/qwen35_unified_structure/run.py` to parse and allow only that empty closed block, audit generated continuations separately, keep candidate suffix scoring after `</think>\n\n`, and define PRED as the token before the first candidate token after the official prefix.
- Ran Stage A and clean behavior gate with local `Qwen/Qwen3.5-4B` on CUDA bfloat16, no quantization/offload, no prompt/template changes. Outputs: `outputs/qwen35_unified_structure/chat_template_audit.json`, `behavior_gate/{predictions.parquet,metrics.json,report.md}`, and `final_report.md`. Behavior on 36 scenes / 540 rows: overall `.9759`; REL `1.000`; RANK `.8796`; SELECT `1.000`; NEAREST `1.000`; RANK rightmost label `.7222`. Unified experiment stopped under original core gate C because RANK failed.
- User then requested a new preregistered Phase O2Y focused only on identity-aware ordinal structure `O` across REL/SELECT. Added `experiments/qwen35_o2y/run.py` and `outputs/qwen35_o2y/`. Reused Qwen3.5-4B, processor, official non-thinking template, REL/SELECT prompts, candidate sequence scoring, 36 scenes, and existing 24/6/6 family split; no model/data/template/prompt changes and no confirmatory data generation.
- Built O2Y manifests: `protocol.md`, `manifests/task_manifest_rel_select.parquet`, `dataset/pilot_scene_manifest.parquet`, `manifests/ordinal_recombination_manifest.parquet`, `manifest_hashes.json`, and copied split files. Core manifest has 324 rows; recombination manifest has 1620 rows; validation/test each have 270 pair-query rows. Test split was not used for causal pilot.
- Extracted train+validation states only (270 REL/SELECT rows) using hooks at macro full-attention boundaries near 0.4/0.6/0.8, saving OBJ_BUNDLE/TASK_END/PRED float32 CPU vectors under `outputs/qwen35_o2y/states/pilot/`. Macro stage manifest: full-attention code indices 7/11/19/23/31 for target depths .2/.4/.6/.8/1.0; pilot used stage_2/3/4.
- Fit O subspaces from same-geometry permutation differences only. Validation diagnostics (`outputs/qwen35_o2y/subspaces/pilot_fit_summary.json`): OBJ_BUNDLE stage_2 slot/scene `1.0/1.0`; TASK_END REL stage_3 `1.0/1.0`; TASK_END SELECT stage_2 `.7037/.3889`; PRED REL stage_4 `.9444/.8333`; PRED SELECT stage_3 `.9444/.8333`. Ridge emitted small-sample singular-matrix warnings; diagnostics are not causal evidence.
- Implemented validation-only intervention pilot with OBJ_BUNDLE source-to-target O swap on one directed pair (`perm00_small_left_large_right_t1 -> perm01_small_left_large_right_t1`) over all 6 REL + 3 SELECT queries and controls noop/U_O/random/label-shuffled/whole-state. Outputs: `interventions/pilot.parquet`, `interventions/pilot_scene_coherence.parquet`, `pilot/pilot_intervention_summary.json`. No-op exactness passed (`max_abs_score_shift=0.0`), but U_O, random, shuffled, and whole-state all had novel O-CAR `0.0`, target retention `1.0`, REL-ESM `0.0`, SELECT-ESM `0.0`; no structural direction signal.
- Added and ran tiny PRED whole-state same-task decision positive control at stage_4 for REL `blue_to_green` and SELECT `middle`: `controls/decision_positive_control.json`. No-op retained target answer; whole PRED state source-answer copy rate was `0.0` (margins shifted toward source but predictions did not flip).
- Phase O2Y stopped by preregistered pilot rules: structural intervention did not outperform controls and whole-state/Y positive control was ineffective. No 210-scene confirmatory dataset, confirmatory behavior gate, validation hyperparameter expansion, held-out test, metric M, RANK-as-core, NEAREST/FARTHEST, prompt search, 9B download, head scan, full-layer tracing, or repair was run. Final report: `outputs/qwen35_o2y/final_report.md`.

## 2026-07-27 01:17:22 CST: Phase O2Y-Capacity Full-State Causal Capacity Ladder

- User requested an independent REL-only full-state causal capacity audit before any further O-subspace work. Added `experiments/qwen35_o2y_capacity/run.py` and `outputs/qwen35_o2y_capacity/`; no U_O fitting/debugging, factorial data, prompt changes, RANK/SELECT/NEAREST, layer/head scan, model switch, or repair was run.
- Built `outputs/qwen35_o2y_capacity/pair_manifest.parquet`: 12 validation directed REL source→target pairs, 6 left→right and 6 right→left, covering all 6 directed REL queries and all 6 validation scenes. REL `left`/`right` candidate suffixes are single tokens (`left=2282`, `right=1246`); PRED index is 262 after the official empty closed think block. Audit saved in `candidate_pred_audit.json`.
- Level 1 final normalized hidden → lm_head arbiter passed exactly on 12/12 pairs: patched target logits reproduced clean source logits with max/mean error `0.0`, no-op error `0.0`, top-1 agreement `1.0`, and source-answer copy `1.0`. This validates the intervention/prediction-position/lm_head-input pipeline.
- PRED whole-state ladder over macro stages showed decision capacity: `stage_5` CE mean `15.8958` CI `[15.3953,16.5680]`, copy `1.0`, retention `0.0`; `stage_4` CE `7.0833` CI `[6.4737,7.7138]`, copy `.4167`; `stage_3` CE `6.3333` CI `[5.9156,6.8234]`, copy `.1667`. No-op rows kept target with CE `0`.
- Extracted exact span states for all 36 validation REL rows at stages 2–5. Initial site-ladder run exposed a control bug: `random_scene` sometimes fell back to normal source because controls were selected only within the 12-pair manifest. Fixed controls to sample from all validation REL clean rows, recached missing states, and reran the full site ladder. Final outputs: `site_capacity_results.parquet`, `controls.parquet`, `site_capacity_summary.json`, and `site_manifest.json`.
- Corrected Level 2 results: pre-decision full-state sites have continuous but non-decision-sufficient effects. Normal-patch Capacity 1 sites: PRE_DESCRIPTOR_EXACT stage_2 CE `.7396` CI `[.5729,.9063]`; PRE_QUERY_ALL_TEXT_EXACT stage_2 CE `.2917` CI `[.0883,.5159]`; PRE_QUERY_ALL_TEXT_EXACT stage_3 CE `2.1771` CI `[1.4479,2.9068]`; QUERY_SPAN_EXACT stage_3 CE `.2292` CI `[.1667,.3021]`. All have source-answer copy `0` and target retention `1`.
- Specificity caveat: random-scene controls were often comparable to normal patches (e.g. PRE_QUERY_ALL_TEXT_EXACT stage_3 random CE `1.8958` vs normal `2.1771`; PRE_DESCRIPTOR_EXACT stage_2 random `.6875` vs normal `.7396`), and same-answer controls were positive for some early broad spans. TASK_END stage_3/4/5 was Capacity 0. Therefore this audit supports a valid causal pipeline and late decision-state capacity, but not a clean portable scene-specific pre-decision ordinal carrier.
- Stop decision: do not continue to factorial O from this audit. The continuation gate required pre-decision capacity with weaker same-answer/random-scene controls; that specificity gate failed. Final report: `outputs/qwen35_o2y_capacity/report.md`.

## 2026-07-26 16:52:19 CST: Qwen3.5 Unified Spatial Structure Stage-A Fix and Behavior Gate Stop

- User corrected the Stage-A non-thinking criterion: official Qwen3.5 non-thinking chat template may render an empty closed `<think>\n\n</think>\n\n` block. Updated `experiments/qwen35_unified_structure/run.py` to allow only that exact empty paired block, verify `enable_thinking=False` is passed directly, save the actual chat template/model revision, audit generated continuations, candidate suffixes, and PRED positions, without modifying official template, prompts, data, split, or model.
- Reused existing `models/Qwen3.5-4B` and 36-scene / 540-row data; did not redownload or regenerate. Stage A now passes: 8 deterministic greedy samples across REL/RANK/SELECT/NEAREST generated direct legal candidates (`left`, `leftmost`, object descriptors, etc.), with no newly generated `<think>` and no reasoning paragraph. Candidate scoring starts after the official `</think>\n\n` prefix and excludes think tags, assistant delimiter, and end-of-turn tokens. PRED is the last input token after the empty closed think block.
- Improved candidate sequence scoring implementation to use `logits_to_keep` at candidate prediction positions and batch candidates within a row, avoiding full `B×S×vocab` logits OOM. A temporary 1-row smoke completed and failed gate only because it was not a full gate.
- Ran full clean behavior gate on all 540 rows with Qwen3.5-4B bf16, single GPU, no cache, no quantization/offload, no prompt search. Results in `outputs/qwen35_unified_structure/behavior_gate/metrics.json`: overall `.9759`; REL `1.0`; RANK `.87963`; SELECT `1.0`; NEAREST `1.0`; mean margin `6.7834`; greedy audit invalid/new-think/reasoning counts all 0.
- Gate verdict: `C_stop_core_behavior_gate_failed`. Failure is the core RANK family: RANK below required `.90`, and answer-label `rightmost` accuracy `.7222` below the `.85` floor; confusion shows 10/36 rightmost rows predicted as `middle`, and 3/36 leftmost rows predicted as `middle`. REL/SELECT/NEAREST all passed perfectly.
- Updated `outputs/qwen35_unified_structure/final_report.md`, `chat_template_audit.json`, `architecture_audit.json`, `behavior_gate/{predictions.parquet,greedy_sanity.parquet,metrics.json,report.md}`. Stopped before states, probes, subspaces, interventions, held-out test, or any M/R/B/Y mechanism claim. Do not continue by changing prompts/thresholds or auto-downloading 9B; a 9B check would need explicit authorization as a new run with the same gates.

## 2026-07-26 12:14:18 CST: Qwen3.5 Unified Spatial Structure Stage-A Stop

- User requested the fixed Qwen3.5-4B unified M/R/B/Y/L spatial-structure and cross-task causal recombination experiment, with strict non-thinking prompts and behavior-gate stop rules before any representation or intervention claims.
- Added independent Stage A/B runner `experiments/qwen35_unified_structure/run.py`; it implements environment/model snapshot audit, minimal three-object data generation, task/pair manifests, candidate-sequence scoring scaffolding, chat-template audit, and behavior-gate stop handling.
- Downloaded `Qwen/Qwen3.5-4B` to `models/Qwen3.5-4B` with `hf download`; snapshot SHA `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Verified config/tokenizer/processor/chat template/safetensors index and both nonzero shards in `outputs/qwen35_unified_structure/model_snapshot.json`.
- Environment audit used `/root/miniconda3/envs/next3d_qwen/bin/python`: Python 3.10.20, torch `2.7.0+cu128`, transformers `5.8.1`, single NVIDIA GeForce RTX 5060 Laptop GPU, 8.15 GiB VRAM, bf16 supported. Default `python` is 3.13.12 without torch, so do not use it for this experiment.
- Generated independent data under `data/qwen35_unified_structure/`: 36 scenes = 6 identity permutations × 6 geometry templates, 540 task rows across REL/RANK/SELECT/NEAREST, train/validation/test families 24/6/6. Audit passed: no nearest ties, balanced answers, same-order translation and gap controls present. Pair manifest written to `data/unified_structure_pairs.parquet` and frozen copy/hash in outputs.
- Fixed an initial smoke-load bug: current transformers accepts `Qwen3_5ForConditionalGeneration` through `AutoModelForMultimodalLM`, but `use_cache` cannot be passed to `from_pretrained`; runner now sets `model.config.use_cache=False` after loading and passes `use_cache=False` during forward/generate.
- Smoke loaded weights in bf16 on the single GPU using native torch fallback for Qwen3.5 linear attention; no fla/causal-conv1d/custom kernel was installed. Stopped before architecture audit and behavior gate because `processor.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` still rendered `<think>\n\n</think>\n\n` in the assistant prefix for all audited samples. This violates the user’s explicit non-thinking rendered-prompt requirement.
- Outputs: `outputs/qwen35_unified_structure/{environment.json,requirements_frozen.txt,model_snapshot.json,dataset_audit.json,task_manifest.parquet,chat_template_audit.json,final_report.md}` plus not-run `architecture_audit.json`, behavior-gate report/metrics/empty predictions, subspace/intervention placeholders. No behavior metrics, hidden states, probes, subspace fitting, interventions, held-out test, or mechanistic claims were produced.
- Follow-up: do not continue this run by prompt/template surgery, prompt search, Qwen3 repair, quantization, offload, or model switch. A future Qwen3.5-9B run is only worth considering if its exact snapshot first passes the same non-thinking template audit; do not auto-download 9B.

## 2026-07-19: Phase 3A Relation Carrier / Answer Compression Pilot — Behavior Gate Stop

- User requested the strictly gated Phase 3A two-object experiment. Added `experiments/phase3a_relation_answer_compression.py`, reusing the Grid-Exact scenes, Phase-2 VLM loader/state convention, and existing decoder-block output patch hook; no renderer, new image data, OOD set, H3 data, learned subspace, or patching matrix was added.
- Built `outputs/phase3a_relation_answer_compression_pilot/manifest.jsonl`: 144 existing images from 24 family-disjoint pilot families, each with AB/BA query direction and M_L/M_R answer mapping (576 rows; each R×M cell=144; train/validation/test images=96/24/24). Relation and answer are recomputed from exact grid coordinates and asserted.
- Fixed the initial prompt-span implementation: Qwen chat-template/image expansion makes bare token sub-sequence search invalid. The script now follows the existing Phase-2 character-offset → actual multimodal-input-ID alignment convention for target/reference, relation-carrier, and prediction positions. `py_compile` passed.
- Extracted all 576 state payloads (h0–h28, target/reference mention means, pre-mapping relation-carrier, prediction position, logits) with local Qwen3-VL-2B CUDA bfloat16. Sanity passed: factorial completeness true, zero prefix failures, zero scene split leakage; across eight mapping-paired reruns vision/target/reference/carrier states have max and mean absolute difference 0 at every layer (minimum float32 cosine .99999976 due to cosine arithmetic).
- Canonical prompt behavior failed the required model-behavior gate on held-out test rows: 0.59375 accuracy. R×M test cells: right/M_L=.50, right/M_R=1.0, left/M_L=.875, left/M_R=0.0. This is mapping-following failure, not base spatial vision failure: the existing Grid-Exact direct left/right audit has test accuracy .99653.
- Added prompt-gate-only equivalent mapping surfaces (`direct`, `binary_question`) to check the failure before interpreting representations. On 16 conditions each, direct=.50 and binary-question=.625, still inadequate. They did not add demonstrations or new data.
- `probe_results.csv` and `coordinate_selectivity.csv` were written but are implementation diagnostics only: the core behavior gate invalidates H1/H2/H4 interpretation. In particular, no carrier transplant, persistent patch, prediction-position patch, causal claim, or Phase 3B decision was run. Gate-stop report: `outputs/phase3a_relation_answer_compression_pilot/gate_stop_report.md`.
- Follow-up: establish a held-out balanced answer-mapping prompt/task protocol with high accuracy, then rerun the existing extraction/analysis/patch stages in this script; do not reuse current probes as mechanistic evidence.

## 2026-07-19: Phase 3A Natural Left/Right Interface Repair — Second Gate Stop

- User requested repair of the Phase 3A behavior interface only: no image regeneration, legacy yes/no probe rerun, Phase 3B, or causal/representation analysis unless a natural-mapping behavior gate passed.
- Incrementally updated `experiments/phase3a_relation_answer_compression.py` with `--interface natural_lr` and the one permitted `natural_selection` fallback. Both reuse existing Grid-Exact images and the previous family-disjoint 16/4/4 family selection. Natural manifests have canonical `R=A relative to B`, exactly two post-carrier mappings per image, and assert final labels: `same/reverse` yields left/right; selection yields A/B. The natural-left/right manifest is 288 rows/144 scenes/24 families, with all four R×M cells=72.
- Added batched behavior-only gate handling and records that report exact-match accuracy, R×M cells, mapping/relation summaries, candidate-token audit, output frequencies, and log-probability margin (left-right or A-B). This stage does not extract/cache states.
- First 24-image held-out screening used the required natural left/right prompt: canonical question `Is object A horizontally to the left or to the right of object B?`, followed by same/reverse final request and exact `left`/`right` answer. Candidate IDs are left=2115/right=1290 (single token). Gate failed: overall .6667; right/same=1.0, right/reverse=0.0, left/same=.6667, left/reverse=1.0; outputs left=.1667/right=.8333.
- Ran the only allowed fallback, A/B object selection. Space-prefixed A=362/B=425 are single tokens. It also failed on the same held-out 24 images: overall .5833; right/select-left=0.0, right/select-right=.9167, left/select-left=1.0, left/select-right=.4167; A output frequency=.875. The existing direct relation task remains .9965 test accuracy, so this identifies inability to execute the required two-stage final request, not loss of base spatial behavior.
- Stopped before full natural behavior, hidden-state extraction, new probes, carrier/prediction patching, or Phase 3B. Details: `outputs/phase3a_relation_answer_compression_natural_mapping_gate_report.md`; gate JSON/records are in `outputs/phase3a_relation_answer_compression_{natural_lr,selection}_screen/`. Do not enumerate additional prompts without a new user decision; both user-permitted natural interfaces are exhausted.

## 2026-07-19: Phase 3A Final Neutral-Carrier Repair — Oracle Pass, Image Gate Stop

- User requested one structurally different final repair only: first a text-only oracle relation-completion calibration, then a neutral visual carrier followed by direct AB/BA left/right sentence completion. No image regeneration, old-prompt rerun, large state extraction, patching, or Phase 3B was permitted unless both gates passed.
- Incrementally extended `experiments/phase3a_relation_answer_compression.py` with `--stage oracle` and `--interface neutral_completion`. The oracle uses exactly two semantically equivalent text templates × four explicit fact-relation/final-subject-order cells. The image manifest reuses the fixed 144 Grid-Exact images/24 families and emits two AB/BA direct-completion prompts per image (288 rows; each canonical-R_AB × Q cell=72). Canonical R_AB, final answer, and geometry are recomputed/asserted. The neutral carrier is the final token of literal `Spatial information retained.` before either suffix.
- Fixed one neutral-span branch syntax error before any model execution; `py_compile` passed thereafter. Text-only oracle results in `outputs/phase3a_oracle_relation_completion/`: one-token left=2115/right=1290; 8/8 exact accuracy across both templates; each of four cells=1.0; left/right output frequency=.5/.5. It passes the specified >=.95/.90 gate and shows the model can perform explicit natural text relation reversal.
- Neutral image screening at `outputs/phase3a_neutral_carrier_pilot/` used only the 24 pre-existing held-out images (48 AB/BA conditions, 12/cell). It failed: overall=.5625; right/AB=0.0, right/BA=.25, left/AB=1.0, left/BA=1.0; AB=.5, BA=.625; canonical-right=.125, canonical-left=1.0; left/right outputs=.5625/.4375. This fails the overall and per-cell behavior gate despite the balanced text oracle.
- Stopped immediately: no full 144-image behavior, no new hidden states, probes, carrier/prediction patches, H1/H2/H4 claim, or Phase 3B launch. This gate failure does not imply missing middle-layer spatial structure and does not rule out Phase 3B independently. Detailed result: `outputs/phase3a_neutral_carrier_pilot/gate_stop_report.md`.

## 2026-07-19: Phase 3A D0/D1/D2 Alias-Binding Diagnostic

- User requested one behavior-only three-condition matrix after neutral-carrier asymmetry; explicitly prohibited new images, state extraction, probes, patching, Phase 3B, or automatic continuation. Added `--stage diagnostic` to `experiments/phase3a_relation_answer_compression.py`; it uses the exact same 24 held-out Grid-Exact images and current batched Phase-3A evaluator for 48 AB/BA conditions per D0/D1/D2 (144 records total).
- D0 is not an approximate recreation: it imports and calls `grid_exact.questions.questions_for_scene` and `grid_exact.behavior.make_prompt`, the original direct-task question/template. D1 uses direct A/B aliases with no carrier. D2 removes aliases and uses a neutral direct descriptor prefix plus the native direct left/right query. Left/right single-token candidates are 2115/1290; records save pair-conditional log probabilities/margins, prompts, labels, outputs, image path, and bbox centers.
- Metadata audit found zero errors: grid recomputation agrees with both geometry labels, all image paths exist, object IDs/colors are valid. Visually inspected 8 images (4 R_AB=left, 4=right), including blue/green and red/blue families; all confirm rendered object location, bbox centers, identity, and labels. Artifact `manual_check_8_images.csv` includes all D0/D1/D2 AB/BA predictions for these samples.
- Results in `outputs/phase3a_d0_d1_d2_diagnostic/diagnostic_gate.json`: D0=1.0 overall and all four cells=1.0; D1=.5625 (right/AB=.25, right/BA=0, left/AB=1, left/BA=1); D2=1.0 overall and all four cells=1.0. D0/D2 output frequency is exactly left=.5/right=.5; D1 left=.4375/right=.5625. This rules out evaluator/preprocessing/image/label regression and shows neutral descriptor carrier is behaviorally sound; the failure is A/B multimodal alias binding, especially canonical-right scenes.
- Stopped after the report `outputs/phase3a_d0_d1_d2_diagnostic/diagnostic_report.md`. If a future authorized Phase 3A resumes, use the D2 descriptor carrier, not A/B aliases. Do not launch A/B/C Phase 3B on this model; it lacks the required alias-binding prerequisite. A stronger model must first pass this same low-cost gate.

## 2026-07-19: Phase 3A Descriptor-Based Neutral Carrier, States, Probes, And Causal Stop

- User explicitly authorized only the D2 behavior-passing descriptor carrier continuation; no new images, prompt redesign, old-gate rerun, aliases, codebook, or Phase 3B. Added `--interface descriptor_carrier` to `experiments/phase3a_relation_answer_compression.py`. It uses real color descriptors from each existing image and the exact D2 shared prompt `Compare the horizontal positions of the <A-color> object and the <B-color> object.`; RB/BR suffixes are native direct left/right queries. Manifest labels recompute/assert R=A-color relative to B-color and query-specific answer from grid coordinates.
- Full gate at `outputs/phase3a_descriptor_carrier_pilot/behavior_gate.json`: 144 images, 288 conditions, overall=1.0, all right/RB/right/BR/left/RB/left/BR cells=1.0, left/right=.5/.5. The full 24-family pilot satisfies the stricter >=.95 overall/.90 cell gate.
- Extracted all 288 h0–h28 payloads: pooled descriptor-A/B mentions, final neutral-carrier token, full neutral-sentence span for span patching, and pre-answer P. Fixed descriptor repeated-span alignment with the existing offset-to-actual-ID method plus declared raw-token occurrence rank (the descriptor appears again in the suffix); no free-text matching. `sanity_report.json`: factorial complete, no split leakage, no prefix failures; all RB/BR paired pre-suffix vision/descriptor/carrier max and mean absolute differences=0 (cosine roundoff only).
- Added resumable per-layer linear analysis to survive process limits. Family-grouped held-out results: descriptor slot selectivity supports object-specific coordinate content (red L18 x_red R2=.962 vs x_blue=.871; blue L6 x_blue=.898 vs x_red=-.318). Carrier R BAcc=1 by L8; L18 carrier x_red/x_blue/pair-center R2=.943/.978/.980. Duplicated RB/BR carrier Q/Y sanity probes are exactly .5. Existing-position median holdout carrier R BAcc reaches 1.0 but does not eliminate a coordinate-package account.
- Prediction: within split R/Q/Y reach 1.0; cross-query Y BAcc=1 from L15 onward while cross-query canonical R is 0 in L15–L28 (best .542 at L12). This is a strong answer-compression signature, not merely a random-split answer probe.
- Added checkpointed descriptor whole-state patching with no-op, same-relation, and opposite-relation same-query pairs from 4 held-out matched ± displacement families (all clean-correct, retention=1). Single/persistent carrier-token and single/persistent neutral-sentence-span patches have zero opposite-relation reinterpretation flips at all layers. Span single L16 had a small +.172 shift but no flip; persistent span still no flips. No-op margin shift=0 and same-relation controls retain target answer, so carrier intervention indexing/hooking works but causal reusable spatial transfer is not established.
- Prediction-position patch at L18–L27 gives source-answer copy rate=1.0 and target-answer retention=0 on opposite-relation pairs (L18 shift +8.07, L27 +9.48), supporting H4 answer compression. Causal evidence does not establish H2: carrier is relation/coordinate readable but whole-state carrier transfer failed; it remains compatible with H1-like distributed coordinate content or direct suffix-to-vision rereading.
- Final report and necessary artifacts: `outputs/phase3a_descriptor_carrier_pilot/phase3a_descriptor_report.md`, behavior/sanity JSON, `descriptor_slot_selectivity.csv`, `carrier_representation.csv`, `prediction_probe_results.csv`, three curves, and four carrier plus one prediction patch result/summary files. Stop: no further automatic Phase 3A or Phase 3B; future H1-vs-H2 discrimination needs separately authorized anti-direct-vision-readout design.

## 2026-07-19: Grid-Exact Step-4 24-Family Extraction Repair and Pilot

- User requested only repair/completion of the Step-4 pooled-state pilot; did not regenerate Step 1–3 data, run full 480-family extraction, or enter Step 5/causal work.
- Reproduced the legacy pre-shard failure at `outputs/grid_exact_pilot_480_v0_2_r1/step4_pilot24/debug_legacy_failure`: `TypeError: 'list' object is not callable`. Root cause: `run_step4_pilot.py` rebounded the `pool` helper name to a split-candidate list before the first pooling call.
- Added strict split extraction/analysis entry points and shared storage helpers:
  - `grid_exact_dataset/src/grid_exact/step4_extract.py`
  - `grid_exact_dataset/scripts/{extract_step4_pilot.py,audit_step4_extraction.py,analyze_step4_pilot.py,reproduce_step4_legacy_failure.py}`
- New stage-specific shards avoid mixed dimensions: processor packed input=1536, local projection=1024, vision-final=1024, merger/h0/decoder=2048. Shards are atomic `states: float32[N,D]` + aligned JSONL, hash-verified, and resume now performs immutable verified-shard checking.
- Level 1 (`debug_one_family`, six scenes, stages plus L1/L16/L28), Level 2 (`debug_two_families`, 12 scenes/all layers including matched-row unavailable), and 24-family extraction (`extraction_v2`, 144 scenes, 231 stage/group shards, ~264 MB) all passed integrity; resume verified existing hashes without a forward pass.
- Pilot selected 8 families per split, balanced 4/4 family types per split. Background coverage: original/far/pooled 24/24, matched-row 16/24; unavailable matched-row records are omitted rather than replaced.
- Ran raw/centered fixed-alpha ridge and fixed-C logistic probes on verified shards and 50 family-internal label permutations shared across all stage/group comparisons. Packed input/local projection fixed-anchor centered probes are exactly null (R2=0, BACC=.5). The 24-family probe is implementation sanity only; some small-control estimates are unstable, so no representational conclusion or 480-family launch decision should rely on individual pilot scores alone.

## 2026-07-19: Grid-Exact Step-4 Sensitivity Gate Stop

- User requested the 24-family positive-control/readout gate before any full 480-family extraction. Added `grid_exact_dataset/scripts/report_step4_sensitivity.py` and wrote `outputs/grid_exact_pilot_480_v0_2_r1/step4_pilot24/extraction_v2/positive_control_sensitivity_report.{json,md}`.
- Audited 31,680 fixed-anchor/moved-object support-metadata records against scene grid coordinates and verified zero state/label/support alignment errors.
- Sensitivity gate failed: moved-object best family-centered test displacement R2 was 0.0 (processor input/local projection); best family-centered side BACC was 0.6042 (decoder late layers), below the required .70 and not materially above the 50-permutation null. Therefore did not add ordered/per-token readouts and did not run 480-family extraction, formal statistics, G/Q/Y, or Step 5.

## 2026-07-18: Phase 2R-v2 Phase A/B Pilot

- User requested only corrected Phase 2R-v2 Phase A/B: no full Phase C state extraction, full-layer probes, relation ablation, mediation, or causal patching.
- Created isolated output root `outputs/clevr_spatial_content_phase2rv2`; old Phase 2/pilot outputs were not overwritten.
- Added:
  - `experiments/clevr_bottleneck/build_phase2rv2_datasets.py`
  - `experiments/clevr_bottleneck/validate_phase2rv2_camera_coordinates.py`
  - `experiments/clevr_bottleneck/validate_phase2rv2_audits.py`
  - `experiments/clevr_bottleneck/phase2rv2_model_sanity.py`
- Ran the builder on CLEVR val scenes (`max_families=12`, `trajectory_families=3`), renderer at default 320×240/8 samples, camera-coordinate audit, render/mask audit, and a one-family CUDA bfloat16 model sanity check.
- Data result: 12 A-fixed and 12 B-fixed source families, 120 anchor-condition rows, 43 valid trajectory points, and 48 geometry×query 2×2 rows; each of the four G/Q/Y cells has 12 rows and all three templates occur four times/cell.
- Camera-right definition is `dot(world_xy, normalize(scene.directions.right[:2]))`; `behind` is used as camera-plane up, while CLEVR `above` is world-z. All 120 directed labels/projections passed.
- Architecture/model sanity: Qwen vision encoder remains bidirectional and h0 is merged vision output in decoder input, not raw patch embedding. Same image with Q_AB vs Q_BA had exact image-token equality through h0–h28 (max abs delta 0), whereas prediction changed (max abs delta 6.0). C0/C7 no-op had exact all-token equality (max delta 0).
- Critical Phase-A result: default 8×10 merged visual grid yielded only 8/120 nonempty, disjoint >=.7 core-mask rows; 112 were rejected as empty core. No rows had nonzero accepted A/B core overlap.
- Critical local-invariance result: C7 is exact (raw core RGB and packed processor input deltas 0), but C3/C4 anchor-core RGB means are 2.5455/1.5217 on 0–255 RGB and packed-input means are .008518/.011685. Visible-area changes also occur. Thus this pilot fails the strict local-anchor-input criterion, plausibly due to shadows/visibility, and must not proceed to Phase C.
- `architecture_state_map.json`, manifests, render/patch/mask audits, same-image/no-op equality files, token overlays, README, and summary JSON are in the new output root. Regenerate a filtered default-resolution set with larger separated anchors and shadow/occlusion control; do not silently increase resolution or use the impossible dual-coordinate shortcut gate.

## 2026-07-10: Dot Test And COCO Prediction-Token Attention

- Paper core overview:
  - The paper studies spatial grounding failure in decoder-only VLM self-attention, especially for fine-grained visual localization.
  - Central claim: prediction-token attention has a layer-wise U-shape. Middle decoder layers retain more spatially useful grounding, while final layers become homogenized and less object-specific.
  - Face experiments show attention maps for different facial feature queries become highly similar, making final-layer attention unreliable for fine-grained face localization.
  - Dot-test experiments show high-contrast synthetic objects are partially localizable in middle layers, but early/late layers show sink-like or homogenized behavior.
  - COCO prediction-token experiments strengthen the claim by analyzing model-generated object tokens, not prompt/query tokens. On 50 COCO val2017 images, generated prediction-token grounding peaks near L19/L20 and declines by L27.
  - Embedding/probe experiments support the interpretation that useful spatial information is more accessible in mid-layer representations than in final-layer representations.
  - The intended follow-up is spatial repair: test whether final-layer vision tokens still contain recoverable spatial structure by training lightweight repair modules while freezing all VLM parameters.
  - Important caveat: attention maps are diagnostic, not direct proof of causality. Claims should distinguish spatial readout failure, representation accessibility, and true causal intervention.
- Verified Qwen dot-test visual token grid is 8x8 for a 256x256 image: `image_grid_thw=[[1,16,16]]`, merged grid 8x8.
- Corrected dot target mapping from original 16x16 block coordinates to actual heatmap grid:
  - red `(1,1)` 16x16 -> `(0,0)` 8x8
  - green `(7,7)` 16x16 -> `(3,3)` 8x8
  - blue `(14,14)` 16x16 -> `(7,7)` 8x8
- Updated dot scripts to use dynamic grid mapping and exact 16x16 rectangle drawing:
  - `run_dot_test.py`
  - `run_attention_fixed.py`
  - `dot_layer_analysis.py`
  - `dot_heatmap_viz.py`
  - `dot_routing_demo.py`
- Ran corrected `dot_layer_analysis.py` on RTX 5060 via non-sandbox GPU access.
  - L1-3 cross_r `0.9178`
  - L8-10 cross_r `0.7017`
  - L18-20 cross_r `0.4949`
  - L25-27 cross_r `0.7565`
  - L18-20 localized all colors within 1 grid cell of target; L25-27 showed bottom-right collapse.
- Downloaded COCO 2017 annotations to `/root/coco/annotations/instances_val2017.json`.
- Created and iteratively improved `run_coco_prediction_attention_ushape.py`.
  - Main analysis uses generated prediction-token attention, not prompt/query token attention.
  - Added COCO annotation/image/bbox sanity checks.
  - Added stricter object-name prompts, alias matching, subword-token averaging, failure summaries, and run reports.
- Ran COCO prediction-token attention experiment on 50 val2017 images.
  - Output directory: `outputs/coco_pred_attn_50`
  - COCO sanity: 50 images, 161 objects, 0 errors, 0 warnings.
  - Successful generated target-token cases: 102 / 161, success rate 63.4%.
  - Exact category match: 73 cases; alias match: 29 cases.
  - Key layer results:
    - Best target mass: L19 = `0.4638`
    - Best distractor gap: L19 = `0.2252`
    - Best pointing hit: L19 = `0.6667`
    - Lowest cross-object Pearson: L20 = `0.1503`
    - Last-layer Pearson: L27 = `0.7850`
  - Interpretation: supports prediction-token attention U-shape and late-layer attention homogenization.
- Created `run_coco_spatial_repair.py`.
  - Diagnostic experiment for whether last-layer vision tokens can be spatially repaired.
  - Includes feature cache extraction, image-id splits, repair modules, bilinear readout, metrics, paired tests, plots, and visualization hooks.
  - Syntax and `--help` checks passed; experiment not run.
- Created `run_coco_logit_lens_repair_analysis.py`.
  - Logit Lens analysis for mid-layer, raw last-layer, repaired last-layer, shuffled/random controls, and fusion representations.
  - Uses original VLM final norm and `lm_head` to measure vision-token semantic readability for generated prediction-token target categories.
  - Supports feature cache reuse, optional feature extraction via `run_coco_spatial_repair.py`, repair checkpoint loading, category synonyms, patch/region logit lens, semantic gaps, attention-logit consistency, top-attended semantic hit, feature preservation, paired tests, plots, logit map export, and visualization hooks.
  - Syntax and `--help` checks passed; experiment not run.

## 2026-07-10 11:41:59 CST: Unrun Experiment Script Check

- User asked which experiment scripts are still not run.
- Checked script list and output directories with `rg --files`, `find outputs -maxdepth 2`, `find outputs -maxdepth 1`, and `rg` over COCO repair/logit-lens scripts.
- Confirmed no output directories or cache/checkpoint artifacts for `run_coco_spatial_repair.py` or `run_coco_logit_lens_repair_analysis.py`; only `__pycache__` files exist.
- Conclusion: `run_coco_spatial_repair.py` and `run_coco_logit_lens_repair_analysis.py` have only syntax/`--help` validation recorded and still need actual experiment runs.

## 2026-07-10 11:52:35 CST: COCO Spatial Repair 50-Image Run

- User asked to choose an appropriate data scale for the COCO repair experiment.
- Chosen scale: reuse the validated `outputs/coco_pred_attn_50` data, all 102 successful prediction-token cases from 50 selected COCO val2017 images.
  - Dry run loaded 102 cases and split by image_id into train=29, val=10, test=10 images.
  - Final test evaluation contained 20 successful target cases per method.
- Fixed `run_coco_spatial_repair.py` before running:
  - `raw_last`, `raw_mid`, and `random_conv_last` had trainable readout parameters but the training loop skipped them because `method_trainable=False`.
  - Changed `train_one_method` to train whenever the system has trainable parameters, while still freezing repair modules according to method.
  - `py_compile` and `--help` checks passed with `/root/miniconda3/envs/next3d_qwen/bin/python`.
- Ran formal experiment on RTX 5060 via non-sandbox CUDA:
  - Command used `/root/miniconda3/envs/next3d_qwen/bin/python run_coco_spatial_repair.py`
  - Inputs: `outputs/coco_pred_attn_50/{selected_samples.jsonl,prompts.jsonl,generation_results.jsonl}`
  - Model: `/root/Qwen3-VL-2B-Instruct`
  - COCO images/annotations: `/root/coco/val2017`, `/root/coco/annotations/instances_val2017.json`
  - Output: `outputs/coco_spatial_repair_50`
  - Args: `--extract_features --mid_layer 19 --last_layer 27 --methods raw_last,raw_mid,conv_last,shuffled_conv_last,mlp_last,fusion --epochs 10 --batch_size 16 --lr 1e-4 --weight_decay 1e-4 --alpha_values 0,0.25,0.5,0.75,1.0 --seed 42 --device cuda --dtype bfloat16 --save_visualizations --num_viz 12`
- Feature extraction completed for all 102 cases; dynamic vision grids included 10x16, 12x16, 13x20, 15x20, 20x15, 19x19, etc.
- Main test summary from `outputs/coco_spatial_repair_50/repair_summary_by_method.csv`:
  - `raw_last`: target mass `0.2000`, pointing hit `0.2000`, distractor gap `0.0000`, cross-object Pearson `1.0000`
  - `raw_mid`: target mass `0.1500`, pointing hit `0.1500`, distractor gap `-0.1500`, cross-object Pearson `1.0000`
  - `conv_last`: target mass `0.2500`, pointing hit `0.2500`, distractor gap `-0.2000`, cross-object Pearson `1.0000`
  - `mlp_last`: target mass `0.2500`, pointing hit `0.2500`, distractor gap `-0.3000`, cross-object Pearson `0.7846`
  - `shuffled_conv_last`: target mass `0.4000`, pointing hit `0.4000`, distractor gap `-0.0509`, cross-object Pearson `0.7839`
  - Best target mass/pointing hit: `fusion_alpha_1.0`, target mass `0.4544`, pointing hit `0.4500`, but distractor gap `-0.2515`, cross-object Pearson `0.8840`
- Paired tests:
  - `conv_last` vs `raw_last`: target mass p=`0.6663`, pointing hit p=`0.6663`; not significant.
  - `conv_last` vs `mlp_last`: target mass p=`1.0`, pointing hit p=`1.0`; no advantage.
  - `conv_last` vs `shuffled_conv_last`: `conv_last` is worse on target mass/pointing hit; no evidence for spatial conv repair.
  - `fusion_alpha_1.0` vs `raw_mid`: target mass p=`0.0081`, pointing hit p=`0.0102`; improves target metrics but not distractor gap.
- Caveat/conclusion:
  - This first repair run does not support a strong claim that last-layer spatial structure is recoverable by `conv_last`; the shuffled conv control is better than `conv_last`.
  - Several methods show near-zero entropy and cross-object Pearson near 1, indicating learned readout collapse or same-image homogenization; results should be treated as diagnostic and followed by objective/readout regularization or a stricter evaluation before paper claims.

## 2026-07-10: COCO Repair Method/Data-Flow Audit

- User questioned whether the repair code design is flawed because shuffled conv should not outperform direct conv, and asked how each method processes data.
- Audited `run_coco_spatial_repair.py` data flow:
  - Cases come from successful `generation_results_jsonl` entries joined with selected samples/prompts; each case stores target bbox, distractor bboxes, prompt, generated text, matched generated-token offsets, and image size.
  - Feature extraction teacher-forces `prompt + generated_text`, caches `mid_vision_tokens`, `last_vision_tokens`, and final-layer `prediction_token_hidden`; grids are dynamic per image.
  - Masks are generated by checking whether patch centers fall inside COCO bboxes.
  - All methods train their own bilinear readout; comparisons are therefore not purely repair-module-only.
- Important design issue found:
  - `ShuffledConvRepair` currently permutes token order, applies conv on the shuffled sequence reshaped to the grid, then applies the inverse permutation back to original order.
  - This does break true spatial neighbors, but it also acts like a learned random-neighbor/global mixing residual rather than a clean "destroy spatial structure" control, so it can plausibly outperform `conv_last`.
  - The permutation cache is keyed only by token count `N`, not `(grid_h, grid_w)`, which is another caveat for variable grids with the same `N`.
- Additional caveats:
  - The trained bilinear readout and target-mask loss can collapse to very peaky same-image attention; near-zero entropy/cross-object Pearson near 1 in outputs supports this.
  - Existing result should not be interpreted as evidence that shuffled spatial conv is genuinely better; it is more likely a control-design/readout-objective issue.

## 2026-07-10 12:24:12 CST: Attention Spatial Recovery Script

- User requested a new attention spatial recovery experiment to test whether last-layer vision tokens preserve spatial structure but fail under the original attention readout.
- Created `experiments/attention_recovery.py`.
  - Reuses COCO successful prediction-token cases from existing jsonl files.
  - Extracts and caches, per case: `A_mid` from actual model mid-layer attention, `A_last_model` from actual last-layer attention, `last_vision_tokens`, last-layer query hidden states for the same generated prediction token, dynamic vision grid, and bbox metadata.
  - Keeps VLM parameters, Q projection, K projection, and attention computation frozen.
  - Trains only optional readout modules to match `A_mid`; no bbox loss, target mass loss, or Q/K training.
  - Methods: `raw` identity, `conv` spatial 2D conv readout, `mlp` per-token MLP readout, `shuffle_conv` fixed per-grid spatial shuffle + conv + inverse shuffle control.
  - Main metric: Spearman rank correlation between `A_method` and `A_mid`; also outputs Pearson, KL, MSE, entropy, and auxiliary bbox metrics: target mass, pointing accuracy, attention center distance.
  - Outputs per-case CSV, summary CSV/JSON, train log, feature cache, and optional heatmap visualizations.
- Verification:
  - `/root/miniconda3/envs/next3d_qwen/bin/python -m py_compile experiments/attention_recovery.py` passed.
  - `--help` passed.
  - Dry run on full `outputs/coco_pred_attn_50` inputs loaded 102 cases and split images train=29, val=10, test=10.
  - GPU smoke run with `--max_cases 6 --epochs 1 --extract_features` completed at `outputs/attention_recovery_smoke`.
  - Reuse-cache output check completed at `outputs/attention_recovery_smoke_v2`; per-case CSV includes `model`, `layer_mid`, `layer_last`, `method`, `spearman_to_mid`, `entropy`, `pointing_accuracy`, etc.
- Implementation caveat:
  - Recomputed readout attention uses the model's frozen last-layer Q/K projection modules over vision tokens only, but does not currently reproduce full transformer RoPE/all-token attention exactly. `last_model` is saved as the actual model attention reference, while `raw` is the frozen-Q/K vision-only identity-readout baseline used for fair comparison with readout modules.

## 2026-07-10 12:28:20 CST: Attention Spatial Recovery 50-Image Run

- Ran full `experiments/attention_recovery.py` experiment on RTX 5060 via non-sandbox CUDA.
- Command:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/attention_recovery.py --coco_image_dir /root/coco/val2017 --selected_samples_jsonl outputs/coco_pred_attn_50/selected_samples.jsonl --prompts_jsonl outputs/coco_pred_attn_50/prompts.jsonl --generation_results_jsonl outputs/coco_pred_attn_50/generation_results.jsonl --model_name_or_path /root/Qwen3-VL-2B-Instruct --output_dir outputs/attention_recovery_50 --extract_features --mid_layer 19 --last_layer 27 --methods raw,conv,mlp,shuffle_conv --epochs 10 --lr 1e-4 --weight_decay 1e-4 --seed 42 --device cuda --dtype bfloat16 --save_visualizations --num_viz 12`
- Output directory: `outputs/attention_recovery_50`
  - Cached 102 successful cases.
  - Split cases: train=59, val=23, test=20; split images train=29, val=10, test=10.
  - Outputs include `attention_recovery_metrics_per_case.csv`, `attention_recovery_summary.csv`, `attention_recovery_summary.json`, `train_log.csv`, feature cache, and 12 visualization PNGs.
- Main test results from `attention_recovery_summary.csv`:
  - `mid_model`: Spearman-to-mid `1.0000`, entropy `4.4098`, target mass `0.4189`, pointing `0.6000`.
  - `last_model`: Spearman-to-mid `0.5989`, entropy `4.7032`, target mass `0.2302`, pointing `0.1500`.
  - `raw`: Spearman-to-mid `0.5741`, entropy `5.1854`, target mass `0.2516`, pointing `0.1500`.
  - `conv`: Spearman-to-mid `0.6932`, entropy `4.9361`, target mass `0.2800`, pointing `0.5000`.
  - `mlp`: Spearman-to-mid `0.7086`, entropy `4.8927`, target mass `0.2742`, pointing `0.4500`.
  - `shuffle_conv`: Spearman-to-mid `0.6621`, entropy `4.9312`, target mass `0.2662`, pointing `0.3500`.
- Paired quick stats on 20 test cases:
  - Spearman `conv - raw`: mean diff `+0.1191`, paired t p=`1.306e-06`, Wilcoxon p=`9.537e-06`.
  - Spearman `conv - last_model`: mean diff `+0.0943`, paired t p=`0.004633`.
  - Spearman `conv - shuffle_conv`: mean diff `+0.0311`, paired t p=`0.0002696`.
  - Spearman `conv - mlp`: mean diff `-0.0154`, paired t p=`0.0009139`; MLP is higher on Spearman.
  - Entropy `conv - raw`: mean diff `-0.2493`, paired t p=`4.473e-12`.
  - Pointing `conv - raw`: mean diff `+0.35`, paired t p=`0.01527`.
- Interpretation:
  - Supports the narrow claim that learned readout over last-layer tokens can recover attention patterns closer to mid-layer attention: `conv` improves over `raw`, `last_model`, and `shuffle_conv`.
  - Does not yet uniquely support "spatial convolution is the reason", because `mlp` is slightly better than `conv` on Spearman-to-mid. Current evidence suggests last-layer information is recoverable by learned readout, with spatial conv better than the shuffled spatial control but not better than tokenwise MLP.

## 2026-07-10 12:39:58 CST: Vision Token Logit Lens Script

- User requested a no-training Vision-Language Model Logit Lens experiment with:
  - Experiment 1: vision token semantic diversity across decoder layers.
  - Experiment 2: COCO object-region semantic alignment across decoder layers.
- Created `experiments/vision_token_logit_lens.py`.
  - Uses existing Qwen/LLaVA-style model loading, image preprocessing, hidden-state extraction, vision token index/grid inference, and COCO bbox-to-token mapping.
  - Applies the model's own final norm and `lm_head` to decoder-layer vision token hidden states only; text/generated tokens are not analyzed.
  - Supports layer sets such as `--layers 1,10,19,27 --layer_names early,mid_low,mid,last`.
  - Experiment 1 outputs per-image and summary CSV/JSON with `mean_JS`, top-k Jaccard overlap, semantic variance, vision token count, and grid metadata.
  - Experiment 2 outputs per-object-region and summary CSV/JSON with correct category logit, random category logit, random region logit, and deltas.
  - Optional visualizations draw COCO bboxes and region top predicted words per layer.
  - JS divergence defaults to an approximate top-vocab subset (`--js_vocab_top 4096`) and sampled token pairs (`--pair_samples`) for tractability; `--js_vocab_top 0` enables full vocab on very small runs.
- Verification:
  - `py_compile` passed.
  - `--help` passed.
  - Dry run selected 5 images / 11 objects with layers `[(1, early), (10, mid_low), (19, mid), (27, last)]`.
  - GPU smoke test completed:
    - Command used `--max_images 2 --max_objects_per_image 2 --layers 1,19,27 --layer_names early,mid,last --pair_samples 256 --js_vocab_top 1024 --save_visualizations`.
    - Output directory: `outputs/vision_token_logit_lens_smoke`.
    - Produced `semantic_diversity_per_image.csv`, `semantic_diversity_summary.csv/json`, `object_alignment_per_region.csv`, `object_alignment_summary.csv/json`, and two visualization PNGs.
- Smoke test example summary:
  - Semantic diversity variance decreased from early `7.2589` to mid `5.0029` to last `3.1016` on 2 images, suggesting the metric can detect homogenization-like trends.
  - Smoke sample is too small for conclusions; it only validates the implementation path.

## 2026-07-10 15:33:46 CST: Vision Token Logit Lens 50-Image Run

- User asked to run the logit lens experiment.
- Ran formal no-training `experiments/vision_token_logit_lens.py` on RTX 5060 via non-sandbox CUDA.
- Command:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/vision_token_logit_lens.py --coco_image_dir /root/coco/val2017 --coco_ann_file /root/coco/annotations/instances_val2017.json --model_name_or_path /root/Qwen3-VL-2B-Instruct --output_dir outputs/vision_token_logit_lens_50 --layers 1,5,10,15,19,23,27 --layer_names L1_early,L5_earlymid,L10_midlow,L15_mid,L19_peak,L23_late,L27_last --max_images 50 --max_objects_per_image 4 --max_objects_total 150 --min_area_ratio 0.02 --pair_samples 2048 --js_vocab_top 4096 --topk 10 --top_words 5 --random_categories 20 --device cuda --dtype bfloat16 --save_visualizations --num_viz 12 --seed 42`
- Output directory: `outputs/vision_token_logit_lens_50`
  - Selected 50 COCO val2017 images and 104 object regions.
  - Outputs include `semantic_diversity_per_image.csv`, `semantic_diversity_summary.csv/json`, `object_alignment_per_region.csv`, `object_alignment_summary.csv/json`, `run.log`, and 12 visualization PNGs.
- Semantic diversity summary:
  - `semantic_variance_mean`: L1 `7.2466`, L5 `3.4950`, L10 `4.5501`, L15 `4.5811`, L19 `4.8220`, L23 `4.7229`, L27 `3.0634`.
  - `topk_overlap_mean`: L1 `0.0116`, L19 `0.0070`, L27 `0.0231`.
  - Paired image-level differences: L27-L19 semantic variance `-1.7586`, t p=`1.392e-41`, Wilcoxon p=`1.776e-15`; L27-L23 semantic variance `-1.6595`, t p=`6.644e-50`.
- Object-region alignment summary:
  - `delta_vs_random_category_mean`: L1 `2.1206`, L5 `0.5206`, L10 `-0.1434`, L15 `-0.0152`, L19 `0.4617`, L23 `1.9651`, L27 `2.1439`.
  - `delta_vs_random_region_mean`: L1 `1.0867`, L19 `0.8629`, L23 `1.9436`, L27 `1.6055`.
  - Paired object-level differences: L27-L19 `delta_vs_random_category` `+1.6822`, t p=`2.555e-17`; L27-L19 `delta_vs_random_region` `+0.7427`, t p=`4.840e-06`.
- Interpretation/caveats:
  - Supports a semantic homogenization signal in final-layer vision tokens by reduced token-wise semantic variance and increased top-k overlap at L27.
  - Does not support a simple "final vision tokens lose all object semantics" claim: final-layer object regions still score strongly above random categories and random same-size regions.
  - The logit lens uses the model's final norm and LM head on intermediate vision-token states; results are diagnostic for semantic readability, not causal evidence about attention routing.

## 2026-07-10 15:43:27 CST: Logit Lens Result Explanation

- User asked for an explanation of the logit lens experiment data/results.
- Checked:
  - `outputs/vision_token_logit_lens_50/semantic_diversity_summary.csv`
  - `outputs/vision_token_logit_lens_50/object_alignment_summary.csv`
  - `outputs/vision_token_logit_lens_50/run.log`
- Key explanation:
  - Experiment 1 measures whether different vision tokens within the same image decode to diverse token distributions through the model final norm + LM head. L27 has much lower semantic variance (`3.0634`) than L19 (`4.8220`) and higher top-k overlap (`0.0231` vs `0.0070`), supporting final-layer semantic homogenization.
  - Experiment 2 measures whether COCO bbox-region vision tokens still favor the true category over random categories or random regions. L27 remains strongly positive on `delta_vs_random_category` (`2.1439`) and `delta_vs_random_region` (`1.6055`), so final-layer tokens still contain object semantics.
  - Overall conclusion: final-layer vision tokens become more semantically homogenized across space, but object semantics are not erased; the attention/localization failure is more consistent with spatial readout/routing degradation than complete semantic loss.

## 2026-07-10 16:11:34 CST: Target Attention Mass Experiment Index

- User asked which experiments measured Target Attention Mass and what the data were.
- Checked scripts and outputs with `rg`, `find`, and `sed`:
  - `run_coco_prediction_attention_ushape.py`
  - `run_coco_spatial_repair.py`
  - `experiments/attention_recovery.py`
  - `outputs/coco_pred_attn_50/summary_by_layer.csv`
  - `outputs/coco_spatial_repair_50/repair_summary_by_method.csv`
  - `outputs/attention_recovery_50/attention_recovery_summary.csv`
- Formal Target Attention Mass results:
  - COCO prediction-token attention: actual model attention from generated object tokens to vision tokens. Best layer L19 target mass `0.4638`; L20 `0.4493`; L23 `0.4334`; L27 drops to `0.2096`.
  - COCO spatial repair: learned readout maps over cached representations, not original model attention. `fusion_alpha_1.0` target mass `0.4544`; `shuffled_conv_last` `0.4000`; `conv_last` `0.2500`; `raw_last` `0.2000`; caveat that readout collapse/control design makes this diagnostic only.
  - Attention recovery: readout modules trained to match L19 attention, with target mass as an auxiliary bbox metric. `mid_model` `0.4189`; `last_model` `0.2302`; `raw` `0.2516`; `conv` `0.2800`; `mlp` `0.2742`; `shuffle_conv` `0.2662`.
- Conclusion:
  - Original prediction-token attention gives the cleanest Target Attention Mass evidence for a mid-layer peak and late-layer drop.
  - Repair/recovery results use target mass as a secondary localization metric over learned or reconstructed maps, so they should not be merged directly with the original-attention layer curve.

## 2026-07-10 16:14:31 CST: COCO Spatial Repair Result Suspicion/Audit

- User suspected `outputs/coco_spatial_repair_50` data were too regular.
- Audited per-case metrics, train log, and code around `BilinearAttentionReadout`, losses, and method training.
- Key findings:
  - Per-case target mass is almost entirely binary. On 20 test cases:
    - `raw_last`: 16 zeros, 4 ones; mean `0.2`.
    - `conv_last`: 15 zeros, 5 ones; mean `0.25`.
    - `mlp_last`: 15 zeros, 5 ones; mean `0.25`.
    - `shuffled_conv_last`: 12 zeros, 7 ones, 1 near-one; mean `0.4000`.
    - `fusion_alpha_1.0`: 10 zeros, 8 ones, 1 `0.9202`, 1 `0.1679`; mean `0.4544`.
  - Entropy is near zero for most methods, showing the learned/readout maps are effectively one-hot. Therefore target mass mostly reduces to whether the argmax patch lands inside the bbox, making the means look like simple fractions of 20 cases.
  - `raw_last` and `raw_mid` are not truly raw frozen baselines: the repair module is absent/frozen, but the bilinear readout `q/k` layers remain trainable.
  - `train_log.csv` has `val_loss=0.0` for all epochs because validation loss is not computed; only val metrics are recorded.
  - The bbox-supervised objective and unconstrained softmax readout can saturate, and cross-object similarity near 1 indicates same-image/readout collapse.
- Conclusion:
  - Treat `coco_spatial_repair_50` as a flawed diagnostic run, not evidence for spatial repair. The cleaner follow-up is `outputs/attention_recovery_50`, where modules are trained to match actual L19 attention and target mass is only auxiliary.

## 2026-07-10 16:28:57 CST: COCO Spatial Repair v2 Redesign And Run

- User asked to redesign and run the repair experiment after finding the previous spatial repair data suspicious.
- Created `experiments/spatial_repair_v2.py`.
  - Uses cached features from `outputs/coco_spatial_repair_50/feature_cache`; no VLM re-extraction required.
  - Two-stage design:
    - Train a middle-layer L19 anchor readout on smoothed uniform bbox target distributions.
    - Freeze the anchor readout.
    - Train only L27 repair modules so last-layer tokens become readable by the same anchor.
  - Adds anti-collapse design:
    - cosine readout with fixed temperature,
    - target smoothing,
    - entropy floor,
    - max-prob cap,
    - CE/KL to soft bbox target instead of only maximizing target mass.
  - Baselines/controls:
    - `fixed_mid_cosine`, `fixed_last_cosine`, `mid_anchor`, `last_no_repair`, `last_linear`, `last_mlp`, `last_conv`, `last_shuffle_conv`, `last_permute`, `last_readout_only`.
  - Outputs per-case metrics, summary, target-mass histogram, train log, paired tests, and run log.
- Verification:
  - `py_compile` passed.
  - `--help` passed.
  - CPU smoke run passed on 12 cached cases.
- Formal run:
  - Command used cached 102-case feature set, old image-id split, 12 anchor epochs, 12 repair epochs.
  - CUDA was unavailable in sandbox, so run fell back to CPU.
  - Output directory: `outputs/coco_spatial_repair_v2_50`.
  - Split: train=59 cases / 29 images, val=23 cases / 10 images, test=20 cases / 10 images.
- Main test summary:
  - `last_no_repair`: target mass `0.2696`, pointing `0.35`, KL `1.4317`, entropy `5.5144`, max prob `0.00723`.
  - `last_conv`: target mass `0.3569`, pointing `0.70`, KL `1.2157`, entropy `5.3163`, max prob `0.01551`.
  - `last_mlp`: target mass `0.3405`, pointing `0.65`, KL `1.3194`.
  - `last_shuffle_conv`: target mass `0.3144`, pointing `0.80`, KL `1.3738`.
  - `last_readout_only`: target mass `0.3438`, pointing `0.75`, KL `1.3368`.
  - `mid_anchor`: target mass `0.3467`, pointing `0.60`, KL `1.3263`.
- Paired tests on 20 test cases:
  - `last_conv` vs `last_no_repair`: target mass diff `+0.0873`, t p=`0.000352`; KL diff `-0.2160`, t p=`0.000600`; pointing diff `+0.35`, t p=`0.00473`.
  - `last_conv` vs `last_shuffle_conv`: target mass diff `+0.0426`, t p=`0.000436`; KL diff `-0.1581`, t p=`2.36e-05`; pointing diff `-0.10`, not significant.
  - `last_conv` vs `last_mlp`: target mass diff `+0.0164`, t p=`0.00736`; KL diff `-0.1037`, t p=`0.000158`; pointing diff `+0.05`, not significant.
  - `last_conv` vs `last_readout_only`: target mass diff `+0.0131`, not significant; KL diff `-0.1211`, t p=`0.000589`.
- Collapse diagnostics:
  - All methods have `max_prob_over_cap_fraction=0.0`.
  - Target mass is no longer mostly binary; histograms include broad intermediate buckets.
  - `near_binary_mass_fraction=0.1` remains because 2/20 test cases have near-one target mass, but entropy/max-prob show this is not global one-hot softmax collapse.
- Interpretation:
  - v2 fixes the main flaw in the original repair run and gives moderate evidence that a spatial conv repair improves calibrated bbox alignment over no-repair and shuffled-neighbor controls.
  - However, `last_readout_only` is close on target mass/pointing, so the strongest claim should be "calibrated learned readout/repair can recover some L27 spatial alignment"; the unique advantage of spatial convolution is supported mainly by KL/soft-target fit and target-mass paired tests, not by pointing accuracy.

## 2026-07-10 19:32:07 CST: COCO Attention Destination Analysis Implementation

- User requested a rigorous COCO final-layer target-object attention destination experiment, without running the full experiment.
- Inspected existing code:
  - `run_coco_prediction_attention_ushape.py`
  - `run_dot_test.py`
  - `experiments/vision_token_logit_lens.py`
  - `outputs/coco_pred_attn_50/config.json`
- Key old-definition finding:
  - Old U-shape `target_mass` is visual-conditional TAM, not absolute full-key TAM.
  - `extract_prediction_token_attention` slices attention to `vision_indices`; `compute_attention_metrics` then normalizes the visual vector before summing target bbox mass.
- Created `experiments/coco_attention_destination_analysis.py`.
  - Reuses old selected samples, prompt JSONL, generation-result success cases, model prompt formatting, generated prediction-token matching, vision token inference, grid inference, and bbox-to-patch mask logic.
  - Computes full-key `visual_attention_ratio`, `absolute_target_mass`, and visual-conditional `conditional_target_mass`.
  - Computes mutually exclusive destination regions: target, same-class-other, different-class-object, overlap, background.
  - Computes destination raw/conditional masses, densities, nTAM, object/background margins, hit@k, top-percent hit, precision, top-location labels, peak margin/ratio, normalized entropy, N_eff, centroid distance, mid-final drift/map distance, cross-query similarity, per-head metrics, best-head/correct-head fraction, and pairwise head JS.
  - Adds optional logit-lens joint analysis by image/layer semantic variance if compatible files exist.
  - Saves requested CSV/JSON/Markdown outputs and layer/head plots; overlay generation is implemented behind `--save_overlays`.
  - Supports `--resume`, `--max_cases`, layer subset, fixed mid/final layers, configurable min region token count, and `query_aware` vs `old_instance` target policy.
- Validation:
  - `py_compile` passed.
  - `--help` passed.
  - `--run_unit_tests` synthetic checks passed:
    - uniform nTAM ≈ 1,
    - one-hot entropy ≈ 0,
    - uniform normalized entropy ≈ 1,
    - pointing/top-location classification works.
  - Dry run loaded 5 successful old cases from `outputs/coco_pred_attn_50`.
  - 1-case GPU smoke with layers `19,27` completed at `outputs/coco_attention_destination_smoke`; generated per-head/per-sample/layer/paired/top-location/head-summary/cross-query/map-distance outputs and figures.
  - Overlay smoke with `--save_overlays` produced example PNGs.
- Caveats:
  - Full experiment was not run, per user instruction.
  - Old TAM reproduction check is only meaningful when running the same 102 cases; 1-case smoke correctly marks sample counts as non-comparable.
  - Bbox-to-token mapping intentionally reuses the old center-in-bbox grid method for comparability; this still does not fully model every processor resize/crop/padding detail.
  - Query-aware target union can differ from old single-bbox target; `legacy_cond_target_mass_old_bbox` is saved for direct old-U-shape reproduction.

## 2026-07-10 19:51:54 CST: COCO Attention Destination Full Run

- User asked to recheck the code and then run the rigorous COCO final-layer attention destination experiment.
- Rechecked and fixed `experiments/coco_attention_destination_analysis.py` before full runs:
  - Fixed full-key attention slicing with `np.take(..., axis=1)` so `vision_indices` do not swap head/key axes.
  - Fixed multi-subtoken aggregation by padding per-token attention rows to the max key length instead of truncating to the shortest sequence.
  - Added mean-head-map metrics and changed old TAM reproduction to compare `mean_map_legacy_cond_target_mass_old_bbox`, matching old U-shape head-mean-map then visual-renormalize behavior.
  - Changed minimum target-token filtering to use raw target bbox union before mutually exclusive overlap removal, preventing overlapped targets from being incorrectly skipped.
  - Kept `query_aware` main policy and `old_instance` compatibility policy separate.
- Validation commands:
  - `python -m py_compile experiments/coco_attention_destination_analysis.py`
  - `python experiments/coco_attention_destination_analysis.py ... --run_unit_tests`
  - 1-case CUDA smoke with layers `19,27` and overlay generation.
- Old-U-shape compatibility run:
  - Output: `outputs/coco_attention_destination_oldbbox_102_v2`
  - Command used old samples/generation results, `--target_policy old_instance --min_region_tokens 1 --mid_layer 19 --final_layer 27`.
  - Completed `analyzed=102 skipped=0`.
  - Old TAM reproduction passed: max absolute diff vs `outputs/coco_pred_attn_50/summary_by_layer.csv` = `4.59e-10`; `likely_matches_old_definition=true`.
- Main query-aware run:
  - Output: `outputs/coco_attention_destination_50_v2`
  - Command used old 50-image COCO prediction-token cases, `--target_policy query_aware --min_region_tokens 4 --mid_layer 19 --final_layer 27 --save_overlays`.
  - Completed `analyzed=102 skipped=0`.
  - Outputs include per-head/per-sample metrics, layer summary, paired stats, top-location distribution, cross-query similarity, map distances, mechanism summary, plots, and 4 overlay PNGs.
- Main paired image-target results, L19 mid vs L27 final, n=102:
  - VAR: `0.1952 -> 0.0489`, drop `0.1463`, Wilcoxon p=`1.82e-18`.
  - AbsTAM: `0.0694 -> 0.0085`, drop `0.0609`, p=`3.73e-16`.
  - head-mean CondTAM: `0.2566 -> 0.1588`, drop `0.0978`, p=`1.05e-12`.
  - mean-map CondTAM: `0.3592 -> 0.1619`, drop `0.1973`, p=`3.54e-14`.
  - nTAM: `2.5373 -> 1.0414`.
  - target-vs-other margin: `1.9648 -> 0.1295`; target-vs-background margin: `1.6330 -> -0.0833`.
  - pointing accuracy, per-head then sample mean: `0.4154 -> 0.1991`; mean-map pointing: `0.6961 -> 0.2255`.
  - normalized entropy: `0.6741 -> 0.8340`; effective token number: `26.50 -> 59.34`.
  - distance to target: `0.1623 -> 0.2056`; mean wrong-direction drift `0.0928`, positive for `81.4%` of samples.
- Top-location average over heads:
  - L19: target `0.3107`, different-class object `0.2488`, background `0.3254`.
  - L27: target `0.1526`, different-class object `0.2904`, background `0.5025`.
- Cross-query similarity within same image:
  - L19: cosine `0.2309`, JS `0.3279`, top-k Jaccard `0.0710`, same top1 `0.0000`, same object region `0.1268`.
  - L27: cosine `0.8462`, JS `0.0391`, top-k Jaccard `0.4285`, same top1 `0.2676`, same object region `0.7183`.
- Mechanism summary generated from metrics:
  - Both VAR and CondTAM drop, so total visual dependence and visual-internal target binding both degrade.
  - CondTAM/pointing drop while entropy/N_eff rise, supporting attention flattening rather than a purely peaky wrong-anchor takeover.
  - Object/background density margins both drop, with final top locations increasingly in background and different-class objects.
  - Cross-query maps become much more similar by L27, supporting shared/homogenized attention destinations.
- Caveats:
  - Bbox-to-token mapping intentionally reuses the old center-in-bbox grid method for direct U-shape comparability; it is not a full inverse of processor resize/crop/padding.
  - Logit-lens joint alignment produced only one aligned row in this run, so semantic-attention correlation is not interpretable yet.
  - Runtime warnings about empty slices come from absent optional regions such as same-class-other in some images; main target/VAR/entropy metrics are present.

## 2026-07-10 20:43:44 CST: Sparse Semantic Source Causal Trace Implementation

- User requested a strict causal experiment to test whether sparse mid-layer localized visual tokens are semantic sources that broadcast target-category semantics to many later vision tokens.
- Inspected/reused existing related scripts and outputs:
  - `experiments/vision_token_logit_lens.py`
  - `run_coco_prediction_attention_ushape.py`
  - `experiments/coco_attention_destination_analysis.py`
  - `run_coco_spatial_repair.py`
  - `run_coco_logit_lens_repair_analysis.py`
  - `outputs/coco_pred_attn_50/{selected_samples.jsonl,prompts.jsonl,generation_results.jsonl}`
- Created new causal tracing implementation:
  - `run_semantic_source_experiment.py`
  - `semantic_source/metrics.py`
  - `semantic_source/model_utils.py`
  - `semantic_source/data.py`
  - `semantic_source/corruption.py`
  - `semantic_source/semantic_scorer.py`
  - `semantic_source/hotspot_selector.py`
  - `semantic_source/matched_controls.py`
  - `semantic_source/activation_patching.py`
  - `semantic_source/statistics.py`
  - `semantic_source/sanity_checks.py`
  - `semantic_source/visualization.py`
  - `configs/semantic_source_causal_trace.yaml`
  - tests under `tests/test_semantic_source_*.py`, `tests/test_reachability.py`, `tests/test_matched_controls.py`, `tests/test_corruption_alignment.py`, `tests/test_activation_patching.py`
- Core design:
  - Reuses the same Qwen3-VL model, old 102 successful image-target pairs, prompts, generated prediction-token text, Qwen `image_grid_thw` token layout, and logit-lens definition `lm_head(final_norm(hidden_state))`.
  - Runs teacher-forced `prompt + generated_text` for clean/corrupted/patched inputs to keep text and prediction-token context aligned.
  - Uses COCO segmentation masks for target corruption and token labels when `pycocotools` is available; otherwise falls back to bbox masks.
  - Corruptions implemented: `local_blur`, `texture_shuffle`, optional `black_mask`, optional OpenCV `inpaint`.
  - Source selection uses clean-run target semantic margin at each layer, with selection/evaluation alias split where possible.
  - Activation patch point is `resid_post`: output of decoder block producing `hidden_states[source_layer]`.
  - Sufficiency: corrupted run with source positions replaced by clean activation.
  - Necessity: clean run with source positions replaced by corrupted activation.
  - Outcomes exclude directly patched source tokens and separately report reachable non-source, outside-target, target-region, other-object, and background effects.
  - Matched controls: random, region-matched, low-semantic target, norm-matched low-semantic, attention hotspot, wrong-category semantic hotspot.
  - Propagation trace records downstream per-layer source/outside/reachable/unreachable effects and BroadcastFraction.
  - Conservative mechanism summary levels are written to `mechanism_level_summary.json`; Level 3+ is required before calling a result semantic broadcast.
- Attention mask report smoke:
  - Output: `outputs/semantic_source_mask_report/attention_mask_report.md`
  - Qwen3-VL sample had vision token range `[4, 304)`, `300` vision tokens.
  - Decoder layer path: `model.model.language_model.layers`.
  - Hidden states returned: `29`; decoder blocks: `28`.
  - `hidden_states[0]` is embedding output; `hidden_states[L]` is output after decoder block `L-1`.
  - Visual tokens use `standard_causal` mask in the inspected sample.
  - Reachability rule: a non-source visual token is reachable iff its sequence position is after at least one source token.
- Validation:
  - `py_compile` passed for `run_semantic_source_experiment.py` and `semantic_source/*.py`.
  - Built-in unit checks passed: uniform/one-hot entropy, BroadcastFraction, region-matched controls.
  - Direct test-function invocation passed because `pytest` is not installed in the env.
  - `--help` works.
  - Dry run loaded 3 smoke cases from old outputs.
  - CUDA smoke `outputs/semantic_source_smoke_l19_k1_v2` completed with `forward_count=16`, `patch_rows=14`; generated attention mask report, patching CSV, propagation trace, matched-control balance, stats, plots, and mechanism summary.
- Smoke result caveat:
  - The 1-case smoke is only implementation validation, not evidence for or against the mechanism.
  - Its automatic mechanism summary correctly marks Level 1-4 as false/data-insufficient.
- Known caveats/follow-up:
  - Full experiment was intentionally not run.
  - The runner supports `--resume` in config but does not yet skip already-written individual interventions; full resumability needs row-level append/skip indexing.
  - Prediction-token secondary outcome is scaffolded as `prediction_logit_effect` but not yet computed.
  - Threshold for `coverage` defaults to `0.0`; for final use it should be calibrated from repeated corrupted baseline or control 95th percentile as specified.
  - `pytest` is unavailable in the current environment; direct function invocation was used instead.

## 2026-07-10 20:54:41 CST: Semantic Source Diagnostic-Lite Run

- User asked to proceed with the causal semantic source experiment.
- Ran validation first:
  - `python -m py_compile run_semantic_source_experiment.py semantic_source/*.py`
  - `python run_semantic_source_experiment.py --output_dir outputs/semantic_source_unit --run_unit_tests`
  - Both passed.
- Initial diagnostic-lite run to `outputs/semantic_source_diagnostic_lite_8` completed but exposed a bug:
  - `KeyError:'background'` in the optional single-token causal influence path.
  - Diagnostic mode also overrode an explicit `--single_token_exact_cases 0`.
- Fixed `run_semantic_source_experiment.py`:
  - Use `partition["masks"]["background"]`.
  - Respect explicit `--single_token_exact_cases 0`.
- Reran fixed diagnostic-lite:
  - Output: `outputs/semantic_source_diagnostic_lite_8_v2`
  - Command used `--mode diagnostic --max_cases 8 --source_layers 19,27 --semantic_layers 19,27 --trace_layers 19,20,21,23,27 --k_specs 1,4,5pct --corruption_types local_blur,texture_shuffle --control_seeds 1 --single_token_exact_cases 0 --device cuda --dtype bfloat16`.
  - Completed `forward_count=608`, `patch_rows=588`.
- Data flow/result sizes:
  - `corruption_validation.csv`: 12 clean-corrupt rows from 6 alias-eligible cases.
  - 7 corruption instances passed gap/artifact filters into patching.
  - `patching_per_sample.csv`: 588 rows.
  - `propagation_trace.csv`: 210 rows.
  - `matched_control_balance.csv`: 294 rows.
  - `statistical_tests.csv`: 520 rows.
- Attention mask report remained:
  - Vision token range `[4, 304)`, 300 vision tokens.
  - Hidden states 29, decoder blocks 28.
  - Visual tokens use standard causal mask; reachable tokens are visual tokens after the source position.
- Automatic mechanism summary from `mechanism_level_summary.json`:
  - Level 1 semantic concentration: `true`.
  - Level 2 causal semantic carrier: `false`.
  - Level 3 causal semantic source/broadcast: `false`.
  - Level 4 sparse mid-layer semantic source: `false`.
  - Supporting means:
    - mid semantic entropy `0.8216`
    - mid target semantic density `1.8167`
    - source outside-target restoration `0.0241`
    - control outside-target restoration `0.0977`
    - source outside-target necessity `0.0077`
    - control outside-target necessity `0.2400`
    - category-specific effect `0.0034`
    - final trace outside-target effect `0.00457`
    - final trace broadcast fraction `0.2550`
    - final unreachable abs effect `0.0`
- Additional extracted diagnostics:
  - L19 source trace shows outside-target effect increasing from `0` at patch layer to about `0.0091` at L27; unreachable effect stayed `0`, passing reachability sanity.
  - However, L19 source restoration/necessity did not beat matched controls overall.
  - L27 patching produced `0` non-source effect, consistent with no downstream layer for propagation.
- Interpretation:
  - This small diagnostic supports only descriptive semantic concentration/localization signals, not causal semantic broadcast.
  - Current evidence is against claiming that the selected sparse semantic hotspots are sufficient/necessary causal sources, because matched controls are stronger on the main outside-target restoration/necessity metrics in this pilot.
  - Pilot sample is small and control seeds are only 1, so treat as a debugging/diagnostic result, not final paper evidence.

## 2026-07-10 21:04:36 CST: Semantic Source Token-Level Audit

- User asked to inspect each sample's source/control token indices, token counts, bbox membership, hidden-state norm, semantic score, attention received, actual patch/ablate count, raw clean/corrupt/patched metrics, and necessity numerator/denominator.
- Generated audit CSV:
  - `outputs/semantic_source_diagnostic_lite_8_v2/token_intervention_audit.csv`
  - 588 rows, matching `patching_per_sample.csv`.
- Audit checks:
  - Actual patched/ablated token count matched the source/control k count for all rows: `count_mismatch_rows=0`.
  - Necessity/restoration denominator `clean_score - corrupt_score` had no zero-denominator rows.
  - Source/control token counts by k were as expected:
    - `k=1`: 1 token
    - `k=4`: 4 tokens
    - `k=5pct`: 13 or 15 tokens depending on visual token count.
  - Source rows: 84; control rows: 504.
- Bbox/region findings:
  - Semantic source rows were not always target-local.
  - For L19 restore rows:
    - `k=1`: source target/overlap in 4/7 rows, background-only in 2/7.
    - `k=4`: source target/overlap in 5/7 rows, background-only in 2/7.
    - `k=5pct`: source target/overlap in 7/7 rows, background-only in 0/7.
  - For L27 restore rows:
    - `k=1`: source target/overlap in 4/7 rows, background-only in 3/7.
    - `k=4`: source target/overlap in 5/7 rows.
    - `k=5pct`: source target/overlap in 5/7 rows.
- Token diagnostics:
  - Source per-token semantic scores, hidden norms, and attention received are available in `semantic_hotspots.csv` and copied into the audit for source rows.
  - Controls have actual token indices and region distributions in `patching_per_sample.csv`, plus semantic/norm means from `matched_control_balance.csv`.
  - Control per-token semantic/norm/attention values were not saved in the original diagnostic run; attention received for non-source controls is therefore blank in the audit.
- Example from `153011_truck_394633`, L19, k=1:
  - Source token `45`, region `background`, semantic `5.7291`, norm `365.31`, attention `0.00475`.
  - Restoration denominator on reachable non-source score `0.10027`; numerator `-0.00299`.
  - Necessity numerator for reverse patch `0.00138`.
- Interpretation caveat:
  - Because semantic top-1 sources are often background tokens and controls lack per-token attention diagnostics, any paper-level claim about localized bbox source tokens needs either target-constrained source analysis or a rerun with full control per-token metadata saved.

## 2026-07-10 21:36:00 CST: BBox Key-Token Causal Smoke Experiment

- User requested a stricter, auditable experiment for the hypothesis that a few middle-layer target-bbox visual tokens causally determine shared final-layer target semantics.
- Audited and reused existing modules:
  - `semantic_source.model_utils` for model loading, vision token layout, forward calls, decoder layer/hook path.
  - `semantic_source.data` for COCO cases, annotations, valid object regions, region partitioning, CSV writing.
  - `semantic_source.semantic_scorer` for final-norm + LM-head logit-lens semantic margins.
  - `semantic_source.activation_patching.run_with_resid_post_patch` for resid-post intervention after decoder block `m-1`.
- Added `experiments/bbox_key_token_causal.py`.
  - Source selection is strictly inside target bbox tokens using `bbox_overlap_ratio >= 0.5`; center-inside mask is also saved for sensitivity.
  - Saves per-token grid/cell coordinates, bbox overlap, center-inside flag, normalized bbox-center distance, semantic scores, hidden norms, prediction-token attention received, all-text attention received, source/control type, donor pairing, matching distance, and intervention metadata.
  - Conditions: clean, source/control bbox mean replacement, zero-ablation sanity, donor-category replacement raw, donor-category replacement norm-rescaled.
  - Controls: bbox random, bbox norm-matched, bbox attention/norm/spatial matched.
  - Region metrics: same_position, bbox_non_source, all_non_source, background_non_source, other_object_regions.
  - Outputs observation, destruction, donor replacement, paired stats, mechanism summary, sanity report, JSONL raw rows, CSV, layerwise trace, and figures.
  - Semantic-component removal is explicitly marked optional/not run because no validated layer-specific category direction exists.
- Validation:
  - `py_compile` passed.
  - `--help` passed.
  - Dry run loaded 5 successful old COCO prediction-attention cases.
- Final smoke run:
  - Command: `/root/miniconda3/envs/next3d_qwen/bin/python experiments/bbox_key_token_causal.py --output_dir results/bbox_key_token_causal/smoke_v3 --max_cases 5 --k_values 1,2 --random_control_repeats 1 --trace_layers 19,20,27 --device cuda --dtype bfloat16`
  - Required non-sandbox GPU access; sandbox CUDA reported no NVIDIA driver.
  - Output directory: `results/bbox_key_token_causal/smoke_v3`
  - Processed 3 valid cases, skipped 2 because no usable single-token target alias.
  - Effect rows: 104; token metadata rows: 72; layerwise trace CSV lines: 91; forward_count: 83; elapsed about 226 s.
- Smoke sanity:
  - clean rerun max abs error `0.0`
  - source tokens inside bbox fraction min `1.0`
  - source/control counts equal and masks non-overlapping: true
  - bbox_non_source/source masks non-overlapping: true
  - direct intervention touched only selected visual tokens: true
  - text tokens unchanged at patch layer: true
  - donor norm ratio range `0.9973` to `1.0034`
  - empty-mask and invalid-index filtered intervention final error `0.0`
  - source semantic score higher than random bbox control fraction mean `1.0`
- Smoke result caveat:
  - This smoke is implementation/audit only. With 3 valid cases and k in `{1,2}`, source interventions did not outperform norm/attention matched controls on all-non-source category shift; mechanism summary therefore gives no causal-broadcast support from the smoke.
  - Full experiment should run larger k values `{1,2,4,8,16}` with 10 random controls after reviewing `sanity_check_report.json` and the saved bbox/source/control visualizations.

## 2026-07-11 CST: Follow-Up Experiment Summary

- User requested a Chinese summary of experiments added beyond the original paper, including design, data, and conclusions.
- Summarized completed extensions: corrected synthetic dot test; 50-image/102-case COCO generated prediction-token attention U-shape; attention-destination decomposition; attention recovery; no-training vision-token logit lens; spatial repair v2.
- Also separated diagnostic-only causal work from completed evidence: semantic-source diagnostic-lite (7 valid corruption instances) did not support causal broadcast; bbox key-token causal test is only a 3-case smoke and not paper evidence.
- Main supported synthesis: L27 loses visual dependence, target binding, and query specificity while becoming more background-directed/homogenized; its vision tokens still retain object semantics and some spatial alignment can be recovered by calibrated learned readout. Strong claims of uniquely convolutional repair or sparse causal semantic broadcasting remain unsupported.

## 2026-07-11 CST: Logit-Lens Prompt Constraint Clarification

- User asked whether the vision-token logit-lens object-semantic result strictly constrained the prompt to one given object.
- Checked `experiments/vision_token_logit_lens.py`: no. Every image used the same generic prompt, `Question: What objects are visible in the image? Answer briefly.`
- Object-specificity is an evaluation-side COCO annotation constraint: up to four largest valid annotated objects per image are mapped from bbox to vision tokens, then their true category-token logit is compared with random category logits in the same region and the true-category logit in random same-size regions. The model is neither told the target object name nor required to generate it.

## 2026-07-11 12:59 CST: Late Visual-Access Causal Ablation Smoke

- User requested a new causal experiment testing whether target-specific visual evidence is written into the prediction-position residual stream by middle layers, so later direct visual access can be removed with little effect.
- Added `experiments/late_visual_access_ablation.py`.
  - Reuses `outputs/coco_pred_attn_50` successful generated-object cases via `build_case_index`, existing prompt preprocessing, cached token-offset matching, vision-token span/grid inference, and Qwen model loader.
  - Teacher-forces the cached continuation by appending its token ids to the original generation prefill; it audits that the main first-content object token is scored by `logits[prediction_position]`, with `prediction_position = generated_input_position - 1`.
  - Uses a `self_attn` forward hook that reconstructs post-softmax per-head visual value contribution and subtracts `W_o(sum_visual alpha V)` only at the selected query position. Main `zero_contribution` does not renormalize text weights; optional `renormalized_text`, random-position, random-layer-count, full-text/nonvisual controls, image-clustered bootstrap/permutation/Wilcoxon/FDR summaries, and optional no-KV-cache regeneration are implemented.
- Validation commands:
  - `python -m py_compile experiments/late_visual_access_ablation.py`
  - `python experiments/late_visual_access_ablation.py --help`
  - CUDA smoke: `python experiments/late_visual_access_ablation.py --smoke-test --output-dir results/late_visual_access_ablation/smoke --layers 19,27 --bootstrap-samples 100 --permutation-samples 1000 --device cuda --dtype bfloat16`
- Smoke output: `results/late_visual_access_ablation/smoke`.
  - Processed 3 images / 3 object cases, 15 result rows, no failures.
  - No-op active-hook max logit error `0.0`; visual + nonvisual output reconstruction max absolute error `1.0` but BF16 relative error `0.00763`, passing the BF16-aware `<0.02` reconstruction criterion; position and layer specificity passed for every intervention.
  - Illustrative only: cumulative L19 ablation mean target Δlogp `-1.1290` with top-1 preservation `0.667`; L27-only cumulative mean Δlogp `-0.0106`, preservation `1.0`. With 3 images this is a hook-validation smoke, not a formal causal result.
- Caveats/follow-up:
  - Hook implementation currently targets Qwen3-VL eager attention and requires returned attention weights; it refuses SDPA/Flash paths rather than silently approximating the intervention.
  - Formal run is intentionally not executed; use the generated command in the script/report with all cached cases and full controls.

## 2026-07-11 13:50 CST: Late Visual-Access Formal Run Stopped

- User asked to run the full late visual-access experiment, then explicitly asked to stop it and summarize available data.
- Full CUDA command used all 102 cached successful object-token cases / 49 images, layers `1,5,10,15,19,20,21,23,25,27`, single/cumulative/random-position/random-layer-count/full-text controls, BF16, and regeneration for the first 12 cases.
- One accidental duplicate process was detected early and stopped; the original PID 228201 was retained and monitored as the sole writer until the user requested termination.
- Original run reached the start of case 61/102 (`210520_dining_table_60`), meaning cases through the preceding case had executed in memory; no traceback or per-case failure appeared in `run.log`.
- Important persistence caveat: the retained process had started before per-case checkpoint support was patched into `experiments/late_visual_access_ablation.py`. After it was stopped, the formal directory contained only `config.json` and `run.log`; no formal `per_sample_results.csv`, `audit.json`, summaries, or report were available. Formal numeric effects therefore cannot be recovered from this interrupted run.
- Available completed data remain the 3-image smoke under `results/late_visual_access_ablation/smoke`:
  - prediction-position audit passed for 3/3; clean target rank was 1 for all.
  - no-op max logit error `0.0`.
  - reconstruction max absolute error `1.0`, BF16-relative max error `0.0076336`; reconstruction check passed.
  - cumulative L19 mean delta logp `-1.12895`, top-1 preservation `0.667`; cumulative L27 mean delta logp `-0.01058`, preservation `1.0`.
  - n=3 image-level permutation/FDR p-values were about `0.24-0.27`, so smoke trends are not formal evidence.
- Added durable per-case `per_sample_results.csv`, regeneration CSV, and `audit_checkpoint.json` writes plus `--resume` support to the script for any future run; this change was not active in the stopped original process.

## 2026-07-11 17:22 CST: Vision-Token Landmark Probe Audit

- User asked why repair experiments find MLP and convolution similarly capable, while the landmark experiment appears to show a large convolution advantage.
- Located the relevant comparison in `geometry_probe_exp.py`, with recorded output in `paper/figures/fixed_test/geometry_probe_results.json`. Other landmark scripts (`midlayer_landmark_probe.py`, `multiseed_midlayer.py`, `landmark_viz.py`, and `skip_connection_probe.py`) use only CNN probes and are not Conv-vs-MLP comparisons.
- The apparent discrepancy is architectural rather than contradictory evidence:
  - Landmark `PoolMLPProbe` first computes `x.mean(dim=1)`, discarding the identities/order of all 64 vision-token grid cells before its MLP. It is permutation-invariant and cannot recover landmark locations from arrangement.
  - Landmark `ConvNetProbe` retains the full 8x8 layout and then flattens it into a position-specific linear head. It therefore differs in both spatial information access and parameter count, not just 3x3 neighbor mixing.
  - Parameter audit: landmark ConvNet `7,147,016`; landmark pooled MLP `1,118,856`; pooled linear `278,664`. By contrast spatial-repair-v2 MLP and Conv modules are closely matched (`2,107,904` vs `2,113,024`) and both preserve one output per vision token; attention recovery likewise uses a per-token MLP, not a pooled MLP.
- Code/result caveats found:
  - `geometry_probe_exp.py` always z-scores features (lines 304-308), but `paper/main.tex` claims its reported main result is unnormalized. The checked artifact instead reports normalized results: L18-20 ConvNet `73.1`, PoolMLP `18.0`, PoolLinear `0.0`; the separate 500-epoch pooled rerun gives PoolMLP `18.5`, PoolLinear `0.0`. These do not match the paper/PROJECT_PROGRESS claim `68.4/29.6/29.6`, for which no matching saved artifact was found.
  - The ConvNet has BatchNorm and a larger position-aware flattening head; the pooled MLP has neither. Thus the comparison supports only that global mean pooling destroys landmark information, not that convolution is uniquely or overwhelmingly better than an MLP that retains grid positions.
  - Vision token span (`vs=4`, special id `151653`) and 8x8 grid are hard-coded without assertions; correct for the recorded AFLW run but brittle for changed processors/resolutions. If an image is skipped, cached features and `lms_sel` can become misaligned.
- Recommended fair follow-up: use one shared retained-grid head for all models, compare (i) identity / no repair, (ii) tokenwise MLP, (iii) 1x1 pointwise network with the same spatially indexed flattening head, (iv) true 3x3 convolution, and (v) shuffled-grid 3x3 convolution. Match parameter budgets, optimizer/normalization/early stopping, and fixed image-level splits. Only Conv vs 1x1/tokenwise + shuffle controls can isolate local-neighbor mixing.

## 2026-07-11 17:45 CST: Fair Spatial-MLP Landmark Probe Implementation

- User asked to modify the landmark MLP and rerun the experiment.
- Updated `geometry_probe_exp.py`:
  - Added `SpatialMLPProbe`: a per-token MLP that preserves all 8x8 token slots and uses the same Flatten + coordinate head as `ConvNetProbe`; it has no local token-neighbour mixing.
  - Set token hidden width to 2304 so its two MLP matrices match the two ConvNet 3x3 convolution matrix budgets. Total parameters: ConvNet `7,147,016`, SpatialMLP `7,148,296` (difference `1,280`).
  - Kept `PoolMLP`/`PoolLinear` as explicitly pooled negative controls; added CLI `--output_dir`, `--output_name`, and `--probes` so old artifacts are not overwritten. The intended formal output is `paper/figures/geometry_fair_spatial_mlp/geometry_fair_spatial_mlp_results.json`.
- Verification passed:
  - `python -m py_compile geometry_probe_exp.py`
  - `python geometry_probe_exp.py --help`
  - CPU shape smoke for ConvNet, SpatialMLP, and PoolMLP: all produce `[2, 68, 2]` from `[2, 64, 2048]`.
- Formal run command was started twice with the requested configuration (200 AFLW images; ConvNet, SpatialMLP, PoolMLP, PoolLinear; 3 layer groups, 3 seeds, 4 data fractions), but it cannot begin feature extraction: the RTX 5060 has `7417-7697 / 8151 MiB` occupied at `99-100%` utilization by an external process that is not exposed in `nvidia-smi`'s process list. Qwen finishes loading and is then terminated by memory pressure; no result JSON was written. Do not treat this as an experiment result. Resume the same command after GPU memory is released.

## 2026-07-11 17:47 CST: Fair Landmark Probe Retry

- User asked to retry the formal fair landmark experiment.
- GPU status remained blocked before retry: `7583 / 8151 MiB` used and `100%` utilization. The same formal command was launched once more; it again reached only Qwen3-VL weight loading and then exited before printing the decoder-layer count, extracting a feature, or writing a result JSON. No experiment data were produced.
- `pgrep` found no surviving `geometry_probe_exp.py` worker after the attempt. The external GPU allocation remains the sole blocker; no external process was terminated.

## 2026-07-11 18:05 CST: Fair Spatial-MLP Landmark Probe Completed

- User asked to run the modified landmark experiment.
- Ran `geometry_probe_exp.py` on RTX 5060 after confirming GPU was free (`0 MiB / 8151 MiB` used).
- Command:
  - `/root/miniconda3/envs/next3d_qwen/bin/python geometry_probe_exp.py --output_dir paper/figures/geometry_fair_spatial_mlp --output_name geometry_fair_spatial_mlp_results.json --probes ConvNet,SpatialMLP,PoolMLP,PoolLinear`
- Output created:
  - `paper/figures/geometry_fair_spatial_mlp/geometry_fair_spatial_mlp_results.json`
- Run configuration:
  - 200 AFLW2000 images, 40-image test split, 160-image train pool, 3 seeds.
  - Layer groups: `L1-3`, `L18-20`, `Last`.
  - Probes: `ConvNet`, parameter-matched position-preserving `SpatialMLP`, pooled `PoolMLP`, pooled `PoolLinear`.
  - Hidden states normalized by train-set per-channel z-score.
- Key PCK@0.05 full-data results:
  - `ConvNet`: L1-3 `74.0±1.5`, L18-20 `73.1±1.8`, Last `66.9±1.0`.
  - `SpatialMLP`: L1-3 `61.7±5.2`, L18-20 `65.0±3.7`, Last `59.4±3.5`.
  - `PoolMLP`: L1-3 `16.9±1.0`, L18-20 `18.5±1.5`, Last `16.9±0.7`.
  - `PoolLinear`: all layer groups `0.0±0.0`.
  - Per-landmark train-mean baseline: `28.6±0.8`.
- Geometry summary:
  - Adjacent-token cosine: L1-3 `0.7766`, L18-20 `0.8494`, Last `0.4809`.
  - Effective rank: L1-3 `34.4`, L18-20 `32.39`, Last `43.29`.
- Conclusion/caveat:
  - The modified fair control shows retained-grid SpatialMLP recovers substantial landmark information and is far stronger than pooled MLP, so the old ConvNet-vs-PoolMLP gap mostly reflected pooling destroying spatial layout.
  - ConvNet still outperforms SpatialMLP by about 6-8 PCK points at full data, suggesting local spatial mixing/ConvNet inductive bias helps, but the evidence is now moderate rather than the old overwhelming pooled-control contrast.

## 2026-07-11 22:36:32 CST: Prediction-Token Attention Homogenization Result Lookup

- User asked to find the experimental data/results for prediction-token attention homogenization.
- Checked:
  - `outputs/coco_pred_attn_50/summary_by_layer.csv`
  - `outputs/coco_pred_attn_50/run_report.md`
  - `outputs/coco_attention_destination_50_v2/layer_summary.csv`
  - `outputs/coco_attention_destination_50_v2/mid_final_paired_statistics.csv`
  - `outputs/coco_attention_destination_50_v2/cross_query_similarity.csv`
  - `outputs/coco_attention_destination_50_v2/top_location_distribution.csv`
- Main original COCO prediction-token result:
  - 50 COCO val2017 images, 102 successful generated prediction-token target cases, 1988 same-image cross-object rows.
  - Target mass peaks at L19 `0.4638`, L20 `0.4493`, then drops to L27 `0.2096`.
  - Cross-object Pearson similarity is lowest at L20 `0.1503`; L19 `0.1569`; L27 rebounds to `0.7850`.
  - Cross-object cosine similarity: L19 `0.2309`, L20 `0.2288`, L27 `0.8462`.
  - This is the cleanest direct evidence for late-layer prediction-token attention homogenization.
- Main destination-analysis confirmation:
  - `outputs/coco_attention_destination_50_v2` reproduces old TAM with max abs diff `4.59e-10`.
  - L19 -> L27, n=102: VAR `0.1952 -> 0.0489`, AbsTAM `0.0694 -> 0.0085`, head-mean CondTAM `0.2566 -> 0.1588`, mean-map CondTAM `0.3592 -> 0.1619`.
  - Cross-query same-image maps: cosine `0.2309 -> 0.8462`, JS `0.3279 -> 0.0391`, top-k Jaccard `0.0710 -> 0.4285`, same top1 `0.0000 -> 0.2676`, same object region `0.1268 -> 0.7183`.
  - Entropy/effective token count increase: normalized entropy `0.6741 -> 0.8340`, N_eff `26.50 -> 59.34`, supporting flattening/shared destination rather than sharper object-specific attention.
  - Top-location distribution over heads shifts from L19 target `0.3107`, background `0.3254` to L27 target `0.1526`, background `0.5025`.
- Conclusion:
  - The strongest phrase is: generated prediction-token attention is object-specific in mid layers but becomes substantially more similar/shared by L27, with lower visual dependence, lower target binding, higher entropy, and more background-directed destinations.

## 2026-07-11 22:?? CST: Semantic Source Layer Intervention Result Lookup

- User asked to review the experiment measuring how intervening on/truncating selected vision tokens at different layers affects semantics.
- Checked `outputs/semantic_source_diagnostic_lite_8_v2/{analysis_summary.md,mechanism_level_summary.json,semantic_concentration_per_sample.csv,statistical_tests.csv,propagation_trace.csv,token_intervention_audit.csv,attention_mask_report.md}` and `AGENTS.md`.
- Relevant run: diagnostic-lite semantic source causal trace, `max_cases=8`, source/semantic layers `19,27`, final layer `27`, trace layers `19,20,21,23,27`, k `1,4,5pct`, corruptions `local_blur,texture_shuffle`; 608 forward passes, 588 patch rows, 210 trace rows, 7 valid corruption instances.
- Descriptive semantics: L19 and L27 both show semantic concentration. Aggregated semantic concentration rows: L19 entropy `0.8216`, effective tokens `88.06`, top-5pct mass `0.2653`, target semantic density `1.8167`; L27 entropy `0.8111`, effective tokens `80.57`, top-5pct mass `0.2982`, target semantic density `2.5408`.
- Causal result: mechanism summary supports only Level 1 semantic concentration. Level 2 causal carrier, Level 3 broadcast, and Level 4 sparse mid-layer source are all false.
- Main causal means: source outside-target restoration `0.0241` vs controls `0.0977`; source outside-target necessity `0.0077` vs controls `0.2400`; category-specific effect `0.0034`; final trace outside-target effect `0.00457`; final broadcast fraction `0.2550`.
- Layer-specific patching caveat: L27 patching yields zero downstream non-source effect because patch point is final-layer resid-post with no later layer for propagation; this should not be read as “L27 has no semantics.”
- Interpretation: The pilot shows vision-token semantic concentration, including at L27, but does not support a causal claim that sparse selected mid-layer vision tokens are sufficient/necessary semantic sources broadcasting category semantics. Sample is small and control seed count is 1; target-constrained rerun or larger bbox-key-token causal run is needed before paper claims.

## 2026-07-12 15:09 CST: Late Visual-Access Ablation Full Rerun

- User asked to rerun the ablation experiment completely.
- Confirmed prior formal `results/late_visual_access_ablation` run had only `config.json`/`run.log` and had stopped around case 61/102; no usable checkpoint files were present, so reran from scratch.
- Command:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/late_visual_access_ablation.py --cache-path outputs/coco_pred_attn_50 --model /root/Qwen3-VL-2B-Instruct --output-dir results/late_visual_access_ablation --device cuda --dtype bfloat16 --layers 1,5,10,15,19,20,21,23,25,27 --ablation-mode zero_contribution --conditions clean,single_layer,cumulative,random_position,random_layer_count --include-full-text-control --max-objects-per-image 9999 --regenerate-subset 12 --bootstrap-samples 10000 --permutation-samples 10000 --epsilons 0.1,0.25,0.5 --seed 42`
- Outputs written in `results/late_visual_access_ablation`:
  - `per_sample_results.csv`, `layer_summary.csv`, `group_summary.csv`, `image_cluster_summary.csv`, `regeneration_results.csv`, `audit.json`, `audit_checkpoint.json`, `progress.json`, `report.md`, `run.log`, and four PNG summary plots.
- Completion/validation:
  - Process exited code 0.
  - Finished `5202` rows over `102` successful object cases / `49` images.
  - Prediction-position audit: 102 audited cases, 0 failed cases.
  - Internal checks: `no_op_pass=true`, `reconstruction_pass=true`; no-op max logit error `0.0`, reconstruction max relative error `0.015075`.
  - `regeneration_results.csv` has 132 rows for the 12-case regeneration subset (`clean` and `cumulative`).
- Main visual cumulative ablation results:
  - L1 Δlogp `-1.7536`, top-1 preserved `0.4915`, BH p `0.000238`.
  - L19 Δlogp `-1.4323`, top-1 preserved `0.5340`, BH p `0.000238`.
  - L20 Δlogp `-1.0501`, top-1 preserved `0.6105`, BH p `0.000238`.
  - L21 Δlogp `-0.5124`, top-1 preserved `0.7721`, BH p `0.000238`.
  - L23 Δlogp `-0.1230`, top-1 preserved `0.9218`, BH p `0.03055`.
  - L25 Δlogp `-0.0647`, top-1 preserved `0.9286`, BH p `0.1062`.
  - L27 Δlogp `-0.00054`, top-1 preserved `0.9524`, BH p `0.8563`.
- Group summaries:
  - `early` Δlogp `-1.7554`, BH p `0.000167`.
  - `middle` Δlogp `-1.7784`, BH p `0.000167`.
  - `L19_neighborhood` Δlogp `-0.9983`, BH p `0.000167`.
  - `late` Δlogp `-0.09385`, BH p `0.01562`.
  - `final` Δlogp `-0.00054`, BH p `0.8325`.
- Interpretation:
  - The completed run supports the narrow causal-readout claim that direct visual value contribution to the prediction position is strong through mid/L19-L20 layers, drops sharply after L21-L23, and is effectively absent by L27 under this intervention.
  - Full-text attention ablation remains disruptive even at L27 (`Δlogp=-0.1036`, top-1 preserved `0.8980`, BH p `0.0004545`), so this should be phrased as loss of direct visual-access dependence, not absence of all late-layer information paths.
- Caveat:
  - SciPy emitted a Wilcoxon runtime warning during summary for degenerate paired differences, but the experiment completed and all required outputs were written.

## 2026-07-13 12:39:44 CST: Vision Token Logit Lens mean_JS Explanation

- User asked what the direct layerwise Logit Lens experiment's `mean_JS` metric means and how to interpret the results.
- Checked `experiments/vision_token_logit_lens.py`, `outputs/vision_token_logit_lens_50/semantic_diversity_summary.csv`, `object_alignment_summary.csv`, and `run.log`.
- `mean_JS` is the mean Jensen-Shannon divergence between logit-lens probability distributions of sampled pairs of vision tokens within the same image/layer.
  - For the 50-image run it sampled up to 2048 vision-token pairs per image and used the top 4096 vocab ids by mean logit as an approximation.
  - Higher `mean_JS` means vision tokens decode to more different semantic distributions; lower `mean_JS` means more similar/homogenized distributions.
- Key values: L1 `0.6057`, L5 `0.4570`, L10 `0.5148`, L15 `0.5026`, L19 `0.5358`, L23 `0.6416`, L27 `0.5811`.
- Interpretation caveat: `mean_JS` alone does not show final-layer homogenization versus L19 because L27 mean_JS is higher than L19. The stronger homogenization evidence in this run comes from L27's much lower semantic variance (`3.0634` vs L19 `4.8220`) and higher top-k overlap (`0.0231` vs `0.0070`), while object alignment shows L27 still preserves category semantics above random controls.

## 2026-07-14 15:09:21 CST: CLEVR Causal Semantic Bottleneck Pipeline Scaffold

- User requested a CLEVR counterfactual intervention experiment for testing middle-layer semantic compression in LVLMs, explicitly not using attention weights to define critical regions.
- Created `experiments/clevr_bottleneck/`:
  - `prepare_data.py`: reads CLEVR scenes, filters 3-5 object scenes, builds unique color/shape questions from ground-truth attributes, optionally enforces mask-based target area and bbox-overlap occlusion filters, writes `cases.jsonl`/`metadata.json`.
  - `generate_cf.py`: creates strict scene-graph counterfactuals for `original`, `critical`, `noncritical`, and `spatial`; validates unchanged objects by field comparison.
  - `render_cf.py`: deterministic renderer wrapper using a user-provided CLEVR renderer command template; no diffusion or pixel editing.
  - `map_tokens.py`: maps CLEVR object masks to LVLM vision-token grid by patch centers, without attention.
  - `inference.py`: loads Qwen3-VL via existing repo utilities and saves layer-wise text hidden vectors/logits for each condition.
  - `analysis.py`: computes layer-wise representation shift, KL logit shift, and semantic concentration score `critical/(critical+noncritical)`, with optional plots.
  - `ablation.py`: uses mask-defined token mappings to replace target/background vision tokens with mean vision token at a chosen decoder layer and measures downstream text hidden shift.
  - `README.md`: documents the end-to-end command sequence and caveats.
- Verification commands run:
  - `/root/miniconda3/envs/next3d_qwen/bin/python -m py_compile experiments/clevr_bottleneck/*.py`
  - `--help` checks for all seven scripts.
  - Synthetic `/tmp/clevr_smoke` data validated `prepare_data.py`, `generate_cf.py`, `render_cf.py --dry_run`, `map_tokens.py`, `inference.py --dry_run`, `ablation.py --dry_run`, and `analysis.py` on fake hidden caches.
- Caveats/follow-up:
  - No formal CLEVR run was executed because no CLEVR dataset/renderer was present under `/root`.
  - Strict area/occlusion filtering and token ablation require object masks from the CLEVR renderer or a compatible mask export.
  - `render_cf.py` expects a renderer command that can consume scene JSONs and write deterministic images/scenes; the exact command depends on the local CLEVR renderer installation.

## 2026-07-14 15:48:03 CST: CLEVR Data, Renderer, And Blender Download

- User asked to directly download required CLEVR data and renderer.
- Downloaded official CLEVR no-images package to `data/clevr/CLEVR_v1.0_no_images.parallel.zip`.
  - Source: `https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip`
  - Size: `89363837` bytes.
  - SHA256: `d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4`.
  - `unzip -t` passed.
  - Extracted to `data/clevr/CLEVR_v1.0/`.
  - Verified counts: `CLEVR_val_scenes.json` 15000 scenes, `CLEVR_train_scenes.json` 70000 scenes, `CLEVR_val_questions.json` 149991 questions.
- Cloned official CLEVR renderer/generation repo to `data/clevr/clevr-dataset-gen`.
  - Source: `https://github.com/facebookresearch/clevr-dataset-gen`
  - Commit: `f0ce2c81750bfae09b5bf94d009f42e055f2cb3a`.
  - Renderer assets present under `image_generation/data/`, including `base_scene.blend`, shapes, materials, and `properties.json`.
- Downloaded Blender 2.79b binary needed by the official renderer.
  - Source: `https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2`
  - Size: `143200446` bytes.
  - SHA256: `43824a4e0b0c6de6fa34ff224eec44c1cc9f26a95f6f3c8c2558d1c05704183c`.
  - Extracted to `data/clevr/blender-2.79b-linux-glibc219-x86_64/` using `tar --no-same-owner`.
  - Added `experiments/clevr_bottleneck/blender279.sh` wrapper to set `LD_LIBRARY_PATH` for bundled `libGLU.so.1`.
  - `experiments/clevr_bottleneck/blender279.sh --version` passed: Blender `2.79`, hash `f4dc9f9d68b`.
- Download notes/caveats:
  - Started the full 18 GB CLEVR zip but stopped after ~23 MB because the no-images package is sufficient for scene graphs when re-rendering originals/counterfactuals; partial file remains at `data/clevr/CLEVR_v1.0.zip`.
  - Partial single-connection no-images file remains at `data/clevr/CLEVR_v1.0_no_images.zip`; complete package is `CLEVR_v1.0_no_images.parallel.zip`.
  - Parallel download part directories remain at `data/clevr/no_images_parts/` and `data/clevr/blender_parts/`.
  - A 1-image low-resolution official renderer smoke test was attempted with Blender background mode but produced no output after several minutes and was interrupted; renderer code and Blender binary are installed, but actual rendering still needs a focused compatibility/debug pass.

## 2026-07-14 16:11:21 CST: CLEVR Renderer Debug And Smoke Experiment Run

- User asked to get the CLEVR renderer working, process data, and run the experiment.
- Fixed/created CLEVR renderer path:
  - Added `experiments/clevr_bottleneck/render_scene_json.py` support for resolving official renderer assets from `data/clevr/clevr-dataset-gen/image_generation`.
  - Added Blender 2.79 world-color compatibility (`horizon_color` fallback).
  - Replaced fragile one-pass color mask rendering with per-object shadeless mask renders and binary PGM instance maps.
  - Updated `experiments/clevr_bottleneck/common.py` to load `.pgm` masks and clamp cosine distance to non-negative numeric range.
  - Updated `experiments/clevr_bottleneck/map_tokens.py` to find masks under `mask_dir/<condition>/<case_id>.pgm`.
  - Added `experiments/clevr_bottleneck/vlm_utils.py`; `inference.py` and `ablation.py` now load Qwen3-VL via `Qwen3VLForConditionalGeneration`/`AutoModelForImageTextToText` instead of the older causal-LM fallback.
- Renderer/data smoke run:
  - Used `prepare_data.py --allow_missing_images --max_cases 4` on `data/clevr/CLEVR_v1.0/scenes/CLEVR_val_scenes.json`; output `outputs/clevr_bottleneck/metadata_smoke`.
  - Used `generate_cf.py` to create 4 case groups / 16 original-critical-noncritical-spatial rows; output `outputs/clevr_bottleneck/dataset_smoke/manifest.jsonl`.
  - Rendered all 16 rows with Blender 2.79 at `224x168`, 8 samples, 0/1 bounces; outputs under `outputs/clevr_bottleneck/dataset_smoke`, `rendered_scenes_smoke`, and `masks_smoke`.
  - Verified PGM masks have instance values `0..4` and no temp PNGs remained.
  - Ran `map_tokens.py` with `/root/Qwen3-VL-2B-Instruct`; Qwen processor inferred a `7x10` visual-token grid for all 16 rows. Target tokens were nonempty, typically 1-5 tokens depending on object and spatial condition.
  - Generated counterfactual montage: `outputs/clevr_bottleneck/visualizations_smoke/counterfactual_grid.png`.
- Qwen3-VL experiment smoke run:
  - Ran `inference.py` on 4 cases / 16 rows using CUDA bfloat16 outside sandbox; saved 16 `.pt` hidden/logit payloads in `outputs/clevr_bottleneck/hidden_states_smoke`.
  - Ran `analysis.py --plot`; outputs in `outputs/clevr_bottleneck/analysis_smoke`.
  - Analysis rows: 4 cases, 348 layer-shift rows, 12 KL rows.
  - Logit KL critical vs original was large: `17.75-21.38`; noncritical/spatial KL was near zero except one shape noncritical case (`4.264`).
  - Semantic concentration smoke results:
    - color: best critical-minus-noncritical at layer 28, critical shift `0.1587`, noncritical `0.00217`, delta `0.1565`; best SC at layer 26, `0.9940`.
    - shape: best critical-minus-noncritical at layer 28, critical shift `0.1943`, noncritical `0.0412`, delta `0.1531`; best SC at layer 26, `0.8630`.
  - Ran `ablation.py` on original condition, intervention layer 19, regions target/background; output `outputs/clevr_bottleneck/ablation_smoke`.
    - color best hidden shift at readout layer 28: background `0.03069`, target `0.03039` (`n=1`).
    - shape best hidden shift at readout layer 28: background `0.01689`, target `0.00224` (`n=3`).
- Caveats/follow-up:
  - This is a pipeline smoke run, not a formal scientific result: only 4 case groups from one CLEVR scene image index (`12858`), low render resolution, and several target objects map to only 1-2 visual tokens.
  - `prepare_data.py` was run without source object masks, so initial area/occlusion filters were not strict; rendered PGM masks now make a stricter second-pass filter possible.
  - Ablation replaces many more background tokens than target tokens, so smoke ablation magnitudes are not directly comparable without token-count-matched controls.
  - Formal run should regenerate a larger filtered set with rendered masks, higher resolution, balanced color/shape tasks, and per-condition validation before paper-level claims.

## 2026-07-14 16:43:30 CST: CLEVR Bottleneck Formal Strict Run

- User asked to proceed with the formal CLEVR counterfactual experiment.
- Added `experiments/clevr_bottleneck/filter_rendered_cases.py`.
  - Filters rendered candidate cases using original-condition PGM object masks.
  - Criteria used: target visible area ratio `>=0.05`, bbox IoU `<=0.35`, estimated target tokens `>=2`, max 40 cases.
- Formal dataset construction:
  - Prepared 700 CLEVR-val candidate cases from scene graphs with `prepare_data.py --allow_missing_images --max_cases 700 --seed 123`.
  - Generated counterfactual groups with `generate_cf.py`.
  - Rendered 700 original candidate images/masks at `320x240`, 8 samples, 0/1 bounces, using Blender 2.79 and `render_scene_json.py`.
  - Strict rendered-mask filter kept 35 cases; rejects were `{'target_area': 665}`.
  - Final formal set: `outputs/clevr_bottleneck/dataset_formal_strict_700`, 35 cases / 140 rows across original, critical, noncritical, spatial.
  - Rendered outputs: 140 PNGs in `dataset_formal_strict_700`, 140 PGM masks in `masks_formal_strict_700`.
  - Token mapping output: `outputs/clevr_bottleneck/token_mapping_formal_strict_700`; Qwen visual grid inferred as `8x10` / 80 merged vision tokens for all rows.
- Qwen3-VL inference and analysis:
  - Ran `experiments/clevr_bottleneck/inference.py` on all 140 rows with `/root/Qwen3-VL-2B-Instruct`, CUDA, bfloat16.
  - Hidden/logit payloads: `outputs/clevr_bottleneck/hidden_states_formal_strict_700`, 140 `.pt` files.
  - Ran `experiments/clevr_bottleneck/analysis.py --plot`; outputs in `outputs/clevr_bottleneck/analysis_formal_strict_700`.
  - Analysis rows: 35 cases, 3045 layer-shift rows, 105 KL rows.
  - Critical representation shift peaks at final hidden state L28: color `0.13646`, shape `0.16784`.
  - Critical-vs-noncritical separation becomes large in L23-L26:
    - color SC peak L26 `0.97975`; L23 SC `0.97273`, critical `0.07859`, noncritical `0.00220`.
    - shape SC peak L26 `0.97281`; L23 SC `0.96905`, critical `0.09459`, noncritical `0.00302`.
  - Final-logit KL mean:
    - color critical `18.5864`, noncritical `0.00535`, spatial `0.23551`.
    - shape critical `16.7985`, noncritical `0.0000096`, spatial `0.000984`.
- Vision-token ablation:
  - Ran mask-defined target/background ablation on original condition at L23 and L18.
  - Outputs:
    - `outputs/clevr_bottleneck/ablation_formal_strict_700_L23`
    - `outputs/clevr_bottleneck/ablation_formal_strict_700_L18`
  - Added normalized per-replaced-token summaries in each ablation output directory.
  - Raw region-level shift is often larger for background because background replaces ~71 tokens vs ~5 target tokens.
  - Per-token normalized L18 readout L28 shift:
    - color target `6.71e-06` vs background `6.26e-07`.
    - shape target `7.93e-05` vs background `5.31e-05`.
  - Per-token normalized L23 readout L28 shift:
    - color target `1.49e-06` vs background `1.58e-07`.
    - shape target `2.33e-05` vs background `6.65e-06`.
- Visualization:
  - Counterfactual grid for first 12 cases: `outputs/clevr_bottleneck/visualizations_formal_strict_700/counterfactual_grid_first12.png`.
- Interpretation/caveats:
  - Results support strong critical-vs-noncritical selectivity in text hidden states and logits, with separation emerging around L18 and becoming highly concentrated by L23-L26.
  - The final hidden state L28 has the largest absolute critical shift, so claims should phrase this as middle-to-late formation/concentration rather than a pure middle-only peak.
  - Spatial interventions are usually much smaller than critical attribute interventions for these color/shape prompts; color spatial KL has a few nonzero cases.
  - Ablation should be interpreted using per-token normalization or followed by token-count-matched background controls; raw target/background region shifts are not directly comparable.

## 2026-07-14: CLEVR Strict Paired Counterfactual Audit

- User requested a strict audit because prior CLEVR results might only be aggregate critical-vs-noncritical comparisons rather than paired counterfactual analysis.
- Added `experiments/clevr_bottleneck/audit_counterfactual_pairs.py`.
  - Generates `counterfactual_pairing.csv`, `validity_report.csv`, `edit_type_balance.csv`, `question_validity_report.csv`, `model_correctness_report.csv`, `paired_hidden_shift.csv`, `paired_stats.csv`, plots, and `audit_report.md`.
  - Uses original-anchored paired shifts: `Dcrit=1-cos(h_original,h_critical)`, `Dnon=1-cos(h_original,h_noncritical)`, `Dspatial=1-cos(h_original,h_spatial)`.
  - Model correctness uses saved final logits with forced-choice scoring over CLEVR color/shape labels; no attention is used.
- Ran audit on:
  - Manifest: `outputs/clevr_bottleneck/dataset_formal_strict_700/manifest.jsonl`
  - Hidden states: `outputs/clevr_bottleneck/hidden_states_formal_strict_700`
  - Output: `outputs/clevr_bottleneck/audit_formal_strict_700`
- Audit counts:
  - Original manifest cases: 35.
  - Complete quadruples: 35.
  - Scene-level counterfactual valid cases: 35.
  - Edit-type matched cases: 17.
  - Question-valid cases: 33 overall; 17 after edit-type filtering.
  - Before correctness filtering: 17.
  - After correctness filtering: 17.
  - Final cases by task: color 17, shape 0.
- Key correction:
  - The formal strict dataset's shape cases fail edit-type matching because `critical_attr=shape` but `noncritical_attr=color`.
  - Therefore previous shape aggregate results should not be used as strict paired evidence.
- Final paired color results:
  - L18: mean Dcrit `0.007339`, mean Dnon `0.002114`, mean diff `0.005226`, prop Dcrit>Dnon `1.0`, paired t p `2.80e-05`, Wilcoxon p `1.53e-05`, bootstrap CI `[0.003591, 0.007027]`.
  - L23: mean Dcrit `0.078585`, mean Dnon `0.002203`, mean diff `0.076382`, prop `1.0`, paired t p `1.91e-09`, Wilcoxon p `1.53e-05`, CI `[0.065433, 0.089156]`.
  - L26: mean Dcrit `0.068131`, mean Dnon `0.001408`, mean diff `0.066723`, prop `1.0`, paired t p `6.18e-11`, Wilcoxon p `1.53e-05`, CI `[0.058896, 0.075489]`.
  - L28: mean Dcrit `0.136460`, mean Dnon `0.004436`, mean diff `0.132025`, prop `1.0`, paired t p `6.30e-11`, Wilcoxon p `1.53e-05`, CI `[0.115877, 0.148344]`.
- Follow-up required:
  - Regenerate shape formal cases with `generate_cf.py --noncritical_attr shape` or task-aware noncritical attr selection before making shape-task paired claims.

## 2026-07-14: CLEVR Three-Part Semantics Phase-1 Reorganization

- User requested reorganizing the CLEVR counterfactual experiment into three connected parts:
  - Part A: Vision-token semantics.
  - Part B: Text-token semantics.
  - Part C: Prediction-position semantics.
  - Required reusing the existing semantic variance metric and using only strict paired cases.
- Added `experiments/clevr_bottleneck/three_part_semantics.py`.
  - Uses only audit-passed paired cases from `outputs/clevr_bottleneck/audit_formal_strict_700`.
  - Defines a unified counterfactual semantic axis `a_orig -> a_cf` and margin `m(h)=z(a_cf)-z(a_orig)` using `final_norm + lm_head`.
  - Uses image + question only; no answer token is teacher-forced.
  - Prediction position is the final prompt token hidden state before the first answer token.
  - Reuses `experiments/vision_token_logit_lens.py::semantic_variance` unchanged: full-vocab logit variance across vision tokens after final norm + lm_head.
  - Uses patch-mask overlap for vision token region mapping and labels target / changed_distractor / other_distractor / background, with mixed/low-overlap flags.
  - Implements text span pooling for object mention, attribute/query token, and other question tokens; excludes prediction position.
  - Implements output-hook activation patching for target, matched distractor, same-count random background, full vision, and reverse target patches.
- Ran phase-1 on 17 strict color paired cases:
  - Command output directory: `outputs/clevr_bottleneck/three_part_phase1_color17`.
  - Outputs include `semantic_axis_metadata.csv`, `vision_semantic_variance.csv`, `vision_token_region_map.csv`, `vision_token_semantic_direction.csv`, `vision_region_semantic_summary.csv`, `text_token_semantic_direction.csv`, `prediction_semantic_direction.csv`, `prediction_final_full_vocab_logits.pt`, `prediction_final_full_vocab_logits_index.csv`, `vision_text_patching_results.csv`, `vision_text_prediction_joint.csv`, plots, prompt-swap audit, and `three_part_report.md`.
  - Full-vocab final prediction logits file size: 20 MB.
- Key phase-1 results:
  - Region map counts: target 107 tokens, changed_distractor 29, other_distractor 68, background 1156; 12 tokens marked mixed.
  - Vision semantic variance compatibility audit confirms reuse of the prior metric and logit-lens path.
  - Vision Part A:
    - L23 original semantic variance `4.0578`; critical `4.0535`.
    - L27 original semantic variance `2.1552`; critical `2.1343`, showing late decoder-layer semantic variance drop versus L23/L26.
    - L23 target semantic effect `10.8069`, changed distractor `-0.6303`, background `0.5981`, local contrast `10.6895`, global strength `1.3046`.
    - L26 target semantic effect `13.6210`, local contrast `13.2412`, global strength `1.7553`.
    - L27 target semantic effect `8.8991`, local contrast `8.6665`, global strength `1.1551`.
    - Interpretation: target semantics are strongly localized in mid/late layers, semantic variance drops at L27 while global semantic strength remains positive; do not phrase this as information loss.
  - Text Part B:
    - Object mention critical margin rises: L18 `0.1041`, L23 `1.0648`, L26 `2.1264`, L28 `1.4118`.
    - Object mention critical-minus-original delta: L23 `2.4508`, L26 `5.0901`, L28 `3.5809`.
  - Prediction Part C:
    - Critical prediction margin flips strongly positive while original/noncritical stay negative at later layers.
    - L23: original `-11.7298`, critical `11.9559`, noncritical `-11.8787`; critical-original `23.6857`.
    - L28: original `-14.4779`, critical `13.9853`, noncritical `-14.6103`; critical-original `28.4632`.
    - Median semantic flip layer across 17 color cases: L13; min L1, max L23.
  - Activation patching:
    - At patch layer L18, target patch at prediction readout L28 recovers sufficiency ratio `0.6614`; matched distractor `0.00894`; random background `0.00335`; full vision `0.9133`.
    - L23 target patch ratio `0.2055`; L26 target patch ratio `0.0189`, consistent with semantic transfer already occurring earlier.
  - Prompt-swap control:
    - Current Qwen chat-template path does not support a clean query-invariant vision-state claim.
    - Same-image prompt swaps had max absolute vision hidden differences up to `24`; report explains that serialized control/user text precedes the vision token block, so causal masking alone is insufficient for invariance.
- Caveats/follow-up:
  - Phase 1 supports strict color paired claims only.
  - Shape remains unsupported until regenerating strict shape-matched cases.
  - Prompt-swap needs a stronger cached image-embedding or image-independent input control in the next phase.

## 2026-07-14: CLEVR Three-Part Phase-1 Anomaly Corrections

- User asked to fix abnormal-looking data in the three-part CLEVR report.
- Diagnosis:
  - L28 vision semantic variance was anomalously high because L28 is the final hidden output and is not scale-compatible with decoder block layers for the old semantic-variance/homogenization claim.
  - Region-level summaries mixed clean and mixed-overlap tokens; Qwen's 8x10 merged grid is coarse for object masks.
  - Prompt-swap control failed for Qwen chat-template inputs because control/user text precedes the vision-token block; same-image prompt swaps produced max absolute vision hidden differences up to 24.
- Code changes:
  - Updated `experiments/clevr_bottleneck/three_part_semantics.py` so future runs also write `vision_region_semantic_summary_clean.csv`.
  - Clean-only summary filters `overlap_quality == clean`.
- Generated corrected outputs in `outputs/clevr_bottleneck/three_part_phase1_color17`:
  - `vision_region_semantic_summary_clean.csv`
  - `vision_semantic_variance_decoder_layers_summary.csv`
  - `vision_semantic_variance_decoder_layers.png`
  - `vision_localization_homogenization_decoder_clean.png`
  - `three_part_report_corrected.md`
- Corrected report changes:
  - Vision semantic variance main analysis is restricted to L1-L27; L28 is excluded from the main homogenization evidence and should be treated separately as final hidden output.
  - Main Part A table now uses clean region tokens only.
  - Prompt-swap is explicitly marked as failed / non-applicable supporting evidence for query-invariant vision states.
- Corrected key numbers:
  - Clean region map counts: target 102, changed distractor 24, other distractor 66, background 1156.
  - L23 clean target effect `11.0046`, clean changed distractor `-1.7215`, clean background `0.5981`, clean local contrast `11.1935`, clean global strength `1.2733`.
  - L26 clean target effect `13.8697`, clean local contrast `13.8744`, clean global strength `1.7187`.
  - L27 semantic variance original `2.1552` and clean global strength `1.1308`, supporting homogenization/shared-field framing rather than information loss.
- Use `three_part_report_corrected.md` for future summaries; avoid citing the old L28 vision variance table as main evidence.

## 2026-07-14: CLEVR Dynamic Semantic Transfer Tracing Phase 1

- User said the prior Part A/B/C report was still static logit-lens comparison and requested true source-layer x destination-layer x destination-stream semantic signal tracing.
- Created `experiments/clevr_bottleneck/semantic_transfer_tracing.py`.
  - Patches decoder block output once at a source layer, then lets downstream layers compute naturally with `use_cache=False` and `output_hidden_states=True`.
  - Layer numbering: `hidden_states[0]` embedding; `hidden_states[1..28]` decoder block outputs; FinalNorm is not treated as a decoder layer.
  - Patch conditions implemented: target patch, matched distractor patch, same-count random background patch, and no-op patch.
  - Destination streams saved/analyzed: target vision, nontarget vision plus changed/other/background subclasses, object text span, query text span, and prediction position.
  - Emits only destination layers `d > s`.
- Smoke command passed on one strict color case:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/clevr_bottleneck/semantic_transfer_tracing.py --manifest_jsonl outputs/clevr_bottleneck/dataset_formal_strict_700/manifest.jsonl --audit_dir outputs/clevr_bottleneck/audit_formal_strict_700 --region_map_csv outputs/clevr_bottleneck/three_part_phase1_color17/vision_token_region_map.csv --output_dir outputs/clevr_bottleneck/semantic_transfer_smoke --model_name_or_path /root/Qwen3-VL-2B-Instruct --source_layers 16 --max_cases 1 --task color --device cuda --dtype bfloat16 --save_patched_hidden --plot --seed 123`
- Formal phase-1 command completed on 17 strict audited color paired cases:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/clevr_bottleneck/semantic_transfer_tracing.py --manifest_jsonl outputs/clevr_bottleneck/dataset_formal_strict_700/manifest.jsonl --audit_dir outputs/clevr_bottleneck/audit_formal_strict_700 --region_map_csv outputs/clevr_bottleneck/three_part_phase1_color17/vision_token_region_map.csv --output_dir outputs/clevr_bottleneck/semantic_transfer_phase1_color17 --model_name_or_path /root/Qwen3-VL-2B-Instruct --source_layers 16,18,20,23,26 --task color --device cuda --dtype bfloat16 --save_patched_hidden --plot --seed 123`
- Output directory: `outputs/clevr_bottleneck/semantic_transfer_phase1_color17`.
  - `semantic_transfer_raw.csv`: 18,907 data rows.
  - `semantic_transfer_normalized.csv`: 18,744 data rows.
  - `semantic_transfer_onset.csv`: 25 data rows.
  - `source_patch_audit.csv`: 325 data rows.
  - `patched_hidden_states_index.csv`: 325 data rows.
  - `tracing_implementation_audit.md` and `semantic_transfer_report.md`.
  - Heatmaps generated for target vision, nontarget vision, object text, query text, and prediction, each with raw and normalized variants.
  - Patched hidden-state files saved under `patched_hidden_states/`; total size about 978M.
- Patch coverage:
  - target patch: 85 source-runs.
  - random background patch: 85 source-runs.
  - no-op patch: 85 source-runs.
  - matched distractor patch: 70 source-runs.
  - 15 matched-distractor source-runs missing because 3 cases have no overlap-mapped changed-distractor tokens in Qwen's coarse 8x10 merged grid: `12793_obj0_color`, `2514_obj3_color`, `5785_obj0_color`, across L16/L18/L20/L23/L26.
- Audit results:
  - No-op max raw transfer effect is `0.0`.
  - Example source patch audit for `10372_obj3_color` L16 target patch: target positions max abs diff `21.75`, nonpatched positions max abs diff `0.0`.
  - Downstream layer debug reports patched-vs-original and patched-vs-critical max diffs for each downstream layer in `tracing_implementation_audit.md`.
- Key target-patch transfer to final destination L28:
  - L16: target vision E `14.2691`, nontarget vision E `1.1599`, object text E `2.5221`, query text E `0.5294`, prediction E `21.8309`, prediction ratio `0.7642`.
  - L18: target vision E `14.3826`, nontarget vision E `1.0901`, object text E `2.2868`, query text E `0.5294`, prediction E `18.9118`, prediction ratio `0.6614`.
  - L20: target vision E `14.5958`, nontarget vision E `1.1360`, object text E `2.1250`, query text E `0.5000`, prediction E `17.5882`, prediction ratio `0.6163`.
  - L23: target vision E `15.3008`, nontarget vision E `0.6526`, object text E `1.4853`, query text E `0.4559`, prediction E `5.8603`, prediction ratio `0.2055`.
  - L26: target vision E `15.7523`, nontarget vision E `0.1820`, object text E `0.1765`, query text E `0.0441`, prediction E `0.5441`, prediction ratio `0.0189`.
- Controls at prediction L28:
  - L16 target/distractor/random/no-op E: `21.8309`, `-0.0714`, `-0.0221`, `0.0`.
  - L18: `18.9118`, `-0.0536`, `0.0956`, `0.0`.
  - L20: `17.5882`, `-0.0179`, `0.0588`, `0.0`.
  - L23: `5.8603`, `0.0357`, `-0.0074`, `0.0`.
  - L26: `0.5441`, `0.0`, `-0.0147`, `0.0`.
- Onset summary:
  - Source L16: target vision L23, nontarget vision L21, object text L18, query text L23, prediction L18.
  - Source L18: target vision L23, nontarget vision L21, object text L20, query text L20, prediction L20.
  - Source L20: target vision L22, nontarget vision L21, object text L21, query text L23, prediction L21.
  - Source L23: target vision L24, nontarget vision L25, object text L25, query text L25, prediction L24.
  - Source L26: target vision L27, nontarget vision none, object text L27, query text L27, prediction L27.
- Interpretation:
  - This is now dynamic transfer tracing, not static critical-original logit-lens.
  - L16/L18/L20 target-region patches strongly transfer to downstream prediction and text streams, while distractor/random/no-op controls are near zero.
  - Non-target vision effects exist but are much smaller, so results are consistent with strong target-vision-to-text/prediction transfer plus possible weak vision broadcast.
  - L26 target patch still has local target-vision effect but little final prediction effect, arguing against a purely late visual readout story.
- Follow-up:
  - Full-vision patch and path restoration were intentionally not run yet; user asked to confirm the matrices first, then add those.
  - Shape task remains excluded until strict shape-matched cases are regenerated.

## 2026-07-14: CLEVR Semantic Transfer Matrix Analysis

- User asked not to run path restoration yet and to fully analyze the existing dynamic source-layer x destination-layer x destination-stream matrices with heatmaps, common-valid controls, random resampling, onset redefinition, peak/AUC, and case-level confidence intervals.
- Modified `experiments/clevr_bottleneck/semantic_transfer_tracing.py`:
  - Added `--patch_conditions` and `--random_repeats`.
  - Random background rows now record `random_repeat_index`, `random_seed`, and `sampled_token_ids`.
  - Repeated random patching keeps token count matched to target patch count.
- Ran 20-repeat random background control only, without saving hidden states and without restoration:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/clevr_bottleneck/semantic_transfer_tracing.py --manifest_jsonl outputs/clevr_bottleneck/dataset_formal_strict_700/manifest.jsonl --audit_dir outputs/clevr_bottleneck/audit_formal_strict_700 --region_map_csv outputs/clevr_bottleneck/three_part_phase1_color17/vision_token_region_map.csv --output_dir outputs/clevr_bottleneck/semantic_transfer_random20_color17 --model_name_or_path /root/Qwen3-VL-2B-Instruct --source_layers 16,18,20,23,26 --patch_conditions same_count_random_background_patch --random_repeats 20 --task color --device cuda --dtype bfloat16 --seed 9001`
  - Output rows: `semantic_transfer_raw.csv` 98,420 data rows; `source_patch_audit.csv` 1,700 data rows.
- Created `experiments/clevr_bottleneck/analyze_semantic_transfer_matrix.py`.
  - Uses standard library + NumPy/Matplotlib only; no pandas dependency.
  - Merges existing target/distractor/no-op rows with the 20-repeat random control.
  - Builds common-valid case/source subset for formal target/distractor/random comparison.
  - Generates raw propagation heatmaps, target-minus-distractor heatmaps, target-minus-random heatmaps, thresholded normalized heatmaps, fixed-source curves, fixed-stream source comparisons, redefined onset tables, peak/AUC/duration summaries, random control summaries, and case-level scatter/violin plots.
- Ran matrix analysis:
  - `MPLCONFIGDIR=/tmp/mpl /root/miniconda3/envs/next3d_qwen/bin/python experiments/clevr_bottleneck/analyze_semantic_transfer_matrix.py --base_dir outputs/clevr_bottleneck/semantic_transfer_phase1_color17 --random_dir outputs/clevr_bottleneck/semantic_transfer_random20_color17 --output_dir outputs/clevr_bottleneck/semantic_transfer_matrix_analysis_color17 --onset_k 2 --bootstrap 5000 --seed 123 --full_effect_floor 0.25`
- Output directory: `outputs/clevr_bottleneck/semantic_transfer_matrix_analysis_color17`.
  - Main report: `semantic_tracing_heatmap_report.md`.
  - Integrity: `data_integrity_summary.csv`.
  - Common-valid set: `common_valid_cases.csv`.
  - Patch Ns: `patch_condition_N_by_source.csv`.
  - Random samples: `random_background_samples.csv` with 1,700 sampled token-id rows.
  - Random summaries: `random_control_case_distribution.csv`, `random_control_case_summary.csv`.
  - Matrix data: `semantic_transfer_case_level_matrix.csv`, `semantic_transfer_normalized_thresholded.csv`, `semantic_transfer_selective_case_level.csv`.
  - Statistics: `semantic_transfer_onset_redefined.csv`, `transfer_peak_summary.csv`, `case_level_key_cell_stats.csv`.
  - Figures: 80 PNGs across raw heatmaps, selective heatmaps, normalized heatmaps, fixed-source curves, source-comparison curves, and per-case plots.
- Integrity results:
  - Rows analyzed after merging random20: 112,406.
  - `destination_layer <= source_layer`: 0 rows.
  - Destination layer range: L17-L28.
  - FinalNorm as decoder destination: false.
  - No-op max absolute raw transfer effect: 0.0.
  - Patched hidden files/index rows from original tracing: 325 / 325.
- Common-valid formal comparison:
  - Each source layer L16/L18/L20/L23/L26 has 14 common-valid cases for target/distractor/random comparison.
  - The 3 excluded case/source pairs per layer are the same missing changed-distractor-token cases from the tracing audit.
- Normalized ratio threshold:
  - Rule: `max(full_effect_floor, p10(abs(full_counterfactual_effect)))`.
  - `full_effect_floor=0.25`, p10 abs full effect `0.09375`, median abs full effect `0.84375`; applied threshold `0.25`.
- Redefined onset highlights using bootstrap CI and 20-repeat random controls:
  - L16: object text and prediction onset L18; all-nontarget onset L20.
  - L18: object text/query/prediction onset L20; all-nontarget control-standardized onset L20.
  - L20: all major streams onset L21.
  - L23: prediction onset L24; text and nontarget onset L25.
  - L26: target vision/text/prediction onset L27, but all-nontarget has no onset.
- Peak/AUC highlights:
  - Prediction AUC: L16 `127.954`, L18 `107.314`, L20 `97.6696`, L23 `18.5379`, L26 `0.90625`.
  - Object text peak layer mostly L26 for L16/L18/L20/L23; L26 source has only weak text effects.
  - Target vision peak remains strong through late layers, but L26 has weak text/prediction propagation, supporting late patch isolation.
- Mechanistic conclusion from matrices:
  - Main pattern supports direct target-vision-to-text/prediction transfer in L16-L20, with weaker non-target vision broadcast.
  - It is not a purely vision-internal broadcast story because object text/prediction emerge early and strongly.
  - It is not a purely late visual readout story because L26 target vision effects do not translate into strong downstream prediction.
- Restoration recommendation only, not executed:
  - Candidate source layers: primary L18, secondary L20.
  - Candidate restoration layers should bracket control-standardized text onsets and nontarget onsets, plus peak prediction layers.

## 2026-07-14: CLEVR Path Restoration Phase 1

- User requested path restoration based on the completed dynamic semantic tracing results, specifically to distinguish text mediation, direct/parallel prediction writing, non-target vision broadcast, and late persistent visual readout.
- Created `experiments/clevr_bottleneck/path_restoration.py`.
  - Reuses the target-region source patch: critical target vision states patched into original at decoder block output `s`.
  - Restoration occurs once at decoder block output `r > s`, then downstream layers run naturally.
  - Restoration regions: `object_text`, `query_text`, `all_question_text`, `prediction_position`, `non_target_vision`, `target_vision`, `no_op`.
  - Uses the same semantic axis `z(a_cf)-z(a_orig)` and excludes answer tokens.
  - Saves final prediction drop/mediation ratio and full downstream restoration trajectories.
- Dry run confirmed scope:
  - 17 strict color cases.
  - Source layers: L16, L18.
  - Restoration layers: L16 -> L18/L20/L21/L23; L18 -> L20/L21/L23.
- First sandbox GPU run failed because CUDA driver was hidden; reran approved non-sandbox GPU command:
  - `/root/miniconda3/envs/next3d_qwen/bin/python experiments/clevr_bottleneck/path_restoration.py --manifest_jsonl outputs/clevr_bottleneck/dataset_formal_strict_700/manifest.jsonl --audit_dir outputs/clevr_bottleneck/audit_formal_strict_700 --region_map_csv outputs/clevr_bottleneck/three_part_phase1_color17/vision_token_region_map.csv --output_dir outputs/clevr_bottleneck/path_restoration_phase1_color17 --model_name_or_path /root/Qwen3-VL-2B-Instruct --source_layers 16,18 --restoration_layers_by_source '16:18,20,21,23;18:20,21,23' --task color --device cuda --dtype bfloat16 --seed 123 --bootstrap 5000`
- Output directory: `outputs/clevr_bottleneck/path_restoration_phase1_color17`.
  - `path_restoration_results.csv`: 833 data rows.
  - `path_restoration_trajectory.csv`: 33,915 data rows.
  - `path_restoration_trajectory_summary.csv`: 1,995 data rows.
  - `path_restoration_stats.csv`: 49 data rows.
  - `path_restoration_paired_comparisons.csv`: 42 data rows.
  - `path_restoration_auc_corrected.csv`: 34 data rows.
  - `path_restoration_audit.csv`: 833 data rows.
  - `path_restoration_span_audit.csv`: 119 data rows.
  - `path_restoration_report.md`.
- Control/audit results:
  - Source/restoration hook calls not equal to one: 0 rows.
  - `all_question_text` includes prediction position: 0 rows.
  - original/critical input ids differ: 0 rows.
  - original/critical vision indices differ: 0 rows.
  - restoration non-restored positions max diff: 0.0.
  - no-op restoration max absolute drop: 0.0.
  - FinalNorm is not counted as a decoder layer.
- AUC correction:
  - Source L16: raw summed AUC `124.455`, mean downstream effect `10.3713`, next-5 AUC `7.83019`, final prediction effect `21.8309`.
  - Source L18: raw summed AUC `103.452`, mean downstream effect `10.3452`, next-5 AUC `23.994`, final prediction effect `18.9118`.
  - This confirms raw summed AUC is length-biased; next-5 AUC is much stronger for L18.
- Key restoration drops:
  - Non-target vision restoration does not reduce final prediction effect; drops are near zero or negative across tested layers.
  - Object/query text restoration is small. Object-text has a small early L16/L18 effect (`0.2721`) but is weak later; query text is near zero/negative.
  - All-question text restoration has modest drops at L16/L18 (`0.8676`) and L16/L23 (`0.6691`), but remains much smaller than prediction-position restoration.
  - Prediction-position restoration causes large drops: L16/L23 `15.2059`; L18/L23 `12.4191`; consistently larger than all-question/object-text in paired tests.
  - Target-vision restoration causes very large drops at early/middle restoration layers: L16/L18 `20.1691`, L16/L20 `19.0147`, L18/L20 `18.125`; still nontrivial at L23 (`5.6176` for L16, `5.8897` for L18).
- Mechanistic conclusion:
  - The result does not support non-target vision broadcast as the main mediator.
  - It does not support a pure object/query text-mediated path.
  - It supports a mixed path dominated by direct/parallel writing into the prediction-position stream plus persistent target-vision contribution through middle layers.
  - Target vision handoff appears incomplete by L20 and partly reduced by L23.

## 2026-07-17 15:38:56 CST: Spatial Content Mechanism Phase-0 Audit

- User goal: distinguish coordinate transfer, explicit relation transfer, object-position-relation binding, and answer compression in decoder-only VLM middle-layer information flow; requested Phase 0 only, with no full experiment run.
- Audited existing README/configs, CLEVR counterfactual/render/mask pipeline, COCO bbox/logit-lens/attention code, semantic-source activation patching, prediction-difference hooks, outputs, and local Qwen3-VL-2B config.
- Verified Qwen3-VL-2B config: 28 decoder blocks, hidden size 2048, 16 attention heads, vision patch size 16, spatial merge size 2; runtime grid code derives actual grid from `image_grid_thw` rather than a fixed grid.
- Reusable core:
  - `experiments/clevr_bottleneck/{common,prepare_data,generate_cf,render_scene_json,map_tokens,inference}.py` for deterministic CLEVR scene/render/mask metadata.
  - `experiments/prediction_difference/hooks.py` for audited pre-LN residual definitions and decoder-layer discovery.
  - `semantic_source/activation_patching.py` and `experiments/clevr_bottleneck/path_restoration.py` for resid-post replacement/no-op checks and source-to-destination restoration patterns.
  - `semantic_source/statistics.py` and existing paired-analysis scripts for bootstrap/Wilcoxon-style reporting.
- Key Phase-1 gaps/risks:
  - Existing CLEVR conditions are only `original/critical/noncritical/spatial` and target color/shape; they do not implement paired two-object relation conditions C2-C7 or Q2-Q4.
  - Existing text-span extraction uses string subsequence lookup and must be replaced by template-origin/token-offset alignment.
  - Existing token mapping defaults to center-point hard assignment and image-space grids; add processor-coordinate transform validation plus hard/fractional overlap modes.
  - Existing LLaVA configuration is explicitly a placeholder; Qwen must be the first validated adapter and LLaVA needs a tested model-specific adapter before use.
- `pytest` is unavailable in the environment; no dependency was installed. Relevant files passed `py_compile`; built-in unit checks passed for prediction difference, COCO destination synthetic metrics, and semantic-source metrics.
- Environment follow-up: the active `/root/miniconda3/envs/next3d_qwen` environment also lacks both `pandas` and `pyarrow`; required Parquet outputs need an approved dependency installation or a separately agreed writer before Phase 1 emits final artifacts.
- No experiment data, model weights, or existing results were overwritten; no full dataset/model experiment was started.

## 2026-07-17 16:04:26 CST: Spatial Content Mechanism Phase-1 Smoke

- User asked to continue the experiment. Completed Phase 1 only: CLEVR spatial counterfactual generation/validation, question construction, family splits, and dynamic token-region alignment. No extraction, probes, or full causal runs were started.
- Added under `experiments/clevr_bottleneck/`:
  - `spatial_content_common.py`
  - `build_spatial_counterfactuals.py` for C0-C7
  - `validate_spatial_counterfactuals.py`
  - `build_spatial_questions.py` for Q1-Q4 and exact template-origin character spans
  - `spatial_splits.py`
  - `spatial_token_alignment.py`
  - `map_spatial_content_tokens.py`
  - `build_spatial_region_match_plan.py`
  - Config: `configs/spatial_content_clevr.json`.
- Counterfactual design:
  - C1 changes only A color; C2 translates A+B jointly; C3 changes A-B distance; C4 flips left/right; C5 swaps A/B color+shape while keeping positions; C6 moves a same-shape/same-size distractor by the exact C4 displacement; C7 is scene-identical no-op.
  - Generation requires Q4 relational referent uniqueness and C6 matching-shape/size eligibility before rendering.
  - Rendered validation checks scene-field changes, relation preservation/flip, bounds/collisions, rendered 2D relation, A/B masks, bbox overlap, visibility, C6/C4 changed-region area ratio, C6/C4 pixel-L2 ratio, and C7 zero pixel delta.
- Smoke output: `outputs/clevr_spatial_content_smoke_v4`.
  - 5 base families, 40 low-resolution renders/masks, all 5 valid after rendered audit.
  - Relation distribution: 2 left / 3 right.
  - 260 questions: Q1=70, Q2=70, Q3=70, Q4=50; no question rejections.
  - Qwen tokenizer check: all used yes/no/left/right/color answers are single-token with leading-space tokenization.
  - Qwen processor check: `image_grid_thw=1x14x20`, merge=2 -> runtime vision grid 7x10 (70 tokens); synthetic red/blue block masks each mapped to 9 distinct tokens, overlap 0.
  - Main hard token mode is `hard_any_overlap`; `hard_majority` gave zero tokens for 12px targets and is retained only as a robustness mode. Region mapping exports A, B, D, A-union-B, between corridor, hull, background, and all-image pools plus fractional weights.
  - C4-A vs C6-D count-match plan: 5 families, 3 usable with `min_matched_tokens=2`; 2 are below this threshold at 7x10 and must be filtered from main patching statistics or rerendered at higher effective resolution.
- Commands run:
  - `py_compile` on all new Phase-1 modules.
  - `spatial_token_alignment.py --run_unit_tests`.
  - Existing prediction-difference and semantic-source built-in unit tests.
  - Existing local Blender 2.79 renderer for the 5-family, 40-image/mask smoke set at 160x120, 4 samples.
- Known limitations/follow-up:
  - This Phase-1 generator currently targets left/right. Above/below needs a separately validated rendered-2D construction because the legacy CLEVR relationship routine skips above/below.
  - Formal data must use family-disjoint splits with adequate train/val/test counts; the 5-family smoke naturally has no useful held-out random split.
  - Required Parquet outputs still need a dependency decision because the environment lacks `pandas` and `pyarrow`.

## 2026-07-17: Spatial Content Mechanism Phase-2 Dependency And Smoke

- User requested that environment dependencies be fixed and subsequent experiments be launched. Installed/verified in `/root/miniconda3/envs/next3d_qwen`: `pandas 2.3.3`, `pyarrow 25.0.0`, and `scikit-learn 1.7.2`.
- Added Phase-2 modules under `experiments/clevr_bottleneck/`:
  - `spatial_content_states.py` shared cache/label/pooling helpers.
  - `extract_spatial_content_states.py` frozen Qwen residual extraction with ground-truth region pools and exact template-span input-ID alignment.
  - `train_spatial_content_probes.py` plus `evaluate_spatial_ood_probes.py` entrypoint for train-standardized linear probes, family-disjoint random splits, OOD split definitions, majority/shuffled baselines, and rejected-fit records.
  - `run_spatial_representation_deltas.py` for paired C0/counterfactual L2 and cosine residual distances.
  - Updated `build_spatial_questions.py` to store an explicit Q3 `relation_options` token span; updated `configs/spatial_content_clevr.json` with Phase-2 definitions and controls.
- Hook/state definition: cached `hidden_states[0]` is embedding output; `hidden_states[L]` is decoder block `L-1` output; final norm excluded; hook point saved as `block_output`. Main prediction position is final prompt token before generation.
- GPU smoke results:
  - `outputs/clevr_spatial_content_phase2_smoke`: two Q1 extractions completed. Qwen produced 29 hidden states (embedding + 28 blocks) and 70 vision tokens on the 7x10 merged grid.
  - `outputs/clevr_spatial_content_phase2_span_smoke`: Q3 extraction verified strict fields: A/B mentions, a three-token `relation_options` span, 16-token question-only `T_all`, and prediction position outside `T_all`.
  - `outputs/clevr_spatial_content_phase2_q3_smoke/deltas/representation_delta_metrics.parquet`: one C0/C4 Q3 paired delta produced 696 rows, covering layers 0-28 and available groups with `hidden_l2` and `cosine_distance`.
  - Tiny probe smoke emitted 30 explicit `fit_status=rejected` rows, as intended: one family cannot form a family-disjoint train/test split. No invalid accuracy was reported.
- Fixed a token-merge edge case exposed by smoke: an empty `V_between` corridor is now recorded as an unavailable optional group rather than rejecting the sample. Required A/B/all-vision/A-mention/prediction groups remain mandatory.
- Commands verified: JSON parsing, `py_compile` for all new modules, Qwen GPU extraction on 2 + 3 + 1 samples, paired delta run, and small probe run.
- Limitations/follow-up: existing Phase-1 smoke question files predate the new Q3 relation-span field; formal generation must regenerate questions. The `7x10` grid can leave corridors empty, so formal main analyses must report group availability and use the planned token-count filters/rerendering. No full dataset, full probe, patching, subspace, or composition run was started.

## 2026-07-17: Spatial Content Phase-2 40-Family Pilot

- User requested the next experiment. Ran a controlled, non-full Phase-2 pilot at `outputs/clevr_spatial_content_pilot40`.
- Data generation/render/audit:
  - Built 40 C0-C7 scene families with seed 314; static scene-graph audit passed 40/40 (left=18, right=22).
  - First Blender invocation rendered zero rows because `render_scene_json.py` defaults to legacy conditions; rerun with explicit C0-C7 and `--skip_existing` completed all 320 images/masks at 160x120.
  - Fixed `validate_spatial_counterfactuals.py`: missing rendered images/scenes are now explicit invalidation errors rather than silently skipped.
  - Rendered audit retained 38/40 families (left=16, right=22). Rejections: `scene_000184_A0_B4` C4 missing key mask; `scene_003196_A2_B3` C6 pixel-change mismatch.
  - Fixed `map_spatial_content_tokens.py` with `--valid_families_jsonl`, preventing invalid families from entering mapping. Produced 304 valid mappings; Qwen runtime geometry remains 7x10/70 tokens and synthetic color-block test passed.
- Questions/splits/states:
  - Regenerated questions with explicit Q3 relation-option spans: 1,976 questions, zero rejection; answer tokens remained single-token.
  - Family split: train=23, val=6, test=9.
  - Added extraction `--template_ids` and append-by-sample-id `--resume` so independently selected condition slices share one cache safely.
  - Extracted 380 frozen Qwen states: 152 C0 samples (38 families × Q1-Q4 template 0) plus 228 Q3 samples for C2-C7. All are valid. No model generation/teacher-forced answer positions were used as the main state.
- Outputs/results:
  - `deltas/representation_delta_metrics.parquet`: 184,962 rows (38 families, 6 counterfactual types, 29 hidden-state indices, 14 token groups, L2/cosine metrics).
  - Mean cosine distances verify C7 no-op is numerically zero. Relation flip exceeds joint translation at key groups: V_A 0.0566 vs 0.0328; query-final 0.0012 vs 0.0006; prediction position 0.0035 vs 0.0008. Identity-position swap is larger still (V_A 0.0977, prediction 0.0080); C6 distractor is much smaller (V_A 0.0047).
  - Random family-disjoint core probes (`probes_core/probe_metrics.parquet`, layers 0/16/28, one seed) were valid. L16: V_A x-coordinate R2=0.8953 and delta-x R2=0.4611; query-final x R2=0.6750 and delta-x R2=0.2690. Relation balanced accuracy at L16: V_A=0.9654, query-final=0.9104, prediction=0.9002; all exceed 0.5 majority and shuffled controls.
  - Cross-absolute-position OOD result (`probes_ood_position/ood_probe_metrics.parquet`, L16 low-rank linear probe): absolute x R2 is negative for every tested group (-1.19 to -2.26), while delta-x R2 remains 0.22-0.31 and relation balanced accuracy remains V_A=0.8864, query-final=0.7898, prediction=0.8333. This is evidence against a purely in-distribution absolute-coordinate shortcut, not conclusive H2/H3 evidence.
- Caveats:
  - Pilot has only 38 families, one seed for reported core/OOD figures, left/right only, and the random full-rank grid has not been used for conclusions. Binding labels, CKA/subspace projections, paired statistics, and all OOD splits remain incomplete.
  - No Phase-3 patching, Phase-4 subspace ablation, or Phase-5 composition run was started.

## 2026-07-18: Spatial Content Phase-2 Pilot Completion And Audit

- User asked to complete the remaining Phase-2 work, without starting Phase 3-5. Completed the 40-family Qwen3-VL-2B pilot consolidation at `outputs/clevr_spatial_content_pilot40`.
- Added `experiments/clevr_bottleneck/summarize_spatial_phase2.py`.
  - Produces `phase2_pilot_summary.json` and `phase2_pilot_selected_metrics.csv` from the authoritative per-sample/per-seed tables.
  - Records the `block_output` state definition, excludes legacy OOD outputs, and records Phase-2 limitations rather than treating them as completed results.
  - Fixed the summary grouping to retain null `question_type` fields emitted by the probe schema (`groupby(..., dropna=False)`).
- Completed all-layer linear probe outputs already scheduled for the pilot:
  - `probes_content_q3_alllayers/probe_metrics.parquet`: 26,796 valid rows, C0/Q3/template-0, 29 cached state indices, 3 seeds, PCA rank 16.
  - `probes_content_q1_alllayers/probe_metrics.parquet`: 19,836 valid and 290 explicit rejected rows; rejection records are retained.
  - `probes_binding_swap/probe_metrics.parquet`: 525 valid rows. This is a swap-condition sensitivity probe only, not evidence of role-filler compositionality.
- Completed paired Phase-2 delta aggregation:
  - `deltas/representation_delta_metrics.parquet`: 211,410 rows for C1-C7 paired against C0, with hidden-L2/cosine metrics across 29 state indices and token groups.
  - `statistics/statistical_tests.csv`: 6,496 paired comparisons with bootstrap 95% CIs, sign-flip permutation, Wilcoxon, and both Holm adjustments. `p_holm_wilcoxon` is the primary multiplicity-adjusted field because finite permutation resolution is inadequate for the large comparison family.
  - `statistics/invariance_metrics.parquet` and five plot files in `figures_phase2/` were produced.
- Corrected an OOD-probe contamination issue before reporting results:
  - Old `probes_ood_distance` used `conditions=all, question_types=all`, so it included multiple counterfactual samples from each family and is excluded from the summary.
  - Re-ran strict clean distance OOD as `probes_ood_distance_v2`: C0/Q3/template-0 only, family-level short/long split, 38 examples, train=12/test=26, layers=16, seeds 13/42/71, rank 16.
  - L16 relation balanced accuracy: V_A=1.0000, T_query_final=1.0000, P_prediction=1.0000, V_A_union_B=0.9688. This is small-pilot evidence against immediate distance-specific failure, not proof of explicit abstract relation representation.
  - `probes_ood_position_v2` remains the strict source for position OOD; template OOD intentionally shares families and is labeled wording-only.
- Selected L16 random-family Q3 results (train=23/test=9 per seed): relation balanced accuracy V_A=1.000, V_A_union_B=0.900, T_query_final=0.875, P_prediction=1.000; delta-x R2 P_prediction=0.658, V_A_union_B=0.509, T_query_final=0.400, V_A=0.229. Majority is 0.5; shuffled-label baselines are retained in source metrics.
- Relation-flip versus joint-translation paired cosine differences at L16 are present before correction (for example V_A right: mean +0.0343, bootstrap CI [0.0253, 0.0427], Wilcoxon p=1e-6), but only this selected V_A/right cell remains below 0.05 after Holm over all 6,496 comparisons (`p_holm_wilcoxon=0.00710`). Do not make broad layer/token claims from uncorrected cells.
- Updated `train_spatial_content_probes.py` to remove the deprecated scikit-learn `multi_class="auto"` argument; re-ran strict distance OOD with no warning. `py_compile` passed for Phase-2 scripts and the summary script parsed output JSON successfully.
- Remaining Phase-2 limitations: left/right only; 38 valid families; CKA and learned attribute/coordinate/relation/answer subspace projections are not implemented; no COCO external-validation pass has been run for this new content framework. No patching, mediation, ablation, or composition experiment was launched.

## 2026-07-18: Spatial Content Mechanism Phase-2 Geometry And Subspace Completion

- User asked to complete remaining Phase 2 before Phase 3. Added and ran held-out representation geometry and content-subspace analyses using existing frozen state caches only; no model forward, patching, mediation, ablation, or composition run was started.
- Added `run_spatial_phase2_geometry.py`, `aggregate_spatial_subspace_statistics.py`, `plot_spatial_phase2_geometry.py`, and `summarize_spatial_external_validation.py` under `experiments/clevr_bottleneck/`.
- Formal output: `outputs/clevr_spatial_content_pilot40/geometry_phase2`.
  - `representation_cka_metrics.parquet`: 1,740 rows, centered linear CKA with 200 bootstrap draws per valid CI.
  - `subspace_projection_metrics.parquet`: 169,650 held-out rows; `subspace_fit_metadata.parquet`: 870 valid fitted subspaces.
  - `subspace_statistical_tests.csv`: 4,640 paired projection-enrichment comparisons, 2,000 bootstrap draws, Wilcoxon and global Holm correction.
  - Inspected figures: `figures_phase2_geometry/counterfactual_cka_curves.png` and `relation_subspace_enrichment_curves.png`.
- Leakage/control design:
  - For every layer/token group, linear subspaces are fit only on 23 C0 training families; projection deltas are evaluated only on 9 held-out test families.
  - Specs are color, shape, absolute x, delta x, relation, and answer direction. Q3 answer direction is explicitly label-equivalent to left/right relation and is not independent evidence.
  - Reports projected-delta fraction, matched-rank random orthogonal-subspace fraction, and enrichment; C7 state delta is exactly zero.
  - Two layer-0 text CKA cells have zero centered variance and are explicit `rejected` rows rather than valid NaNs.
- Key L16 held-out results:
  - V_A CKA: C2 joint translation=0.8736, C4 relation flip=0.7140, C6 distractor control=0.9975, C7=1.0000.
  - C4 relation-flip relation-subspace enrichment over matched random rank: V_A=+0.1995, T_query_final=+0.1851, P_prediction=+0.2506.
  - C4-vs-C2 enrichment tests have n=4/5 after relation-direction stratification; none survives global Holm. These are descriptive pilot effects, not a statistically general relation-subspace result.
- External natural-image context: wrote `outputs/clevr_spatial_content_pilot40/external_validation_coco.json` from existing COCO diagnostics. It preserves L19 vs L27 target-attention mass (0.46385 vs 0.20964) but explicitly states that these outputs lack paired relation flips/swaps and cannot distinguish H1-H4.
- Updated `summarize_spatial_phase2.py` to include geometry/subspace tables and filter rejected CKA rows. Verified `py_compile`, summary schema, C7 no-op invariants, figures, and statistics.
- Remaining limitation: this is still a 38-family left/right pilot with only 9 held-out families; CKA/projection are correlational, and a newly generated COCO relation-counterfactual set is still absent. Phase 3 is required for causal claims.

## 2026-07-18: Relation-Coordinate Confound Audit

- User flagged the apparently anomalous L16 `V_A` relation balanced accuracy of 1.0. Audited the data instead of interpreting it as evidence for an explicit relation representation.
- Added `experiments/clevr_bottleneck/audit_relation_coordinate_confound.py` and ran it at `outputs/clevr_spatial_content_pilot40/audit_relation_coordinate_confound/`.
  - On the exact C0/Q3/template-0 random-family probe split, a logistic baseline using only scene-graph `a_camera_right` has held-out balanced accuracy=1.0000; `b_camera_right`=0.7750, `a+b_camera_right`=1.0000, and raw world `a_x`=0.7750.
  - The generator labels left/right from the camera-right axis, but the current family distribution has non-overlapping A camera-right ranges by relation: train left [-3.938,-1.733], train right [0.340,3.016]; test left [-3.136,-1.101], test right [0.599,2.984].
  - Verdict is `invalid_for_relational_claim`: current random-split `V_A` relation accuracy can be explained by an absolute-coordinate shortcut and is not evidence for H2/H3.
- Secondary token-resolution audit: V_A/V_B masks overlap in 18/38 samples (maximum 3 merged tokens) at the 7x10 grid. A no-overlap subset still gave V_A BAcc=1.0 but has only 13 train/4 test examples, so it neither rescues nor changes the coordinate-confound verdict.
- Updated `summarize_spatial_phase2.py` and `phase2_pilot_summary.json` to embed the coordinate-confound audit and state that random-split V_A relation numbers must not be used as H2/H3 evidence. Regenerate or coordinate-balance CLEVR relations before interpreting relation probes, CKA, or relation-subspace changes as abstract relation selectivity.

## 2026-07-18: Phase 2R Step 0-3 Feasibility Smoke

- User requested Phase 2R and explicitly paused Phase 3. Created isolated output root `outputs/clevr_spatial_content_phase2r`; old `pilot40` outputs were not changed or overwritten.
- Added:
  - `audit_phase2r_architecture.py`
  - `build_phase2r_datasets.py`
  - `audit_phase2r_shortcuts.py`
  - `audit_phase2r_render_tokens.py` (implemented but deliberately not run after shortcut gate failure)
  - `outputs/clevr_spatial_content_phase2r/phase2r_readme.md` and `phase2r_summary.json`.
- Architecture audit output: `architecture_audit.json`.
  - Qwen3-VL-2B visual tower has 24 bidirectional vision blocks (source passes `attention_mask=None`); patch=16, merge=2, vision hidden=1024 -> merged output=2048.
  - Decoder has 28 causal layers, h0 is multimodal LLM input after vision merger placeholder replacement, h1..h28 are decoder block outputs, final RMSNorm excluded.
  - Therefore V_A can already carry full-image information before decoder L16; old caches do not separately save raw vision-tower output.
- Implemented literal symmetric candidate data smoke from 12 CLEVR val families: 24 probe rows (P_left/P_right) and 120 anchor-fixed rows (A-fixed and B-fixed C0/C3/C4/C6/C7). Scene files/manifests are retained as a failure case.
- Shortcut smoke at `shortcut_audit.json` failed: A-only camera-right BACC=1.0, B-only camera-right BACC=1.0 on held-out families.
  - This exposed a mathematical incompatibility in the requested marginal conditions: for deterministic strict left/right labels on an ordered camera-right axis, both `p(x_A|left)=p(x_A|right)` and `p(x_B|left)=p(x_B|right)` cannot hold simultaneously. Equal marginals imply equal expected delta, but labels require E[x_B-x_A|left] > 0 and E[x_B-x_A|right] < 0.
  - No render, token audit, activation extraction, full probes, subspace ablation, or Phase 3 was launched after this gate failed.
- Required next decision: use a feasible anchor-specific target (balance the anchored object's coordinate while acknowledging the moved object's coordinate carries the relation) and test symmetric A-fixed/B-moved and B-fixed/A-moved experiments, or redefine the relation/control objective. Do not claim a fully marginal-balanced direct left/right dataset with both single-object coordinate baselines random.

## 2026-07-18: Grid-Exact Spatial Relations Dataset Smoke

- User requested a standalone, minimal Pillow 2D dataset generator only; no activation extraction, probing, patching, mediation, logit lens, repair, COCO/CLEVR/SpatialSense experiment, or formal-scale generation was run.
- Added `grid_exact_dataset/` with validated grid/config/scene/renderer/question/split/audit modules, Qwen3-VL processor impulse calibration, generation/audit/preview CLIs, README, example config, and pytest-style tests.
- Qwen3-VL processor calibration was actually run against local Qwen3-VL-2B-Instruct at 448x448: `image_grid_thw=[[1,28,28]]`, patch=16, merge=2, merged grid=14x14, cell=32px. Five impulses showed contiguous 2x2 packed-patch blocks in merged-block raster order; no resize/crop/pad. Calibration artifact: `/tmp/grid_exact_qwen_calibration2/`.
- Important validation: 2x2 objects plus one required background cell make requested `dx=±1,±2` impossible. Config validation rejects these values; smoke uses `[-5,-4,-3,3,4,5]`.
- Generated only `outputs/grid_exact_smoke`: 8 families, 48 lossless PNG scenes, 192 questions, family-disjoint 6/1/1 train/validation/test split, audit JSON/Markdown, and contact sheet.
- Smoke audit passed with raw anchor pixels identical in every family (max difference 0), processor-local packed patch input identical (max difference 0), no geometry/question/split leakage errors, and four direct G×Q×Y cells each 24. Attribute-answer colors are exactly balanced at 24 each.
- `pytest` is absent from the available environment; `py_compile` passed and all 12 test functions were directly executed using temporary directories, including intentional anchor-pixel, relation-label, and split tampering cases. Patch-embedding audit remains explicitly `not_available`: no final visual state is misrepresented as pre-self-attention embedding.

## 2026-07-19: Grid-Exact Step 1-3 Gates

- User requested gated Step 1-5 continuation. Completed only Step 1-3 and began the conservative Step-4 architecture report; no representation extraction, causal intervention, mediation, repair, or three-object dataset was run yet.
- Step 1 changes in `grid_exact_dataset/`: direct-relation answer now uses one subject/reference pure function; two-object attribute questions default off and, if enabled, carry `shortcut_sensitive=true`, `valid_for_relation_transfer=false`, and exclusion reason; formal manifests include direct questions only. Added a fixed uncovered 1x1 merged-cell control region per family, raw/processor control audits, 20-cell impulse calibration, formal-manifest audit, metadata logistic shortcut baselines, and collision-aware family splitting.
- Fresh Step-1 outputs: `outputs/grid_exact_step1_calibration_20` and `outputs/grid_exact_step1_smoke_v0_2`. Calibration passed 20/20 impulses. Smoke audit passed: anchor/control raw max diff=0, anchor/control processor packed-input max diff=0, difference-support errors=0, split leakage=0. All 16 family-disjoint train-only metadata logistic baselines had test BACC=0.5 with family-bootstrap CI [0.5,0.5].
- Initial 480 pilot at `outputs/grid_exact_pilot_480_v0_2` intentionally retained as failed evidence: exact duplicate two-object raster scenes from reversed colors/anchor modes crossed splits, producing three split-leakage errors. No labels or samples were changed.
- Fixed leakage by assigning visual-duplicate family connected components as indivisible units while retaining exact stratum quotas; regenerated only at `outputs/grid_exact_pilot_480_v0_2_r1` (480 families: 240/240 types; 2880 scenes; 5760 direct/formal questions; 336/72/72 family split). Audit passed: 0 errors, raw anchor/control max diff=0, processor anchor/control max diff=0, difference-support errors=0, 20/20 mapping passed, no cross-split image hash leak. All shortcut baselines BACC=0.5 / CI [0.5,0.5].
- Step 3 executed all 5760 questions locally with Qwen3-VL-2B-Instruct, greedy decoding, CUDA bfloat16, batch 4, no download. `left`/`right` are one-token (2359/1291); records save prompt, input/generated IDs, raw/normalized output, logits/probabilities/margins. `outputs/grid_exact_pilot_480_v0_2_r1/behavior_qwen3vl2b`: parse=1.0; all-valid accuracy/BACC=0.9974; AB=0.9965; BA=0.9983; paired=0.9948; minimum distance accuracy=0.9958; preferred gate passed and hard stop false. A report-only summary repaired missing derived grouping metadata without rerunning or filtering the 5760 raw records.
- Created and ran `report_grid_exact_architecture.py` to `outputs/grid_exact_pilot_480_v0_2_r1/representation_origin_architecture`; it is conservative and marks runtime hooks as not-yet-captured pending an identity/shape audit.

## 2026-07-19: Grid-Exact Step 4 Runtime Hook Preflight

- User requested only Step-4 runtime hook preflight and a later small extraction pilot; no data regeneration, Step 5, causal patching, mediation, SAE, or repair was run.
- Ran two real local CUDA bfloat16 forwards on one formal pilot scene. The first proved hooks executed but failed before report writing due to a report-only `decoder_h0` name parsing bug; no dataset/model output changed. The corrected second forward wrote `outputs/grid_exact_pilot_480_v0_2_r1/representation_origin_runtime_preflight/{runtime_hook_report.json,md,token_span_report.json}`.
- Runtime-proven module paths/shapes: `model.visual.patch_embed` input `[784,1536]`, direct tensor output `[784,1024]`; `model.visual.merger` pre-hook input `[784,1024]` (vision-final pre-merger) and output `[196,2048]`; `model.language_model.layers.0` pre-hook (decoder h0) `[1,227,2048]`; all 28 decoder blocks ran exactly once and outputs are `[1,227,2048]`. Hooks are not inferred from names alone.
- Runtime token span was derived from actual `input_ids` / `<|vision_end|>`: `[4,200)`, count=196; `image_grid_thw=[[1,28,28]]`; hidden-state sequence length=227 at h0 and all 28 decoder outputs. This passes the one-sample shape/token-count preflight. Local patch projection is available at `model.visual.patch_embed`, before vision blocks.
- The requested 8-family local-projection identity, 100-scene AB/BA invariance, multi-control derivation, and 24-family extraction/sanity pilot are not yet run; do not treat the one-sample runtime preflight as a full Step-4 gate pass.

## 2026-07-22: Descriptor-Based Phase 3A Minimal Evidence Audit

- User requested an analysis-only audit of the behavior-gated descriptor D2 experiment: no new data/image/prompt/model run, no carrier search, attention scan, or repair. Added `experiments/phase3a_descriptor_evidence_audit.py`; it loads only the existing 288 manifest/cache payloads and five saved patch-logit CSVs. `py_compile` and the offline audit ran with the local next3d environment.
- New artifacts are under `outputs/phase3a_descriptor_carrier_pilot/evidence_audit/`: cache/split audit, full behavior table, coordinate distributions, every-layer raw/residual coordinate probes and selectivity CIs, every-layer family-disjoint prediction probe/confusion audit, every-layer continuous patch-effect audit, and `evidence_audit_report.md`.
- Integrity: shared-prefix S_A/S_B/C analysis keeps one RB state per image (144, never duplicates RB/BR); P keeps 288 query-specific states. Train/validation/test are 16/4/4 families and 96/24/24 images, with empty family and image intersections. The actual descriptor cache has variable A/B colors rather than a literal fixed red/blue corpus; conclusions are slot- rather than color-specific.
- Full 144-image behavior gate remains perfect: 288/288 exact, each R×Q cell N=72/accuracy=1, RB=BR=1, left/right=.5/.5. Mean left-minus-right candidate margins are left/RB +4.646, left/BR -5.330, right/RB -4.875, right/BR +4.960.
- H1 audit: coordinate correlation is low (test corr(x_A,x_B)=.086), but residual selectivity is asymmetric. B-slot own-residual selectivity has positive family-bootstrap CIs at several layers (e.g. L6 +1.177 [.576,17.987]); no A-slot own-minus-other residual selectivity CI excludes zero (L18 +.049 [-.697,.410]). Therefore retain spatial decodability with suggestive B-slot selectivity, but downgrade the two-slot object-specific coordinate-binding claim.
- Prediction audit refit scaler on train only and selected C on validation only. From L16–L28, both family-disjoint cross-query Y directions are 24/24/BACC=1; both R directions are 0/24/BACC=0 and flipped accuracy=1, confirming systematic query-direction reversal rather than random R failure.
- Carrier patch audit uses raw left/right logits and CE_carrier. Despite 0/16 flips, some saved interventions have positive directional CE CIs (span-single L16 +.172 [.078,.266]; span-persistent L1 +.938 [.483,1.570]); no-op is exactly zero and same-relation controls are not source-aligned, though they can lower target margin without a flip. Classify carrier evidence as weak/continuous, not sufficient reusable spatial transfer; only four held-out families are independent.
- Prediction whole-state patch: main opposite-answer condition has 16 directed clean-correct pairs/4 families. L18–L27 copy is 16/16 with CE_prediction L18 +8.070 [6.984,8.898] and L27 +9.477 [8.172,10.617]; no-op CE is exactly 0 and available same-answer controls retain 16/16. Supports within-set causal answer-aligned sufficiency, but retain the four-family and answer-semantic-vs-lexical limitations. Stop after this audit; do not automatically launch a new mechanism experiment.

## 2026-07-22: Phase 3A Composition Localization — Minimal h_out Causal Pilot Stop

- User requested the restricted follow-up: only the existing descriptor-D2 clean-correct quadruplets, human blocks 14–16, and first the block-15 whole-output positive control; no new data/prompt/probe/model, head scan, carrier search, Qwen3.5, repair, or Phase 3B. Updated `experiments/phase3a_composition_localization.py` to enforce one deterministic existing clean quadruplet per family (smallest existing displacement), assert unique pilot families, and make shard output labels unambiguous. `py_compile` passed.
- Runtime forward implementation was rechecked: `h_in` is block input; `a_out` is self-attention output before first residual addition; `h_post_attn=h_in+a_out`; `m_out` is MLP output before second addition; `h_out=h_post_attn+m_out`. A one-sample clean forward, h_out self/no-op, and Q-flip h_out swap all executed successfully, isolating later process interruption to the interactive execution lifetime rather than hook semantics.
- Environment foreground lifetime required four two-family execution shards (batch 8) rather than one 8-family process. Shard merge is exact and contains 192 h_out Q/R-flip rows: eight independent families, 32 directed pairs per pair type/block. This changed only process lifetime/model reloads, not data, pair definitions, hooks, or statistics.
- Family-disjoint boundary probes: human block 15 / code index 14 has h_in Y cross-query=.5/.5; h_post_attn Y=1/1 and canonical R=0/0 with flipped R=1/1. Thus the decodable answer-aligned/R-reversed transition starts at the attention residual boundary. Block-15 h_out Y=.1/.917 and R=.0/.083 across directions; block 16 subsequently stabilizes the representation.
- Required h_out positive control failed at block 15: Q-flip CE +.016 [-.020,.055], R-flip +.043 [-.024,.109], 0/32 source-answer copies for either, and target accuracy 1.0. Block 16 had small positive CE (Q +.500 [.426,.578], R +.473 [.344,.586]) but still 0 copying and target accuracy 1.0, so it is not an answer-control positive control. h_out self/no-op merged 96 rows (8 families × 4 conditions × 3 blocks) with max absolute margin shift exactly 0 and target accuracy 1.
- Therefore stopped exactly at the prescribed gate: did not run a_out/m_out, same-Y, paired branch comparisons, h_post_attn, or full-72 validation. The only supported conclusion is representation-boundary localization at block-15 attention residual; portable causal localization to h_out or an additive branch is not established. Report/artifacts: `outputs/phase3a_composition_localization/minimal_causal_pilot_report.md`, merged positive/self result and summary CSVs, selection JSONs, and existing `boundary_probe_results.csv`.

## 2026-07-22: Phase 3A Composition Pilot Post-Run Integrity Recheck

- User asked whether the minimal causal pilot is credible. Performed read-only recheck of source script, all merged raw rows, selection JSONs, cached h_out vectors, and self/no-op rows; no model forward, prompt/image/data change, or new causal condition was run.
- The merged positive set has exactly 192 rows, 8 unique families, and no duplicate `(family, displacement, pair type, direction, block, component)` key. Each Q-flip row has same R and different Q/Y; each R-flip row has same Q and different R/Y; all source/target rows are clean-correct. Each block/pair type has 32 directed pairs from all 8 families. Selection shards contain families 000000–000007, each existing minimum displacement |dx|=3.
- The self set has exactly 96 rows, 8 families, source_id=target_id for every row, and exact clean_margin=patched_margin for every row (max absolute shift 0). This independently validates batch padding/prediction-position alignment and target-side cached h_out replacement.
- Source/target h_out states are not accidentally identical: across the 64 opposite-answer pairs per block, median source-target L2 is 3.076 (L14), 5.624 (L15), and 12.167 (L16), with zero exact-equal pairs (median cosines .99963/.99910/.99684). Thus the L15 zero-copy result is not a null replacement artifact. Corrected one report typography typo only (`1.00`, not `.1`).
- Conclusion unchanged: methodology is internally credible for the stated *negative L15 h_out positive-control result*, but its inferential scope remains a small, deterministic 8-family pilot at |dx|=3; it does not prove a universal absence of causal composition or identify a branch.

## 2026-07-22: Phase 3A Causal Commitment Boundary — 8-Base-Family Bridge Pilot

- User requested h_out-only causal bridge with no new images/prompts/probes/branches/heads/models. Extended `experiments/phase3a_composition_localization.py` with finite prediction-position h_out windows and base-family clustered summaries/MF; reused existing unified h_out cache and verified legacy all-layer prediction h_out equals unified block-18 h_out exactly before reusing the missing block-19 cache for W[15,20].
- Stage 0 replicated late control over 8 base families × 3 existing variants=24 quadruplets: L18 Q/R CE=8.408/8.753, copy=.990/1; L20 CE=9.583/9.699, copy=1/1; self/no-op exactly zero. Thus L15 negative is not a pipeline failure.
- L15 h_out remained null; L16 is earliest stable continuous causal influence (Q/R CE=.613/.566, both CIs positive, copy=0); L18 is first causal commitment/sufficiency (high copy, CE CIs positive, L20/L23 retain). Q and R have the same onset here. W[15,e] exactly follows its endpoint; W[16,16] equals W[15,16], so block-15 source alignment is overwritten/not required rather than needing maintenance. Mode C. Artifacts: `bridge_stage0_merged_*`, `bridge_stage1_merged_*`, `bridge_stage2_merged_*`, and `causal_commitment_boundary_pilot.md`.
- This is a successful 8-base-family pilot, not the optional all-family full validation; do not claim the model is frozen for Qwen3.5 until the user authorizes/runs the restricted full validation.

## 2026-07-22 23:05:47 CST: Phase 3A Functional Reusable Spatial Retention Gate Stop

- User requested the minimal three-object text-mediated spatial retention experiment: red square / blue circle Q1, then RB/BR/GR/GB Q2, with Q2 vision cutoff only if Gates A/B/C passed. No hidden states, probes, carrier search, head scan, Qwen3.5, repair, or large data generation.
- Added `experiments/phase3a_functional_spatial_retention.py`. It reuses the Grid-Exact renderer/config, Qwen loader, descriptor-style natural left/right scoring, runtime vision-token inference, and offset-based token span alignment. It creates only the requested 18 paired families / 36 images under `outputs/phase3a_functional_spatial_retention/`; each pair changes only green x, preserves red/blue and Q1 RB label, and matches Type A/B/C flip definitions.
- Command run: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/phase3a_functional_spatial_retention.py --device cuda --dtype bfloat16 --batch-size 8 --bootstrap 400`. The sandbox CUDA attempt failed due missing NVIDIA driver; the approved external CUDA run completed.
- Gate A single-question three-object ability passed: 144/144 correct, overall=1.0; RB/BR/GR/GB all 1.0; left/right output balanced .5/.5.
- Gate B two-question full-vision context failed and the experiment stopped before Q2 vision cutoff / Gate C / F-C-S retention analysis: overall=0.9097; RB=1.0, BR=1.0, GR=0.8056, GB=0.8333; left output fraction=.5764. Green errors were mostly `right` labels answered as `left`; RB/BR relation reuse remained perfect.
- Main report/artifacts: `outputs/phase3a_functional_spatial_retention/functional_spatial_retention_report.md`, `paired_family_table.csv`, `manifest_scenes.jsonl`, `records.csv`, `summary.csv`, `gates.json`.
- Caveat: the reused Grid-Exact renderer draws colored rectangles, while the prompt uses fixed shape descriptors red square / blue circle / green triangle. Gate A passing shows the model handled the rendered pilot under these descriptors, but this should be reported as an implementation caveat. Conclusion is only that the current model is unreliable in the dual-question green-query context, so reusable-retention claims cannot be tested from this run.

## 2026-07-24 CST: Q1-to-Q2 Interference Audit Stop

- User requested a minimal causal interference audit for the failed three-object Gate B only: reuse the existing 18 paired base families/36 images, Gate A/B prompts, Qwen3, full-vision scoring, and saved logits; analyze only Q2 GR/GB. No new data/prompt, vision cutoff, hidden-state patching, layer/head scan, prompt search, repair, Qwen3.5 run, or retention continuation.
- Added `experiments/q1_to_q2_interference_audit.py`; outputs are under `outputs/phase3a_q1_to_q2_interference_audit/`. `py_compile` and `--help` passed. The sandbox CUDA attempt failed due no NVIDIA driver; the approved external CUDA run completed with `--bootstrap 400 --batch-size 8 --device cuda --dtype bfloat16`.
- Stage 0 reused cached Gate A/B logits for GR/GB: Gate A accuracy=1.0; Gate B green accuracy=.8194. Overall `I_Q1 = -2.236` with family-bootstrap 95% CI `[-2.641,-1.793]`, so it is significantly opposite the previous-answer direction, not previous-answer-aligned. Gate B accuracy is worse for `A1==Y2` (.6389) than for `A1!=Y2` (1.0); errors are mainly right-truth answered left.
- Stage 1 attention-mask implementation checks passed: B0 rerun exactly matched cached Gate B margins (max shift 0), no-op 4D causal mask matched B0 (max shift 0), span audit passed, Q2 retained full vision-token access, and Q2 internal causal structure was unchanged. BA masked only the Q1 answer token; BQ masked the Q1 semantic sentence; BQA masked question+answer; BC masked `Answer with exactly one word`.
- Span ablations did not recover signed margin: B0 overall green acc/SM=.8194/2.661; BA=.8333/2.142 with improvement `-0.519 [-0.839,-0.109]`; BQ=.8194/1.972 with `-0.689 [-0.938,-0.351]`; BQA=.7778/1.804 with `-0.858 [-1.229,-0.382]`; BC=.8611/1.849 with `-0.812 [-1.083,-0.519]`. Stage 2 answer-swap was not run because Mode A was not supported.
- Predefined mechanism classification: Mode D, generic sequence-position/format failure. The failure cannot be localized to the Q1 lexical answer token, previous red-blue query span, or complete Q1 semantic history. Conservative decision: freeze Qwen3 for this reusable-retention protocol and move the same Gate A/B audit to Qwen3.5 before any retention, cutoff, hidden-state patching, or repair.

## 2026-07-24 CST: Q1-to-Q2 Interference Audit Recheck

- User asked whether the audit data/results were reasonable and requested an error check. Re-read `experiments/q1_to_q2_interference_audit.py`, reran offline consistency checks on saved CSV/JSON only, and did not rerun model forwards or add new conditions.
- Verified 36 scenes / 18 families; each of GateA_cached/B0/BA/BQ/BQA/BC has 72 GR/GB rows, 18 families, 36 images, and 36 rows per query. Stage 2 files are absent. B0 saved records exactly match cached Gate B margins (max shift 0). Formula checks for prediction, correctness, signed margins, Stage-0 `I_Q1`, and `Delta_correct` had zero mismatches.
- Verified mask audit rows: 5 samples per BA/BQ/BQA/BC; BA blocks one token decoded ` left`; BQ blocks 12-token Q1 semantic sentence; BQA blocks those plus the answer token; BC blocks 5-token non-semantic `Answer with exactly one word`; audited Q2 prediction rows can access all vision tokens and Q2 causal structure is unchanged.
- Found and fixed one non-conclusion bookkeeping issue: `mechanism_classification.json` had the sign of `bias_reductions_vs_B0` reversed. Updated `experiments/q1_to_q2_interference_audit.py` and regenerated the JSON field as `bias_shift_vs_B0`; Mode D and `run_stage2=false` are unchanged because all BA/BQ/BQA signed-margin improvement CIs remain negative.
- Added report caveat: BC is a format-control sanity span but is not length-matched to BQ/BQA, so it should not support a strong positive BC comparison. This does not weaken the main negative result that semantic ablations do not recover margin. Overall result remains reasonable and internally consistent.

## 2026-07-24 CST: Functional Text-Mediated Spatial Relay Gate-0 Stop

- User requested the single-turn 2x2 stage-access relay audit using the existing three-object 18-family / 36-image pilot, fixed neutral workspace prompt, Qwen3, left/right logit scoring, 4D attention masks, and true pre-query cache branching. No Q1/Q2 protocol, prior-answer interference, layer/head scan, hidden-state probe, carrier search, activation patch, new data, Qwen3.5, or repair.
- Added `experiments/functional_text_relay.py`; `py_compile` and `--help` passed. Ran `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/functional_text_relay.py --device cuda --dtype bfloat16 --batch-size 8 --bootstrap 400` with approved external CUDA. Outputs are under `outputs/phase3a_functional_text_relay/`.
- Span audit for the fixed prompt succeeded before stopping: sequence length 232, prediction position 231. V includes wrappers and image tokens `[3,201)` length 198 (`<|vision_start|> ... <|vision_end|>`); P `[201,213)` includes `Examine...` plus `Question:`; Q `[213,225)` is the full object-pair query sentence; A `[225,232)` includes `Answer:` and assistant delimiter through the prediction row. No overlap/boundary error was detected.
- Gate 0 failed. F full-vision one-shot did not reproduce Gate A behavior under the fixed neutral-workspace prompt: overall accuracy=.5, RB/BR/GR/GB each=.5, left prediction fraction=1.0, mean SM=.087 with CI crossing zero. The model predicted `left` for every first answer token.
- The no-op 4D mask path exactly matched F (`max_abs_margin_shift=0`), but cache-branch full-access vs one-shot F differed by max margin .375, above the strict cache equivalence tolerance. Because F itself failed, the protocol is already invalid; T00/T11/T10/T01 were not run.
- Predefined classification: Mode E protocol invalid. This run cannot answer scene-first vs query-conditioned extraction and should not be interpreted as evidence for or against reusable pre-query scene relay. Do not expand prompts or continue relay conditions without an explicit new user decision.

## 2026-07-25 00:03 CST: Functional Relay Baseline Gate-0 Interface Audit

- User requested a strict Gate-0-only interface audit explaining why original Gate A was 144/144 while the new fixed-neutral one-shot F collapsed to fixed `left`, and checking cache-branch vs one-shot equivalence. No T00/T10/T01/T11, mechanism interpretation, masks, hidden-state probe/patch, layer/head scan, new data, prompt search, Qwen3.5, or repair protocol was run.
- Created `experiments/functional_relay_baseline_audit.py` and outputs in `outputs/phase3a_functional_relay_baseline_audit/`: 144-row audit manifest, rendered prompt/token dumps, candidate tokenization, S1/S2/S3 scoring, greedy generation/top-20 next-token audit, G0-G5 prompt factor scores, query/image checks, cache equivalence diagnostics, and final report.
- Manifest checks passed: 144 rows, 36 images x 4 queries, RB/BR/GR/GB each 36 rows, each query left/right 18/18. Original Gate A baseline G reproduced exactly: accuracy=1.0, left/right predictions 72/72, mean signed margin=3.1936. Original fixed-neutral N failed: accuracy=0.5, predictions left=144, mean signed margin=0.0871.
- Candidate scoring was not the cause. `left/right` are bare tokens 2359/1291; ` left/ right` are 2115/1290. G was 1.0 under bare, space-prefixed, and full continuation scoring. N stayed 0.5 under S1/S2/S3. Greedy N generated explanatory text beginning `Based...`; top-1 next token was `Based` with probability ~0.92-0.98 on audited samples, so N was not in a constrained left/right answer context.
- Prompt factor audit localized the break: G0 original=1.0; G1 adding neutral sentence=1.0; G2 adding `Question:`=.986; G3/G4 N-like layout with one-word instruction=.979; G5 original N without `Answer with exactly one word, "left" or "right".`=.5 fixed-left. Classification for F collapse: chat-template/answer-cue/answer-contract mismatch, not scoring, prediction-index, or query/image alignment bug.
- Updated `experiments/functional_text_relay.py` minimally: fixed `prompt_for()` to include the one-word left/right answer contract and added `--gate0-only`; Gate-0 failure/success now saves F+NOOP+CB_FULL records before stopping. `py_compile` and `--help` passed.
- Reran only Gate-0 paths: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/functional_text_relay.py --device cuda --dtype bfloat16 --batch-size 8 --bootstrap 400 --gate0-only --output-dir outputs/phase3a_functional_text_relay_gate0_fixed`. Fixed F: overall=.9792; RB/BR/GR/GB=1.0/1.0/.944/.972; NOOP max margin shift=0.0. CB_FULL overall=.9861, but cache-full max margin shift vs F=0.5, so strict Gate 0 still failed.
- Cache audit conclusion: original G0 8-sample cache branch had prediction/top1 agreement=1.0 but margin diff median=.25, p95/max=.5. Fixed 144-row cache audit had F-vs-CB_FULL prediction agreement 143/144, mean abs margin diff=.1476, median=.125, p95/max=.5; one GB row changed prediction (`typeA_m0_t2_I1`, F margin -0.125 vs cache margin 0.0). Cache branching remains not acceptable for downstream relay experiments.
- Final report: `outputs/phase3a_functional_relay_baseline_audit/functional_relay_baseline_audit_report.md`. Current decision: fixed-neutral F no longer has the fixed-left interface collapse, but it is not an exact 144/144 Gate-A ceiling and cache continuation equivalence is still below the required standard; do not rerun 2x2 relay Gate 0 or T conditions until cache equivalence is resolved under the strict criterion.

## 2026-08-01 CST: VR Batch1-Only Pilot Status Check

- User requested a read-only status check of the Qwen3.5 selective visual-restoration `batch1_only` pilot. No model forward, resume, analysis, prompt/data/model/configuration change, or output mutation was performed; this AGENTS entry is the only write.
- No `qwen35_visual_restoration/run.py` process is currently running. `outputs/qwen35_visual_restoration/pilot_results_batch1_partial.csv` was last modified 2026-07-31 13:05:27 CST and has grown to 3,286 unique condition records.
- Recomputed against the runner's exact expected manifest: 3,286/4,352 rows complete (75.51%), 1,066 remaining; 145/192 pair-stage units complete, with no duplicate condition keys or partial pair-stage units. Completed: all four stages for TypeA_m0_t0, TypeA_m1_t2, TypeB_m0_t0, TypeB_m1_t2; TypeC_m0_t0 has five of eight directed pairs complete. Remaining work is TypeC_m0_t0 (330 rows) and TypeC_m1_t2 (736 rows).
- All checkpoint rows declare `execution_mode=batch1_only`; 1,896 older resumable rows lack the later-added `batch_size` column value, while 1,390 explicitly state batch_size=1. The runner normalizes this field to 1 only in the final artifact. No `pilot_results.parquet` or `implementation_gates.json` exists, so no analysis/stage selection or causal conclusion is authorized. Resume only in a durable CUDA session with `--stage pilot --execution-mode batch1_only --patch-batch-size 1 --batch-size 1`, then run `--stage analyze` after completion; do not run confirmatory work.

## 2026-08-01 CST: VR Batch1-Only Pilot Resumed

- User requested resuming the existing batch1-only pilot from checkpoint. Started `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_visual_restoration/run.py --stage pilot --execution-mode batch1_only --patch-batch-size 1 --batch-size 1` in the external CUDA environment. The initial output channel returned during model loading, but process audit confirmed the continuing job as PID 4333; it occupies ~7.9 GiB GPU memory at 99–100% utilization.
- The run is active at this handoff. Its first newly completed pair-stage wrote successfully: `pilot_results_batch1_partial.csv` advanced from 3,286 to 3,308 rows, with 146/192 pair-stage units complete. No configuration, input, state, prompt, or protocol was changed; do not launch a second copy. Await completion of pilot before running `--stage analyze`; no confirmatory work is authorized.

## 2026-08-01 CST: VR Batch1-Only Pilot Late-Run Status

- User requested another read-only status check. The resumed external CUDA process remains PID 4333 (`run.py --stage pilot --execution-mode batch1_only --patch-batch-size 1 --batch-size 1`) and has run for about 2h54m at audit.
- During the check, the partial CSV advanced from 4,044 to 4,132 records: 4,132/4,352 expected condition records complete (94.95%), with 220 remaining. `pilot_results.parquet` and `implementation_gates.json` are still absent, so pilot finalization and analysis have not yet occurred. No duplicate job, output alteration, or protocol change was made.

## 2026-08-01 CST: VR Batch1-Only Pilot Completed And Formal Analysis

- User requested inspection and analysis. The batch1-only pilot completed all 4,352 expected rows (1,088/stage over stages 2–5), all declaring `execution_mode=batch1_only` and `batch_size=1`; final `pilot_results.parquet` and `implementation_gates.json` were produced. Ran the fixed CPU-only `--stage analyze` to regenerate `pilot_summary.json`, `frozen_restoration_config.json`, and `audit_report.md`; no model forward, prompt/data/model/protocol change, or confirmatory run occurred.
- Implementation verdict is **fail** despite clean core checks: NOOP max margin shift=0 and source=target full-vocab max logit difference=0. The strict shuffled-critical delta L2 check is `3.0517578125e-05`, above the configured `1e-05` threshold. Under the prescribed gate, stop and do not treat this as visual causal evidence or proceed to confirmation.
- Descriptive (gate-invalid) changed-pair effects are very small: CRITICAL CE/RR at stages 2/3/4/5 = `.0938/.0131`, `-.0156/-.0022`, `.0391/.0056`, `0/0`; source-answer copy is 0 and target retention 1 at every stage/condition. Stage-2 FULL_VISION CE=.8984 but copy=0; critical-region effect remains only ~1.3% restoration. Invariant rows have zero erroneous flips. Thus even without the gate failure there is no decisive restoration/copy effect.
- The current analyzer writes `selected_primary_stage=stage_2` and `proceed_to_confirmatory_allowed_by_pilot_gate=true` based solely on its pilot score. This is logically inconsistent with `implementation_gates.passed=false`; treat those two fields as non-operative metadata, not authorization. Do not run confirmatory work without an explicit protocol decision and a gate-corrected/re-executed audit.

## 2026-08-01 CST: VR Code Re-Audit And Early-Arbiter Stop

- User requested a code re-audit of the already run Visual Restoration scripts. Read-only review plus `py_compile` covered `run.py`, `batch_arbiter.py`, and the new `early_boundary_arbiter.py`; no model forward was rerun for this review.
- Original `run.py` batch1-only pilot collection is not invalidated by the review: 4,352 batch1-only rows, exact NOOP/source=target checks, and the separate VR-Batch-Arbiter Verdict B remain valid for their stated scope. The known bug is in `summarize_and_freeze`: it computes `proceed` and a primary stage solely from stage scores, without requiring `implementation_gates.passed`; hence its `stage_2` / confirmatory metadata remains non-operative.
- `batch_arbiter.py` is consistent with its restricted batch-execution purpose: batch1 repeat/no-op exact, duplicate alignment exact, same-length task signs stable, and it correctly fixes formal execution to batch1-only. It does not authorize a scientific confirmation on its own.
- Stopped two early-arbiter attempts before results were written. First accidentally inherited forbidden bbox-only diagnostic conditions; second review found that it interleaved regional conditions before finishing the mandatory h0 arbiter, hard-coded an h0 patched-vision error rather than checking it, omitted several required early implementation/scientific gates, and fixed a C/D verdict instead of implementing the directional rules. Terminated PIDs 92187 and 131936. No `h0_full_vision_arbiter`, early result, scientific-gate, or verdict artifact exists and no early forward is interpretable. The early arbiter must be redesigned as a two-phase h0-first gate before any rerun.

## 2026-08-02 20:45 CST: Phase V2D-Boundary Completed — Verdict A

- User requested an independent Qwen3.5 Local Visual Evidence → Context-Bound Computation → Localized Decision experiment. Added `experiments/qwen35_v2d_boundary/{common.py,model_runtime.py,boundary_arbiter.py,boundary_confirmatory.py,summarize_boundary.py}` and `tests/qwen35_v2d_boundary/test_protocol.py`; outputs are isolated under `outputs/qwen35_v2d_boundary/`. No old early-runner verdict/authorization/summary code was imported, no old raw file was overwritten, and no prompt/data/model/region/threshold/layer search or PRED forward was run.
- All five pre-model unittest groups passed: authorization truth table, phase/hash order, data-dependent A–E verdict fixtures, patch no-op/source=target/identity, and condition-row completeness. The confirmatory manifest remained the frozen 12 families: 32 changed and 64 invariant directed rows; 22 condition repeats per boundary; 4,224 total batch1-only forward rows.
- Architecture discovery fixed VIS_LOCAL0 at `model.visual.blocks.0` pre-hook (`[784,1024]`, grid `1×28×28`), DEC_H0 at `model.language_model.layers.0` pre-hook (`[263,2560]`), and DEC_STAGE1 at full block-7 output (normalized depth .25, full attention). VIS local difference support was exactly 32 rows per directed image pair, with zero red/blue overlap or diffusion; local patching reconstructed source hidden/logits exactly on 48/48 pilot rows.
- An initial H0 arbiter used the D2L V coalition partition of 198 rows, which included identical vision-start/end special tokens. Its authorization was revoked and artifacts renamed `*.invalid_v198.*`; `checkpoint_migration_audit.json` records the correction. Reran all 48 H0 checks with exactly the 196 mapped image rows while retaining independently unaffected VIS/stage1 checks. Final VIS, H0 full-vision, and stage1 full-sequence arbiters all had hidden/full-vocab/no-op/source=target max error 0, top-1 agreement 1, and source copy 1. All mapping, batch1, permutation, manifest, and immutable-hash gates passed; final authorization is true.
- Ran the full authorized confirmatory in one continuous CUDA model session: 4,224/4,224 rows, 32 changed and 64 invariant directed pairs, no duplicates/missing/unexpected rows, `batch_size=patch_batch_size=1`, `execution_mode=batch1_only`, NOOP max full-vocab error 0, shuffled inverse-recovery max error 0, and float64 norm relative error at numerical precision (`<=2.73e-16`). Raw outputs: `raw/confirmatory_changed.parquet` and `raw/confirmatory_invariant.parquet`.
- Family-cluster results: DEC_H0 critical family-mean RR=.1189, Spec_shuffle=.0618, Spec_noncritical=.1117, 12/12 positive families, both directions positive; DEC_STAGE1 RR=.0608, Spec_shuffle=.0213, Spec_noncritical=.0538, 12/12 positive families, both directions positive. Changed-row means were H0 CE/RR=.8945/.1160 and stage1=.4336/.0578; invariant critical absolute drift=.3672/.2813, erroneous flips 0. FULL_VISION RR=1.0 at H0 and .7789 at stage1.
- All H1–H4 at both co-primary boundaries passed the preregistered family bootstrap/Holm directional gate (Holm-adjusted p=.002 throughout; corrected lower bounds positive). Equivalence against ±.05 RR failed at H0 for both contrasts; at stage1 only Spec_shuffle was equivalent, while Spec_noncritical was not. Old stage2–5 remained descriptive only: critical RR .0131/-.0022/.0056/0; PRED ladder remained the validated stage3/4/5 CE 6.33/7.08/15.90 and copy .167/.417/1.0.
- Final verdict: **A — Persistent localized visual evidence**. Under the protocol, PRED mediation may be proposed next but was not run. Claim only selective moved-object-region causal portability at decoder input/earliest macro stage under this confirmatory protocol; do not claim a scene graph, absence/deletion of visual evidence elsewhere, or commensurate information transfer between region RR and PRED CE. A summary-only figure pivot bug was fixed offline after raw completion (`summary_only_fix_audit.json`); no model/raw/statistical specification changed. Final report: `outputs/qwen35_v2d_boundary/report.md`.
## 2026-08-03 CST: Phase V2P-Mediation In Progress

- User requested an independent Qwen3.5 selective visual evidence → PRED mediation experiment. Added `experiments/qwen35_v2p_mediation/{common.py,model_runtime.py,mediation_arbiter.py,run_mediation.py,summarize_mediation.py}` and `tests/qwen35_v2p_mediation/test_protocol.py`; no V2D authorization/verdict logic is imported.
- Protocol tests pass 5/5. V2D input audit passes: Verdict A, implementation true, 196 image rows, 8-token critical union, 4,224/4,224 prior rows, withdrawn 198-row artifacts visible. V2P arbiter passes no-op/source=target, hook order, fixed-row total-effect direction, K5 endpoint reproduction, hashes, and batch1-only gates; `arbiter/mediation_authorization.json` is true.
- Main runner is batch1-only with 32 changed and 64 invariant directed rows, K3/K4/K5, critical/shuffled/background conditions, PRED/TASK_END sufficiency and reset. It writes resumable partial parquet rows and clean-state caches. Current checkpoint at audit: 8/192 pair-boundaries complete; one Python runner remains active (PID 508765). Do not start a duplicate runner; resume only from `raw/mediation_checkpoint.json`.
- No mediation verdict or summary exists yet. After 192/192, run `summarize_mediation.py` offline, verify raw completeness, then append final Verdict/report here. No new layer/region/data/model/PRED site is authorized.

## 2026-08-04 CST: V2P-Mediation Status Audit

- User requested a read-only status check of the Selective Visual Evidence → PRED-Mediated Decision Commitment experiment. No model forward, resume, analysis, script/data/model/protocol change, or output mutation was performed; only this AGENTS entry was appended.
- No `qwen35_v2p_mediation` process is running. The implementation arbiter remains passed across all recorded gates: batch1-only, no-op/source=target, token alignment, hook order, hash/input audits, K5 endpoint reproduction, and fixed-row total-effect reproduction. The two frozen decoder visual boundaries are `DEC_H0` and `DEC_STAGE1`; mediator sites remain K3/K4/K5 at code indices 19/23/31, with an 8-token critical union from 196 decoder image rows.
- All 96 required clean sample caches are present and loadable: 64 under `raw/states/` plus 32 under `arbiter/state_cache/`. The formal manifests remain 32 changed and 64 invariant directed rows across 12 families.
- Mediation collection is partial and stopped: checkpoint metadata reports 40/192 pair-boundaries; partial parquet keys show 41/192 boundaries (40 fully checkpointed plus one fully written boundary before checkpoint advancement), all in the changed subset. This is 20 complete directed changed pairs plus the first boundary of the 21st; no invariant boundary has started. Current rows are `base_states_partial=451/960`, `sufficiency_partial=1312/2688`, `reset_partial=205/960`, and `invariant_controls_partial=0/128`; all nonempty partial rows are unique and explicitly `execution_mode=batch1_only`, `batch_size=1`, `patch_batch_size=1`.
- No finalized `raw/*.parquet`, `summaries/*`, `figures/*`, `summaries/verdict.json`, or `report.md` exists. Partial rows must not be interpreted as mediation evidence. Resume the fixed batch1-only runner from the existing checkpoint, then run the offline summarizer only after raw completeness reaches 192/192; do not add sites, layers, regions, strengths, data, models, or repair.

## 2026-08-04 11:15 CST: V2P-Mediation Checkpoint Resume Active

- User requested continuing the V2P mediation project from checkpoint. Started `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_v2p_mediation/run_mediation.py` in the approved external CUDA environment; no script, data, model, prompt, protocol, or configuration changes were made.
- The runner loaded and validated all 96 clean caches, skipped the existing rows, and began the remaining changed-pair boundaries. At this audit the unique process is PID 5562, GPU utilization is approximately 100%, and partial outputs advanced to `base_states_partial=473`, `sufficiency_partial=1376`, `reset_partial=215`; invariant controls remain 0. This corresponds to approximately 43/192 pair-boundaries, with checkpoint metadata briefly at 42 during a concurrent write window.
- The job remains active; do not launch a duplicate. No mediation summary/verdict/report exists yet. Let the same runner finish all 192 boundaries, then run the offline summarizer only after raw completeness passes; no scientific interpretation is authorized from current partial rows.

## 2026-08-04 15:26 CST: V2P-Mediation Status Audit

- User requested a read-only progress check. Ran process/GPU/file audits and read-only parquet/checkpoint integrity checks; no model forward, resume, summarization, script/data/model/protocol change, or result mutation was performed.
- The previously resumed runner is no longer running: no V2P process or GPU compute process is present, and the latest partial outputs/checkpoint were written at 15:16 CST. Checkpoint metadata is `batch1_only`, `78/192` pair-boundaries complete (40.625%).
- Partial coverage is internally consistent and duplicate-free: `base_states_partial=746/1088`, `sufficiency_partial=2118/2688`, `reset_partial=390/960`, `invariant_controls_partial=28/128`. The 78 complete boundaries are 64 changed boundaries (all 32 changed directed pairs) plus 14 invariant boundaries (7 invariant pairs); all completed invariant rows are in the first two Type-A families. All nonempty rows declare `execution_mode=batch1_only`, `batch_size=1`, `patch_batch_size=1`.
- Implementation authorization remains true, including no-op/source=target, token alignment, hook order, hash/input, K5 endpoint reproduction, and fixed-row total-effect reproduction. The 96 clean caches remain present (64 formal plus 32 arbiter). No finalized raw parquet, summaries, figures, verdict, or report exists; partial rows are not mediation evidence.
- Follow-up: resume the fixed runner from `outputs/qwen35_v2p_mediation/raw/mediation_checkpoint.json`; only after `192/192` and raw completeness pass, run offline `summarize_mediation.py`. Do not launch a duplicate or add layers/regions/strengths/data/models/repair.

## 2026-08-04 15:56 CST: V2P-Mediation Checkpoint Resume Running

- User requested continuing the fixed experiment. Started `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_v2p_mediation/run_mediation.py` from the existing checkpoint; no script, data, model, prompt, protocol, or configuration changes were made.
- The runner loaded all 96 clean caches, skipped the existing 78 boundaries, and completed pairs 40–44, advancing the checkpoint to `88/192` pair-boundaries. The same batch1-only session remains active and is processing the remaining invariant pairs; no duplicate runner was started.
- No final raw parquet, summary, verdict, or report exists yet. Do not run `summarize_mediation.py` until raw collection reaches `192/192` and completeness is verified.

## 2026-08-04 20:14 CST: V2P-Mediation Collection And Offline Analysis Completed

- User requested resuming the fixed V2P-Mediation run from checkpoint and processing it after completion. Ran `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_v2p_mediation/run_mediation.py`; it resumed without protocol/model/data changes and exited successfully at `192/192` pair-boundaries in `batch1_only` mode.
- Final raw coverage is complete and duplicate-free: `base_states=1088`, `sufficiency=2688`, `reset=960`, `invariant_controls=256`; all formal rows declare batch size 1 and patch batch size 1. The checkpoint is marked `status=complete`; GPU is idle after completion.
- Fixed three offline summarizer-only bugs in `experiments/qwen35_v2p_mediation/summarize_mediation.py`: invariant base expected count (`1088`), preservation of `pair_set` in collapse keys, `SE_TASK` selection for TASK_END sufficiency, and numeric LOFO range serialization. No raw rows or scientific protocol were changed.
- Ran `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_v2p_mediation/summarize_mediation.py` successfully. Final Verdict A at both `DEC_H0` and `DEC_STAGE1`; implementation and endpoint controls passed, `causal_handoff_allowed=true`. K4 bootstrap means (95% CI): DEC_H0 M1/M2/M3=`0.3486 [0.2775,0.4334]` / `0.5208 [0.4375,0.6224]` / `0.4479 [0.3802,0.5234]`; DEC_STAGE1=`0.1564 [0.1009,0.2096]` / `0.2760 [0.2005,0.3568]` / `0.2943 [0.2318,0.3542]`. SE_P and BE_P were positive at both boundaries.
- Final artifacts include `raw/*.parquet`, `summaries/{raw_effects,primary_contrasts,bootstrap,fractions,family_robustness,verdict}.parquet/json`, six figures, and `report.md` under `outputs/qwen35_v2p_mediation/`. Interpretation remains intervention-based: do not claim PRED is the unique natural mediator or that patching fractions are information quantities; the report stops without additional sites/layers/regions/models/repair.

## 2026-08-04 20:18 CST: V2P-Mediation Data Reasonableness Recheck

- User requested a second read-only data-reasonableness audit after completion. No model forward, resume, raw-data, protocol, or experiment change was performed; only independent pandas/hash/statistical checks and this AGENTS entry were added.
- Final raw structure is internally consistent: manifest metadata matches every raw row; changed/invariant keys do not overlap; final and partial parquet files are identical as multisets; all required keys/conditions are complete and duplicate-free. Authorization hashes still match, checkpoint is `status=complete`, and all model-derived formal tables are `batch1_only` with batch/patch batch size 1.
- Independent recomputation exactly reproduces all stored K4 family means and bootstrap CIs for M1/M2/M3, SE_P, and BE_P at both DEC_H0 and DEC_STAGE1. Verdict A is therefore not caused by an aggregation mismatch: all five CIs are strictly positive at both boundaries, and SE_P/BE_P are positive in all 12 families at both boundaries.
- Reasonableness caveats: invariant controls have zero erroneous flips but nonzero continuous margin drift (mean `0.1572`, max `1.25`); changed critical patch source-answer copy is only `0.03125` at DEC_H0 and `0` at DEC_STAGE1, so the result is primarily continuous-margin intervention evidence, not categorical source-answer recovery. DEC_STAGE1 M1 is negative for one family (`typeA_m0_t2`, `-0.007812`) although its leave-one-family-out range remains positive. Descriptive SF/BF ranges are DEC_H0 `SF 0.4375–0.7407`, `BF 0.3750–0.7778`; DEC_STAGE1 `SF 0.4286–0.7500`, `BF 0.4444–0.8000`.
- Conclusion: no data-integrity or summary-recalculation problem was found. Retain Verdict A under the preregistered rule, but report it strictly as controlled activation-intervention mediation with the continuous-margin and invariant-drift caveats; do not upgrade it to unique natural mediation or information-transfer claims.

## 2026-08-04 19:41 CST: V2P-Mediation Status Audit

- User requested a read-only Phase V2P-Mediation progress check. No model forward, resume, summarization, experiment-script/data/model/protocol change, or output mutation was performed; only this AGENTS entry was appended.
- No V2P runner or GPU compute process is running. The fixed checkpoint is `batch1_only`, `180/192` pair-boundaries (93.75%); partial parquet keys contain 181 boundaries because one boundary was fully written before checkpoint advancement. All 64 changed boundaries are complete; 117/128 invariant boundaries are complete, leaving 11 invariant boundaries to collect. Current rows: base `1055`, sufficiency `2633`, reset `905`, invariant controls `234`; all expected missing rows are respectively 33, 55, 55, and 22 under the runner's actual write logic.
- Implementation authorization remains true: all recorded gates pass, including batch1-only, no-op/source=target, token alignment, hook order, hash/input audits, K5 endpoint reproduction, and fixed-row total-effect reproduction. All 96 clean caches are present and loadable (64 formal plus 32 arbiter), and the checkpoint authorization hash matches the authorization file.
- No finalized `raw/*.parquet`, summaries, figures, verdict, or report exists. Partial rows are not mediation evidence. Before the offline summarizer is run after raw completion, note that `run_mediation.py` writes 3 base rows per invariant boundary (`TARGET`, `CRITICAL`, `BACKGROUND`), so the complete base table should be 1088 rows; `summarize_mediation.py` currently expects 960 base rows and will likely fail its completeness check. Do not interpret or repair this mismatch without explicit follow-up.

## 2026-08-05 10:01:58 CST: Phase Spatial-Flow Synthesis Completed Offline

- User requested unified evidence synthesis, claim/experiment freeze, cross-model functional-stage mapping, figure specifications, and a pre-registered LLaVA-NeXT replication protocol. Strictly read-only with respect to model science: no model load, forward, intervention, new image/task, prompt/data/split change, probe fit, second-model run, natural-image run, repair, commit, reset, or checkout.
- Added only the independent output root `outputs/spatial_flow_synthesis/` and its offline audit builder `build_synthesis.py`. It contains `source_registry.json`, `evidence_matrix.csv`, `claim_registry.json`, `claim_evidence_map.csv`, `main_experiment_table.csv`, `cross_model_functional_stages.json`, `statistics_audit.json`, invalidation/claims/limitations/outline/protocol/figure documents, six offline PNG figures, recomputed audit CSVs, `synthesis_report.md`, and `synthesis_complete.json`.
- Commands run: `/root/miniconda3/envs/next3d_qwen/bin/python outputs/spatial_flow_synthesis/build_synthesis.py` (offline pandas/matplotlib only; first run stopped on a plotting-only error); then `MPLCONFIGDIR=/tmp/spatial_flow_mpl /root/miniconda3/envs/next3d_qwen/bin/python outputs/spatial_flow_synthesis/build_synthesis.py`; plus offline schema/hash/status checks and `py_compile`/forward-string audit.
- Raw Qwen3.5 V2D recomputation: 4,224 rows (1,408 changed; 2,816 invariant), 32/64 directed pairs, 12 families, both DEC_H0/DEC_STAGE1, all `batch1_only`, implementation arbiter passed, exact 196 vision rows. Family-weighted critical RR: DEC_H0 `0.118906` (95% bootstrap CI `[0.101323,0.137930]`), DEC_STAGE1 `0.060781` (`[0.047859,0.074968]`); all eight Holm hypotheses positive; both directions and all Type A/B/C families retained.
- Raw Qwen3.5 V2P recomputation: base/sufficiency/reset/invariant rows `1088/2688/960/256`, 192/192 pair-boundaries, 12 families, changed/invariant key overlap 0, all formal tables `batch1_only` with batch/patch batch size 1 and implementation/endpoint gates passed. K4 family means M1/M2/M3: DEC_H0 `0.348633/0.520833/0.447917`; DEC_STAGE1 `0.156413/0.276042/0.294271`; all six Holm tests positive and leave-one-family-out remains positive.
- Invariant drift was retained: mean `0.157227`, median `~0`, p90 `0.500000`, p95 `0.625000`, max `1.250000`, erroneous flip rate `0`. Maximum is `typeB_m1_t1`, query `RB`, boundary `DEC_H0`, visual/CRITICAL condition, source `typeB_m1_t1_I1__REL__RB` → target `typeB_m1_t1_I0__REL__RB`; it is not deleted from the audit.
- Frozen synthesis conclusion: `Selective visual evidence → contextual computation → pre-final PRED-mediated decision → late causal commitment`, explicitly under controlled activation interventions. Tier A: Qwen3.5 V2D, V2P K4, and validated PRED commitment ladder. Qwen3 target-region restoration/tracing and commitment bridge remain Tier B; COCO attention is Tier C; D2L/O2Y-capacity/carrier results are narrow Tier D boundaries. Legacy visual-restoration stage-2 authorization, old early arbiter verdicts, 198-row results, mixed batch formal results, one-pair O2Y mechanism conclusion, missing-contract fixed-neutral result, and retracted coordinate/neutral-carrier/Qwen3-retention claims are X_invalidated.
- `synthesis_complete.json` is `completed=true`. Second-priority LLaVA-NeXT protocol is frozen in `replication_protocol.md`; it reproduces only early selective restoration, pre-final PRED mediation, and late PRED commitment with the same 18-family data, behavior gate, arbiters, family bootstrap, directions, and changed/invariant controls. No second architecture or natural-image work was started.

## 2026-08-05 16:15 CST: Phase CrossArch-LLaVA Read-Only Status Audit

- User requested inspection of the frozen LLaVA-NeXT cross-architecture replication. No model forward, resume, analysis, script/data/model/protocol change, partial-output deletion, or mechanism-stage execution was performed; only this AGENTS entry was appended.
- Completed artifacts: protocol inheritance, fp16 LLaVA-NeXT-7B snapshot (`llava-hf/llava-v1.6-mistral-7b-hf`, revision `2424fdd47412fccc66d91719126b420e9fbd7065`), architecture audit, 36-image runtime packed mapping, 12 overlays, pair manifest, and mapping/layout tests. Mapping has 105,408 rows (`36×2,928`), 103,680 spatial and 1,728 newline rows; row/layout gates pass.
- Behavior is incomplete: `predictions_partial_semantic.parquet` has 72/144 intended image×query rows, 18 families, only I0, no duplicates; all 72 predictions are `left`, balanced answer labels 36/36, provisional accuracy `.50`. The older `predictions_partial.parquet` has only 39 rows and is stale. No formal `predictions.parquet`, `metrics.json`, `report.md`, calibration, restoration, commitment, mediation, or final verdict exists.
- Blocking code issue found at `experiments/llava_next_crossarch/run.py:416`: `behavior_manifest()` uses `family_id` in `sample_id` although pair manifests use image-specific IDs. I0/I1 rows collide; resume deduplication skips I1, and `clean_pair_rows()` cannot join behavior rows to pair rows. Existing semantic partial must not be interpreted or resumed as-is; fix the identifier scheme and regenerate/recompute behavior partials before continuing the frozen gate.
- Current verdict: protocol/architecture/mapping preparation only; behavior gate is not passed or failed. Do not run calibration, early restoration, PRED mediation, or late commitment until the behavior manifest/checkpoint identity issue is corrected under the same frozen protocol.

## 2026-08-05 16:27 CST: Phase CrossArch-LLaVA Identity Repair And Static Design Audit

- User requested fixing the CrossArch behavior checkpoint bug and rechecking design/code. Updated `experiments/llava_next_crossarch/run.py` and `test_protocol.py`; no model forward, weight load, mechanism run, prompt/data/model change, or partial-result deletion was performed.
- `behavior_manifest()` now uses image-specific IDs (`{image_id}__REL__{query}`), asserts 144 unique rows, and matches the directed pair manifest. Resume loading quarantines stale family-only partial rows; `clean_pair_rows()` now hard-fails on duplicate/missing behavior↔pair IDs. Existing partial files remain untouched and are not evidence.
- Fixed restoration H4 to aggregate invariant absolute drift by `family_id,boundary` rather than joining incompatible changed/invariant sample IDs. Added Holm-adjusted p-values, leave-one-family-out ranges, direction/type summaries, explicit source=target identity/no-op/full-vision implementation arbiter fields, and strict stage guards. Commitment now records no-op full-vocabulary error and checks the ordered early→pre-final→final ladder. Behavior records the fixed single-sample/no-heterogeneous-padding applicability audit.
- Pure checks passed: `8/8` protocol unittests, AST parse, Holm/LOFO helper checks, and static manifest contract: behavior `144/144` unique IDs, pair IDs `144/144` covered, queries `36×4`, labels `72/72`, pairs changed/invariant `48/96`, directions `72/72`. No `metrics.json`, final behavior table, calibration, restoration, commitment, mediation, or verdict exists yet.
- Next authorized step is the fixed behavior gate only; it will rewrite the semantic partial with image-specific rows and must pass before any mechanism stage. Do not interpret the old `72-row I0-only` semantic partial or the older `39-row` partial.

## 2026-08-05 17:01 CST: Phase CrossArch-LLaVA Corrected Behavior Gate Run

- User requested running the corrected CrossArch-LLaVA behavior gate. The sandbox CPU attempt was rejected by the protocol; the same fixed command was then run once in the approved external CUDA environment: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/llava_next_crossarch/run.py --stage behavior`. No further code, data, prompt, model, or protocol change was made.
- The corrected runner completed all `144/144` rows and discarded all `72` stale family-only partial rows. Final `predictions.parquet` is complete and duplicate-free: 144 unique `sample_id`s, 36 images, 4 queries × 36, balanced labels 72/72, image-specific ID format valid, and no duplicate image×query rows. Single-sample padding audit passed (`batch_size=1`, `expanded_seq_len=2981`, `image_rows=2928`, heterogeneous padding not applicable); candidate left/right lengths are both one token.
- Behavior gate failed: candidate-sequence semantic accuracy `0.5000`; every prediction was `left` (`left` answer accuracy `1.0`, `right` answer accuracy `0.0`); each query and Type A/B/C accuracy was `.5`; planned changed source→target clean rate `0/48`. Greedy audit separately had first-answer accuracy `.75`, candidate/greedy agreement `.875`, all 8 audited continuations valid and no-think, but failed the required `.95` greedy threshold. Gate failure flags: overall/query/left-right/changed/direction/no-label-collapse/greedy thresholds; batch padding, valid first answer, and no-think checks passed.
- Formal artifacts now include `outputs/llava_next_crossarch/behavior/{predictions.parquet,metrics.json,report.md}` and `tests/batch_padding_audit.json`. Calibration, restoration, commitment, mediation, and final verdict stages were not run. Do not interpret mechanism results or proceed past the behavior gate; the all-left semantic-scoring behavior failure requires a separate authorized diagnosis before any rerun.

## 2026-08-08 20:31 CST: Phase LLaVA-Semantic-Behavior Completed — Verdict B

- User requested freezing case-insensitive exact-word semantic class scoring and rerunning the clean 144-row behavior gate, without changing prompt, answer contract, model, tokenizer, processor, chat template, data, thresholds, or running any mechanism stage. Added `experiments/llava_next_crossarch/semantic_behavior.py` and `tests/llava_semantic_behavior/`; semantic unit tests passed `7/7`.
- Frozen hash audit matched the legacy arbiter: checkpoint `llava-hf/llava-v1.6-mistral-7b-hf`, revision `2424fdd47412fccc66d91719126b420e9fbd7065`, prompt/data/processor/tokenizer/model hashes unchanged, prediction row `2980`. Result-blind vocab classes were frozen before logits: `T_LEFT={1749,1828,7549,18365,20528}` and `T_RIGHT={1103,1246,7552,9554,14166}` based on decoded NFKC/strip/casefold exact words; SentencePiece raw boundary markers were retained only as audit metadata.
- Semantic scorer validity passed on the fixed 16-row audit: classes nonempty/disjoint, specials excluded, finite/no ties, batch-1 repeat stable, hashes and prediction row valid. Formal score is `m_sem=logsumexp(z[T_LEFT])-logsumexp(z[T_RIGHT])`; greedy agreement was diagnostic only. Full run used batch=1, eval/inference mode, `use_cache=False`, `do_sample=False`, and deterministic greedy `max_new_tokens=8`; a durable tmux session was needed because foreground CUDA jobs were reaped.
- Final full behavior is complete and duplicate-free: `144/144` rows, RB/BR/GR/GB each 36, 18 families, Type A/B/C each 48. Semantic accuracy: overall `.85417`; RB `1.0000`, BR `.77778`, GR `.80556`, GB `.83333`; ground-truth left `1.0000`, right `.70833`; Type A `.83333`, B `.87500`, C `.85417`; predicted-left fraction `.64583`.
- Frozen behavior gate failed: semantic overall, BR, right-label, and changed-pair clean retention (`34/48=.70833`) were below thresholds; both clean directions were present (`17/17`). Diagnostics: lowercase exact accuracy `.50` with all-left predictions; capitalized exact accuracy `.85417`; greedy semantic accuracy `.86806`, semantic-valid fraction `1.0`, strict-format fraction `0`, semantic/greedy agreement `.98611`; mean casing biases `Bias_left=10.7815`, `Bias_right=12.0174` logits.
- Final verdict is `SEMANTIC_BEHAVIOR_FAIL`, `mechanism_allowed=false`. Created `outputs/llava_next_crossarch/semantic_behavior/` artifacts including frozen classes/spec, audit/full predictions, metrics, planned/clean changed and invariant manifests, downstream adaptation, report, tests, and verdict. `crossarch_semantic_protocol.json` was correctly not created because behavior failed. No visual mapping, calibration, restoration, commitment, mediation, attention, probe, prompt search, model/checkpoint search, family deletion, or threshold change was run. Stop this LLaVA-NeXT checkpoint for the preregistered causal replication; retain old behavior outputs read-only.

## 2026-08-09 CST: Phase R1 Natural Local Visual State Geometric Recomposition Pilot — Verdict Support

- User requested the six-family R1 treatment-semantics pilot: test whether the audited Qwen3.5 DEC_H0 natural critical visual state follows source relation or recomposes under a new target anchor. Reused the formal Qwen3.5 non-thinking REL renderer/scorer (`left=2282`, `right=1246`), current Qwen3.5-4B checkpoint, batch1-only execution, V2D-Boundary DEC_H0 decoder-layer-0 pre-hook, runtime 14×14/196 mapping, V2D shuffled-delta definition, matched background/blue controls, and family bootstrap helpers. No confirmatory family, DEC_STAGE1, PRED mediation, layer search, probe, prompt, checkpoint, or existing confirmatory output was changed.
- Added `experiments/phase_r1_geometric_recomposition/run.py` and its README. Corrected the initial geometry feasibility choice per user instruction: restored the frozen 2×2 object footprint and used one left/right target anchor per family with fixed legal slots `x_old,L=0 < r_L=3 < x_G*=6 < r_R=9 < x_old,R=12`; blue is fixed on a separate noninterfering y row, red/green y is fixed. Generated 36 current-manifest images from the six frozen pilot family IDs and 72 GR/RG tasks. The first temporary 1×1 manifest was overwritten before model execution and is not scientific input.
- Pixel/mapping gates passed: 12 source-left/source-right/hybrid C-region checks had exact RGB equality and red-only outside-C differences; all 36 images mapped to exactly 196 decoder image rows with exact 2×2 bbox→merged-token alignment. Clean behavior passed 72/72: overall/GR/RG = `1.0/1.0/1.0`, every S/T/H role = `1.0`, no clean failures.
- DEC_H0 implementation gates passed in batch1: query-invariance GR-vs-RG visual-row max diff `0.0`; 12 full-image H→T reconstruction rows had max full-vocab logit diff `0.0` and top-1 agreement `1.0`; 48 source=self/no-op checks had max full-vocab diff `0.0`. Full raw completeness passed `336/336` rows with zero duplicate keys; NOOP raw margin shift was exactly `0.0` for 24 rows. A transient first-run permutation-width bug (`8-token critical union` vs `2-row permutation`) was captured in `runtime_error.json`, fixed to the actual V2D 8-row width, resumed from partial checkpoint, and produced the complete valid raw output; the transient artifact was removed after resolution.
- Scientific results on all predefined clean-correct rows: HYBRID_LOCAL_ORACLE mean `E_H=1.21875` (6/6 family-positive); SOURCE_MATCH `1.21875`; SOURCE_CONFLICT `0.53125` (6/6 family-positive); CRITICAL_SHUFFLED `0.386719`; BACKGROUND `0.03125`; BLUE_CONTROL `0.06250`. SOURCE_CONFLICT raw direction matched the clean H direction at `24/24`, recomposition accuracy `1.0` overall, by family, query (GR/RG), and anchor side. Raw conflict delta sign flipped as required: GR left `-0.75`, GR right `+0.50`; RG left `+0.50`, RG right `-0.375`. Two-factor estimates: target-geometry coefficient `0.8750`, source-relation coefficient `0.34375`, interaction `-0.15625`; target term was larger.
- All six frozen stop-rule criteria A–F passed, so data-driven verdict is `PILOT_SUPPORTS_GEOMETRIC_RECOMPOSITION`. This supports only the permitted treatment-semantics interpretation: in this controlled cross-context interchange, source-conflict critical states still redirected target decisions according to the target-anchor-defined hybrid relation. It does not claim an explicit coordinate code, absence of relation representation, uniqueness of DEC_H0, or token-wise information percentages. No 12-family confirmatory run was automatically started.
- Primary artifacts are under `outputs/phase_r1_geometric_recomposition/`: `config.json`, `family_manifest.json`, `image_manifest.json`, `pixel_equivalence_audit.json`, `mapping_audit.json`, `behavior_audit.json`, `query_invariance_audit.json`, `implementation_audit.json`, `raw_rows.parquet`, `family_summary.csv`, `pilot_summary.json`, `fig_anchor_sweep.{png,pdf}`, `fig_match_vs_conflict.{png,pdf}`, `row_completeness.json`, `verdict.json`, and supporting state/mapping/arbiter files. Commands used: fixed `--stage manifest`, `--stage mapping`, durable CUDA `--stage behavior --batch-size 1`, durable CUDA `--stage arbiters --batch-size 1`, durable CUDA `--stage intervention --batch-size 1` with resume, and local `--stage analyze`.

## 2026-08-10 CST: Phase R1 Post-Pilot Integrity + Specificity Audit

- User requested a strictly artifact-only post-pilot audit. Added `experiments/phase_r1_geometric_recomposition/audit/run_audit.py` and generated all audit artifacts under `experiments/phase_r1_geometric_recomposition/audit/`. No model forward, checkpoint load, prompt/geometry/object-size/shuffle change, layer search, DEC_STAGE1, PRED mediation, probe, confirmatory-family access, or R1 output mutation was performed. Commands: `py_compile` and `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/phase_r1_geometric_recomposition/audit/run_audit.py`.
- Independent raw recomputation passed exactly: `336/336` rows, no duplicate/missing keys, all expected cells and metadata complete, stored `delta_m/E_H/oracle_direction/RR_H` exactly reproduced, and all six primary means matched the prior summary (`oracle/match=1.218749924`, conflict `0.531249875`, shuffled `0.386718644`, background `0.031249929`, blue `0.062499878`). Clean R1 behavior and existing implementation gates remain unchanged and passing.
- Critical finding: for all `24` family×anchor×query pairs, `HYBRID_LOCAL_ORACLE` and `SOURCE_MATCH` are whole-image pixel-identical and have the same red anchor. With frozen slots `x_old,L=0 < r_L=3 < x_G*=6 < r_R=9 < x_old,R=12`, left-target H is exactly `S_RIGHTREL`, and right-target H is exactly `S_LEFTREL`. Existing DEC_H0 cache confirms clean logits and critical DEC_H0 states are bit-identical for H/M in `24/24`; H–CONFLICT critical-state Frobenius L2 mean is `18.732628`. This is not a quantization-only explanation and triggers the user-defined source-match identity failure.
- Specificity diagnostics: SOURCE_CONFLICT `A_recompose=1.0` (`24/24`); shuffled `0.90625`; background/blue `0.5/0.5`. Family-paired `D_CS=E_conflict−E_shuffled=0.144531231`, positive in `6/6`; conflict-background/blue contrasts are `0.499999947/0.468749998`, positive in `6/6`. Conflict signs flip in all four GR/RG×left/right cells; shuffled preserves most direction but with weaker magnitude.
- Corrected factorial audit is balanced and full-rank on the observed design (Model A rank `10/10`, condition number `6.8541`; Model B `9/9`, same condition number). However, the old R1 factorial fields were query-conditioned answer labels and the old model omitted an explicit query term, so `OLD_FACTORIAL_ANALYSIS_INVALID=true`; corrected canonical Model A coefficients are `R_H=.281250043`, `R_S=.187499966`, query `-.468749982`, interaction `-.156250010`. Six-family fixed-effect bootstrap resamples are rank-deficient in most samples and are diagnostic only.
- Final data-driven audit verdict: `AUDIT_IMPLEMENTATION_FAIL`; `recommend_confirmatory=false`. Do not interpret the prior R1 `PILOT_SUPPORTS_GEOMETRIC_RECOMPOSITION` as a clean H-vs-M specificity result and do not run the 12-family confirmatory experiment. Required outputs include `audit_recomputed_summary.json`, `audit_raw_completeness.csv`, `audit_oracle_vs_match_rows.csv`, `audit_recomposition_accuracy.csv`, `audit_family_paired_contrasts.csv`, `audit_anchor_query_signs.csv`, `audit_factorial_design.csv`, `audit_factorial_models.json`, `audit_shuffle_semantics.md`, `audit_activation_distances.csv`, `audit_verdict.json`, and `AUDIT_REPORT.md`.

## 2026-08-10 CST: R/O Decoding Read-Only Retrieval

- User requested the R/O decoding experiment's design and results, explicitly excluding carrier-transplant results. Read-only retrieval only: no model load/forward, probe refit, intervention, prompt/data/model change, or output creation occurred beyond this log entry.
- Located the relevant Phase O2Y materials: `outputs/qwen35_o2y/{protocol.md,final_report.md,manifests/manifest_summary.json,subspaces/pilot_o_validation_grid.csv,subspaces/pilot_fit_summary.json}` and implementation in `experiments/qwen35_o2y/run.py`.
- Here `O` is the 3×3 red/blue/green identity-to-rank assignment; pairwise relation `R` is an equivalent diagnostic readout of O, not an independently posited latent. The decoder is standardized ridge on same-geometry scene-permutation activation differences, fit on 24 train families and selected on six validation families; the six-family test split was unused. Best validation O results: OBJ_BUNDLE/stage_2 OBJ slot/scene exact `1.000/1.000`; TASK_END stage_3 REL `1.000/1.000`; PRED stage_4 REL `.9444/.8333`; PRED stage_3 SELECT `.9444/.8333`; TASK_END stage_2 SELECT `.7037/.3889`.
- These are representation-only validation diagnostics, not causal or reusable-O evidence. Carrier-token/span transplants and any other patch results were deliberately excluded from the retrieval.

## 2026-08-10 CST: Phase O2Y Complete Design/Result Read-Only Review

- User asked for a detailed end-to-end review of the R/O (O2Y) experiment, including concepts and calculations, while excluding Phase 3A carrier-transplant results. Read-only source/artifact audit only; no model forward, refit, intervention, output regeneration, prompt/data/model change, or new experiment was run.
- Verified the complete pipeline in `experiments/qwen35_o2y/run.py` and fixed artifacts under `outputs/qwen35_o2y/`. O is the 9-dimensional identity-to-rank one-hot target; R is derived from O (rank comparison), while `REL` and `SELECT` are task-conditioned contexts. The 36 existing scenes are six identity permutations × six geometry templates; split is four templates/24 scenes train, one template/six scenes validation, and one template/six scenes test. Only 270 train+validation REL/SELECT rows were state-cached; test remained unread.
- Decoder procedure: cache post-block residuals at full-attention stage_2/3/4 (code indices 11/19/23) for OBJ_BUNDLE, TASK_END, PRED; form within-same-geometry ordered source-target differences. This produces 120 OBJ, 720 REL, and 360 SELECT training difference rows. Fit StandardScaler+Ridge (no intercept) to map activation differences to O differences; validation centers activations inside its geometry family, applies Ridge, adds the 1/3 O baseline, and argmaxes each identity's three rank logits. Alpha grid is 1e-4…1e4. Important implementation fact: rank is used only to SVD/QR-truncate the saved intervention basis; the validation decoder remains the full Ridge model, so selected rank=1 does not establish a 1-D decoder.
- Best per stage/site/mode validation decoding: stage_2 OBJ_BUNDLE OBJ 1.000 slot/1.000 all-three; TASK_END REL .889/.694 and SELECT .704/.389; PRED REL .935/.861 and SELECT .833/.556. stage_3 OBJ_BUNDLE 1/1; TASK_END REL 1/1 and SELECT .556/.278; PRED REL .898/.750 and SELECT .944/.833. stage_4 OBJ_BUNDLE 1/1; TASK_END REL .972/.917 and SELECT .648/.389; PRED REL/.944/.833 and SELECT .944/.833. Reported values are validation representation diagnostics, not test generalization or causal evidence.
- The O2Y-specific validation causal pilot (not Phase 3A carrier transplant) executed only one same-geometry source→target pair and only OBJ_BUNDLE/stage_2. Across the nine target queries, the four novel-composed-answer rows had O-CAR 0 for U_O, random, shuffled, and whole-state conditions; target retention was 1 and no-op score difference 0. A two-example stage_4 PRED whole-state positive control also had source-answer-copy rate 0. Therefore the preregistered stop fired: no other recombination configurations, no test use, no 210-scene confirmation, and no reusable-O causal conclusion.

## 2026-08-10 CST: Direct R/O Decoder Retrieval

- User requested an artifact/code search for an independent direct R/O decoder, rather than O2Y's source-target difference decoder. Read-only search only; no model forward, refit, artifact mutation, or experiment execution occurred.
- Found two completed direct canonical-relation R probes on single prediction-position states: `outputs/phase3a_descriptor_carrier_pilot/evidence_audit/prediction_probe_audit_all_layers.csv` (standardized LogisticRegression, train/validation/test family split 192/48/48; ordinary all-to-all direct R BACC reaches 1.0 at layers 15–27, L28 `.9792`, but it is a two-object D2 task and has no O target) and `outputs/phase3a_composition_localization/boundary_probe_results.csv` (same direct logistic setup from one residual state at each boundary; direct R reaches BACC 1.0 from block 15 h_post_attn onward). These probe fits are independent analyses; no carrier patch/transplant result was used in the retrieval.
- Found no completed direct single-state decoder with the full 9D identity-to-rank O target. `qwen35_o2y/run.py` is the only O decoder and explicitly trains on within-geometry differences then evaluates centered states. `qwen35_unified_structure` supplies direct RANK behavior data, but no hidden-state O/R decoder. Direct coordinate ridge probes exist in the Phase 3A audit, but are neither direct R nor full O decoders.

## 2026-08-10 CST: O2Y Independent Pairwise Difference Validation And Held-Out Test

- User requested a direct pairwise validation of the O2Y hidden-state-difference mapping with train and test separated. Added `experiments/qwen35_o2y_pairwise_validation.py` and independent output root `outputs/qwen35_o2y_pairwise_validation/`; original O2Y artifacts and its causal pilot were not modified or rerun. `py_compile` passed. The first sandbox CUDA attempt failed due no NVIDIA driver; approved durable CUDA extraction initially hit `CUBLAS_STATUS_INTERNAL_ERROR` after seven rows, then resumed from `.npz` checkpoint and completed all 54 test states. Offline evaluation was rerun after changing the script so test labels are materialized only after validation has fixed alpha.
- Frozen split/coverage: original train/validation pilot cache has 216/54 task rows; newly extracted held-out test has 54 rows from six `small_left_large_right_t2` scenes, with sample-ID overlap 0. Test OBJ_BUNDLE state is exact across all nine prompts per image at stage_2/3/4 (max abs diff 0). Train/validation/test pair counts are OBJ 120/30/30, REL 720/180/180, SELECT 360/90/90 for each stage/site mode; 1,710 test pair prediction rows were saved.
- Protocol: StandardScaler+Ridge fit only to directed within-geometry train differences `d_h -> d_O`, alpha in {1e-4,.01,1,100,10000} chosen solely by validation 9D matrix MSE, then a fresh train-only model evaluated test once. New numerical endpoints are all-9-entry matrix MSE/MAE/R2 and per-identity scalar-rank-delta MAE/RMSE (left/middle/right=0/1/2); pair rows are reported descriptively and not treated as independent scenes. Low-alpha Ridge issued expected small-n/high-dimensional singular/ill-conditioned solver warnings.
- Held-out test results: OBJ_BUNDLE is strong at every depth: stage_2 MSE `.005583`, R2 `.989532`, rank-delta MAE `.083632`; stage_3 `.005467`, `.989749`, `.053941`; stage_4 `.007745`, `.985478`, `.056436` (30 directed pairs each). Best task-conditioned site is stage_3 PRED SELECT: MSE `.052809`, R2 `.900984`, rank-delta MAE `.250636` over 90 pairs. Stage_3 TASK_END REL/SELECT: MSE `.065863/.058620`, R2 `.876507/.890088`, rank-delta MAE `.213071/.205347` over 180/90 pairs. PRED REL is weaker but positive (test R2 `.740308/.805764/.757472` for stages 2/3/4); stage_4 TASK_END has R2 `.826028` REL and `.852518` SELECT. This is the first held-out numerical pairwise test of the mapping: it supports robust linear prediction of O differences, especially OBJ_BUNDLE, but does not alter the prior O2Y causal-stop conclusion (U_O recomposition remained unestablished).

## 2026-08-12 CST: V2D Selective Visual Causality Retrieval

- User asked to locate the completed V2D — Selective Visual Causality experiment. Read-only artifact retrieval only; no model forward, analysis rerun, or experiment-output modification occurred beyond this log entry.
- Located the completed Phase V2D-Boundary experiment: implementation in `experiments/qwen35_v2d_boundary/`; final artifacts in `outputs/qwen35_v2d_boundary/`, led by `report.md`, raw confirmatory changed/invariant Parquets, implementation arbiter/authorization, summaries, figures, manifests, and overlays.
- Final status is Verdict A, implementation passed, with `batch1_only` execution. Confirmatory raw coverage is complete: 4,224 expected rows, no missing, duplicates, or unexpected rows. Qwen3.5 DEC_H0 and DEC_STAGE1 both show positive selective critical-region RR across all 12 families; family-weighted RR `.118906` (95% CI `[.1013,.1379]`) and `.060781` (`[.0479,.0750]`), respectively. Scope remains continuous-margin causal evidence; source-answer copy is rare and no portable carrier/complete scene-graph claim is licensed.

## 2026-08-12 CST: V2D Complete Plain-Text Design/Results Explanation

- User requested a complete pure-text explanation of the completed V2D selective-visual-causality dataset, design, formulas, and results. Read-only reconstruction from the final V2D report, manifests, implementation scripts, arbiters, and summary Parquets; no model forward, analysis rerun, or V2D output modification occurred beyond this log entry.
- Explained the 18-family/36-image source Grid-Exact corpus, fixed 6-pilot/12-confirmatory split, three-object/14×14 image-token geometry, four relation questions, A/B/C green-move contrasts, changed versus invariant directed pairs, exact batch1-only activation restoration at DEC_H0 and DEC_STAGE1, the 8-token critical green union and matched/shuffled/random/full/no-op controls, cluster-bootstrap/Holm inference, and final scoped Verdict A.

## 2026-08-12 CST: V2D Prompt/Renderer Mismatch Read-Only Audit

- User asked why the red/blue/green objects are all squares despite the V2D prompt calling them a red square, blue circle, and green triangle. Read-only code/artifact audit only; no images, prompts, models, or V2D outputs were changed.
- Confirmed a real visual-text mismatch: `experiments/phase3a_functional_spatial_retention.py` hard-codes shape metadata and prompt labels (`square/circle/triangle`) in `make_scene`, but `grid_exact_dataset/src/grid_exact/renderer.py:render_scene` ignores `shape` and uses `Image.paste` with each object's rectangular bounding box. Thus all 36 source images are colored 2×2-cell rectangles. The source report itself states this renderer caveat.
- Consequence: colors still uniquely identify the queried object and source/target prompts are identical, so this does not by itself negate the controlled within-pair patch contrast; however it is an avoidable multimodal semantic inconsistency. The V2D conclusion must be scoped to color-addressed grid rectangles under mismatched shape language. A new protocol would need a frozen prompt-only behavior/implementation audit using neutral color-object descriptors (or a genuine shape renderer) before treating the effect as shape-faithful spatial visual causality.

## 2026-08-12 CST: B/O-Readable Subspace Causal Interchange — Runtime-Blocked Before Capacity Gate

- User requested a new independent B/O-readable-versus-decoder-orthogonal natural-difference causal interchange pilot. Added `experiments/qwen35_bo_subspace_interchange.py` and `outputs/qwen35_bo_subspace_interchange/`; original O2Y and pairwise-validation trees were not modified. The runner freezes primary `OBJ_BUNDLE/stage_2/OBJ` and only authorizes the specified `REL TASK_END/stage_3` fallback after primary capacity failure.
- Audited and rebuilt the original `qwen35_o2y_pairwise_validation` decoder because it had no saved model: exact train-only directed within-template differences, validation-selected alpha `.01`, `StandardScaler -> Ridge(fit_intercept=False)`. Held-out complete-pair reproduction passed to numerical precision: matrix MSE `.0055831564`, R2 `.9895316`, rank-delta MAE `.0836316`. Saved frozen estimator and artifacts under the new independent output root.
- Raw projector uses `W_raw=W_z diag(1/scale)`; numerical SVD rank is 4 at `1e-6*sigma_max`. Pipeline-equivalence max error is `2.24e-7`, reconstruction `1.39e-17`, relative parallel/orth dot `3.39e-16`, and decoder-orthogonal residual max `2.42e-7`. The 108 frozen held-out REL directed pairs have exact same prompt/query, same template, clean-correct source/target, and opposite answers. Caveat: inherited held-out split has only one geometry template, so family/template bootstrap is degenerate.
- Two predeclared-pair implementation sanity rows exist in `raw/implementation_sanity.parquet`: NOOP effect exactly zero; raw decoder BO displacement matched FULL and decoder-orthogonal residual was near null. These rows are wiring-only, not causal evidence. Full 108-pair primary FULL capacity gate and decomposition pilot did not run.
- Runtime blocker: foreground CUDA runs were killed by the execution environment; a durable session then hit CUDA OOM loading frozen Qwen3.5-4B bf16/no-offload (1.19GB allocation, ~6.87GB available on 8GB GPU), including after skipping Transformers' transient allocator warm-up. No quantization/offload/model/dtype/prompt/data/site/rank change or fallback was used. `report.md` explicitly gives no Outcome A/B/C verdict. Resume only on a runtime where the frozen original model configuration loads, then run primary FULL capacity first.

## 2026-08-12 CST: Qwen3.5 Loader Audit — V2D Formal Runner vs B/O Interchange

- User requested a strict loader-only comparison before any B/O retry; B/O experiment code and causal pilot were not changed or run. Added independent `experiments/qwen35_v2d_boundary/loader_audit.py` and `outputs/qwen35_loader_audit/{v2d_exact_loader_audit.json,bo_exact_loader_audit.json,report.md}`. It calls the formal V2D loader or the existing B/O loader helper, then performs only one existing clean V2D REL forward.
- Both paths successfully loaded `models/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, native `Qwen3_5ForConditionalGeneration`, and completed the same clean forward. Both have 4,539,265,536 parameters in 723 tensors, all `torch.bfloat16` on `cuda:0`; CPU parameter count 0; theoretical CUDA parameter bytes 9,078,531,072; no `hf_device_map`/offload/quantization/max_memory/attn-implementation/trust-remote-code configuration. Accelerate 1.13.0 is installed but neither helper imports/uses an Accelerate dispatch map.
- Both call the same underlying `load_model("bfloat16")` with `device_map={"":0}`. The sole code difference is B/O's existing temporary no-op replacement of Transformers `caching_allocator_warmup`; it does not change model placement. Thus initial B/O OOM was not an accidental auto/offload→full-CUDA change; exact old V2D full-CUDA loading still works in a fresh process. CUDA allocated/reserved accounting after load exceeded nominal 8GB in both processes, so it is recorded verbatim and the initial OOM is scoped as transient allocator/loading state, not explained by a declared loader configuration mismatch.

## 2026-08-13 CST: B/O Interchange Primary Capacity And Decomposition Stop

- User authorized continuing the frozen `experiments/qwen35_bo_subspace_interchange.py` protocol after loader audit. No decoder/projector/rank/site/model/dtype/device placement/prompt/pair/data change was made. Ran primary `OBJ_BUNDLE/stage_2` FULL gate only first, then the two remaining predeclared BO_READABLE and DECODER_ORTHOGONAL conditions after capacity passed; no fallback and no new controls were run.
- FULL gate covered all 108 same-template/same-query/opposite-answer held-out REL pairs plus 108 NOOP rows. NOOP max absolute CE was 0.0. FULL mean CE `.559028`, mean RR `.0342963`, positive CE/RR on 100/108 pairs, source-answer copy 0; capacity gate passed on directional margin effect, so fallback `REL TASK_END/stage_3` was not run.
- Decomposition completed all 216 BO/ORTH forwards and combined raw coverage is 432 rows (108 per NOOP/FULL/BO_READABLE/DECODER_ORTHOGONAL). The final frozen manipulation check then failed: BO decoder displacement fraction mean `.9999999999999906`, but ORTH maximum decoder-displacement fraction `1.8687476544446164e-7` exceeded that check's `1e-7` cutoff (absolute residual max `2.421084e-7`). Although this is below the separate prior raw-projector audit tolerance `1e-5`, thresholds were not reconciled or loosened post hoc.
- Per the specified stop rule, no BO-vs-ORTH downstream causal interpretation, bootstrap, Outcome A/B/C claim, fallback, tolerance change, rerun, or additional experiment was made. `outputs/qwen35_bo_subspace_interchange/report.md` records an implementation/manipulation-check stop; raw rows remain for traceability only. Single-template discovery limitation remains.

## 2026-08-13 CST: B/O Interchange Numerical Arbitration — N1 Projector Construction Failure

- User requested a sealed numerical-only arbitration of the `1.8687e-7` ORTH decoder-displacement fraction. Added independent `experiments/qwen35_bo_subspace_numerical_arbitration.py` and `outputs/qwen35_bo_subspace_interchange/numerical_arbitration/`; did not modify or rerun the B/O causal runner, decoder, projector, rank, site, model, dtype, prompt, pairs, or data. The script never opens the causal-intervention parquet; BO-vs-ORTH downstream CE/RR/copy outcomes were not inspected, aggregated, or reported.
- Located frozen gate at `experiments/qwen35_bo_subspace_interchange.py:395`: `||W_raw delta_orth||₂ / ||W_raw delta_full||₂ <= 1e-7`. It is computed on the pre-cast `ds` vector, before float32/bf16 hook casting or a target hidden-state write. No rationale for the hard-coded threshold was found in code, artifacts, reports, or available history.
- Pure float64 reconstruction on all 108 pairs already reproduces the failure: ORTH residue median/max `2.889e-7/4.304e-7`, fraction median/max `1.368e-7/1.869e-7`; reconstruction error is ~`1e-17` and normalized BO/ORTH dot ~`1e-16`. The frozen rank cutoff is `1.3255065e-6` and retains rank 4, excluding a nonzero fifth singular value `1.2959106e-6`; this thresholded direction causes the mathematical residual. Float32 preserves it.
- A no-forward bf16 pooled-state write proxy (exact tokenwise states were not saved) shows a separate, much larger expected write/cancellation floor: ORTH residue median/max `4.454e-4/7.613e-4`, with median relative write error `.00709`. This does not cause the original gate failure because the original gate precedes the runtime write, but shows the cutoff was not derived from bf16 write precision.
- Numerical verdict N1: projector/manipulation construction failure under the requested definition. No precision-aware gate revision is proposed/applied, causal outcomes remain sealed, and no Outcome A/B/C conclusion is licensed.

## 2026-08-13 CST: Treatment-Specific Text-State Coalition Interchange — Implementation Complete, CUDA Execution Not Completed

- User requested a strictly frozen discovery pilot asking whether the existing V2D `DEC_H0 + CRITICAL_EXACT` treatment-induced K4 text-state delta is individually or jointly causally efficacious across SUBJECT, REFERENCE, and TASK_END. Added `experiments/qwen35_v2d_text_coalition_pilot/run.py`, package init, and `tests/qwen35_v2d_text_coalition_pilot/test_protocol.py`; independent output root is `outputs/qwen35_v2d_text_coalition_pilot/`.
- Runner resolves exactly the 32 V2D changed GR/GB directed pairs across 12 confirmatory families; rejects invariant/non-GR/GB/incorrect/opposite-answer violations. It freezes 196 decoder image rows, V2D `CRITICAL_EXACT` 8-token `DEC_H0` write, bf16, batch size 1, and K4 uniquely at `model.language_model.layers.23`, code index 23, full-block post-output, normalized depth .75. No V2D treatment, region, prompt, PRED definition, model, layer selection, scaling, projection, or factorial expansion is changed.
- Token-role hard gates use the exact rendered prompt and existing TASK_END convention (final ordinary token before the answer contract), save rendered text/token IDs/positions, demand identical CLEAN_TARGET/VISUAL_TREATMENT IDs and role locations, exclude image rows 4–199, vision controls, specials and PRED, and reject overlap between SUBJECT/REFERENCE/TASK_END. Phase S has only S0–S6; raw deltas are unscaled. Statistics first average directed rows within each family, use a 12-family paired bootstrap, and report CE/RR, direction, count, and delta L2. Phase R reset is programmatically forbidden except S-A.
- Unit tests passed with `/root/miniconda3/envs/next3d_qwen/bin/python -m unittest tests/qwen35_v2d_text_coalition_pilot/test_protocol.py -v`; frozen source audit also passed (32 rows / 12 families / GB+GR / frozen candidate IDs 2282/1246 / unique K4 mapping). CUDA `--stage audit` and `--stage all` foreground invocations were terminated by the execution environment while loading the frozen model and wrote only `audit/{frozen_v2d_reference.json,fixed_k4_site.json}`; no token role audit, trajectories, intervention rows, Phase S verdict, reset, or scientific conclusion exists. Resume only on a durable CUDA execution path with the same command; do not interpret absence/partial output as a Phase-S outcome.

## 2026-08-14 CST: Difference-Decoding Stage Semantics Read-Only Audit

- User asked whether stage_3/4/5 in the difference-decoding work denote Qwen3.5 layers or experiment steps, whether full depth is traversed, and whether fits exist by depth. Read only `experiments/qwen35_o2y_pairwise_validation.py`, O2Y stage manifests/state caches, V2P runtime code, and saved pairwise summaries; no forward, fit, artifact, protocol, or model change was made.
- `stage_n` is a macro-boundary alias for a fixed Qwen3.5 decoder block output—not an independent experiment phase. From the frozen macro manifest: stage_2=`layers.11` (normalized depth .375), stage_3=`layers.19` (.625), stage_4=`layers.23` (.75), stage_5=`layers.31` (1.0). Each normal model call traverses the complete 32-layer decoder; forward hooks merely capture chosen outputs. The difference-decoder cache hooks only stage_2/3/4, so it does not extract every layer output or conduct a full layer sweep.
- The independent complete-pair O-difference validator fits distinct `StandardScaler -> Ridge(fit_intercept=False)` models for every `stage ∈ {stage_2,stage_3,stage_4}` and every site/mode (OBJ_BUNDLE/OBJ; TASK_END/REL and SELECT; PRED/REL and SELECT): 15 final selected-alpha fits, with train-only pair differences, validation-only alpha selection, and one held-out test evaluation. Saved summary has five configurations per stage. It has no stage_5 decoder fit/data; K5/stage_5 occurs in separate PRED-capacity/V2P mediation work, not in that difference-decoding fit. For OBJ_BUNDLE/OBJ test MSE/R² were stage_2 `.005583/.989532`, stage_3 `.005467/.989749`, stage_4 `.007745/.985478`.

## 2026-08-14 CST: V2P Follow-up Text-Coalition Read-Only Review

- User requested a retrospective of the V2P follow-up that replaces source-scene PRED transplantation with partial text-token intervention. Read only `experiments/qwen35_v2d_text_coalition_pilot/run.py`, its unit test, the V2D/V2P frozen reports and current output root; no model forward, resume, script/protocol/data/model change, or result interpretation was performed.
- The follow-up is a distinct frozen discovery pilot, not another V2P PRED-mediation run. It starts from the same target trajectory after frozen V2D `DEC_H0 + CRITICAL_EXACT` (8 of 196 image rows), captures K4/layer 23 post-block, and writes only the resulting within-target delta `h_K4^(visual-treatment) - h_K4^(clean)` to target text positions. It never transplants a source-scene text state or PRED; it uses 32 changed GR/GB directed pairs across 12 confirmatory families, bf16 and batch1-only.
- Predeclared sufficiency groups are NOOP, SUBJECT, REFERENCE, TASK_END, SUBJECT+REFERENCE, SUBJECT+REFERENCE+TASK_END semantic coalition, and ALL_NONPRED_TEXT. Exact rendered-prompt token alignment excludes image/control/special/PRED rows and requires the three semantic groups to be disjoint. Family-first CE/RR summaries and a 12-family bootstrap test whether the semantic coalition exceeds the best single token group and is supra-additive; only an S-A result can unlock reset/blockade, otherwise execution stops.
- Current progress remains pre-execution: the protocol unit test and frozen-input audit passed, but foreground CUDA invocations ended during model load. Only `audit/frozen_v2d_reference.json` and `audit/fixed_k4_site.json` exist; there is no token-role audit, trajectory/state cache, intervention parquet, Phase-S verdict, reset, final report, or scientific outcome. Resume unchanged only in a durable CUDA session with `experiments/qwen35_v2d_text_coalition_pilot/run.py --stage all`.

## 2026-08-14 CST: Treatment-Specific Text-State Coalition CUDA Resume — Token-Alignment Implementation Stop

- User authorized continuing the frozen experiment. Launched the unchanged command in durable external CUDA tmux session `v2d_text_coalition`: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_v2d_text_coalition_pilot/run.py --stage all`. The frozen Qwen3.5-4B loaded successfully (all 723 tensors); no model/data/prompt/region/site/dtype/batch/protocol change was made.
- The runner stopped before its first model forward and before writing `token_alignment_audit.json`: in `_align_prompt_tokens`, `ALL_NONPRED_TEXT` attempts to align one combined prompt-token sequence via `locate_subseq`, but that sequence has zero occurrences in the actual decoder input (`RuntimeError: subsequence occurrence 0 not found; matches=0`, `run.py:185`). SUBJECT/REFERENCE/TASK_END had not yet been completed/audited for the first row, and no trajectory/state cache/sufficiency/reset/summary/result file was written.
- This is an implementation gate failure, not a Phase-S result or scientific outcome. Preserve the current two initial audit files and the run log; do not repair token alignment, substitute a token convention, launch a duplicate, or interpret the failed run without explicit authorization for an implementation-repair protocol.

## 2026-08-14 CST: Treatment-Specific Text-State Coalition — Authorized ALL_NONPRED_TEXT Alignment Repair and Resume Active

- User explicitly authorized only a repair of `ALL_NONPRED_TEXT` alignment, with every other protocol component frozen. Diagnosed the failure with the first frozen target prompt: whitespace-only raw tokens (newlines) were dropped before a contiguous raw token subsequence was aligned, producing an artificial mismatch even though the full ordinary prompt text occurs at decoder rows 201–253. Changed only `experiments/qwen35_v2d_text_coalition_pilot/run.py`: retain raw whitespace text tokens while forming `ALL_NONPRED_TEXT`; the pre-existing image/control/special/PRED exclusion gates remain unchanged.
- `tests/qwen35_v2d_text_coalition_pilot/test_protocol.py -v` still passed (2/2). A fresh native CUDA `--stage audit` run loaded the frozen model and passed exact token alignment for all 32 pairs; `audit/token_alignment_audit.json` records `passed=true`, and GB/GR each have 53 all-non-PRED text positions. No intervention conditions were run in this audit.
- Started the unchanged `--stage all` runner in the sole durable external CUDA tmux session `v2d_text_coalition_run`. It reuses the passed audit and writes resumable `raw/trajectories_partial.parquet`; at latest check it is live and has completed 5/32 trajectories. No sufficiency, Phase-S, reset, summary, or scientific conclusion exists yet; partial trajectories must not be interpreted. Do not launch another runner or alter the frozen protocol.

## 2026-08-14 CST: Text-Coalition CUDA Progress Check

- User requested a read-only running-status check. The sole `v2d_text_coalition_run` tmux session and CUDA process remain live; no duplicate or intervention change was made.
- All 32/32 trajectory pairs are complete in both `raw/trajectories.parquet` and its partial checkpoint, with 96 rows and zero duplicate `(sample_key, trajectory)` keys. Sufficiency collection is active: `raw/sufficiency_partial.parquet` has 68 unique rows across all seven predeclared conditions (9 complete pairs plus 5 rows of the tenth pair; log progress is `sufficiency 7/32` at the check boundary). No error traceback, finalized summaries, Phase-S verdict, reset, or report exists.
- Do not interpret the partial sufficiency rows; leave the single durable runner active until collection reaches its existing completeness gate.

## 2026-08-14 CST: Treatment-Specific Text-State Coalition — Collection and Offline Verdict Completed

- User requested another progress check. The durable runner exited normally after `trajectory 32/32` and `sufficiency 32/32`; no traceback or concurrent runner remained. Final raw coverage is 96 trajectory rows (32 each CLEAN_SOURCE/CLEAN_TARGET/VISUAL_TREATMENT) and 224 sufficiency rows (32 each across NOOP, SUBJECT, REFERENCE, TASK_END, SUBJECT_REFERENCE, SEMANTIC_COALITION, ALL_NONPRED_TEXT), duplicate-free.
- Existing offline summary completed under the frozen protocol. V2D family-weighted baseline RR is `0.118906` (95% CI `[0.101616,0.137720]`). Text conditions have mean RR: SUBJECT `.002082` (CI `[-.001942,.006495]`), REFERENCE `.000208` (`[-.003027,.003239]`), TASK_END `-.001281` (`[-.005414,.002246]`), SUBJECT_REFERENCE `.000208` (`[-.002844,.003232]`), SEMANTIC_COALITION `-.000140` (`[-.004718,.003891]`), ALL_NONPRED_TEXT `-.000012` (`[-.004486,.004004]`); NOOP is exactly zero.
- Coalition contrasts are non-positive: `J_max=-.004310` (95% CI `[-.007561,-.001543]`, 1/12 positive families) and `S_ABC=-.001149` (CI `[-.008244,.005882]`, 5/12). Phase-S verdict is `S-C` (no all-text capacity), so Phase R reset/blockade did not run. Final pilot verdict is `T-B`: no sharp cooperative text-state configuration beyond ordinary distributed routing. Final artifacts are under `outputs/qwen35_v2d_text_coalition_pilot/{raw,summaries,report.md}`; no layer/region/factorial/norm-matching follow-up was run.

## 2026-08-14 CST: Text-Coalition Condition-Level Effect Read-Only Audit

- User requested concrete treatment effects. Read-only recomputation from final raw/summaries confirmed the primary family-weighted values: NOOP RR/CE `0/0`; SUBJECT `.002082/.015625`; REFERENCE `.000208/.002604`; TASK_END `-.001281/-.007812`; SUBJECT_REFERENCE `.000208/.002604`; SEMANTIC_COALITION `-.000140/~0`; ALL_NONPRED_TEXT `-.000012/.002604`. All 32 rows per condition retained the target prediction and none copied the source answer; NOOP full-vocabulary error was exactly `0`.
- The audit did not change any artifact, protocol, or conclusion. Family-bootstrap CIs, token counts, and raw treatment norms remain in `summaries/sufficiency_condition_effects.parquet`.

## 2026-08-14 CST: V2D Critical Region, V2P PRED, and Non-PRED Text Integrated Read-Only Analysis

- User requested a comprehensive comparison of the three causal layers. Read-only review combined the frozen V2D critical-region report/raw tables, completed V2P-Mediation summaries, and completed K4 non-PRED text-coalition summaries; no model forward, artifact, protocol, or result change was made.
- Integrated conclusion: the 8-row moved-object critical visual region has robust controlled causal portability at DEC_H0/DEC_STAGE1; K4 PRED has positive treatment/site/blockade mediation contrasts under activation intervention; selected non-PRED text token groups at K4 have near-zero RR and no coalition advantage. The evidence supports a critical visual evidence -> contextual computation -> late PRED commitment path, not a proven non-PRED token coalition or unique natural PRED mediator.
- Primary numbers recorded for the analysis: V2D critical raw changed-row CE/RR are DEC_H0 `.894531/.115988` and DEC_STAGE1 `.433594/.057826`; family-bootstrap critical RR are `.118906` (CI `[.101481,.137870]`) and `.060781` (CI `[.047992,.075591]`), with all 12 families positive. V2P K4 DEC_H0 M1/M2/M3 are `.348633/.520833/.447917`, CIs all strictly positive; DEC_STAGE1 are `.156413/.276042/.294271`, also all positive. Non-PRED family-weighted RR are SUBJECT `.002082`, REFERENCE `.000208`, TASK_END `-.001281`, SUBJECT_REFERENCE `.000208`, SEMANTIC_COALITION `-.000140`, ALL_NONPRED_TEXT `-.000012`; Phase-S `S-C`, final `T-B`.
- Caveats retained: V2D/PRED/non-PRED CE/RR and contrasts have different intervention definitions and denominators; V2P SF/BF are descriptive, not information fractions. V2P invariant controls have zero erroneous flips but continuous drift; non-PRED used only the frozen H0/K4 target-context write with no layer sweep, norm matching, random coalition, or Phase-R follow-up. Single-model/single-template/family-cluster generalization remains limited.

## 2026-08-14 CST: V2D versus V2P Patch/Measurement Position Clarification

- User asked whether V2D uses pre-decision PRED data and whether V2P computes a post-patch PRED at stage 4. Read-only code check: V2D patches decoder visual rows only at DEC_H0 or DEC_STAGE1, then runs the remaining model forward and reads final candidate logits at the PRED position; it does not transplant a PRED hidden state. V2D PRED is the readout location, not the intervention location.
- V2P first applies the V2D visual treatment at DEC_H0/DEC_STAGE1, captures treatment-derived PRED and TASK_END vectors at K3/K4/K5, then separately patches the selected vector into a clean target at K3/K4/K5. K4 is layers[23] post-block: the K4 PRED hidden vector is the intervention, while the final left/right logits are measured at the PRED position after layers 24–31 run. K3/K5 are additional onset/endpoint stages.

## 2026-08-14 CST: V2P M1/M2/M3 Formula Read-Only Clarification

- User asked how V2P mediation metrics are computed. Read-only code/data check: for each changed directed pair and boundary/stage, the signed PRED sufficiency effect is `SE_P = sign*(margin_patched - margin_target)`, where the patch is the treatment-derived PRED vector written into a clean target. Eight shuffled repeats are averaged within each directed row before family aggregation.
- M1 (`pred_treatment_specificity`) is `SE_P_CRITICAL - mean(SE_P_SHUFFLED, SE_P_BACKGROUND)`. M2 (`pred_site_specificity`) is `SE_P_CRITICAL - SE_TASK_CRITICAL`, where the latter writes the same critical treatment state at TASK_END. M3 (`block_site_specificity`) is `BE_P - BE_TASK`; `BE_P = critical_visual_effect - effect_after_clean-target-PRED reset`, and `BE_TASK` is the analogous TASK_END reset.
- At K4 family means, DEC_H0 expands numerically as `SE_P_CRITICAL=.523437`, `SE_P_SHUFFLED=.245443`, `SE_P_BACKGROUND=.104167`, control mean `.174805`, hence M1 `.348633`; `SE_TASK=.002604`, hence M2 `.520833`; `BE_P=.453125`, `BE_TASK=.005208`, hence M3 `.447917`. DEC_STAGE1 analogously has critical `.273437`, shuffle `.168945`, background `.065104`, control `.117025`, task `-.002604`, BE_P `.302083`, BE_TASK `.007812`, yielding M1 `.156413`, M2 `.276042`, M3 `.294271`. All are signed margin contrasts, not RR or information fractions; 12-family bootstrap (4,000 draws) is applied after family averaging.

## 2026-08-15 CST: V2P Margin Normalization Read-Only Clarification

- User asked whether the V2P values use softmax normalization. Read-only code check: `scores()` applies `log_softmax` to the full vocabulary, selects the left/right candidate log probabilities, and defines `margin = log p_left - log p_right`. Since the shared log-sum-exp term cancels, this equals the raw-logit difference exactly. CE, SE_P, BE_P, M1, M2, and M3 therefore are signed logit-margin effects/contrasts, not softmax probability differences or probability-normalized information fractions. RR additionally divides CE by the clean source-target margin D; bootstrap is applied after family aggregation.

## 2026-08-14 CST: V2P PRED Hidden-State Stage Semantics Read-Only Clarification

- User asked whether V2P-Mediation uses the stage-4 PRED hidden state. Read-only code/manifest check: yes, K4 is the preregistered primary mediation stage at decoder code index 23 (`model.language_model.layers[23]`, normalized depth .75), and `SUFF_*` PRED interventions patch the PRED position with the treatment-derived K4 state.
- It is not K4-only: the same PRED sufficiency and reset/blockade framework evaluates K3/code index 19 (secondary onset), K4/code index 23 (primary), and K5/code index 31 (final endpoint control). TASK_END sufficiency/reset is evaluated at K3/K4; K5 TASK_END is excluded. Treatment captures retain PRED and TASK_END vectors separately, and clean caches contain full sequence states.
- The PRED state is not a source-scene transplant: for each target trajectory, V2D visual treatment is applied first, then its PRED hidden vector at the selected stage is patched into a clean target forward. Verdict A names K4 as primary, with K3/K5 as temporal/endpoint controls.

## 2026-08-14 CST: Text-Token Coalition Stage Clarification

- User asked whether the partial text-token migration also uses stage 4. It does: the frozen coalition runner fixes K4/code index 23, post-full-block output, normalized depth .75, and writes the visual-treatment K4 state only at selected SUBJECT/REFERENCE/TASK_END text rows (or their coalitions). It does not sweep or use K3/K5.
- `DEC_H0 + CRITICAL_EXACT` is the upstream visual treatment boundary (8 image rows); it is distinct from the downstream K4 text-token write site. The text pilot excludes PRED and tests target-context treatment deltas, not source-scene PRED transplantation.

## 2026-08-15 CST: V2P Mediation Core-Conclusion Clarification

- User asked for the core conclusion of V2P Mediation. The frozen run reached Verdict A at both DEC_H0 and DEC_STAGE1: K4 PRED sufficiency was stronger than shuffled/background controls (M1), stronger than writing the same state at TASK_END (M2), and PRED reset/blockade removed more effect than TASK_END reset/blockade (M3). K4 family means were DEC_H0 M1/M2/M3 `.348633/.520833/.447917` and DEC_STAGE1 `.156413/.276042/.294271`, with all preregistered bootstrap CIs positive.
- Interpretation: under controlled activation interventions, critical visual evidence is causally handed off to the late PRED decision state, with K4 as the primary mediation site; K3 is onset context and K5 is endpoint control. This supports a visual evidence -> contextual computation -> PRED commitment path, but does not prove PRED is the unique natural mediator or that the effects are information fractions. The pilot used 32 changed directed pairs from 12 family clusters on one frozen model/template.

## 2026-08-15 CST: V2P Non-PRED Text-Coalition Core-Conclusion Clarification

- User asked for the core conclusion of the non-PRED follow-up. The frozen K4 text-state intervention writes within-target visual-treatment deltas (not source-scene text/PRED states) into SUBJECT, REFERENCE, TASK_END, their coalitions, or ALL_NONPRED_TEXT, after `DEC_H0 + CRITICAL_EXACT`. All conditions had near-zero RR, retained the target answer on all 32 directed rows, and copied no source answer; the V2D critical baseline RR was `.118906`.
- Family-weighted RR: SUBJECT `.002082`, REFERENCE `.000208`, TASK_END `-.001281`, SUBJECT_REFERENCE `.000208`, SEMANTIC_COALITION `-.000140`, ALL_NONPRED_TEXT `-.000012`. Coalition contrasts were non-positive (`J_max=-.004310`, `S_ABC=-.001149`), so Phase S was `S-C`, Phase R blockade did not run, and final verdict was `T-B`: no sharp cooperative non-PRED text-state configuration beyond ordinary distributed routing. This does not prove non-PRED tokens are causally irrelevant; it rules out the tested frozen K4 configuration under this protocol.

## 2026-08-15 CST: V2P K4 PRED RR Clarification

- User asked for K4 PRED RR. The primary V2P report stores signed PRED sufficiency effects rather than a primary RR; applying the same family-first `RR = effect / D` definition used by the non-PRED summary to `SUFF_CRITICAL` gives family-weighted K4 PRED RR `0.070066` at DEC_H0 and `0.037576` at DEC_STAGE1. Corresponding family-mean SE_P effects are `.523437` and `.273437`; row-wise RR means are `.069339` and `.035719`.
- These are distinct from descriptive SF (`SE_P/TE`), whose family means are `.613822` and `.584807`, and from the V2D total-effect RR baselines `.118906` and `.060781`. No single combined K4 RR is defined because DEC_H0 and DEC_STAGE1 are separate treatment boundaries.

## 2026-08-15 CST: V2P PRED/non-PRED RR Cross-Consistency Audit

- User questioned why PRED and non-PRED RR do not sum to the visual-transfer RR. Read-only parquet/code audit plus an authorized single-pair CUDA hook comparison found two separate issues. First, the sum is not a valid decomposition: PRED and non-PRED are separate marginal interventions from clean target, non-PRED excludes 196 image rows and control/special rows, and downstream transformer interactions are nonlinear. K4 state-delta norms illustrate the missing support: image mean L2 `104.68`, ALL_NONPRED_TEXT `9.97`, PRED `2.83`, combined PRED+text `10.40` versus total `105.30`.
- Second, there is a real implementation mismatch affecting cross-experiment comparability. `experiments/qwen35_v2p_mediation/model_runtime.py::_replace_delta` uses bf16 `base + (source-target)` even when `visual_permutation=None`; frozen V2D `patch_hidden` and the non-PRED `_forward` use direct bf16 source replacement for unpermuted critical visual rows. Same-model/same-pair comparison produced full-vocabulary max diff `.2890625` and margin CE `0.750000` (V2P delta-add) versus `.625000` (V2D/direct). Across the 32 DEC_H0 critical pairs, 14 had material CE differences (>1e-5), max `.25`, mean absolute `.058594`; D denominators and clean margins matched exactly.
- Therefore the RR formula/softmax calculation is not the cause, but the saved V2P K4 PRED RR and non-PRED RR are not a clean apples-to-apples decomposition. Existing V2P mediation remains internally reproducible under its own delta-add path, while the integrated PRED-vs-non-PRED quantitative comparison is provisional. Do not silently edit or rerun the frozen experiment; a separately authorized matched direct-replacement repair/recompute is required before using their RR sum or comparing mediation magnitudes.

## 2026-08-15 CST: Non-PRED Visual Boundary Clarification

- User asked whether non-PRED uses DEC_H0 or DEC_STAGE1. The frozen non-PRED text-coalition pilot uses only `DEC_H0 + CRITICAL_EXACT` upstream (8 critical decoder image rows); it does not run DEC_STAGE1. Its separate downstream write site is K4/stage_4 layer 23 post-block, where the within-target visual-treatment text-state delta is written to selected non-PRED text roles.

## 2026-08-15 CST: Matched Direct-Replacement Repair And Joint Text Extension In Progress

- User authorized a matched direct-replacement repair and an added PRED/non-PRED joint replacement comparison. Changed `experiments/qwen35_v2p_mediation/common.py`, `model_runtime.py`, `mediation_arbiter.py` inputs, `summarize_mediation.py`, `experiments/qwen35_v2d_text_coalition_pilot/run.py`, and added `matched_direct_repair.py` plus `matched_joint_extension.py`; old output roots remain untouched. Direct unpermuted visual writes now assign the source hidden rows directly; explicitly shuffled rows retain the frozen delta-add path. Added S7 `PRED_ONLY` and S8 `PRED_PLUS_ALL_NONPRED_TEXT` while retaining all prior conditions.
- Protocol tests and py_compile checks pass. New matched V2P arbiter at `outputs/qwen35_v2p_mediation_matched_direct/arbiter/` passes all gates: direct replacement, NOOP/source=target, token alignment, endpoint reproduction, and fixed-row total-effect reproduction. Fixed-row family RR is DEC_H0 `.116667` versus old `.118971` (old CI `[.101481,.137870]`), DEC_STAGE1 `.041667` versus old `.060866` (old CI `[.047992,.075591]`); comparisons pass the preregistered tolerance gate.
- Full matched repair is running persistently in batch1-only mode via `matched_direct_repair.py`; it reuses verified clean/unaffected rows and recomputes only direct-sensitive CRITICAL/BACKGROUND, PRED sufficiency, TASK sufficiency, reset, and invariant-critical rows. At this entry it has completed 7/96 pairs (14/192 boundaries); no final mediation result is authorized yet. A watcher will run the offline V2P summarizer after checkpoint completion and then run the two-new-condition text extension. Do not interpret partial repair rows.

## 2026-08-15 CST: Matched Repair Progress Audit

- User requested a read-only progress check. The persistent `v2p_matched_repair_fast` session is still active on GPU PID 446803; checkpoint is `running`, `42/192` pair-boundaries (`21/96` pairs), all in the changed subset, with no invariant boundary started. Direct-sensitive partial rows are base `84`, sufficiency `336`, reset `210`, invariant `0`, matching the completed-boundary write pattern; no final mediation parquet/summary exists.
- The `v2p_joint_followup` watcher remains idle and will not run the offline summarizer or PRED/non-PRED joint extension until the repair checkpoint becomes `complete`. Partial rows remain non-interpretable.

## 2026-08-15 CST: Matched Repair Progress Audit 2

- Read-only check at 14:59 CST: durable repair checkpoint remains `status=running`, `60/192` pair-boundaries (`30/96` pairs), all still in the 32-pair changed subset. Partial parquet writes contain one additional complete boundary (`61` unique boundaries: base `122`, sufficiency `488`, reset `305`, invariant `0`) but the checkpoint/log did not advance after that write, so resume logic must reconcile it; no final merged raw tables, summary, or report exists.
- No repair process was visible in the current PID namespace after the last write at 14:58:39; do not assume completion. The joint PRED/non-PRED extension has not started: no `matched_joint_partial.parquet`, S7/S8 rows, or extension manifest/report. Existing text-coalition S0-S6 outputs are prior completed artifacts only.

## 2026-08-15 CST: Matched Repair Progress Audit 3

- Read-only check at 15:32 CST: repair checkpoint advanced to `76/192` boundaries (`38/96` pairs), with all 64 changed boundaries complete and 12 invariant boundaries collected (6 invariant pairs). Partial rows are base `152`, sufficiency `548`, reset `380`, invariant `12`; execution remains batch1-only and checkpoint status is still `running`.
- Last durable write/log line is 15:30:47 (`matched repair 38/96`); no process is visible in the current PID namespace and no further write occurred during a 25-second observation. No merged final raw tables, summary, report, or joint S7/S8 extension artifacts exist; partial rows remain non-interpretable.

## 2026-08-15 CST: Matched Repair Process Check

- Read-only process check at 17:08 CST: no `matched_direct_repair.py`, joint-extension, summarizer, or known PID 446803 process is visible. Checkpoint nevertheless advanced to `126/192` boundaries (`63/96` pairs) with status still `running`; last checkpoint/log write was 17:07:04 (`matched repair 63/96`).
- Partial coverage is now all 64 changed boundaries plus 62 invariant boundaries (31 invariant pairs): base `252`, sufficiency `698`, reset `630`, invariant `62`. No final merged outputs or S7/S8 joint-extension artifacts exist; remaining work is 66 invariant boundaries, and partial rows remain non-interpretable.

## 2026-08-15 CST: Matched Repair Completed; Postprocessing and Joint Extension Status

- Read-only check at 20:46 CST: matched repair is complete (`192/192` boundaries, checkpoint `status=complete`, merge manifest written at 19:27:18). Final merged raw tables exist: base `1088`, reset `960`, invariant `256`, sufficiency `2432`.
- The automatic `summarize_mediation.py` then failed its completeness gate: expected sufficiency `2688`, found `2432` (missing `256` invariant sufficiency rows); no matched-repair mediation summary/report/verdict was produced. Do not interpret the matched raw output as final mediation statistics until this completeness issue is addressed under an explicit repair.
- The authorized joint extension completed separately at `outputs/qwen35_v2d_text_coalition_pilot_matched_joint/`: 32 pairs, 64 new S7/S8 rows, merged 288-row S0-S8 table, report and summaries present. Its frozen verdict is `S-C` / `T-B`; PRED-only, non-PRED-only, and joint comparison artifacts are complete.

## 2026-08-15 CST: Remaining Work Quantified

- Read-only comparison of old versus matched sufficiency keys found the exact missing set: `256 = 64 invariant pairs × 2 boundaries × 2 stages (K3,K4)` rows, all `SUFF_TASK_CRITICAL`; invariant `SUFF_CRITICAL` and K5 rows are present. No changed rows are missing.
- Thus model execution is 100% complete and the joint extension is 100% complete; only the matched mediation raw-completeness repair/remerge and offline summarizer remain. No new CUDA forwards should be needed if the frozen invariant task-sufficiency rows are authorized for reuse, but no such repair was performed in this audit.

## 2026-08-15 CST: Status Check After Remaining-Work Audit

- Read-only check at 23:14 CST: no repair, summarizer, or joint-extension process is running. Matched checkpoint remains complete `192/192`; raw sufficiency remains `2432/2688` with the same missing `256` invariant `SUFF_TASK_CRITICAL` K3/K4 rows. The only summarizer attempt remains the 19:27 completeness-gate failure; no new report or summary was produced.

## 2026-08-15 CST: Three-Object Relation Probe vs Orthogonal-Complement Retrieval

- User asked to locate the three-object relation probe-decoding and orthogonal-complement comparison. Read-only audit; no model run, output mutation, or protocol change.
- Located the representation probes under `outputs/clevr_spatial_content_pilot40/` and the held-out O difference decoder under `outputs/qwen35_o2y_pairwise_validation/`; the latter is the decoder reused by the B/O causal interchange, not an independent relation-label probe.
- Located the corrected full-rowspace causal comparison under `outputs/qwen35_bo_subspace_interchange/corrected_causal/` and the final intrinsic-4D rerun under `outputs/qwen35_bo_intrinsic4_causal/`. Intrinsic final means: BO RR `.00100454`, decoder-orthogonal RR `.0171461`, FULL RR `.0342963`, BO-minus-orthogonal `-.0161415` (paired bootstrap 95% CI `[-.0206336,-.0121526]`), Outcome B.
- Relation probe random-split claims are confounded by absolute camera-right coordinate (`audit_relation_coordinate_confound`, verdict `invalid_for_relational_claim`); Phase-2 projection enrichment is a matched-rank random-subspace descriptive control, not the exact orthogonal-complement causal intervention.

## 2026-08-15 CST: Difference-Decoding Audit Clarification

- User clarified that the requested audit concerns source-target O/rank difference decoding, not the absolute Q3 relation probe. The camera-right shortcut audit does not directly invalidate the difference decoder.
- Difference labels are exact 4D, although nominal O is 9D; the frozen 9D Ridge has small nonzero redundant singular directions and unconstrained output components. The prior hard-coded rank-4 projector therefore truncated nonzero decoder directions and was marked `IMPLEMENTATION_INVALID_PROJECTOR_TRUNCATION`; those causal outcomes are quarantined.
- The repaired complete-rowspace and final intrinsic-4D audits passed. Held-out difference decoding remains strong at OBJ_BUNDLE (stage_2/3/4 R2 `.989532/.989749/.985478`), with caveats that pair rows are dependent, test is one held-out geometry template, and low-alpha Ridge fits issued small-n/high-dimensional conditioning warnings.

## 2026-08-15 CST: Why Intrinsic BO RR Is Below Orthogonal RR

- User asked why the final intrinsic causal effect is much lower in BO than in the decoder-orthogonal residual. Read-only code/report audit confirmed the write is `full = source-target`, `BO = Q(Q^T full)`, `ORTH = full-BO`; projector/runtime gates passed.
- The natural difference is not norm matched: mean BO L2 norm `2.168835` versus ORTH `3.580601`; BO is `.516791` of full norm and `.268019` of squared energy, ORTH `.731981` of squared energy. BO+ORTH also has positive downstream non-additivity: interaction RR `.0161457` (FULL RR `.0342963` vs BO+ORTH `.0181506`).
- Interpretation recorded: decoder-orthogonal means null to the frozen linear O readout only, not null to the model's downstream nonlinear/Jacobian pathways. The result limits disproportionate causal privilege of the probe-readable component; it does not show that ORTH is noise or that the probe is incorrect. Norm matching was not authorized after Outcome B.

## 2026-08-15 23:19 CST: Matched V2P Sufficiency Completion And Final Summary

- User authorized completing the matched direct-replacement repair, offline summarization, and live process monitoring. Added `experiments/qwen35_v2p_mediation/complete_matched_sufficiency.py`; no model forward was run and the original raw V2P output rows were not changed.
- Exact-key audit confirmed the only gap was 256 frozen invariant `SUFF_TASK_CRITICAL` rows (`64` invariant pairs × `K3/K4` × `DEC_H0/DEC_STAGE1`), all `batch1_only`. Reused those rows from the frozen old sufficiency table into both matched raw locations; matched sufficiency is now `2688/2688`. Merge manifest records the post-merge reuse and `model_forward=false`.
- Ran `QWEN35_V2P_OUT_ROOT=outputs/qwen35_v2p_mediation_matched_direct ... summarize_mediation.py`; the process exited 0 and generated `report.md`, `summaries/verdict.json`, bootstrap/contrast tables, and figures. Completeness passed: base `1088`, sufficiency `2688`, reset `960`, invariant controls `256`; verdict `A`, with strong mediation at both DEC_H0 and DEC_STAGE1 and K5 endpoint control passed.
- K4 family-bootstrap means (95% CI): DEC_H0 M1 `.365560` `[.292794,.445475]`, M2 `.549479` `[.468750,.640625]`, M3 `.468750` `[.411458,.531250]`; DEC_STAGE1 M1 `.179850` `[.145671,.213216]`, M2 `.294271` `[.242188,.348958]`, M3 `.278646` `[.223958,.335937]`. All 12/12 families were positive for each K4 contrast at both boundaries.
- The first post-repair summary invocation used an obsolete environment-variable name and regenerated only the default-root summary from unchanged old raw rows; the corrected invocation targeted the matched root and produced the final artifacts above. No mediation or model raw data were affected. No matching process remains running. Joint S7/S8 extension remains complete with its prior `S-C`/`T-B` result.

## 2026-08-15 23:24 CST: Detailed Matched V2P And Joint Text Report

- Read-only detailed audit used only the completed matched roots. The margin scorer applies `log_softmax` then uses `log p(left)-log p(right)`; normalization cancels in the margin, so `effect`, M1/M2/M3, and RR are margin-based intervention quantities, not probability fractions or cross-entropy losses.
- Matched K4 PRED family-first RR is `.072431` (DEC_H0) and `.039553` (DEC_STAGE1), versus total visual-treatment RR `.118906`/`.060781`; K4 TASK critical RR is `-.001281`/`-.000808`. K4 family-bootstrap M1/M2/M3 are DEC_H0 `.365560/.549479/.468750` and DEC_STAGE1 `.179850/.294271/.278646`, with all 12/12 families positive and 95% CIs strictly above zero.
- Temporal row means show PRED sufficiency at K3/K4/K5 of `.484375/.558594/.894531` (DEC_H0) and `.257813/.285156/.433594` (DEC_STAGE1); K5 is endpoint localization only and is not counted as mediation timing. K4 PRED rows had no source-answer copies and retained the target answer on all changed rows.
- Joint S7/S8 comparison at K4 + DEC_H0: PRED_ONLY RR `.072431` CI `[.062200,.082944]`; ALL_NONPRED_TEXT `-.0000119` CI `[-.004383,.004111]`; joint `.073909` CI `[.064461,.083058]`; joint-minus-best `.001478` CI crosses zero; joint-minus-sum `.001490` CI crosses zero. The separate coalition verdict remains S-C/T-B: no sharp cooperative non-PRED configuration under the frozen target-context delta protocol.
- Detailed interpretation: corrected V2P supports intervention-based PRED site/treatment specificity at both frozen boundaries, while the tested non-PRED text deltas add no reliable causal effect or synergy. It does not establish a unique natural mediator, an information decomposition, or irrelevance of all non-PRED tokens.

## 2026-08-16 CST: State-Source Audit Clarification

- User asked whether the completed experiments patch full source text states or critical-treatment-derived text states. Read-only code/output audit confirms V2P PRED/TASK sufficiency uses `treatment[condition]["states"][stage]`, captured from a target-row forward after source visual hidden rows are written only into the frozen critical visual region; it never patches `src["states"][stage]` into the target PRED/TASK position.
- The matched V2P chain is therefore: source visual critical rows -> target-context CRITICAL/BACKGROUND forward -> treatment-derived PRED or TASK_END state -> clean-target state overwrite. Source full text states are used only indirectly for the visual donor and source-margin D/reference, not as text-state donors.
- The non-PRED S0-S8/joint extension likewise loads `treatment_k4` and copies selected positions from that target-context critical-treatment state to clean target K4; PRED_ONLY copies only PRED, while PRED_PLUS_ALL_NONPRED_TEXT copies PRED plus selected non-PRED text rows. It is not a full-source text-state transplant.

## 2026-08-20 CST: Position-ID Versus Ordinary Left/Right Semantics Experiment

- User requested an independent, minimal Qwen3.5 experiment testing only whether a pure-text left/right semantic direction can substitute for the visual paired “position ID” direction in selecting the other object's color. Added `experiments/position_id_semantic_test/`, `tests/position_id_semantic_test/`, and independent outputs under `outputs/position_id_semantic_test/`; no existing experiment or result was modified, no model was downloaded, and no Q/K, relation-operator, COCO, repair, training, or follow-up mechanism experiment was run.
- Generated 192 exact paired synthetic families / 768 trials with family-level 96/32/64 train/validation/test splits and disjoint geometry templates. All balance, prompt-ban, mirror-pair, and text-`OK` audits passed. The 16-family pilot passed PRED, batch1 scorer, NOOP, alpha=0, and clean gates; formal clean full-sequence and unconstrained exact-color accuracy were both `768/768`, including LEFT/RIGHT `384/384`. PRED is index 108 after the official empty closed think block; all 32 post-block states were saved.
- Train paired `h_RIGHT-h_LEFT` and validation-only selection fixed L17: validation AUC `1.0`, balanced accuracy `1.0`, Cohen d `25.3525`, paired d_z `16.4854`. Independently estimated text directions at L17 gave cos(ID,LR) `.305589`, cos(ID,ORDER) `.121024`, cos(ID,TB) `-.016772`; textual LR separated held-out visual side with AUC/balanced accuracy `1.0/1.0`, Cohen d `22.0233`.
- Validation ID-only sweep fixed alpha `2.0` (mean delta margin `7.81567`, family-bootstrap 95% CI `[7.29926,8.31821]`). Exact cached multimodal-input optimization passed two 4-row bitwise full-vocabulary equivalence gates (max error 0) and retained independent decoder batch1 execution. Formal causal coverage is complete: 6144 unique rows = 256 clean-correct test trials x (ID/LR/ORDER/TB + 20 norm-matched random), all 64 families and LEFT/RIGHT `128/128`.
- Test causal means: ID delta margin `7.70947` (95% CI `[7.38400,8.01686]`), opposite-object greedy flip `.375`; LR `2.63989` (`[2.44553,2.82571]`), flip `0`; ORDER `1.17993`, TB `.788574`, 20-random trial-mean `.061011` with CI crossing zero, all flip `0`. `RE_LR=.342422`, family-bootstrap 95% CI `[.324121,.361824]`. Corrected report retention is unconstrained greedy retention: ID `.625`, LR/controls `1.0`; restricted-candidate retention remains a separate secondary field.
- Verdict is preregistered Pattern D / mixed representation: ordinary LR supplies a real, LR-specific margin-level component and is closer than ordinal semantics, but at the common validation-fixed intervention it does not causally reproduce the decisive opposite-object greedy color switch. Thus the experiment weakens a wholly semantics-independent ID account but does not support the stronger claim that textual LR is sufficient to substitute for visual object selection. Final artifacts include `summary.json`, `REPORT.md`, all required manifests/CSVs/NPZs, and exactly three core figures. Five protocol unit tests and py_compile pass. Stop here; do not add subspace decomposition, alternate alpha/site/layer search, or new mechanism experiments without a new protocol.

## 2026-08-20 CST: Position-ID Factorial Physical-Position vs Enumeration Pilot

- User requested a synthetic, strictly factorial test of physical position `S`, enumeration slot `E`, and object identity `O`, with faithful Position-ID extraction before cross-slot intervention. Added `experiments/position_id_factorial/` and independent outputs under `outputs/position_id_factorial/`; no old PRED-side-difference experiment, COCO, Q/K circuit, relation operator, repair, or model training was used.
- Paper audit recorded the residual object-word Position-ID definition from *Linear Mechanisms* (arXiv:2601.12626) and the distinct residual/head-level definitions from *Visual Symbolic Mechanisms* (arXiv:2506.15871). Qwen3.5 has no faithful paper-2 CMA head map here, so only its residual selection/comma diagnostics are saved; no head-level claim is made.
- Dataset and tokenizer audits passed. The 8-family pilot passed the faithful gate: clean accuracy `1.0`, validation/audit side projection `1.0`, selected layer `21`, alpha `5`, ID margin effect `11.8950` versus norm-matched random `9.6026`, ID switch `.625` versus random `.275`, alpha-zero max score difference `0`.
- The 16-family causal pilot completed batch1-only on 4 held-out test families/16 factorial rows. Same-slot margin effect `7.8438` (95% family bootstrap `[7.6942,7.9933]`), cross-slot `7.2057` (`[7.0304,7.3810]`), `CTR=.9187`; actual opposite-object switch `.625` same-slot versus `.375` cross-slot. Enumeration switch `.1875`; random switch `.2083`; object-direction `.1875`. Cross-object exploratory `COR=.9163` covers only 12 rows because one unseen-shape source prototype was not imputed. Attribute switch was `0` for all tested interventions.
- Correct first-token plausibility audit passed on validation/test: both forced first-object prefixes above `-12` on `16/16` rows in each split; mean first-token logprob `-4.3037` validation and `-5.1320` test. Natural generation was kept separate (`20/32` valid two-shape orders). Raw arrays are six `[64,32,2560]` activation sites with complete traceable index rows; 10,000 family-bootstrap intervals and four core figures are present.
- Verdict: `MIXED/INCONCLUSIVE`. Margin transfer is close across slots and stronger than the tested enumeration/random controls, but decisive full-object switch and attribute follow-through are not clean enough for a single-factor H-POS conclusion. The predeclared formal `128/32/64` run was not started; no formal result should be inferred from this pilot.

## 2026-08-20 CST: Position-ID Free-Generation Correction Pilot

- User requested only two corrections to `position_id_factorial`: alpha dose-response with fair norm-matched random controls, and true free second-object generation plus conditional post-switch attribute querying. Added `experiments/position_id_factorial/free_generation.py`; extended `protocol.py`/`dataset.py`; outputs are independent under `outputs/position_id_factorial/free_generation_pilot/`. The frozen Linear-Mechanisms object-word residual Position ID at L21 and old pilot outputs were preserved; no PRED-side difference, Q/K, circuit, COCO, training, or repair was used.
- Prepared 24 natural families / 48 mirror images. Clean eligibility was 28/48 (`LR=28`, `RL=0`), so natural enumeration has a strong LR bias and no RL causal estimate is identifiable. `natural_order_bias.json` now records identity, shape, color, size, x-coordinate, and first/second shape-token confidence; eligibility was clean-only and pre-treatment.
- Natural gate causal subset was fixed before intervention at 8 eligible LR rows, 280 rows total, alpha `{0,.25,.5,1,2,3,5}`, conditions ID same-slot/cross-slot/random/enumeration/no-op, one random direction per sample. Cross-slot free opposite-object switch was 0 at every alpha; random was also 0. Low/mid gate A failed, random corruption gate B passed (`.05` in the gate subset), manual/parser gate C passed (`20` examples); formal 500-family pool was not authorized or run.
- Controlled-prefix secondary completed 16/16 clean rows and 560 rows with the same alpha grid. Cross-slot ID, random, and enumeration free switch rates were 0 at all alpha. Same-slot had one isolated 1/16 switch at alpha=2, a duplicated current noun (`triangle red, triangle red,`), not stable position transfer. One conditional attribute query matched the new color (`1/1`) after that switch; natural had zero actual switches, so no natural attribute estimate exists.
- Raw margin fields were added as secondary diagnostics. At natural alpha=5, mean cross-ID margin effect was `6.45` versus random `6.26`, while free object switch remained `0`; this confirms raw margin specificity is insufficient at the paper-faithful high alpha. All refreshed primary classifications exactly matched the pre-margin snapshots. Alpha=0/no-op and actual injected norm ratio gates passed; 10,000 family-bootstrap draws are recorded.
- Required manifests/CSVs, `summary.json`, `REPORT.md`, and five figures are complete. The formal run remains stopped by the predeclared gate; interpret this as a pilot null for cross-slot free object switching, not a universal negative claim about Position ID.

## 2026-08-21 CST: Latent Binding Interface MVP Implementation

- User requested an isolated VLM binding-repair MVP for Qwen3.5-4B: strictly controlled synthetic 2–5-object images, M0 frozen baseline, M1 ordinary LoRA SFT, M2 query-conditioned Latent Binding Interface with target-attention and counterfactual identity-consistency losses, plus binding/selectivity/error/causal evaluation. No COCO or existing experiment output was used or modified.
- Added `experiments/lbi_binding/` with `dataset/` (448×448 renderer, exact 14×14 merged-token bboxes, object metadata, shared-color/shared-shape/shared-feature and multi-object challenges, counterfactual relocations, processor collation), `model/` (dependency-free LoRA and LBI cross-attention wrapper), `training/`, `evaluation/` (candidate scorer, attention overlays, error categories, M0/M1/M2 comparison, BIND-state replacement), and README. Added smoke tests under `tests/lbi_binding/`.
- Static compile, dataset/metadata audits, processor alignment audit, and manual unit tests passed. Processor produced 196 visual tokens; every target footprint is exactly four tokens. Real external CUDA Qwen3.5-4B forward passed: M2 logits `[1,228,248320]`, attention `[1,196]`, attention sum `1.0`, finite LM/binding losses. One-step real CUDA M1 and M2 training both completed with trainable-only checkpoints; one-row M0/M1/M2 scorer/evaluation smoke and one causal replacement pair completed. Smoke causal donor-follow rate was `0.0` after only one training step and is implementation-only, not scientific evidence.
- Full dataset generation, multi-epoch training, full test evaluation, lambda ablation, and any M2>M1>M0 scientific conclusion were not run in this implementation turn. MVP intentionally uses batch size 1 for auditable visual-token alignment; run the README commands in a durable CUDA session before interpreting results.

## 2026-08-22 CST: LBI Validation Phase — Main Results And Explicit Stop

- User requested strict validation of the existing LBI MVP without changing the core architecture: full M0/M1/M2 comparison, binding errors, target/distractor selectivity, query sensitivity, counterfactual consistency, causal z replacement, and lambda/architecture ablations. Added `experiments/lbi_binding/validation/{run.py,features.py,aggregate.py}`; extended only explicit ablation flags (`disable_query_summary`, `visual_pooling`) and prompt-next-token scoring/answer-boundary LM projection. Outputs are under `outputs/lbi_validation/`.
- Fixed protocol generated 32 scene-disjoint scenes / 128 rows with train/validation/test `16/8/8`, exact 448×448 / 14×14 merged grid, and test coverage of `same_color`, `same_shape`, and `shared_feature` plus multi-object scene metadata. Main M1/M2 each completed one full 64-row epoch at seed `20260821`; M1 train/val LM `0.4502/0.6843`, M2 train total `2.4502` (LM `0.3804`, bind `2.0695`, inv `0.0031`), val LM/bind `0.5053/1.4755`, target mass `0.2890`.
- Held-out test has 32 rows: M0 accuracy `.96875`, M1 `1.0`, M2 `1.0`; shared-feature accuracy is `1.0/1.0/1.0`, so the desired M2>M1 accuracy ordering is not shown on this small split. Observable BER is M0 `.03125`, M1/M2 `0`; true two-attribute mixed binding is not identifiable under the MVP one-word answer contract and is explicitly marked in CSV/report.
- M2 attention mean target mass `.39483`, max distractor mass `.46271`, selectivity `-.06788`; shared-feature selectivity `.00041`. M0/M1 attention is `not_applicable`, not zero. Query sensitivity over 8 test images: M2 `1-cos=3.64e-6`, random-query control `.00375`, vision-average approximately `0`, so this checkpoint does not support query-conditioned z variation. Counterfactual test has 17 rows: z_bind cosine `.99733`, ordinary decoder-boundary hidden cosine `.93466`.
- Causal replacement found only 3 exact same-question/different-answer held-out pairs; actual answer flip and donor-follow rates are both `0.0`, mean replacement z cosine `.99030`. This is a null for causal control in this run, not evidence for H4.
- Added required `main_results.csv`, `binding_error_analysis.csv`, `attention_selectivity.csv`, `query_sensitivity.csv`, `counterfactual_binding.json`, `causal_replacement.csv`, figures (attention heatmap, BER, selectivity, causal, training curve), checkpoints, logs, and `validation_report.md`. Added `validation/sweep.py` to keep one Qwen process resident and complete all 16 ablations: the full 4×3 lambda grid plus four architecture ablations. Every short configuration used 2 optimization steps and 8 test rows, reached accuracy `1.0`, and had selectivity from `0` to `-.1133`; these are exploratory and not comparable to the full one-epoch main protocol. The refreshed report explicitly records that M1/M2 tie at the accuracy ceiling, M2 query sensitivity is near zero, and causal answer flips are `0/3`; no scientific hypothesis conclusion or benchmark expansion was authorized.

## 2026-08-22 16:45 CST: OCBW OBSB Workspace Validation — Null Mechanistic Pilot

- User requested the independent Object-Centric Binding Workspace (OCBW) repair experiment extending LBI: fixed `N=8` latent slots, Hungarian slot matching, query-conditioned selection, identity/selection/relation objectives, M0/M1/M2/M3 comparison, object-count scaling, identity consistency, and causal selected-state replacement. Added `experiments/ocbw_binding/` with OBSB generator/loader, OCBW model/runtime, training, evaluation, causal/identity analysis, aggregation, and figures. Existing `experiments/lbi_binding/` and its outputs were not modified.
- Generated `outputs/ocbw_validation/dataset/`: 20 scene-disjoint scenes, 130 rows (`train/validation/test = 52/26/52`), 2/4/6/8 objects, levels 1–5, all 130 counterfactual rows, 448×448 images with exact 14×14 merged-token alignment, and manifest SHA256 `85bd8f4386bb28465faafae835d34e555e3d528f552d301b32c3d04168edc081`. Trained one seed `20260822` for one epoch each on Qwen3.5-4B bf16 CUDA: M1 LoRA, M2 single-token LBI, and M3 OCBW; base vision/LLM weights remained frozen.
- Formal test evaluation completed all M0/M1/M2/M3 at 52 rows. Exact answer accuracy was `1.0` for every model and every object-count/task group, so this small protocol is at an accuracy ceiling and does not show `M3>M2>M1>M0`. M3 object retrieval was only `.2692` overall and fell `0.50/.4286/.2143/0.0` at 2/4/6/8 objects; M3 relation-head accuracy was `.2737` over 1,392 ordered pairs. M0/M1/M2 retrieval/relation-head fields are not applicable, not zeros. Binding error rate was `0` for all models because no answer errors occurred.
- M3 counterfactual identity evaluation over 52 test rows gave target-slot cosine `.9895` and mean matched-object cosine `.9891`; this indicates position-stable slot states but does not establish correct query selection. Causal replacement over 16 same-question/different-answer pairs (8 attribute-collision and 8 cross-attribute) gave clean accuracy `1.0`, donor-follow `0.0`, actual answer flip `0.0`, mean z cosine `.9708`, and mean donor-minus-source target-answer logit `-14.13`.
- Added heterogeneous right-padding in `OCBWModel.score_candidate_batch` so multi-token candidate scoring retains original prediction positions under a masked decoder batch. Static compile/diff checks passed; the post-fix causal run completed cross-attribute candidates without error. The saved full M3 evaluation was already exact teacher-forced scoring but its pre-fix spatial-relation rows used the slower per-candidate fallback; this changes efficiency only, not the score definition.
- Final artifacts: `outputs/ocbw_validation/main_results.csv`, `binding_error_analysis.csv`, `object_retrieval.csv`, `relation_decoding.csv`, `identity/identity_consistency.{csv,json}`, `causal/causal_replacement.{csv,jsonl}`, `figures/`, and `validation_report.md`. Conclusion: this one-seed, small OBSB pilot does not support an OCBW performance or causal-binding claim; the workspace states are highly position-consistent but not reliably object-selective or causally answer-controlling. Follow-up should first remove the accuracy ceiling/increase scene diversity and validate selection supervision before any larger benchmark.

## 2026-08-22 21:53 CST: Native Binding Failure Localization — Behavioral Calibration

- User requested a new, mechanism-agnostic native binding-failure dataset and frozen Qwen3.5-4B behavioral calibration; no LBI/OCBW/LoRA training, probes, latent modules, or causal patching were run. Added `experiments/native_binding_failure/` and independent outputs under `outputs/native_binding_failure/`.
- Generated 180 geometry families / 720 scene variants: N=`4,6,8` × I=`0,1,2,3`, 60 matched families per cell, exact 448×448 rendering with 14×14/196-token processor alignment. Pair pixel audits passed for all 180 families; 20-family contact sheet/manual audit passed; full processor audit passed `720/720`.
- Full CUDA bf16 batch1 greedy free-generation run completed `720/720` main rows, `4320/4320` primitive-control rows, and `5040/5040` raw generations. Cell coverage is exactly 60 rows for every N/I cell; no restricted candidate scoring was used as the primary metric.
- Overall strict/semantic bundle accuracy=`.8750`; errors: correct `630`, wrong-object `83`, hallucinated `7`, invalid `0`, mixed/illusory `0`. Object-count scaling is clear: N4/N6/N8 accuracy `.9583/.8542/.8125`, wrong-object rate `.0417/.1333/.1708`. Full N×I accuracy: N4 `.9667/.9500/.9667/.9500`; N6 `.8333/.8833/.8500/.8500`; N8 `.8000/.8000/.8333/.8167`.
- The preregistered interference hypothesis is not supported in this run: I0→I3 paired deltas are `-.0167,+.0167,+.0167`; matched transitions are 11 high-correct/low-wrong versus 10 low-correct/high-wrong. The dominant confound is relation direction: left/right accuracy `.9944/.9722`, above/below `.7611/.7722`.
- Primitive controls: target feature presence `1.0000`, reference recognition `1.0000`, target conjunction `.8208`, primitive spatial relation `.8069`, overall `.9380`. Post-hoc subsets: Type A `11` families, primitive-clean Type A `5`; Type B `146`; Type C `23`. Five clean Type-A families are suitable only for a small exploratory case-control audit.
- Final conclusion: the dataset exposes native wrong-object selection failures and an object-count degradation curve, but not observable mixed/illusory feature binding or an interference-specific high/low effect. N=8 has the largest generic burden, while I3 is not worse than I0. Do not claim that this run localizes a binding mechanism or that I3 causes mixed binding; first recalibrate above/below relation controls and/or strengthen the behavioral construction before any mechanistic follow-up.
- Commands run: dataset generation/audit/contact sheet, processor alignment audit, frozen behavior command `python -u -m experiments.native_binding_failure.evaluation.behavior --split all --device cuda --dtype bf16`, aggregate command `python -u -m experiments.native_binding_failure.evaluation.aggregate --root outputs/native_binding_failure`, compile/diff checks, and result completeness audit. Final artifacts include `REPORT.md`, `summary.json`, CSVs, raw generations, mechanistic subset manifest, and five figures.

## 2026-08-22 22:41 CST: Legacy Feature-Binding Regime Replication — Minimal Smoke Test

- User requested exactly one frozen 16-image diagnostic: disjunctive/conjunctive visual search at N=`10,50`, present/absent, two seeds; no training, repair, probes, attention analysis, causal tracing, prompt tuning, or automatic sample expansion. Added `experiments/legacy_binding_smoke/{generate.py,audit.py,evaluate.py,report.py}` and independent outputs under `outputs/legacy_binding_smoke/`.
- Generated exactly 16 deterministic 448×448 RGB images once using integer PIL primitives: two fixed seed layouts per N, shared positive/negative position sets, no overlap, no unknown/antialias pixels. Audit and contact sheet passed; all eight positive/negative pair pixel diffs were confined to the target slot. Conjunctive-negative metadata verified `green L=0`, with both green T and red L distractors.
- Reused local `models/Qwen3.5-4B`, the native-binding processor/tokenizer/chat-template path, official `enable_thinking=False`, CUDA bfloat16, frozen parameters, unrestricted greedy generation (`do_sample=False`, temperature recorded `0.0`, max 8 new tokens). External CUDA run completed all 16 rows and preserved raw generations; no restricted candidate scoring was used.
- Results: overall `11/16=.6875`; disjunctive N10/N50 present and absent were all `2/2`; conjunctive positive was N10 `1/2`, N50 `2/2`; conjunctive-negative was `0/2` at both N10 and N50. All four conjunctive-negative cases were false positives (`Yes` with no green L); there was one conjunctive-positive false negative at N10 and no invalid outputs.
- Interpretation: the strongest qualitative signal is conjunction-specific-looking false positives with a perfect disjunctive negative control, compatible with a legacy feature-binding/illusory-conjunction pattern. It is not a clean N10→N50 degradation interaction because the conjunctive-negative failure is already present at N10 and persists at N50; 16 images/two seeds cannot establish a binding mechanism or quantify improvement versus older VLMs. Stop here; do not expand, tune, or run mechanistic follow-up without a new protocol.
- Commands run: `python -m experiments.legacy_binding_smoke.generate`, `python -m experiments.legacy_binding_smoke.audit`, persistent external CUDA `python -u -m experiments.legacy_binding_smoke.evaluate`, `python -m experiments.legacy_binding_smoke.report`, and `py_compile`. Final artifacts include `images/`, `metadata.jsonl`, `contact_sheet.png`, `audit.json`, `raw_generations.jsonl`, `results.csv`, `summary.json`, `runtime_audit.json`, and `REPORT.md`.

## 2026-08-23 01:48 CST: Task-Conditioned Binding Computation — Behavioral Gate Stop

- User requested an isolated Qwen3.5-4B experiment comparing direct, explicit same-entity, marginal-scaffold, and entity-bundled protocols on matched conjunctive-search scenes, with causal localization only after a valid DME contrast. Added `experiments/task_conditioned_binding/{common.py,dataset.py,behavior.py,causal.py,report.py,run.py}` and `tests/task_conditioned_binding/test_protocol.py`; no prior experiment directory or output was overwritten.
- Generated exactly 120 448×448 images: 60 matched positive/negative families, 30 at N=10 and 30 at N=50. Eight target color/shape definitions are counterbalanced; every pair shares positions and differs at exactly one target-slot identity. Dataset audit passed with zero failures; contact sheet was visually inspected. Outputs are under `outputs/task_conditioned_binding/dataset/`.
- Frozen CUDA bf16 Qwen3.5-4B greedy evaluation completed all 480 rows (120 images × D0/D1/M/E) with the official `enable_thinking=False` template, temperature 0, and raw generations preserved. Exact-length batching was used; five initial singleton checkpoints were recomputed through the finalized batch path before final analysis. Runtime audit records model hashes, 4,539,265,536 frozen parameters, CUDA bf16, and 480 unique keys.
- Parsed semantic negative-conjunction FPR by N: D0 `.433/.900`, D1 `.267/.533`, M `.267/.667`, E `.167/.867` for N=10/N=50. D1 reduces parsed FPR versus D0 at both counts, but strict D1 output-format validity is only `.033/.000`; M/E scaffold and answer validity are also imperfect. E has no stable count-generalized advantage and is worse at N=50.
- Paired negative McNemar: D1 vs E A-only/B-only correct `22/4`, exact two-sided `p=.0005335`; M vs E `1/8`, `p=.03906`. Family-bootstrap FPR differences: D1−E `-.1167`, 95% CI `[-.2667,.0333]`; M−E `-.0500`, CI `[-.2000,.1000]`.
- Behavioral gate required at least 10 negative scenes with D1=wrong Yes, M=wrong Yes, E=correct No, exact target-free E scaffold. Only 2 semantic candidates existed; both E scaffolds incorrectly included the target conjunction, so valid triplets=`0`. **STOP after behavior; no model forward, residual patching, donor control, attention/MLP decomposition, or head scan was run.**
- Final report: `outputs/task_conditioned_binding/REPORT.md`; required behavior CSV/JSON/figure artifacts and empty `causal/critical_triplets.jsonl` plus `causal/behavioral_gate.json` are present. Static compilation, dataset checks, raw-key completeness, parser section check, and direct protocol checks passed. Do not proceed to causal or repair work without a new explicit protocol.

## 2026-08-23 16:53 CST: Cross-Entity Evidence Integration — Causal Gate Stop

- User requested the frozen N=50 D0 cross-entity integration test before any repair. Added `experiments/cross_entity_integration/{__init__.py,common.py,dataset.py,behavior.py,causal.py,report.py}` and independent outputs under `outputs/cross_entity_integration/`; no existing experiment output was overwritten and no repair/training code was run.
- Selected 27 behavioral N=50 D0 negative false-positive source families. Generated 135 matched five-state images (B00/C10/S01/F11-X/T11-S), preserving 448×448 geometry, object count, positions, bboxes, and rendering. Counterfactual audit and contact sheet passed. Frozen CUDA bf16 Qwen3.5-4B greedy behavior completed all 135 rows: B00 No `27/27`, C10 No `25/27`, S01 No `27/27`, F11-X Yes `27/27`, T11-S Yes `27/27`; clean families `25/27`, behavioral gate passed.
- Coarse visual-region causal sweep completed all `25 × 17 × 6 = 2550` rows at layers `0,2,...,30,31`. Runtime/precomputed-visual equivalence passed exactly; all parameters were frozen; no-op batch/suffix margin difference was `0.0`; random donors carried neither destination target marginal. Equalization is by object-region count (4 color, 4 shape, 8 random); merged 14×14 visual-token footprints vary and are recorded explicitly.
- Primary causal gate **STOP**: no layer had both `INT` bootstrap lower CI > 0 and `ΔCS−ΔR` lower CI > 0. Strongest mean `INT=.035` at layer 18, with `INT` CI `[-.005,.070]` and `CS−R` CI `[0,.080]`; candidate layers were empty. The coarse sweep did include the IT true-target positive-control condition (largest mean Δ margin `.555` at layer 0); only the gate-dependent fine true-target file, attention/MLP decomposition, head scan, and D1 auxiliary control were not run.
- Final report and figures are in `outputs/cross_entity_integration/REPORT.md` and `figures/`. Commands covered dataset generation, frozen behavior, runtime mapping, durable CUDA coarse sweep, offline metadata correction, report generation, and `py_compile`. Conclusion: the matched behavior isolates the false-positive regime, but this protocol does not causally support a supra-additive cross-entity integration mechanism. Do not design ECR, consistency gates, LoRA, slots, or other repair from this result without a new protocol.

## 2026-08-23 CST: Marginal-Matched Entity Pairing Diagnostic — H2 Result

- User requested a behavioral-only Qwen3.5-4B test of entity-level color-shape correspondence under exactly matched A/B marginals, with no training, probes, causal patching, head search, prompt changes, or repair. Added `experiments/marginal_matched_pairing/{__init__.py,common.py,dataset.py,behavior.py,report.py,run.py}` and independent outputs under `outputs/marginal_matched_pairing/`.
- Generated 120 deterministic geometry families / 240 images at `m={1,4,12,24}`, N=50, with 8 counterbalanced validated color/shape templates. A/B geometry, bboxes, sizes, background, marginal counts, joint counts, neutral objects, and image metadata audited with zero failures. The neutral circle uses the already validated integer PIL primitive; no new visual style was introduced.
- Token audit passed: `Yes=[9175]`, `No=[2665]`; CUDA bf16 official Qwen3.5-4B, frozen parameters, `enable_thinking=False`, greedy temperature 0. Full behavior completed 960/960 unique rows with zero invalid predictions, noncandidate first tokens, or free-generation/full-vocabulary mismatches.
- Results: mean PS=`8.767,9.396,7.077,2.306` for m=`1,4,12,24`; strict pair accuracy=`1.000,.967,.800,.100`; PS(1)−PS(24)=`6.460`, family bootstrap 95% CI `[5.917,7.029]`. QC and QS accuracy were `1.000` at every m; marginal A−B margin CIs included zero at every m. Template/factor summaries are saved in `behavior/summary_by_template.csv` and `behavior/confound_summary.csv`.
- Best-supported interpretation is **H2: repetition-dependent pairing degradation**. The endpoint decline is not strictly monotonic because m=4 exceeds m=1, but every template has lower mean PS at m=24 than m=1. The result conceptually clears the user’s gate for a future native entity/spatial-index analysis, but that follow-up was not run; stop at `outputs/marginal_matched_pairing/REPORT.md`.

## 2026-08-24 11:00 CST: Native Index Availability vs Utilization Pilot — Causal Validation Stop

- User requested the small frozen Qwen3.5-4B pilot separating native feature-to-object indexed retrieval availability from causal utilization, reusing only the matched-marginal m=4/m=24 dataset. Added `experiments/index_availability_use_pilot/{__init__.py,common.py,run.py,report.py}` and independent outputs under `outputs/index_availability_use_pilot/`; no training, probe, repair, prompt, dataset expansion, or head/layer scan was run.
- Deterministic subset and processor audit passed: m=4 calibration/held-out=`10/10`, m=24 held-out=`10`; A/B geometry and matched marginals passed; 120 Q1/Q2 rows had identical A/B tokenization, 196 visual tokens, and one-token color/shape spans. Native pre-softmax q·k extraction used only full-attention layers.
- Frozen m=4 calibration selected zero-based layer `27`, heads `[13,1,8,6]`. Held-out availability: IA=`.7851` (m=4, bootstrap 95% CI `[.6285,.9430]`) and `.6127` (m=24, `[.1148,1.3070]`); m24−m4 CI `[-.7118,.5211]`. Color/shape localization remained positive but weakened at m=24 (`1.000/.750` and `.975/.800`).
- CUDA bf16 causal stage completed 80 selected-channel donor swaps, 1,600 random-four-head controls, and 40 QS controls. No-op audit passed exactly: max full-vocabulary logit and Yes–No margin differences `0.0`; QC/QS before/after accuracy remained `1.0/1.0`. PS on held-out families was `9.4812` (m=4) and `2.0312` (m=24). `CE_spec` was `-.008125/-.003125`; `U_idx` was `-.000794/-.000918`.
- Final classification is **Case D**: the selected channel localized individual features but did not pass the m=4 directional causal validation gate (`CE_index=-.00625`, `CE_spec=-.008125`, both bootstrap intervals crossed zero). The small numerical U_idx decrease at m=24 is not evidence for H_use, and IA’s between-m CI crosses zero. The bf16 batch/singleton clean-margin audit had max absolute difference `.25`, recorded in `causal/clean_margin_audit.csv`; own-value branch no-op remained exact. Stop; do not run native Position-ID analysis, repair, layer/head search, or new data without a new protocol. Final report: `outputs/index_availability_use_pilot/REPORT.md`.

## 2026-08-24 16:24 CST: Crossed Binding-State Transplant — NOT_ESTABLISHED

- User requested the mechanism-agnostic text-side entity-carrier causal diagnostic after the Native Index Availability pilot’s Case D stop. Added `experiments/crossed_binding_state_transplant/{__init__.py,run.py}` and independent outputs under `outputs/crossed_binding_state_transplant/`; no Position-ID/head search, repair, training, or old-experiment rerun was run.
- Built and audited 48 family-disjoint geometry templates (24 selection/24 test), four crossed scenes S00/S10/S01/S11 each, 192 images, exact matched geometry/shape/color marginals, and official prompts `Target object: <shape>. Report its color.` plus the auxiliary left/right prompt. Gate 0 completed all 768 rows: color was `384/384` correct, position was `348/384` correct, with color-of-A/B `1.0/1.0` and position-of-A/B `0.854167/0.958333`; clean families were selection `19/24`, test `12/24`, with every exclusion recorded.
- Runtime audit used local `Qwen/Qwen3.5-4B` at `/root/Qwen3-VL/models/Qwen3.5-4B`, CUDA bfloat16, 32 `Qwen3_5DecoderLayer` blocks, hidden size 2560. Actual full-attention indices were `[3,7,11,15,19,23,27,31]`; other blocks exposed `Qwen3_5GatedDeltaNet`/linear attention. All blocks remained eligible for residual hooks. Image tokens were `[4,199]`; main color prediction position `218`, auxiliary position `224`; candidate color/position answers were all one-token. Tokenizer and chat-template hashes are in `model_audit.json`.
- Phase 1 family-CV probes over all 32 block residuals selected frozen `l*=25` (`model.language_model.layers.25`): selection BAcc_pos `1.0`, BAcc_color `.421053`, `G=.578947`, with 10,000 family bootstrap draws. Held-out test was opened only after freeze: position BAcc `1.0`, color BAcc `.75`.
- First Phase-2 implementation attempt was quarantined in `raw_invalid_batch_state/`: state cache used batch 16 while causal forward used batch 1, failing NO-OP with max full-vocabulary error `.1875`. Formal state recapture and causal run used batch 1 throughout; formal NO-OP passed exactly (`0.0`). The state audit also records a fallback-runtime main-vs-auxiliary carrier max difference `.625`; this caveat is reported and was not treated as causal evidence.
- Formal Phase 2 completed 192 rows (48 held-out pairs × NOOP/SPATIAL_TRANSPLANT/ATTRIBUTE_ASSIGNMENT/RANDOM_FAMILY_NORM_MATCHED). Results: spatial CE L→R `.0104167`, CI `[-.03125,.0520833]`, TRFR `0`; R→L CE `-.0078125`, CI `[-.0416667,.0286458]`, TRFR `0`; pooled CE `.00130208`, CI `[-.0260417,.0325521]`. Spatial−attribute `-.0664063` CI `[-.0950521,-.0390625]`; spatial−random `-.0494792` CI `[-.0729167,-.0247396]`; color-reversal consistency spatial `.08333` versus random `.33333`. Gates A–E all failed; implementation gate passed.
- Final verdict is **NOT_ESTABLISHED**. Phase 3 direction transport was not run because Phase 2 did not pass all gates. No answer change is treated as binding evidence, no explicit Position-ID/binding-matrix claim is made, and no repair follow-up is authorized from this result. Final report: `outputs/crossed_binding_state_transplant/REPORT.md`.

## 2026-08-24 CST: Crossed Binding-State Transplant — Independent Review

- User requested a read-only recheck; no model forward, causal rerun, old-experiment rerun, protocol search, or repair was performed. Recomputed manifest/family completeness, behavior counts, stable-family filtering, frozen-layer metadata, formal-row uniqueness, NO-OP, family bootstrap CIs, gate values, output completeness, and static compilation/diff checks.
- Corrected a documentation ambiguity: 768/768 means behavior rows completed, not all correct. Color was 384/384; position was 348/384, leaving 31 all-correct families (selection 19, test 12). Formal causal rows use only the 12 held-out test stable families, 192 unique rows, batch=1 state capture and batch=1 causal execution.
- Independent recomputation matched the saved result: L→R CE `.0104167` CI `[-.03125,.0520833]`, R→L CE `-.0078125` CI `[-.0416667,.0260417]`, spatial−attribute `-.0664063` CI `[-.0950521,-.0390625]`, spatial−random `-.0494792` CI `[-.0729167,-.0247396]`, NO-OP max error `0.0`; Gates A–E all false and no Phase 3 artifacts exist.
- Caveats retained: main/auxiliary carrier residuals differ by up to `.625` despite shared token IDs through the carrier, so task-invariant carrier-state claims are not supported; the dormant Phase-3 function does not yet implement the required repeat of all Phase-2 controls, but it was correctly blocked and never executed. Final verdict remains **NOT_ESTABLISHED**.
- Review also found 12 inherited behavior rows with missing `candidate_scoring_mode` provenance (all are correct selection rows and none enter formal test causal rows); they were not rerun. Report generation was aligned to the saved gate-specific R→L CI (`[-.0416667,.0260417]`); no result files or causal values were changed. `py_compile` and `git diff --check` passed.

## 2026-08-24 17:15 CST: CBST L25 Causal Audit — AUDIT_INCONCLUSIVE

- User requested a minimal preregistered explanation audit distinguishing causally-live/non-routing from readable/causally-dead L25 carrier. Added `experiments/cbst_l25_causal_audit/{__init__.py,run.py}` and independent outputs under `outputs/cbst_l25_causal_audit/`. No layer/head/channel search, repair, training, Phase 3, or alternate hook was run; `raw_invalid_batch_state/` was explicitly not read.
- Audit A reconstructed the original fixed selection-only probe because CBST did not save probe weights. On the 12 held-out stable test families at frozen L25: position BAcc `1.0`, AUC `1.0`, family permutation `p=.000488`; bound-color BAcc `.75`, AUC `.815972`, family permutation `p=.002197`; outcome **A-PARTIAL** (role generalizes, but color is also significantly readable).
- Audit B ran 288 batch=1 rows over the same 12 held-out families: NOOP/SAME_IMAGE_ALTERNATE/RANDOM_FAMILY_NORM_MATCHED each 96, with the exact L25 post-block carrier hook at position 204. NO-OP full-vocabulary and margin errors were `0.0`; A←B CE `.0742188` CI `[.0273438,.121094]`, B←A CE `-.0494792` CI `[-.138021,.0351563]`, pooled CE `.0123698` CI `[-.0208333,.0436198]`, ITR `0/0`; same-image-minus-random CE `-.0494792` CI `[-.103516,.00390625]`. Stable same-image KL/JS changes without reliable bidirectional directional margin gave **WEAK_CARRIER_LIVE** under the predeclared viability rule.
- Joint result is **AUDIT_INCONCLUSIVE** because Audit A is A-PARTIAL and Audit B is WEAK_CARRIER_LIVE. Original CBST remains **NOT_ESTABLISHED**; the audit does not establish absence of binding and does not justify another-layer search. All required audit artifacts and limited plots are under `outputs/cbst_l25_causal_audit/`; external CUDA execution used the frozen local Qwen3.5-4B checkpoint and batch=1.

## 2026-08-27 CST: Minimal Faithful Position-ID Replication on Qwen3.5 — Pilot Case B

- User requested the standalone Stage 1 prerequisite experiment reproducing the original scene-description representation chain and full-attention image-key causal retrieval protocol. Added `experiments/qwen35_position_id_replication/{protocol.py,dataset.py,run.py,README.md,__init__.py}` and independent outputs under `outputs/qwen35_position_id_replication/`. No GDN state patching, KV/GDN factorial, CBST, L25 transplant, repair, training, head search, or automatic formal expansion was run.
- Model/processor audit used local `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, official `enable_thinking=False`, CUDA bfloat16, frozen parameters, 32 decoder layers. Runtime discovery found full attention `[3,7,11,15,19,23,27,31]` and Gated DeltaNet in the other 24 layers. Full-attention audit recorded `q_proj/k_proj/v_proj`, 16 query heads, 4 KV heads, head dim 256, and `k_proj -> k_norm -> transpose -> multimodal RoPE`; the primary key hook is post-`k_norm`, pre-RoPE.
- Geometry audit passed on 20 deterministic sanity images and all 504 images: runtime `image_grid_thw=[1,28,28]`, patch 16, merge 2, merged grid 14×14, 196 visual tokens; each object maps to exactly four equal/disjoint post-merger visual tokens. `visual_token_mapping.csv` contains 2,736 image/object rows with bboxes, grid cells, and decoder token IDs.
- Pilot behavior gates passed. Scene description used 36 balanced rows (12 per split): color/shape/joint accuracy all `.944444` (weakest target slot `.666667`). Color retrieval used 12 selection + 12 test queries; template calibration selected `template_a`, and clean A-color accuracy was `1.0` overall, balanced, and at every query position. An initial local label-field bug had compared clean predictions to B rather than queried A; it was corrected before final scoring and old rows were invalidated.
- Representation extraction covered all 32 decoder layers at the exact final `a` token. Held-out pilot RSA gave position peak layer 18 and feature peak layer 28; early positive position-advantage layers were `[13,15,16,18]`, feature rise over the pre-peak baseline `.476792`. The frozen validation-RSA feature window was full-attention layers `[3,7]`; PCA, RSA, and probe CSVs/figures were saved.
- Selection-only key directions produced 96 layer×A→B directions from 12 scenes. Native batch=1 causal pilot completed all `12 × 8 × 3 = 288` rows for A→B, A→C, and norm-matched random controls. No-op passed: D=0 and A→A logits/actual hook delta were exact zero; float32 synthetic add/undo max error was `2.384e-7 < 1e-6`. Candidate-window A→B `CE_key=.0104167`, 95% CI `[-.0520833,.0703125]`, median `.015625`; TRSR `0` with CI `[0,0]`; random superiority, A→C specificity, and color reversal all failed. All full-layer A→B TRSRs were zero; layer 27 had the only positive CE CI lower bound (`.015625`).
- Frozen pilot verdict is **`REPRESENTATIONAL_SEQUENCE_REPLICATED_BUT_CAUSAL_KEY_MECHANISM_NOT_REPLICATED`** (pre-registered Case B), explicitly scoped as a pilot rather than a formal 100–200-query causal claim. Six-class identity probes for directions are NaN because the pilot has one selection scene per ordered pair and are not interpreted. Final report: `outputs/qwen35_position_id_replication/REPORT.md`. Stop Stage 1 here; do not infer that Qwen3.5 lacks Position ID or that GDN stores a missing Position ID.

## 2026-08-28 17:05 CST: Visual-Prefix Persistent-State Causal Factorial — Implementation Gate Stop

- User requested the frozen Stage-2 whole-prefix causal factorial following Stage-1 verdict `REPRESENTATIONAL_SEQUENCE_REPLICATED_BUT_CAUSAL_KEY_MECHANISM_NOT_REPLICATED`, testing complete full-attention KV (`K`) versus complete GDN continuation state (`G`) only. Reused the read-only crossed spatial-assignment family/scene manifests from `outputs/crossed_binding_state_transplant/`; no new data, prompts, model, Position-ID search, layer/head search, subspace, repair, training, or Stage 3 work was run.
- Added/used `experiments/qwen35_kv_gdn_factorial_pilot.py` and independent output root `outputs/qwen35_kv_gdn_factorial_pilot/`. Dataset audit passed for 48 families (24 selection/24 test), behavior screening completed 192 rows over the 24 test families with target/source and all four query types exact, and 48 immutable target/source prefix snapshots were captured at boundary 201. Runtime inventory found 8 full-attention layers and 24 GDN layers; `K` contains `keys`/`values`, while `G` contains both `conv_states` and `recurrent_states` on every GDN layer. State checksum/component-construction audit passed (48 snapshots, 96 condition constructions, 6,144 component-field checks).
- Commands run included prepare, behavior, state capture, integrity, two CUDA segmented-equivalence attempts, read-only singleton diagnostics, static `py_compile`, and `git diff --check`. The installed runtime used CUDA bf16, default text attention `sdpa`, and the Transformers torch GDN fallback (`fast_path_available=false`). A no-padding logical-boundary chunk shim was tested to remove synthetic final-tail state updates; it was not a causal intervention.
- Phase 0A failed before NO-OP/all-source/factorial: on 16 calibration rows, prediction winners agreed 16/16, but max candidate score error was `0.6252756` and max source-target margin error `0.125`, both above frozen `0.05`. Singleton full-versus-segmented diagnostic differences remained about `0.125–0.437` score units; an eager-attention diagnostic did not resolve the discrepancy. Prompt/full-candidate mRoPE prefix IDs were exact (`[3,1,219]`, boundary 201, `rope_delta=-182`), so the failure was not attributed to position metadata.
- Verdict is **`SEGMENTED_FORWARD_INVALID`** / implementation stop. `no_op`, all-source completeness, TS/ST/SS mixed-state rows, identity mixed-state controls, bootstrap statistics, plots, and Stage 3 were not run or interpreted. Final report: `outputs/qwen35_kv_gdn_factorial_pilot/REPORT.md`; stop manifest: `stage_stop_manifest.json`. This does not support any claim about GDN, KV, or Position-ID causal storage; future work requires a separately authorized cache-semantics protocol.

## 2026-08-28 18:48 CST: Stage 2A Intervention-Compatible Execution Semantics — Established

- User requested an engineering-only audit of the prior `SEGMENTED_FORWARD_INVALID` stop, preserving its `0.05` gate and forbidding TS/ST/SS. Added `experiments/qwen35_execution_semantics_audit.py` and independent outputs under `outputs/qwen35_execution_semantics_audit/`; no model weights, processor/template, prefix boundary, data, causal factorial, layer search, repair, or training changed.
- Runtime hooks on Transformers `5.8.1` / CUDA bf16 confirmed all 24 GDN layers: standard multi-token uses `torch_chunk_gated_delta_rule` plus full convolution; cached `seq_len=1` uses `torch_recurrent_gated_delta_rule` plus in-place `torch_causal_conv1d_update`; cached multi-token uses the chunk rule with initial recurrent state and cached conv context. Chunk size is 64 and formal `b=201` has remainder 9. Fast path is unavailable. Zero-initializing active GDN conv/recurrent cache forces recurrent semantics from token 0 while full-attention KV length remains 0.
- Continuous-recurrent versus split-recurrent completed 16/16 calibration rows (four families × Left/Right/identity-A/identity-B). All results were exact: winner agreement `1.0`, max candidate-score error `0.0`, max margin error `0.0`, first-answer full-vocabulary max/L2 error `0.0`, signed drift `0.0`; 5,832 traced GDN kernel events were recurrent and chunk events were 0. Prefix save/reload and final state hashes were exact on every row, all cache storages disjoint, immutable snapshots/unrelated split caches unmutated, position IDs and final FA cache lengths exact. An initial pre-row implementation attempt mistakenly expanded an already-batched snapshot from B=4 to B=16; it failed before any score row was saved and was corrected to preserve native batch.
- Standard versus continuous recurrent behavior passed: prediction agreement and clean accuracy `1.0`, spatial source-target margin-sign agreement `1.0`, Pearson `.9987888`, Spearman `1.0`, median/max relative spatial-margin error `.0228639/.0833333`. The optional one-sample boundary diagnostic did not show a simple alignment fix: score/margin errors were `.74846/.1875` at aligned b=192, `.31163/.0625` at b=197, and `.37410/.0625` at formal b=201.
- Final verdict is **`RECURRENT_EXECUTION_EQUIVALENCE_ESTABLISHED`** and Stage 2 recurrent-regime factorial is technically authorized, but was not run. Important diagnostic limit: the prior formal segmented suffix has `seq_len=21` and runtime hooks show it also used the chunk rule, so the narrow “original discrepancy was chunk-vs-recurrent kernel switching” claim is false; the result is consistent with a broader chunk prefill/resume semantics mismatch, whose exact cause remains unidentified. Future TT_rec/TS_rec/ST_rec/SS_rec must all use the unified recurrent regime; standard forward is behavior control only. Final report: `outputs/qwen35_execution_semantics_audit/REPORT.md`.

## 2026-08-29 17:19 CST: Stage 2B KV × GDN Recurrent Factorial — KV_DOMINANT

- User requested the authorized whole-prefix KV-versus-GDN persistent-state causal factorial after Stage 2A established the recurrent execution regime. Added `experiments/qwen35_kv_gdn_factorial_recurrent_pilot.py` and independent outputs under `outputs/qwen35_kv_gdn_factorial_recurrent_pilot/`; reused the frozen crossed spatial-assignment families and read Stage 1/2A artifacts. No Position-ID search, layer/head/state decomposition, repair, training, or Stage 3 was run.
- All 384 formal rows completed on 24 selected clean families × Left/Right/identity-A/identity-B queries × TT/TS/ST/SS. Every row used `RECURRENT_REFERENCE` tokenwise `seq_len=1`; canonical CSV integrity is 384 rows, 96 family-query keys, and exactly 96 rows per condition. Prefix boundary remained `b=201`; `K` was all FA keys/values and `G` was all GDN conv/recurrent state fields.
- TT resume and SS completeness passed on 16/16 calibration rows each with max candidate-score, margin, first-answer-logit, and L2 errors all `0.0`. State integrity passed: `K_TS==K_T`, `G_ST==G_T`, 6,144 component-field checks, disjoint condition storage, immutable snapshots, and fresh cache reconstruction all exact.
- Pooled spatial effects: `CE_GDN=.341146` (95% CI `[.256478,.429688]`, normalized `R_GDN=.027266` CI `[.019657,.035274]`), versus `CE_KV=11.3841` (CI `[11.0807,11.7227]`, `R_KV=.964148` CI `[.955787,.972502]`). GDN was much smaller and directionally unstable (Right CI `[-.033854,.226563]`) with `SATR_GDN=0`; KV was positive and stable in both directions with `SATR_KV=1`. The paired recovery difference was `R_GDN-R_KV=-.936882` CI `[-.943331,-.930248]`.
- Factorial interaction was not clearly established: `I=.091146`, 95% CI `[-.078125,.256510]`. Identity controls remained stable (accuracy `1.0` in TT/TS/ST/SS; no answer flips; generic-state-perturbation flag false). Source-color reversal checks were source-specific for both effects, but only KV met the bidirectional stability requirement.
- Frozen verdict is **`KV_DOMINANT`** for this pilot. This supports a larger source-specific causal effect through full-attention KV than through whole-prefix GDN state under the controlled recurrent regime; it does not show that GDN stores Position IDs. Stage 3 was not run and is not automatically authorized. Final report: `outputs/qwen35_kv_gdn_factorial_recurrent_pilot/REPORT.md`.

## 2026-08-29 20:08 CST: Stage 2C Full-Attention Key–Value Decomposition — INCONCLUSIVE

- User requested the frozen whole-FA-cache K/V factorial after Stage 2A `RECURRENT_EXECUTION_EQUIVALENCE_ESTABLISHED` and Stage 2B `KV_DOMINANT`: TT/KS/VS/KVS only, `G=G_T`, all eight FA layers, same 24 clean families/queries/prefix boundary/scorer, no STANDARD_FULL_FORWARD causal baseline and no follow-up search. Used `experiments/qwen35_fa_key_value_factorial.py` and independent outputs under `outputs/qwen35_fa_key_value_factorial/`. Added only strict EOF-incomplete-row checkpoint sanitation and an explicit Stage-1/Stage-2B comparison paragraph; no model/data/protocol change.
- Source, target-baseline, KVS→Stage-2B-ST positive-control, and state-integrity gates passed. Formal output has 384 rows: 24 families × Left/Right/identity-A/identity-B × TT/KS/VS/KVS, 96 per condition; all rows are `RECURRENT_REFERENCE`, all `m_TT<0` and `m_KVS>0`, and all 48 spatial rows satisfy `D>=0.5`. All 96 KVS rows exactly matched the corresponding Stage-2B ST rows for candidate scores/probabilities, margin, prediction, and first-answer logits in an offline full-file comparison.
- Pooled spatial results: `CE_K=1.41927` (95% CI `[.742188,2.14456]`), `R_K=.134941`, `SATR_K=.104167`; `CE_V=9.73958` (CI `[8.85677,10.6276]`), `R_V=.851908`, `SATR_V=.916667`. Left/Right were both positive and stable for K and V; source-color reversal passed for V but not K. Factorial `E_K=1.53190`, `E_V=9.85221`; `I_KV=.225260` (CI `[-.294271,.729167]`), so strong spatial interaction was not established.
- Identity controls failed for isolated components: TT/KVS accuracy `1.0`, but KS=`.125` (flip `.875`) and VS=`.104167` (flip `.895833`), with negative correct-answer margins and large JS/KL shifts. Therefore `GENERIC_STATE_PERTURBATION=true`; despite a raw spatial pattern that is V-dominant, the pre-registered specificity gate blocks `VALUE_DOMINANT`. Final verdict is **`KEY_VALUE_CAUSAL_DECOMPOSITION_INCONCLUSIVE`**; no K/V standalone causal interpretation, Position-ID claim, or Stage 3/follow-up experiment is authorized. Final report: `outputs/qwen35_fa_key_value_factorial/REPORT.md`.
- Commands run: static compile, `--stage prepare`, resumed `--stage factorial` from an EOF-repaired checkpoint, `--stage analyze`, and read-only row/formula/completeness/positive-control audits. Required JSON/CSV/report/plot artifacts are present; no subsequent experiment was run.

## 2026-08-30 16:36 CST: Stage 2D Orthogonal Address–Payload — Behavior-Gate Stop

- User requested the independent frozen `Stage 2D — Orthogonal Address–Payload Counterfactual` on the same 24 Stage-2B families, with S00/S10/S01/S11, Address/Payload balanced reverse edges, whole-prefix FA K/V only, target `G_T`, and no follow-up mechanism search. Added `experiments/qwen35_address_payload_orthogonal.py` and `outputs/qwen35_address_payload_orthogonal/`; generated 96 new four-scene images and a 192-row pair manifest. Source Stage-2A/2B/2C reports and required artifacts were hashed/read-only audited.
- Code review/fixes: corrected spatial-query candidate typing; added the existing allocator warm-up environment workaround with immediate restoration; fixed B=4-to-B=1 target baseline comparison; added Stage-2A provenance; fixed zero-family stop handling in integrity/gates. A fixed raw-vs-embedding scorer diagnostic failed candidate-score equivalence (max `0.0622635`), so it was quarantined as `behavior_scorer_equivalence_audit.json` and never used; formal rows used the canonical raw-processor full-candidate scorer.
- CUDA behavior gate completed all `576/576` rows (24 families × 4 scenes × 6 queries). Identity-color accuracy was A=`1.000`, B=`.989583`; identity-position A/B were each `.53125`; spatial-left/right were each `.500`. Consequently `0/24` families passed the required all-query gate; every exclusion is recorded in `family_manifest.csv`, with no rescreening or result-dependent selection.
- Because no family was behavior-clean, `states`, formal implementation positive control, factorial K/V rows, invariant K/V controls, bootstrap, and plots were not run. The zero-family integrity path was fixed and finalized with explicit not-run summaries. Verdict: **`ORTHOGONAL_DECOMPOSITION_INCONCLUSIVE`**, stop reason `NO_BEHAVIOR_CLEAN_FAMILIES`, `mechanism_allowed=false`, `stage_3_run=false`. Final report: `outputs/qwen35_address_payload_orthogonal/REPORT.md`. This does not support or refute address/payload separation and makes no claim that keys encode Position IDs or values encode object features.

## 2026-08-30 19:51 CST: Stage 2D-A Orthogonal Stimulus Behavior Equivalence Audit — MIXED_BEHAVIOR_AUDIT

- User requested a behavior-only audit of the frozen Stage 2D orthogonal stimulus after its zero-family gate stop. Added `experiments/qwen35_stage2d_behavior_audit.py` and outputs under `outputs/qwen35_stage2d_behavior_audit/`; no K/V capture/intervention, GDN, prompt search, family reselection, repair, or follow-up mechanism experiment was run.
- Ran `--stage prepare`, then the prescribed CUDA behavior stage in a durable tmux session, followed by offline `--stage analyze`. Audit A independently used renderer object metadata, exact pixel RGB masks, centroid-x ordering, and image hashes over all 96 scenes: position/color/left-payload/right-payload/pixel-color/image-hash match rates were all `1.0`; mismatch rows `0`.
- Audit B replayed 8 deterministic frozen Stage 2B families / 64 rows with the unchanged local bf16 checkpoint, official chat rendering, historical prompts, and canonical raw full-candidate scorer. Predictions and correctness reproduced `64/64`; candidate-score max difference was `.62517643` and margin max difference `.25` under the host-CPU-remainder runtime, so scores were retained as diagnostic rather than called numerically identical.
- Audit C used the exact Stage 2B spatial prompts/scorer on all 96 Stage 2D scenes: spatial-left/right accuracy and balanced accuracy were both `1.0` in every S00/S10/S01/S11 group; identity-color A=`1.0`, B=`.989583`, pooled=`.994792`. Thus the frozen Stage 2D spatial chance result is protocol-sensitive.
- Audit D direct identity-position queries remained near chance: A/B=`.520833`, pooled balanced accuracy=`.520833`; predictions were left `188/192` and right `4/192`. Factor breakdown localized the failure to the identity-position assignment factor (A I0/I1=`1.0/.041667`; B I0/I1=`.041667/1.0`), while color-position levels were approximately chance.
- Matched S00 identity-color baseline was `1.0` for both original Stage 2B and Stage 2D, with image-hash match `1.0`. The exact spatial protocol restored spatial behavior, but direct position remained fixed-left and `0/24` families passed the original six-query all-clean gate. Final verdict: **`MIXED_BEHAVIOR_AUDIT`**, `mechanism_allowed=false`, `stage2d_causal_eligibility=false`; stop the Qwen3.5 mechanism sub-branch and do not run Stage 2D K/V causal decomposition.

## 2026-08-30 20:06 CST: Stage 2D-B Position-to-Identity Behavioral Direction Audit — POSITION_TO_IDENTITY_VALIDATED

- User requested a behavior-only Position→Identity audit on the frozen 24-family / 96-scene Stage 2D stimulus, using only the two fixed queries `Which object is on the left/right, the <A> or the <B>?`, full-candidate identity scoring, and no K/V or causal follow-up. Added `experiments/qwen35_position_to_identity_audit.py` and outputs under `outputs/qwen35_position_to_identity_audit/`; images/manifests were read-only and unchanged.
- Ran static compile, source-artifact prepare audit, durable CUDA behavior stage, and offline analyze. Source gates passed: Stage 2D frozen verdict/stop and Stage 2D-A mixed verdict matched; Stage 2D-A renderer/pixel/hash audit had 96 scenes and zero mismatches; prior Stage2B exact spatial-feature BAcc was `1.0` and identity-color accuracy `.994792`.
- All `192/192` P2I rows completed with the canonical Qwen3.5 chat template/image placement and raw-processor full-candidate sequence scorer. P2I-Left and P2I-Right each had accuracy/BAcc `1.0`; pooled accuracy/BAcc `1.0`. A-answer and B-answer accuracy were both `1.0`, predicted A/B rates were `.5/.5`, and the identity confusion matrix had no off-diagonal rows.
- Direction and factor checks were fully stable: A@Left and A@Right each accuracy/BAcc `1.0`; all S00/S10/S01/S11 groups, I0/I1 identity-position assignments, and C0/C1 color-position assignments were `1.0`. The optional no-image lexical-bias sanity check was not run.
- Side-by-side behavioral asymmetry: old Identity→Position from Stage 2D-A remained BAcc `.520833` with Left=`188/192` and Right=`4/192`, while Position→Identity was perfect without identity bias. This is only a directional/readout asymmetry, not proof that binding is intact or absent.
- Final verdict: **`POSITION_TO_IDENTITY_VALIDATED`**, `mechanism_allowed=false`, `stage2d_prime_causal_eligibility=true`, but `stage2d_prime_run=false`. A revised causal design may be behaviorally feasible only with validated position-to-identity/position-to-feature invariant readouts; do not automatically run Stage 2D′ or any K/V intervention. Final report: `outputs/qwen35_position_to_identity_audit/REPORT.md`.

## 2026-08-31 12:07 CST: Workspace Cleanup

- User requested a cleanup of files with no remaining use. Performed a read-only inventory first and removed only confirmed caches, download remnants, duplicate/invalid checkpoints, and Linux download metadata; no model, source code, formal result, valid checkpoint, or audit evidence was removed.
- Deleted `.git/objects/pack/tmp_pack_hOvsfo`, which `git count-objects -vH` reported as garbage (`3.59 GiB`) and which had no active Git process using it. Deleted byte-identical formal-output copies `outputs/qwen35_fa_key_value_factorial/factorial_rows_partial.csv` and `outputs/qwen35_kv_gdn_factorial_recurrent_pilot/factorial_rows_condition_batch_partial.csv`; deleted the obsolete `condition_batch2` and legacy 30-row partial checkpoints from the latter experiment. Deleted two explicitly stale D2L `*.invalid_batch_expand.csv` files and the obsolete address-payload candidate-type partial.
- Deleted the documented incomplete CLEVR archives `data/clevr/CLEVR_v1.0.zip` and `data/clevr/CLEVR_v1.0_no_images.zip`, their completed-download part directories `data/clevr/no_images_parts/` and `data/clevr/blender_parts/`, all repository `__pycache__`, `.pytest_cache`, and `.ipynb_checkpoints` directories, and all `*:Zone.Identifier` files. The four tracked OCR checkpoint images were not referenced by `cookbooks/ocr.ipynb`; normal OCR assets remain.
- Verification: workspace usage fell from about `87G` to `70G`; Git now reports `garbage: 0`; no targeted cleanup path remains; both formal factorial CSVs remain complete at `384` data rows each; `data/clevr/CLEVR_v1.0_no_images.parallel.zip` and the extracted Blender binary remain present.
- Deliberately retained the duplicate root-level LLaVA snapshot (`/root/Qwen3-VL/model-*.safetensors`) because legacy scripts explicitly load `/root/Qwen3-VL`; retained valid resumable partial outputs and invalid-run forensic artifacts cited by experiment reports. Removing those requires a separate decision about breaking legacy scripts or losing audit provenance.

## 2026-09-01 CST: Stage 2D′ Validated Orthogonal Address–Payload K/V Factorial — Case E Stop

- User requested the final frozen Qwen3.5 K/V branch experiment after the Stage 2A/2B/2C/2D/2D-A/2D-B read-only audit. Added `experiments/qwen35_stage2d_prime_orthogonal_kv.py` and independent outputs under `outputs/qwen35_stage2d_prime_orthogonal_kv/`; reused the frozen Stage 2D images and manifests without regeneration, and recorded all prior frozen conclusions plus permanent exclusion of the unreliable Identity→Position readout.
- Behavior gate reused the validated Stage 2D-A/Stage 2D-B rows: `576/576` rows, `23/24` clean families, minimum `20`, passed. One family (`family_test_45`) was excluded before state capture/intervention. State capture and implementation gates passed for `RECURRENT_REFERENCE`, target `GDN`, all eight full-attention layers, and TT/KS/VS/KVS; all required cache/component/checksum/storage audits were exact.
- Factorial and invariant controls completed all four files at `736` rows each (`23 families × 8 directed pairs × 2 queries × 4 conditions`), with no duplicate/NUL rows. Address and Payload whole-KV transport both passed: `SATR_KV=1.0`; source-directed KVS margins were positive with family-bootstrap CIs `[4.7644,4.9742]` and `[4.4494,4.6512]`. Validated invariant controls passed: Address KS/VS/KVS accuracy `.929348/.940217/1.0`, Payload KS/VS/KVS `1.0/1.0/1.0`; no generic K/V mismatch.
- Pooled primary contrasts were positive: Address `R_K=.798261`, `R_V=.176872`, `A_KV=.621388` CI `[.531700,.699523]`; Payload `R_K=-.018976`, `R_V=1.006375`, `P_VK=1.025351` CI `[1.009316,1.040939]`; `DD_R=1.646739` CI `[1.559133,1.724948]`, and `DD_CE=15.7334` CI `[14.7486,16.6081]`. However Payload failed preregistered forward/reverse, A/B-query, and source-color specificity consistency, so positive pooled DD cannot be interpreted as a stable double dissociation.
- An implementation-only repair was required after the first controls attempt: P2I prefix values were exact, but the metadata hash included four canonical color-candidate rows while P2I has two candidate rows. The new runner now normalizes candidate-row count while requiring exact per-candidate prefix mRoPE equality; controls then completed. No prompt, image, model, intervention condition, or statistical criterion changed.
- Final verdict is **`ORTHOGONAL_KV_DECOMPOSITION_INCONCLUSIVE`** (pre-registered Case E, directional/subgroup inconsistency), `mechanism_allowed=false`, with required stop: **STOP QWEN3.5 K/V MECHANISM BRANCH**. No GDN follow-up, layer/head localization, Position-ID search, nonlinear search, PCA/CCA, residual patch, CBST, repair, training, prompt search, natural-image extension, or result-dependent intervention was run. Final report: `outputs/qwen35_stage2d_prime_orthogonal_kv/REPORT.md`.

## 2026-09-01: Stage 2E Failure-Interface Pilot — Behavior Screen In Progress

- User authorized the strictly gated `Stage 2E — Address–Payload Failure Localization and Causal Rescue Pilot`. Added `experiments/qwen35_failure_interface_pilot.py` and independent output root `outputs/qwen35_failure_interface_pilot/`; no K/V state capture or intervention has been run.
- CPU-only prepare completed: 256 new deterministic families / 1024 scenes, exact Stage 2D/2D′ renderer design, 448×448 images, geometry/position/color/payload/hash audits all passed. Source artifact audit records the frozen Stage 2A/2B/2C/2D/2D-A/2D-B/2D′ conclusions and hashes all required Stage 2C artifacts; old Identity→Position readout is not used.
- CUDA command started in durable tmux session `stage2e_behavior`: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_failure_interface_pilot.py --stage behavior`. It uses the canonical raw-processor full-candidate scorer, behavior batch size 4, and checkpoints `behavior_all_rows.partial.csv`; at the last audit 52/6144 behavior rows were complete and the process remained healthy at 100% GPU utilization.
- The behavior stage must finish all six canonical queries, deterministic rerun, recurrent confirmation, and fixed pure-error selection before any state capture. Current run is incomplete; do not interpret the partial rows or start causal stages until `behavior_gate_summary.json` passes.

## 2026-09-01 20:33 CST: Stage 2E Behavior Resume

- A read-only status check found the original `stage2e_behavior` tmux server/PID gone after the checkpoint reached 920 data rows; GPU was idle, and `dmesg` showed no OOM or Python error. Partial integrity passed: 920 unique keys, no duplicates or malformed rows.
- Resumed the unchanged behavior command in a new durable `stage2e_behavior` session from the existing checkpoint. The model reloaded successfully; no completed rows were rerun, and no state capture or causal intervention has started. Latest observed checkpoint remains 920/6144 while the first resumed batch runs.

## 2026-09-02 13:30 CST: Stage 2E Behavior Resume After Environment Restart

- A subsequent status audit found the second behavior process stopped at 3236/6144 rows. Checkpoint integrity remained exact: 3236 unique keys, no duplicates or malformed rows, covering families 0–134. `uptime` showed a fresh system boot at 13:29 with unclean journal replacement, so this was an environment restart rather than a protocol or model failure.
- Restarted the unchanged durable `stage2e_behavior` session from the 3236-row checkpoint; Qwen3.5 weights loaded successfully and GPU execution resumed. Behavior gate remains pending; no K/V state capture or causal intervention has started.

## 2026-09-02 14:55 CST: Stage 2E Behavior Resume After Second Environment Restart

- The next status audit found the resumed process stopped at 3400/6144 rows. Checkpoint integrity again passed: 3400 unique keys, no duplicates or malformed rows, covering families 0–141. `uptime` was one minute and journald again reported unclean shutdown, confirming an environment restart rather than an experiment failure.
- Restarted the unchanged durable `stage2e_behavior` session from the 3400-row checkpoint; model loading completed and GPU execution resumed. Behavior gate remains pending; no state capture or causal intervention has started.

## 2026-09-02 19:11 CST: Stage 2E Behavior Resume After Third Environment Restart

- The next status audit found the behavior process stopped at 3860/6144 rows. Checkpoint integrity passed again: 3860 unique keys, no duplicates or malformed rows, covering families 0–160. `uptime` showed a one-minute-old system boot at 19:10; GPU was idle, so the interruption was another environment restart rather than a protocol failure.
- Restarted the unchanged durable `stage2e_behavior` session from the 3860-row checkpoint; model loading completed and execution resumed. Behavior gate remains pending; no K/V state capture or causal intervention has started.

## 2026-09-03 17:22 CST: Stage 2E Behavior Gate Complete; State Capture Started

- The 256-family behavior screen completed normally with `6144/6144` rows. There were `241/256` behavior-clean families and `15` stable pure-error families, so the preregistered `>=12` gate passed and extension to 512 was not performed. All 15 matched-correct controls were selected behavior-only; both orthogonal donors were correct for every selected error and donor answers differed from target gold.
- Behavior outputs are complete: `behavior_all_rows.csv`, `behavior_rerun.csv`, `recurrent_confirmation.csv`, `pure_error_manifest.csv`, and `matched_correct_manifest.csv`; no causal rows existed yet. Started durable CUDA session `stage2e_states` for the permitted 120 recurrent state snapshots (30 selected families × 4 scenes). Model loaded successfully; at first audit 0 state files had been written and causal intervention had not started.

## 2026-09-03 18:50 CST: Stage 2E Implementation Gate Passed

- Continued the authorized Stage 2E pilot. State capture completed `120/120` immutable recurrent snapshots with unique hashes, cache length `201`, and `state_capture_summary.passed=true`. The first implementation retry exposed a runner-only `orth.sem` reference error; a second retry exposed an invalid cross-batch TT comparison. The runner was repaired to lazy-load only the current calibration/control snapshots and to reproduce the exact four-scene capture batch; no model, data, prompt, intervention condition, or threshold changed.
- Re-ran `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_failure_interface_pilot.py --stage implementation` in durable CUDA tmux. Final gate passed for 4 calibration errors: `TT_or_T0_exact=true` with max candidate-score error `0.0`; all `K/V/G` component tensors exact, all cache reconstructions exact, all condition cache storages disjoint, and immutable snapshots unchanged. Formal causal rows had not been run before this gate.
- Current outputs include `implementation_target_rows.csv`, `implementation_condition_rows.csv`, `implementation_component_rows.csv`, `implementation_audit.json`, and a pending `verdict.json`. Next authorized step is the fixed Stage 2E causal rescue stage; no repair is authorized.

## 2026-09-03 18:52 CST: Stage 2E Causal Rescue In Progress

- After the implementation gate passed, started durable CUDA session `stage2e_causal` with `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_failure_interface_pilot.py --stage causal`. The fixed six conditions are scored in audited 4+2 batches because the reused scorer caps one batch at four conditions; no condition, state, query, threshold, or statistical unit changed.
- At the latest checkpoint, 2/15 pure errors had complete rescue blocks (`12/90` rescue rows) and both had complete 24-row invariant blocks (`48/360` invariant rows). The process remained live; final rescue/control summaries and verdict are not yet available. Continue from the partial CSVs if interrupted; do not interpret partial rows.

## 2026-09-03 19:36 CST: Stage 2E Causal OOM Recovery

- A WSL restart interrupted the causal process after the third error; checkpoint validation found no duplicate keys. Kernel logs also recorded the Python process killed by the host OOM killer while an external `cpptools` process requested memory. No scientific output was finalized.
- Preserved the prior mixed-batch partials as `error_rescue_rows.partial.csv.pre_batch1` and `invariant_preservation.partial.csv.pre_batch1`, then resumed the same six-condition causal protocol with `CAUSAL_CONDITION_BATCH_SIZE=1` to bound memory and keep all formal rows on one batch path. This is an execution-memory change only; all conditions, donors, queries, gates, and statistics remain fixed.
- The resumed run is live in `stage2e_causal`; first error has `6/90` rescue rows and `18/360` invariant rows at the latest check, with Python RSS about `2.9 GB` and no new OOM. Final analysis remains pending.

## 2026-09-03 22:02 CST: Stage 2E Error Blocks Complete; Controls Resumed

- The batch=1 causal run completed all 15 pure-error blocks: `90/90` rescue rows and `360/360` invariant rows. It then hit a runner-only sort-key typo on the first matched-correct control (`error_family_index` instead of `control_family_index`); no control result was written and no scientific analysis ran.
- Fixed that column reference and resumed the unchanged causal stage. Existing error/invariant rows were accepted by the batch=1 checkpoint gate; the first correct-control block is now complete (`6/90` control rows), with the process live and RSS about `3.1 GB`. Final control outputs, analysis, bootstrap, and verdict remain pending.

## 2026-09-03 22:39 CST: Stage 2E Causal Rescue Pilot Complete; Generic Perturbation Stop

- The fixed batch=1 CUDA causal stage completed all formal rows: `error_rescue_rows.csv` 90/90, `invariant_preservation.csv` 360/360, `correct_control_rows.csv` 90/90, and `wrong_donor_controls.csv` 30/30. Offline analysis completed with `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_failure_interface_pilot.py --stage analyze`; required output files are present, all formal causal CSV keys are unique, and all rows record `condition_batch_size=1`.
- Final behavior screen retained `15` stable pure errors from `256` families; both atomic Position→Color and Position→Identity readouts were correct, and both donors were behaviorally correct for every error. Implementation gates passed: TT/T0 exact, all K/V/G tensors and cache reconstructions exact, storage disjoint, immutable snapshots unchanged.
- Rescue results (valid invariant-preserving rows): K-rescue `13/15=.8667` (95% bootstrap CI `[.6667,1.0000]`), V-rescue `7/15=.4667` (`[.2000,.7333]`), joint `9/15=.6000` (`[.3333,.8667]`); wrong-K and wrong-V rescue were `0/15` each. However V-rescue and joint conditions altered invariant/correct-control behavior: invariant accuracy was `.9833/.9500` for V/joint, and correct-control preservation was `.5333/.4667`.
- Final verdict is `GENERIC_PERTURBATION`, `mechanism_allowed=false`, `stage_3_run=false`. All 15 cases classify as `GENERIC_STATE_PERTURBATION`; matched rescue cannot be interpreted as an identifiable address–payload failure. No repair or further K/V intervention was run or authorized.

## 2026-09-04 17:08 CST: Stage 2F Key-Guided Recurrent Refinement Pilot — No-Loop-Repair Stop

- User requested the frozen donor-free, training-free Stage 2F pilot. Added `experiments/qwen35_key_guided_loop_pilot.py` and independent outputs under `outputs/qwen35_key_guided_loop_pilot/`; used all 15 frozen Stage-2E pure errors, all 15 matched-correct controls, and 30 result-blind additional clean controls. Conditions were fixed to L19/L23 critical, L7 control, KEY-REFINE T2/T3, eta=.5; no layer search, donor, cache append, or training was run.
- `implementation_audit.json` passed: T=1 candidate/attention-hidden/layer-output errors were all `0.0`, predictions exact, sequence length/mask/position/mRoPE unchanged, no cache entries or duplicate KV append, and KEY Q/V plus VALUE Q/K freeze checks passed. Formal rows are complete and unique: `error_rows.csv` 105, `correct_control_rows.csv` 315, `atomic_control_rows.csv` 420; natural baselines reproduce exactly and all rows use full-sequence/no-cache scoring.
- Phase A results: every condition had `LRR=0/15` and negative mean margin change. L19-T2/T3 mean ΔM `-0.091667/-0.145833`, CPR `.955556/.911111`; L23-T2/T3 `-0.087500/-0.100000`, CPR `.977778/.977778`; L7 controls `-0.187500/-0.200000`, CPR `.911111/.866667`. Atomic readouts were `1.0` accuracy with `0` flip rate for every condition. No critical condition passed all gates; Phase B was correctly not run.
- Auxiliary `key_refinement_diagnostics.csv`/`key_refinement_diagnostics_summary.json` were completed in a separate result-blind pass over the same 15 errors after checkpoint resume: 18 batch-round rows across all six conditions, K Frobenius change `695.319–1533.566`, cosine-to-K0 `.929632–.982810`, Q/V exact in every record. No object-mask attention mass was computed because no Stage-2E mask existed and new probes were forbidden.
- Final verdict: **`MECHANISM_GUIDED_LOOP_NOT_SUPPORTED`**, case D / secondary `NO_LOOP_REPAIR`; `phaseA_positive=false`, `phaseB_run=false`, `no_training_run=true`. The pilot does not support simple recurrent key refinement reproducing Stage-2E K-donor rescue; stop here and do not proceed to loop-aware training or tuning without a new protocol. Final report: `outputs/qwen35_key_guided_loop_pilot/REPORT.md`.

## 2026-09-04 22:55 CST: Stage 2G-A Partitioned-K Rescue Audit — Visual-K-Mediated Stop

- User requested the frozen causal decomposition of the Stage-2E whole-prefix K rescue. Added `experiments/qwen35_partitioned_k_rescue.py` and independent outputs under `outputs/qwen35_partitioned_k_rescue/`; used exactly the 15 Stage-2E stable pure errors, 15 matched-correct controls, and frozen D+/D− donor aliases. No new data, probes, layer/head search, repair, or training was run.
- Processor/layout-only partition audit passed: prefix boundary `201`; `VISUAL_PREFIX=[3,201)` (198 tokens, including the formal vision span delimiters) and `TEXT_PREFIX=[0,3)` (3 tokens), disjoint and exactly covering the Stage-2E K prefix. Implementation gates passed: T0 candidate-score error `0.0`; K-ALL+ prediction agreement `15/15`; K-ALL+ historical rescue `13/15`; composition, cache reconstruction, component exactness, and immutable snapshot checks passed.
- Formal causal coverage passed with batch size 1: `error_rows.csv` `75`, `correct_control_rows.csv` `75`, and `atomic_control_rows.csv` `180`, all in `RECURRENT_REFERENCE`. Error rescue: K-ALL+ `13/15`, K-VIS+ `13/15`, K-TEXT+ `0/15`, K-VIS− `0/15`; mean ΔM was `.720833` for K-VIS+, `0` for K-TEXT+, and `-6.070833` for K-VIS−. Visual recovery ratio was `1.0` (pooled `1.0`, all `15` eligible families).
- Safety/readouts: K-VIS+ CPR `1.0` with one flip-versus-T0 and mean correct-answer margin change `.654167`; K-TEXT+ CPR `.933333` with zero flips and zero mean margin change. K-VIS+ atomic accuracy `1.0`, flip rate `0`; K-VIS− atomic accuracy `.633333`, flip rate `.366667`; the visual atomic gate passed.
- Final verdict is **`VISUAL_K_MEDIATED_RESCUE`**, `mechanism_allowed=false`, `repair_run=false`, `next_stage_run=false`. The result supports only that the established Stage-2E matched-K rescue is mainly reproducible by visual-prefix K with address specificity; it does not establish endogenous visual-K corruption or a root cause. Report: `outputs/qwen35_partitioned_k_rescue/REPORT.md`.

## 2026-09-05 00:23 CST: Stage 2G-B Bidirectional Visual-K Error Transfer — Nontransferable Stop

- User requested the final frozen reverse audit. Added `experiments/qwen35_reverse_visual_k_transfer.py` and independent outputs under `outputs/qwen35_reverse_visual_k_transfer/`; used all 15 Stage-2E E/C pairs, `RECURRENT_REFERENCE`, boundary `201`, visual K `[3,201)`, all eight full-attention layers, and only E0/E-RESCUE/C0/C-ERRK plus the four existing atomic queries. No repair, training, search, donor expansion, or new mechanism branch ran.
- Commands completed: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/qwen35_reverse_visual_k_transfer.py --stage prepare`, `--stage implementation`, `--stage causal`, and offline `--stage analyze`; the CUDA stages ran only after the implementation gate passed.
- Source-pair, natural-phenotype, and implementation audits passed: 15/15 pair checks; 15/15 natural errors were opposite-slot; E0 exact 15/15; E-RESCUE exact 15/15 with historical 13/15 rescue; C0 exact/correct 15/15; all 60 component rows, 15 position/RoPE rows, and 16 cache reconstructions passed.
- C-ERRK (natural-error visual K into clean C) remained correct `15/15`: EIR/WSIR/OPCR `0/15`, mean ΔM `+0.0333335`, median `+0.0624981`, family-bootstrap 95% CI `[-0.0833334,+0.15]`, MRR `0/15`. All four atomic readouts remained 1.0 accuracy with zero flips for both C0 and C-ERRK (P2I/P2C each 1.0). C-ERRK matched E's lexical wrong token `15/15`, but that token was C's own correct answer in every pair; it never induced an error or recipient-opposite-slot output.
- Final verdict: **`NONTRANSFERABLE_ERROR_STATE`**, `mechanism_allowed=false`, stop=true. The visual-K rescue surface is corrective but natural-error visual K is not sufficient to transfer the failure to a clean matched-address recipient; no endogenous K/root-cause or repair implication follows. Analysis had one runner-only report-field typo, fixed without rerunning causal rows. Report: `outputs/qwen35_reverse_visual_k_transfer/REPORT.md`.

## 2026-09-05 13:58 CST: Stage 3A Counterfactual Binding Routing Calibration Pilot — Phase 0 Stop

- User requested the repair-before-repair feasibility pilot: test only context-dependent Q–visual-K compatibility, with no donor K/V, natural-error supervision, red-yellow training/validation, backbone update, layer/head search, or tuning. Added `experiments/qwen35_counterfactual_routing_pilot.py` and independent outputs under `outputs/qwen35_counterfactual_routing_pilot/`.
- Ran `--stage prepare`, a resumable durable CUDA `--stage phase0`, offline `--stage report`, and final `py_compile`. Phase 0 completed all `4096` rows: 256 complete families, 241 behavior-clean and 15 frozen Stage2E errors, using raw pre-softmax QK, direct renderer object masks, A/B identity→color query rows, and all eight full-attention layers. No adapter training or checkpoint selection ran.
- Error `L_struct` mean/median was `.0631121/.0292669`; clean was `.0688534/.0481444`. Error-minus-clean mean was `-.00574133`, family-bootstrap 95% CI `[-.0350825,.0306327]`; only `6/15=.4` errors exceeded the clean median. `routing_defect_signal=false`, so the preregistered stop rule passed as `NO_ROUTING_DEFECT_SIGNAL`.
- Required audit/manifests/report artifacts are present, including `phase0_routing_audit.csv`, `phase0_summary.json`, `data_manifest.csv`, `split_manifest.csv`, and `REPORT.md`; fixed training split remains 167/42 with zero red-yellow families in train/validation. Checkpoint directory is intentionally empty because Phase 0 blocked training.
- Final verdict: **`NO_ROUTING_REPAIR`**, Case D, `mechanism_allowed=false`, `training_run=false`. This does not support an error-associated Q–visual-K routing defect in this pilot; stop the branch. Do not train STRUCTURAL or GROUNDING_ONLY adapters, expand the benchmark, or tune rank/lr/layer/head settings without a new protocol.

## 2026-09-05 14:03 CST: Stage 3A Full Results Report Expanded

- User requested the complete result report. Expanded `outputs/qwen35_counterfactual_routing_pilot/REPORT.md` without rerunning the model or changing any result: added the exact Phase 0 protocol, data/exclusion topology, component/query/layer summaries, all 15 error-family values, direct answers 1–10, and explicit not-applicable status for blocked Phase 1 metrics.
- Confirmed the recorded result remains `NO_ROUTING_REPAIR` / Case D: error mean/median `.0631121/.0292669` versus clean `.0688534/.0481444`, error-minus-clean bootstrap CI `[-.0350825,.0306327]`, and `6/15` errors above the clean median. No adapter or checkpoint was run.

## 2026-09-05 CST: Repair Direction and CVPR 2027 Evidence Review

- User requested project review, the most promising repair, and a CVPR 2027 strategy. Read current Stage2E/2F/2G/3A reports, earlier LBI/OCBW and spatial-flow results, and relevant primary literature. No model forward, training, intervention, protocol change, or old-output mutation ran; only this repository note was appended.
- Read-only Python/CSV recomputation of all 6,144 Stage2E behavior rows: 1,024 scenes, 2,048 identity-color queries; direct answers correct 2,031/2,048. All 17 errors are yellow→red (A=3, B=14); the 15 selected stable errors likewise are all yellow→red (A=2, B=13).
- Deterministically joined cached predicted Position→Identity with cached predicted Position→Color, matching the identity parsed from the original question. No gold labels entered answer construction; gold used only for scoring. Composed accuracy was 2,048/2,048, ambiguous matches 0, corrected direct errors 17, broken correct answers 0. This is post-hoc offline recombination on historical candidate-scored data, not a new inference/repair trial, independent test, or proof of an endogenous compositional mechanism.
- Recommended research direction: first compare simple color/calibration baselines with explicit atomic-answer composition; if fresh family/color/template tests support it, investigate distilling compositional consistency into direct answers with equal-data SFT and equal-compute inference controls. Do not restart frozen K/routing/BO branches. Stage3A's null concerns the measured aggregate routing observable; uniform slot attention also satisfies its invariance/equivariance losses, so the metric alone cannot identify correct grounding.
- CVPR strategy must address narrow yellow/red phenotype, historical-test reuse, small error count, prior object-slot nulls, cross-architecture and natural-image transfer, and overlap with VISER (NeurIPS 2025) and Binding Visual Features Point by Point (arXiv:2605.25427). Official CVPR 2027 Dates page checked: registration Nov 10, 2026; paper Nov 16; supplementary Nov 23 (AoE); CFP text not yet available. This entry records analysis/proposals only, not experiment authorization.

## 2026-09-07 16:10 CST: Phase 1 Executable Atomic Composition Reproduction Complete

- User requested the frozen behavior-level reproduction of direct Identity→Color versus deterministic composition from fresh Position→Identity and Position→Color readouts. Added `experiments/qwen35_executable_atomic_composition.py` and `outputs/qwen35_executable_atomic_composition/`; added an optional in-memory image-embedding override to `experiments/qwen35_kv_gdn_factorial_pilot.py` so the unchanged per-query recurrent scorer could reuse one scene visual prefix without changing formal query execution.
- Ran `py_compile`, `--stage prepare`, the complete CUDA `--stage execute` in durable sessions, and `--stage analyze`. Fresh coverage passed: 2,048 direct + 4,096 atomic rows, 1,024 scenes, 256 families; all model forwards and outer jobs batch size 1, `RECURRENT_REFERENCE`, no ground truth during execution, and no historical read before completion. A resumed-run prefix-audit merge issue was repaired by rebuilding the lightweight audit from complete fresh-row metadata; no formal rows were rerun for that audit repair.
- Final fresh results: direct `2030/2048=.991211`; P2I `2048/2048=1.0`; P2C `2048/2048=1.0`; AtomicComplete `2048/2048=1.0`; hard executable composition `2048/2048=1.0`; Δ=`.008789`. There were 18 direct errors, all 18 were repaired, 0 harms, 18 disagreements, and 0 invalid bindings. CUG=`18/2048=.008789`; DER=`18/18=1.0`; soft composition accuracy was `1.0` as descriptive secondary analysis.
- Composition purity passed: all 2,048 rows recomputed exactly from `I_hat_L/I_hat_R/C_hat_L/C_hat_R` and query identity, `composition_oracle_usage=false`, no ground-truth or direct-answer input, and no fallback. All 18 fresh direct errors were `OPPOSITE_SLOT` and yellow→red; all were stable across run0/run1/run2. Atomic rechecks on the 18 error scenes all agreed.
- Historical audit read `outputs/qwen35_failure_interface_pilot/behavior_all_rows.csv` only after fresh completion. Historical direct errors were 17 versus fresh 18; yellow→red was 17 versus fresh 18; direct prediction agreement was `2047/2048`, atomic prediction agreement `4096/4096`, and deterministic historical composition reconstruction was `2048/2048`, agreeing with fresh composition `2048/2048`.
- Family bootstrap used 10,000 draws: direct mean `.991263`, 95% CI `[.986816,.995117]`; composition mean `1.0`, CI `[1.0,1.0]`; Δ mean `.008737`, CI `[.004883,.013184]`; repair−harm observed `18`, CI `[10,27]`. Final verdict: **`EXECUTABLE_COMPOSITION_GAP_REPRODUCED`**, with `mechanism_allowed=false`, `next_stage_allowed=false`; stop Phase 1 and require an explicitly new protocol before any composition-load experiment. The P2I post-execution evaluator maps frozen shape labels to the row's semantic `A/B` space; this was corrected before the final analysis without new model forwards.

## 2026-09-09 CST: RealBind Phase 2 Dataset Builder Implemented; Raw-Data Gate Stop

- User requested deterministic RealBind Phase 2 dataset construction only: VAW val/test instance colors, Visual Genome v1.4 object uniqueness, VG images, 100-family dry-run, fixed ten QA templates, strict audit, local HTML viewer, and reports; no synthetic images, LLM question generation, or VLM inference.
- Added `experiments/realbind_phase2/{README.md,scripts/common.py,scripts/inspect_data.py,scripts/build_candidates.py,scripts/sample_families.py,scripts/generate_qa.py,scripts/audit_dataset.py,scripts/build_audit_html.py}`. The pipeline uses only pathlib-based paths, seed 42, explicit gate statistics and one-gate relaxation counts, conservative object normalization/optional official alias metadata, raw deterministic rebuild hashes, and non-zero failure exits.
- Ran `python experiments/realbind_phase2/scripts/inspect_data.py --data-root /mnt/d/datasets/RealBind`: `/mnt/d` accessible, free disk `72.00 GiB`; `/mnt/d/datasets/RealBind` had no VAW val/test files, VG image folders, or VG objects annotation. The report is `/mnt/d/datasets/RealBind/realbind/reports/missing_file_report.json`.
- Ran `python experiments/realbind_phase2/scripts/build_candidates.py --data-root /mnt/d/datasets/RealBind`; input resolution failed explicitly with no candidate output. Ran the strict audit, which wrote `/mnt/d/datasets/RealBind/realbind/audit/dryrun_100_audit.json`, `dryrun_100_audit_schema.{jsonl,csv}`, `deterministic_rebuild_checks.jsonl`, and `reports/dryrun_report.{json,md}` with `passed=false` and four missing-input failures.
- `py_compile`, all six CLI `--help` checks, the deterministic ten-QA template smoke test, and the 100-row in-memory sampling smoke test passed. The actual 100-family dry-run did not run because the required raw data are absent; no `all_candidates.jsonl`, `dryrun_100.jsonl`, `dryrun_100_qa.jsonl`, or HTML viewer was fabricated. Do not provide/run the 2000-family release command until raw inputs are installed and the 100-family audit passes.

## 2026-09-10 CST: RealBind Phase 2 Raw Data Download and Dry-Run Complete

- User requested the actual RealBind Phase 2 raw-data acquisition on D: followed by validation and the existing dry-run pipeline; no VLM inference, synthetic images, or LLM question generation was run. `/mnt/d` was writable with `77,309,460,480` bytes free before downloads and `29,819,957,248` bytes free after. Raw `vaw+vg+downloads` usage is `46,959,044,260` bytes; `realbind/` derived outputs use `168,071,327` bytes. Downloads retain Range parts, partial/test files, and verification logs.
- Cloned `https://github.com/adobe-research/vaw_dataset` at `/mnt/d/datasets/RealBind/vaw`, commit `3a4ebffc5e8194d19a77f51d846c2cd47c477d13`. Actual evaluation files are `data/val.json` (11,590,564 bytes, 12,286 records) and `data/test.json` (29,992,834 bytes, 31,819 records); train files were not used. The real schema includes all six required fields plus `instance_polygon`; a fixed 1,000-record parse test had zero failures. Full normalization recorded one malformed test record with empty `object_name` (`record_index=3667`) and preserved it in the report rather than fabricating a candidate.
- Downloaded official Visual Genome image archives `images.zip` (`9,731,705,982` bytes) and `images2.zip` (`5,471,658,058` bytes) from the official Stanford Visual Genome links, plus `objects.json.zip` (`55,323,929` bytes) and `image_data.json.zip` (`1,780,854` bytes) from the official Visual Genome data directory. All four `unzip -t` checks passed and logs are under `/mnt/d/datasets/RealBind/downloads/`. Extracted image counts are `VG_100K=64,346` and `VG_100K_2=43,903`; `objects.json` and `image_data.json` each load as 108,077 records. A seed-42 sample of 100 VAW val/test IDs mapped exactly once to VG images (`100/100`, missing 0, duplicate 0).
- Added the resumable local helper `experiments/realbind_phase2/scripts/download_vg_ranges.sh` for the large archives. Updated the existing RealBind resolver/config paths to discover `/mnt/d/datasets/RealBind/vaw`, `/mnt/d/datasets/RealBind/vg/images`, `/mnt/d/datasets/RealBind/vg/annotations/objects.json`, and `/mnt/d/datasets/RealBind/vg/annotations/image_data.json`; no raw files were renamed or copied into the repository.
- Ran `inspect_data.py` successfully, then `build_candidates.py`, `sample_families.py --count 100 --seed 42`, `generate_qa.py`, `audit_dataset.py`, and `build_audit_html.py`. Candidate extraction produced 2,008 geometry-passing pairs, 794 uniqueness-eligible pairs, 476 unique candidate images, and cross-negative counts strict/single/none `113/492/1403`; dry-run sampling selected 100 unique images with `34 strict/66 single/0 none`, balanced `left_is_A` and candidate option order `50/50`, and max unordered object-pair repeat 1. QA output has 1,000 rows, exactly 10 per family, deterministic-template-only and no LLM use.
- Strict audit passed with `failures=0`, including image/bbox/geometry/uniqueness/QA/composition/reverse-option and deterministic raw-rebuild assertions. Reports and outputs are under `/mnt/d/datasets/RealBind/realbind/`, including `candidates/all_candidates.jsonl`, `families/dryrun_100.jsonl`, `qa/dryrun_100_qa.jsonl`, `audit/dryrun_100.html`, `audit/dryrun_100_audit.json`, and `reports/dryrun_report.{json,md}`. The full 2,000-family target is not available: largest sequential rejection gate is `object_identity` (`5,713`); among one-gate relaxation diagnostics, `x_separation` would add 290 unique images and `area_threshold` 190. Do not lower gates or run full 2,000 inference without the user's decision.

## 2026-09-11 CST: Strict RealBind Atomic-Composition Baseline Complete

- User requested the frozen 100-family RealBind atomic-composition baseline only: no dataset edits, new images, prompt repair, CoT/decomposition, training, or intervention. Added `experiments/realbind_phase2/atomic_composition_baseline.py`; outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/atomic_composition_baseline_qwen35_4b/`.
- Read and audited `common.expected_qa`/`generate_qa.py` semantics before inference. The ten types are eight atomic mappings (`P2I_LEFT/RIGHT`, `P2C_LEFT/RIGHT`, `I2P_A/B`, `I2C_A/B`) plus `COMPOSITION_FORWARD/REVERSED`; both composition types have the exact same minimal dependency `I2C_A -> I2C_B` (color(A), color(B)). Preflight passed: 100 families, 1,000 QA, exactly 10/family, existing audit PASS, all images/GT/candidate pairs valid, 100/100 dependency coverage, no missing/ambiguous/circular/equivalent-final dependency or leakage.
- Commands: formal `--stage audit`; CUDA `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/realbind_phase2/atomic_composition_baseline.py --stage infer`; offline `--stage analyze`. Used the project Qwen3.5-4B loader/template, one independent batch-1 query per QA, greedy `do_sample=false`, no answer/context sharing. Inference completed 1,000/1,000 rows. A scorer implementation issue was corrected offline from saved raw responses: unique canonical candidate mentions are valid regardless of explanatory prefix; conflicting/unmapped responses remain invalid. No model forward was rerun.
- Final results: atomic pooled accuracy `.365000` (292/800); `I2C_A=.71`, `I2C_B=.75`. Composition raw accuracy `.020000` (4/200), both forward/reversed `.02`. There were 108 composition rows with both required atoms correct and 104 pure failures, giving `PCFR=.962963`; `P(composition correct | all atoms correct)=.037037`. Family bootstrap seed 42, 10,000 replicates: 95% CI `[.9134615385, 1.0]`, stable. Pure failures by type: forward 52, reversed 52.
- Important caveat: all 104 pure failures have `invalid_prediction=true` for the final composition response; valid-but-wrong canonical composition labels are `0`. Composition invalid rate is 196/200. Under the user-specified rule that invalid counts as incorrect, the fixed verdict is **`REAL-WORLD_ATOMIC_COMPOSITION_GAP_ESTABLISHED`**, but the result is not clean evidence of 104 valid wrong A/B answers and should be interpreted with this output-format caveat. `pure_composition_failures.jsonl` contains all 104 cases and the HTML page shows only their original images.

## 2026-09-11 CST: Max64 use_cache Gate Complete; Formal False-Cache Run Started

- User requested pausing the prior max64 full run, fixing the first 50 frozen QA rows (40 atomic / 10 composition; all ten QA types, five each), and comparing one exact-prompt `use_cache=False` pass with one `use_cache=True` pass at `max_new_tokens=64`, `do_sample=False`, `batch_size=1`. Added `experiments/realbind_phase2/cache_gate.py`; parameterized `experiments/realbind_phase2/atomic_composition_baseline.py` so its default remains `use_cache=False` and explicit `--use-cache` is the only formal change.
- Both cache-gate modes used the project Qwen3.5-4B loader, standard render/encode path, independent current-image/current-question queries, and the deterministic scorer. Outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/atomic_composition_baseline_qwen35_4b_max64/`: `cache_gate_manifest.json`, both complete 50-row JSONL files, `cache_gate_comparison.jsonl`, `cache_gate_summary.json`, and `cache_gate_report.md`. The earlier paused 10-row full-run partial was preserved at `paused_before_cache_gate/inference_results.partial.jsonl` and was not mixed into the formal run.
- Cache gate results: exact generated-token-ID agreement `0/50=0.000000`; raw-response agreement `0/50`; normalized-answer agreement `29/50=.580000`; correctness agreement `33/50=.660000`; generated-length agreement `40/50=.800000`. `use_cache=False` had accuracy `.22` (11/50), invalid 36, mean length 63.5, hit-max 46/50; `use_cache=True` had accuracy `.16` (8/50), invalid 39, mean length 56.52, hit-max 43/50. Because normalized answers mismatched on 21 rows, the formal choice is strictly `use_cache=False`; no A/B instruction control was run.
- After the gate, started the fresh formal command in durable CUDA session `realbind_formal_max64_false`: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/realbind_phase2/atomic_composition_baseline.py --stage infer --max-new-tokens 64 --output-root /mnt/d/datasets/RealBind/realbind/experiments/atomic_composition_baseline_qwen35_4b_max64`. Preflight passed and the fresh formal run reached 10/1000 at this entry; analysis has not yet run. The run is expected to be long because false-cache decoding recomputes token states; resume from its checkpoint if interrupted, keeping `use_cache=False`.

## 2026-09-11 CST: Cache-Gate Divergence Diagnosis

- Read-only analysis of `cache_gate_comparison.jsonl`: all 50 first generated tokens agree, but full token-ID agreement is `0/50`; longest common prefixes are only 1–4 tokens (mean 1.82). Normalized agreement is `29/50=.58`, but 27 of those 29 are `None/None` invalid pairs and only 2 are same valid answers. The 21 normalized mismatches are exactly validity flips: 12 false-valid/true-invalid and 9 false-invalid/true-valid; no valid-vs-valid different-answer pair was observed.
- The local Qwen3.5 implementation is hybrid. With cached generation, after prefill it uses cached single-token `causal_conv1d_update` plus `recurrent_gated_delta_rule`; without cache it repeatedly runs the multi-token convolution plus `chunk_gated_delta_rule`. Both are intended to represent the same computation, but they are distinct stateful numerical paths under the frozen bf16/fallback runtime. Raw true-cache responses include malformed/gibberish continuations, including rows where false-cache gives a coherent valid candidate; this is a cache-path compatibility/runtime divergence, not a scorer-only effect. No protocol or formal output was changed.

## 2026-09-11 CST: Max64 Invalid-Prediction Cause Clarified

- Read-only gate audit showed max length is not synonymous with validity. In `use_cache=False`, 36/50 rows were invalid, including 3/36 that stopped before the 64-token cap (lengths 61, 53, 54); all three were invalid because the explanatory response mentioned both candidates. Example: an `I2C_B` response clearly says `blue`, then says `not white`, and ends with `Final Answer: Blue`; strict deterministic parsing still sees two candidate mentions and records `zero_or_multiple_candidate_mentions` without manual repair. The remaining false-cache invalid rows hit max (33/36), but hitting max is only a termination statistic, not a unique-answer guarantee.
- In `use_cache=True`, 39/50 invalid rows included 32/39 at the cap and 7/39 early malformed/ambiguous responses. Therefore increasing `max_new_tokens` can reduce genuine truncation but cannot force a final answer format, prevent alternative-candidate mentions, or repair cache-path gibberish. No protocol or output was changed.

## 2026-09-11 CST: RealBind A/B Direct-Choice Baseline Complete

- User requested a fresh strict RealBind 100-family/1,000-QA baseline with the frozen images, question semantics, candidate contents, and option order unchanged; every independent prompt received only the suffix `Answer with only A or B. Do not provide any explanation.` Added the A/B direct protocol to `experiments/realbind_phase2/atomic_composition_baseline.py`; no dataset, image, model, or prior output was modified.
- Preflight passed all gates: 100 families, 1,000 QA, exactly 10 per family, existing audit PASS, valid images/GT/candidate pairs, and complete non-circular dependency mapping. Formal CUDA inference used the project Qwen3.5-4B loader, batch size 1, greedy decoding, `do_sample=False`, `use_cache=False`, `max_new_tokens=4`; outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/atomic_composition_baseline_qwen35_4b_ab_direct_max4/`.
- Final results: atomic accuracy `486/800=.6075`; composition accuracy `189/200=.945`; 102 compositions had all required atoms correct and 3 were pure composition failures, so pooled PCFR=`3/102=.029412`, conditional composition correctness=`.970588`. Family-cluster bootstrap (seed 42, 10,000 replicates) 95% CI for PCFR is `[0,.065217]`. Pure failures were forward 2 and reversed 1.
- Format violations were `2/1,000=.002`; both were short outputs (`A. a blue`, `A. a white`) on composition-forward rows, not long generations. No constrained-scoring fallback was run. Generated length was 4 for all rows because the model emitted answer/newline/control tokens through the four-token budget; this termination statistic was reported separately from semantic A/B validity.
- Final verdict: **`REAL-WORLD_ATOMIC_COMPOSITION_GAP_WEAK`**. The effect exists in 3 cases but is too small and partly format-related for a stable claim; one of the three was a valid-but-wrong A/B answer and two were format violations counted incorrect under protocol. Final artifacts include `inference_results.jsonl`, `composition_analysis.jsonl`, `pure_composition_failures.jsonl`, `summary.{json,md}`, and `pure_composition_failures.html`.

## 2026-09-11 CST: RealBind Pair-Conditioned Direct QA v1 Audit Stop

- User requested a new inference-free seven-QA-per-family protocol using the frozen RealBind 100 families, with no raw-data/candidate/family/old-QA changes, no images, and no VLM inference. Added `experiments/realbind_phase2/scripts/generate_pair_direct_qa.py`, which deterministically generates `I2C_A`, `I2C_B`, `PAIR_REL_A_TO_B`, `REL2I_LEFT`, `REL2I_RIGHT`, `REL2C_LEFT`, and `REL2C_RIGHT`, writes minimum/strict dependency metadata, performs strict audit, and builds an original-image-only HTML page.
- Commands run: `py_compile`; `python .../generate_pair_direct_qa.py --stage all --data-root /mnt/d/datasets/RealBind` (generation and audit); then `--stage audit` and `--stage html` after classifying the audit issue. No model, torch, transformer, inference, candidate extraction, family sampling, or old QA generation was run.
- Exact outputs: 100 families, 700 QA rows, 100 of each new type; levels `atomic_attribute=200`, `atomic_relation=100`, `relational_readout=200`, `composition=200`; `left_is_A=True/False=50/50`; all prompts are pair-conditioned, contain no candidates/options/A-B labels, and all dependency mappings are same-family and acyclic. Frozen family/candidate/existing-audit hashes remain `64bb9019...751e3` / `3bd41310...d3e0c` / `4efc37ff...986c1`; new QA SHA256 is `81149908...c8a0`.
- Strict audit stopped with one genuine canonical-identity/template collision: family `realbind_test_2346035_2346035016_2346035001` uses object identity `object`, producing `Between the couch and the object, what color is the object on the left?`. The canonical identity collides with the required generic phrase `the object on the requested side`; no renaming, template relaxation, exclusion, or resampling was performed. Audit status is **`REALBIND_PAIR_DIRECT_V1_INVALID`** under the requested stop rule. The diagnostic HTML contains all 100 families and 700 QA rows, original images only, with no bbox overlays.
- Outputs: `/mnt/d/datasets/RealBind/realbind/qa/dryrun_100_pair_direct_v1.jsonl`, `pair_direct_v1_dependency_spec.json`, `/mnt/d/datasets/RealBind/realbind/audit/dryrun_100_pair_direct_v1_audit.json`, `dryrun_100_pair_direct_v1.html`, and `/mnt/d/datasets/RealBind/realbind/reports/dryrun_100_pair_direct_v1_report.md`. Do not run inference on this release unless the user authorizes a new decision for the `object` identity collision.

## 2026-09-11 CST: RealBind Pair-Conditioned Direct QA v1.1 Ready

- User requested an inference-free v1.1 protocol correction: replace only generic pair-member template references with `which one` / `the one`, add a separate conservative referential-identity audit, replace the known selected family containing canonical identity `object`, and preserve raw VAW/VG data, candidate extraction, geometry gates, images, and all old outputs. Added `experiments/realbind_phase2/scripts/generate_pair_direct_qa_v1_1.py`; no model, torch, transformer, VLM, API, or image generation was run.
- Read-only pool audit scanned 794 existing eligible candidate pairs and found 4 generic-denylist candidate rows (the final selected invalid family was one of them). The deterministic replacement filtered unchanged gates, referential denylist, unused image/pair, and `left_is_A=False`, then ranked by source-split match, cross-negative priority distance, geometry distance, and candidate ID. It selected `realbind_test_2384340_2384340003_2384340005` (`floor`/`pillar`, `test`, `single` cross-negative); no unused strict cross-negative candidate existed. The removed family was `realbind_test_2346035_2346035016_2346035001` because of `generic_canonical_identity` (`object`).
- Commands: `python -m py_compile experiments/realbind_phase2/scripts/generate_pair_direct_qa_v1_1.py`; CUDA/model/inference commands were not run; `python -u experiments/realbind_phase2/scripts/generate_pair_direct_qa_v1_1.py --stage all --data-root /mnt/d/datasets/RealBind`. Output files are independent v1.1 artifacts under `/mnt/d/datasets/RealBind/realbind/`.
- Final audit passed: 100 families, 700 QA rows, exactly 100 of each type; levels `atomic_attribute=200`, `atomic_relation=100`, `relational_readout=200`, `composition=200`; `left_is_A=True/False=50/50`; annotation/referential/template/dependency/geometry/pair-conditioning/answer-leakage violations all 0; explicit dependency graph edges 1000 and cycles 0. New family SHA256 is `b8d366c296403646a8c9a71c09cf8dd8ba007eea2d1d7a3c930cd95edd1816ee`; new QA SHA256 is `433ccc24f7a7070ae56f756e8936f10c181105e6fa31b3fa9bf206fcf06c3c87`. Verdict: **`REALBIND_PAIR_DIRECT_V1_1_READY`**. Do not run inference until separately authorized.

## 2026-09-11 CST: RealBind Pair-Direct v1.1 Qwen3.5-4B Baseline Complete

- User requested the formal measurement-only baseline on the frozen v1.1 100-family/700-QA release. Added `experiments/realbind_phase2/pair_direct_v1_1_baseline.py`; no dataset, family, QA, image, prompt, model, or intervention was changed. Used the existing Qwen3.5-4B loader/renderer, independent full-image batch-1 queries, greedy `do_sample=False`, `use_cache=False`, `max_new_tokens=16`.
- Preflight and 28-row smoke gate passed. Formal CUDA inference completed 700/700 unique rows; all 7 QA types were 100 each, all rows recorded raw response/token IDs/length/EOS/scoring fields, and all 700 had EOS with 0 hit-max and 0 semantic truncation. A scorer vocabulary correction was applied offline to saved raw generations only: canonical colors outside the current pair are valid semantic wrong answers, while identity uses canonical identities in the frozen eval families; no model forward was rerun.
- Final per-type accuracy: `I2C_A=.66`, `I2C_B=.73`, `PAIR_REL=.70`, `REL2I_LEFT=.76`, `REL2I_RIGHT=.80`, `REL2C_LEFT=.46`, `REL2C_RIGHT=.52`; pooled composition `98/200=.49`. Corrected invalid/format-violation count is `11/700`; non-invalid semantic wrong rows are retained as wrong rather than invalid.
- Conditional results: `PCFR_min=32/100=.320000`, `P(composition correct | minimum-ready)=.680000`; `PCFR_strict=25/68=.367647`; `PCFR_path=28/88=.318182`. Family-cluster bootstrap used seed 42 and 10,000 draws: CIs `[.239130,.401869]`, `[.272727,.464286]`, and `[.230769,.405412]`, respectively; zero-denominator draws were 0. Strong-path failures were 28 and matched strong-path successes 60.
- The result is not dominated by one checked composition template/side/left-is-A/cross-negative stratum (largest checked failure share `.59375`); both `REL2C_LEFT` and `REL2C_RIGHT` contribute failures, and 30 of 32 minimum failures are valid semantic wrong answers. Final verdict: **`REALBIND_DIRECT_COMPOSITION_GAP_ESTABLISHED`**. This is behavioral availability/conditional-failure evidence, not evidence that the model internally executes a specific symbolic program.
- Outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/pair_direct_v1_1_qwen35_4b_baseline/`: `preflight.json`, `protocol.json`, `smoke_test_results.jsonl`, `smoke_test_report.md`, `inference_results.jsonl`, `composition_analysis.jsonl`, `pure_failures_min.jsonl`, `pure_failures_strict.jsonl`, `strong_path_failures.jsonl`, `strong_path_successes.jsonl`, `summary.{json,md}`, `failure_audit.html`, `reproducibility.json`, and `runtime_metadata.json`. Input hashes recorded: family `b8d366c2...1816ee`, QA `433ccc24...f06c3c87`, candidates `3bd41310...d3e0c`; final protocol SHA256 `90510c77...5be0efe8`.

## 2026-09-13 00:26 CST: DeepSeek RealBind Pair-Direct v1.1 Behavioral Runner Implemented And Dry-Run Complete

- User requested a DeepSeek multimodal cross-model behavioral baseline for the frozen pair-direct v1.1 inputs, with strict runtime-only credential handling, no API key supplied, no real API inference, default dry-run, one independent request per QA, resumable checkpoints, shared Qwen semantic scoring, PCFR analysis, and secret-safety auditing.
- Added `experiments/realbind_phase2/deepseek_api/{__init__.py,config.py,client.py,image.py,scorer.py,analyze.py,secret_safety.py,run_baseline.py}`. The runner uses the OpenAI-compatible DeepSeek Chat Completions shape with the default model identifier centralized as `deepseek-v4-flash-vision-exp`, fixed `thinking={type:disabled}`, temperature 0, max_tokens 16, frozen question text, and one original full image. The base64 backend is default; an optional in-process Files API backend is encapsulated but was not uploaded during dry-run.
- Dry-run commands only: `python -u experiments/realbind_phase2/deepseek_api/run_baseline.py --dry-run` (default output root) and a workspace-only optional uploaded-file backend dry-run. Both completed without creating a client or network request. Final output is `/mnt/d/datasets/RealBind/realbind/experiments/pair_direct_v1_1_deepseek_baseline/`; manifest has 700 unique rows, each QA type 100, 100 unique images, 600 cache hits, preflight/dependency/image/question checks PASS, network requests 0, and secret-safety PASS. Formal `inference_results.jsonl`/summary files were not fabricated.
- The shared scorer adapter reuses `pair_direct_v1_1_baseline.py` directly, preserving color/identity/left-right normalization, conservative invalid/format rules, and scoring revision `realbind_pair_direct_v1_1_semantic_vocab_v2`. `analyze.py` reuses the Qwen analysis/PCFR definitions and family bootstrap seed 42 with 10,000 replicates. Results save only final assistant content, normalized prediction, GT/correctness, latency, retry count, usage metadata, and protocol/model/QA hashes; reasoning content, request headers, client object, environment, and credential material are excluded.
- Input hashes after dry-run remain family `b8d366c296403646a8c9a71c09cf8dd8ba007eea2d1d7a3c930cd95edd1816ee`, QA `433ccc24f7a7070ae56f756e8936f10c181105e6fa31b3fa9bf206fcf06c3c87`, dependency `9238308038aa552f307b8cbb25058a373e0d979dd408656dacc82ace7e57cdf9`, audit `576eb3c4eb7da3f2a251aa804478434fdfd6514fe9553c0eaa8bfea49e460b8c`. No `--execute` command was run; formal inference remains a manual action outside Codex.

## 2026-09-13 15:25 CST: DeepSeek RealBind Pair-Direct v1.1 Baseline Analysis Complete

- User reported that the formal DeepSeek run had produced `summary.json` and requested analysis. Read-only analysis used the completed `/mnt/d/datasets/RealBind/realbind/experiments/pair_direct_v1_1_deepseek_baseline/` results and the same frozen Qwen baseline; no API request, key access, prompt/data/model change, or scorer change was made.
- DeepSeek coverage is complete: 700/700 unique QA rows, 100/100 families, all seven QA types at 100 each, `network_requests_sent=700`, base64 image cache 100 unique images/600 reuse hits, zero retries, and secret-safety audit PASS. Shared deterministic scorer and family bootstrap remain `realbind_pair_direct_v1_1_semantic_vocab_v2`, seed 42, 10,000 replicates.
- DeepSeek accuracy: `I2C_A=.58`, `I2C_B=.68`, `PAIR_REL=.58`, `REL2I_LEFT=.57`, `REL2I_RIGHT=.56`, `REL2C_LEFT=.34`, `REL2C_RIGHT=.39`; pooled groups are I2C `.63`, PAIR_REL `.58`, REL2I `.565`, composition `.365` (73/200). There are 19 invalid/format rows, including four truncated PAIR_REL responses; no internal reasoning content was saved.
- Conditional composition failure is established under the frozen rule: `PCFR_min=35/72=.486111` with family bootstrap 95% CI `[.371795,.602740]`; `PCFR_strict=19/42=.452381`, CI `[.305556,.600000]`; `PCFR_path=20/48=.416667`, CI `[.277778,.552632]`. Of the 35 minimum-ready failures, 34 are valid semantic wrong answers; 13 select the opposite object's color, 21 output another canonical color, and 1 is invalid.
- Compared with the same-QA Qwen3.5-4B baseline, DeepSeek is lower on every raw QA group (composition `.365` vs `.490`; I2C `.630` vs `.695`; PAIR_REL `.580` vs `.700`; REL2I `.565` vs `.780`) and has fewer minimum-ready composition cases (72 vs 100). Conditional composition failure is nevertheless higher for DeepSeek (`.486` vs `.320` at minimum, `.452` vs `.368` strict, `.417` vs `.318` path); these intervals overlap, so this is descriptive cross-model evidence rather than a claim of statistically significant model difference.
- Final interpretation: DeepSeek replicates the behavioral pattern of atomic/relational availability with a substantial conditional composition gap, but its lower prerequisite accuracy means the result is not a pure composition-only deficit. This remains behavioral evidence only and does not establish an internal symbolic relation-to-identity-to-color mechanism. Main artifacts: `summary.{json,md}`, `composition_analysis.jsonl`, `strong_path_failures.jsonl`, and `strong_path_successes.jsonl` in the DeepSeek output root.

## 2026-09-13 CST: DeepSeek Model-Routing And Protocol Audit

- User questioned whether the completed run used V4.1 and whether the low score could reflect a design issue. Read-only audit inspected `protocol.json`, `reproducibility.json`, all 700 result rows, and the runner/client source; no API request, key access, result rewrite, or code protocol change was made.
- The submitted/requested identifier is `deepseek-v4-flash-vision-exp` in the protocol and every result row, not the current canonical `deepseek-flash`. Official DeepSeek's 2026-09-10 release says V4.1-Flash is live as `deepseek-flash` and that the legacy `deepseek-v4-flash-vision-exp` temporarily routes to V4.1-Flash, so the Sep 13 run likely received V4.1-Flash through compatibility routing. However, the client discarded the API response `model` and `system_fingerprint`; saved `model_identifier` is only the CLI/request value, so the actual served backend cannot be proven retrospectively.
- The multimodal request shape is valid: `/chat/completions`, one user message, frozen question plus raw base64 `image_url`, no system/family context/options, and the official vision format. The scorer remains the shared Qwen deterministic scorer. No local image preprocessing defect was found.
- Two protocol limitations are material: `thinking={type:disabled}` intentionally evaluates direct/non-thinking mode rather than full reasoning capability; official docs state non-thinking mode fixes temperature at 1.0 and ignores the submitted `temperature=0.0`, so this is not guaranteed server-side greedy/deterministic decoding. `max_tokens=16` is too small for verbose direct answers: 4/700 PAIR_REL rows ended with `finish_reason=length`, producing truncated invalid responses. This affects a small number of rows but is a genuine protocol caveat.
- Audit conclusion: the result is valid only as a legacy-alias, non-thinking direct behavioral run; it should not be presented as a fully verified canonical V4.1/full-capability baseline. Before any new paid run, use the canonical `deepseek-flash` identifier, capture the non-secret response `model`/fingerprint, increase the output cap, and keep any thinking-enabled sanity run as a separately labeled protocol rather than mixing it with the current summary.

## 2026-09-13 CST: Qwen RealBind Thinking-Mode Status Verified

- User asked whether the Qwen3.5 comparison baseline enabled thinking. Read-only source/output audit confirmed that `qwen35_kv_gdn_factorial_pilot.render_chat()` passes `enable_thinking=False`, and the Qwen RealBind protocol records the same setting. Generation used `do_sample=False`, `use_cache=False`, batch size 1, and max_new_tokens 16.
- The Qwen chat template may still render an empty `<think>\n\n</think>\n\n` scaffold when thinking is disabled; this is template syntax, not generated reasoning. Saved Qwen responses contain short answer continuations and no reasoning field. Therefore both the Qwen baseline and the DeepSeek run were direct/non-thinking evaluations; a thinking-enabled DeepSeek result would not be a like-for-like comparison and must be separately labeled.

## 2026-09-13 CST: DeepSeek Thinking-Mode Runner Prepared And Dry-Run Passed

- User requested enabling DeepSeek thinking mode for a second manual RealBind test and asked whether it changes the output length budget. Updated the DeepSeek runner to expose explicit `--thinking` and `--max-tokens` flags, use `thinking.type=enabled`, record the mode/config in protocol, manifest, result, summary, and reproducibility metadata, and isolate thinking outputs under a separate default root. The canonical default model identifier is now `deepseek-flash`; legacy aliases remain command-line selectable.
- Added non-secret API response metadata capture for served model, system fingerprint, and response ID; reasoning content remains excluded from saved results. The existing direct/non-thinking output root is protected from accidental thinking-mode reuse. No frozen input was modified.
- Validation commands only: Python compilation; thinking dry-run with `deepseek-flash`, `max_tokens=4096`; legacy direct-mode compatibility dry-run with `deepseek-v4-flash-vision-exp`. Both validated 700/700 payloads, 100 images plus 600 cache hits, frozen questions, dependency checks, and secret-safety PASS; network requests sent 0 and no credential was requested.
- Protocol conclusion: DeepSeek `max_tokens` is the total completion budget in thinking mode, shared by reasoning and final visible answer. The new thinking dry-run records `max_tokens=4096`; if a formal row ends with `finish_reason=length`, rerun it as a new protocol with a larger cap rather than mixing checkpoints. Formal `--execute` was not run in Codex.

## 2026-09-13 CST: DeepSeek Thinking Baseline Offline Analysis

- User reported that the second thinking-enabled summary was complete. Read-only analysis inspected the frozen-compatible thinking output root; no API request, credential access, frozen-input change, scorer change, or result rewrite was performed.
- Coverage and protocol are valid: 700/700 unique QA rows, 200 composition analysis rows, server-reported model deepseek-flash on all 700 rows, one system fingerprint, thinking.type=enabled, max_tokens=4096, and secret-safety PASS. Reproducibility recorded 704 transport attempts because four requests used retries; this does not change the 700 successful QA count.
- Main metrics: I2C .645, PAIR_REL .660, REL2I .655, composition .330 (66/200). PCFR values are .444 minimum (40/90), .441 strict (30/68), and .406 path (28/69); verdict remains REALBIND_DIRECT_COMPOSITION_GAP_ESTABLISHED.
- The key caveat is budget truncation: 107/700 rows ended with finish_reason=length and empty final content, including 61/200 composition rows; all 700 rows had reasoning_content_present=true. On non-truncated rows only, the diagnostic composition accuracies were .420 left and .529 right, and diagnostic PCFR was .286 minimum, .283 strict, .212 path; these are not replacement protocol metrics but show that 4096 truncation materially depresses the aggregate.
- Compared with the prior non-thinking DeepSeek run, thinking improved I2C/PAIR_REL/REL2I raw accuracy (.645/.660 vs .630/.580) and conditional composition correctness, but raw composition was lower (.330 vs .365) because of truncation. Follow-up requires a separately fingerprinted larger-token thinking run (for example --max-tokens 8192); do not mix it with the 4096 checkpoint.

## 2026-09-13 CST: DeepSeek Thinking Token-Length Diagnosis

- User asked why the thinking output is long and whether 8192 tokens will solve it. Read-only usage analysis found 107/700 rows at the 4096 cap, all with empty final content; composition alone had 61/200 capped rows. Completion-token medians were 1419 for REL2C_LEFT and 1902 for REL2C_RIGHT, with reasoning tokens accounting for nearly all completion tokens. The runner does not save chain-of-thought text.
- The current request omitted reasoning_effort, so the DeepSeek thinking API used its default high effort; the frozen short-answer suffix constrains visible content only and does not bound hidden reasoning. Thinking-mode temperature is not an effective length control. The composition questions are intrinsically longer in this run because they combine visual identity, relation, and color selection.
- A new max_tokens=8192 run will likely remove many current cap failures and improve the aggregate, but it cannot guarantee no truncation or fix stop-time format/conflicting-answer failures. It also increases worst-case latency/output usage. If the goal is shorter thinking, reasoning_effort=low is a separate protocol and should not be mixed with the current high-effort 4096 results.

## 2026-09-13 CST: DeepSeek Thinking 8192 Baseline Analysis

- User reported that the separately fingerprinted 8192-token thinking summary was complete. Read-only comparison inspected its summary, protocol, reproducibility, secret-safety audit, and 700 result rows against the 4096 thinking and prior direct outputs; no API request, credential access, frozen-input change, scorer change, or result rewrite was performed.
- The 8192 protocol is valid: 700/700 unique rows, server model deepseek-flash on all rows, thinking enabled, max_tokens=8192, 200 composition-analysis rows, two retry transport attempts beyond the 700 successful QA completions, and secret-safety PASS.
- Increasing the cap reduced length termination from 107/700 to 41/700 and invalid rows from 170 to 94. Composition cap rows fell from 61/200 to 27/200, but 8192 did not eliminate truncation; composition still has 47 invalid rows total, including 27 empty cap responses, 20 stop-time format/conflict failures, and 80 valid-but-wrong semantic responses.
- Main 8192 metrics: I2C .670, PAIR_REL .730, REL2I .710, composition .365 (73/200); PCFR_min .408 (42/103), PCFR_strict .446 (33/74), and PCFR_path .386 (34/88). The verdict remains REALBIND_DIRECT_COMPOSITION_GAP_ESTABLISHED. Compared with 4096, prerequisite accuracy and raw aggregate accuracy improved, while conditional metrics changed modestly and are not evidence that thinking solved composition.
- Conclusion: 8192 substantially fixes the 4096 budget confound but is not fully budget-clean and does not remove the residual composition gap. A further high-effort run would need a new output root/protocol (for example 16384); a low reasoning-effort run would answer a different protocol question and must remain separate.

## 2026-09-13 CST: RealBind Case Figure v3 — Bowl and Knife

- User rejected the uncommon tape referent. Selected and visually inspected DeepSeek thinking-8192 family `realbind_test_2402473_2402473004_2402473002`: bowl left of knife, right identity knife, knife color Black all correct; right composition green versus frozen GT black. All four answers valid and non-truncated. Knife's handle is visibly black; full original kitchen image retained.
- Updated `render_case_figure.py`; rendered and visually checked PNG/PDF/SVG with provenance at `outputs/realbind_case_figure_v3/` and `/mnt/c/Users/张俊豪/Desktop/RealBind_case_figure_v3/`. Prior versions preserved; no inference or source-result changes.

## 2026-09-13 CST: RealBind Case Figure Reselection

- User requested a clearer color error than orange/yellow. Visually inspected alternatives and selected saved DeepSeek thinking-8192 family `realbind_test_2385540_2385540000_2385540001`: tape left of vase, right identity vase, vase color Green all correct; right composition Pink versus GT green. All displayed responses valid and non-truncated.
- Updated `render_case_figure.py`, checked rendered layout, and saved PNG/PDF/SVG with provenance to `outputs/realbind_case_figure_v2/` and `/mnt/c/Users/张俊豪/Desktop/RealBind_case_figure_v2/`; retained original desktop figure. No inference or source output changes.

## 2026-09-13 CST: RealBind Paper Case Figure

- User requested a paper-style original-image case with correct identity/color/position questions and wrong composition, saved to Desktop. Added `experiments/realbind_phase2/render_case_figure.py`; rendered and visually checked PNG plus PDF/SVG and source provenance in `outputs/realbind_case_figure/` and `/mnt/c/Users/张俊豪/Desktop/RealBind_case_figure/`.
- Selected DeepSeek thinking-8192 family `realbind_test_2319488_2319488001_2319488002`: apple left of pear=left, right identity=pear, pear color=yellow all correct; right composition=Orange versus GT yellow. All four displayed outputs valid and non-truncated; questions taken verbatim from frozen QA. Original full VG image preserved; source result bytes checked unchanged. No inference or API call.

## 2026-09-13 CST: RealBind Four-Protocol Cross-Model Analysis Complete

- User requested a strict offline comparison of Qwen direct, DeepSeek direct, DeepSeek thinking 4096, and DeepSeek thinking 8192 on the frozen pair-direct v1.1 release. Added `experiments/realbind_phase2/cross_model_analysis.py`; no VLM/API inference, API key access, scorer rule change, frozen-input change, or raw-result rewrite was performed.
- Commands: `python -m py_compile experiments/realbind_phase2/cross_model_analysis.py`; then `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/realbind_phase2/cross_model_analysis.py` (offline only). Auto-discovery found the four completed result files by protocol content; validation passed for 100 families, 700 QA, 100 rows/type, exact `(family_id, qa_id)` alignment, image paths, questions, answer types, and dependency spec.
- Unified in-memory scorer revision is `realbind_pair_direct_v1_1_semantic_vocab_v2`; all four raw inference JSONL files were byte-for-byte unchanged and saved scorer fields matched the recomputation. Family/QA/dependency SHA256 are `b8d366c2...1816ee`, `433ccc24...f06c3c87`, and `92383080...e57cdf9`.
- Main results: raw I2C/PAIR_REL/REL2I/composition are Qwen `.695/.700/.780/.490`, DeepSeek direct `.630/.580/.565/.365`, thinking-4096 `.645/.660/.655/.330`, and thinking-8192 `.670/.730/.710/.365`. Thinking 4096→8192 reduces total truncation `107→41` and invalid `170→94`; 8192 composition has `73` correct, `80` valid-semantic wrong, `27` truncated, and `20` stop-time format/conflict invalid rows.
- Raw PCFR `(min, strict, path)` is Qwen `(.320,.368,.318)`, direct `(.486,.452,.417)`, thinking-4096 `(.444,.441,.406)`, and thinking-8192 `(.408,.446,.386)`. Valid-semantic 8192 composition is `73/153=.477`; family bootstrap uses seed 42 and 10,000 replicates. Final artifacts are under `/mnt/d/datasets/RealBind/realbind/analysis/qwen_deepseek_cross_protocol_v1/`.
- Caveat: the direct source records requested identifier `deepseek-v4-flash-vision-exp` and does not record a served-model value; thinking sources record `deepseek-flash`. The analysis reports this provenance fact and does not retrospectively claim that the direct alias was served by a particular backend. Conclusions remain behavioral and do not establish architecture, mechanism, or global model superiority.

## 2026-09-13 23:17:41 CST: RealBind Qwen Visual-Interference Facts Pilot Complete

- User requested a strict local causal pilot comparing `FACTS_ONLY` (`Facts + frozen REL2C question`) with `VISION_PLUS_FACTS` (`original full image + the byte-identical text`), using only Qwen3.5-4B direct/non-thinking inference. Added `experiments/realbind_phase2/qwen35_visual_interference_pilot/run.py`; no frozen dataset, original Qwen baseline, or shared scorer was modified.
- Dry-run and execution commands: `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/realbind_phase2/qwen35_visual_interference_pilot/run.py --dry-run`; `/root/miniconda3/envs/next3d_qwen/bin/python -u experiments/realbind_phase2/qwen35_visual_interference_pilot/run.py --execute`. A post-completion `--execute` invocation only reused complete checkpoints to rebuild transition/metric artifacts; it did not load the model or issue new generations.
- Selection and audits passed: strong-path ready `88`, all `28` strong failures, exactly `28` deterministic matched successes across `(REL2C_LEFT/RIGHT, left_is_A)`, `56` cases, `112` generations. Three GT facts were injected from frozen family/QA fields; paired text hashes matched for every case; source result, frozen family/QA/dependency files, and scorer SHA256 remained unchanged.
- Both conditions completed `56/56` with Qwen3.5-4B, bf16, batch 1, `do_sample=False`, `use_cache=False`, `max_new_tokens=16`. Facts-only accuracy was `56/56=1.000`; Image+Facts was `45/56=.803571`, Δvision `-.196429` with family bootstrap 95% CI `[-.298246,-.100000]`; exact paired McNemar p=`.0009766`.
- Subsets: original strong failures Facts-only `28/28`, Image+Facts `22/28`, VIR `.214286`, FRR `1.0`, and rescue reversal `6/28=.214286`; matched strong successes Image+Facts `23/28`, VIR `.178571`. Overall VIR was `11/56=.196429`; no visual rescues occurred because all Facts-only answers were correct. Six cases were separately marked `FAILURE_RESCUE_REVERSED_BY_VISION`; all 11 basic visual-interference cases remain in the general interference file.
- Final status is `VISUAL_INTERFERENCE_SIGNAL_FOUND`, interpreted only as a paired behavioral association under this intervention, not proof that vision corrupts reasoning or of an internal mechanism. Outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/qwen35_visual_interference_pilot_v1/`, including provenance, selection/fact/prompt audits, both inference JSONL checkpoints, paired results, transitions, metrics, bootstrap/statistics, and human-readable reversal cases.

## 2026-09-14 09:14:42 CST: RealBind Visual-Interference Evidence Figure

- User requested a top-conference-style two-part figure using a real saved error record: `IMAGE + FACTS` wrong versus `FACTS ONLY` correct, with a rescue marker. Added `experiments/realbind_phase2/render_visual_interference_figure.py`; it reads only the completed pilot's `failure_rescue_reversed_by_vision.jsonl`, verifies the transition and paired prompt hash, and preserves the original Visual Genome image.
- Deterministically selected real case `realbind_test_2320720_2320720004_2320720002__REL2C_RIGHT`: facts are fire=orange, pole=gray, fire left of pole; frozen question asks the color on the right. Saved Qwen outputs are Image+Facts=`orange` (wrong; GT gray) and Facts-only=`gray` (correct), with original baseline=`green` (wrong). No image generation, model inference, prompt change, or source-result modification was performed.
- Rendered and visually checked `RealBind_visual_interference_2320720.{png,pdf,svg}` plus `figure_provenance.json` under `outputs/realbind_visual_interference_case_figure_v1/`. Renderer assertion confirmed the source reversal JSONL was unchanged; the provenance records image/source hashes, exact displayed prompt/facts/question, outputs, and direct/non-thinking generation configuration.

## 2026-09-14: RealBind Qwen/DeepSeek Data Tables Reissued

- User requested separate data tables for the atomic/relational-constituent versus composition experiment and for the `Image+Facts` versus `Facts-only` experiment. Read-only extraction used `analysis/qwen_deepseek_cross_protocol_v1/table_raw_performance.csv`, `conditional_metrics.csv`, and `experiments/qwen35_visual_interference_pilot_v1/metrics.md`; no inference, scorer, or result-file change was made.
- Atomic/composition metrics reissued for Qwen direct and DeepSeek direct/thinking-4096/thinking-8192. Key pooled values are Qwen `.695/.700/.780/.490`, DeepSeek direct `.630/.580/.565/.365`, thinking-4096 `.645/.660/.655/.330`, thinking-8192 `.670/.730/.710/.365` for I2C/PAIR_REL/REL2I/REL2C. Raw PCFR(min) is `.320`, `.486`, `.444`, `.408` respectively; denominators/failures remain in the CSV.
- The paired `Facts-only` versus `Image+Facts` experiment exists only for Qwen: `56/56` versus `45/56`, Δ=`-.196429`, VIR=`.196429`, six original-failure rescue reversals. No equivalent DeepSeek facts-only/image+facts pilot has been run, so no DeepSeek value was inferred or fabricated.

## 2026-09-14: Core RealBind Metrics Table Reduced

- User requested a strictly reduced table retaining only pooled atomic-fact accuracy, pooled composition accuracy, and atomic-facts-correct-but-composition-wrong rate, while showing DeepSeek direct, thinking-4096, and thinking-8192 separately.
- Definition used: atomic facts = pooled `I2C_A`, `I2C_B`, and `PAIR_REL` (`N=300`); composition = pooled `REL2C_LEFT/RIGHT` (`N=200`); conditional failure = `PCFR_min` with its ready-case denominator. Values reissued from the existing cross-model CSVs; no inference or result modification.

## 2026-09-14: RealBind Visual-Interference Error-Type Audit

- User requested an offline classification of errors introduced by `Image+Facts` relative to `Facts-only` in the completed Qwen visual-interference pilot. Read-only audit used `experiments/qwen35_visual_interference_pilot_v1/paired_results.jsonl` and `visual_interference_cases.jsonl`; no inference, scorer, frozen data, or result output was changed.
- All 56 selected cases were correct under `Facts-only`; `Image+Facts` introduced 11 errors. Nine were non-target-object color outputs (`9/11=81.8%`), specifically the injected `object_A` color; two were valid third-color outputs (`red`/`brown`, `2/11=18.2%`). All 11 were exact valid color responses: invalid=`0`, format violation=`0`, semantic truncation=`0`.
- The 11 errors occurred only when the queried target was object B: `11/28=39.3%` for target-B cases versus `0/28` for target-A cases in this matched pilot. Original strong failures contributed 6 errors (4 non-target-color, 2 third-color); matched strong successes contributed 5 (all non-target-color). This is a behavioral asymmetry in the selected pilot, not a mechanism claim; the third-color source is not identified by this experiment.

## 2026-09-14: RealBind AB-Order Reversal Control Complete

- User requested a minimal reversal control over exactly the completed pilot's 56 paired cases. Added `experiments/realbind_phase2/qwen35_visual_interference_pilot/ab_order_reversal.py`; it reuses the original selected-case set, frozen inputs, original images, Qwen3.5 direct/non-thinking loader/config, and `realbind_pair_direct_v1_1_semantic_vocab_v2`. Original baseline/result/scorer/frozen-input files were not modified.
- Ran dry-run and exactly 112 local generations (`REVERSED_FACTS_ONLY` and `REVERSED_VISION_PLUS_FACTS`), then an offline `--analyze-only` refresh. Reversed Facts-only was `55/56=.9821`; Reversed Image+Facts was `40/56=.7143`; reversed VIR was `15/55=.2727`. All reversed outputs were valid complete semantic color responses: invalid/format/truncation all `0`.
- Among the 11 original visual-interference cases, AB reversal rescued `8/11`, all by outputting the newly first-mentioned B color (now the target), while `3/11` persisted as the original object_A color. Among 12 new reversal interference cases, 9 output the new first-mentioned B color, 1 the original A color, and 2 a third color. Thus the pilot supports a substantial first-mentioned correspondence with a residual original-entity pattern; it does not establish mechanism and cannot separate first mention from object identity/text-layout without another control.
- Manually inspected the two original third-color images: red is clearly present as a bright boat/buoy in `2364553.jpg`; brown/tan is visibly present in broad water/shore/railway-ground regions in `2390136.jpg`. This is qualitative compatibility only, not automated color attribution or proof of visual causality. Artifacts are under `/mnt/d/datasets/RealBind/realbind/experiments/qwen35_visual_interference_pilot_ab_order_reversal_v1/`.

## 2026-09-14: RealBind Visual-Interference Residual Mechanism Test Complete

- User requested the minimal Qwen3.5-4B prediction-residual mechanism experiment over nine valid wrong-target-color cases from the completed visual-interference pilot; the two third-color cases were retained separately and excluded from main statistics. Added `experiments/realbind_phase2/qwen35_visual_interference_mechanism.py`; frozen data, original pilot outputs, scorer, and model files were not modified.
- Ran syntax check, dry-run, then CUDA bf16 execution in durable tmux. The first two implementation gates stopped safely (final-layer lens shape and raw-response field mapping); both were corrected without relaxing gates. Final execution passed exact clean replay, completed 576/576 self-patch no-op rows and 288/288 forward plus 288/288 reverse patch rows. No-op max first-token-logit difference was 0.0; all patch writes were exact.
- Main artifacts are under `/mnt/d/datasets/RealBind/realbind/experiments/qwen35_visual_interference_mechanism_v1/`: provenance, selected cases, paired audit, hidden-state metadata, layerwise color/identity scores, patch JSONL checkpoints, no-op audit, CSV curves, four PNG figures, classifications, and summary. Frozen family/QA/dependency hashes were unchanged before/after.
- Results: forward Facts-only residual into Image+Facts produced 113 correct final outputs across 288 case×layer rows; reverse patch retained/correctly answered 274/288 rows. Best rescue window was layers 28–31 with rescue rate `1.0`; best reverse-corruption layer was 31 with corruption rate `1.0`. Case diagnostics: 1 selection-like, 5 binding/readout-like, 3 unclear. Final status: `CAUSAL_RESIDUAL_BRIDGE_FOUND`.
- Caveat: the causal bridge is a paired residual intervention result on nine selected cases; identity trajectories are diagnostic only and do not prove an internal symbolic pipeline or mechanism. Historical stop sentinels from the corrected preflight attempts remain in the output directory as audit traces.

## 2026-09-15 16:46:11 CST: RealBind Qwen3.5 DaH Oracle Pilot Complete

- User requested a unified training-free Qwen3.5-4B evaluator comparing Original, Blur-only, Decompose-only, Handoff+TargetLock, and Full DaH on the frozen pair-direct v1.1 oracle-bbox pilot. Added `experiments/realbind_phase2/qwen35_dah_oracle_eval.py`; outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/qwen35_dah_oracle_v1/`. Frozen family/QA/dependency files and the scorer were read-only; no training, attention intervention, grounding, AB/BA gate, DeepSeek, CoT, or thinking was used.
- Commands: compiled the unified script; ran `--prepare-images`; ran the complete CUDA `--run --variant all` in durable tmux with the existing Qwen loader/renderer (`bf16`, batch 1, direct, `do_sample=False`, `use_cache=False`, `max_new_tokens=16`); ran offline `--analyze` after completion. A report-only patch fixed failure-breakdown percentages and added explicit A–E ablation comparisons; raw generations were not rerun.
- Alignment and execution completed: 100 families, 200 composition cases (100 LEFT/100 RIGHT), 300 deterministic PNGs, 20-family seed-42 HTML audit, 1,167 unique cache calls. All five raw result files and both Stage-1 checkpoints have 200 unique rows. Historical Qwen baseline vs rerun agreement was 200/200 for raw response, normalized prediction, and correctness. Live SHA256 checks matched recorded frozen family, QA, dependency, and scorer hashes; no `EXPERIMENT_INVALID` sentinel exists.
- Pooled composition accuracy: Original `.490` (`.460/.520` LEFT/RIGHT), Blur-only `.595` (`.590/.600`), Decompose-only `.580` (`.530/.630`), Handoff+TargetLock `.610` (`.570/.650`), Full DaH `.620` (`.590/.650`). Repair-vs-Original family bootstrap Δ CIs were Blur `[.055,.160]`, Decompose `[.020,.160]`, Handoff `[.055,.185]`, Full DaH `[.065,.195]`; exact McNemar p-values were `.001065`, `.024746`, `.004272`, and `.002186` respectively.
- Decomposed metrics: binding accuracy was `.780/.780/.820` for Decompose-only/Handoff/Full; attribute accuracy conditional on correct binding was `.711538/.743590/.719512`; final accuracy was `.580/.610/.620`. Full DaH had 76 failures: 30 binding-source and 46 attribute-after-correct-binding. Wrong-binding same-color accidental-correction count was zero; there were 6 wrong-binding but final-correct cases in Full DaH, all with different selected/target colors, so they are not binding successes.
- Historical strong-path mask was recomputed as 88 ready cases, 28 original strong failures, and 98 original successes. Rescue was Blur `15/28=.535714`, Decompose `28/28=1.0`, Handoff `25/28=.892857`, Full DaH `25/28=.892857`; Full DaH preserved `77/98=.785714`. The 9 wrong-target-color visual-interference cases had 8 Full DaH rescues. Final descriptive status: `DAH_STRONG_SIGNAL`.
- Caveat: this is an oracle-localization pilot only: GT bboxes build pair/target-lock images, while runtime target selection always follows normalized Stage-1 A/B prediction. All Stage2 calls were fresh independent requests using exact frozen I2C questions; no target correction or prompt rewriting was performed.

## 2026-09-15 18:41:54 CST: RealBind DaH Paired Transition Analysis Complete

- User requested case-level paired transitions from the completed DaH oracle pilot: Full vs Blur-only, Full vs Handoff+TargetLock, and Stage1 original-image REL2I vs pair-defocus REL2I. Added `experiments/realbind_phase2/analyze_dah_transitions.py`; analysis was read-only over existing result/checkpoint JSONL files, with no model inference, rescoring, frozen-input mutation, or baseline rewrite.
- Outputs: `full_vs_blur_transitions.csv`, `full_vs_handoff_transitions.csv`, `stage1_original_vs_pair_transitions.csv`, `paired_transition_analysis.json`, and `paired_transition_analysis.md` under `/mnt/datasets/RealBind/realbind/experiments/qwen35_dah_oracle_v1/`. Validation passed: each transition CSV has 200 unique cases.
- Full vs Blur: 27 newly repaired, 22 broken, net `+5`; baseline Blur correct `119/200`, Full correct `124/200`, prediction changed on 69. Full-vs-Blur net gain was LEFT `0` (10/10) and RIGHT `+5` (17/12).
- Full vs Handoff: 6 newly repaired, 4 broken, net `+2`; Handoff correct `122/200`, Full correct `124/200`, prediction changed on 16. Net gain was LEFT `+2` and RIGHT `0`.
- Stage1 original vs pair-defocus: 12 binding fixes, 4 binding breaks, net `+8`; original correct `156/200`, pair-defocus correct `164/200`, normalized identity changed on 16. Fixes were balanced LEFT/RIGHT at 6/6; breaks were 2/2.
- Detailed case IDs and base/comparison predictions are in the Markdown/CSV artifacts. These are descriptive paired transitions; no new significance test or causal claim was added.

## 2026-09-15 19:20:05 CST: RealBind DaH Stage1 Pair-Corridor Pilot Complete

- User requested a Stage1 REL2I-only validation comparing Original, the existing pair-defocus, and a new pair+corridor defocus. Added `experiments/realbind_phase2/qwen35_dah_rel2i_corridor_eval.py`; Original and current pair-defocus raw rows were reused from the completed DaH evaluator, while exactly 200 new pair+corridor Qwen3.5-4B calls were run. No Stage2/composition inference was executed.
- Corridor images use the unchanged bbox expansion, blur radius, feather, processor, and generation config. The new fixed geometry is a soft capsule between raw A/B bbox centers with radius `max(4px, 0.25 * smaller expanded bbox max-side)`; no parameter search. Existing pair-defocus PNGs matched recomputation byte-for-byte for all 100 families. Corridor image audit passed for resolution, determinism, and no pixel annotations; visual HTML contains 20 seed-42 families.
- Binding accuracy: Original `.780` (`.760/.800` LEFT/RIGHT), pair-defocus `.820` (`.800/.840`), pair+corridor `.815` (`.800/.830`). All three had invalid/format/truncation counts zero.
- Pair-defocus -> corridor had 1 newly repaired case and 2 broken cases, net `-1` (`-.005`). The new repair was `realbind_test_2381904_2381904006_2381904002__REL2C_LEFT`; regressions were `realbind_test_2397782_2397782000_2397782007__REL2C_LEFT` and `realbind_test_716_716004_716008__REL2C_RIGHT`.
- Of the original pair-defocus 12 new binding repairs, corridor retained 10/12 and lost 2/12; of the four pair-defocus regressions relative to Original, corridor restored 0/4. Pair->corridor exact McNemar p=`1.0`, family bootstrap Δ CI `[-.025,+.010]` (seed 42, 10,000 replicates). Original->corridor was 11 repairs/4 breaks, p=`.118469`, family bootstrap CI `[-.005,+.080]`.
- Outputs are under `/mnt/datasets/RealBind/realbind/experiments/qwen35_dah_rel2i_corridor_v1/`: config/provenance, audits, 100 corridor PNGs, visual audit, raw checkpoint, case comparison, summary, paired statistics, and regression CSV. Final status is `VALID`; frozen input/scorer/model-config gates passed.

## 2026-09-15 20:10:55 CST: RealBind DaH Stage1 Pair-Crop Pilot Complete

- User requested a Stage1 REL2I-only Pair-Centric Re-Encoding / Pair-Crop validation on the same 100 families / 200 linked cases. Added `experiments/realbind_phase2/qwen35_dah_rel2i_pair_reencode_eval.py`; Original and current Pair-defocus raw outputs were reused, with exactly 200 new Pair-Crop Qwen3.5-4B calls. No Stage2/composition inference, target lock, corridor, dual-view, prompt/config search, intervention, training, or thinking was run.
- Pair-Crop is the raw original-image crop of the minimal joint A/B bbox expanded by fixed `0.20` in each axis, clipped and passed to the existing processor without blur, annotation, stretch, or manual resize. Image audit passed for 100/100 families: A/B contained, left/right ordering preserved, exact original pixels, deterministic PNGs; visual audit shows 20 seed-42 families.
- Binding accuracy: Original `.780` (`.760/.800` LEFT/RIGHT), Pair-defocus `.820` (`.800/.840`), Pair-Crop `.810` (`.800/.820`). All variants had invalid/format/truncation counts zero; sanity checks passed for frozen input/scorer/model-generation hashes and Stage2 call count 0.
- Original→Pair-Crop had 11 repairs and 5 breaks, net `+.030` (McNemar p=`.210114`, family bootstrap 95% CI `[-.015,+.080]`). Pair-defocus→Pair-Crop had 8 repairs and 10 breaks, net `-.010` (p=`.814529`, CI `[-.055,+.035]`). Pair-Crop retained 7/12 existing Pair-defocus repairs and restored 4/4 existing Pair-defocus regressions. Full case-level three-prediction ledger is in `summary_metrics.md` and `transition_analysis.csv`.
- Fixed magnification bins showed Pair-Crop−Pair-defocus deltas of `+.0345` for `m<1.5` (N=116), `-.0517` for `1.5≤m<2.5` (N=58), `-.1667` for `2.5≤m<4` (N=18), and `0` for `m≥4` (N=8). Transition-group geometry did not support the effective-resolution hypothesis: Pair-Crop repaired cases had lower effective magnification than Pair-Crop broken cases (mean `1.351` vs `2.091`). Final status is `PAIR_REENCODE_WEAK_SIGNAL` because Pair-Crop was below Pair-defocus and repair/regression changes were not favorable.
- Outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/qwen35_dah_rel2i_pair_reencode_v1/`: config/provenance, data/bbox/image audits, 100 Pair-Crop PNGs, visual audit, raw/cache checkpoints, comparison/transition/magnification CSVs, summary, and paired statistics. Commands: `python -m py_compile ...`; `--prepare-images`; durable CUDA `--run`; offline `--analyze`.

## 2026-09-15 22:27:10 CST: RealBind DaH Stage2 Role-Handoff Pilot Complete

- User requested a strict Stage2-only Qwen3.5-4B experiment on the existing Full DaH pair-defocus Stage1 predictions, comparing Original/TargetLock visuals with frozen Plain identity, fixed Explicit Selected Role, and fixed Contrastive Selected Role prompts. Added `experiments/realbind_phase2/qwen35_stage2_role_handoff_eval.py`; no Stage1 rerun, frozen input/scorer change, target correction, prompt search, training, attention intervention, CoT, or thinking was used.
- Commands: compiled the unified evaluator; `--prepare-images`; durable CUDA `--run --condition original_plain` followed by `--run --condition all`; offline `--analyze`. Existing T-P0 was read from Full DaH raw; O-P0 reused 193 base-cache calls and added 7 exact misses; four P1/P2 condition families used 728 unique new model generations with row-level checkpoints. All six condition outputs have 200 unique rows; Stage1 calls=0 and Stage2 condition rows=1,200.
- Protocol gates passed: 100 families / 200 composition cases (100 LEFT/100 RIGHT), 200/200 valid frozen pair-defocus Stage1 rows and 164/200 binding-correct, exact P1/P2 prompt audit 200/200, byte-identical TargetLock reuse, fresh Stage2 contexts, model/generation/scorer/frozen-input hashes unchanged. Experiment cache contains 735 unique new model generations; no `EXPERIMENT_INVALID` sentinel exists.
- Stage1-correct primary attribute accuracy: O-P0 `.695122` (114/164), O-P1 `.695122`, O-P2 `.701220` (115/164), T-P0 `.719512` (118/164), T-P1 `.750000` (123/164), T-P2 `.737805` (121/164). T-P0→T-P1 repaired 7 and broke 2 (`Δ=.030488`, exact McNemar p=`.179688`, family-bootstrap CI `[-.005882,.067073]`); T-P1→T-P2 repaired 4 and broke 6 (`Δ=-.012195`). Original visual role gain was zero; only O-P1 reduced competitor-attribute errors (`8→7`), while T-P1's gain reduced third/invalid errors (`34→30`, `3→2`) but did not reduce competitor errors (`9→9`).
- All-200 final composition accuracy: O-P0/O-P1/O-P2 `.590/.590/.590`; T-P0/T-P1/T-P2 `.620/.640/.620`, with T-P0→T-P1 `7/3` repaired/broken and `Δ=.020`. Fixed existing T-P0 failure subset was 46 cases: O-P1 rescued `10`, O-P2 `10`, T-P1 `7`, T-P2 `11`; rescue-by-type is recorded in the summary/CSV. Nine wrong-target-color visual-interference diagnostics were correct `9/9` for all Original conditions and `8/9` for all TargetLock conditions.
- Final status: `ROLE_HANDOFF_MODERATE_SIGNAL`. The positive TargetLock P1 paired gain is structured but not statistically decisive and does not reduce competitor-attribute errors; the stricter combined strong-signal condition therefore did not pass. Outputs are under `/mnt/d/datasets/RealBind/realbind/experiments/qwen35_stage2_role_handoff_v1/`, including six raw JSONL files, prompt/stage1/TargetLock audits, transition/error/failure CSVs, paired statistics, `summary_metrics.json`, and `summary_metrics.md`.
-
## 2026-09-16 CST: Stage2 Binding-Guided Visual Routing V1 Invalid Random-Control Gate

- User requested implementation and execution of the strictly frozen PHASE 1–4 Oracle Binding-Guided Visual Routing feasibility test: reuse Full DaH pair-defocus Stage1 predictions, original images, byte-identical EXPLICIT_SELECTED_ROLE/P1, predicted-entity bbox masks, only standard self-attention visual-budget routing at W1/W2, and stop before Phase 5. Added `experiments/realbind_phase2/qwen35_stage2_binding_guided_routing_v1.py`; no Stage1 rerun, image/prompt/model/data change, TargetLock, state/residual patch, search, training, or Phase 5 was run.
- Static checks and a one-case CUDA smoke passed for the custom eager attention path: beta=0 matched the copied installed eager implementation exactly; routing was limited to the first prompt-prefill query, W1 modified only L19, mass conservation error was about `1.19e-7`, non-visual keys were unchanged, and the hook count was exactly one modified layer. Actual loaded model audit found 32 decoder layers, standard self-attention `[3,7,11,15,19,23,27,31]`, untouched Gated DeltaNet layers `[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30]`, W1 actual `[19]`, W2 actual `[23,27]`, 16 Q heads / 4 KV heads / head_dim 256.
- Formal output root was auto-resolved to `/mnt/datasets/RealBind/realbind/experiments/qwen35_stage2_binding_guided_routing_v1/`. Visual mapping passed for all 100 families with runtime `image_grid_thw`, processor geometry, post-merge cells, exact decoder image-token positions, and 20 seed-42 HTML audit overlays. The first all-run reached only a stopped partial patched baseline `45/200` before attempting a random mask and is quarantined as non-evidence.
- Offline/preflight random-region feasibility found `16/200` frozen cases where an exact-count mask outside both predicted target and competitor masks is mathematically impossible; 38 cases have overlapping bbox token masks. Example: predicted B has 98 target tokens, only 82 legal non-target/non-competitor tokens remain. The implementation was tightened to audit this before any further Stage2 generation and to write `random_region_feasibility.json` plus `EXPERIMENT_INVALID.json`/provenance status.
- Final status: **`EXPERIMENT_INVALID_RANDOM_REGION`**. No formal TARGET/COMPETITOR/RANDOM routing conditions, paired statistics, behavior analysis, or Phase 4 signal were produced. Do not resume or relax overlap/token-count requirements without an explicitly new protocol; the existing 45-row partial baseline must not be interpreted.

## 2026-09-26 23:48 CST: RealBind-v2 Metadata-First Construction Reaches Human Audit

- User requested a model-agnostic RealBind-v2 builder and resumable metadata-first streaming construction. Added `build_realbind_v2.py`, `realbind_v2/{mining,downloader,gates,referents,state_db,audit_export,question_generator,validate}.py`, `realbind_v2/protocol.yaml`, README, and `tests/test_realbind_v2.py`; installed a runnable copy under `/mnt/d/vlm_repair_v2/experiments/realbind_v2_builder/`. No VLM was loaded or scored.
- Reused local VAW train/val/test and Visual Genome annotations/images. Metadata inventory: 108,077 VG image records, 2,516,939 VG object annotations, and 124,688 one-canonical-color VAW instances. Excluded all 100 v1.1 development image IDs. Fixed gates reuse v1.1 geometry values: min object area ratio `.02`, normalized center separation `.20`, max IoU `.10`; frozen colors are the existing 11-color vocabulary. The VAW instance ID is not the VG object ID, so cross-source consistency checks category presence per image and leaves referent/object identity to human review.
- Metadata snapshot under `/mnt/d/datasets/RealBind_v2/`: 9,256 candidate images; 18,294 candidate pairs before cap; 13,302 selected metadata families with at most 2/image; 136,928 pair-level rejection records; `metadata_candidates.sqlite` includes all candidate families and selected-for-stream flags. Same-category candidates are 1,603/13,302; left-is-A is 6,693/13,302. Inputs are fingerprinted in `manifests/mine_summary.json`.
- The actual 50-image dry-run passed, then the same resumable command continued to target: 1,400 `AUTO_VALID` families from 973 locally reused VG images, 973 image files / 0.172 GB in accepted, zero remote downloads, zero download retries/failures, 48.807 GB free, zero working bytes. 8,283 candidate images remain pending for future extensions; run stopped because the preregistered 1,400-family target was met.
- Generated `audit/index.html` with 1,400 family cards, A/B overlays, seven deterministic QA, and decision controls. All 1,400 remain human pending; no human acceptance was fabricated and no frozen benchmark export was run. `construction_report.md` and `dry_run_report.md` explicitly state the model-free construction guarantee and human-review status.
- Verification: `python -m unittest -v tests.test_realbind_v2` passed 9 tests, then passed again with `TMPDIR=/mnt/d/datasets/RealBind_v2/working` so SQLite commit/fsync/delete, resume, and audit/export fixture behavior ran on D:. Read-only SQLite checks confirmed 9,256 candidate image rows, 18,294 candidate families, 13,302 selected, 136,928 pair rejects, max 2 selected families/image, and no candidate family missing an image record. The production 50-image sample had 0 image-level rejects; the transactional reject/delete behavior was verified on the isolated synthetic D: fixture and appears in `dry_run_report.md`.

## 2026-09-30 CST: RealBind2 Repository Packaging Audit And Fail-Closed Refactor

- User requested a direct audit/refactor/test/smoke/package/private-GitHub workflow without changing dataset, prompt, scorer, causal, or repair semantics. Added the self-contained `realbind2-vlm/` project with parameterized model/data paths, frozen prompt/scorer/hash manifests, model adapters, append-only resume logging, deterministic two-GPU shards, automatic F0/F1 derivation, Strong Integrated cache/handoff contracts, cross-model runners, environment isolation, mandatory tests, full runbook, audit, and packaging report. Historical source was copied without deleting or overwriting originals; compact evidence is under `realbind2-vlm/legacy/evidence/`.
- Audit located the repair workspace at `/mnt/d/vlm_repair_v2`. The final route is explicitly R4 MM-GDINO-L + frozen Qwen selector, deterministic target geometry, SAM3 target mask isolation/fallback, crop-Qwen attribute reading, minimal state, and image-free final reasoning. Direct SAM3 selection/consolidation is labeled `HISTORICAL_FAILED_BRANCH`; DaH oracle and 40-case R4/Strong Integrated runners are retrospective only. MM-GDINO remains isolated at Transformers 4.55.4; the invalid 5.8.1 detector run remains excluded.
- Critical blocker found: `/mnt/d/datasets/RealBind_v2/state/construction_state.sqlite` contains 1,400 `AUTO_VALID` families over 973 unique source images, but 0 `HUMAN_ACCEPTED` and 0 `FROZEN`; no frozen 9,800-row QA export or formal splits exist. `data/realbind2_manifest.json` records all current IDs and `formal_run_allowed=false`. RealBind2 VIR defocus semantics also lack a located frozen implementation hash. Dataset-scale VIR/causal/retrospective and fresh live smoke therefore fail closed; no benchmark rows or model forwards were fabricated.
- Verification: new pytest suite `13 passed`; preserved historical harness `73 passed` in its original `/mnt/d` environment; `python -m compileall`, all shell `bash -n`, `data/HASHES.json` (12 files), and artifact-backed `bash experiments/qwen35_main/00_smoke.sh` all passed with `SMOKE_PASS`. Live smoke correctly stops on missing frozen dataset/model configuration. Work is on branch `realbind2-packaging`; publishing status is recorded in `realbind2-vlm/PACKAGING_REPORT.md`.


## 2026-09-30 CST: RealBind2 Local Pre-Server Certification

- User requested local-only dataset tracing, QA/count certification, historical VIR defocus recovery, and a server live-certification bundle; no GPU inference/model download was run. Work is in `realbind2-vlm/`.
- Immutable construction DB audit: 1400 AUTO_VALID families, 973 unique source images; 546 images contribute one family and 427 contribute two. Maximum two matches the protocol. No duplicate family/entity-pair/QA IDs; all 1400 images decode and all families have seven complete QA, totaling 9800 QA/2800 REL2C. Unit is FAMILY, not unique image.
- Human gate is explicitly required by preserved builder protocol/export code; HUMAN_ACCEPTED=0, FROZEN=0, DATASET_FROZEN events=0. No alternate final frozen export found. Manifest stays nonfrozen; no REALBIND2_FROZEN.sha256 was created. Audit and blockers are recorded in REALBIND2_DATA_AUDIT.md, REALBIND2_VERSION_COMPARISON.md, and DATASET_FREEZE_BLOCKERS.md.
- Recovered historical DaH/VIR pair-defocus into src/failure/defocus_frozen.py. Five historical input/output fixtures reproduce exact pixels and PNG bytes. Historical REMOVE==FOCUS defect is documented separately. Source artifacts remain untouched.
- Prepared server_certification scripts for GPU/path/Qwen/MM-GDINO/SAM3/Strong Integrated/self-patch, fixed six metadata-only smoke families, runtime profiling, and receipt-gated deterministic two-GPU baseline/stability shards. Cohort builder requires two complete independent runs and no-replacement matching. Server live certification remains NOT_RUN_LOCAL_BY_DESIGN.
- Local checks: next3d_qwen Python pytest 21 passed/1 live_gpu skipped; compile, shell syntax, HASH_PASS files=18, five DEFOCUS exact regressions, artifact-backed SMOKE_PASS. Base Python has no pytest. SERVER_HANDOFF.md and PRE_SERVER_STATUS.md record dataset BLOCKED, DEFOCUS PASS, bundle prepared, and formal transfer/run blocked pending human freeze. GitHub CLI absent; GITHUB_PUSH_BLOCKED_AUTH.
