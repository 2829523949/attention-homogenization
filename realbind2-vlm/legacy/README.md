# Historical source map

Nothing in this directory is the headline final repair.

| Preserved source | Packaged location | Status / purpose |
|---|---|---|
| `/mnt/d/vlm_repair_v2/harness/` | `legacy/vlm_repair_v2/harness/` | Full historical repair research harness; source snapshot, including context-preserving binder, P0 Qwen, P1 ObjEmbed interfaces, joint assignment, and direct grounding work. |
| `/mnt/d/vlm_repair_v2/docs/` | `legacy/vlm_repair_v2/docs/` | Historical contracts and audit notes. |
| `/mnt/d/vlm_repair_v2/manifests/` | `legacy/vlm_repair_v2/manifests/` | Small historical development manifests only. |
| `experiments/realbind_phase2/qwen35_dah_oracle_eval.py` | `src/repair/dah/qwen35_oracle_retrospective.py` | Qwen3.5-only oracle retrospective; GT bboxes make it diagnostic, not final repair. |
| `/mnt/d/vlm_repair_v2/harness/scripts/run_region_bottleneck_joint_repair.py` | `src/repair/retrospective/r4_40case_historical.py` | Historical R4 40-case implementation. |
| `/mnt/d/vlm_repair_v2/harness/scripts/run_strong_integrated_repair.py` | `src/repair/retrospective/strong_integrated_40case_historical.py` | Historical 40-case integrated implementation. |
| `/mnt/d/vlm_repair_v2/harness/scripts/run_sam3_instance_hypothesis_binding.py` | `src/repair/retrospective/sam3_consolidation_failed.py` | Historical failed SAM3 direct-selection/consolidation branch. |
| `experiments/realbind_phase2/qwen35_visual_interference_mechanism.py` | `legacy/qwen35_visual_interference_mechanism.py` | Validated nine-case Qwen3.5 mechanism history. |

Large outputs, masks, images, model weights, and caches remain in their
original locations and are excluded from Git. Compact reports in
`legacy/evidence/` are byte copies. All copied historical runners retain their
original hard-coded paths and are intentionally not official entrypoints.

`UNKNOWN_LEGACY`: older position-ID, index, recurrent/GDN, KV orthogonal, KEY
refine, K-rescuable, LBI, OCBW, and unrelated Qwen experiments remain in the
parent workspace. Their semantics were not rewritten or reclassified as
repair.

