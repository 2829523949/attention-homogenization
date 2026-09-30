# Environment and checkpoint report

## Frozen cohort

- Locked cohort: `manifests/acquisition_semantic_support_20family.json`, SHA-256 `f1d7096598f054e8288dcdcecf5e53c4252b11e256e80fade9c69fb048ee384f`, 20 families and 40 REL2C QA.
- Prepared legal-QA input: `outputs/minimal_decomposition_route_pilot/prepared.jsonl`, SHA-256 `16fbf958bc75225e50f4bd8769a332c644d17b7b5b25d9f80837d2b1dba1e404`.
- Original candidate source (frozen): `outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl`, SHA-256 `0b1b058199ccc17b7b4d42307cbefa2eef261617a03ec168973f5f71ad8d3b8b`.

## MM-GDINO-L

- Checkpoint: `openmmlab-community/mm_grounding_dino_large_all`, revision `0445d0a0ae5f0fa6be689cf7243d4d0ef4ba7de0`; local path `models/mm_grounding_dino/mm_grounding_dino_large_all`.
- `model.safetensors` SHA-256 `1802856b98c66cdf6c4b1e9841320f7b76b039c9cb54e88061de1c2517b53822`.
- `config.json` SHA-256 `7669f371f9ae2d642881ecd7b1769095384439c8a9e01b5f958eaed298919cb2`; model type `mm-grounding-dino`, Swin-L backbone, 343.55M parameters.
- Corrected inference runtime: Python 3.10.20; PyTorch 2.7.0+cu128; CUDA runtime 12.8; Transformers 4.55.4; tokenizers 0.21.4; huggingface_hub 0.36.2; NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB.
- Inference settings: float32 weights and inputs, no autocast; no NMS; box threshold 0.25; text threshold 0.25; retained top 16 candidates by native score with deterministic raw-index tie-breaking.
- Prompt construction: the frozen `ReferentSpec.text_description.value` is passed verbatim as a singleton candidate-label list; the official processor performs its deterministic lower-casing/separator conversion. No question, side, relation, answer, attribute enrichment, target-side, or GT information is seen by the detector.
- MMDetection is not installed or used: the official OpenMMLab Hugging Face conversion is executed directly through Transformers, so there is no MMDetection version to report.

## Original Grounding DINO

- `IDEA-Research/grounding-dino-base`, revision `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`.
- Candidate pools reused exactly from the frozen prior acquisition pilot; box threshold 0.25; text threshold 0.25; no NMS.

## Qwen3.5-4B local processors

- Model: `Qwen/Qwen3.5-4B`, local path `/root/Qwen3-VL/models/Qwen3.5-4B`; config SHA-256 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.
- Runtime: Python 3.10.20; PyTorch 2.7.0+cu128; Transformers 5.8.1; CUDA 12.8; same local GPU.
- Deterministic generation: `do_sample=false`, `use_cache=false`, `max_new_tokens=16`, `batch_size=1`, `enable_thinking=false`.
- Selector role: numbered candidate-crop grid plus legal referent phrase only. It never sees the REL2C question, LEFT/RIGHT target side, final answer, GT, or relation answer.
- Color role: exact selected target crop plus the fixed generic color prompt only. No relation question, target side, GT, or full-scene reasoning.

## Invalid pre-repair run disclosure

The first MM-GDINO-L pass in this pilot used Transformers 5.8.1, which produced incorrect box semantics for this checkpoint. Those 40 strong-candidate records, selector rows, color rows, and logs are archived under `outputs/region_bottleneck_joint_repair/invalid_5_8_1_run/` and are excluded from every reported metric. The corrected pass used an isolated Transformers 4.55.4 runtime and reran all 40 strong referents consistently. The Qwen selector/color and final text reasoner continued to use the project's existing frozen Transformers 5.8.1 runtime.

## Prohibitions

No hosted API, DINO-X API, Grounding DINO 1.6 API, SAM/SAM2, segmentation refinement, detector ensemble, score fusion, blur, certification, soft handoff, or full-scene Qwen reasoning is used.
