#!/usr/bin/env python3
"""Minimal prediction-residual mechanism test for RealBind visual interference.

The main sample is frozen to the completed pilot's valid wrong-target-color
transitions: Facts-only correct, Image+Facts wrong, and the image-conditioned
answer is the other RealBind object's canonical color.  Third-color cases are
audited and excluded from all main mechanism statistics.

Default execution is dry-run.  ``--execute`` uses the same local Qwen3.5-4B
loader and direct/non-thinking protocol as the completed behavioral pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.realbind_phase2 import pair_direct_v1_1_baseline as qwen_baseline  # noqa: E402
from experiments.realbind_phase2.qwen35_visual_interference_pilot import run as behavioral_pilot  # noqa: E402


REALBIND_ROOT = Path("/mnt/d/datasets/RealBind/realbind")
PILOT_ROOT = REALBIND_ROOT / "experiments" / "qwen35_visual_interference_pilot_v1"
OUTPUT_ROOT = REALBIND_ROOT / "experiments" / "qwen35_visual_interference_mechanism_v1"
MODEL_DIR = ROOT / "models" / "Qwen3.5-4B"
MODEL_IDENTIFIER = "Qwen/Qwen3.5-4B"
FAMILIES_PATH = REALBIND_ROOT / "families" / "dryrun_100_pair_direct_v1_1_frozen.jsonl"
QA_PATH = REALBIND_ROOT / "qa" / "dryrun_100_pair_direct_v1_1.jsonl"
DEPENDENCY_PATH = REALBIND_ROOT / "qa" / "pair_direct_v1_1_dependency_spec.json"
SCORER_PATH = ROOT / "experiments" / "realbind_phase2" / "pair_direct_v1_1_baseline.py"

FACTS = "FACTS_ONLY"
VISION = "VISION_PLUS_FACTS"
MAX_NEW_TOKENS = 16
USE_CACHE = False
BATCH_SIZE = 1
EXPECTED_MAIN_CASES = 9
EXPECTED_THIRD_COLOR_CASES = 2
EXPECTED_LAYERS = 32
SCORER_REVISION = "realbind_pair_direct_v1_1_semantic_vocab_v2"
LENS_RECONSTRUCTION_TOLERANCE = 1e-4

# Fixed before outcome inspection.  A wrong-sign onset is the start of the
# terminal run of strictly negative target-minus-other margins.
CLASSIFICATION_RULE = {
    "wrong_sign_onset": "earliest layer l such that margin(l..final) is strictly negative",
    "selection_like": "F color and identity margins are positive at their respective VF onsets; VF identity terminal-negative onset exists and is no later than VF color onset",
    "binding_readout_like": "VF color terminal-negative onset exists while F color, F identity, and VF identity margins are positive at the color onset, and identity onset is absent or later",
    "unclear": "all remaining trajectories",
    "no_fitted_threshold": True,
}


class MechanismStop(RuntimeError):
    def __init__(self, code: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = dict(details or {})


def bool_value(value: Any) -> bool:
    return behavioral_pilot.bool_value(value)


def read_json(path: Path) -> Any:
    return behavioral_pilot.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return behavioral_pilot.read_jsonl(path)


def write_json(path: Path, value: Any) -> None:
    behavioral_pilot.write_json(path, value)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    behavioral_pilot.write_jsonl(path, rows)


def stable_hash(value: Any) -> str:
    return behavioral_pilot.stable_hash(value)


def sha256_file(path: Path) -> str:
    return behavioral_pilot.sha256_file(path)


def sha256_bytes(value: bytes) -> str:
    return behavioral_pilot.sha256_bytes(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "provenance": root / "provenance.json",
        "protocol": root / "protocol.json",
        "selected": root / "selected_cases.jsonl",
        "excluded_third": root / "excluded_third_color_cases.jsonl",
        "paired_audit": root / "paired_audit.json",
        "hidden_meta": root / "hidden_state_metadata.json",
        "color_scores": root / "layerwise_color_scores.jsonl",
        "identity_scores": root / "layerwise_identity_scores.jsonl",
        "baseline_audit_rows": root / "clean_baseline_replay_audit.jsonl",
        "noop_rows": root / "noop_patch_results.jsonl",
        "noop_audit": root / "noop_patch_audit.json",
        "forward": root / "forward_patch_results.jsonl",
        "reverse": root / "reverse_patch_results.jsonl",
        "rescue_csv": root / "rescue_by_layer.csv",
        "corruption_csv": root / "corruption_by_layer.csv",
        "classification": root / "case_classification.jsonl",
        "summary": root / "summary.md",
        "color_figure": root / "color_margin_curve.png",
        "identity_figure": root / "identity_margin_curve.png",
        "rescue_figure": root / "rescue_curve.png",
        "corruption_figure": root / "corruption_curve.png",
        "runtime": root / "runtime_metadata.json",
        "state_cache": root / "state_cache",
    }


def family_view(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "A": {"object": str(case["object_A"]), "canonical_color": str(case["color_A"])},
        "B": {"object": str(case["object_B"]), "canonical_color": str(case["color_B"])},
    }


def other_fields(row: Mapping[str, Any]) -> tuple[str, str, str]:
    target_label = str(row["target_identity_label"])
    other_label = "B" if target_label == "A" else "A"
    return other_label, str(row[f"object_{other_label}"]), str(row[f"color_{other_label}"])


def load_and_select_cases(
    families: Sequence[Mapping[str, Any]],
    qa_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    paired_path = PILOT_ROOT / "paired_results.jsonl"
    facts_path = PILOT_ROOT / "inference_facts_only.jsonl"
    vision_path = PILOT_ROOT / "inference_vision_plus_facts.jsonl"
    required = (paired_path, facts_path, vision_path)
    if any(not path.is_file() for path in required):
        raise MechanismStop("COMPLETED_PILOT_NOT_FOUND", {"required": [str(path) for path in required]})
    paired = read_jsonl(paired_path)
    facts_rows = read_jsonl(facts_path)
    vision_rows = read_jsonl(vision_path)
    facts_by_id = {str(row["case_id"]): row for row in facts_rows}
    vision_by_id = {str(row["case_id"]): row for row in vision_rows}
    family_by_id = {str(row["family_id"]): row for row in families}
    qa_by_id = {str(row["qa_id"]): row for row in qa_rows}
    failures: list[dict[str, Any]] = []
    main: list[dict[str, Any]] = []
    third: list[dict[str, Any]] = []

    if len(paired) != 56 or len(facts_rows) != 56 or len(vision_rows) != 56:
        raise MechanismStop("PILOT_RESULT_COUNT_MISMATCH", {"paired": len(paired), "facts": len(facts_rows), "vision": len(vision_rows)})
    if set(facts_by_id) != {str(row["case_id"]) for row in paired} or set(vision_by_id) != set(facts_by_id):
        raise MechanismStop("PILOT_RESULT_ALIGNMENT_MISMATCH")

    for row0 in paired:
        row = dict(row0)
        cid = str(row["case_id"])
        family = family_by_id.get(str(row["family_id"]))
        qa = qa_by_id.get(str(row["qa_id"]))
        facts = facts_by_id[cid]
        vision = vision_by_id[cid]
        local: list[str] = []
        if family is None or qa is None:
            local.append("missing_frozen_source")
        else:
            for label in ("A", "B"):
                if str(row[f"object_{label}"]) != str(family[label]["object"]):
                    local.append(f"object_{label}")
                if str(row[f"color_{label}"]) != str(family[label]["canonical_color"]):
                    local.append(f"color_{label}")
            if str(row["question"]) != str(qa["question"]) or str(row["target_color"]) != str(qa["ground_truth"]):
                local.append("composition_qa")
        if str(row.get("text_prompt", "")).encode("utf-8") != str(facts.get("text_prompt", "")).encode("utf-8"):
            local.append("facts_prompt_bytes")
        if str(row.get("text_prompt", "")).encode("utf-8") != str(vision.get("text_prompt", "")).encode("utf-8"):
            local.append("vision_prompt_bytes")
        prompt_hash = sha256_bytes(str(row.get("text_prompt", "")).encode("utf-8"))
        if prompt_hash != str(row.get("facts_only_text_hash")) or prompt_hash != str(row.get("vision_plus_facts_text_hash")):
            local.append("paired_prompt_hash")
        if prompt_hash != str(facts.get("text_prompt_sha256")) or prompt_hash != str(vision.get("text_prompt_sha256")):
            local.append("checkpoint_prompt_hash")
        if bool_value(facts.get("image_present")) or not bool_value(vision.get("image_present")):
            local.append("image_presence")
        if facts.get("request_content_structure") != [{"role": "user", "content": [{"type": "text", "text": row["text_prompt"]}]}]:
            local.append("facts_content_structure")
        expected_vision_content = [{"role": "user", "content": [{"type": "image", "image": row["image_path"]}, {"type": "text", "text": row["text_prompt"]}]}]
        if vision.get("request_content_structure") != expected_vision_content:
            local.append("vision_content_structure")
        other_label, other_identity, other_color = other_fields(row)
        row.update({"other_identity_label": other_label, "other_identity": other_identity, "other_color": other_color})
        if local:
            failures.append({"case_id": cid, "failures": local})
            continue
        is_interference = bool_value(row["facts_only_correct"]) and not bool_value(row["vision_facts_correct"])
        valid = not bool_value(row["vision_facts_invalid"]) and not bool_value(row["vision_facts_format_violation"]) and not bool_value(row["vision_facts_semantic_truncation"])
        if is_interference and valid and str(row["vision_facts_prediction"]) == other_color:
            row["mechanism_sample"] = "wrong_target_color"
            main.append(row)
        elif is_interference and valid and str(row["vision_facts_prediction"]) not in {str(row["color_A"]), str(row["color_B"])}:
            row["mechanism_sample"] = "third_color_excluded"
            third.append(row)
    if failures:
        raise MechanismStop("PAIRED_AUDIT_FAILED", {"failures": failures[:50]})
    main.sort(key=lambda row: str(row["case_id"]))
    third.sort(key=lambda row: str(row["case_id"]))
    if len(main) != EXPECTED_MAIN_CASES or len(third) != EXPECTED_THIRD_COLOR_CASES:
        raise MechanismStop("MECHANISM_CASE_COUNT_MISMATCH", {"wrong_target": len(main), "third_color": len(third)})
    main_reversal = sum(bool_value(row.get("failure_rescue_reversed_by_vision")) for row in main)
    third_reversal = sum(bool_value(row.get("failure_rescue_reversed_by_vision")) for row in third)
    audit = {
        "passed": True,
        "paired_cases_read": len(paired),
        "main_wrong_target_color_cases": len(main),
        "main_rescue_reversal_cases": main_reversal,
        "excluded_third_color_cases": len(third),
        "excluded_third_color_rescue_reversal_cases": third_reversal,
        "all_rescue_reversal_cases": main_reversal + third_reversal,
        "facts_text_equals_vision_text_byte_for_byte": True,
        "only_input_difference": "Image+Facts adds the original image content block",
        "frozen_fact_fields_verified": ["target_identity", "other_identity", "target_color", "other_color"],
        "source_files": {
            "paired": {"path": str(paired_path), "sha256": sha256_file(paired_path)},
            "facts": {"path": str(facts_path), "sha256": sha256_file(facts_path)},
            "vision": {"path": str(vision_path), "sha256": sha256_file(vision_path)},
        },
    }
    return main, third, audit, facts_by_id, vision_by_id


def build_protocol(frozen_audit: Mapping[str, Any], paired_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "realbind_qwen35_visual_interference_mechanism_v1",
        "experiment": "Qwen3.5-4B RealBind visual-interference prediction-residual mechanism test",
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "path": str(MODEL_DIR),
            "dtype": "bfloat16",
            "chat_template": "completed visual-interference pilot renderer with enable_thinking=False",
        },
        "generation": {"do_sample": False, "use_cache": False, "max_new_tokens": 16, "batch_size": 1},
        "selection": {
            "main": "Facts-only correct AND Image+Facts wrong AND Image+Facts prediction equals frozen other-object color",
            "third_color_excluded_from_main": True,
            "paired_audit": dict(paired_audit),
        },
        "residual_site": {
            "position": "final processed prompt token used to predict the first answer token",
            "layers": "all true text decoder block outputs, indexed 0..num_hidden_layers-1",
            "state": "post-block residual before the model final norm",
            "patch": "replace exactly one prediction-position residual vector; no other token or state is changed",
        },
        "logit_lens": {
            "projection": "model-native final text RMSNorm then tied lm_head",
            "color_margin": "full candidate score(target color) - full candidate score(other-object color)",
            "identity_margin": "full teacher-forced candidate score(target identity) - full teacher-forced candidate score(other identity)",
            "multi_token_rule": "sum token log-probabilities over the complete assistant candidate sequence",
            "single_token_rule": "exact prediction-position token log-probability",
        },
        "classification_rule": CLASSIFICATION_RULE,
        "best_layer_rules": {
            "divergence": "argmax across-case mean(F margin - VF margin), ties resolved to the earliest layer",
            "causal_window": "longest contiguous run at the global maximum rate, ties resolved to the earliest run",
        },
        "status_rule": {
            "CAUSAL_RESIDUAL_BRIDGE_FOUND": "no-op PASS and at least one layer has both nonzero forward rescue and nonzero reverse corruption",
            "DESCRIPTIVE_DIVERGENCE_ONLY": "no-op PASS, no bidirectional causal layer, but paired answer-aligned trajectories differ",
            "NO_CLEAR_MECHANISTIC_SIGNAL": "no-op PASS and neither criterion above is met",
        },
        "frozen_inputs": dict(frozen_audit),
        "scorer": {"revision": SCORER_REVISION, "implementation": str(SCORER_PATH), "llm_judge": False},
        "forbidden": ["third-color main statistics", "attention-head search", "KV sweep", "position-ID intervention", "recurrent-state intervention", "training", "repair", "DeepSeek", "prompt change"],
    }


def extract_hidden(output: Any) -> Any:
    import torch

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, dict):
        for key in ("hidden_states", "last_hidden_state"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"unsupported decoder block output: {type(output)}")


def replace_hidden(output: Any, patched: Any) -> Any:
    if isinstance(output, tuple):
        return (patched,) + output[1:]
    if isinstance(output, list):
        return [patched] + output[1:]
    if isinstance(output, dict):
        value = dict(output)
        if "hidden_states" in value:
            value["hidden_states"] = patched
        elif "last_hidden_state" in value:
            value["last_hidden_state"] = patched
        else:
            raise TypeError("decoder block mapping lacks hidden state")
        return value
    return patched


def discover_decoder(model: Any) -> tuple[list[Any], list[str], list[str], str, Any, str, Any]:
    import torch.nn as nn

    config = getattr(model.config, "text_config", model.config)
    count = int(config.num_hidden_layers)
    candidates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == count:
            candidates.append((name, module))
    if not candidates:
        raise MechanismStop("DECODER_LAYER_DISCOVERY_FAILED", {"expected_layers": count})
    candidates.sort(key=lambda item: ("language_model" not in item[0], len(item[0])))
    layer_root, module_list = candidates[0]
    layers = list(module_list)
    layer_paths = [f"{layer_root}.{index}" for index in range(len(layers))]
    layer_types = list(getattr(config, "layer_types", []) or [])
    if len(layer_types) != len(layers):
        layer_types = ["unknown"] * len(layers)
    parent_path = layer_root.rsplit(".", 1)[0]
    parent = model.get_submodule(parent_path)
    final_norm = getattr(parent, "norm", None)
    if final_norm is None:
        raise MechanismStop("FINAL_NORM_DISCOVERY_FAILED", {"layer_root": layer_root, "parent_path": parent_path})
    norm_path = f"{parent_path}.norm"
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise MechanismStop("LM_HEAD_DISCOVERY_FAILED")
    head_path = next((name for name, module in model.named_modules() if module is lm_head), "get_output_embeddings")
    if len(layers) != EXPECTED_LAYERS:
        raise MechanismStop("DECODER_LAYER_COUNT_MISMATCH", {"observed": len(layers), "expected": EXPECTED_LAYERS})
    return layers, layer_types, layer_paths, norm_path, final_norm, head_path, lm_head


def module_execution_device(module: Any, fallback: Any) -> Any:
    hook = getattr(module, "_hf_hook", None)
    value = getattr(hook, "execution_device", None)
    if value is not None:
        return value
    for parameter in module.parameters(recurse=False):
        if str(parameter.device) != "meta":
            return parameter.device
    return fallback


def render_condition(processor: Any, case: Mapping[str, Any], condition: str, assistant: str | None = None) -> str:
    from experiments import qwen35_kv_gdn_factorial_pilot as frozen_base

    if condition == VISION:
        return frozen_base.render_chat(processor, str(case["image_path"]), str(case["text_prompt"]), assistant)
    if condition != FACTS:
        raise ValueError(condition)
    content = [{"role": "user", "content": [{"type": "text", "text": str(case["text_prompt"])}]}]
    if assistant is not None:
        content.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})
    return processor.apply_chat_template(content, tokenize=False, add_generation_prompt=assistant is None, enable_thinking=False)


def encode_condition(processor: Any, case: Mapping[str, Any], condition: str, device: Any, assistant: str | None = None) -> tuple[str, dict[str, Any]]:
    import torch
    from experiments import qwen35_kv_gdn_factorial_pilot as frozen_base

    rendered = render_condition(processor, case, condition, assistant)
    if condition == VISION:
        encoded = frozen_base.encode_single(processor, rendered, str(case["image_path"]), device)
    else:
        encoded = processor(text=rendered, return_tensors="pt")
        encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
    return rendered, encoded


def clone_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def input_ids(inputs: Mapping[str, Any]) -> list[int]:
    return [int(x) for x in inputs["input_ids"][0].detach().cpu().tolist()]


def candidate_spec(processor: Any, case: Mapping[str, Any], condition: str, candidate: str, device: Any) -> dict[str, Any]:
    from experiments import qwen35_kv_gdn_factorial_pilot as frozen_base

    prompt_rendered, prompt_inputs = encode_condition(processor, case, condition, device, None)
    full_rendered, full_inputs = encode_condition(processor, case, condition, device, candidate)
    prompt_ids = input_ids(prompt_inputs)
    full_ids = input_ids(full_inputs)
    try:
        suffix = frozen_base.candidate_suffix(processor, prompt_ids, full_ids)
    except Exception as error:
        raise MechanismStop("CANDIDATE_SERIALIZATION_FAILED", {"case_id": case["case_id"], "condition": condition, "candidate": candidate, "error": str(error)}) from error
    positions = [len(prompt_ids) - 1 + offset for offset in range(len(suffix))]
    return {
        "candidate": candidate,
        "candidate_token_ids": suffix,
        "candidate_token_strings": [getattr(processor, "tokenizer", processor).decode([token]) for token in suffix],
        "prompt_length": len(prompt_ids),
        "prediction_positions": positions,
        "prompt_rendered_sha256": sha256_bytes(prompt_rendered.encode("utf-8")),
        "full_rendered_sha256": sha256_bytes(full_rendered.encode("utf-8")),
        "full_inputs": full_inputs,
    }


def capture_prompt_states(model: Any, layers: Sequence[Any], inputs: Mapping[str, Any], pred_position: int, device: Any) -> tuple[Any, Any]:
    import torch

    captured: dict[int, Any] = {}
    handles = []
    for index, layer in enumerate(layers):
        def make_hook(layer_index: int):
            def hook(_module: Any, _args: Any, output: Any) -> Any:
                hidden = extract_hidden(output)
                captured[layer_index] = hidden[0, pred_position].detach().cpu().clone()
                return output
            return hook
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        with torch.inference_mode():
            keep = torch.tensor([pred_position], device=device, dtype=torch.long)
            output = model(**clone_inputs(inputs), return_dict=True, use_cache=False, logits_to_keep=keep)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(len(layers))):
        raise MechanismStop("HIDDEN_CAPTURE_INCOMPLETE", {"captured": sorted(captured), "expected": len(layers)})
    states = torch.stack([captured[index] for index in range(len(layers))], dim=0)
    return states, output.logits[0, 0].detach().float().cpu().clone()


def capture_candidate_states(model: Any, layers: Sequence[Any], spec: Mapping[str, Any]) -> Any:
    import torch

    positions = [int(value) for value in spec["prediction_positions"]]
    position_tensor = torch.tensor(positions, device=spec["full_inputs"]["input_ids"].device, dtype=torch.long)
    captured: dict[int, Any] = {}
    handles = []
    for index, layer in enumerate(layers):
        def make_hook(layer_index: int):
            def hook(_module: Any, _args: Any, output: Any) -> Any:
                hidden = extract_hidden(output)
                captured[layer_index] = hidden[0].index_select(0, position_tensor).detach().cpu().clone()
                return output
            return hook
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        with torch.inference_mode():
            model(**clone_inputs(spec["full_inputs"]), return_dict=True, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(len(layers))):
        raise MechanismStop("CANDIDATE_HIDDEN_CAPTURE_INCOMPLETE", {"candidate": spec["candidate"]})
    return torch.stack([captured[index] for index in range(len(layers))], dim=0)


def lens_token_logps(
    states: Any,
    token_ids: Sequence[int],
    final_norm: Any,
    lm_head: Any,
    fallback_device: Any,
    chunk_size: int = 4,
) -> tuple[np.ndarray, Any | None]:
    """Return [layer, token-position] log-probs under the native final norm/head."""
    import torch

    if states.ndim == 2:
        states = states[:, None, :]
    layers, positions, hidden = states.shape
    if positions != len(token_ids):
        raise ValueError(f"states/token mismatch: {states.shape=} {len(token_ids)=}")
    flat = states.reshape(layers * positions, hidden)
    repeated_ids = torch.tensor(list(token_ids) * layers, dtype=torch.long)
    norm_device = module_execution_device(final_norm, fallback_device)
    head_device = module_execution_device(lm_head, norm_device)
    values: list[Any] = []
    last_full_logits = None
    with torch.inference_mode():
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            x = flat[start:end].to(norm_device)
            normalized = final_norm(x)
            logits = lm_head(normalized.to(head_device)).float()
            ids = repeated_ids[start:end].to(logits.device)
            logps = torch.log_softmax(logits, dim=-1).gather(1, ids[:, None])[:, 0]
            values.append(logps.detach().cpu())
            if positions == 1 and end == flat.shape[0]:
                last_full_logits = logits[-1].detach().cpu().clone()
    array = torch.cat(values).reshape(layers, positions).numpy().astype(np.float64)
    return array, last_full_logits


def native_single_position_logits(
    state: Any,
    final_norm: Any,
    lm_head: Any,
    fallback_device: Any,
) -> Any:
    """Project one residual with the same [batch, sequence, hidden] shape as the model.

    Keeping both singleton leading dimensions matters for the bf16 lm-head kernel:
    flattening multiple layer probes into one matrix can legitimately select a
    different GEMM kernel and differ by one bf16 ULP even at the final layer.
    This helper is therefore used for the final-layer reconstruction gate.
    """
    import torch

    norm_device = module_execution_device(final_norm, fallback_device)
    head_device = module_execution_device(lm_head, norm_device)
    with torch.inference_mode():
        value = state.reshape(1, 1, -1).to(norm_device)
        normalized = final_norm(value)
        logits = lm_head(normalized.to(head_device))
    return logits[0, 0].detach().float().cpu().clone()


def candidate_layer_scores(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    final_norm: Any,
    lm_head: Any,
    device: Any,
    case: Mapping[str, Any],
    condition: str,
    candidate: str,
    prompt_states: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = candidate_spec(processor, case, condition, candidate, device)
    token_ids = [int(value) for value in spec["candidate_token_ids"]]
    if len(token_ids) == 1:
        token_logps, _ = lens_token_logps(prompt_states, token_ids, final_norm, lm_head, device)
        source = "prompt_prediction_residual_single_token_exact"
        prefix_max = 0.0
    else:
        candidate_states = capture_candidate_states(model, layers, spec)
        prefix_max = float((candidate_states[:, 0].float() - prompt_states.float()).abs().max().item())
        token_logps, _ = lens_token_logps(candidate_states, token_ids, final_norm, lm_head, device)
        source = "full_candidate_teacher_forced_sequence"
    scores = token_logps.sum(axis=1)
    meta = {
        key: value for key, value in spec.items() if key != "full_inputs"
    }
    meta.update({"scoring_mode": source, "prefix_state_max_abs_difference_from_prompt_capture": prefix_max})
    return scores, meta


def parse_generated(
    processor: Any,
    case: Mapping[str, Any],
    sequences: Any,
    input_length: int,
    semantic_vocabulary: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    continuation = sequences[0, input_length:]
    raw_with_special_tokens, decoded, token_ids = qwen_baseline.frozen_baseline.decode_generation(processor, continuation)
    parsed = qwen_baseline.parse_semantic_answer(decoded, "color", str(case["target_color"]), family_view(case), semantic_vocabulary)
    return {
        # Match the frozen baseline schema: raw_response is the decoded semantic
        # continuation, while the special-token-preserving form is separate.
        "raw_response": decoded,
        "raw_response_with_special_tokens": raw_with_special_tokens,
        "decoded_response": decoded,
        "generated_token_ids": token_ids,
        "normalized_prediction": parsed["normalized_prediction"],
        "invalid_prediction": bool(parsed["invalid_prediction"]),
        "parser_reason": parsed["parser_reason"],
        "correct": parsed["normalized_prediction"] == str(case["target_color"]) and not bool(parsed["invalid_prediction"]),
    }


def generate_with_patch(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    inputs: Mapping[str, Any],
    case: Mapping[str, Any],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    target_token_id: int,
    other_token_id: int,
    layer_index: int | None = None,
    replacement: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    import torch

    pred_position = int(inputs["input_ids"].shape[-1] - 1)
    handle = None
    write_meta: dict[str, Any] = {"patch_applied": False, "patch_call_count": 0}
    if layer_index is not None:
        if replacement is None:
            raise ValueError("patch layer requires replacement")
        patched_once = False

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            nonlocal patched_once
            write_meta["patch_call_count"] += 1
            if patched_once:
                return output
            hidden = extract_hidden(output)
            if hidden.shape[1] <= pred_position:
                raise RuntimeError("prediction position outside hidden output")
            edited = hidden.clone()
            before = hidden[:, pred_position, :]
            source = replacement.to(device=edited.device, dtype=edited.dtype).reshape(1, -1)
            edited[:, pred_position, :] = source
            actual = edited[:, pred_position, :]
            write_meta.update(
                {
                    "patch_applied": True,
                    "source_norm": float(torch.linalg.vector_norm(source.float()).detach().cpu()),
                    "target_before_norm": float(torch.linalg.vector_norm(before.float()).detach().cpu()),
                    "actual_delta_norm": float(torch.linalg.vector_norm((actual - before).float()).detach().cpu()),
                    "write_exact": bool(torch.equal(actual, source)),
                    "patched_sequence_length": int(hidden.shape[1]),
                }
            )
            patched_once = True
            return replace_hidden(output, edited)

        handle = layers[int(layer_index)].register_forward_hook(hook)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(
                **clone_inputs(inputs),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=USE_CACHE,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    elapsed = time.perf_counter() - started
    if not generated.scores:
        raise MechanismStop("GENERATION_SCORES_MISSING", {"case_id": case["case_id"], "layer": layer_index})
    first_logits = generated.scores[0][0].detach().float().cpu().clone()
    first_logps = torch.log_softmax(first_logits, dim=-1)
    result = parse_generated(processor, case, generated.sequences, int(inputs["input_ids"].shape[-1]), semantic_vocabulary)
    result.update(
        {
            "target_color_score": float(first_logps[int(target_token_id)]),
            "other_color_score": float(first_logps[int(other_token_id)]),
            "color_margin": float(first_logps[int(target_token_id)] - first_logps[int(other_token_id)]),
            "first_answer_token_id": int(generated.sequences[0, int(inputs["input_ids"].shape[-1])].detach().cpu()),
            "latency_seconds": elapsed,
            "layer": layer_index,
            **write_meta,
        }
    )
    return result, first_logits


def checkpoint_rows(path: Path, protocol_fingerprint: str, key_fields: Sequence[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row.get("protocol_fingerprint")) != protocol_fingerprint:
            raise MechanismStop("CHECKPOINT_PROTOCOL_MISMATCH", {"path": str(path)})
        key = tuple(str(row.get(field)) for field in key_fields)
        if key in result:
            raise MechanismStop("CHECKPOINT_DUPLICATE", {"path": str(path), "key": key})
        result[key] = row
    return result


def save_checkpoint(path: Path, rows: Mapping[tuple[str, ...], Mapping[str, Any]]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    write_jsonl(path, ordered)


def state_cache_path(root: Path, case_id: str) -> Path:
    return root / f"{case_id}.pt"


def save_state_cache(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_state_cache(path: Path, protocol_fingerprint: str, layer_paths: Sequence[str]) -> dict[str, Any] | None:
    import torch

    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("protocol_fingerprint")) != protocol_fingerprint or list(payload.get("layer_paths", [])) != list(layer_paths):
        raise MechanismStop("STATE_CACHE_PROTOCOL_MISMATCH", {"path": str(path)})
    return payload


def full_candidate_score_with_patch(
    model: Any,
    layers: Sequence[Any],
    spec: Mapping[str, Any],
    layer_index: int,
    replacement: Any,
) -> float:
    """Exact full-sequence score used only if a selected color is multi-token."""
    import torch

    pred_position = int(spec["prompt_length"] - 1)
    patched_once = False

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        nonlocal patched_once
        if patched_once:
            return output
        hidden = extract_hidden(output)
        edited = hidden.clone()
        edited[:, pred_position, :] = replacement.to(device=edited.device, dtype=edited.dtype)
        patched_once = True
        return replace_hidden(output, edited)

    handle = layers[int(layer_index)].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            keep = torch.tensor(spec["prediction_positions"], device=spec["full_inputs"]["input_ids"].device, dtype=torch.long)
            output = model(**clone_inputs(spec["full_inputs"]), return_dict=True, use_cache=False, logits_to_keep=keep)
    finally:
        handle.remove()
    logits = output.logits[0].detach().float()
    values = []
    for offset, token_id in enumerate(spec["candidate_token_ids"]):
        values.append(torch.log_softmax(logits[offset], dim=-1)[int(token_id)])
    return float(torch.stack(values).sum().detach().cpu())


def build_case_states_and_scores(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    layer_types: Sequence[str],
    layer_paths: Sequence[str],
    final_norm: Any,
    lm_head: Any,
    device: Any,
    case: Mapping[str, Any],
    protocol_fingerprint: str,
    cache_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    cache_path = state_cache_path(cache_root, str(case["case_id"]))
    cached = load_state_cache(cache_path, protocol_fingerprint, layer_paths)
    if cached is not None:
        return cached, list(cached["color_rows"]), list(cached["identity_rows"]), dict(cached["metadata"])

    condition_states: dict[str, Any] = {}
    condition_prompt_logits: dict[str, Any] = {}
    condition_inputs: dict[str, Any] = {}
    condition_positions: dict[str, int] = {}
    token_meta: dict[str, Any] = {}
    reconstruction: dict[str, float] = {}
    score_cache: dict[tuple[str, str], np.ndarray] = {}

    _, _, target_color = str(case["target_identity_label"]), str(case["target_identity"]), str(case["target_color"])
    other_identity = str(case["other_identity"])
    other_color = str(case["other_color"])
    target_identity = str(case["target_identity"])

    for condition in (FACTS, VISION):
        rendered, inputs = encode_condition(processor, case, condition, device)
        pred = int(inputs["input_ids"].shape[-1] - 1)
        states, prompt_logits = capture_prompt_states(model, layers, inputs, pred, device)
        condition_states[condition] = states
        condition_prompt_logits[condition] = prompt_logits
        condition_inputs[condition] = inputs
        condition_positions[condition] = pred
        if int(inputs["attention_mask"][0].sum().item()) != int(inputs["input_ids"].shape[-1]):
            raise MechanismStop("UNEXPECTED_PROMPT_PADDING", {"case_id": case["case_id"], "condition": condition})
        candidates = {
            "target_color": target_color,
            "other_color": other_color,
            "target_identity": target_identity,
            "other_identity": other_identity,
        }
        for label, candidate in candidates.items():
            scores, meta = candidate_layer_scores(
                processor, model, layers, final_norm, lm_head, device, case, condition, candidate, states
            )
            score_cache[(condition, label)] = scores
            token_meta[f"{condition}:{label}"] = meta
        reconstructed = native_single_position_logits(states[-1], final_norm, lm_head, device)
        reconstruction[condition] = float((reconstructed.float() - prompt_logits.float()).abs().max().item())
        if reconstruction[condition] > LENS_RECONSTRUCTION_TOLERANCE:
            raise MechanismStop(
                "LOGIT_LENS_RECONSTRUCTION_FAILED",
                {"case_id": case["case_id"], "condition": condition, "max_abs_error": reconstruction[condition]},
            )
        del rendered

    if input_ids(condition_inputs[FACTS])[-1] != input_ids(condition_inputs[VISION])[-1]:
        raise MechanismStop("PREDICTION_TOKEN_MISMATCH", {"case_id": case["case_id"]})
    color_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for layer in range(len(layers)):
        color_f_target = float(score_cache[(FACTS, "target_color")][layer])
        color_f_other = float(score_cache[(FACTS, "other_color")][layer])
        color_v_target = float(score_cache[(VISION, "target_color")][layer])
        color_v_other = float(score_cache[(VISION, "other_color")][layer])
        color_rows.append(
            {
                "protocol_fingerprint": protocol_fingerprint,
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "rescue_reversal_subset": bool_value(case.get("failure_rescue_reversed_by_vision")),
                "layer": layer,
                "layer_path": layer_paths[layer],
                "layer_type": layer_types[layer],
                "target_color": target_color,
                "other_color": other_color,
                "target_color_score_facts": color_f_target,
                "other_color_score_facts": color_f_other,
                "margin_facts": color_f_target - color_f_other,
                "target_color_score_vision_facts": color_v_target,
                "other_color_score_vision_facts": color_v_other,
                "margin_vision_facts": color_v_target - color_v_other,
                "margin_difference_facts_minus_vision": (color_f_target - color_f_other) - (color_v_target - color_v_other),
            }
        )
        id_f_target = float(score_cache[(FACTS, "target_identity")][layer])
        id_f_other = float(score_cache[(FACTS, "other_identity")][layer])
        id_v_target = float(score_cache[(VISION, "target_identity")][layer])
        id_v_other = float(score_cache[(VISION, "other_identity")][layer])
        identity_rows.append(
            {
                "protocol_fingerprint": protocol_fingerprint,
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "rescue_reversal_subset": bool_value(case.get("failure_rescue_reversed_by_vision")),
                "layer": layer,
                "layer_path": layer_paths[layer],
                "layer_type": layer_types[layer],
                "target_identity": target_identity,
                "other_identity": other_identity,
                "target_identity_score_facts": id_f_target,
                "other_identity_score_facts": id_f_other,
                "margin_facts": id_f_target - id_f_other,
                "target_identity_score_vision_facts": id_v_target,
                "other_identity_score_vision_facts": id_v_other,
                "margin_vision_facts": id_v_target - id_v_other,
                "margin_difference_facts_minus_vision": (id_f_target - id_f_other) - (id_v_target - id_v_other),
            }
        )
    metadata = {
        "case_id": case["case_id"],
        "facts_prompt_length": int(condition_inputs[FACTS]["input_ids"].shape[-1]),
        "vision_facts_prompt_length": int(condition_inputs[VISION]["input_ids"].shape[-1]),
        "facts_prediction_position": condition_positions[FACTS],
        "vision_facts_prediction_position": condition_positions[VISION],
        "prediction_token_id_facts": input_ids(condition_inputs[FACTS])[-1],
        "prediction_token_id_vision_facts": input_ids(condition_inputs[VISION])[-1],
        "prediction_token_string": getattr(processor, "tokenizer", processor).decode([input_ids(condition_inputs[FACTS])[-1]]),
        "tokenization": token_meta,
        "last_layer_logit_lens_max_abs_error": reconstruction,
    }
    payload = {
        "protocol_fingerprint": protocol_fingerprint,
        "case_id": case["case_id"],
        "layer_paths": list(layer_paths),
        "facts_states": condition_states[FACTS],
        "vision_states": condition_states[VISION],
        "color_rows": color_rows,
        "identity_rows": identity_rows,
        "metadata": metadata,
    }
    save_state_cache(cache_path, payload)
    return payload, color_rows, identity_rows, metadata


def clean_replay(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    device: Any,
    case: Mapping[str, Any],
    condition: str,
    semantic_vocabulary: Mapping[str, Sequence[str]],
    source_row: Mapping[str, Any],
    color_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Any]:
    _, inputs = encode_condition(processor, case, condition, device)
    target_ids = list(color_specs["target"]["candidate_token_ids"])
    other_ids = list(color_specs["other"]["candidate_token_ids"])
    if not target_ids or not other_ids:
        raise MechanismStop("EMPTY_COLOR_CANDIDATE", {"case_id": case["case_id"], "condition": condition})
    result, logits = generate_with_patch(
        processor, model, layers, inputs, case, semantic_vocabulary, int(target_ids[0]), int(other_ids[0])
    )
    checks = {
        "generated_token_ids_exact": list(result["generated_token_ids"]) == [int(value) for value in source_row["generated_token_ids"]],
        "raw_response_exact": str(result["raw_response"]) == str(source_row["raw_response"]),
        "normalized_prediction_exact": result["normalized_prediction"] == source_row["normalized_prediction"],
        "correct_exact": bool(result["correct"]) == bool_value(source_row["correct"]),
    }
    row = {
        "case_id": case["case_id"],
        "condition": condition,
        "source_prediction": source_row["normalized_prediction"],
        "replayed_prediction": result["normalized_prediction"],
        "source_generated_token_ids": source_row["generated_token_ids"],
        "replayed_generated_token_ids": result["generated_token_ids"],
        "checks": checks,
        "passed": all(checks.values()),
        **result,
    }
    if not row["passed"]:
        raise MechanismStop("CLEAN_BASELINE_REPLAY_FAILED", row)
    return row, logits


def patch_result_row(
    protocol_fingerprint: str,
    case: Mapping[str, Any],
    direction: str,
    layer: int,
    source_condition: str,
    target_condition: str,
    result: Mapping[str, Any],
    target_color_score: float,
    other_color_score: float,
) -> dict[str, Any]:
    return {
        "protocol_fingerprint": protocol_fingerprint,
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "direction": direction,
        "source_condition": source_condition,
        "target_condition": target_condition,
        "layer": layer,
        "rescue_reversal_subset": bool_value(case.get("failure_rescue_reversed_by_vision")),
        "target_identity": case["target_identity"],
        "other_identity": case["other_identity"],
        "target_color": case["target_color"],
        "other_color": case["other_color"],
        "patched_target_color_score": target_color_score,
        "patched_other_color_score": other_color_score,
        "patched_color_margin": target_color_score - other_color_score,
        "final_raw_output": result["raw_response"],
        "final_generated_token_ids": result["generated_token_ids"],
        "final_normalized_answer": result["normalized_prediction"],
        "final_correct": bool(result["correct"]),
        "final_invalid": bool(result["invalid_prediction"]),
        "parser_reason": result["parser_reason"],
        "first_answer_token_id": result["first_answer_token_id"],
        "source_norm": result.get("source_norm"),
        "target_before_norm": result.get("target_before_norm"),
        "actual_delta_norm": result.get("actual_delta_norm"),
        "write_exact": result.get("write_exact"),
        "patch_applied": result.get("patch_applied"),
        "patch_call_count": result.get("patch_call_count"),
        "latency_seconds": result.get("latency_seconds"),
    }


def run_noop_phase(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    device: Any,
    cases: Sequence[Mapping[str, Any]],
    state_payloads: Mapping[str, Mapping[str, Any]],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    source_facts: Mapping[str, Mapping[str, Any]],
    source_vision: Mapping[str, Mapping[str, Any]],
    protocol_fingerprint: str,
    out_path: Path,
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    rows = checkpoint_rows(out_path, protocol_fingerprint, ("case_id", "condition", "layer"))
    clean_by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        cid = str(case["case_id"])
        payload = state_payloads[cid]
        color_meta = payload["metadata"]["tokenization"]
        specs = {
            FACTS: {
                "target": color_meta[f"{FACTS}:target_color"],
                "other": color_meta[f"{FACTS}:other_color"],
            },
            VISION: {
                "target": color_meta[f"{VISION}:target_color"],
                "other": color_meta[f"{VISION}:other_color"],
            },
        }
        clean_by_case[cid] = {}
        for condition, source_map, state_key in (
            (FACTS, source_facts, "facts_states"),
            (VISION, source_vision, "vision_states"),
        ):
            clean, clean_logits = clean_replay(
                processor, model, layers, device, case, condition, semantic_vocabulary, source_map[cid], specs[condition]
            )
            clean_by_case[cid][condition] = clean
            _, inputs = encode_condition(processor, case, condition, device)
            target_id = int(specs[condition]["target"]["candidate_token_ids"][0])
            other_id = int(specs[condition]["other"]["candidate_token_ids"][0])
            states = payload[state_key]
            for layer in range(len(layers)):
                key = (cid, condition, str(layer))
                if key in rows:
                    continue
                patched, patched_logits = generate_with_patch(
                    processor,
                    model,
                    layers,
                    inputs,
                    case,
                    semantic_vocabulary,
                    target_id,
                    other_id,
                    layer,
                    states[layer],
                )
                max_abs = float((patched_logits - clean_logits).abs().max().item())
                exact_logits = bool(__import__("torch").equal(patched_logits, clean_logits))
                checks = {
                    "full_first_token_logits_exact": exact_logits,
                    "generated_token_ids_exact": list(patched["generated_token_ids"]) == list(clean["generated_token_ids"]),
                    "raw_response_exact": str(patched["raw_response"]) == str(clean["raw_response"]),
                    "normalized_prediction_exact": patched["normalized_prediction"] == clean["normalized_prediction"],
                    "correct_exact": bool(patched["correct"]) == bool(clean["correct"]),
                    "write_exact": bool(patched.get("write_exact")),
                    "patch_applied": bool(patched.get("patch_applied")),
                }
                row = {
                    "protocol_fingerprint": protocol_fingerprint,
                    "case_id": cid,
                    "family_id": case["family_id"],
                    "condition": condition,
                    "layer": layer,
                    "max_abs_first_token_logit_difference": max_abs,
                    "baseline_generated_token_ids": clean["generated_token_ids"],
                    "patched_generated_token_ids": patched["generated_token_ids"],
                    "baseline_prediction": clean["normalized_prediction"],
                    "patched_prediction": patched["normalized_prediction"],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
                rows[key] = row
                save_checkpoint(out_path, rows)
                print(f"NOOP {len(rows)}/{len(cases) * len(layers) * 2}: {cid} {condition} L{layer}", flush=True)
                if not row["passed"]:
                    audit = {"passed": False, "failure": row, "completed_rows": len(rows)}
                    return rows, clean_by_case, audit
    failures = [row for row in rows.values() if not bool_value(row.get("passed"))]
    audit = {
        "passed": not failures and len(rows) == len(cases) * len(layers) * 2,
        "rows": len(rows),
        "expected_rows": len(cases) * len(layers) * 2,
        "max_abs_first_token_logit_difference": max((float(row["max_abs_first_token_logit_difference"]) for row in rows.values()), default=None),
        "generated_output_exact_rate": sum(bool_value(row["checks"]["generated_token_ids_exact"]) for row in rows.values()) / len(rows) if rows else None,
        "failures": failures[:20],
    }
    return rows, clean_by_case, audit


def patched_color_scores(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    device: Any,
    case: Mapping[str, Any],
    condition: str,
    layer: int,
    replacement: Any,
    result: Mapping[str, Any],
    target_meta: Mapping[str, Any],
    other_meta: Mapping[str, Any],
) -> tuple[float, float, str]:
    if len(target_meta["candidate_token_ids"]) == 1 and len(other_meta["candidate_token_ids"]) == 1:
        return float(result["target_color_score"]), float(result["other_color_score"]), "single_token_generation_first_step"
    target_spec = candidate_spec(processor, case, condition, str(case["target_color"]), device)
    other_spec = candidate_spec(processor, case, condition, str(case["other_color"]), device)
    target_score = full_candidate_score_with_patch(model, layers, target_spec, layer, replacement)
    other_score = full_candidate_score_with_patch(model, layers, other_spec, layer, replacement)
    return target_score, other_score, "full_candidate_teacher_forced_sequence"


def run_cross_patch_phase(
    processor: Any,
    model: Any,
    layers: Sequence[Any],
    device: Any,
    cases: Sequence[Mapping[str, Any]],
    state_payloads: Mapping[str, Mapping[str, Any]],
    semantic_vocabulary: Mapping[str, Sequence[str]],
    protocol_fingerprint: str,
    forward_path: Path,
    reverse_path: Path,
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    forward = checkpoint_rows(forward_path, protocol_fingerprint, ("case_id", "layer"))
    reverse = checkpoint_rows(reverse_path, protocol_fingerprint, ("case_id", "layer"))
    expected = len(cases) * len(layers)
    for case in cases:
        cid = str(case["case_id"])
        payload = state_payloads[cid]
        meta = payload["metadata"]["tokenization"]
        facts_states = payload["facts_states"]
        vision_states = payload["vision_states"]
        _, facts_inputs = encode_condition(processor, case, FACTS, device)
        _, vision_inputs = encode_condition(processor, case, VISION, device)
        f_target = meta[f"{FACTS}:target_color"]
        f_other = meta[f"{FACTS}:other_color"]
        v_target = meta[f"{VISION}:target_color"]
        v_other = meta[f"{VISION}:other_color"]
        for layer in range(len(layers)):
            fwd_key = (cid, str(layer))
            if fwd_key not in forward:
                result, _ = generate_with_patch(
                    processor,
                    model,
                    layers,
                    vision_inputs,
                    case,
                    semantic_vocabulary,
                    int(v_target["candidate_token_ids"][0]),
                    int(v_other["candidate_token_ids"][0]),
                    layer,
                    facts_states[layer],
                )
                target_score, other_score, score_mode = patched_color_scores(
                    processor, model, layers, device, case, VISION, layer, facts_states[layer], result, v_target, v_other
                )
                row = patch_result_row(
                    protocol_fingerprint, case, "FACTS_TO_VISION", layer, FACTS, VISION, result, target_score, other_score
                )
                row.update({"score_mode": score_mode, "rescue": bool(result["correct"])})
                forward[fwd_key] = row
                save_checkpoint(forward_path, forward)
                print(f"FORWARD PATCH {len(forward)}/{expected}: {cid} L{layer}", flush=True)
            rev_key = (cid, str(layer))
            if rev_key not in reverse:
                result, _ = generate_with_patch(
                    processor,
                    model,
                    layers,
                    facts_inputs,
                    case,
                    semantic_vocabulary,
                    int(f_target["candidate_token_ids"][0]),
                    int(f_other["candidate_token_ids"][0]),
                    layer,
                    vision_states[layer],
                )
                target_score, other_score, score_mode = patched_color_scores(
                    processor, model, layers, device, case, FACTS, layer, vision_states[layer], result, f_target, f_other
                )
                row = patch_result_row(
                    protocol_fingerprint, case, "VISION_TO_FACTS", layer, VISION, FACTS, result, target_score, other_score
                )
                row.update({"score_mode": score_mode, "corruption": not bool(result["correct"])})
                reverse[rev_key] = row
                save_checkpoint(reverse_path, reverse)
                print(f"REVERSE PATCH {len(reverse)}/{expected}: {cid} L{layer}", flush=True)
    if len(forward) != expected or len(reverse) != expected:
        raise MechanismStop("PATCH_RESULTS_INCOMPLETE", {"forward": len(forward), "reverse": len(reverse), "expected": expected})
    return forward, reverse


def terminal_negative_onset(values: Sequence[float]) -> int | None:
    last_nonnegative = -1
    for index, value in enumerate(values):
        if not math.isfinite(float(value)) or float(value) >= 0.0:
            last_nonnegative = index
    onset = last_nonnegative + 1
    if onset >= len(values):
        return None
    return onset


def classify_case(color_rows: Sequence[Mapping[str, Any]], identity_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    color = sorted(color_rows, key=lambda row: int(row["layer"]))
    identity = sorted(identity_rows, key=lambda row: int(row["layer"]))
    color_f = [float(row["margin_facts"]) for row in color]
    color_v = [float(row["margin_vision_facts"]) for row in color]
    id_f = [float(row["margin_facts"]) for row in identity]
    id_v = [float(row["margin_vision_facts"]) for row in identity]
    color_onset = terminal_negative_onset(color_v)
    id_onset = terminal_negative_onset(id_v)
    label = "unclear"
    reason = "trajectory does not satisfy the fixed sign/onset rules"
    if color_onset is not None:
        f_color_positive = color_f[color_onset] > 0
        f_id_positive_at_color = id_f[color_onset] > 0
        vf_id_positive_at_color = id_v[color_onset] > 0
        if id_onset is not None and id_onset <= color_onset and id_f[id_onset] > 0 and f_color_positive:
            label = "selection-like"
            reason = "VF identity enters its terminal wrong-sign run no later than VF color while the paired Facts-only margins are target-positive"
        elif f_color_positive and f_id_positive_at_color and vf_id_positive_at_color and (id_onset is None or id_onset > color_onset):
            label = "binding/readout-like"
            reason = "VF color is terminally wrong-sign while VF identity remains target-positive at the color onset"
    return {
        "classification": label,
        "classification_reason": reason,
        "color_terminal_wrong_onset_layer": color_onset,
        "identity_terminal_wrong_onset_layer": id_onset,
        "final_color_margin_facts": color_f[-1],
        "final_color_margin_vision_facts": color_v[-1],
        "final_identity_margin_facts": id_f[-1],
        "final_identity_margin_vision_facts": id_v[-1],
        "classification_rule": CLASSIFICATION_RULE,
    }


def longest_max_window(layer_rows: Sequence[Mapping[str, Any]], rate_field: str) -> dict[str, Any]:
    rows = sorted(layer_rows, key=lambda row: int(row["layer"]))
    if not rows:
        return {"start": None, "end": None, "rate": None, "layers": []}
    maximum = max(float(row[rate_field]) for row in rows)
    runs: list[list[int]] = []
    current: list[int] = []
    for row in rows:
        layer = int(row["layer"])
        if math.isclose(float(row[rate_field]), maximum, rel_tol=0.0, abs_tol=1e-15):
            if current and layer != current[-1] + 1:
                runs.append(current)
                current = []
            current.append(layer)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    runs.sort(key=lambda run: (-len(run), run[0]))
    best = runs[0]
    return {"start": best[0], "end": best[-1], "rate": maximum, "layers": best, "all_maximal_runs": runs}


def grouped_mean(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(float(row[field]))
    return [float(np.mean(by_layer[layer])) for layer in sorted(by_layer)]


def write_figures(
    out: Mapping[str, Path],
    color_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    rescue_rows: Sequence[Mapping[str, Any]],
    corruption_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted({int(row["layer"]) for row in color_rows})
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(layers, grouped_mean(color_rows, "margin_facts"), marker="o", markersize=3, label="Facts-only")
    plt.plot(layers, grouped_mean(color_rows, "margin_vision_facts"), marker="o", markersize=3, label="Image+Facts")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Decoder layer (post-block residual)")
    plt.ylabel("Target − other color score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out["color_figure"], dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(layers, grouped_mean(identity_rows, "margin_facts"), marker="o", markersize=3, label="Facts-only")
    plt.plot(layers, grouped_mean(identity_rows, "margin_vision_facts"), marker="o", markersize=3, label="Image+Facts")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Decoder layer (post-block residual)")
    plt.ylabel("Target − other identity score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out["identity_figure"], dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    plt.plot([int(row["layer"]) for row in rescue_rows], [float(row["rescue_rate"]) for row in rescue_rows], marker="o")
    plt.xlabel("Patched decoder layer")
    plt.ylabel("Forward rescue rate")
    plt.ylim(-0.03, 1.03)
    plt.tight_layout()
    plt.savefig(out["rescue_figure"], dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    plt.plot([int(row["layer"]) for row in corruption_rows], [float(row["corruption_rate"]) for row in corruption_rows], marker="o")
    plt.xlabel("Patched decoder layer")
    plt.ylabel("Reverse corruption rate")
    plt.ylim(-0.03, 1.03)
    plt.tight_layout()
    plt.savefig(out["corruption_figure"], dpi=220)
    plt.close()


def analyze_outputs(
    out: Mapping[str, Path],
    cases: Sequence[Mapping[str, Any]],
    color_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    forward_rows: Sequence[Mapping[str, Any]],
    reverse_rows: Sequence[Mapping[str, Any]],
    noop_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    expected_scores = len(cases) * EXPECTED_LAYERS
    if len(color_rows) != expected_scores or len(identity_rows) != expected_scores:
        raise MechanismStop("LAYERWISE_SCORE_COUNT_MISMATCH", {"color": len(color_rows), "identity": len(identity_rows), "expected": expected_scores})
    if len(forward_rows) != expected_scores or len(reverse_rows) != expected_scores:
        raise MechanismStop("PATCH_RESULT_COUNT_MISMATCH", {"forward": len(forward_rows), "reverse": len(reverse_rows), "expected": expected_scores})
    case_ids = {str(case["case_id"]) for case in cases}
    color_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identity_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in color_rows:
        color_by_case[str(row["case_id"])].append(row)
    for row in identity_rows:
        identity_by_case[str(row["case_id"])].append(row)
    if set(color_by_case) != case_ids or set(identity_by_case) != case_ids:
        raise MechanismStop("LAYERWISE_CASE_ALIGNMENT_FAILED")

    rescue_rows: list[dict[str, Any]] = []
    corruption_rows: list[dict[str, Any]] = []
    for layer in range(EXPECTED_LAYERS):
        frows = [row for row in forward_rows if int(row["layer"]) == layer]
        rrows = [row for row in reverse_rows if int(row["layer"]) == layer]
        rescue_count = sum(bool_value(row.get("rescue")) for row in frows)
        corruption_count = sum(bool_value(row.get("corruption")) for row in rrows)
        rescue_rows.append({"layer": layer, "N": len(frows), "rescue_count": rescue_count, "rescue_rate": rescue_count / len(frows)})
        corruption_rows.append({"layer": layer, "N": len(rrows), "corruption_count": corruption_count, "corruption_rate": corruption_count / len(rrows)})
    write_csv(out["rescue_csv"], rescue_rows, ("layer", "N", "rescue_count", "rescue_rate"))
    write_csv(out["corruption_csv"], corruption_rows, ("layer", "N", "corruption_count", "corruption_rate"))

    classification_rows: list[dict[str, Any]] = []
    case_by_id = {str(case["case_id"]): case for case in cases}
    for cid in sorted(case_ids):
        value = classify_case(color_by_case[cid], identity_by_case[cid])
        classification_rows.append(
            {
                "case_id": cid,
                "family_id": case_by_id[cid]["family_id"],
                "rescue_reversal_subset": bool_value(case_by_id[cid].get("failure_rescue_reversed_by_vision")),
                **value,
            }
        )
    write_jsonl(out["classification"], classification_rows)

    mean_color_f = grouped_mean(color_rows, "margin_facts")
    mean_color_v = grouped_mean(color_rows, "margin_vision_facts")
    mean_identity_f = grouped_mean(identity_rows, "margin_facts")
    mean_identity_v = grouped_mean(identity_rows, "margin_vision_facts")
    color_divergence = [a - b for a, b in zip(mean_color_f, mean_color_v)]
    identity_divergence = [a - b for a, b in zip(mean_identity_f, mean_identity_v)]
    best_color = int(max(range(EXPECTED_LAYERS), key=lambda layer: (color_divergence[layer], -layer)))
    best_identity = int(max(range(EXPECTED_LAYERS), key=lambda layer: (identity_divergence[layer], -layer)))
    rescue_window = longest_max_window(rescue_rows, "rescue_rate")
    corruption_window = longest_max_window(corruption_rows, "corruption_rate")
    bridge_layers = [
        layer for layer in range(EXPECTED_LAYERS)
        if rescue_rows[layer]["rescue_count"] > 0 and corruption_rows[layer]["corruption_count"] > 0
    ]
    classification_counts = Counter(str(row["classification"]) for row in classification_rows)
    trajectory_diff = any(abs(value) > 0 for value in color_divergence) or any(abs(value) > 0 for value in identity_divergence)
    if bool_value(noop_audit.get("passed")) and bridge_layers:
        status = "CAUSAL_RESIDUAL_BRIDGE_FOUND"
    elif bool_value(noop_audit.get("passed")) and trajectory_diff:
        status = "DESCRIPTIVE_DIVERGENCE_ONLY"
    else:
        status = "NO_CLEAR_MECHANISTIC_SIGNAL"
    summary = {
        "cases": len(cases),
        "rescue_reversal_cases_main": sum(bool_value(case.get("failure_rescue_reversed_by_vision")) for case in cases),
        "best_color_divergence_layer": best_color,
        "best_color_divergence_value": color_divergence[best_color],
        "best_identity_divergence_layer": best_identity,
        "best_identity_divergence_value": identity_divergence[best_identity],
        "best_rescue_window": rescue_window,
        "best_reverse_corruption_window": corruption_window,
        "bidirectional_causal_layers": bridge_layers,
        "classification_counts": dict(classification_counts),
        "no_op_patch": dict(noop_audit),
        "mean_color_margin_facts": mean_color_f,
        "mean_color_margin_vision_facts": mean_color_v,
        "mean_identity_margin_facts": mean_identity_f,
        "mean_identity_margin_vision_facts": mean_identity_v,
        "final_status": status,
    }
    write_figures(out, color_rows, identity_rows, rescue_rows, corruption_rows)

    def layer_span(window: Mapping[str, Any]) -> str:
        if window.get("start") is None:
            return "NA"
        return str(window["start"]) if window["start"] == window["end"] else f"{window['start']}–{window['end']}"

    lines = [
        "# Qwen3.5 RealBind visual-interference mechanism test",
        "",
        "Main statistics contain exactly the completed pilot's nine valid wrong-target-color cases. The two third-color cases are saved separately and excluded. Of the six total behavioral rescue-reversal cases, four are in this main sample and two are excluded third-color cases.",
        "",
        "## Fixed protocol",
        "",
        "The two clean runs use byte-identical text and differ only by the original image. The analyzed state is the final processed prompt token's post-block residual, immediately before the first answer token. Layers are the 32 true decoder blocks indexed 0–31. Logit-lens scores use the model-native final norm and tied lm_head. Multi-token candidates use summed full-sequence teacher-forced token log-probabilities.",
        "",
        "Case labels were fixed before outcome inspection: selection-like requires a terminal wrong-identity sign onset no later than the wrong-color onset, with paired Facts-only target-positive margins; binding/readout-like requires the color to turn terminally wrong while identity remains target-positive at that onset; all other cases are unclear. These are diagnostic labels, not explicit-token or symbolic-pipeline claims.",
        "",
        "## Results",
        "",
        f"- Cases: {len(cases)}",
        f"- Main rescue-reversal cases: {summary['rescue_reversal_cases_main']}",
        f"- Best mean color-divergence layer: {best_color} (Facts−Image margin difference {color_divergence[best_color]:.6f})",
        f"- Best mean identity-divergence layer: {best_identity} (Facts−Image margin difference {identity_divergence[best_identity]:.6f})",
        f"- Best rescue window: {layer_span(rescue_window)}, max rate {float(rescue_window['rate']):.6f}",
        f"- Best reverse-corruption window: {layer_span(corruption_window)}, max rate {float(corruption_window['rate']):.6f}",
        f"- Layers with both forward rescue and reverse corruption: {bridge_layers}",
        f"- Selection-like / binding-readout-like / unclear: {classification_counts.get('selection-like', 0)} / {classification_counts.get('binding/readout-like', 0)} / {classification_counts.get('unclear', 0)}",
        f"- No-op patch: {'PASS' if noop_audit.get('passed') else 'FAIL'}; max first-token-logit difference {noop_audit.get('max_abs_first_token_logit_difference')}",
        "",
        "## Layer-wise causal curves",
        "",
        "| Layer | Rescue | Rescue rate | Reverse corruption | Corruption rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rescue, corruption in zip(rescue_rows, corruption_rows):
        lines.append(f"| {rescue['layer']} | {rescue['rescue_count']}/{rescue['N']} | {rescue['rescue_rate']:.6f} | {corruption['corruption_count']}/{corruption['N']} | {corruption['corruption_rate']:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Layer-wise margin divergence alone establishes only a different answer-aligned residual trajectory. A successful cross-condition patch establishes that the paired residual difference at that layer is causally relevant to the behavioral error. Identity/color ordering is interpreted only through the fixed diagnostic rule and does not establish an internal symbolic R→I→C pipeline.",
            "",
            f"FINAL STATUS: **{status}**",
            "",
        ]
    )
    out["summary"].write_text("\n".join(lines), encoding="utf-8")
    return summary, status


def build_provenance(
    out: Mapping[str, Path],
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    paired_audit: Mapping[str, Any],
    frozen_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "realbind_qwen35_visual_interference_mechanism_provenance_v1",
        "mode": "dry-run",
        "model_identifier": MODEL_IDENTIFIER,
        "model_path": str(MODEL_DIR),
        "protocol_fingerprint": protocol_fingerprint,
        "paired_audit": dict(paired_audit),
        "frozen_inputs": dict(frozen_audit),
        "scorer": {
            "revision": SCORER_REVISION,
            "path": str(SCORER_PATH),
            "sha256_before": sha256_file(SCORER_PATH),
        },
        "frozen_hashes_before": {
            "families": sha256_file(FAMILIES_PATH),
            "qa": sha256_file(QA_PATH),
            "dependency": sha256_file(DEPENDENCY_PATH),
        },
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "artifacts": {name: str(path) for name, path in out.items() if name not in {"root", "state_cache"}},
        "forbidden_work_not_run": ["DeepSeek", "attention-head search", "KV sweep", "position-ID experiment", "recurrent-state experiment", "training", "repair"],
    }


def execute(args: argparse.Namespace) -> int:
    out = paths(args.output_root.expanduser().resolve())
    out["root"].mkdir(parents=True, exist_ok=True)
    families, qa_rows, frozen_audit = behavioral_pilot.validate_frozen_inputs()
    main_cases, third_cases, paired_audit, source_facts, source_vision = load_and_select_cases(families, qa_rows)
    protocol = build_protocol(frozen_audit, paired_audit)
    protocol_fingerprint = stable_hash(protocol)
    provenance = build_provenance(out, protocol, protocol_fingerprint, paired_audit, frozen_audit)
    write_json(out["protocol"], {**protocol, "protocol_fingerprint": protocol_fingerprint})
    write_json(out["provenance"], provenance)
    write_jsonl(out["selected"], main_cases)
    write_jsonl(out["excluded_third"], third_cases)
    write_json(out["paired_audit"], paired_audit)

    print("VISUAL INTERFERENCE MECHANISM PREFLIGHT")
    print(f"cases = {len(main_cases)}")
    print(f"rescue-reversal cases (main) = {paired_audit['main_rescue_reversal_cases']}")
    print(f"excluded third-color cases = {paired_audit['excluded_third_color_cases']}")
    print(f"all rescue-reversal cases including excluded third-color = {paired_audit['all_rescue_reversal_cases']}")
    print(f"planned no-op rows = {len(main_cases) * EXPECTED_LAYERS * 2}")
    print(f"planned cross-patch rows = {len(main_cases) * EXPECTED_LAYERS * 2}")
    print("paired audit = PASS")
    if not args.execute and not args.analyze_only:
        print("DRY_RUN_PASS")
        return 0

    if args.analyze_only:
        color_map = checkpoint_rows(out["color_scores"], protocol_fingerprint, ("case_id", "layer"))
        identity_map = checkpoint_rows(out["identity_scores"], protocol_fingerprint, ("case_id", "layer"))
        forward_map = checkpoint_rows(out["forward"], protocol_fingerprint, ("case_id", "layer"))
        reverse_map = checkpoint_rows(out["reverse"], protocol_fingerprint, ("case_id", "layer"))
        if not out["noop_audit"].is_file():
            raise MechanismStop("ANALYSIS_INPUT_INCOMPLETE", {"missing": str(out["noop_audit"])})
        noop_audit = read_json(out["noop_audit"])
        summary, status = analyze_outputs(out, main_cases, list(color_map.values()), list(identity_map.values()), list(forward_map.values()), list(reverse_map.values()), noop_audit)
        provenance.update({"mode": "analyze-only", "summary": summary, "final_status": status})
        write_json(out["provenance"], provenance)
        print_terminal(summary, paired_audit, status)
        return 0

    processor, model, device = qwen_baseline.frozen_baseline.load_frozen_model()
    layers, layer_types, layer_paths, norm_path, final_norm, head_path, lm_head = discover_decoder(model)
    runtime = qwen_baseline.runtime_metadata(
        processor,
        model,
        device,
        {
            "families": FAMILIES_PATH,
            "qa": QA_PATH,
            "candidates": REALBIND_ROOT / "candidates" / "all_candidates.jsonl",
            "dependency": DEPENDENCY_PATH,
            "dataset_audit": REALBIND_ROOT / "audit" / "dryrun_100_pair_direct_v1_1_audit.json",
            "dataset_report": REALBIND_ROOT / "reports" / "pair_direct_v1_1_report.md",
        },
        protocol_fingerprint,
    )
    write_json(out["runtime"], runtime)
    semantic_vocabulary = qwen_baseline.build_semantic_vocabulary(families)
    color_map = checkpoint_rows(out["color_scores"], protocol_fingerprint, ("case_id", "layer"))
    identity_map = checkpoint_rows(out["identity_scores"], protocol_fingerprint, ("case_id", "layer"))
    state_payloads: dict[str, Mapping[str, Any]] = {}
    case_metadata: list[dict[str, Any]] = []
    for index, case in enumerate(main_cases, start=1):
        payload, color_rows, identity_rows, metadata = build_case_states_and_scores(
            processor,
            model,
            layers,
            layer_types,
            layer_paths,
            final_norm,
            lm_head,
            device,
            case,
            protocol_fingerprint,
            out["state_cache"],
        )
        state_payloads[str(case["case_id"])] = payload
        for row in color_rows:
            color_map[(str(row["case_id"]), str(row["layer"]))] = row
        for row in identity_rows:
            identity_map[(str(row["case_id"]), str(row["layer"]))] = row
        save_checkpoint(out["color_scores"], color_map)
        save_checkpoint(out["identity_scores"], identity_map)
        case_metadata.append(metadata)
        print(f"STATE/LENS {index}/{len(main_cases)}: {case['case_id']}", flush=True)
    hidden_meta = {
        "model_identifier": MODEL_IDENTIFIER,
        "decoder_layer_count": len(layers),
        "layer_indexing": "0-based true decoder block index; h_l is post-block residual before final norm",
        "layer_paths": layer_paths,
        "layer_types": layer_types,
        "final_norm_path": norm_path,
        "final_norm_class": final_norm.__class__.__name__,
        "lm_head_path": head_path,
        "lm_head_class": lm_head.__class__.__name__,
        "prediction_position": "last processed prompt token immediately before first generated answer token",
        "cases": case_metadata,
    }
    write_json(out["hidden_meta"], hidden_meta)

    _, clean_by_case, noop_audit = run_noop_phase(
        processor,
        model,
        layers,
        device,
        main_cases,
        state_payloads,
        semantic_vocabulary,
        source_facts,
        source_vision,
        protocol_fingerprint,
        out["noop_rows"],
    )
    baseline_rows = []
    for cid in sorted(clean_by_case):
        for condition in (FACTS, VISION):
            baseline_rows.append({"protocol_fingerprint": protocol_fingerprint, **clean_by_case[cid][condition]})
    write_jsonl(out["baseline_audit_rows"], baseline_rows)
    write_json(out["noop_audit"], noop_audit)
    if not noop_audit["passed"]:
        raise MechanismStop("PATCH_IMPLEMENTATION_FAIL", noop_audit)

    forward_map, reverse_map = run_cross_patch_phase(
        processor,
        model,
        layers,
        device,
        main_cases,
        state_payloads,
        semantic_vocabulary,
        protocol_fingerprint,
        out["forward"],
        out["reverse"],
    )
    summary, status = analyze_outputs(
        out,
        main_cases,
        list(color_map.values()),
        list(identity_map.values()),
        list(forward_map.values()),
        list(reverse_map.values()),
        noop_audit,
    )
    source_after = {
        key: sha256_file(Path(record["path"])) for key, record in paired_audit["source_files"].items()
    }
    frozen_after = {"families": sha256_file(FAMILIES_PATH), "qa": sha256_file(QA_PATH), "dependency": sha256_file(DEPENDENCY_PATH)}
    source_before = {key: record["sha256"] for key, record in paired_audit["source_files"].items()}
    if source_after != source_before:
        raise MechanismStop("ORIGINAL_PILOT_CHANGED", {"before": source_before, "after": source_after})
    if frozen_after != provenance["frozen_hashes_before"]:
        raise MechanismStop("FROZEN_INPUT_CHANGED", {"before": provenance["frozen_hashes_before"], "after": frozen_after})
    scorer_after = sha256_file(SCORER_PATH)
    if scorer_after != provenance["scorer"]["sha256_before"]:
        raise MechanismStop("SCORER_CHANGED")
    provenance.update(
        {
            "mode": "execute",
            "runtime": runtime,
            "hidden_state_metadata": hidden_meta,
            "source_hashes_after": source_after,
            "original_pilot_unchanged": True,
            "frozen_hashes_after": frozen_after,
            "frozen_inputs_unchanged": True,
            "scorer": {**provenance["scorer"], "sha256_after": scorer_after, "unchanged": True},
            "no_op_patch": noop_audit,
            "summary": summary,
            "final_status": status,
        }
    )
    write_json(out["provenance"], provenance)
    del model
    del processor
    print_terminal(summary, paired_audit, status)
    return 0


def print_terminal(summary: Mapping[str, Any], paired_audit: Mapping[str, Any], status: str) -> None:
    rescue = summary["best_rescue_window"]
    corruption = summary["best_reverse_corruption_window"]

    def span(value: Mapping[str, Any]) -> str:
        if value.get("start") is None:
            return "NA"
        return str(value["start"]) if value["start"] == value["end"] else f"{value['start']}-{value['end']}"

    counts = summary["classification_counts"]
    print()
    print("VISUAL INTERFERENCE MECHANISM TEST")
    print()
    print(f"cases = {summary['cases']}")
    print(f"rescue-reversal cases = {summary['rescue_reversal_cases_main']} (main; {paired_audit['all_rescue_reversal_cases']} including 2 excluded third-color cases)")
    print()
    print(f"best color-divergence layer = {summary['best_color_divergence_layer']}")
    print(f"best identity-divergence layer = {summary['best_identity_divergence_layer']}")
    print()
    print(f"best rescue layer/window = {span(rescue)}")
    print(f"max rescue rate = {rescue['rate']:.6f}")
    print()
    print(f"best reverse-corruption layer/window = {span(corruption)}")
    print(f"max corruption rate = {corruption['rate']:.6f}")
    print()
    print(f"selection-like cases = {counts.get('selection-like', 0)}")
    print(f"binding/readout-like cases = {counts.get('binding/readout-like', 0)}")
    print(f"unclear cases = {counts.get('unclear', 0)}")
    print()
    print("NO-OP PATCH:")
    print(f"    {'PASS' if summary['no_op_patch'].get('passed') else 'FAIL'}")
    print()
    print("FINAL STATUS:")
    print(f"    {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return execute(args)
    except MechanismStop as error:
        try:
            root = args.output_root.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "mechanism_stop.json", {"code": error.code, "details": error.details})
            (root / error.code).write_text("\n", encoding="utf-8")
        except Exception:
            pass
        print(error.code, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("MECHANISM_TEST_INTERRUPTED", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
