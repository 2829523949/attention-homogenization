"""Image-free final reasoner boundary for the corrected main method."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.evaluators.leakage_checks import assert_inference_safe, assert_no_image_payload
from harness.proof_state.compiler import CertifiedFact, MinimalProofState
from harness.proof_state.serializer import serialize_proof_state


@dataclass(frozen=True)
class HardHandoffRequest:
    question: str
    proof_state: MinimalProofState

    def to_dict(self) -> dict[str, Any]:
        if self.proof_state.diagnostic or not self.proof_state.proof_complete or not self.proof_state.certified_proof_complete:
            raise AssertionError("controller must not hand an incomplete or diagnostic proof state to the final reasoner")
        payload = {"question": self.question, "proof_state": self.proof_state.to_dict()}
        assert_inference_safe(payload, context="hard-handoff request")
        assert_no_image_payload(payload, context="hard-handoff request")
        return payload


class HardHandoffReasoner:
    """Abstract API: no image-bearing argument exists by construction."""

    def generate(self, question: str, proof_state: MinimalProofState) -> dict[str, Any]:
        raise NotImplementedError


class DeterministicProofStateReasoner(HardHandoffReasoner):
    """Architecture smoke-test reasoner; it consumes facts, never pixels."""

    def generate(self, question: str, proof_state: MinimalProofState) -> dict[str, Any]:
        request = HardHandoffRequest(question, proof_state)
        request.to_dict()
        color_facts = [fact for fact in proof_state.facts if isinstance(fact, CertifiedFact) and fact.predicate == "COLOR" and fact.certificate_status == "CERTIFIED"]
        answer = None
        if len(color_facts) == 1:
            answer = color_facts[0].value
        elif len(color_facts) > 1:
            answer = "UNCERTAIN"
        serialized = serialize_proof_state(proof_state)
        return {
            "reasoner": "deterministic_proof_state_smoke",
            "answer": answer,
            "raw_response": str(answer) if answer is not None else "UNCERTAIN",
            "question": question,
            "proof_state_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "image_received": False,
            "input_schema": "question + MinimalProofState",
        }


class Qwen35HardHandoffReasoner(HardHandoffReasoner):
    """Optional Qwen3.5-4B text-only adapter for the final symbolic step."""

    generation_config = {"do_sample": False, "use_cache": False, "max_new_tokens": 16, "enable_thinking": False}

    def __init__(self, model_path: Path = Path("/root/Qwen3-VL/models/Qwen3.5-4B"), device: str | None = None):
        # Loading is deliberately lazy and text-only. The model path is an
        # existing historical Qwen asset, not a new download.
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32).to(self.device).eval()

    def generate(self, question: str, proof_state: MinimalProofState) -> dict[str, Any]:
        request = HardHandoffRequest(question, proof_state)
        request.to_dict()
        import torch

        text = "Certified visual proof state:\n" + serialize_proof_state(proof_state) + "\n\nFrozen question:\n" + question
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=rendered, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, **self.generation_config)
        continuation = output[0, int(inputs["input_ids"].shape[-1]):]
        response = self.processor.decode(continuation, skip_special_tokens=True).strip()
        return {
            "reasoner": "Qwen3.5-4B-hard-handoff",
            "answer": response,
            "raw_response": response,
            "question": question,
            "proof_state_sha256": hashlib.sha256(serialize_proof_state(proof_state).encode("utf-8")).hexdigest(),
            "image_received": False,
            "input_schema": "question + MinimalProofState",
            "generation_config": dict(self.generation_config),
        }


def hard_handoff_signature_is_image_free() -> bool:
    parameters = inspect.signature(HardHandoffReasoner.generate).parameters
    return not any(token in name.casefold() for name in parameters for token in ("image", "pixel", "embedding"))

