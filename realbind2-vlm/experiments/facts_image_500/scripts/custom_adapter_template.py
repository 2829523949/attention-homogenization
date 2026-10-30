#!/usr/bin/env python3
"""Template for adding a new VLM adapter.

Copy this file to a new module in ``scripts/`` (or any importable module), set
the model loader/preprocessing, and point a model config at it:

    adapter: custom
    adapter_class: my_adapter.MyVLMAdapter
    path_key: my_vlm
    model_id: My-VLM
    runtime: {device: cuda, dtype: bfloat16, max_new_tokens: 16}

The only contract required by ``run_facts_image.py`` is:

* ``prepare_prompt(text, image_path=None) -> prepared``
    - ``image_path is None`` must produce the facts-only request.
    - otherwise it must add exactly the original image as a vision content block.
* ``generate(prepared) -> str``.
* ``get_model_metadata() -> dict`` (inherited implementation is sufficient).

Do not add bboxes, crops, marks, object labels, a different prompt, a different
question, or few-shot examples.  The facts-only and image-plus-facts branches
must receive byte-identical ``text``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base_adapter import VLMAdapter


class MyVLMAdapter(VLMAdapter):
    model_family = "my-vlm"

    def _load(self) -> None:
        if hasattr(self, "model"):
            return
        # Import heavy dependencies lazily so the runner can be inspected without
        # installing a model runtime.
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = getattr(torch, str(self.runtime.get("dtype", "bfloat16")))
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(self.runtime.get("device", "cuda"))

    def prepare_prompt(self, prompt: str, image_path: str | Path | None = None) -> Any:
        self._load()
        # Replace this block with the model's documented chat/image rendering.
        # For image_path=None the rendered request must contain only the text.
        messages = [{"role": "user", "content": []}]
        images = None
        if image_path is not None:
            images = [self.prepare_image(image_path)]
            messages[0]["content"].append({"type": "image"})
        messages[0]["content"].append({"type": "text", "text": prompt})
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.processor(text=[rendered], images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items()}

    def generate(self, prepared: dict[str, Any]) -> str:
        import torch

        with torch.inference_mode():
            generated = self.model.generate(
                **prepared,
                do_sample=False,
                use_cache=bool(self.runtime.get("use_cache", False)),
                max_new_tokens=int(self.runtime.get("max_new_tokens", 16)),
            )
        prompt_length = int(prepared["input_ids"].shape[1])
        return self.processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
