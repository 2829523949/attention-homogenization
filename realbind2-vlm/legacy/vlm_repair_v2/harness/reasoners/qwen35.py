from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT = Path("/root/Qwen3-VL")
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))


class Qwen35Reasoner:
    """Thin wrapper around the project's audited frozen loader and decoding."""

    generation_config = {"do_sample": False, "use_cache": False, "max_new_tokens": 16, "enable_thinking": False, "batch_size": 1}

    def __init__(self):
        from experiments.realbind_phase2.atomic_composition_baseline import load_frozen_model
        self.processor, self.model, self.device = load_frozen_model()

    def generate(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        import torch
        from experiments.realbind_phase2.atomic_composition_baseline import decode_generation
        from experiments import qwen35_kv_gdn_factorial_pilot as runtime

        vision_module = getattr(self.model, "visual", None)
        if vision_module is None:
            vision_module = getattr(getattr(self.model, "model", None), "visual", None)
        vision_calls = [0]
        hook = vision_module.register_forward_hook(lambda *_: vision_calls.__setitem__(0, vision_calls[0] + 1)) if vision_module is not None else None
        if image_path is not None:
            rendered = runtime.render_chat(self.processor, str(image_path), prompt)
            inputs = runtime.encode_single(self.processor, rendered, str(image_path), self.device)
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.processor(text=rendered, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            if any(token in rendered for token in ("<|image", "<|vision", "<|video")) or any("pixel" in key or "image" in key or "video" in key for key in inputs):
                raise AssertionError("text-only Qwen call contains visual tokens or inputs")
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=False)
        finally:
            if hook is not None:
                hook.remove()
        continuation = output[0, int(inputs["input_ids"].shape[-1]):]
        raw, decoded, token_ids = decode_generation(self.processor, continuation)
        return {"raw_response_with_special_tokens": raw, "response": decoded, "token_ids": token_ids, "generation_config": dict(self.generation_config), "prompt": prompt, "image_path": str(image_path) if image_path else None, "rendered_prompt_sha256": __import__("hashlib").sha256(rendered.encode()).hexdigest(), "input_keys": sorted(inputs), "visual_tokens_in_rendered_prompt": any(token in rendered for token in ("<|image", "<|vision", "<|video")), "vision_module_found": vision_module is not None, "vision_encoder_called": vision_calls[0] > 0, "vision_encoder_call_count": vision_calls[0]}
