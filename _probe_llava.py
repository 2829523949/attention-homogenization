"""Probe LLaVA model structure"""
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

model = LlavaNextForConditionalGeneration.from_pretrained(
    "/root/Qwen3-VL", quantization_config=bnb,
    device_map="auto", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True,
    torch_dtype=torch.bfloat16)

print("=== Model type ===")
print(type(model).__name__)

print("\n=== All named_children ===")
for name, mod in model.named_children():
    sub_type = type(mod).__name__
    has_layers = hasattr(mod, "layers")
    has_model_layers = hasattr(mod, "model") and hasattr(mod.model, "layers")
    ml_len = len(mod.model.layers) if has_model_layers else 0
    l_len = len(mod.layers) if has_layers else 0
    print(f"  {name}: {sub_type}  layers={l_len}  model.layers={ml_len}")

print("\n=== Deep search for .layers ===")
def search(obj, path="obj", depth=0):
    if depth > 3:
        return
    if hasattr(obj, "layers") and isinstance(getattr(obj, "layers"), torch.nn.ModuleList):
        print(f"  >> {path}.layers  (len={len(obj.layers)})")
    if depth < 2:
        for name, child in obj.named_children():
            search(child, f"{path}.{name}", depth + 1)

search(model)

print("\n=== Done ===")
