import os, sys, torch
sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
print("=== module tree (layers/norm/embed) ===")
for name, mod in model.named_modules():
    if any(x in name.lower() for x in ['layer', 'norm', 'embed', 'model', 'language']):
        if name.count('.') <= 3:
            print(f"  {name or '(root)'}: {type(mod).__name__}")
del model
torch.cuda.empty_cache()
