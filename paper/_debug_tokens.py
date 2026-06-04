"""Debug: check token ordering in Qwen3-VL input."""
import sys, os, torch
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)

img = Image.open("/root/output_phase1_test/右侧15度平视.png").convert("RGB")

query = "nose"
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": f"describe the {query}"}
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=img, return_tensors="pt")

input_ids = inputs["input_ids"][0]
print(f"Total tokens: {len(input_ids)}")
print(f"\nFirst 30 tokens:")
for i in range(min(30, len(input_ids))):
    decoded = processor.tokenizer.decode([input_ids[i]])
    print(f"  [{i:3d}] id={input_ids[i]:6d} -> {repr(decoded)}")

if "image_grid_thw" in inputs:
    g = inputs["image_grid_thw"][0]
    h_patches = g[1].item() // 2
    w_patches = g[2].item() // 2
    nvt = h_patches * w_patches
    print(f"\nVisual tokens: {nvt} ({h_patches}x{w_patches})")

print(f"\nLast 20 tokens:")
for i in range(max(0, len(input_ids)-20), len(input_ids)):
    decoded = processor.tokenizer.decode([input_ids[i]])
    print(f"  [{i:3d}] id={input_ids[i]:6d} -> {repr(decoded)}")

# Find positions of visual tokens
print(f"\nSearching for visual token markers...")
for i in range(len(input_ids)):
    decoded = processor.tokenizer.decode([input_ids[i]])
    if "vision" in decoded.lower() or "image" in decoded.lower() or "visual" in decoded.lower():
        print(f"  [{i:3d}] id={input_ids[i]:6d} -> {repr(decoded)}")
