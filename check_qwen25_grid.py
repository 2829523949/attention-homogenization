import os, sys, math, torch
from PIL import Image
sys.path.insert(0, '/root/Qwen3-VL')

from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen2.5-VL-3B-Instruct",
    local_files_only=True, trust_remote_code=True)

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
img = Image.open(os.path.join(BASE, "AFLW2000-3D_crop", "image00002.jpg")).convert('RGB')
print(f"Image size: {img.size}")

msg = [{"role":"user","content":[{"type":"image","image":img},
        {"type":"text","text":"describe this image"}]}]
prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inp = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
ids = inp['input_ids'][0]
ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
nvt = ve - 4
grid_sz = int(math.isqrt(nvt))
print(f"Visual tokens: {nvt}, grid: {grid_sz}x{grid_sz}")

# Qwen2.5-VL config is nested (text_config)
cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
print(f"Total layers: {cfg.num_hidden_layers}")
print(f"Hidden dim: {cfg.hidden_size}")

del model, processor
torch.cuda.empty_cache()
print("Done!")
