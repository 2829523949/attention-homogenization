import os, sys, math, torch
from PIL import Image
sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    '/root/Qwen3-VL-2B-Instruct', torch_dtype=torch.bfloat16,
    device_map='cuda', attn_implementation='eager',
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    '/root/Qwen3-VL-2B-Instruct', local_files_only=True, trust_remote_code=True)

cfg = model.config.text_config
print(f'Config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}')

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
with open(os.path.join(BASE, "AFLW2000-3D_crop.list")) as f:
    all_images = [l.strip() for l in f if l.strip()]

img = Image.open(os.path.join(BASE, "AFLW2000-3D_crop", all_images[0])).convert('RGB')
print(f'Image: {all_images[0]}, size: {img.size}')

msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this image"}]}]
prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inp = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
ids = inp['input_ids'][0]
ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
nvt = ve - 4
grid_sz = int(math.isqrt(nvt))
print(f'Visual tokens: {nvt}, grid: {grid_sz}x{grid_sz}')

# Test hidden state hook on last layer
lm = model.model.language_model
last_l = cfg.num_hidden_layers - 1  # 27
collected = {}
hook = lm.layers[last_l].register_forward_hook(
    lambda m, i, o: collected.update({'out': o[0].detach() if isinstance(o, tuple) else o.detach()})
)
with torch.no_grad():
    model(**inp)
hook.remove()
h = collected['out']
print(f'Hidden state at L{last_l+1}: shape={h.shape}, dtype={h.dtype}')
vis_h = h[0, 4:4+nvt]
print(f'  Visual tokens only: {vis_h.shape}, mean={vis_h.mean().item():.6f}, std={vis_h.std().item():.6f}')

del model, processor
torch.cuda.empty_cache()
print('Done!')
