"""Quick check: Can Qwen2.5-VL-3B-Instruct run at bfloat16 on RTX 5060 8GB?"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # fallback for processor
import sys, math, torch
sys.path.insert(0, '/root/Qwen3-VL')

MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
print(f"All model files present: {all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in ['config.json', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'])}")

from transformers import AutoModelForImageTextToText, AutoProcessor
print("Loading Qwen2.5-VL-3B-Instruct at bfloat16 (local)...")

try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    loaded_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded! VRAM used: {loaded_gb:.2f} GB")
    print(f"Remaining: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    
    cfg = model.config
    hs = cfg.text_config.hidden_size if hasattr(cfg, 'text_config') else cfg.hidden_size
    nl = cfg.text_config.num_hidden_layers if hasattr(cfg, 'text_config') else cfg.num_hidden_layers
    print(f"Config: hidden={hs}, layers={nl}")
    
    from PIL import Image
    BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
    img = Image.open(os.path.join(BASE, "AFLW2000-3D_crop", "image00002.jpg")).convert('RGB')
    msg = [{"role":"user","content":[{"type":"image","image":img},
            {"type":"text","text":"describe this image"}]}]
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
    ids = inp['input_ids'][0]
    ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
    nvt = ve - 4
    grid_sz = int(math.isqrt(nvt))
    print(f"Image: 120x120 -> {nvt} vis tokens ({grid_sz}x{grid_sz})")
    
    with torch.no_grad():
        out = model(**inp)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Forward pass OK, VRAM peak: {peak:.2f} GB")
    del model, processor
    torch.cuda.empty_cache()
    print("SUCCESS!")

except Exception as e:
    print(f"FAILED: {e}")
    print(f"VRAM at failure: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
