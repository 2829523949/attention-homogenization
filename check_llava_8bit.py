"""Check if LLaVA-NeXT-7B runs at 8-bit on RTX 5060 8GB."""
import os, sys, torch
sys.path.insert(0, '/root/Qwen3-VL')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# 4-bit (current, baseline)
print("\n--- 4-bit NF4 (current) ---")
bnb_4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
m4 = LlavaNextForConditionalGeneration.from_pretrained(
    "/root/Qwen3-VL", quantization_config=bnb_4, device_map="auto",
    attn_implementation="eager", local_files_only=True, trust_remote_code=True)
m4.config.image_grid_pinpoints = []
del m4; torch.cuda.empty_cache()
print(f"Free after 4-bit: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# 8-bit
print("\n--- 8-bit (bitsandbytes) ---")
bnb_8 = BitsAndBytesConfig(load_in_8bit=True)
try:
    m8 = LlavaNextForConditionalGeneration.from_pretrained(
        "/root/Qwen3-VL", quantization_config=bnb_8, device_map="auto",
        attn_implementation="eager", local_files_only=True, trust_remote_code=True)
    m8.config.image_grid_pinpoints = []
    print(f"8-bit loaded! Free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    
    # Forward on one AFLW2000 image
    from PIL import Image
    proc = AutoProcessor.from_pretrained("/root/Qwen3-VL", local_files_only=True, trust_remote_code=True)
    BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
    img = Image.open(os.path.join(BASE, "AFLW2000-3D_crop", "image00002.jpg")).convert('RGB')
    conv = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this image"}]}]
    prompt = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inp = proc(text=prompt, images=img, return_tensors='pt').to('cuda')
    with torch.no_grad(): m8(**inp)
    peak = torch.cuda.max_memory_allocated()/1e9
    print(f"Forward OK! Peak: {peak:.2f} GB")
    del m8; torch.cuda.empty_cache()
    print("SUCCESS: 8-bit fits on RTX 5060!")
except Exception as e:
    print(f"FAILED: {e}")
    print(f"Free at failure: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
