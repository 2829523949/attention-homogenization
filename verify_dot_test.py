"""Verify synthetic dot test: token counts, grid sizes, block position mapping."""
import os, sys, torch
from PIL import Image
sys.path.insert(0, "/root/Qwen3-VL")

img = Image.open("/root/Qwen3-VL/paper/figures/dot_test/synthetic_dots.png")
print(f"Input image: {img.size}")

# Qwen3-VL
print(f"\n{'='*60}\nQwen3-VL-2B: Token structure")
from transformers import AutoProcessor
qproc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",local_files_only=True,trust_remote_code=True)
msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe the red area"}]}]
prompt = qproc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
inputs = qproc(text=prompt,images=img,return_tensors="pt")
ids = inputs["input_ids"][0]
print(f"  Input IDs: {len(ids)}")
ve = (ids==151653).nonzero(as_tuple=True)[0]
ie = (ids==151645).nonzero(as_tuple=True)[0]
print(f"  vision_end at: {ve.tolist()}, im_end at: {ie.tolist()}")
nvt = ve.item() - 4
print(f"  Visual tokens: {nvt}")
dim = int(nvt**0.5)
print(f"  Grid: {dim}x{dim}" if dim*dim==nvt else f"  NOT square ({nvt})")
print(f"  image_grid_thw: {inputs.get('image_grid_thw', 'N/A')}")
if dim*dim==nvt:
    for name,gx,gy in [("red",2,2),("green",11,11),("blue",21,21)]:
        qx,qy=gx*dim//24,gy*dim//24
        print(f"  {name} in {dim}x{dim} grid: ({qx},{qy})")

# LLaVA-NeXT
print(f"\n{'='*60}\nLLaVA-NeXT-7B: Token structure")
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
lmod = LlavaNextForConditionalGeneration.from_pretrained(
    "/root/Qwen3-VL",quantization_config=bnb,device_map="auto",
    attn_implementation="eager",local_files_only=True,trust_remote_code=True,
    torch_dtype=torch.bfloat16)
lmod.config.image_grid_pinpoints = [[336, 336]]
lproc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL",local_files_only=True,trust_remote_code=True)
lproc.chat_template = ("{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0%2==0) %}{{ raise_exception('') }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ '[INST] '+message['content']|trim+' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}{{ ' '+message['content']|trim+' '+eos_token }}{% endif %}{% endfor %}")

msg2 = [{"role":"user","content":"<image>\ndescribe the red area"}]
prompt2 = lproc.apply_chat_template(msg2,tokenize=False,add_generation_prompt=True)
inputs2 = lproc(text=prompt2,images=img,return_tensors="pt")
ids2 = inputs2["input_ids"][0]
print(f"  Input IDs: {len(ids2)}")
ip = (ids2==32000).nonzero(as_tuple=True)[0]
print(f"  IMAGE(32000) count: {len(ip)}")
vs,ve2=int(ip[0]),int(ip[-1])+1
nvt2=ve2-vs
dim2=int(nvt2**0.5)
print(f"  Visual range: [{vs}:{ve2}), tokens={nvt2}")
print(f"  Grid: {dim2}x{dim2}" if dim2*dim2==nvt2 else f"  NOT square ({nvt2})")
if dim2*dim2==nvt2:
    for name,gx,gy in [("red",2,2),("green",11,11),("blue",21,21)]:
        lx,ly=gx*dim2//24,gy*dim2//24
        print(f"  {name} in {dim2}x{dim2} grid: ({lx},{ly})")
