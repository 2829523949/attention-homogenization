"""Check LLaVA visual token layout for face vs dot images."""
import os,sys,torch,numpy as np
from PIL import Image
sys.path.insert(0,"/root/Qwen3-VL");os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["HF_HUB_OFFLINE"]="1"

from transformers import LlavaNextForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
model=LlavaNextForConditionalGeneration.from_pretrained("/root/Qwen3-VL",
    quantization_config=bnb,device_map="auto",attn_implementation="eager",
    local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16)
model.config.image_grid_pinpoints=[[336,336]]
proc=AutoProcessor.from_pretrained("/root/Qwen3-VL",local_files_only=True,trust_remote_code=True)
proc.chat_template=("{{ bos_token }}{% for message in messages %}"
    "{% if (message['role']=='user')!=(loop.index0%2==0)%}{{ raise_exception('')}}{% endif %}"
    "{% if message['role']=='user'%}{{'[INST] '+message['content']|trim+' [/INST]'}}"
    "{% elif message['role']=='assistant'%}{{' '+message['content']|trim+' '+eos_token}}{% endif %}{% endfor %}")

# Test both image types
test_imgs={
    "face":Image.open("/root/output_phase1_test/右侧15度平视.png").convert("RGB"),
    "dot":Image.new("RGB",(256,256),(0,0,0)),
}
# Draw a dot on the synthetic
from PIL import ImageDraw
d=ImageDraw.Draw(test_imgs["dot"])
d.rectangle([16,16,31,31],fill=(255,0,0))

for name,img in test_imgs.items():
    rimg=img.resize((336,336),Image.LANCZOS)
    msg=[{"role":"user","content":"<image>\ndescribe this"}]
    pr=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    inp=proc(text=pr,images=rimg,return_tensors="pt")
    ids=inp["input_ids"][0]
    ip=(ids==32000).nonzero(as_tuple=True)[0]
    nvt=int(ip[-1])-int(ip[0])+1
    print(f"\n{name}: nvt={nvt}, IMAGE tokens={len(ip)}")
    print(f"  IMAGE IDs at: {ip[:5].tolist()}...{ip[-5:].tolist()}")
    # Check if tokens are contiguous
    gaps=[int(ip[i+1]-ip[i]) for i in range(min(30,len(ip)-1))]
    print(f"  First 30 gaps: {gaps}")
    max_gap=max(gaps)
    print(f"  Max gap: {max_gap}")
    if max_gap>1:
        print(f"  WARNING: non-contiguous! Max gap={max_gap}")
    else:
        print(f"  ✓ Contiguous IMAGE tokens")
