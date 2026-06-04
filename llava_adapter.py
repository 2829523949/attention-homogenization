"""LLaVA-NeXT Adapter for Attention Analyzer (4-bit)"""
import os, sys, math, torch, numpy as np
from PIL import Image
from attention_viz import BaseVLMAdapter, VLMConfig, AttentionAnalyzer

MISTRAL_TMPL = ("{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('') }}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content']|trim + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content']|trim + ' ' + eos_token }}"
    "{% endif %}{% endfor %}")

class LlavaNextAdapter(BaseVLMAdapter):
    IMAGE_TOKEN_ID = 32000

    def load_model(self, config):
        from transformers import (
            LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            config.model_path, quantization_config=bnb,
            device_map="auto", attn_implementation="eager",
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.bfloat16)
        proc = AutoProcessor.from_pretrained(
            config.model_path, local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code)
        proc.chat_template = MISTRAL_TMPL
        model.eval()
        return model, proc

    def get_lm(self, model):
        return model.model.language_model

    def get_num_layers(self, lm):
        return len(lm.layers)

    def build_messages(self, image, query):
        return [{"role": "user", "content": f"<image>\ndescribe the {query}"}]

    def get_token_ranges(self, input_ids):
        ip = (input_ids == self.IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(ip) == 0:
            return {"vis_start": 0, "vis_end": 0,
                    "q_start": 1, "q_end": len(input_ids)}
        return {"vis_start": int(ip[0]), "vis_end": int(ip[-1]) + 1,
                "q_start": int(ip[-1]) + 1, "q_end": len(input_ids),
                "nvt": int(ip[-1]) + 1 - int(ip[0])}

    def _debug_token_structure(self, input_ids):
        ip = (input_ids == self.IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        print(f"  input_ids len={len(input_ids)}, IMAGE count={len(ip)}")
        print(f"  Range: [{ip[0]}:{ip[-1]+1})")


def dry_run():
    cfg = VLMConfig(model_path="/root/Qwen3-VL",
                    output_dir="/root/Qwen3-VL/paper/figures/llava",
                    image_dir="/root/output_phase1_test", max_images=1)
    os.makedirs(cfg.output_dir, exist_ok=True)
    adapter = LlavaNextAdapter()

    print("Loading LLaVA-NeXT Mistral-7B (4-bit)...")
    model, proc = adapter.load_model(cfg)
    lm = adapter.get_lm(model)
    print(f"Loaded. {len(lm.layers)} LM layers.")

    imgs = sorted([f for f in os.listdir(cfg.image_dir)
                   if f.endswith((".png",".jpg",".jpeg"))
                   and f != "grid_9views.png"])
    if not imgs: return print("No images")
    fname = imgs[0]
    print(f"\nImage: {fname}")
    img = Image.open(os.path.join(cfg.image_dir, fname)).convert("RGB")

    msg = adapter.build_messages(img, "nose")
    prompt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    print(f"Prompt: {repr(prompt[:120])}...")
    inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda")
    tr = adapter.get_token_ranges(inputs["input_ids"][0])
    print(f"Token ranges: vis=[{tr['vis_start']}:{tr['vis_end']}) "
          f"q=[{tr['q_start']}:{tr['q_end']}) nvt={tr['nvt']}")

    for lidx in [0, 1, 2]:
        lm.layers[lidx].self_attn.output_attentions = True

    attn = []
    handles = [lm.layers[l].self_attn.register_forward_hook(
        lambda m, i, o: attn.append(o[1].detach())
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None else None)
        for l in [0, 1, 2]]

    with torch.no_grad():
        out = model(**inputs)
    for h in handles: h.remove()

    if attn:
        a = torch.stack(attn).mean(dim=0).mean(dim=[0, 1])
        vis_attn = a[tr["q_start"]:tr["q_end"],
                     tr["vis_start"]:tr["vis_end"]].mean(dim=0).cpu().float().numpy()
        print(f"vis_attn shape: {vis_attn.shape}")
        # Try reshape to nearest square (2928 -> first 576=24x24 or similar)
        for s in [24, 32, 16]:
            if s*s <= len(vis_attn):
                hm = vis_attn[:s*s].reshape(s, s)
                print(f"Heatmap ({s}x{s}): [{hm.min():.4f},{hm.max():.4f}]")
                break
        print(f"SUCCESS: {len(vis_attn)} visual tokens extracted")
    else:
        print("No attention captured")

    del attn, a, out, inputs, model
    torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    dry_run()
