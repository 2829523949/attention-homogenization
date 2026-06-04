"""
Gradient Attribution - Single Image (§4.1)
============================================
Processes one COCO image and appends result to a shared JSON.
Each invocation is a fresh process → complete GPU memory cleanup.
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

QUERY_PAIRS = [("left side", "right side"), ("top", "bottom")]
LAYER_GROUPS = {"L1-3": [0, 1, 2], "L18-20": [17, 18, 19], "L28": [27]}
GRAD_LAYERS = {"L1-3": [0, 1, 2], "L18-20": [17, 18, 19], "L28": [26]}

COCO_DIR = "/datasets/coco/val2017"
if not os.path.exists(COCO_DIR):
    COCO_DIR = "/coco/val2017"
if not os.path.exists(COCO_DIR):
    COCO_DIR = os.path.expanduser("~/coco/val2017")


def process_one_image(img_name: str):
    """Process a single image and return results dict."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    results = {}
    img_path = os.path.join(COCO_DIR, img_name)
    if not os.path.exists(img_path):
        return {"image": img_name, "error": "not found"}
    img = Image.open(img_path).convert("RGB")

    for gk in LAYER_GROUPS:
        grad_layers = GRAD_LAYERS[gk]
        for q1, q2 in QUERY_PAIRS:
            for q in [q1, q2]:
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"describe the {q}"}
                ]}]
                prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
                inp["pixel_values"].requires_grad_(True)
                ids = inp["input_ids"][0]
                ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
                vs, nvt = 4, ve - 4
                q_pos = ve + 1

                hs_out = []
                def hs_hook(m, i, o):
                    hs_out.append(o[0])
                hs_handle = lm.layers[grad_layers[0]].register_forward_hook(hs_hook)
                out = model(**inp)
                hs_handle.remove()

                key = f"{gk}_{q1}-{q2}"
                if hs_out:
                    h_q = hs_out[0][0, q_pos]
                    grad = torch.autograd.grad(h_q.norm(), inp["pixel_values"], retain_graph=False)[0]
                    if grad is not None:
                        sal = grad.norm(dim=-1).detach().cpu().float().numpy()
                        sal = sal[:nvt] if len(sal) >= nvt else np.pad(sal, (0, nvt - len(sal)))
                    else:
                        sal = np.zeros(nvt)
                    results[key] = sal.tolist()
                else:
                    results[key] = [0.0] * nvt

                del hs_out, out, inp, grad, h_q
                torch.cuda.empty_cache()
                gc.collect()
            # end q
        # end pair
    # end gk
    
    del model, processor, lm
    torch.cuda.empty_cache()
    gc.collect()
    return {"image": img_name, "data": results}


if __name__ == "__main__":
    img_name = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT_DIR, "gradient_single.json")
    result = process_one_image(img_name)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"DONE: {img_name}")
