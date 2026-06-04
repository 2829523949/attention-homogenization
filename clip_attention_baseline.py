"""
CLIP Attention Baseline: within-face r on AFLW2000 using CLIP embeddings.
Tests whether the high within-face correlation (r>0.9) is unique to VLM decoder-only
architectures or also present in CLIP.

Method:
  For each image + query:
    1. Get patch embeddings from ViT post_layernorm
    2. Project via visual_projection to projection_dim (CLIP's shared space)
    3. L2 normalize both patches and text
    4. Compute cosine similarity map
    5. within-face r across 6 face queries
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

CLIP_PATH = "/root/clip-vit-large-patch14"
if not os.path.isdir(CLIP_PATH):
    CLIP_PATH = "openai/clip-vit-large-patch14"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 200
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_within_r(flat_maps):
    nq = len(flat_maps)
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r_val, _ = pearsonr(flat_maps[i], flat_maps[j])
            sim[i, j] = r_val
    face_sims = [sim[i, j] for i in range(nq) for j in range(i + 1, nq)]
    return float(np.mean(face_sims))


def main():
    print("=" * 70)
    print("CLIP ATTENTION BASELINE ON AFLW2000")
    print("=" * 70)

    # ---- 1. Load CLIP ----
    print("\n[1/4] Loading CLIP...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(
        CLIP_PATH, torch_dtype=torch.float16,
        device_map=DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    vis_dim = model.config.vision_config.hidden_size
    proj_dim = model.config.projection_dim
    print(f"   CLIP loaded. Vision dim={vis_dim}, Projection dim={proj_dim}")

    # Pre-get projection layers for use in the loop
    visual_proj = model.visual_projection

    # ---- 2. Load images ----
    print("\n[2/4] Loading AFLW2000 image list...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    img_names = [all_names[i] for i in list(range(0, len(all_names), step))[:N_IMAGES]]
    print(f"   {len(all_names)} total, sampling {N_IMAGES}.")

    # ---- 3. Compute ----
    print("\n[3/4] Computing CLIP similarity maps...")
    results = []
    t_start = time.time()

    for img_idx, img_name in enumerate(img_names):
        elapsed = time.time() - t_start
        eta = (elapsed / max(1, img_idx)) * (N_IMAGES - img_idx) if img_idx > 0 else 0
        print(f"   [{img_idx + 1}/{N_IMAGES}] {img_name[:30]:30s}  "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")

        # --- Get patch embeddings via vision_model output ---
        proc_img = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vis_out = model.vision_model(pixel_values=proc_img["pixel_values"],
                                         output_hidden_states=False)
        # vis_out.last_hidden_state: [1, 1+n_patches, vis_dim]
        # last_hidden_state includes CLS at [:, 0, :] + patches at [:, 1:, :]
        patches_raw = vis_out.last_hidden_state[0, 1:, :]  # [n_patches, vis_dim], model dtype
        patches_proj = visual_proj(patches_raw)   # [n_patches, proj_dim]
        patches_proj = patches_proj.float()        # fp32 for downstream
        patches_norm = torch.nn.functional.normalize(patches_proj, dim=-1)

        # --- For each query, compute text-patch similarity ---
        sim_maps = []
        for query in QUERIES:
            proc_text = processor(text=[f"face {query}"],
                                  return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                txt_out = model.text_model(**proc_text)
                # Pool from EOS token (standard CLIP text embedding)
                txt_pooled = txt_out[0][range(len(proc_text['input_ids'])), proc_text['attention_mask'].sum(dim=1)-1]
                text_emb = model.text_projection(txt_pooled)
                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
            # Cosine similarity in shared projection space
            sim = (text_emb.float() @ patches_norm.T)
            # Scale and softmax to get attention-like distribution
            sim = (sim * 10.0).softmax(dim=-1)
            sim_maps.append(sim[0].detach().cpu().float().numpy())

        within_r = compute_within_r(sim_maps)
        results.append({"image": img_name, "within_face_r": within_r})

        if (img_idx + 1) % 50 == 0:
            interim = np.mean([r["within_face_r"] for r in results])
            print(f"      interim mean r = {interim:.4f}")

        del patches_raw, patches_proj, patches_norm, proc_img
        torch.cuda.empty_cache()

    # ---- 4. Summary ----
    print("\n[4/4] Summary...")
    r_vals = [r["within_face_r"] for r in results]
    print(f"\nCLIP within-face r on {len(r_vals)} AFLW2000 images:")
    print(f"  Mean: {np.mean(r_vals):.4f}")
    print(f"  Std:  {np.std(r_vals):.4f}")
    print(f"  Min:  {np.min(r_vals):.4f}")
    print(f"  Max:  {np.max(r_vals):.4f}")

    save_data = {
        "config": {
            "model": str(CLIP_PATH),
            "n_images": len(results),
            "queries": QUERIES,
            "vis_dim": vis_dim,
            "proj_dim": proj_dim,
            "method": "cosine(text_emb_proj, patch_emb_proj)_L2normed",
            "dataset": "AFLW2000-3D_crop",
        },
        "summary": {
            "mean_within_r": round(float(np.mean(r_vals)), 4),
            "std_within_r": round(float(np.std(r_vals)), 4),
            "min_within_r": round(float(np.min(r_vals)), 4),
            "max_within_r": round(float(np.max(r_vals)), 4),
        },
        "per_image": results,
    }
    save_path = os.path.join(OUT_DIR, "clip_attention_baseline.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")

    total = time.time() - t_start
    print(f"Total time: {total:.0f}s ({total / 60:.1f}min)")
    print("Done!")


if __name__ == "__main__":
    main()
