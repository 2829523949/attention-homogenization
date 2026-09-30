#!/usr/bin/env python3
"""ObjEmbed member scoring for the frozen SAM3 top64 proposal set.

Run in a subprocess with external/objembed_pydeps before site-packages so the
ObjEmbed-compatible Transformers runtime is isolated from the Qwen runtime.
"""
from __future__ import annotations
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path("/mnt/d/vlm_repair_v2")
sys.path.insert(0, str(ROOT / "external/objembed_pydeps"))
sys.path.insert(0, str(ROOT / "external/ObjEmbed"))
OUT = ROOT / "outputs/sam3_instance_hypothesis_binding"
RAW = OUT / "raw_candidate_manifest.json"
CACHE = OUT / "objembed_cache"
SCORES = OUT / "objembed_member_scores.jsonl"
QUERY_TEMPLATE = "Find an object that matches the given caption. {referent} <local_text>"
CHECKPOINT = Path("/root/vlm_objembed_2b") if (Path("/root/vlm_objembed_2b") / "model-00001-of-00002.safetensors").exists() else ROOT / "models/objembed_2b"


def rows(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def valid_boxes(boxes, size):
    w, h = size
    out = []
    for b in boxes:
        c = [max(0.0, min(float(w), b[0])), max(0.0, min(float(h), b[1])), max(0.0, min(float(w), b[2])), max(0.0, min(float(h), b[3]))]
        assert 0 <= c[0] < c[2] <= w and 0 <= c[1] < c[3] <= h, (b, c)
        out.append(c)
    return out


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_objembed import ObjectEmbed
    from models.vision_process import process_vision_info

    CACHE.mkdir(parents=True, exist_ok=True)
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    existing = {f"{r['family_id']}|{r['logical_ref']}": r for r in rows(SCORES)}
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = ObjectEmbed.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto").eval()
    obj_id = processor.tokenizer.convert_tokens_to_ids("<object>")
    text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")
    global_id = processor.tokenizer.convert_tokens_to_ids("<global>") if model.use_global_caption else None
    global_text_id = processor.tokenizer.convert_tokens_to_ids("<global_text>") if model.use_global_caption else None
    model.model.object_token_id = obj_id
    for index, ref in enumerate(raw["referents"], 1):
        key = f"{ref['family_id']}|{ref['logical_ref']}"
        cache_path = CACHE / f"{ref['family_id']}__{ref['logical_ref']}.json"
        if key in existing and cache_path.exists():
            continue
        image = Image.open(ref["image_path"]).convert("RGB")
        boxes = valid_boxes([c["bbox_xyxy"] for c in ref["candidates"]], image.size)
        assert len(boxes) == len(ref["candidates"])
        msg = [{"role": "user", "content": [{"type": "text", "text": QUERY_TEMPLATE.format(referent=ref["referent_phrase"])}]}]
        texts = [processor.apply_chat_template(msg, tokenize=False).strip()]
        tinputs = processor(text=texts, return_tensors="pt", padding=True, do_resize=False).to(model.device)
        with torch.inference_mode():
            q = model(text_processed=tinputs, global_id=global_id, local_text_id=text_id, global_text_id=global_text_id)["local_text_embeddings"]
        obj_str = "".join(f"Object {j}: <object><object>. " for j in range(len(boxes)))
        if model.use_two_tokens == 0:
            obj_str += "The global image is <global>"
        elif model.use_two_tokens == 1:
            obj_str = "The global image is <global>. " + obj_str + "The global image is <global>"
        else:
            obj_str = "The coarse global image is <global>. " + obj_str + " The detailed global image is <global>. "
        msg = [{"role": "user", "content": [
            {"type": "image", "image": copy.deepcopy(image)},
            {"type": "text", "text": "Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. " + obj_str},
        ]}]
        images, videos = process_vision_info(msg, image_patch_size=16)
        texts = [processor.apply_chat_template(msg, tokenize=False).strip()]
        inputs = processor(text=texts, images=images, videos=videos, return_tensors="pt", padding=True, do_resize=False).to(model.device)
        with torch.inference_mode():
            pred = model(**inputs, bboxes=[torch.tensor(boxes, device=model.device, dtype=model.dtype)], ori_shapes=[image.size], bboxes_id=obj_id, global_id=global_id, local_text_id=text_id, global_text_id=global_text_id)
        emb = pred["object_embeddings"]
        loc = pred["objness"].sigmoid().float().flatten()
        semantic = (F.normalize(q, dim=-1) @ F.normalize(emb, dim=-1).T * model.logit_log_scale.exp() + model.logit_bias).float().sigmoid().flatten()
        score = (semantic * loc).detach().cpu().tolist()
        assert len(score) == len(boxes)
        record = {
            "family_id": ref["family_id"],
            "logical_ref": ref["logical_ref"],
            "referent_phrase": ref["referent_phrase"],
            "query_template": QUERY_TEMPLATE,
            "candidate_ids": [c["candidate_id"] for c in ref["candidates"]],
            "native_ranks": [c["native_rank"] for c in ref["candidates"]],
            "bboxes_xyxy": boxes,
            "scores": [float(x) for x in score],
            "semantic_scores": semantic.detach().cpu().tolist(),
            "localization_quality": loc.detach().cpu().tolist(),
            "status": "OK",
        }
        append_jsonl(SCORES, record)
        write_json(cache_path, record)
        existing[key] = record
        print(f"[objembed] {index}/40 {key} n={len(boxes)}", flush=True)
        del q, emb, loc, semantic, score, pred, inputs, images, videos, texts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
