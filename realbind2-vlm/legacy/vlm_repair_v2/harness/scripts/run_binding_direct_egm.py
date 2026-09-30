#!/usr/bin/env python3
"""EGM-4B direct referent->bbox grounding with the official Qwen3 eval prompt."""
from __future__ import annotations
import hashlib,json,math,os,re,shutil,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/high_recall_binding_repair/p3_direct_grounding'
POOL=ROOT/'outputs/high_recall_binding_repair/p0_qwen_independent/candidate_pool_manifest.json'
MODEL=Path('/root/vlm_egm_4b')
PROMPT='Locate {referent}, output its bbox coordinates using JSON format'
MODEL_ID='nvidia/EGM-4B'
def digest_bytes(x):return hashlib.sha256(x).hexdigest()
def digest_file(p):return digest_bytes(Path(p).read_bytes())
def cache_key(row,manifest_hash,image_hash,checkpoint_hash):
    payload={'family_id':row['family_id'],'logical_ref':row['logical_ref'],'image_sha256':image_hash,'referent_sha256':digest_bytes(row['referent_phrase'].encode()),'prompt_sha256':digest_bytes(PROMPT.format(referent=row['referent_phrase']).encode()),'manifest_sha256':manifest_hash,'model':MODEL_ID,'checkpoint_fingerprint':checkpoint_hash,'generation':{'max_new_tokens':2048,'do_sample':False,'use_cache':True,'dtype':'bfloat16'}}
    return digest_bytes(json.dumps(payload,sort_keys=True).encode())
def load_existing(path,expected):
    done=set()
    if not path.exists():return done
    raw=path.read_bytes();lines=raw.splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:r=json.loads(line)
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise ValueError(f'malformed nonfinal JSONL record {i+1}')
            backup=path.with_suffix('.jsonl.crash_original')
            if not backup.exists():shutil.copy2(path,backup)
            recovery=path.with_suffix('.jsonl.recovered_complete')
            with recovery.open('wb') as h:h.write(b''.join(lines[:-1]));h.flush();os.fsync(h.fileno())
            os.replace(recovery,path)
            return load_existing(path,expected)
        k=(r.get('family_id'),r.get('logical_ref'))
        if k not in expected:raise ValueError(f'unexpected P3 key {k}')
        if k in done:raise ValueError(f'duplicate P3 key {k}')
        e=expected[k]
        if r.get('cache_key')!=e['cache_key'] or r.get('prompt')!=e['prompt'] or r.get('image_path')!=e['image_path'] or r.get('model')!=MODEL_ID or r.get('status') not in ('OK','INVALID_BBOX'):raise ValueError(f'incompatible P3 record {k}')
        done.add(k)
    return done
def validate_checkpoint():
    index=MODEL/'model.safetensors.index.json'
    if not index.is_file():raise FileNotFoundError(f'checkpoint index missing: {index}')
    shards=set(json.loads(index.read_text())['weight_map'].values())
    missing=[x for x in shards if not (MODEL/x).is_file() or (MODEL/x).stat().st_size<1000000]
    if missing:raise FileNotFoundError(f'incomplete checkpoint shards: {missing}')
    return shards
def checkpoint_fingerprint(shards):
    files=[MODEL/'config.json',MODEL/'model.safetensors.index.json']
    payload={'model':MODEL_ID,'metadata_sha256':{p.name:digest_file(p) for p in files},'shards':{name:(MODEL/name).stat().st_size for name in sorted(shards)}}
    return digest_bytes(json.dumps(payload,sort_keys=True).encode())
NUM=r'[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?'
def parse_box(raw,size):
    w,h=size
    m=re.search(r'"bbox_2d"\s*:\s*\[\s*('+NUM+r')\s*,\s*('+NUM+r')\s*,\s*('+NUM+r')\s*,\s*('+NUM+r')\s*\]',raw,re.S)
    if not m:m=re.search(r'\[\s*('+NUM+r')\s*,\s*('+NUM+r')\s*,\s*('+NUM+r')\s*,\s*('+NUM+r')\s*\]',raw,re.S)
    if not m:return None
    b=[float(m.group(i)) for i in range(1,5)]
    if not all(math.isfinite(x) for x in b):return None
    b=[min(1000.,max(0.,x)) for x in b]
    x0,y0,x1,y1=b
    out=[float(round(x0*w/1000)),float(round(y0*h/1000)),float(round(x1*w/1000)),float(round(y1*h/1000))]
    out=[min(w-1,max(0,out[0])),min(h-1,max(0,out[1])),min(w-1,max(0,out[2])),min(h-1,max(0,out[3]))]
    return out if out[2]>out[0] and out[3]>out[1] else None
def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    OUT.mkdir(parents=True,exist_ok=True)
    shards=validate_checkpoint()
    checkpoint_hash=checkpoint_fingerprint(shards)
    manifest_hash=digest_file(POOL)
    p=json.loads(POOL.read_text())['referents'];assert len(p)==40
    expected={(r['family_id'],r['logical_ref']):{'cache_key':cache_key(r,manifest_hash,digest_file(r['image_path']),checkpoint_hash),'prompt':PROMPT.format(referent=r['referent_phrase']),'image_path':r['image_path']} for r in p}
    path=OUT/'direct_regions.jsonl';done=load_existing(path,expected)
    processor=AutoProcessor.from_pretrained(MODEL)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='auto').eval()
    for r in p:
        key=(r['family_id'],r['logical_ref'])
        if key in done:continue
        image=Image.open(r['image_path']).convert('RGB');prompt=PROMPT.format(referent=r['referent_phrase'])
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=2048,do_sample=False,use_cache=True)
        raw=processor.decode(out[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
        bbox=parse_box(raw,image.size)
        rec={'family_id':key[0],'logical_ref':key[1],'referent_phrase':r['referent_phrase'],'image_path':r['image_path'],'image_size':image.size,'prompt':prompt,'raw_output':raw,'selected_bbox_xyxy':bbox,'status':'OK' if bbox else 'INVALID_BBOX','model':MODEL_ID,'normalized_bbox_0_1000':True,'cache_key':expected[key]['cache_key'],'manifest_sha256':manifest_hash,'checkpoint_fingerprint':checkpoint_hash,'image_sha256':digest_file(r['image_path']),'prompt_sha256':digest_bytes(prompt.encode()),'generation_config':{'max_new_tokens':2048,'do_sample':False,'use_cache':True,'dtype':'bfloat16'}}
        with path.open('a') as h:h.write(json.dumps(rec)+'\n');h.flush();os.fsync(h.fileno())
        print('EGM',len(done)+1,key,rec['status'],bbox,flush=True);done.add(key)
if __name__=='__main__':main()
