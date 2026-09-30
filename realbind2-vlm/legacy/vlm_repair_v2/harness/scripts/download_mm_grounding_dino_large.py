#!/usr/bin/env python3
"""Download the published OpenMMLab HF conversion with ranged transfers and hash check."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import requests

URL='https://huggingface.co/openmmlab-community/mm_grounding_dino_large_all/resolve/main/model.safetensors'
DEST=Path('/mnt/d/vlm_repair_v2/models/mm_grounding_dino/mm_grounding_dino_large_all/model.safetensors')
SIZE=1376363740
DIGEST='1802856b98c66cdf6c4b1e9841320f7b76b039c9cb54e88061de1c2517b53822'
N=12

def segment(i):
 start=i*SIZE//N;end=(i+1)*SIZE//N-1
 part=DEST.with_suffix(f'.part{i:02d}')
 if part.exists() and part.stat().st_size==end-start+1:return part
 offset=part.stat().st_size if part.exists() else 0
 if end-start+1-offset>16<<20:
  spans=[];begin=start+offset
  while begin<=end:
   finish=min(end,begin+(8<<20)-1);spans.append((begin,finish));begin=finish+1
  def chunk(span):
   a,b=span;tmp=DEST.with_suffix(f'.part{i:02d}.{a}.chunk')
   if tmp.exists() and tmp.stat().st_size==b-a+1:return tmp
   for attempt in range(30):
    try:
     offset=tmp.stat().st_size if tmp.exists() else 0
     if offset==b-a+1:return tmp
     response=requests.get(URL,headers={'Range':f'bytes={a+offset}-{b}'},stream=True,timeout=(20,35));response.raise_for_status()
     assert response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {a+offset}-{b}/')
     with tmp.open('ab') as out:
      for block in response.iter_content(1<<20):out.write(block)
     assert tmp.stat().st_size==b-a+1
     return tmp
    except Exception as e:print('chunk retry',i,a,attempt+1,repr(e),flush=True)
   raise RuntimeError(f'chunk {i} {a}-{b} failed')
  with ThreadPoolExecutor(max_workers=10) as pool:chunks=list(pool.map(chunk,spans))
  with part.open('ab') as out:
   for tmp in chunks:
    with tmp.open('rb') as source:
     while block:=source.read(1<<20):out.write(block)
    tmp.unlink()
  assert part.stat().st_size==end-start+1
  print('segment',i,'complete',flush=True)
  return part
 for attempt in range(20):
  try:
   offset=part.stat().st_size if part.exists() else 0
   if offset==end-start+1:return part
   response=requests.get(URL,headers={'Range':f'bytes={start+offset}-{end}'},stream=True,timeout=(20,35))
   response.raise_for_status()
   assert response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {start+offset}-{end}/')
   with part.open('ab') as out:
    for chunk in response.iter_content(1<<20):out.write(chunk)
   assert part.stat().st_size==end-start+1
   print('segment',i,'complete',flush=True)
   return part
  except Exception as e:print('segment',i,'attempt',attempt+1,'failed',repr(e),flush=True)
 raise RuntimeError(f'segment {i} failed')

def main():
 with ThreadPoolExecutor(max_workers=N) as pool:parts=list(pool.map(segment,range(N)))
 digest=sha256()
 with DEST.open('wb') as out:
  for p in parts:
   with p.open('rb') as source:
    while chunk:=source.read(1<<20):out.write(chunk);digest.update(chunk)
 assert digest.hexdigest()==DIGEST,(digest.hexdigest(),DIGEST)
 for p in parts:p.unlink()
 print('verified',DEST,DIGEST,flush=True)

if __name__=='__main__':main()
