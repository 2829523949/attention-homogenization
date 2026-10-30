#!/usr/bin/env python3
"""Contact sheets for annotation-overlap and curated-referent spot checks."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def draw_sheet(rows, dataset_root, out, title, max_items=16):
    W,H,cols,pad,header=420,360,4,10,38
    canvas=Image.new('RGB',(cols*W+(cols+1)*pad,4*(H+header+pad)+pad),'white'); d=ImageDraw.Draw(canvas)
    try:
        f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13); fs=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
    except Exception: f=fs=ImageFont.load_default()
    for idx,r in enumerate(rows[:max_items]):
        rr=idx//cols; cc=idx%cols; x0=pad+cc*(W+pad); y0=pad+rr*(H+header+pad)
        p=dataset_root/'images'/r['image_file']; im=Image.open(p).convert('RGB'); scale=min(W/im.width,H/im.height)
        im2=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
        canvas.paste(im2,(x0+(W-im2.width)//2,y0+header+(H-im2.height)//2))
        ox=x0+(W-im2.width)//2; oy=y0+header+(H-im2.height)//2
        for key,col in [('A_bbox_xyxy','#1f77b4'),('B_bbox_xyxy','#d62728')]:
            x1,y1,x2,y2=r[key]; X1=ox+x1*scale;Y1=oy+y1*scale;X2=ox+x2*scale;Y2=oy+y2*scale
            d.rectangle([X1,Y1,X2,Y2],outline=col,width=3); d.text((X1+3,Y1+3),key[0],fill=col,font=f,stroke_width=2,stroke_fill='white')
        text=r.get('label') or f"{r['family_id']} iou={r.get('ab_annotation_iou', r.get('ab_bbox_iou',0)):.3f}"
        # wrap first two lines
        words=text.split(); lines=[];cur=''
        for w in words:
            t=(cur+' '+w).strip()
            if len(t)>62: lines.append(cur);cur=w
            else:cur=t
        if cur:lines.append(cur)
        d.text((x0+2,y0+4),title,fill='black',font=f)
        for i,line in enumerate(lines[:2]): d.text((x0+2,y0+20+i*13),line,fill='black',font=fs)
    out.parent.mkdir(parents=True,exist_ok=True); canvas.save(out,quality=91)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); ap.add_argument('--curated-families',type=Path,required=True); ap.add_argument('--dataset-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    audit=load(args.audit); test=[r for r in audit if r['paco_split']=='test']
    hi=sorted(test,key=lambda r:r.get('ab_annotation_iou',0),reverse=True)[:16]
    # Add labels and flatten boxes for high-overlap sheet.
    for r in hi: r['label']=f"{r['family_id']} iou={r['ab_annotation_iou']:.3f} {r['A_category']}/{r['B_category']}"
    draw_sheet(hi,args.dataset_root,args.output_dir/'top_annotation_overlap_16.jpg','Highest annotation A/B IoU')
    contain=sorted(test,key=lambda r:max(r.get('A_inside_B_fraction',0),r.get('B_inside_A_fraction',0)),reverse=True)[:16]
    for r in contain: r['label']=f"{r['family_id']} contain={max(r['A_inside_B_fraction'],r['B_inside_A_fraction']):.3f} {r['A_category']}/{r['B_category']}"
    draw_sheet(contain,args.dataset_root,args.output_dir/'top_annotation_containment_16.jpg','Highest annotation containment')
    # Curated referents: choose examples with non-bare descriptors and some with bare category.
    fams=load(args.curated_families)
    descriptive=[f for f in fams if f.get('referent_A_info',{}).get('basis')!='unique_category' or f.get('referent_B_info',{}).get('basis')!='unique_category']
    rng=random.Random(20261029); rng.shuffle(descriptive); sample=descriptive[:16]
    # Enrich sample with annotation boxes from audit and label with descriptors.
    amap={r['family_id']:r for r in audit}
    outrows=[]
    for f in sample:
        a=amap[f['family_id']]; rr={'image_file':f['image_relpath'].replace('images/',''), 'A_bbox_xyxy':f['bbox_A_xyxy'], 'B_bbox_xyxy':f['bbox_B_xyxy'],
            'label':f"{f['referent_A']} / {f['referent_B']}"}
        outrows.append(rr)
    draw_sheet(outrows,args.dataset_root,args.output_dir/'curated_referent_preview_16.jpg','Curated referents')
if __name__=='__main__':main()
