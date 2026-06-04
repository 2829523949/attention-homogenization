#!/bin/bash
source /root/miniconda3/bin/activate next3d_qwen
cd /root/Qwen3-VL

COCO_DIR="/datasets/coco/val2017"
TMP_DIR="/tmp/grad_results"
mkdir -p "$TMP_DIR"
N_IMAGES=30

IMAGES=($(ls "$COCO_DIR" | head -$N_IMAGES))
echo "Processing ${#IMAGES[@]} images, one per process (GPU fully reset each time)..."

for i in "${!IMAGES[@]}"; do
    img="${IMAGES[$i]}"
    echo "[$((i+1))/${#IMAGES[@]}] $img"
    python gradient_single.py "$img" "$TMP_DIR/result_$i.json"
    echo "  Done."
done

# Combine all results into one JSON
python3 -c "
import json, os, glob
results = []
for p in sorted(glob.glob('$TMP_DIR/result_*.json'), key=lambda x: int(x.split('_')[-1].split('.')[0])):
    with open(p) as f:
        results.append(json.load(f))
with open('/root/Qwen3-VL/paper/figures/fixed_test/gradient_30_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Combined {len(results)} results.')
"

rm -rf "$TMP_DIR"
echo "ALL DONE."
