#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eg3d
cd /root/Qwen3-VL
rm -rf ~/.cache/torch_extensions
python invert_pti.py
