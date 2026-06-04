#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eg3d

echo "CUDA_HOME=$CUDA_HOME"
rm -rf ~/.cache/torch_extensions

cd /root/Qwen3-VL
python -c "
import sys
sys.path.insert(0, '/root/eg3d/eg3d')
import legacy, dnnlib
print('CUDA OK')
print('GPU:', __import__('torch').cuda.get_device_name(0))
"
