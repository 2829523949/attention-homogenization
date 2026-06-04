import sys, json, struct, pickle
import numpy as np

# Try to read attention_data.npy - might be old data
# Instead, check the similarity PNG files by reading pixel data from images
from PIL import Image
import os

fig_dir = '/root/Qwen3-VL/paper/figures'
for f in sorted(os.listdir(fig_dir)):
    if f.startswith('similarity_') and f.endswith('.png'):
        img = Image.open(os.path.join(fig_dir, f))
        print(f'{f}: {img.size}, {img.mode}')
