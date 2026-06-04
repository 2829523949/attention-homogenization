"""诊断并修复 CUDA 编译问题"""
import os, sys, subprocess, glob

# 1. 检查 CUDA_HOME
cuda_home = os.environ.get('CUDA_HOME', 'NOT SET')
print(f"CUDA_HOME={cuda_home}")

# 2. 找 cuda_runtime_api.h
for p in ['/usr/include/cuda_runtime_api.h', '/usr/local/cuda/include/cuda_runtime_api.h',
          '/usr/lib/cuda/include/cuda_runtime_api.h']:
    print(f"  {p}: {'EXISTS' if os.path.exists(p) else 'NOT FOUND'}")

# 3. 找 nvcc
nvcc = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
print(f"nvcc: {nvcc.stdout.strip()}")

# 4. 找 CUDA toolkit 安装位置  
result = subprocess.run(['dpkg', '-L', 'nvidia-cuda-dev'], capture_output=True, text=True)
if result.returncode == 0:
    cuda_files = [l for l in result.stdout.split('\n') if 'include' in l]
    print(f"nvidia-cuda-dev include files: {len(cuda_files)}")

# 5. 测试 PyTorch cpp_extension 如何找 CUDA
import torch
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"torch.cuda.is_compiled(): {torch.cuda.is_compiled()}")

# 试试直接设置正确的 CUDA_HOME 路径
# /usr/lib/cuda 是 toolkit 位置
os.environ['CUDA_HOME'] = '/usr/lib/cuda'
os.environ['CPLUS_INCLUDE_PATH'] = '/usr/include:' + os.environ.get('CPLUS_INCLUDE_PATH', '')

# 编译测试
from torch.utils.cpp_extension import load
try:
    ext = load(name='test_cuda', sources=[], verbose=True)
    print("COMPILE OK ✅")
except Exception as e:
    print(f"COMPILE FAILED ❌: {e}")
