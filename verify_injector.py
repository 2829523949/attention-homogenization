"""验证 injector 结构"""
import sys
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import DeepStackInjector, QwenEG3DBridge
b = QwenEG3DBridge()
inj = b.injector
print(f"Injector convs: {len(inj.inject_convs)}")
for k, c in inj.inject_convs.items():
    print(f"  res={k}: {c.out_channels}ch")
print(f"res_to_ds: {inj.res_to_ds}")
print("OK")
