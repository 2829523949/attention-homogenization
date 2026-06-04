import numpy as np
data = np.load("/root/Qwen3-VL/paper/figures/attention_data.npy", allow_pickle=True)
d = data.item() if hasattr(data, 'item') else data
print("Images:", list(d['heatmaps'].keys()))
for fname in list(d['heatmaps'].keys())[:1]:
    print(f"\n{fname}:")
    for q, hm in d['heatmaps'][fname].items():
        arr = np.array(hm)
        print(f"  {q:20s}: min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}")
    break
