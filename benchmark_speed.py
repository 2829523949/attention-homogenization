"""VLM vs CLIP 推理速度基准测试
输出: 延迟统计 + 保存结果到 benchmark_results.json + 生成论文用 Figure
"""

import sys, os, json, time, math
sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import numpy as np

# ===== 测试配置 =====
N_WARMUP = 10
N_TRIALS = 100
TEXT = "A person looking to the right at 30 degrees"

# ===== 工具 =====
def timed(fn, n_warmup=5, n_trials=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    mems = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        mems.append(torch.cuda.max_memory_allocated() / 1024**3)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_per_sec": 1000.0 / float(np.mean(latencies)),
        "peak_gpu_gb": float(np.max(mems)),
        "raw_ms": latencies,
    }

def print_stats(name, s):
    print(f"\n  {name}")
    print(f"    Mean: {s['mean_ms']:8.2f} ms | Median: {s['median_ms']:8.2f} ms | P95: {s['p95_ms']:8.2f} ms")
    print(f"    Throughput: {s['throughput_per_sec']:7.1f} samples/sec | Peak VRAM: {s['peak_gpu_gb']:.2f} GB")

# ===== 1. VLM =====
print("[1/4] Loading VLM...")
from bridge import QwenEG3DBridge, get_llm_hidden

vlm_b = QwenEG3DBridge()
vlm_b.eval()
ckpt = torch.load("/root/Qwen3-VL/checkpoints/vlm_en_epoch_080.pt", map_location="cuda")
vlm_b.pose.load_state_dict(ckpt["pose"])
vlm_params = sum(p.numel() for p in vlm_b.qwen.parameters())
vlm_params += sum(p.numel() for p in vlm_b.pose.parameters())

# warmup
_ = get_llm_hidden(vlm_b.qwen, vlm_b.processor, TEXT)
_ = vlm_b.pose(torch.randn(1, 2048, device='cuda'))
torch.cuda.synchronize()
print("  VLM warmup OK")

# ===== 2. CLIP =====
print("\n[2/4] Loading CLIP...")
from lightweight_bridge import LightweightBridge, get_embedding_for_training

clip_b = LightweightBridge(encoder_name='clip')
ckpt = torch.load("/root/Qwen3-VL/checkpoints/clip_en_epoch_080.pt", map_location="cuda")
clip_b.projection.load_state_dict(ckpt["projection"])
clip_b.pose.load_state_dict(ckpt["pose"])
clip_b.eval()
clip_params = sum(p.numel() for p in clip_b.text_model.parameters())
clip_params += sum(p.numel() for p in clip_b.projection.parameters())
clip_params += sum(p.numel() for p in clip_b.pose.parameters())

# warmup
emb = get_embedding_for_training(clip_b, TEXT)
_ = clip_b.projection(emb.unsqueeze(0))
torch.cuda.synchronize()
print("  CLIP warmup OK")

# ===== 3. Benchmark =====
print(f"\n[3/4] Benchmarking ({N_TRIALS} trials, {N_WARMUP} warmup)...")

def vlm_full():
    h = get_llm_hidden(vlm_b.qwen, vlm_b.processor, TEXT)
    _ = vlm_b.pose(h.unsqueeze(0))

def vlm_enc():
    _ = get_llm_hidden(vlm_b.qwen, vlm_b.processor, TEXT)

def clip_full():
    emb = get_embedding_for_training(clip_b, TEXT)
    h = clip_b.projection(emb.unsqueeze(0))
    _ = clip_b.pose(h)

def clip_enc():
    _ = get_embedding_for_training(clip_b, TEXT)

vlm_s = timed(vlm_full, N_WARMUP, N_TRIALS)
vlm_e = timed(vlm_enc, N_WARMUP, N_TRIALS)
clip_s = timed(clip_full, N_WARMUP, N_TRIALS)
clip_e = timed(clip_enc, N_WARMUP, N_TRIALS)

print("\n--- Results ---")
print_stats("VLM 2B — Full Pipeline", vlm_s)
print_stats("VLM 2B — Encoder Only", vlm_e)
print_stats("CLIP 63M — Full Pipeline", clip_s)
print_stats("CLIP 63M — Encoder Only", clip_e)

# ===== 汇总表 =====
print(f"\n{'='*65}")
print(f"  {'':25s} {'VLM 2B':>12s} {'CLIP 63M':>12s} {'Speedup':>12s}")
print(f"  {'─'*63}")
print(f"  {'Full Pipeline (ms)':25s} {vlm_s['mean_ms']:>8.1f}ms {clip_s['mean_ms']:>8.1f}ms {vlm_s['mean_ms']/clip_s['mean_ms']:>7.1f}×")
print(f"  {'Encoder Only (ms)':25s} {vlm_e['mean_ms']:>8.1f}ms {clip_e['mean_ms']:>8.1f}ms {vlm_e['mean_ms']/clip_e['mean_ms']:>7.1f}×")
print(f"  {'Throughput (/sec)':25s} {vlm_s['throughput_per_sec']:>8.0f}  {clip_s['throughput_per_sec']:>8.0f}  {clip_s['throughput_per_sec']/vlm_s['throughput_per_sec']:>7.1f}×")
print(f"  {'Peak VRAM (GB)':25s} {vlm_s['peak_gpu_gb']:>8.2f}  {clip_s['peak_gpu_gb']:>8.2f}  {vlm_s['peak_gpu_gb']/clip_s['peak_gpu_gb']:>7.1f}×")
print(f"  {'Parameters (M)':25s} {vlm_params/1e6:>8.0f}M  {clip_params/1e6:>8.0f}M  {vlm_params/clip_params:>7.1f}×")
print(f"{'='*65}")

# ===== 保存 JSON =====
results = {
    "vlm": {"params_m": vlm_params/1e6, "full": vlm_s, "enc": vlm_e},
    "clip": {"params_m": clip_params/1e6, "full": clip_s, "enc": clip_e},
    "speedup_full": vlm_s["mean_ms"] / clip_s["mean_ms"],
    "gpu": torch.cuda.get_device_name(0),
}

out = "/root/Qwen3-VL/checkpoints/benchmark_results.json"
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults -> {out}")

# ===== 生成 Figure =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.patch.set_facecolor('white')

colors = {'VLM': '#E74C3C', 'CLIP': '#3498DB'}
labels = ['VLM 2B', 'CLIP 63M']
full = [vlm_s['mean_ms'], clip_s['mean_ms']]
full_s = [vlm_s['std_ms'], clip_s['std_ms']]
enc = [vlm_e['mean_ms'], clip_e['mean_ms']]
enc_s = [vlm_e['std_ms'], clip_e['std_ms']]
vram = [vlm_s['peak_gpu_gb'], clip_s['peak_gpu_gb']]
param = [vlm_params/1e6, clip_params/1e6]

# A: Full latency
ax = axes[0]
bars = ax.bar(labels, full, yerr=full_s, capsize=6, color=['#E74C3C','#3498DB'], edgecolor='black', error_kw={'lw':2})
for b, v in zip(bars, full):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(full_s)*0.8, f"{v:.0f} ms", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('A. Full Pipeline Latency', fontsize=13, fontweight='bold')

# B: Encoder latency
ax = axes[1]
bars = ax.bar(labels, enc, yerr=enc_s, capsize=6, color=['#E74C3C','#3498DB'], edgecolor='black', error_kw={'lw':2})
for b, v in zip(bars, enc):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(enc_s)*0.8, f"{v:.0f} ms", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_title('B. Encoder-Only Latency', fontsize=13, fontweight='bold')

# C: Params + VRAM
ax = axes[2]
x = np.arange(2)
w = 0.3
b1 = ax.bar(x-w/2, param, w, color='#2ECC71', edgecolor='black', label='Parameters (M)')
ax.set_ylabel('Parameters (M)', fontsize=12, color='#2ECC71')
ax.tick_params(axis='y', labelcolor='#2ECC71')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax2 = ax.twinx()
b2 = ax2.bar(x+w/2, vram, w, color='#9B59B6', edgecolor='black', label='Peak VRAM (GB)')
ax2.set_ylabel('Peak VRAM (GB)', fontsize=12, color='#9B59B6')
ax2.tick_params(axis='y', labelcolor='#9B59B6')
ax2.legend([b1, b2], ['Parameters (M)', 'Peak VRAM (GB)'], loc='upper left', fontsize=9)
ax.set_title('C. Model Size & Memory', fontsize=13, fontweight='bold')

sp = vlm_s['mean_ms'] / clip_s['mean_ms']
fig.suptitle(f'CLIP is {sp:.1f}× faster than VLM in full-pipeline inference',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fp = "/root/Qwen3-VL/paper_figures/speed_benchmark.png"
plt.savefig(fp, dpi=200, bbox_inches='tight')
plt.close()
print(f"Figure -> {fp}")
print("\nDone!")
