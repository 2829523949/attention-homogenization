"""
Fair comparison: VLM EN vs CLIP EN on IDENTICAL fixed test set
"""
import os, sys, math, torch, random
sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Fix seed for deterministic val split
random.seed(42)

from train_mixed_v2_en import MixedPoseDatasetEn, generate_diverse_text_en, EN_TEMPLATES_RAW, EN_TEMPLATE_TEXTS
from bridge import pose_6d_to_camera_params

# ============================================================
# Create a FIXED test set (not the one from training, a new one)
# ============================================================
print("Creating fixed test set...")
test_texts = []
# Add all 21 templates as test cases
for yaw, pitch_off, text in EN_TEMPLATES_RAW:
    test_texts.append({'text': text, 'yaw': yaw, 'pitch_offset': pitch_off})

# Add 30 additional diverse texts generated with fixed seed
random.seed(123)
for _ in range(30):
    yaw = random.uniform(-0.79, 0.79)
    pitch_off = random.uniform(-0.6, 0.6)
    text = generate_diverse_text_en(yaw, pitch_off)
    test_texts.append({'text': text, 'yaw': round(yaw, 2), 'pitch_offset': round(pitch_off, 2)})
random.seed(None)

print(f"Fixed test set: {len(test_texts)} samples")

# Build fixed test data
test_data = []
for item in test_texts:
    pitch = math.pi / 2 + item['pitch_offset']
    raw_6d = torch.tensor([[item['yaw'], pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32)
    test_data.append({
        'text': item['text'],
        'yaw_gt': item['yaw'],
        'pitch_gt': pitch,
    })
print()

# ============================================================
# Test VLM EN
# ============================================================
print("=" * 60)
print("Testing VLM EN...")
print("=" * 60)
from bridge import QwenEG3DBridge, get_llm_hidden

vlm = QwenEG3DBridge()
ckpt = torch.load("/root/Qwen3-VL/checkpoints/vlm_en_epoch_080.pt", map_location="cuda")
vlm.pose.load_state_dict(ckpt["pose"])
vlm.eval()

vl_yaw_errs, vl_pit_errs = [], []
with torch.no_grad():
    for v in test_data:
        hidden = get_llm_hidden(vlm.qwen, vlm.processor, v['text'])
        p6d = vlm.pose(hidden.unsqueeze(0))
        ye = abs(p6d[0, 0].item() - v['yaw_gt'])
        ye = min(ye, 2 * math.pi - ye)
        vl_yaw_errs.append(ye)
        vl_pit_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))

vl_ye = sum(vl_yaw_errs)/len(vl_yaw_errs) * 180 / math.pi
vl_pe = sum(vl_pit_errs)/len(vl_pit_errs) * 180 / math.pi
vl_ym = sorted(vl_yaw_errs)[len(vl_yaw_errs)//2] * 180 / math.pi
vl_pm = sorted(vl_pit_errs)[len(vl_pit_errs)//2] * 180 / math.pi
print(f"VLM EN  yaw: mean={vl_ye:.2f}°  median={vl_ym:.2f}°")
print(f"VLM EN  pitch: mean={vl_pe:.2f}°  median={vl_pm:.2f}°")
print()

# ============================================================
# Test CLIP EN
# ============================================================
print("=" * 60)
print("Testing CLIP EN...")
print("=" * 60)
from lightweight_bridge import LightweightBridge, get_embedding_for_training

clip = LightweightBridge(encoder_name="clip")
ckpt2 = torch.load("/root/Qwen3-VL/checkpoints/clip_en_epoch_080.pt", map_location="cuda")
clip.projection.load_state_dict(ckpt2["projection"])
clip.pose.load_state_dict(ckpt2["pose"])
clip.eval()

cl_yaw_errs, cl_pit_errs = [], []
with torch.no_grad():
    for v in test_data:
        emb = get_embedding_for_training(clip, v['text'])
        h = clip.projection(emb)
        p6d = clip.pose(h)
        ye = abs(p6d[0, 0].item() - v['yaw_gt'])
        ye = min(ye, 2 * math.pi - ye)
        cl_yaw_errs.append(ye)
        cl_pit_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))

cl_ye = sum(cl_yaw_errs)/len(cl_yaw_errs) * 180 / math.pi
cl_pe = sum(cl_pit_errs)/len(cl_pit_errs) * 180 / math.pi
cl_ym = sorted(cl_yaw_errs)[len(cl_yaw_errs)//2] * 180 / math.pi
cl_pm = sorted(cl_pit_errs)[len(cl_pit_errs)//2] * 180 / math.pi
print(f"CLIP EN  yaw: mean={cl_ye:.2f}°  median={cl_ym:.2f}°")
print(f"CLIP EN  pitch: mean={cl_pe:.2f}°  median={cl_pm:.2f}°")
print()

# ============================================================
# Final Comparison
# ============================================================
print("=" * 60)
print("FAIR COMPARISON (same test set)")
print("=" * 60)
print(f"{'Model':15s} {'Params':>8s} {'Yaw Mean':>10s} {'Yaw Med':>10s} {'Pitch Mean':>10s} {'Pitch Med':>10s}")
print("-" * 65)
print(f"{'VLM (Qwen3-VL)':15s} {'2,000M':>8s} {vl_ye:>8.2f}°  {vl_ym:>8.2f}°  {vl_pe:>8.2f}°  {vl_pm:>8.2f}°")
print(f"{'CLIP (63M)':15s} {'63M':>8s} {cl_ye:>8.2f}°  {cl_ym:>8.2f}°  {cl_pe:>8.2f}°  {cl_pm:>8.2f}°")
print("-" * 65)
print(f"{'Ratio':15s} {'32x':>8s} {'':>8s} {'':>8s} {'':>8s} {'':>8s}")

# Print some sample predictions
print("\nSample predictions:")
print(f"{'Text':40s} {'GT yaw':>8s} {'VLM yaw':>8s} {'CLIP yaw':>8s}")
for i in range(min(5, len(test_data))):
    v = test_data[i]
    print(f"{v['text'][:38]:38s} {v['yaw_gt']:>+7.2f}  ", end="")
    # Already computed above, just show
