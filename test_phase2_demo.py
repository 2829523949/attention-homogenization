"""
Phase 2 Demo: PTI 反演 + 角度编辑
用法: cd /root/Qwen3-VL && python test_phase2_demo.py
"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

bridge = QwenEG3DBridge()
bridge.pose.load_state_dict(torch.load(
    "/root/Qwen3-VL/checkpoints/mix_v2_epoch_030.pt", map_location="cuda")["pose"])
bridge.eval()

from invert_face import invert_face_wplus

zid, save_dir = 5, "/root/output_phase2_demo"
img_path = f"/root/edit_data/images/z{zid:03d}_y+0.00_p+0.00.png"

w_plus = invert_face_wplus(bridge, img_path, steps=1000, save_dir=save_dir)

edit_texts = ["正面平视", "左侧45度平视", "右侧45度平视", "正面俯视", "正面仰视"]
os.makedirs(f"{save_dir}/edits", exist_ok=True)

print("\n" + "="*40)
print("角度编辑")
print("="*40)
for text in edit_texts:
    with torch.no_grad():
        out = bridge(text, w_plus=w_plus.cuda(), return_raw_6d=True)
    img, raw = out[0], out[2]
    py, pp = raw[0,0].item(), raw[0,1].item()
    print(f"{text:16s} → yaw={py:+.2f} pitch={pp:+.2f}")
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1),
                                  f"{save_dir}/edits/{text}.png")

print(f"\n结果: {save_dir}/")
