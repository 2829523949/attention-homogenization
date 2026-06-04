"""测试训练后的 Injector 效果"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

# 加载训练好的 injector
ckpt = torch.load("/root/Qwen3-VL/checkpoints/injector_trained.pt", map_location='cuda')
b.injector.load_state_dict(ckpt['injector'])
print("加载 injector_trained.pt")

# 加载 pose
mix = sorted([f for f in os.listdir("/root/Qwen3-VL/checkpoints") if f.startswith('mix_epoch_') and f.endswith('.pt')],
             key=lambda x: int(x.split('_')[-1].split('.')[0]))
if mix:
    b.pose.load_state_dict(torch.load(os.path.join("/root/Qwen3-VL/checkpoints", mix[-1]), map_location='cuda')['pose'])

b.eval()
os.makedirs("/root/output_injector_test", exist_ok=True)

# 测试样本: 不同身份同一角度
tests = [
    ("/root/edit_data/images/z004_y+0.00_p+0.00.png", "正面平视", "身份4正面"),
    ("/root/edit_data/images/z005_y+0.00_p+0.00.png", "正面平视", "身份5正面"),
    ("/root/edit_data/images/z004_y+0.79_p+0.00.png", "左侧45度平视", "身份4左45"),
]

for img_path, text, name in tests:
    if not os.path.exists(img_path): continue
    with torch.no_grad():
        out = b(text, image_path=img_path, return_gates=True)
    img, gates = out[0], out[2]
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), f"/root/output_injector_test/{name}.png")
    print(f"{name}: gates={gates.cpu().numpy().round(2)}")

print("\n保存到 /root/output_injector_test/")
print("对比: 身份4正面 vs 身份5正面 应该五官轮廓相似但细节不同")
