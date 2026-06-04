"""演示: 模拟"真实图→角度编辑"流程 (用预计算w+代替反演)"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge
from PIL import Image

b = QwenEG3DBridge()
b.pose.load_state_dict(torch.load("/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")["pose"])
b.eval()

# 加载预计算的 w+ (真实场景需用 e4e/PTI 从图提取)
identity_w = torch.load("/root/edit_data/identity_w.pt", map_location="cuda")

os.makedirs("/root/output_real_demo", exist_ok=True)

# 模拟: 输入"真实图" (实际是数据集图, 但用 w+ 固定身份)
for zid in [10, 25, 40]:  # 不同人
    input_img = f"/root/edit_data/images/z{zid:03d}_y+0.00_p+0.00.png"
    w = identity_w[zid]
    
    # 显示"输入"
    ref = torchvision.io.read_image(input_img).float() / 255.0
    torchvision.utils.save_image(ref, f"/root/output_real_demo/input_z{zid}.png")
    
    # 文本编辑角度
    for text in ["正面平视", "左侧45度平视", "右侧45度仰视"]:
        with torch.no_grad():
            out = b(text, w_plus=w, return_raw_6d=True)
        img, raw = out[0], out[2]
        torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), 
                                     f"/root/output_real_demo/z{zid}_{text}.png")
        print(f"输入z{zid} | '{text}' → yaw={raw[0,0]:+.2f} pitch={raw[0,1]:+.2f}")

print("\n保存到 /root/output_real_demo/")
print("input_z*.png = 模拟输入的真实人脸")
print("z*_* = 同一人不同角度")
print("\n注: 真实场景需先用 e4e/PTI 从图提取 w+, 此处用预计算 w+ 演示")
