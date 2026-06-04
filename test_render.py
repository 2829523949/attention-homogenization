"""最终渲染测试：加载 adapter + pose，生成多样化文本的人脸图像"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

bridge = QwenEG3DBridge()
ckpt_dir = "/root/Qwen3-VL/checkpoints"

# 加载 adapter 权重
adapter_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('adapter_') and f.endswith('.pt')],
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
if adapter_ckpts:
    ckpt = torch.load(os.path.join(ckpt_dir, adapter_ckpts[-1]), map_location='cuda')
    bridge.adapter.load_state_dict(ckpt['adapter'])
    print(f"加载 adapter: {adapter_ckpts[-1]}")

# 加载 pose 权重 (自动加载最新的)
pose_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('text_only_') and f.endswith('.pt')],
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
if pose_ckpts:
    bridge.pose.load_state_dict(torch.load(os.path.join(ckpt_dir, pose_ckpts[-1]), map_location='cuda')['pose'])
    print(f"加载 pose: {pose_ckpts[-1]}")

# 测试文本 (多样化描述)
test_texts = [
    "正面平视", "正面仰视", "正面俯视",
    "从左侧45度方向平视拍摄", "从左前方高位俯视拍摄",
    "右侧90度水平拍摄", "从正后方平视拍摄",
    "左后方仰角度拍摄", "右后方方向俯视拍摄",
    "从后脑勺方向平视拍摄",
]

os.makedirs("/root/output_render", exist_ok=True)
bridge.eval()

print("=" * 60)
print("渲染测试")
print("=" * 60)

with torch.no_grad():
    for text in test_texts:
        img, cam = bridge(text)
        yaw, pitch = bridge.pose(bridge.adapter(
            sys.modules['bridge'].get_llm_hidden(bridge.qwen, bridge.processor, text)
        ).unsqueeze(0))[0, :2].tolist()
        print(f"'{text:16s}' → yaw={yaw:+.2f} pitch={pitch:+.2f}")
        
        img_out = img[0] * 0.5 + 0.5
        torchvision.utils.save_image(img_out.clamp(0,1), f"/root/output_render/{text}.png")

print(f"\n图像已保存到 /root/output_render/")
print("=" * 60)
