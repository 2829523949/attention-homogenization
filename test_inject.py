"""测试注入式图文角度编辑"""
import sys, torch, torchvision
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge
from train_edit import EditDataset

b = QwenEG3DBridge()
# 加载预训练
import os
ckpt_dir = "/root/Qwen3-VL/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('mix_epoch_') and f.endswith('.pt')],
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
if ckpts:
    ckpt = torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location='cuda')
    b.pose.load_state_dict(ckpt['pose'])
    print(f"pose: {ckpts[-1]}")
else:
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('text_only_') and f.endswith('.pt')],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if ckpts:
        b.pose.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location='cuda')['pose'])
        print(f"pose: {ckpts[-1]}")

d = EditDataset()
img_path, text, _ = d.train_pairs[0], d.train_pairs[0]['text'], d.train_pairs[0]['target_raw_6d']
print(f"\n测试: img={img_path['image']} text='{text}'")
img = b(text, image_path=f"/root/edit_data/images/{img_path['image']}", return_gates=True)
print(f"输出: img={img[0].shape} cam={img[1].shape}")
print(f"门控: {img[2].data.cpu().numpy().round(2)}")
torchvision.utils.save_image((img[0][0]*0.5+0.5).clamp(0,1), "/root/output_edit_test/inject_test.png")
print("保存到 /root/output_edit_test/inject_test.png")
