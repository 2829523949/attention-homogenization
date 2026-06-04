"""最终验证: 多样化自然语言 + 多身份 + 角度精度"""
import sys, torch, torchvision, os, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params

b = QwenEG3DBridge()
b.pose.load_state_dict(torch.load(
    "/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")["pose"])
b.eval()

identity_w = torch.load("/root/edit_data/identity_w.pt", map_location="cuda")
data = torch.load("/root/edit_data/dataset.pt")

os.makedirs("/root/output_diverse_test", exist_ok=True)

# 多样化自然语言测试集
diverse_texts = [
    # 模板文本 (精确)
    ("正面平视", 0.0, 0.0),
    ("正面仰视", 0.0, 0.79),
    ("正面俯视", 0.0, -0.79),
    ("左侧45度平视", -0.79, 0.0),
    ("右侧45度平视", 0.79, 0.0),
    # 多样化自然语言
    ("正脸看我", 0.0, 0.0),
    ("从正面看我", 0.0, 0.0),
    ("抬头往上看", 0.0, 0.79),
    ("45度角向左看", -0.79, 0.0),
    ("往右边转45度", 0.79, 0.0),
    ("右侧30度俯视", 0.52, -0.35),
    ("左侧45度仰视", -0.79, 0.79),
    ("俯拍角度", 0.0, -0.52),
    ("抬头45度向左看", -0.79, 0.79),
    ("稍微向右转20度", 0.35, 0.0),
]

print(f"测试集: {len(diverse_texts)} 组文本 × 10个身份 = {len(diverse_texts)*10} 个样本")
print(f"{'':4s}{'文本':24s}{'目标yaw':>8s}{'预测yaw':>8s}{'误差':>6s}{'目标pit':>8s}{'预测pit':>8s}{'误差':>6s}")
print("-" * 80)

total_yaw_err = 0
total_pit_err = 0
count = 0

for zid in [0, 1, 2, 3, 5, 10, 15, 20, 25, 30]:
    w = identity_w[zid]
    for text, target_yaw, target_pitch in diverse_texts:
        with torch.no_grad():
            out = b(text, w_plus=w, return_raw_6d=True)
        raw_6d = out[2]
        pred_yaw, pred_pitch = raw_6d[0,0].item(), raw_6d[0,1].item()
        
        yaw_err = abs(pred_yaw - target_yaw)
        pit_err = abs(pred_pitch - target_pitch)
        
        total_yaw_err += yaw_err
        total_pit_err += pit_err
        count += 1
        
        if count <= 20:  # 只打印前20行
            print(f"z{zid:2d} {text:24s} {target_yaw:+.2f}  {pred_yaw:+.2f}  {yaw_err:.2f}  {target_pitch:+.2f}  {pred_pitch:+.2f}  {pit_err:.2f}")
        
        # 保存第一组文本的图像示例
        if zid == 5 and count <= 15:
            torchvision.utils.save_image((out[0][0]*0.5+0.5).clamp(0,1),
                                          f"/root/output_diverse_test/z{zid}_{text}.png")

print("-" * 80)
avg_yaw_err = total_yaw_err / count
avg_pit_err = total_pit_err / count
print(f"平均: yaw误差={avg_yaw_err:.4f}rad ({avg_yaw_err*180/math.pi:.2f}°)  pitch={avg_pit_err:.4f}rad ({avg_pit_err*180/math.pi:.2f}°)")
print(f"总样本: {count}")
print(f"结果见 /root/output_diverse_test/")
