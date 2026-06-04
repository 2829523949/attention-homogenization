"""
Phase 1 角度精度测试: 固定模板 vs 多样化文本
==============================================
关键设定:
  - 21个固定模板 (7 yaw × 3 pitch), yaw ∈ ±45°, 文本-角度硬绑定
  - 多样化文本从 dataset.pt 取, 用其预配对的 target_raw_6d 作为真值
  - 全部 yaw ≤ ±45°, 不测大角度 (EG3D 渲染质量差)
  - forward(text): 全部文本统一走 PoseMappingLayer (已移除 Adapter)

使用方法:
  python test_angle_accuracy.py
"""
import sys, math, random
import torch
import numpy as np

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden


# ============================================================
# 21个固定模板 (7 yaw × 3 pitch, yaw≤±45°)
# (yaw, pitch_offset, text)
# yaw: 负=左侧, 正=右侧, 0=正面
# pitch_offset: 负=俯视, 0=平视, 正=仰视
# ============================================================
TEMPLATES = [
    # 正面
    ( 0.00, -0.35, "正面俯视"),      ( 0.00,  0.00, "正面平视"),      ( 0.00,  0.35, "正面仰视"),
    # 左侧15度
    (-0.26, -0.35, "左侧15度俯视"),  (-0.26,  0.00, "左侧15度平视"),  (-0.26,  0.35, "左侧15度仰视"),
    # 左侧30度
    (-0.52, -0.35, "左侧30度俯视"),  (-0.52,  0.00, "左侧30度平视"),  (-0.52,  0.35, "左侧30度仰视"),
    # 左侧45度
    (-0.79, -0.35, "左侧45度俯视"),  (-0.79,  0.00, "左侧45度平视"),  (-0.79,  0.35, "左侧45度仰视"),
    # 右侧15度
    ( 0.26, -0.35, "右侧15度俯视"),  ( 0.26,  0.00, "右侧15度平视"),  ( 0.26,  0.35, "右侧15度仰视"),
    # 右侧30度
    ( 0.52, -0.35, "右侧30度俯视"),  ( 0.52,  0.00, "右侧30度平视"),  ( 0.52,  0.35, "右侧30度仰视"),
    # 右侧45度
    ( 0.79, -0.35, "右侧45度俯视"),  ( 0.79,  0.00, "右侧45度平视"),  ( 0.79,  0.35, "右侧45度仰视"),
]
TEMPLATE_TEXTS = {t[2] for t in TEMPLATES}  # 用于数据集筛选

# 角度名映射 (打印用)
YAW_NAMES = {
    -0.79: "左45°", -0.52: "左30°", -0.26: "左15°",
     0.00: "正面",
     0.26: "右15°",  0.52: "右30°",  0.79: "右45°",
}
PITCH_NAMES = {-0.35: "俯视", 0.00: "平视", 0.35: "仰视"}


def make_gt_6d(yaw, pitch_offset, radius=2.7, fov=18.837):
    """生成与模板对应的 6D 真值"""
    pitch = math.pi / 2 + pitch_offset
    return torch.tensor([[yaw, pitch, radius, fov, 0.0, 0.0]], dtype=torch.float32)


# ============================================================
# 测试函数
# ============================================================
def test_templates(bridge):
    """测试 21 个固定模板文本"""
    print("\n" + "=" * 120)
    print("【固定模板测试】7 yaw × 3 pitch = 21 个文本")
    print("=" * 120)
    print(f"{'文本':>30s} {'角度':>10s} {'预测yaw':>9s} {'目标yaw':>9s} {'Δyaw(rad)':>10s} {'Δyaw(°)':>8s} |"
          f"{'预测pit':>9s} {'目标pit':>9s} {'Δpit(rad)':>10s} {'Δpit(°)':>8s}")
    print("-" * 120)

    yaw_errs, pit_errs = [], []
    for yaw, pitch_off, text in TEMPLATES:
        gt_6d = make_gt_6d(yaw, pitch_off)
        yaw_gt = gt_6d[0, 0].item()
        pit_gt = gt_6d[0, 1].item()

        # 模板文本用 forward (直接 Pose, 无 Adapter)
        with torch.no_grad():
            out = bridge(text, return_raw_6d=True)
        pred = out[2][0]

        ye = pred[0].item() - yaw_gt
        pe = pred[1].item() - pit_gt
        yaw_errs.append(abs(ye))
        pit_errs.append(abs(pe))

        angle_label = f"{YAW_NAMES.get(yaw, f'{yaw:.2f}')} {PITCH_NAMES.get(pitch_off, f'{pitch_off:.2f}')}"
        print(f"{text:>30s} {angle_label:>10s} {pred[0]:+9.3f} {yaw_gt:+9.3f} {ye:+10.4f} {abs(ye)*180/math.pi:>7.2f}° |"
              f"{pred[1]:+9.3f} {pit_gt:+9.3f} {pe:+10.4f} {abs(pe)*180/math.pi:>7.2f}°")

    # 统计
    print("-" * 120)
    avg_y = np.mean(yaw_errs)
    avg_p = np.mean(pit_errs)
    med_y = np.median(yaw_errs)
    med_p = np.median(pit_errs)
    max_y = max(yaw_errs)
    max_p = max(pit_errs)
    print(f"模板汇总:  yaw平均={avg_y*180/math.pi:.2f}° | yaw中位数={med_y*180/math.pi:.2f}° | yaw最大={max_y*180/math.pi:.2f}°")
    print(f"          pitch平均={avg_p*180/math.pi:.2f}° | pitch中位数={med_p*180/math.pi:.2f}° | pitch最大={max_p*180/math.pi:.2f}°")

    return yaw_errs, pit_errs


def test_diverse_texts(bridge, dataset_path="/root/edit_data/dataset.pt", max_samples=200):
    """测试多样化文本 (从 dataset.pt 取, 使用预配对的 target_raw_6d)"""
    print("\n" + "=" * 120)
    print("【多样化文本测试】从 dataset.pt 取非模板文本, 使用预配对真值")
    print("=" * 120)

    data = torch.load(dataset_path)
    all_pairs = data.get('train_pairs', []) + data.get('val_pairs', [])
    print(f"数据集总量: {len(all_pairs)} 条")

    # 筛选: 1) 非模板文本  2) yaw ≤ ±45° (≈±0.79 rad)
    diverse_pairs = []
    for item in all_pairs:
        text = item["text"]
        if text in TEMPLATE_TEXTS:
            continue
        target = torch.tensor(item["target_raw_6d"], dtype=torch.float32)
        yaw = target[0].item()
        if abs(yaw) > 0.79:
            continue
        diverse_pairs.append((text, target))

    # 按目标角度去重取样 (每角度最多取 3 条, 避免某角度过多)
    from collections import defaultdict
    by_angle = defaultdict(list)
    for text, target in diverse_pairs:
        key = (round(target[0].item(), 2), round(target[1].item(), 2))
        by_angle[key].append((text, target))

    sampled = []
    for key, items in by_angle.items():
        sampled.extend(items[:3])  # 每角度最多3条

    # 如果还是太多, 随机采样到 max_samples
    if len(sampled) > max_samples:
        random.shuffle(sampled)
        sampled = sampled[:max_samples]

    print(f"多样化测试样本: {len(sampled)} 条 (来自 {len(by_angle)} 个目标角度)")

    if not sampled:
        print("⚠️ 没有找到多样化文本, 跳过!")
        return [], []

    # 打印前10条预览
    print("\n前10条样本预览:")
    for i, (text, target) in enumerate(sampled[:10]):
        yaw, pitch = target[0].item(), target[1].item()
        print(f"  [{i+1}] '{text[:40]:40s}' → yaw={yaw:+.2f} pitch={pitch:+.2f} ({pitch-math.pi/2:+.2f})")

    # 测试
    print(f"\n{'文本':>45s} {'预测yaw':>9s} {'目标yaw':>9s} {'Δyaw(rad)':>10s} {'Δyaw(°)':>8s} |"
          f"{'预测pit':>9s} {'目标pit':>9s} {'Δpit(rad)':>10s} {'Δpit(°)':>8s}")
    print("-" * 120)

    yaw_errs, pit_errs = [], []
    for text, target in sampled:
        target = target.unsqueeze(0).cuda()
        yaw_gt = target[0, 0].item()
        pit_gt = target[0, 1].item()

        with torch.no_grad():
            hidden = get_llm_hidden(bridge.qwen, bridge.processor, text)
            pred_6d = bridge.pose(hidden.unsqueeze(0))

        ye = pred_6d[0, 0].item() - yaw_gt
        pe = pred_6d[0, 1].item() - pit_gt
        yaw_errs.append(abs(ye))
        pit_errs.append(abs(pe))

        print(f"{text[:45]:>45s} {pred_6d[0,0]:+9.3f} {yaw_gt:+9.3f} {ye:+10.4f} {abs(ye)*180/math.pi:>7.2f}° |"
              f"{pred_6d[0,1]:+9.3f} {pit_gt:+9.3f} {pe:+10.4f} {abs(pe)*180/math.pi:>7.2f}°")

    # 统计
    print("-" * 120)
    avg_y = np.mean(yaw_errs) * 180 / math.pi
    avg_p = np.mean(pit_errs) * 180 / math.pi
    med_y = np.median(yaw_errs) * 180 / math.pi
    med_p = np.median(pit_errs) * 180 / math.pi
    max_y = max(yaw_errs) * 180 / math.pi
    max_p = max(pit_errs) * 180 / math.pi
    print(f"多样化汇总: yaw平均={avg_y:.2f}° | yaw中位数={med_y:.2f}° | yaw最大={max_y:.2f}°")
    print(f"            pitch平均={avg_p:.2f}° | pitch中位数={med_p:.2f}° | pitch最大={max_p:.2f}°")

    return yaw_errs, pit_errs


# ============================================================
# 生成多样化文本 → 角度测试 (不依赖数据集, 使用 generate_diverse_text)
# ============================================================
def test_generated_diverse(bridge, num_samples=50):
    """
    随机生成多样化文本, 构造与之对应的 GT 角度
    验证: 即使非常用措辞, pose 预测是否仍能正确映射
    """
    print("\n" + "=" * 120)
    print(f"【生成多样化文本测试】随机生成 {num_samples} 个文本, 构造配对真值")
    print("=" * 120)

    # 从 train_mixed_v2.py 复制生成函数 (避免 import 依赖)
    def _gen_text(yaw, pitch_offset):
        templates = []
        abs_yaw = abs(yaw)
        if abs_yaw < 0.08:
            dir_words = ["正面", "正脸", "正对镜头", "看向前方"]
        elif abs_yaw < 0.35:
            side = "左侧" if yaw > 0 else "右侧"
            dir_words = [f"{side}15度", f"{side}微侧", f"{side}偏一点",
                         f"{side}稍微侧", f"稍向{side}转", f"微微{side}偏"]
        elif abs_yaw < 0.65:
            side = "左侧" if yaw > 0 else "右侧"
            dir_words = [f"{side}30度", f"{side}三分之一个侧面",
                         f"{side}中幅侧转", f"{side}偏转30度"]
        else:
            side = "左侧" if yaw > 0 else "右侧"
            dir_words = [f"{side}45度", f"{side}半侧面", f"{side}侧向",
                         f"{side}大幅侧转"]
        if pitch_offset < -0.3:
            pit_words = ["俯视", "从上往下看", "高角度", "俯瞰"]
        elif pitch_offset > 0.3:
            pit_words = ["仰视", "从下往上看", "低角度", "仰望"]
        else:
            pit_words = ["平视", "水平视角", "正常高度"]
        dir_w = random.choice(dir_words)
        pit_w = random.choice(pit_words)
        if random.random() > 0.5:
            templates.append(f"{dir_w}{pit_w}")
        if random.random() > 0.5:
            templates.append(f"{random.choice(['一张', ''])}{dir_w}{random.choice(['的', '角度', '视角'])}{pit_w}{random.choice(['照片', '人脸', '拍摄'])}")
        if random.random() > 0.7:
            templates.append(f"{random.choice(['从', '在'])}{dir_w}{random.choice(['方向拍的', '角度看的', '视角下的'])}{pit_w}")
        return random.choice(templates).strip()

    samples = []
    for _ in range(num_samples):
        yaw = random.uniform(-0.79, 0.79)
        pitch_offset = random.uniform(-0.6, 0.6)
        text = _gen_text(yaw, pitch_offset)
        # 确保不重复模板文本
        while text in TEMPLATE_TEXTS:
            text = _gen_text(yaw, pitch_offset)
        pitch = math.pi / 2 + pitch_offset
        gt = torch.tensor([[yaw, pitch, 2.7, 18.84, 0.0, 0.0]], dtype=torch.float32)
        samples.append((text, gt))

    print(f"\n{'文本':>45s} {'预测yaw':>9s} {'目标yaw':>9s} {'Δyaw(rad)':>10s} {'Δyaw(°)':>8s} |"
          f"{'预测pit':>9s} {'目标pit':>9s} {'Δpit(rad)':>10s} {'Δpit(°)':>8s}")
    print("-" * 120)

    yaw_errs, pit_errs = [], []
    for text, gt in samples:
        gt = gt.cuda()
        with torch.no_grad():
            hidden = get_llm_hidden(bridge.qwen, bridge.processor, text)
            pred_6d = bridge.pose(hidden.unsqueeze(0))

        ye = pred_6d[0, 0].item() - gt[0, 0].item()
        pe = pred_6d[0, 1].item() - gt[0, 1].item()
        yaw_errs.append(abs(ye))
        pit_errs.append(abs(pe))

        print(f"{text[:45]:>45s} {pred_6d[0,0]:+9.3f} {gt[0,0]:+9.3f} {ye:+10.4f} {abs(ye)*180/math.pi:>7.2f}° |"
              f"{pred_6d[0,1]:+9.3f} {gt[0,1]:+9.3f} {pe:+10.4f} {abs(pe)*180/math.pi:>7.2f}°")

        ye = pred_6d[0, 0].item() - gt[0, 0].item()
        pe = pred_6d[0, 1].item() - gt[0, 1].item()
        yaw_errs.append(abs(ye))
        pit_errs.append(abs(pe))

        print(f"{text[:45]:>45s} {pred_6d[0,0]:+9.3f} {gt[0,0]:+9.3f} {ye:+10.4f} {abs(ye)*180/math.pi:>7.2f}° |"
              f"{pred_6d[0,1]:+9.3f} {gt[0,1]:+9.3f} {pe:+10.4f} {abs(pe)*180/math.pi:>7.2f}°")

    print("-" * 120)
    avg_y = np.mean(yaw_errs) * 180 / math.pi
    avg_p = np.mean(pit_errs) * 180 / math.pi
    med_y = np.median(yaw_errs) * 180 / math.pi
    med_p = np.median(pit_errs) * 180 / math.pi
    max_y = max(yaw_errs) * 180 / math.pi
    max_p = max(pit_errs) * 180 / math.pi
    print(f"生成多样化汇总: yaw平均={avg_y:.2f}° | yaw中位数={med_y:.2f}° | yaw最大={max_y:.2f}°")
    print(f"                pitch平均={avg_p:.2f}° | pitch中位数={med_p:.2f}° | pitch最大={max_p:.2f}°")

    return yaw_errs, pit_errs


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 120)
    print("  Qwen3-VL → 6D Pose → EG3D  Phase 1 角度精度测试")
    print("  固定模板(21) + 多样化文本(数据集) + 生成多样化文本(随机)")
    print("  所有 yaw ≤ ±45° (弧度 ±0.79)")
    print("=" * 120)

    # 加载 Bridge (自动加载 pose 权重)
    bridge = QwenEG3DBridge()
    bridge.eval()

    # ===== 测试1: 固定模板 =====
    t_y_errs, t_p_errs = test_templates(bridge)

    # ===== 测试2: 多样化文本 (来自数据集) =====
    d1_y_errs, d1_p_errs = test_diverse_texts(bridge, max_samples=200)

    # ===== 测试3: 随机生成多样化文本 =====
    d2_y_errs, d2_p_errs = test_generated_diverse(bridge, num_samples=50)

    # ===== 最终汇总 =====
    print("\n" + "=" * 120)
    print("最终汇总")
    print("=" * 120)
    print(f"{'类别':>20s} {'样本数':>6s} {'yaw平均':>10s} {'yaw中位':>10s} {'yaw最大':>10s} |"
          f"{'pitch平均':>10s} {'pitch中位':>10s} {'pitch最大':>10s}")
    print("-" * 120)

    def deg(v): return v * 180 / math.pi

    for name, y_errs, p_errs in [
        ("固定模板", t_y_errs, t_p_errs),
        ("多样化(数据集)", d1_y_errs, d1_p_errs),
        ("多样化(生成)", d2_y_errs, d2_p_errs),
    ]:
        if y_errs:
            print(f"{name:>20s} {len(y_errs):>6d} "
                  f"{deg(np.mean(y_errs)):>9.2f}° {deg(np.median(y_errs)):>9.2f}° {deg(max(y_errs)):>9.2f}° |"
                  f"{deg(np.mean(p_errs)):>9.2f}° {deg(np.median(p_errs)):>9.2f}° {deg(max(p_errs)):>9.2f}°")

    print("=" * 120)
    print("测试完成!")
