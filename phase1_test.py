import sys, math, json, random, os
import torch, torchvision
import numpy as np

sys.path.insert(0, '/root/Qwen3-VL')
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

# ===== 21个模板 =====
TEMPLATES = [
    ( 0.00, -0.35, '正面俯视'),    ( 0.00,  0.0, '正面平视'),    ( 0.00,  0.35, '正面仰视'),
    (-0.26, -0.35, '左侧15度俯视'), (-0.26,  0.0, '左侧15度平视'), (-0.26,  0.35, '左侧15度仰视'),
    (-0.52, -0.35, '左侧30度俯视'), (-0.52,  0.0, '左侧30度平视'), (-0.52,  0.35, '左侧30度仰视'),
    (-0.79, -0.35, '左侧45度俯视'), (-0.79,  0.0, '左侧45度平视'), (-0.79,  0.35, '左侧45度仰视'),
    ( 0.26, -0.35, '右侧15度俯视'), ( 0.26,  0.0, '右侧15度平视'), ( 0.26,  0.35, '右侧15度仰视'),
    ( 0.52, -0.35, '右侧30度俯视'), ( 0.52,  0.0, '右侧30度平视'), ( 0.52,  0.35, '右侧30度仰视'),
    ( 0.79, -0.35, '右侧45度俯视'), ( 0.79,  0.0, '右侧45度平视'), ( 0.79,  0.35, '右侧45度仰视'),
]

print('='*60)
print('Phase 1 验证: 21模板精度')
print('='*60)
print(f'{"文本":>16s}  {"目标yaw":>8s}  {"预测yaw":>8s}  {"Δyaw°":>8s}  {"Δpitch°":>8s}')
print('-'*60)

all_ye, all_pe = [], []
for yaw_gt, po_gt, text in TEMPLATES:
    _, _, raw = b(text, return_raw_6d=True)
    py = raw[0,0].item()
    pp = raw[0,1].item()
    gt_pitch = math.pi/2 + po_gt
    ye = min(abs(py-yaw_gt), 2*math.pi-abs(py-yaw_gt)) * 180/math.pi
    pe = abs(pp-gt_pitch) * 180/math.pi
    all_ye.append(ye); all_pe.append(pe)
    print(f'{text:>16s}  {yaw_gt:+8.3f}  {py:+8.3f}  {ye:7.2f}°  {pe:7.2f}°')

print('-'*60)
print(f'平均 yaw 误差:  {np.mean(all_ye):.2f}°  (中位数: {np.median(all_ye):.2f}°)')
print(f'平均 pitch 误差: {np.mean(all_pe):.2f}°  (中位数: {np.median(all_pe):.2f}°)')
print(f'yaw 最佳: {min(all_ye):.2f}°  yaw 最差: {max(all_ye):.2f}°')
print()

# ===== 多样化文本 =====
print('='*60)
print('多样化文本测试 (text_variants.json 随机20条)')
print('='*60)
with open('/root/Qwen3-VL/text_variants.json') as f:
    variants = json.load(f)
random.shuffle(variants)
tests = [v for v in variants if abs(v['yaw'])<=0.79][:20]
dv_ye = []
for v in tests:
    _, _, raw = b(v['text'], return_raw_6d=True)
    py = raw[0,0].item()
    ye = min(abs(py-v['yaw']), 2*math.pi-abs(py-v['yaw'])) * 180/math.pi
    dv_ye.append(ye)
print(f'多样化文本平均 yaw 误差: {np.mean(dv_ye):.2f}°')
print()

# ===== 渲染 =====
print('='*60)
print('EG3D 渲染: 9视图保存到 /root/output_phase1_test/')
print('='*60)
save_dir = '/root/output_phase1_test'
os.makedirs(save_dir, exist_ok=True)
views = ['正面平视', '正面俯视', '正面仰视',
         '左侧15度平视', '左侧30度平视', '左侧45度平视',
         '右侧15度平视', '右侧30度平视', '右侧45度平视']
all_imgs = []
for text in views:
    img, cam, raw = b(text, return_raw_6d=True)
    path = f'{save_dir}/{text}.png'
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), path)
    print(f'  {text}: yaw={raw[0,0].item():+.2f} → {path}')
    all_imgs.append(img.cpu())
grid = torchvision.utils.make_grid([im[0]*0.5+0.5 for im in all_imgs], nrow=3, padding=4, pad_value=1)
torchvision.utils.save_image(grid.clamp(0,1), f'{save_dir}/grid_9views.png')
print(f'  网格图: {save_dir}/grid_9views.png')
print()

# ===== 大角度 =====
print('='*60)
print('大角度测试 (超训练范围 ±45°)')
print('='*60)
for text, gt_yaw in [('左侧60度平视',-1.05), ('左侧90度平视',-1.57),
                      ('右侧60度平视',1.05), ('右侧90度平视',1.57), ('背面',3.14)]:
    _, _, raw = b(text, return_raw_6d=True)
    py = raw[0,0].item()
    ye = min(abs(py-gt_yaw), 2*math.pi-abs(py-gt_yaw)) * 180/math.pi
    print(f'{text:>16s}  预测yaw={py:+.3f}  Δ={ye:.2f}°')

print('\n' + '='*60)
print('Phase 1 测试完成!')
print('='*60)
