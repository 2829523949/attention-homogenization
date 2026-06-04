# 论文核心分析点 (修正后 v2, 2026-05-22)

## 重要修正: spec 方法≠空间定位
spec 方法产生了不同的 attention 模式 (within-face r=0.478)，
但这些模式 NOT reliably correspond to facial anatomy:
- eyes peak at bottom (15,9) instead of upper ✓ region
- beard peak at top-left (2,2) instead of bottom
- eyes varies wildly across images: (15,9) -> (6,15) -> (6,8)
Conclusion: spec differentiation =/= accurate spatial localization.

## 1. 最后层 Attention Collapse — 跨模型成立
Qwen3-VL-2B 和 LLaVA-NeXT-7B 的最后层 within-face r≈0.995。
这是 decoder-only VLM 架构层面的根本性限制。

## 2. U-shaped 层模式
Qwen: L1-3 0.999 -> L8-10 0.979 -> L18-20 0.923 -> L25-27 0.995
LLaVA: L1-3 1.000 -> L8-10 1.000 -> L18-20 0.948 -> L29-31 0.995
中间层有更好的空间区分，最后层 collapse。语义抽象 vs 空间精度 trade-off。

## 3. 色点测试: 能定位鲜艳色块
Qwen: red(0,1) green(3,4) blue(7,6). LLaVA: red(2,1) green(11,11) blue(20,21).
Peaks match expected positions across both models.
Cross-r=0.50-0.56 表明 attention 仍共享大量成分。

## 4. Embedding Collapse — 不受方法论影响
VLM 可区分度 0.0028 vs CLIP 0.0459 (16.4x)。
隐空间将不同角度压缩到几乎相同的表示。

## 5. 下游任务: CLIP 全面优于 VLM
Yaw: CLIP 3.58 deg vs VLM 3.84 deg. Speed: 7.0ms vs 75.7ms (10.8x).
Parameter: 67M vs 1224M (18.3x).

## 6. 跨模型对比: 更大模型 != 更好定位
LLaVA-7B face-ctrl r=0.71 vs Qwen-2B 0.25 (worse at face differentiation).
参数数量不改善 spatial attention 能力。

## 修正了的叙事框架
1. 最后层 attention collapse (r=0.995) 跨模型一致
2. U-shaped 层模式: 中间层较好，语义抽象压低了空间精度
3. spec vs avg 方法: avg 方法稀释信号，spec 方法揭示更多结构
4. spec 方法显示的"区分"不等于解剖正确的空间定位
5. 色点测试: 鲜艳特征能被定位但精度有限
6. Embedding collapse 是独立且更严重的发现
7. CLIP 在下游任务中全面优于 VLM
