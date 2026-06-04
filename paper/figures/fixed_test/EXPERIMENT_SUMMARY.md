# 全部实验结果总结 (May 22, 2026)

## 方法说明
| 方法 | 说明 | 状态 |
|:----|:-----|:----|
| **spec** ✅ | 只取特征词对应 token | **正确方法** |
| last | 最后一个 query token | ≈ spec |
| **avg** 🚫 | 所有 query token 平均 | **已废弃** |

## 1. 人脸特征 Attention (Qwen3-VL-2B, L18-20)
spec within-face=**0.478**, face-ctrl=**0.251** | avg: 0.923, 0.804
不同 face query 有不同 peak：nose(7,7), eyes(15,9), mouth(11,8), chin(11,9), forehead(4,7), beard(2,2)

## 2. 色点测试 (Qwen3-VL-2B, L18-20, 8x8)
spec: red(0,1) green(3,4) blue(7,6) ✅ | avg 全部 (0,7) ❌
Cross-r: spec=0.50, avg=0.78

## 3. 色点测试 (LLaVA-NeXT-7B, L18-20, 24x24)
spec: red(2,1) green(11,11) blue(20,21) ✅ | avg 全部 (2,9) ❌
Cross-r: spec=0.56, avg=0.88

## 4. 人脸 Attention (LLaVA-NeXT-7B, L18-20)
spec within-face≈0.80, face-ctrl≈0.71 — 全部 peak (19,6)，无法区分

## 5. 层分析 U-shaped (avg method)
Qwen: L1-3 0.999→L8-10 0.979→L18-20 **0.923**→L25-27 0.995
LLaVA: L1-3 1.000→L8-10 1.000→L18-20 **0.948**→L29-31 0.995
最后层 collapse r≈0.995 跨模型一致

## 6. 原始结果 (不受影响)
Embedding: VLM 0.0028 vs CLIP 0.0459 (16.4x)
Camera: CLIP 3.58deg vs VLM 3.84deg
Speed: CLIP 7.0ms vs VLM 75.7ms (10.8x)
Language: CN 10.10deg vs EN 3.84deg

## 关键结论
1. spec 方法正确，avg 方法稀释信号
2. Qwen3-VL 可部分区分面部特征 (r=0.478)
3. LLaVA-NeXT 无法区分面部特征 (r=0.80)
4. 两模型都能定位色点 (spec)
5. 最后层 collapse 跨模型一致 (r=0.995)
6. U-shaped 层模式跨模型一致
7. Embedding + Camera 结论不受影响
