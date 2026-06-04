# Attention Homogenization Project — Progress Summary
## Date: June 4, 2026

## Paper Status: Full update cycle completed

### Experiments Re-run & Fixed

#### 1. Face Attention (§4.1)
- Primary: AFLW2000 N=200 (spec L25-27 r=0.949±0.035)
- Validation: EG3D N=9 for LLaVA cross-model
- U-shape confirmed: L1-3=0.962→L8-10=0.849→L18-20=0.584→L25-27=0.949

#### 2. Dot Test (§4.2) — **NEW per-layer results**
- Per-layer cross_r U-shape: L1-3=0.819→L8-10=0.688→L18-20=0.374→L25-27=0.730
- Peak positions at L18-20: red(1,0), green(4,3), blue(6,6)
- Multi-contrast configs stable (range 0.36-0.57)

#### 3. Mid-layer Readout Probe (Table V) — **Multi-seed (5 seeds)**
- L18-20: 63.0±2.3 / 76.7±1.2 / 85.4±0.4 / 90.1±1.0
- Last: 48.4±2.5 / 69.6±2.3 / 78.9±1.1 / 86.0±1.3
- Gap at 10%=14.6pp → at 100%=4.1pp (statistically significant)

#### 4. Geometry Probe (Table VII)
- No normalization (fair comparison across probes)
- ConvNet L18-20 68.4% >> PoolMLP 29.6% ≈ PoolLinear 29.6% ≈ baseline 28.6%
- Z-score normalization caveat added (would inflate ConvNet to 73.1% but breaks linear probes)

#### 5. Gate Ablation (Table VIII) — Unchanged, stable
- g=0.730±0.001 across 3 seeds

#### 6. Causal Chain Fix
- §5.1: "Why Homogenization Occurs" → "The Co-occurrence of Homogenization and Spatial Degradation"
- New limitation: correlation vs causation unestablished

#### 7. Citation Fixes
- Removed: Park2025bridge (fabricated arXiv:2511.11526)
- Removed: GRIT-LP, eigen-query decoder, STNet (uncited)
- Fixed author names in bib: Yichen→Yifan Shen, Zhi→Zhou Tao, Yun→Yanchun Cheng, Kunyu→Kelly Cui

### Remaining Issues
- LLaVA only at 4-bit (8GB GPU insufficient for bf16 7B)
- Qwen2.5-VL-3B probe results not re-run (old numbers remain)
- Table labels still use wrong \ref references (tab:gate for geometry table)
- Limitations mention multi-seed missing from single-run experiments (now fixed)

### Key Experiment Scripts
- /root/Qwen3-VL/aflw2000_u_shape.py — AFLW2000 face attention
- /root/Qwen3-VL/dot_layer_analysis.py — Per-layer dot test
- /root/Qwen3-VL/midlayer_landmark_probe.py — Single-run mid-layer probe
- /root/Qwen3-VL/multiseed_midlayer.py — 5-seed mid-layer probe
- /root/Qwen3-VL/geometry_probe_exp.py — Geometry + probe decomposability
- /root/Qwen3-VL/pool_probe_test.py — Pool-only 500 epoch test
- /root/Qwen3-VL/paper/main.tex — Paper source

### Next: Uniface Project
- Unified 3D face generation/editing using EG3D/Next3D
