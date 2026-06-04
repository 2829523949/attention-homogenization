# Experiment Results Summary

## 1. VLM vs CLIP Fair Comparison (English, same test set)

| Model | Params | Yaw Mean | Yaw Median | Pitch Mean | Pitch Median |
|-------|--------|----------|------------|------------|-------------|
| VLM (Qwen3-VL 2B) | 2,000M | 3.84° | 2.40° | 5.60° | 2.29° |
| CLIP (63M) | 63M | 3.58° | 2.23° | 4.60° | 2.22° |
| **Improvement** | **32x smaller** | **7% better** | **7% better** | **18% better** | **3% better** |

## 2. VLM Chinese Baseline
- yaw mean: 10.10° (median 5.36°)
- pitch mean: 4.66° (median 3.74°)
- Language mismatch significantly degrades VLM performance

## 3. Cross-Attention Spatial Grounding Failure
- All face features (nose, eyes, mouth, chin, forehead, beard) → r > 0.9999
- Even control queries ("whole image", "bottom-left") → r > 0.9999
- **Conclusion**: Qwen3-VL cross-attention CANNOT localize facial features
- Evidence saved to: `/root/Qwen3-VL/attention_evidence/`

## 4. Phase 2 Visual Fusion Failures
- LLMControlledFeatureFusion: removed from bridge.py
- DINOv2 injector: trained but insufficient data
- Adapter: abandoned, removed from training script
- Checkpoints exist: adapter_epoch_*.pt, injector_trained.pt

## 5. Key Files
- `bridge.py` - Qwen3-VL bridge (VLM)
- `lightweight_bridge.py` - CLIP/BERT/XLM bridge
- `train_mixed_v2_en.py` - Training with English dataset
- `train_lightweight_en.py` - Lightweight encoder training
- `run_attention_experiment.py` - Cross-attention experiment
- `final_compare.py` - Fair comparison VLM vs CLIP
- `check_pitch.py` - Validation script (supports --encoder vlm|clip|bert)

## 6. Checkpoints
- VLM EN: `/root/Qwen3-VL/checkpoints/vlm_en_epoch_080.pt`
- CLIP EN: `/root/Qwen3-VL/checkpoints/clip_en_epoch_080.pt`
- VLM CN: `/root/Qwen3-VL/checkpoints/mix_v2_epoch_045.pt`

## 7. Attention Evidence
- Heatmaps: `/root/Qwen3-VL/attention_evidence/heatmap_*.png`
- Similarity matrices: `/root/Qwen3-VL/attention_evidence/similarity_*.png`
- Numerical data: `/root/Qwen3-VL/attention_evidence/similarity_data.json`

## 8. Paper Direction
- Title: "Revisiting Text Encoders for 3D Face Control: VLM Overkill"
- Core claim: CLIP (63M) with 1/32 parameters outperforms VLM (2B) in 3D face pose control
- Contributions:
  1. VLM cross-attention spatial grounding failure in face localization
  2. Phase 2 visual fusion 3 attempts all failed
  3. Fair comparison proves VLM overkill hypothesis

## 9. Inference Speed Benchmark

| Metric | VLM 2B | CLIP 63M | Speedup |
|:-------|:------:|:--------:|:-------:|
| Full Pipeline (ms) | 75.71 ± 3.58 | **6.99 ± 1.02** | **10.8×** |
| Encoder Only (ms) | 75.72 ± 3.06 | **6.74 ± 1.05** | **11.2×** |
| Throughput (samples/sec) | 13.2 | **143.0** | **10.8×** |
| Peak VRAM (GB) | 2.12 | **2.09** | 1.01× |
| Parameters (M) | 1,224 | **67** | **18.3×** |

**Key takeaway**: CLIP is **10.8× faster** in inference, uses **18.3× fewer parameters**, and yet achieves **better pose accuracy** (yaw 3.58° vs 3.84°).

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Script: `benchmark_speed.py`
- Figure: `paper_figures/speed_benchmark.png`

## 10. Embedding Sensitivity Ablation

| Metric | VLM 2B | CLIP 63M | CLIP Advantage |
|:-------|:------:|:--------:|:--------------:|
| Mean pairwise similarity (non-diag) | 0.9962 | 0.9469 | More diverse |
| Adjacent (10°) similarity | 0.9976 | 0.9752 | — |
| Far (40°+) similarity | 0.9948 | 0.9293 | — |
| **Discriminability (adj-far)** | **0.0028** | **0.0459** | **16.4× better** |

**Conclusion**: VLM's LLM hidden states collapse fine-grained angle information (0° vs 90° are 99.5% similar). CLIP preserves angle differences in its embedding space, explaining its superior pose control accuracy.

- Script: `ablation_sensitivity.py`
- Figure: `paper_figures/encoding_sensitivity.png`
- Data: `checkpoints/encoding_sensitivity.json`
