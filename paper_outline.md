# Paper Outline: "Revisiting Text Encoders for 3D Face Camera Control"

## Target: WACV 2027 (deadline July 15, 2026)

---

## Title Options

1. "Revisiting Text Encoders for 3D Face Camera Control"
2. "VLM Overkill: Lightweight Encoders Outperform Large Vision-Language Models for 3D Face Pose Control"
3. "Are Large Vision-Language Models Necessary for 3D Face Control?"

---

## Structure (8 pages + refs)

### 1. Introduction (~1 page)
- **Problem**: Text-driven 3D face generation needs a text encoder. CLIP? BERT? VLM?
- **Gap**: Nobody has systematically compared them.
- **This paper**: Compare VLM (Qwen3-VL 2B) vs CLIP (63M) for EG3D camera pose control.
- **Findings**: 
  - Cross-attention cannot localize facial features (r > 0.9999)
  - LLM hidden states cannot distinguish fine-grained angles (discriminability 0.0028 vs 0.0459)
  - CLIP outperforms VLM in accuracy (yaw 3.58 vs 3.84), speed (10.8x), and size (18x)

### 2. Related Work (~1 page)
### 3. Method (~1 page)
### 4. Experiments (~3 pages)
- 4.1 Cross-Attention Spatial Grounding Failure
- 4.2 Embedding Sensitivity Analysis
- 4.3 Pose Control Accuracy
- 4.4 Inference Speed & Model Size
- 4.5 Language Mismatch
### 5. Discussion (~0.5 page)
### 6. Conclusion (~0.5 page)

---

## Figures

| Figure | Content | Status |
|:-------|:--------|:------:|
| Fig 1 | Method pipeline | Need to draw |
| Fig 2 | Attention heatmap grid | Already have |
| Fig 3 | Similarity matrix | Already have |
| Fig 4 | Embedding sensitivity | Already have |
| Fig 5 | Speed benchmark (3-panel) | Already have |
| Table 1 | Accuracy comparison | Data ready |
| Table 2 | Speed/size summary | Data ready |
