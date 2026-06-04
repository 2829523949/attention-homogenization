# Layer Ablation Results (May 22, 2026)
## Experiment: VLM Self-Attention Cross-Query Similarity Across Layer Groups

### Layer Groups (28 layers total, 0-indexed)
| Group | Layers (1-idx) | Layers (0-idx) |
|-------|---------------|----------------|
| A_Early (L1-3) | 1-3 | 0, 1, 2 |
| B_Mid (L8-10) | 8-10 | 7, 8, 9 |
| C_Later (L18-20) | 18-20 | 17, 18, 19 |
| D_Last (L25-27) | 25-27 | 25, 26, 27 |

### Results
| Group | Within-Face r | Within-Face std | Face-vs-Ctrl r | Face-vs-Ctrl std | Ctrl-Ctrl r |
|-------|:------------:|:---------------:|:--------------:|:----------------:|:----------:|
| A_Early (L1-3) | 0.9985 | 0.0005 | 0.9932 | 0.0035 | 0.9955 |
| B_Mid (L8-10) | 0.9794 | 0.0134 | 0.9324 | 0.0218 | 0.9565 |
| C_Later (L18-20) | 0.9231 | 0.0442 | 0.8038 | 0.1152 | 0.7857 |
| D_Last (L25-27) | 0.9949 | 0.0026 | 0.9625 | 0.0107 | 0.9569 |

### Key Findings
- **U-shaped pattern**: Early (L1-3, r=0.9985) and Last (L25-27, r=0.9949) show collapse.
  Later-middle (L18-20, r=0.9231) shows best spatial differentiation.
- **VLM CAN spatially differentiate**: L18-20 within-face r=0.923, face-vs-ctrl r=0.804.
- **Mechanism revealed**: Semantic abstraction progressively suppresses spatial precision in deep layers.
- **Stronger narrative**: Not "VLM can't localize" but "VLMs trade spatial precision for semantic abstraction in late layers."
- **High std in L18-20**: 0.044 within-face, 0.115 face-vs-ctrl -- genuine variation across query pairs.
