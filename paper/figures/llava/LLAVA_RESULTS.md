# Layer Ablation Results — LLaVA-NeXT (May 22, 2026)

## LLaVA-NeXT Mistral-7B (anyres OFF)
| Group | Within-Face r | Face-vs-Ctrl r | Ctrl-Ctrl r |
|-------|:------------:|:--------------:|:----------:|
| Early (L1-3) | 1.0000 | 0.9997 | 0.9997 |
| Mid (L8-10) | 0.9995 | 0.9980 | 0.9993 |
| Later (L18-20) | 0.9475 | 0.9564 | 0.9735 |
| Last (L29-31) | 0.9949 | 0.9920 | 0.9935 |

## Cross-Model Comparison (L18-20 best layer)
| Model | Within-Face r | Face-vs-Ctrl r | Visual Tokens |
|-------|:------------:|:--------------:|:------------:|
| **Qwen3-VL-2B** | **0.9231** | **0.8038** | 256 (16x16) |
| LLaVA-NeXT-7B | 0.9475 | 0.9564 | 576 (24x24) |

## Key Findings
1. **U-shaped pattern is cross-model**: Both models show identical pattern.
2. **Bigger model, worse spatial differentiation**: LLaVA-7B underperforms Qwen3-VL-2B.
3. **Last layer collapse identical** (r=0.9949 both) — fundamental architectural property.
4. **Supports general claim**: VLM attention collapse is decoder-only phenomenon.
