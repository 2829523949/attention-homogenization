"""
Layer-wise VLM Self-Attention Analysis — Reusable Template
==========================================================
Design: Config → Run → Visualize → Report
Can be imported and reused for any decoder-only VLM.

Quick start:
    from attention_viz import VLMConfig, Qwen3VLAdapter, AttentionAnalyzer
    config = VLMConfig(model_path="/path/to/model")
    adapter = Qwen3VLAdapter(config)
    analyzer = AttentionAnalyzer(config, adapter)
    analyzer.run()
"""

import os, sys, json, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from scipy.stats import pearsonr

# ═══════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════

@dataclass
class VLMConfig:
    """Configuration for attention analysis experiments."""
    model_path: str = "/root/Qwen3-VL-2B-Instruct"
    image_dir: str = "/root/output_phase1_test"
    output_dir: str = "/root/Qwen3-VL/paper/figures"
    layer_groups: Dict[str, List[int]] = field(default_factory=lambda: {
        "Early (L1-3)":   [0, 1, 2],
        "Mid (L8-10)":    [7, 8, 9],
        "Later (L18-20)": [17, 18, 19],
        "Last (L25-27)":  [25, 26, 27],
    })
    queries: List[str] = field(default_factory=lambda: [
        "nose", "eyes", "mouth", "chin", "forehead", "beard",
        "whole image", "bottom-left area",
    ])
    num_images: int = 3
    max_images: int = 3
    torch_dtype: str = "bfloat16"
    device_map: str = "cuda"
    attn_implementation: str = "eager"
    local_files_only: bool = True
    trust_remote_code: bool = True


# ═══════════════════════════════════════════════
#  Model Adapter — override for other VLMs
# ═══════════════════════════════════════════════

class BaseVLMAdapter:
    """Abstract base. Subclass to adapt for different VLMs."""
    
    def load_model(self, config: VLMConfig):
        raise NotImplementedError
    
    def get_lm(self, model):
        """Return the decoder-only language model (with .layers)."""
        raise NotImplementedError
    
    def count_visual_tokens(self, processor, image, text: str) -> int:
        """Return number of visual tokens for a given image."""
        raise NotImplementedError
    
    def get_token_ranges(self, input_ids: torch.Tensor) -> Dict[str, int]:
        """
        Return dict with keys: 'vis_start', 'vis_end', 'q_start', 'q_end'.
        These define the token index ranges for visual region and query text region.
        """
        raise NotImplementedError
    
    def build_messages(self, image, query: str):
        """Return messages list for processor.apply_chat_template."""
        raise NotImplementedError


class Qwen3VLAdapter(BaseVLMAdapter):
    """Adapter for Qwen3-VL (and similar HuggingFace VLMs)."""
    
    # Token IDs (Qwen3-VL specific)
    T_VISION_START = 151652
    T_VISION_END = 151653
    T_IM_END = 151645
    
    def load_model(self, config: VLMConfig):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_path,
            torch_dtype=getattr(torch, config.torch_dtype),
            device_map=config.device_map,
            attn_implementation=config.attn_implementation,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            config.model_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        model.eval()
        return model, processor
    
    def get_lm(self, model):
        return model.model.language_model
    
    def count_visual_tokens(self, processor, image, text: str) -> int:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        if "image_grid_thw" in inputs:
            g = inputs["image_grid_thw"][0]
            h_patches = g[1].item() // 2
            w_patches = g[2].item() // 2
            return h_patches * w_patches
        return 256
    
    def get_token_ranges(self, input_ids: torch.Tensor) -> Dict[str, int]:
        ve_pos = (input_ids == self.T_VISION_END).nonzero(as_tuple=True)[0].item()
        ie_pos = (input_ids == self.T_IM_END).nonzero(as_tuple=True)[0].item()
        return {
            "vis_start": 4,
            "vis_end": ve_pos,
            "q_start": ve_pos + 1,
            "q_end": ie_pos,
        }
    
    def build_messages(self, image, query: str):
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"describe the {query}"},
        ]}]


# ═══════════════════════════════════════════════
#  Core Analyzer
# ═══════════════════════════════════════════════

class AttentionAnalyzer:
    """Runs layer-wise attention analysis, produces data + figures."""
    
    def __init__(self, config: VLMConfig, adapter: BaseVLMAdapter = None):
        self.config = config
        self.adapter = adapter or Qwen3VLAdapter()
        self.model = None
        self.processor = None
        self.lm = None
        self.results: Dict = {}  # results[group_key][fname][query] = heatmap_2d
        self.agg_stats: Dict = {}
    
    # ── Setup ──
    
    def setup(self):
        print(f"Loading model from {self.config.model_path}...")
        self.model, self.processor = self.adapter.load_model(self.config)
        self.lm = self.adapter.get_lm(self.model)
        print(f"Model loaded. {len(self.lm.layers)} LM layers.")
        print(f"Layer groups: {list(self.config.layer_groups.keys())}")
        print(f"Queries: {self.config.queries}")
        return self
    
    def _get_images(self):
        all_files = sorted([
            f for f in os.listdir(self.config.image_dir)
            if f.endswith((".png", ".jpg", ".jpeg")) and f != "grid_9views.png"
        ])
        return all_files[:min(self.config.num_images, self.config.max_images)]
    
    def _compute_heatmap(self, image, query: str, layer_indices: List[int]) -> np.ndarray:
        """Run one forward pass with hooks on specified layers, return attention heatmap."""
        messages = self.adapter.build_messages(image, query)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.config.device_map)
        input_ids = inputs["input_ids"][0]
        
        # Token ranges
        tr = self.adapter.get_token_ranges(input_ids)
        nvt = tr["vis_end"] - tr["vis_start"]
        nvt_dim = int(math.isqrt(nvt))
        if nvt != nvt_dim * nvt_dim:
            nvt_dim = int(np.sqrt(nvt))
        
        # Hook specified layers
        attn_weights = []
        def make_hook(weights_list):
            def hook(m, i, o):
                if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                    weights_list.append(o[1].detach())
            return hook
        
        handles = []
        for lidx in layer_indices:
            handles.append(
                self.lm.layers[lidx].self_attn.register_forward_hook(make_hook(attn_weights)))
        
        with torch.no_grad():
            out = self.model(**inputs, output_attentions=True)
        for h in handles:
            h.remove()
        
        if attn_weights:
            attn = torch.stack(attn_weights).mean(dim=0)      # avg layers
            attn = attn.mean(dim=[0, 1])                       # avg batch+heads
            vis_attn = attn[tr["q_start"]:tr["q_end"], tr["vis_start"]:tr["vis_end"]]
            vis_attn = vis_attn.mean(dim=0).cpu().float().numpy()
            return vis_attn.reshape(nvt_dim, nvt_dim)
        return np.zeros((nvt_dim, nvt_dim))
    
    # ── Run ──
    
    def run(self):
        """Full experiment: run all images × queries × layer groups."""
        self.setup()
        image_files = self._get_images()
        print(f"\nImages ({len(image_files)}): {image_files}")
        
        self.results = {gk: {} for gk in self.config.layer_groups}
        
        for fname in image_files:
            img = Image.open(os.path.join(self.config.image_dir, fname)).convert("RGB")
            print(f"\n{'='*60}\nImage: {fname}")
            
            for gk, layer_indices in self.config.layer_groups.items():
                print(f"  [{gk}] layers={layer_indices}", end="", flush=True)
                img_results = {}
                
                for query in self.config.queries:
                    heatmap = self._compute_heatmap(img, query, layer_indices)
                    img_results[query] = heatmap
                    print(".", end="", flush=True)
                
                print(f" done")
                self.results[gk][fname] = img_results
                
                # Compute similarity for this image+group
                flat = {q: img_results[q].flatten() for q in self.config.queries}
                nq = len(self.config.queries)
                sim = np.zeros((nq, nq))
                for i in range(nq):
                    for j in range(nq):
                        r, _ = pearsonr(flat[self.config.queries[i]], flat[self.config.queries[j]])
                        sim[i, j] = r
                
                # Quick report
                if nq >= 8:
                    face_sims = [sim[i, j] for i in range(6) for j in range(i+1, 6)]
                    fc_sims = [sim[i, j] for i in range(6) for j in range(6, 8)]
                    print(f"    within-face={np.mean(face_sims):.4f}  face-ctrl={np.mean(fc_sims):.4f}")
                
                self.results[gk][fname + "_sim"] = sim.tolist()
        
        return self
    
    # ── Statistics ──
    
    def compute_stats(self) -> Dict:
        """Aggregate statistics across all images."""
        self.agg_stats = {}
        print(f"\n{'='*70}")
        print("SUMMARY: Layer-wise Cross-Query Similarity")
        print(f"{'='*70}")
        
        for gk in self.config.layer_groups:
            all_face, all_fc, all_cc = [], [], []
            for fname in self._get_images():
                sim = np.array(self.results[gk][fname + "_sim"])
                all_face.extend([sim[i, j] for i in range(6) for j in range(i+1, 6)])
                all_fc.extend([sim[i, j] for i in range(6) for j in range(6, 8)])
                all_cc.append(sim[6, 7])
            
            fm, fs = float(np.mean(all_face)), float(np.std(all_face))
            fcm, fcs = float(np.mean(all_fc)), float(np.std(all_fc))
            ccm = float(np.mean(all_cc))
            
            self.agg_stats[gk] = {
                "layers": self.config.layer_groups[gk],
                "within_face_mean": round(fm, 4),
                "within_face_std": round(fs, 4),
                "face_vs_ctrl_mean": round(fcm, 4),
                "face_vs_ctrl_std": round(fcs, 4),
                "ctrl_ctrl_mean": round(ccm, 4),
            }
            print(f"  {gk:<22s}  within-face={fm:.4f}+/-{fs:.4f}  "
                  f"face-ctrl={fcm:.4f}+/-{fcs:.4f}  ctrl-ctrl={ccm:.4f}")
        
        # Save JSON
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, "attention_stats.json")
        with open(path, "w") as f:
            json.dump(self.agg_stats, f, indent=2)
        print(f"Saved: {path}")
        return self.agg_stats
    
    # ── Visualization ──
    
    def visualize(self):
        """Generate all figures."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        gks = list(self.config.layer_groups.keys())
        qs = self.config.queries
        nq = len(qs)
        image_files = self._get_images()
        
        theme = {
            "facecolor": "white",
            "cmap_sim": "viridis",
            "cmap_hm": "inferno",
            "vmin_sim": 0.5,
            "vmax_sim": 1.0,
        }
        
        # ── 1. Per-group similarity matrix ──
        for fname in image_files:
            base = Path(fname).stem
            for gk in gks:
                sim = np.array(self.results[gk][fname + "_sim"])
                fig, ax = plt.subplots(figsize=(9, 7))
                fig.patch.set_facecolor(theme["facecolor"])
                im = ax.imshow(sim, cmap=theme["cmap_sim"],
                               vmin=theme["vmin_sim"], vmax=theme["vmax_sim"])
                ax.set_xticks(range(nq)); ax.set_yticks(range(nq))
                ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(qs, fontsize=9)
                for i in range(nq):
                    for j in range(nq):
                        ax.text(j, i, f"{sim[i,j]:.4f}", ha="center", va="center", fontsize=6.5)
                plt.colorbar(im, ax=ax, label="Pearson r")
                ax.set_title(f"[{gk}] Cross-Query Attention Similarity", fontsize=13, fontweight="bold")
                plt.tight_layout()
                p = os.path.join(self.config.output_dir, f"sim_{base}_{gk.replace(' ','_')}.png")
                plt.savefig(p, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Figure: {p}")
        
        # ── 2. Four-panel comparison ──
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.patch.set_facecolor("white")
        axes = axes.flatten()
        
        for idx, gk in enumerate(gks):
            ax = axes[idx]
            sim_sum = None
            count = 0
            for fname in image_files:
                s = np.array(self.results[gk][fname + "_sim"])
                sim_sum = s if sim_sum is None else sim_sum + s
                count += 1
            sim_avg = sim_sum / count
            
            im = ax.imshow(sim_avg, cmap=theme["cmap_sim"],
                           vmin=theme["vmin_sim"], vmax=theme["vmax_sim"])
            ax.set_xticks(range(nq)); ax.set_yticks(range(nq))
            ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(qs, fontsize=8)
            for i in range(nq):
                for j in range(nq):
                    ax.text(j, i, f"{sim_avg[i,j]:.3f}", ha="center", va="center", fontsize=6)
            rect = plt.Rectangle((-0.5, -0.5), 6, 6, fill=False,
                                 edgecolor="cyan", linewidth=2, linestyle="--")
            ax.add_patch(rect)
            ax.set_title(f"{gk}", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
        
        plt.suptitle("Layer-wise Ablation: VLM Self-Attention Similarity (avg over images)",
                     fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        p = os.path.join(self.config.output_dir, "layer_ablation_4panel.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure: {p}")
        
        # ── 3. Bar chart ──
        group_labels = [gk for gk in gks]
        face_means = [self.agg_stats[gk]["within_face_mean"] for gk in gks]
        face_stds = [self.agg_stats[gk]["within_face_std"] for gk in gks]
        fc_means = [self.agg_stats[gk]["face_vs_ctrl_mean"] for gk in gks]
        fc_stds = [self.agg_stats[gk]["face_vs_ctrl_std"] for gk in gks]
        
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor("white")
        x = np.arange(len(gks))
        w = 0.35
        bars1 = ax.bar(x - w/2, face_means, w, yerr=face_stds, capsize=4,
                       label="Within-face", color="#4A6FA5")
        bars2 = ax.bar(x + w/2, fc_means, w, yerr=fc_stds, capsize=4,
                       label="Face vs Control", color="#D16B6B")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Pearson r", fontsize=12)
        ax.set_title("Cross-Query Attention Similarity by Layer Group", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        
        for bar, val in zip(bars1, face_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        for bar, val in zip(bars2, fc_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        
        plt.tight_layout()
        p = os.path.join(self.config.output_dir, "layer_ablation_bars.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure: {p}")
        
        # ── 4. NEW: Heatmap grid per group (qualitative view) ──
        # Show each layer group's 8 heatmaps side by side
        for fname in image_files[:1]:  # Just first image
            base = Path(fname).stem
            rows, cols = 2, 4
            fig, axes = plt.subplots(len(gks), nq, figsize=(3*nq, 3*len(gks)))
            fig.patch.set_facecolor("white")
            
            for row_idx, gk in enumerate(gks):
                for col_idx, query in enumerate(qs):
                    ax = axes[row_idx, col_idx] if len(gks) > 1 else axes[col_idx]
                    hm = self.results[gk][fname][query]
                    vmin, vmax = np.percentile(hm, 5), np.percentile(hm, 99.5)
                    im = ax.imshow(hm, cmap=theme["cmap_hm"],
                                   interpolation="bilinear", vmin=vmin, vmax=vmax)
                    ax.set_title(f'"{query}"' if row_idx == 0 else "", fontsize=7)
                    ax.axis("off")
                    if col_idx == 0:
                        ax.set_ylabel(gk, fontsize=8, fontweight="bold")
            
            plt.suptitle(f"Attention Heatmaps by Layer Group ({base})",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            p = os.path.join(self.config.output_dir, f"heatmap_grid_{base}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Figure: {p}")
        
        # ── 5. NEW: Same query across groups (horizontal comparison) ──
        for fname in image_files[:1]:
            base = Path(fname).stem
            sample_queries = qs[:3] + [qs[-1]]  # nose, eyes, mouth, bottom-left area
            fig, axes = plt.subplots(len(sample_queries), len(gks),
                                     figsize=(3*len(gks), 3*len(sample_queries)))
            fig.patch.set_facecolor("white")
            
            for qi, query in enumerate(sample_queries):
                for gi, gk in enumerate(gks):
                    ax = axes[qi, gi]
                    hm = self.results[gk][fname][query]
                    vmin, vmax = np.percentile(hm, 5), np.percentile(hm, 99.5)
                    im = ax.imshow(hm, cmap=theme["cmap_hm"],
                                   interpolation="bilinear", vmin=vmin, vmax=vmax)
                    ax.axis("off")
                    if qi == 0:
                        ax.set_title(gk, fontsize=9, fontweight="bold")
                    if gi == 0:
                        ax.set_ylabel(f'"{query}"', fontsize=9, fontweight="bold")
            
            plt.suptitle(f"Same Query Across Layer Groups ({base})",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            p = os.path.join(self.config.output_dir, f"query_comparison_{base}.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Figure: {p}")
        
        print("\nAll figures saved.")
        return self
    
    # ── Full pipeline ──
    
    def pipeline(self):
        """Run everything: experiment → stats → figures."""
        self.run()
        self.compute_stats()
        self.visualize()
        return self


# ═══════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════

def main():
    """Run with default config (Qwen3-VL-2B layer ablation)."""
    config = VLMConfig()
    analyzer = AttentionAnalyzer(config, adapter=Qwen3VLAdapter())
    analyzer.pipeline()


if __name__ == "__main__":
    main()
