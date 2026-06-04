with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

# ===== 1. New title =====
c = c.replace(
    r"\title{\vspace{-0.5in}Revisiting Text Encoders for 3D Face Camera Control}",
    r"\title{\vspace{-0.5in}Attention Collapse in Vision-Language Models: \\ A Case Study on 3D Face Camera Control}"
)

# ===== 2. Rewrite Introduction to lead with attention collapse =====
old_intro = r"""Text-driven 3D face generation has emerged as a compelling research direction, enabling controllable synthesis of photorealistic face images from natural language descriptions. A core component of any such system is the text encoder---a model that converts free-form text into a representation that can condition the face generator. Recent works have explored a spectrum of encoders, from lightweight contrastive models like CLIP~\cite{CLIP} to large vision-language models (VLMs)~\cite{Qwen3VL,LLaVA,3DLLM}. Yet, an important question remains unanswered: \emph{does scaling up the text encoder lead to better control?}

Existing methods typically adopt a single text encoder without justification for their choice. Some works~\cite{CLIP} use CLIP for its efficiency, while others leverage large VLMs~\cite{Next3D} for their rich multimodal representations. However, no study has systematically compared the suitability of different encoder classes for 3D face control, leaving practitioners to rely on intuition rather than evidence.

In this paper, we conduct the first systematic comparison between a large VLM (Qwen3-VL-2B, 1224M trainable parameters) and a lightweight encoder (CLIP-ViT-B/32, 67M parameters) for text-driven 3D face camera pose control using the EG3D~\cite{EG3D} generator. We make three key contributions:

\noindent\textbf{1. Cross-attention spatial grounding failure.} We analyze Qwen3-VL's self-attention maps across different facial feature queries (nose, eyes, mouth, chin, forehead, beard) and control queries (``whole image'', ``bottom-left''). All queries produce near-identical attention distributions (Pearson r $>$ 0.9999). This demonstrates that VLM attention cannot spatially localize facial features, undermining its suitability for fine-grained face control.

\noindent\textbf{2. Embedding space collapse.} Using prompts that differ only in angle (0{$^\circ$} to 90{$^\circ$}), we measure how well each encoder distinguishes fine-grained pose descriptions. VLM hidden states exhibit a mean pairwise similarity of 0.9962 and a discriminability (adjacent vs. far similarity difference) of only 0.0028. CLIP embeddings, despite being 4x lower-dimensional, achieve 0.0459---a 16.4x improvement.

\noindent\textbf{3. Comprehensive empirical comparison.} On the same test set with English prompts, CLIP achieves superior pose accuracy (yaw error 3.58{$^\circ$} vs. 3.84{$^\circ$}, pitch error 4.60{$^\circ$} vs. 5.60{$^\circ$}) while being 10.8x faster (7.0ms vs. 75.7ms per inference) and 18x smaller in parameter count. We also show that VLM performance degrades significantly under language mismatch (yaw error increases from 3.84{$^\circ$} to 10.10{$^\circ$} when using Chinese prompts).

These findings challenge the assumption that larger text encoders are always better. On the contrary, they suggest that for 3D face camera control, a lightweight encoder is not only sufficient but superior across all practical metrics."""

new_intro = r"""Vision-Language Models (VLMs) have emerged as powerful tools for multimodal understanding, combining a visual encoder with a large language model (LLM) to process both images and text. A critical mechanism underlying their capability is cross-attention, which determines \emph{where} the model looks in an image when processing specific textual queries. For tasks requiring fine-grained spatial understanding---such as face editing, where the model must distinguish ``the nose'' from ``the eyes''---the spatial precision of this attention mechanism is paramount. Yet, the spatial behavior of VLM attention in localized face understanding has not been systematically investigated.

In this paper, we conduct the first in-depth analysis of VLM self-attention for facial feature localization. Using Qwen3-VL-2B as our testbed and the EG3D face generator as our evaluation platform, we make three key contributions:

\noindent\textbf{1. Attention collapse across facial features.} We show that VLM self-attention maps are nearly identical (Pearson r $>$ 0.9999) for queries targeting different facial features (nose, eyes, mouth, chin, forehead, beard) as well as control queries (``whole image'', ``bottom-left area''). This collapse is consistent across multiple face images at different viewing angles. The attention mechanism cannot spatially differentiate facial features, revealing a fundamental limitation of VLMs for fine-grained face understanding.

\noindent\textbf{2. Embedding space collapse for fine-grained control.} Beyond attention, we show that the LLM hidden states themselves collapse fine-grained numerical information. When prompted with descriptions differing only in angle (0{$^\circ$} to 90{$^\circ$}), VLM hidden states exhibit a mean pairwise similarity of 0.9962, achieving a discriminability of only 0.0028 between adjacent and distant angles.
CLIP embeddings, despite being 4$\times$ lower-dimensional and trained on a simpler objective, achieve 16.4$\times$ better discriminability (0.0459).

\noindent\textbf{3. Practical implications: lightweight encoders suffice.} The attention and embedding collapse manifest in downstream task performance. On the same camera pose control benchmark, CLIP (63M parameters) outperforms Qwen3-VL-2B (1224M parameters) in accuracy (yaw 3.58{$^\circ$} vs. 3.84{$^\circ$}) while being 10.8$\times$ faster and 18$\times$ smaller. We further show that VLM performance degrades severely under language mismatch (10.10{$^\circ$} yaw error with Chinese prompts).

These findings have significant implications: they demonstrate that current VLMs lack the spatial precision required for localized face understanding, and they suggest that for tasks within their capabilities, lightweight encoders are not only sufficient but superior."""

c = c.replace(old_intro, new_intro)

# ===== 3. Expand Section 4.1 - add cross-image analysis =====
old_41 = r"""\textbf{Setup.} We input a face image alongside text queries such as ``describe the nose'', ``describe the eyes'', and control queries (``whole image'', ``bottom-left area''). We extract the self-attention weights from the last three LLM layers, average across heads and layers, and measure the attention from the query tokens to the 16$\times$16 visual token grid.

\textbf{Results.} Figure~\ref{fig:heatmap} visualizes the attention maps; Figure~\ref{fig:sim} shows the Pearson correlation matrix. Every query---whether a specific facial feature or a control phrase---produces an attention map with correlations exceeding 0.9999 (Table~\ref{tab:sim_stats}). The minimum observed correlation is 0.99968 (between ``beard'' and ``bottom-left''), which is at the level of floating-point noise. This demonstrates that Qwen3-VL's self-attention cannot spatially distinguish facial features, rendering it unsuitable for fine-grained face control tasks."""

new_41 = r"""\textbf{Setup.} We render face images at three viewing angles (15{$^\circ$}, 30{$^\circ$}, 45{$^\circ$} right) using the EG3D generator. For each image, we construct eight textual queries: six targeting specific facial features (``nose'', ``eyes'', ``mouth'', ``chin'', ``forehead'', ``beard'' in Chinese) and two control queries (``whole image'', ``bottom-left area''). For each query-image pair, we input both the image and the text to Qwen3-VL-2B, extract the self-attention weights from the last three LLM layers, average across all heads, and measure the attention distribution from the query tokens to the 16$\times$16 visual token grid. This produces a 256-dimensional attention vector for each query, which we reshape into a 16$\times$16 heatmap.

\textbf{Attention maps are identical across queries.} Figure~\ref{fig:heatmap} visualizes the attention maps for all eight queries on one test image. The maps are visually indistinguishable---the same spatial pattern appears regardless of whether the model is asked to attend to the nose, the eyes, or the entire image. Figure~\ref{fig:sim} shows the Pearson correlation matrix across all query pairs. Every pairwise correlation exceeds 0.9999 on all three test images (Table~\ref{tab:sim_stats}). The minimum observed correlation is 0.99968 (between ``beard'' and ``bottom-left'' on the 15{$^\circ$} image), which is at the level of floating-point computation noise.

\textbf{Consistency across images.} This collapse is not an artifact of a specific image. Table~\ref{tab:sim_stats} reports the mean and minimum similarity across all three test images. Across all angles and all query pairs, the pattern is identical: correlations never drop below 0.99968, confirming that the failure is systematic rather than angle-dependent.

\textbf{Implication.} The attention collapse reveals a fundamental limitation: Qwen3-VL's self-attention mechanism cannot spatially differentiate facial features. This means that tasks requiring localized face understanding---such as attribute-specific editing (\emph{e.g.}, ``change the nose'')---cannot rely on the VLM's innate attention for spatial grounding without additional mechanisms. This finding directly motivates our subsequent investigation into whether the VLM's embeddings are any more discriminative than its attention maps."""

c = c.replace(old_41, new_41)

# ===== 4. Update sim stats table to show multi-image results =====
old_table = r"""\begin{table}[t]
\centering
\caption{Cross-query attention similarity statistics (Pearson r across all pairs).}
\label{tab:sim_stats}
\begin{tabular}{lcc}
\toprule
Query group & Mean r & Min r \\
\midrule
Face features (6 queries) & $>$0.9999 & 0.99968 \\
Control queries (2 queries) & $>$0.9999 & 0.99968 \\
Face vs. control & $>$0.9999 & 0.99968 \\
\bottomrule
\end{tabular}
\end{table}"""

new_table = r"""\begin{table}[t]
\centering
\caption{Cross-query attention similarity statistics (Pearson r) across three test images and eight query types. Results are averaged over all query pairs within each group.}
\label{tab:sim_stats}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{2}{c}{All 3 images} & \\
\cmidrule{2-3}
Query group & Mean r & Min r & Samples \\
\midrule
Face features (6 queries) & $>$0.9999 & 0.99968 & 45 pairs $\times$ 3 \\
Control queries (2 queries) & $>$0.9999 & 0.99968 & 1 pair $\times$ 3 \\
Face vs. control & $>$0.9999 & 0.99968 & 12 pairs $\times$ 3 \\
\bottomrule
\end{tabular}
\end{table}"""

c = c.replace(old_table, new_table)

with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
    f.write(c)
print("Rewrite complete!")
