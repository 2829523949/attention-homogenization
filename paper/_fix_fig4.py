with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

# Add architecture figure after the paragraph about pose mapping
target = r"where the projection follows the polar angle convention"
replacement = target + r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/architecture.png}
    \caption{Overview of our text encoder comparison framework. Text is encoded by either Qwen3-VL-2B (VLM branch, 1224M params) or CLIP-ViT-B/32 (CLIP branch, 67M params). Both outputs are mapped to a shared 2048-dimensional space, then to 6D pose via the same Pose MLP. The 6D pose is converted to EG3D camera parameters for face rendering.}
    \label{fig:arch}
\end{figure}

""" + target

c = c.replace(target, replacement)

with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
    f.write(c)
print("OK")
