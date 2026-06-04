with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    content = f.read()

# Add speed_benchmark figure
target = r"\textbf{Setup.} We measure end-to-end inference latency on an NVIDIA RTX 5060 Laptop GPU. Both models execute the complete text-to-pose pipeline (encoder forward + pose MLP). We report statistics over 100 trials after 10 warmup iterations."

replacement = target + r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/speed_benchmark.png}
    \caption{Inference speed benchmark comparing VLM 2B and CLIP 63M. CLIP achieves 10.8$\times$ lower latency and 10.8$\times$ higher throughput with 18$\times$ fewer parameters.}
    \label{fig:speed}
\end{figure}"""

if target in content:
    content = content.replace(target, replacement)
    with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAIL - target not found")
    # Find help
    idx = content.find("inference latency")
    print(f"Found at: {idx}")
    print(f"Context: {repr(content[idx:idx+200])}")
