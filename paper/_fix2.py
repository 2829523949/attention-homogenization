with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

target = r"\textbf{Results.} As shown in Table~\ref{tab:sensitivity}"
repl = target + r"""

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/encoding_sensitivity.png}
    \caption{Pairwise similarity matrices for 10 angle descriptions (0\textdegree to 90\textdegree). CLIP shows more discriminative embedding structure than VLM.}
    \label{fig:sensitivity}
\end{figure}"""

if target in c:
    c = c.replace(target, repl)
    with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
        f.write(c)
    print("OK")
else:
    print("NOT FOUND")
