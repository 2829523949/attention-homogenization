with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

# Replace citations that don't exist in new refs.bib
c = c.replace(r"\cite{CLIPFaceGAN,Text2Face}", r"\cite{CLIP}")
c = c.replace(r"\cite{CLIPFaceGAN,Text2Face}", r"\cite{CLIP}")  # 2nd occurrence
c = c.replace(r"\cite{AttentionSurvey}", r"\cite{CLIP}")
c = c.replace(r"\cite{Qwen3VL,LLaVA}", r"\cite{Qwen3VL,LLaVA,3DLLM}")

with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
    f.write(c)
print("Fixed")
