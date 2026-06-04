with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

# 1. Add FaceGPT to Related Work
c = c.replace(
    'Recent works incorporate CLIP~\\cite{CLIP} for text-driven editing.',
    'Recent works incorporate CLIP~\\cite{CLIP} for text-driven editing. Most recently, FaceGPT~\\cite{FaceGPT} embeds 3DMM parameters directly into VLM token space for 3D face reconstruction.'
)

# 2. Add FaceGPT to Discussion
c = c.replace(
    '\\textbf{Implications for face editing.}',
    '\\textbf{Relation to FaceGPT.} Concurrent work FaceGPT~\\cite{FaceGPT} demonstrates that VLMs can predict 3DMM parameters for reconstruction, which does not conflict with our findings since 3DMM parameters are global and do not require spatial attention. Our attention collapse result specifically impacts localized editing.\n\n\\noindent\\textbf{Implications for face editing.}'
)

# 3. Fix duplicate in Methodology
c = c.replace(
    'where the projection follows the polar angle convention\n\n\\begin{figure}',
    'where the projection follows the polar angle convention ($\\pi/2 =$ forward-facing) and normalized intrinsics.\n\n\\begin{figure}'
)

# 4. Remove duplicate sentence
c = c.replace(
    '\\end{figure}\n\nwhere the projection follows the polar angle convention ($\\pi/2 =$ forward-facing) and normalized intrinsics.',
    '\\end{figure}'
)

with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
    f.write(c)
print("Fixed!")
