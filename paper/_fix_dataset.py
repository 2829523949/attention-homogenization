with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

# Add dataset paragraph after training paragraph
old = r"Both bridges are trained for 80 epochs with the Adam optimizer ($\text{lr}=1\times10^{-4}$). The text encoder is frozen in both cases."
new = r"""Both bridges are trained for 80 epochs with the Adam optimizer ($\text{lr}=1\times10^{-4}$). The text encoder is frozen in both cases.

\textbf{Dataset.} We construct a lightweight dataset of 21 camera pose templates covering yaw angles from $-45^\circ$ to $45^\circ$ and pitch variations around the forward-facing direction. Each pose is paired with 5 text templates (e.g., ``A person looking to the right at \{yaw\} degrees'') plus 500 automatically generated diverse descriptions per training epoch. The dataset is split by identity to prevent leakage: 3 identities for validation and the remainder for training. Since all face images are rendered by the public EG3D~\cite{EG3D} generator pretrained on FFHQ~\cite{StyleGAN2}, the entire dataset can be reproduced from a lightweight JSON specification of pose templates and text templates."""

c = c.replace(old, new)

with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
    f.write(c)
print("Dataset paragraph added!")
