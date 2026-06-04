
with open("/root/Qwen3-VL/paper/main.tex", "r") as f:
    c = f.read()

# 1. Replace deprecated times package
c = c.replace(r"\usepackage{times}", r"\usepackage{mathptmx}")

# 2. Add textcomp after inputenc (for \textdegree compatibility)
c = c.replace(
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[utf8]{inputenc}" + "\n" + r"\usepackage{textcomp}"
)

# 3. Replace all \textdegree with $^\circ$ (most reliable)
c = c.replace(r"\textdegree", r"$^\circ$")

with open("/root/Qwen3-VL/paper/main.tex", "w") as f:
    f.write(c)
print("Fixed")
