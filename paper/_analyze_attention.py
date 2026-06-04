import json

with open("/root/Qwen3-VL/attention_evidence/similarity_data.json") as f:
    d = json.load(f)

print("Images:", list(d["cross_similarities"].keys()))
print("Queries:", d["queries"])
for img, mat in d["cross_similarities"].items():
    n = len(mat)
    off_diag = [mat[i][j] for i in range(n) for j in range(n) if i != j]
    print(f"{img}: min={min(off_diag):.4f}, mean={sum(off_diag)/len(off_diag):.4f}")
