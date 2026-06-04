"""测试不同语义编辑词的 LLM hidden state 区分度"""
import os,sys,torch
import numpy as np
sys.path.insert(0,'/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
from bridge import load_qwen, get_llm_hidden

qwen,processor=load_qwen()

# 编辑词分组 - 测试同类别内的区分度 vs 跨类别区分度
EDIT_GROUPS = {
    "年龄": ["年轻人", "老年人", "中年人", "青少年"],
    "胡子": ["有胡子", "没有胡子", "胡须", "络腮胡"],
    "表情": ["微笑", "悲伤", "愤怒", "惊讶", "面无表情"],
    "性别": ["男性", "女性"],
    "眼镜": ["戴眼镜", "不戴眼镜"],
    "头发": ["长发", "短发", "光头", "卷发"],
    "相机": ["正面平视", "左侧30度平视", "右侧45度仰视"],  # Phase1的对比组
    "无关": ["今天天气真好", "我爱吃苹果", "深度学习"],
}

all_hiddens = {}
all_labels = []

for group, texts in EDIT_GROUPS.items():
    for text in texts:
        h = get_llm_hidden(qwen, processor, text)
        all_hiddens[text] = h.cpu().numpy()
        all_labels.append((group, text))

texts_list = [t for _, t in all_labels]
n = len(texts_list)
print(f"{'='*70}")
print(f"语义编辑词 LLM hidden state 余弦相似度矩阵")
print(f"{'='*70}")
print(f"共 {n} 个文本, hidden dim = {all_hiddens[texts_list[0]].shape[0]}")

# 计算所有pair的cosine similarity
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a = all_hiddens[texts_list[i]]
        b = all_hiddens[texts_list[j]]
        sim_matrix[i][j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 打印表格
print(f"\n{'文本':<16s}", end="")
for t in texts_list[:10]:
    print(f"{t[:10]:>10s}", end="")
print()
print("-" * (16 + 11 * min(n, 10)))

for i in range(n):
    print(f"{texts_list[i]:<16s}", end="")
    for j in range(min(n, 10)):
        print(f"{sim_matrix[i][j]:>10.3f}", end="")
    print()

# 统计: 同类内vs跨类
intra_sims = []  # 同类内相似度
inter_sims = []  # 跨类相似度
for i in range(n):
    for j in range(i+1, n):
        g1, t1 = all_labels[i]
        g2, t2 = all_labels[j]
        s = sim_matrix[i][j]
        if g1 == g2:
            intra_sims.append(s)
        else:
            inter_sims.append(s)

print(f"\n{'='*70}")
print(f"统计结果")
print(f"{'='*70}")
print(f"同类内平均相似度:  {np.mean(intra_sims):.4f} ± {np.std(intra_sims):.4f}")
print(f"跨类平均相似度:    {np.mean(inter_sims):.4f} ± {np.std(inter_sims):.4f}")
print(f"区分度(M_diff):     {np.mean(intra_sims) - np.mean(inter_sims):.4f}")
print(f"全部平均相似度:    {np.mean(sim_matrix):.4f}")

# 关键对比: 编辑词 vs 角度词vs噪声词
相机组 = [t for g, t in all_labels if g == "相机"]
编辑组 = [t for g, t in all_labels if g not in ["相机", "无关"]]
噪声组 = [t for g, t in all_labels if g == "无关"]

编辑内 = []
编辑跨 = []
for i, t1 in enumerate(编辑组):
    for j, t2 in enumerate(编辑组):
        if i < j:
            idx1 = texts_list.index(t1)
            idx2 = texts_list.index(t2)
            编辑内.append(sim_matrix[idx1][idx2])

编辑相机 = []
for t1 in 编辑组:
    for t2 in 相机组:
        idx1 = texts_list.index(t1)
        idx2 = texts_list.index(t2)
        编辑相机.append(sim_matrix[idx1][idx2])

编辑噪声 = []
for t1 in 编辑组:
    for t2 in 噪声组:
        idx1 = texts_list.index(t1)
        idx2 = texts_list.index(t2)
        编辑噪声.append(sim_matrix[idx1][idx2])

print(f"\n{'='*70}")
print(f"编辑属性专项分析")
print(f"{'='*70}")
print(f"编辑词之间:           {np.mean(编辑内):.4f}")
print(f"编辑词 vs 相机角度:    {np.mean(编辑相机):.4f}")
print(f"编辑词 vs 无关文本:    {np.mean(编辑噪声):.4f}")
print(f"\n{'='*70}")
print(f"结论: ", end="")
if np.mean(intra_sims) - np.mean(inter_sims) > 0.05:
    print("有一定区分度, MLP可能学到编辑方向")
elif np.mean(intra_sims) - np.mean(inter_sims) > 0.02:
    print("区分度较弱, MLP可能困难")
else:
    print("几乎没有区分度, 这条路可能行不通")
print(f"{'='*70}")
