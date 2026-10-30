# 实验协议：RealBind2 Facts-only vs Image+Facts (500 images)

## 1. 目的

检验当正确答案所需的文本事实已经充分给出时，加入原始图像是否会损害 VLM 对事实的使用。条件名：

- `FACTS_ONLY`
- `VISION_PLUS_FACTS` / `image_plus_facts`

这不是 image-only 对照，也不是把图像与文本做模态冲突；图像和注入事实来自同一冻结标注并保持语义一致。

## 2. 数据来源

- 数据集：RealBind2 / RealBind-v2.0 frozen release
- 默认数据根目录：`/mnt/d/datasets/RealBind_v2`
- 冻结文件：
  - `frozen/families.jsonl`
  - `frozen/questions.jsonl`
  - `frozen/construction_stats.json`
  - `frozen/splits.json`
- 来源 SHA256 记录在 `manifest.json` 和 `selection_audit.json`。

## 3. 选择设计

固定 seed：`realbind2-facts-image-500-v1-20261030`

1. 仅使用 `split == test`。
2. 排除 `referent_A` 与 `referent_B` 标准化后完全相同的 family（默认 75 个 test families 被排除）。原因是这种情况下三条注入事实不能仅凭文本区分对象，不满足 facts-only 文本自足的前提。
3. 按 `image_id` 聚合，每个来源图片最多选一个 family。
4. 对图片计算：
   `SHA256(seed + ":image:" + image_id)`。
5. 按该 hash 升序取前 500 张。
6. 每个入选图片只选一个组合题；按筛选 rank 交替选择 `REL2C_LEFT` 和 `REL2C_RIGHT`，得到 250/250。
7. 对有两个 family 的图片，按 family_id 的固定 hash 选择 family；选择同时独立于模型输出。
8. 最终得到 500 个唯一图片、500 个 case、1000 次生成。

模型预测不参与选择。该子集不是按模型难度或失败率筛选出的。

## 4. 案例字段

`cases.jsonl` 每行至少包含：

- `case_id`：本实验内部 ID，例如 `RB2_FI500_0001`
- `family_id`, `image_id`, `image_file`, `image_sha256`
- `split`, `qa_id`, `qa_type`, `question`
- `target_side`, `target_identity`, `target_color`
- `competitor_identity`, `competitor_color`
- `referent_A`, `referent_B`, `color_A`, `color_B`, `relation_A_to_B`
- `facts`：三条事实
- `text_prompt`：实际共享文本
- `text_prompt_sha256`
- `condition_contract`

## 5. Prompt

三条事实从 family 标注注入，文本模板为：

```text
The {A_phrase} is {color_A}.
The {B_phrase} is {color_B}.
The {A_phrase} is to the {relation} of the {B_phrase}.

{frozen REL2C question}
```

其中 `A_phrase` / `B_phrase` 是去掉开头 `the ` 的 referent，避免出现 `The the ...`。问题保持冻结的 REL2C 原样。

### 两条件唯一差异

- `facts_only`：text content block only；image_present=false。
- `image_plus_facts`：original full image content block + 同一 text content block；image_present=true。
- 两条件的 `text_prompt_sha256` 必须相等。
- 图像不做 crop、bbox overlay、mark、arrow、object label。

## 6. 模型与生成

由模型配置控制。默认建议与原始实验一致：

```yaml
runtime:
  device: cuda
  dtype: bfloat16
  use_cache: false
  max_new_tokens: 16
```

生成应为 greedy / non-thinking。不同模型的 decode 细节可以不同，但同一模型的两个条件必须严格一致；runner 对每个 `(case_id, condition)` 使用同一份模型配置。

## 7. 输出与可恢复性

每个成功的生成写一行 JSONL，主键为：

```text
record_id = case_id + "::" + condition
```

`run_facts_image.py` 默认恢复模式：已有 `status=ok` 的记录会被跳过；失败记录写入 `.failed.jsonl`，可重新执行补跑。

## 8. 指标定义

记：

- `F` = facts-only 正确
- `VF` = image+facts 正确

主指标：

- `Δ = Acc(VF) − Acc(F)`
- `VIR = P(VF wrong | F correct)`
- `VRR = P(VF correct | F wrong)`
- McNemar exact test
- family-cluster bootstrap，默认 seed=20261030，10,000 次

错误分类：

- `other_object_color`：预测为另一对象颜色；
- `third_color`：预测既不是目标也不是另一对象颜色；
- `invalid_or_ambiguous`：无法唯一解析为一个颜色。

## 9. 解释边界

该子集是诊断性 paired sample，不是 RealBind2 总体的干扰发生率估计。`VIR` 描述本子集内的事实正确条件下图像导致错误的频率，不能直接外推为 benchmark prevalence。

此外，选择规则排除了 exact same-referent family；因此本子集的结论应表述为“在文本事实可唯一指代且自足的条件上”。如果要研究同名词歧义，需要单独建一个 cohort。
