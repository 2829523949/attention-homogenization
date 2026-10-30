# RealBind2 Facts-only vs Image+Facts：500 图实验包

本包从冻结的 **RealBind2 (RealBind-v2.0)** 中确定性选取 **500 张唯一来源图片**，每张图片对应一个组合题案例，运行两个共享同一文本的配对条件：

- `facts_only`：三条正确事实 + 冻结问题，无图；
- `image_plus_facts`：完全相同的文本 + 原始整图。

两条件文本逐字节一致，唯一差异是图像 content block。图像不裁剪、不画框、不做标记、不添加对象标签。

## 包内容

```text
cases.jsonl                 500 个自包含案例（事实、问题、prompt、hash、元数据）
images/                     500 张原始图片
manifest.json               选择协议、条件契约、来源 hash
selection_audit.json        案例选择与平衡审计
selected_images.csv         案例与图片索引
README.md                   本文件
PROTOCOL.md                 实验设计细节
scripts/
  run_facts_image.py        模型无关的 paired 推理 runner
  summarize_facts_image.py  配对指标、VIR、McNemar、bootstrap
  audit_package.py          包完整性审计
  custom_adapter_template.py 新 VLM adapter 模板
  select_500_cases.py       可复现的选择脚本（需要 RealBind2 冻结源）
configs/
  qwen35_4b.yaml
  qwen35_9b.yaml
  qwen3vl_4b.yaml
  qwen3vl_8b.yaml
  internvl35_8b.yaml
  llava_next_7b.yaml
  paths.example.yaml
src/                        运行所需的最小 RealBind2 adapter/scorer 代码
run_qwen35_4b.sh
run_qwen3vl_4b.sh
run_internvl35_8b.sh
run_llava_next_7b.sh
...
```

## 1. 安装运行环境

参考：

```bash
pip install -r requirements-main-vlm.txt
```

运行前把 `configs/paths.example.yaml` 复制为 `configs/paths.yaml`，并填写每个模型的本地路径：

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

`configs/paths.yaml` 至少需要包含当前要跑的 `path_key`，例如：

```yaml
models:
  qwen35_4b: /path/to/Qwen3.5-4B
```

## 2. 先做无模型预检

```bash
python scripts/run_facts_image.py \
  --config configs/qwen35_4b.yaml \
  --paths configs/paths.example.yaml \
  --cases cases.jsonl \
  --images-root images \
  --output outputs/qwen35_4b/results.jsonl \
  --dry-run
```

预期输出包含：

```text
"preflight": {"passed": true, ...}
DRY_RUN_PASS
```

注意：`--dry-run` 不检查 model path，因此可以使用示例 paths；正式运行必须使用真实 `paths.yaml`。

## 3. 运行不同 VLM

Qwen3.5-4B：

```bash
bash run_qwen35_4b.sh
```

Qwen3-VL-4B：

```bash
PATHS=configs/paths.yaml bash run_qwen3vl_4b.sh
```

其他模型同理：

```bash
bash run_qwen35_9b.sh
bash run_qwen3vl_8b.sh
bash run_internvl35_8b.sh
bash run_llava_next_7b.sh
```

每个案例产生两条记录。输出默认写到：

```text
outputs/<model_key>/facts_image.jsonl
outputs/<model_key>/facts_image.failed.jsonl
outputs/<model_key>/facts_image.run_meta.json
```

Runner 默认支持断点续跑：重复执行同一命令会跳过已完成的 `(case_id, condition)`。

## 4. 汇总指标

```bash
python scripts/summarize_facts_image.py \
  --results outputs/qwen35_4b/facts_image.jsonl \
  --cases cases.jsonl \
  --output-dir outputs/qwen35_4b
```

输出：

```text
outputs/<model_key>/metrics.json
outputs/<model_key>/SUMMARY.md
outputs/<model_key>/paired_cases.jsonl
```

核心指标：

- Facts-only / Image+Facts accuracy；
- `Δ = Image+Facts − Facts-only`；
- `VIR = P(image wrong | facts correct)`；
- `VRR = P(image correct | facts wrong)`；
- McNemar exact test；
- 按 `family_id` 聚类的 bootstrap 95% CI；
- 错误类型：另一对象颜色 / 第三种颜色 / 无效输出。

## 5. 多 GPU / 分片运行

Runner 支持按 case 的确定性分片。例如使用两张 GPU：

```bash
bash run_qwen35_4b.sh --num-shards 2 --shard-index 0   --output outputs/qwen35_4b/shard_0.jsonl
bash run_qwen35_4b.sh --num-shards 2 --shard-index 1   --output outputs/qwen35_4b/shard_1.jsonl
```

合并并检查重复：

```bash
python scripts/merge_results.py   --inputs outputs/qwen35_4b/shard_0.jsonl outputs/qwen35_4b/shard_1.jsonl   --output outputs/qwen35_4b/facts_image.jsonl
```

然后执行第 4 节的汇总命令。

## 6. 适配新的 VLM

已有 adapter：

- `qwen35`：Qwen3.5-4B / 9B
- `qwen3vl`：Qwen3-VL-4B / 8B
- `internvl35`：InternVL3.5-8B
- `llava_next`：LLaVA-NeXT-7B

要增加新模型，复制 `scripts/custom_adapter_template.py` 并修改载入、图像预处理和生成逻辑。然后创建配置：

```yaml
model_id: My-VLM
adapter: custom
adapter_class: my_adapter.MyVLMAdapter
path_key: my_vlm
runtime:
  device: cuda
  dtype: bfloat16
  use_cache: false
  max_new_tokens: 16
```

新 adapter 必须满足：

1. `prepare_prompt(text, image_path=None)`：无图时只发送文本；有图时只额外加入原始图像；
2. `generate(prepared) -> str`；
3. 不改变 prompt 文本，不添加示例、bbox、crop、mark 或提示性说明。

## 7. 注意事项

- 本子集只使用 **test split**。
- 为保证 facts-only 条件在原则上文本自足，排除了 `referent_A` 与 `referent_B` 文本完全相同的 family；这些 family 的注入事实无法仅靠文本区分两个对象。
- 每个来源图片最多选择一个 family 和一个组合题；左右目标题各 250 个。
- 选择完全确定、与模型输出无关。图片选择由固定 seed 的 SHA256 排序决定。
- 这 500 题是诊断性配对子集，不是 RealBind2 总体 VIR 的 prevalence estimate。
- 运行后应先检查 `prompt_hash_equal_all_pairs=true` 和 `image_presence_contract_ok_all_pairs=true`，再解释模型间差异。
