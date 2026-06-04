"""模板组合生成多样化视角描述 (确保每条包含完整方位)"""
import sys, json, math, random
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

bridge = QwenEG3DBridge()
qwen = bridge.qwen
processor = bridge.processor

BASE_VIEWS = [
    (-3.14,  0.0,   "背面"),
    (-2.36, -0.35,  "左后方俯视"),  (-2.36, 0.0, "左后方平视"),  (-2.36, 0.35, "左后方仰视"),
    (-1.57, -0.4,   "左侧90度俯视"), (-1.57, 0.0, "左侧90度平视"), (-1.57, 0.4, "左侧90度仰视"),
    (-0.79, -0.35,  "左侧45度俯视"), (-0.79, 0.0, "左侧45度平视"), (-0.79, 0.35, "左侧45度仰视"),
    ( 0.00, -0.35,  "正面俯视"),      ( 0.00, 0.0, "正面平视"),      ( 0.00, 0.35, "正面仰视"),
    ( 0.79, -0.35,  "右侧45度俯视"), ( 0.79, 0.0, "右侧45度平视"), ( 0.79, 0.35, "右侧45度仰视"),
    ( 1.57, -0.4,   "右侧90度俯视"), ( 1.57, 0.0, "右侧90度平视"), ( 1.57, 0.4, "右侧90度仰视"),
    ( 2.36, -0.35,  "右后方俯视"),   ( 2.36, 0.0, "右后方平视"),   ( 2.36, 0.35, "右后方仰视"),
]

# 水平方位词库
H_DIRS = {
    (-3.14, 0.0): ["正后方", "背面", "脑后方向", "后脑勺方向"],
    (-2.36, -0.35): ["左后方", "左肩后方", "左后方偏上"],
    (-2.36, 0.0): ["左后方", "左肩后方", "左后方水平"],
    (-2.36, 0.35): ["左后方", "左肩后方", "左后方偏下"],
    (-1.57, -0.4): ["正左侧", "左侧90度", "左侧偏上"],
    (-1.57, 0.0): ["正左侧", "左侧90度", "左侧水平"],
    (-1.57, 0.4): ["正左侧", "左侧90度", "左侧偏下"],
    (-0.79, -0.35): ["左前方", "左侧45度", "左前上方"],
    (-0.79, 0.0): ["左前方", "左侧45度", "左前水平"],
    (-0.79, 0.35): ["左前方", "左侧45度", "左前下方"],
    ( 0.00, -0.35): ["正前方", "正面", "正前上方"],
    ( 0.00, 0.0): ["正前方", "正面", "正前水平"],
    ( 0.00, 0.35): ["正前方", "正面", "正前下方"],
    ( 0.79, -0.35): ["右前方", "右侧45度", "右前上方"],
    ( 0.79, 0.0): ["右前方", "右侧45度", "右前水平"],
    ( 0.79, 0.35): ["右前方", "右侧45度", "右前下方"],
    ( 1.57, -0.4): ["正右侧", "右侧90度", "右侧偏上"],
    ( 1.57, 0.0): ["正右侧", "右侧90度", "右侧水平"],
    ( 1.57, 0.4): ["正右侧", "右侧90度", "右侧偏下"],
    ( 2.36, -0.35): ["右后方", "右肩后方", "右后方偏上"],
    ( 2.36, 0.0): ["右后方", "右肩后方", "右后方水平"],
    ( 2.36, 0.35): ["右后方", "右肩后方", "右后方偏下"],
}

# 垂直方位词库 (包含所有pitch_offset值)
V_DIRS = {
    -0.4:  ["俯视", "从上往下", "高处向下", "向下拍摄", "高位俯拍"],
    -0.35: ["俯视", "从上往下", "高处向下", "向下拍摄", "高位俯拍", "顶视"],
    0.0:   ["平视", "水平", "与视线齐平", "正面水平", "侧视水平"],
    0.35:  ["仰视", "从下往上", "低处向上", "向上拍摄", "低位仰拍", "底视"],
    0.4:   ["仰视", "从下往上", "低处向上", "向上拍摄", "低位仰拍"],
}

# 句式模板
TEMPLATES = [
    "从{h}{v}拍摄",
    "{v}看{h}",
    "相机在{h}{v}拍摄",
    "在{h}进行{v}拍摄",
    "镜头位于{h}{v}拍摄",
    "从{h}视角{v}",
    "{h}方向的{v}角度",
    "拍摄{h}并{v}",
]

all_data = []
for yaw, pitch_offset, base_text in BASE_VIEWS:
    h_list = H_DIRS[(round(yaw, 2), round(pitch_offset, 2))]
    v_list = V_DIRS[round(pitch_offset, 2)]
    
    variants = set()
    variants.add(base_text)  # 保留原始模板
    
    # 组合生成至少20种
    for template in TEMPLATES:
        for h in h_list:
            for v in v_list[:2]:  # 每个垂直方向取前2种
                text = template.format(h=h, v=v)
                if len(text) <= 20:
                    variants.add(text)
    
    # 如果不够，用LLM补充
    if len(variants) < 10:
        prompt = f"用5种不同方式描述'{base_text}'的相机位置，要求包含具体方位。格式: '从...方向...拍摄':"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], return_tensors="pt").to(qwen.device)
        generated_ids = qwen.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.8, top_p=0.9)
        output = processor.decode(generated_ids[0], skip_special_tokens=True)
        raw_parts = output.split("assistant\n")[-1].strip().split("，")
        for part in raw_parts:
            part = part.strip().strip("，。. ")
            if 5 < len(part) < 20 and any(k in part for k in ["左","右","正前","后","俯","仰"]):
                variants.add(part)
    
    for v in list(variants)[:15]:
        all_data.append({'text': v, 'yaw': yaw, 'pitch_offset': pitch_offset})
    print(f"'{base_text:12s}' → {min(len(variants),15)}种")

with open("/root/Qwen3-VL/text_variants.json", "w") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n总数: {len(all_data)} 条")
print("\n示例:")
for d in all_data[:12]:
    print(f"  '{d['text']:18s}' yaw={d['yaw']:+5.2f} pitch={d['pitch_offset']:+5.2f}")
