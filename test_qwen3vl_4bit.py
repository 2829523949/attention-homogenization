import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model with 4bit quantization...")
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("/root/Qwen3-VL-2B-Instruct")

# 查看显存
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"显存占用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已预留)")

# 测试推理
messages = [
    {"role": "user", "content": [{"type": "text", "text": "你好，请用一句话介绍自己"}]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=64)
response = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nQwen: {response}")
print(f"\n推理后显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
