import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("/root/Qwen3-VL-2B-Instruct")
print("Model loaded!")

messages = [
    {"role": "user", "content": [{"type": "text", "text": "你好，请介绍一下你自己"}]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nQwen: {response}")
print("\n推理完成!")
