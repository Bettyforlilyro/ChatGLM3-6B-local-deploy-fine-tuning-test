from typing import Tuple, List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
import config.data_config as data_config

model_dir = data_config.model_dir
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

peft_model_dir = data_config.peft_model_path
# 加载LoRA参数
model = PeftModel.from_pretrained(base_model, data_config.peft_model_path)
# 合并到基础模型
model = model.merge_and_unload()
model.eval()    # 评估模式

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

history: List[Tuple[str, str]] = []
role = "USER"

while True:
    # 通过键盘接收用户输入
    query = input("请输入你的问题：")
    if query.lower() == "exit":
        break
    # 使用tokenizer构建聊天输入
    prompt = ""
    for i, (old_prompt, old_response) in enumerate(history):
        prompt += f"[Round {i + 1}]\n问：{old_prompt}\n答：{old_response}\n"
    prompt += f"[Round {len(history)}\n问：{query}]\n答："
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=2048, do_sample=True, top_p=0.8,
                                 temperature=0.7, eos_token_id=tokenizer.eos_token_id)
    # 解码并提取最新回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_response = response.split("答：")[-1].strip()
    # 更新历史记录
    history.append((query, new_response))
    print("ASSISTANT:", new_response)