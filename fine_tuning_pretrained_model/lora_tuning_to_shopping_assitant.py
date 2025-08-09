import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from hello_lora_tuning import print_params_of_model
import config.data_config as data_config
import config.model_config as model_config
import data.process_assitant_conversations as process_conversations

model_dir = data_config.model_dir
# 在不需要计算梯度的环境下加载模型
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

batch_size = model_config.batch_size
total_epochs = model_config.total_epochs
learning_rate = model_config.learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("before LoRA peft model.........")
print_params_of_model(model)


peft_model = get_peft_model(model, lora_config)     # 仅训练LoRA参数，其余参数冻结
print("after LoRAConfig.........")
print_params_of_model(peft_model)
print("============================")
# print(peft_model)   # 调试代码，最后一个线性层输出的output_features = 65024
# print("Vocab size:", tokenizer.vocab_size)  # 词汇表大小只有64798，与output_features不相等，详情可查阅HuggingFace官方文档

# 初始化数据集
train_dataset = process_conversations.ConversationsDataset(data_config.conversations_file_path, tokenizer)
# 数据整形（主要是padding所有input_ids至相同长度，并得到注意力掩码）
data_collect = process_conversations.DataCollatorForConversations()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collect)

# 损失函数、优化器、调度器
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_config.padding_label_id)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2400, eta_min=2e-5, last_epoch=-1)

# 使用Accelerator加速训练
accelerate = Accelerator(mixed_precision='fp16')    # 与加载模型的时候half()保持一致，此处取fp16半浮点精度
# 准备模型，数据加载器，优化器，学习率调度器
train_loader, peft_model, optimizer, lr_scheduler = accelerate.prepare(train_loader, peft_model, optimizer, lr_scheduler)

# 开始训练
peft_model.train()
for epoch in range(total_epochs):
    # 使用tqdm创建进度条
    pbar = tqdm(train_loader, total=len(train_loader))
    for data_dict in pbar:
        with accelerate.accumulate(peft_model):     # 支持梯度累积
            # 获取输入的token_ids序列和labels序列，并转移到GPU上
            input_ids = data_dict["input_ids"].to(device)
            input_ids = input_ids[:, :-1]
            labels = data_dict["labels"].to(device)
            labels = labels[:, 1:]
            # 获取模型输出的logits，并计算loss，反向传播计算梯度更新，同时更新学习率
            logits = peft_model(input_ids)["logits"]
            logits = logits.reshape(-1, logits.shape[-1])
            # 查阅ChatGLM3官方文档，输出的logits中的维度为65024，与词汇表大小64798不相等，需要手动截断
            logits = logits[:, :tokenizer.vocab_size]
            # labels = labels.view(-1)    # view要求张量必须连续存储，无法复制，但更加高效，reshape没有要求连续存储，必要时会复制
            labels = labels.reshape(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            # 反向传播可使用Accelerator加速替换
            accelerate.backward(loss)
            # loss.backward()
            if accelerate.sync_gradients:   # 旨在同步时执行
                accelerate.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            # 设置进度条的显示
            pbar.set_description(f"Epoch [{epoch}/{total_epochs}], Loss: {loss.item():.4f}, lr: {lr_scheduler.get_last_lr()[0] * 1000:.5f}")

# 保存训练好的模型参数
peft_model.save_pretrained(data_config.peft_model_path)



