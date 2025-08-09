import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from hello_lora_tuning import print_params_of_model
import config.data_config as data_config
import config.model_config as model_config
import data.process_text_dataset as preprocess


model_dir = data_config.model_dir
# 在不需要计算梯度的环境下加载模型
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

# 打印模型在内存/显存中的大小
print(f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 准备对模型进行kbit训练。该函数专门用于优化低精度量化模型如INT4/INT8/NF4。不使用量化可以不用这个
# 主要做了以下工作：
# 1 可选：启用梯度检查点
# 2 冻结非LoRA训练参数
# 3 处理量化层兼容性，修正量化模块反向传播行为，避免梯度计算错误
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
# 打印模型在内存/显存中的大小
# print(f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB")

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

text_train_dataset = preprocess.TextDataset(data_config.text_file_path, tokenizer, data_config.max_seq_length)
text_data_collector = preprocess.SFTTextDataCollator(tokenizer, data_config.max_seq_length)
text_train_loader = DataLoader(text_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=text_data_collector)

# 损失函数、优化器、调度器
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_config.padding_label_id)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200, eta_min=2e-6, last_epoch=-1)

# 使用Accelerator加速训练
accelerate = Accelerator(mixed_precision='fp16')    # 与加载模型的时候half()保持一致，此处取fp16半浮点精度
# 准备模型，数据加载器，优化器，学习率调度器
train_loader, peft_model, optimizer, lr_scheduler = accelerate.prepare(text_train_loader, peft_model, optimizer, lr_scheduler)


# 开始训练
peft_model.train()
for epoch in range(total_epochs):
    # 使用tqdm创建进度条
    pbar = tqdm(text_train_loader, total=len(text_train_loader))
    for data_dict in pbar:
        with accelerate.accumulate(peft_model):     # 支持梯度累积
            # 获取输入的token_ids序列和labels序列，并转移到GPU上（在数据处理时已经转移过设备）
            input_ids = data_dict["input_ids"][:, :-1]
            # print("\n", input_ids[0].tolist())
            target_mask = data_dict["target_mask"]
            labels = torch.where(target_mask == 1, data_dict["input_ids"], data_config.padding_label_id)
            labels = labels[:, 1:]
            # print(labels[0].tolist())
            # 获取模型输出的logits，并计算loss，反向传播计算梯度更新，同时更新学习率
            logits = peft_model(input_ids)["logits"]
            logits = logits.reshape(-1, logits.shape[-1])
            # 查阅ChatGLM3官方文档，输出的logits中的维度为65024，与词汇表大小64798不相等，需要手动截断?
            # logits = logits[:, :tokenizer.vocab_size]
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

peft_model.save_pretrained(data_config.text_lora_param_save_path)






