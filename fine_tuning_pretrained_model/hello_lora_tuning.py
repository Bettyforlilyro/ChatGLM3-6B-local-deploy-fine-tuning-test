import torch.nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModel

import config.data_config as data_config

# model_dir = data_config.model_dir
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

# 打印模型所有的参数情况
def print_params_of_model(model):
    """
    Prints parameters of model
    :param model:
    :return:
    """
    trainable_params_num = 0
    all_params_num = 0
    for name, param in model.named_parameters():
        all_params_num += param.numel()
        if param.requires_grad:
            trainable_params_num += param.numel()
    print('Total number of trainable parameters: {}'.format(trainable_params_num))
    print('Total number of parameters: {}'.format(all_params_num))
    print('Trainable parameters: {}'.format(100 * trainable_params_num / all_params_num))

# 寻找模型中所有指定类型的Layers
def find_all_target_layers_in_model(model, target_module=torch.nn.Linear):
    """
    Find all target layers in model
    :param model:
    :return:
    """
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, target_module):
            names = name.split('.')     # 将模块名称按'.'进行分割
            module_names.add(names[0] if len(names) == 1 else names[-1])
    # 如果存在"lm_head"，移除。这是对16位情况的特殊处理
    if "lm_head" in module_names:
        module_names.remove("lm_head")
    return list(module_names)

# 创建一个LoraConfig对象，设置相关参数
lora_config = LoraConfig(
    r=64,       # LoRA矩阵的秩，低秩矩阵B/A的内部维度
    lora_alpha=16,      # 缩放系数，控制ΔW=BA对权重矩阵W的更新强度，即W‘ = W + (lora_alpha / r) * ΔW
    target_modules=["query_key_value"],     # lora更新的目标模块名称，Transformer架构中，常见的模块名称有query_key_value, dense
    lora_dropout=0.05,      # 低秩矩阵A的输出会应用一层Dropout防止过拟合，丢弃概率，如果数据集小应该适当增大dropout率
    bias="none",    # 可选"none", "all", "lora_only", 表示不训练偏置/训练所有偏置/仅训练LoRA偏置
    task_type="CAUSAL_LM",      # LoraConfig父类PeftConfig中的参数，用于设定任务的类型
)

# layer_names = find_all_target_layers_in_model(model)
# print("layer_names: ", layer_names)
# print_params_of_model(model)        # LoRA替换参数前的训练参数情况
# model = get_peft_model(model, lora_config)  # 两个步骤：找到目标模块参数并替换；设置只有LoRA相关的参数可训练，冻结其他参数
# print("After Lora-Config--------------")
# print_params_of_model(model)        # LoRA替换参数后的训练参数情况
# print("----------- training ------------")
# model.train()   # 设置训练模式
# ------
# 训练模型，此处代码略
# ------
# 保存训练后的LoRA可调参数（不包括冻结的参数）
# print("----------- trained over ------------")
# model.save_pretrained(save_directory=data_config.lora_param_save_path)
# print("----------- lora param saved ------------")
# lora_model = PeftModel.from_pretrained(model=model, model_id=data_config.lora_param_save_path)
# print("----------- lora param loaded ------------")
# lora_model = lora_model.merge_and_unload()      # 合并lora参数和原始模型参数
# lora_model.train(mode=False)    # 推断模式，不用进行训练准备，省掉一些步骤节省显存和计算资源
