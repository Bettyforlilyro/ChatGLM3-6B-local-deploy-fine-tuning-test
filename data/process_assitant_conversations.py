import torch
import json
import config.data_config
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


def preprocess_conversations(conversations, tokenizer, max_tokens=None):
    """
    Preprocesses conversations according to tokenizer
    :param conversations:
    :param tokenizer:
    :param max_tokens:
    :return:
    """
    all_input_token_ids = []    # 存储所有处理后的输入token_id
    all_labels = []     # 存储所有的labels
    for conversation in conversations:
        roles = [msg["role"] for msg in conversation]   # 一轮对话中每个角色
        messages = [msg["content"] for msg in conversation]     # 一轮对话中每个role的消息内容
        # 第一个role不是"ASSISTANT"，最后一个role是"ASSISTANT"
        assert roles[0] != "ASSISTANT"
        assert roles[-1] == "ASSISTANT"
        input_messages = []     # 需要输入的信息
        # 根据role将消息添加到input_messages中，"ASSISTANT"和"USER"的消息都被添加
        for role, msg in zip(roles, messages):
            if role == "ASSISTANT" or role == "USER":
                input_messages.append(msg)
        # 使用ChatGLM3的tokenizer对文本token化
        tokenized_input = tokenizer(input_messages, add_special_tokens=False)
        input_ids = []
        labels = []
        # 根据第一个角色是"SYSTEM"还是其他角色来添加初始的输入token_id和label
        if roles[0] == "SYSTEM":
            input_ids.extend([64790, 64792, 64794, 30910, 13])
            input_ids.extend(tokenized_input.input_ids[0])
            # data_config.padding_token_id=-100作为padding符号的token_id，不参与损失计算
            labels.extend([config.data_config.padding_label_id] * (len(tokenized_input.input_ids[0]) + 5))
        else:
            input_ids.extend([64790, 64792])
            labels.extend([config.data_config.padding_label_id] * 2)
        # 根据每个role和token化的文本，添加输入token_ids和labels
        for role, msg in zip(roles, tokenized_input.input_ids):
            if role == "USER":
                # USER提问部分，不参与损失函数计算，将label设置成padding_token_id
                if roles[0] == "SYSTEM":
                    labels.extend([config.data_config.padding_label_id] * (len(msg) + 5))
                    input_ids.extend([13, 64795, 30910, 13])
                else:
                    labels.extend([config.data_config.padding_label_id] * (len(msg) + 4))
                    input_ids.extend([64795, 30910, 13])
                input_ids.extend(msg)   # 当前消息的token_ids
                input_ids.extend([64796])   # 添加USER对话结束符
            elif role == "ASSISTANT":
                msg += [tokenizer.eos_token_id]     # 在消息后面加一个结束token
                labels.extend([30910, 13])      # 添加ASSISTANT对话开始的起始符
                labels.extend(msg)       # 将当前的消息token添加到标签列表中
                input_ids.extend([30910, 13])   # 添加ASSISTANT对话开始的起始符号
                input_ids.extend(msg)       # 将当前消息token添加到输入token_id列表中
        if max_tokens is None:
            max_tokens = tokenizer.model_max_length     # 设置默认最大token数量为tokenizer模型最大长度
        # 将输入token_id和labels列表转换为LongTensor，并截取前max_tokens个token
        input_ids = torch.LongTensor(input_ids)[:max_tokens]
        labels = torch.LongTensor(labels)[:max_tokens]
        # 确保形状相同一一对应
        assert input_ids.shape == labels.shape
        all_input_token_ids.append(input_ids)
        all_labels.append(labels)
    return dict(input_ids=all_input_token_ids, labels=all_labels)


tokenizer = AutoTokenizer.from_pretrained(config.data_config.model_dir, trust_remote_code=True)


class ConversationsDataset(Dataset):

    def __init__(self, conversations_file_path=config.data_config.conversations_file_path, tokenizer=tokenizer, max_tokens=None):
        super(ConversationsDataset, self).__init__()
        # 预处理对话文件中的数据，token化
        with open(conversations_file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        data_dict = preprocess_conversations(conversations, tokenizer, max_tokens)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])


# 由于需要使用每轮对话进行批量训练微调，需要对输入的token数据进行标准化整形（padding每个input_ids至相同长度）
class DataCollatorForConversations(object):

    def __init__(self):
        self.padding_token_id = config.data_config.padding_token_id
        self.padding_label = config.data_config.padding_label_id

    def __call__(self, instances):
        # instances 是一个包含多个实例的列表，每个实例都是一个dict，包含input_ids和labels
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ["input_ids", "labels"])
        # 对所有input_ids和labels进行填充，确保所有用于训练的序列长度一致
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.padding_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.padding_label)
        # 返回一个dict，包含经过处理后的input_ids, labels和根据input_ids生成的attention_mask
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.padding_token_id))