from typing import Dict, List, Any

import torch
import json
import config.data_config
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from config import data_config


class TextDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id      # 起始符token_id
        self.eos_token_id = tokenizer.eos_token_id      # 终止符token_id
        self.pad_token_id = tokenizer.pad_token_id      # padding符号token_id
        self.max_length = max_length                    # 序列最大长度
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = f.readlines()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # text格式：
        # {"instruction": "类型#裙*版型#显瘦*风格#性感*裙型#包臀裙*裙型#鱼尾裙",
        #  "output": "修身包臀版型结合性感鱼尾裙摆设计，彰显婉约优雅风情之余，为整体注入几分俏皮灵动气息。且下摆辅以律动感摺裥元素，更烘托出女性浪漫精致的一面。"}
        text = json.loads(text)     # 转换为python对象dict

        # 收集多轮对话
        utterances = [text['instruction'], text['output']]
        # utterances格式：
        # ["类型#裙*版型#显瘦*风格#性感*裙型#包臀裙*裙型#鱼尾裙",
        # "修身包臀版型结合性感鱼尾裙摆设计，彰显婉约优雅风情之余，为整体注入几分俏皮灵动气息。且下摆辅以律动感摺裥元素，更烘托出女性浪漫精致的一面。"]
        final_input_ids = []
        final_attention_mask = []
        final_target_mask = []
        for i, utterance in enumerate(utterances):
            encode = self.tokenizer(utterance, add_special_tokens=False, return_tensors='pt')
            final_input_ids += encode['input_ids'].squeeze(0).tolist()
            final_attention_mask += encode['attention_mask'].squeeze(0).tolist()
            if (i % 2) == 0:    # instruction部分，target_mask全0
                final_target_mask += ([0] * len(encode['input_ids'].squeeze(0)))
            else:   # output部分，target_mask全1，注意加一个eos_token_id
                final_input_ids += [self.eos_token_id]
                final_attention_mask += [1]
                final_target_mask += ([1] * (len(encode['input_ids'].squeeze(0)) + 1))
        return dict(
            input_ids=final_input_ids,
            attention_mask=final_attention_mask,
            target_mask=final_target_mask,
        )


# 数据整形器，主要目的是将所有input_ids都padding至相同的长度，并调整合适的attention_mask和target_mask
class SFTTextDataCollator(object):

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch_inst_text: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 计算一个batch中所有输入序列的最大长度
        lengths = [len(input_seq['input_ids']) for input_seq in batch_inst_text]
        max_len = min(max(lengths), self.max_length)    # 如果最大长度超过max_length，取max_length截断数据
        batch_input_token_ids = []
        batch_input_attention_mask = []
        batch_input_target_mask = []
        for intput_seq in batch_inst_text:
            input_ids = intput_seq['input_ids']
            attention_mask = intput_seq['attention_mask']
            target_mask = intput_seq['target_mask']
            padding_length = max_len - len(input_ids)   # 需要padding的长度
            # 对输入token_ids序列padding至合适长度，并补充目标掩码和注意力掩码
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            target_mask += [0] * padding_length
            # 将处理后的序列和掩码加到列表中
            batch_input_token_ids.append(input_ids)
            batch_input_attention_mask.append(attention_mask)
            batch_input_target_mask.append(target_mask)
        # 将batch输入转换成tensor输入给模型
        input_ids_batch = torch.tensor(batch_input_token_ids, dtype=torch.long, device=self.device)
        attention_mask_batch = torch.tensor(batch_input_attention_mask, dtype=torch.long, device=self.device)
        target_mask_batch = torch.tensor(batch_input_target_mask, dtype=torch.long, device=self.device)
        return dict(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            target_mask=target_mask_batch,
        )



# model_dir = data_config.model_dir
# text_file_path = data_config.text_file_path
# # 在不需要计算梯度的环境下加载模型
# with torch.no_grad():
#     tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#     txt_dataset = TextDataset(text_file_path, tokenizer, tokenizer.model_max_length)
#     print(txt_dataset[0])








