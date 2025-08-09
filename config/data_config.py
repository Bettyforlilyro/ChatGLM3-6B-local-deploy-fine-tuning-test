model_dir = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\model\chatglm3'
model_name = 'ChatGLM3-6B'
model_32k_dir = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\model\chatglm3-6b-32k'
model_32k_name = 'chatglm3-6b-32k'

lora_param_save_path = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\model\lora_param'
text_lora_param_save_path = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\fine_tuning_pretrained_model\text_generate_lora_params'
conversations_file_path = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\data\chatGLM3_dataFormatted_sample.json'
peft_model_path = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\fine_tuning_pretrained_model\lora_query_key_value'
text_file_path = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\data\train.jsonl'
business_report_path = r'D:\projects\python\ChatGLM3-6B-local-deploy-fine-tuning\data\alltxt'

host_ip = '127.0.0.1'
host_port = 7866

padding_token_id = 0
padding_label_id = -100
max_seq_length = 256

BOS_TOKEN = "[BOS]"  # 句子开始
EOS_TOKEN = "[EOS]"  # 句子结束
PAD_TOKEN = "[PAD]"  # 填充
UNK_TOKEN = "<UNK>"  # 未知词
SEP_TOKEN = "[SEP]"  # 未知词



