from langchain_core.prompts import AIMessagePromptTemplate, PromptTemplate

from langchain_for_LLM_test01 import  GoChatGLM
from langchain.chains import LLMChain

from langchain.prompts import (
ChatPromptTemplate,
SystemMessagePromptTemplate,
AIMessagePromptTemplate,
HumanMessagePromptTemplate
)

# template = "你是一个有用的翻译助手，现在帮我翻译下面的文本。"
# system_message_template = SystemMessagePromptTemplate.from_template(template)
# example_human = HumanMessagePromptTemplate.from_template("Hi")
# example_ai = AIMessagePromptTemplate.from_template("中文：我爱中国。英文：I lova China.")
# human_template = "{text}"
# human_message_template = HumanMessagePromptTemplate.from_template(human_template)
# chat_prompt = ChatPromptTemplate.from_messages([system_message_template, example_human, example_ai, human_message_template])

prompt = PromptTemplate(
    template="你是一个专业的翻译助手，现在需要将 {input_language} 翻译为 {output_language}",
    input_variables={"input_language", "output_language"},
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=GoChatGLM(), prompt=chat_prompt)
print(chain.run({"input_language": "中文", "output_language": "French", "text": "我喜欢徐林蓉"}))