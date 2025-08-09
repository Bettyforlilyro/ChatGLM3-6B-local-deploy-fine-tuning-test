from langchain_for_LLM_test01 import GoChatGLM
llm = GoChatGLM()

from langchain.prompts import (
ChatPromptTemplate,
SystemMessagePromptTemplate,
MessagesPlaceholder,    # 用于在模板中插入消息占位的类
HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# 创建一个聊天提示模板，包括系统消息，历史消息占位符和人类消息输入
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个强大的人工智能程序，可以知无不答，但是你不懂的东西会直接回答不知道。"),
    MessagesPlaceholder(variable_name="history"),       # 历史消息占位符
    HumanMessagePromptTemplate.from_template("{input}")     # 人类消息输入模板
])

# 创建一个用于存储对话的内存示例，并设置return_message=True 以返回消息内容
memory = ConversationBufferMemory(return_messages=True)
print("init memory", memory.chat_memory)    # 初始内存状态
print("***************\n***************\n***************")

# 使用内存，提示模板和LLM模型创建一个对话链实例
conversation_chain = ConversationChain(memory=memory, prompt=prompt, llm=GoChatGLM())

response = conversation_chain.predict(input="你好")
print(response)
print("***************\n***************\n***************")

response = conversation_chain.predict(input="请介绍一下南京有什么好玩的地方？")
print(response)
print("***************\n***************\n***************")

response = conversation_chain.predict(input="以上几个地方中，第一个地方请再具体介绍一下")
print(response)

print("***************\n***************\n***************")
print("current memory", memory.chat_memory)     # 更新之后的内存状态，包括对话历史记录