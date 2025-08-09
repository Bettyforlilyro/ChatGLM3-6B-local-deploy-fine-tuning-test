from langchain_for_LLM_test01 import GoChatGLM
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# 生成剧情梗概的LLMChain
title = "青春爱情故事"
characters = "男主：周烨；女主：徐林蓉"
template = """
你是一个剧作家。给定剧名和主要角色，你的工作就是根据这个剧名写一个剧情梗概。
Title: {title}
Characters: {characters}
剧作家：这是上述剧本的剧情梗概：
"""
llm = GoChatGLM()
prompt_template = PromptTemplate(input_variables=["title", "characters"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# 生成剧情评判的LLMChain
llm = GoChatGLM()
template = """
你是《杨子晚报》的戏剧评论家。根据戏剧的梗概，你的工作是为戏剧写一篇评论。
戏剧梗概：{synopsis}
戏剧评论家对上述剧本的评论：
"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# 撰写戏剧海报
llm = GoChatGLM()
template = """
你是一家戏剧公司的社交媒体经理。给定剧本的标题、故事发生的时代和地点、剧本的梗概以及对剧本的评论，你的工作就是为该剧写一篇社交媒体帖子。
这里有一些关于演出时间和地点的背景资料：
上映时间：{time}
上映地点：{location}
剧作家创作的剧情梗概：{synopsis}
戏剧评论家对上述剧本的评论：{review}
"""
prompt_template = PromptTemplate(input_variables=["time", "location", "synopsis", "review"], template=template)
social_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="social_post_text")

from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
overall_chain = SequentialChain(memory=SimpleMemory(memories={"time": "8月26日晚上九点整", "location": "上海大剧院"}),
                                chains=[synopsis_chain, review_chain, social_chain],
                                input_variables=["title", "characters"],
                                output_variables=["social_post_text"],
                                verbose=True)
social_post_text = overall_chain({"title": title, "characters": characters})
print(social_post_text)

