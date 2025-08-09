from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
import langchain_for_LLM_test01

llm = langchain_for_LLM_test01.GoChatGLM()

prompt = PromptTemplate(input_variables=['location', 'street'],
                        template='作为一名专业的旅游顾问，简单说一下{location}有什么好玩的景点，特别是在{street}？只要说一个就可以。')

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run({"location": "南京", "street": "新街口"}))