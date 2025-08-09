from langchain_core.prompts import AIMessagePromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain.prompts import LengthBasedExampleSelector, NGramOverlapExampleSelector

# 创建一个中英文对照样本的列表
examples = [
    {"中文": "你好", "English": "hello"},
    {"中文": "你好吗", "English": "How are you"},
    {"中文": "好久不见", "English": "Long time no see"},
]

# 创建一个提示模板，用于格式化输入和输出
example_prompt = PromptTemplate(
    input_variables=["中文", "English"],      # 输入变量
    template="Input: {中文}\nOutput: {English}\n",        # 格式化的template
)

# 创建一个基于长度的样本选择器，用于从样本列表中选择合适长度的样本
# example_selector = LengthBasedExampleSelector(
#     examples=examples,  # 可供选择的样本列表
#     example_prompt=example_prompt,  # 用于格式化样本的提示模板
#     max_length=25,      # 格式化后样本的最大长度
# )

example_selector = NGramOverlapExampleSelector(
    examples=examples,  # 可供选择的样本列表
    example_prompt=example_prompt,  # 用于格式化样本的提示模板
    threshold=-1.0
    # 对于负数阈值，选择器按照n-gram重叠得分对prompts进行排序，并不排除任何prompt
    # 对于大于1.0的阈值，选择器排除所有prompts，返回一个空列表
    # 对于0，选择器按照n-gram重叠得分对prompts进行排序，并派出与输入prompt示例没有n-gram重叠的prompt
)

# 创建一个基于少量样本的提示模板，用于动态生成提示
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="作为一个专业翻译，请翻译下面的文本内容：\n",    # prompt的前缀
    suffix="Input: {text}\nOutput:",    # prompt的后缀
    input_variables=["text"],       # 输入变量为待翻译的文本
)

from langchain_for_LLM_test01 import GoChatGLM
from langchain.chains import LLMChain

llm = GoChatGLM()
chain = LLMChain(llm=llm, prompt=dynamic_prompt)
print(chain.run("我喜欢徐林蓉"))