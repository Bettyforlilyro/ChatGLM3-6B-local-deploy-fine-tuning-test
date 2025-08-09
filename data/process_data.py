import json
from langchain.indexes import GraphIndexCreator
from langchain_for_LLM_test01 import GoChatGLM

file_name = '工商银行2021年度报告.txt'
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()
text = text.strip().split('\n')

context = ""
for line in text:
    line = json.loads(line)
    if line["type"] == 'text':
        try:
            context += line["inside"]
        except:
            pass

document = context[:480]
print(document)

llm = GoChatGLM()
index_creator = GraphIndexCreator(llm=llm)

times = 20
for i in range(times):
    graph = index_creator.from_text(document)
    print(graph.get_triples())