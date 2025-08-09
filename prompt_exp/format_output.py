from langchain_for_LLM_test01 import GoChatGLM
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the user's question, which should be a website."),
]
output_paser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_paser.get_format_instructions()

prompt_template = PromptTemplate(
    template="answer the user's question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

llm = GoChatGLM()
prompt = prompt_template.format_prompt(question="What is capital of England?")
response = llm(prompt.to_string())
print(response)
