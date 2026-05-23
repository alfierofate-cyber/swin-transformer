from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

model = ChatTongyi(model="qwen3-max")
str_parser = StrOutputParser()

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}， 刚生了{gender}，请起名,仅告诉我名字无需其它内容。"
)

second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请分析这个名字的含义"
)

#函数的入参：AIMessage -> dict({"name": "xxx"})
my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

chain = first_prompt | model | my_func | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "王", "gender": "女"}):
    print(chunk,end="",flush=True)