from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate

#创建所需的解析器
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

#创建模型
model = ChatTongyi(model="qwen3-max")

#第一个提示词模板
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，并封装为json格式。要求key为name，value为名字"
)

#第二个提示词模板
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请分析这个名字的含义"
)

#构建链条
chain = first_prompt | model | json_parser | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "苏", "gender": "女"}):
    print(chunk,end="",flush=True)

