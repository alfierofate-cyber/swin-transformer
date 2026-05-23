from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

parser = StrOutputParser()#输出解析器
model = ChatTongyi(model="qwen3-max")#模型
prompt = PromptTemplate.from_template(
     "我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知我名字无需其它内容。"
)

chain = prompt | model | parser | model

res = chain.invoke({"lastname": "苏", "gender": "女"})
print(res)


