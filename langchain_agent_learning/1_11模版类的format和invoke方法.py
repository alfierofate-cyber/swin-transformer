from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

"""
PromptTemplate -> StringPromptTemplate -> BasePromptTemplate
FewShotPromptTemplate -> StringPromptTemplate -> BasePromptTemplate
ChatPromptTemplate -> StringPromptTemplate -> BasePromptTemplate
"""
template = PromptTemplate.from_template("我的邻居是{lastname}，最喜欢{hobby}")

res = template.format(lastname="张艺谋", hobby="看电影")
print(res, type(res))

res2 =template.invoke(input={"lastname": "周杰伦", "hobby": "唱歌"})
print(res2, type(res2))