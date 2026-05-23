import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOllama(model="qwen3:4b")

messages = [
    ("system", "你是一个诗人。"),
    ("human", "写一首唐诗。"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "按照你上一个回复的格式，在写一首唐诗。")
]
#    SystemMessage(content="你是一个边塞诗人。"),
#    HumanMessage(content="写一首唐诗"),
#    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
#    HumanMessage(content="按照你上一个回复的格式，在写一首唐诗。")



res = model.stream(input = messages)
for chunk in res:
    print(chunk.content,end="",flush=True)