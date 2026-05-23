# langchain_community
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")


res = model.invoke(input="你是谁呀能做什么？")

print(res.content)
