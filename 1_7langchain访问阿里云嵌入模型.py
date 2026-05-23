from langchain_community.embeddings import DashScopeEmbeddings
import os

# 请确保已设置环境变量 DASHSCOPE_API_KEY，或在代码中指定
# os.environ["DASHSCOPE_API_KEY"] = "your-api-key"

model = DashScopeEmbeddings(model="text-embedding-v1")

print(model.embed_query("What is langchain?"))
print(model.embed_query("What is langchain?"))
print(model.embed_documents(["What is langchain?", "What is langgraph?", "晚上吃啥"]))