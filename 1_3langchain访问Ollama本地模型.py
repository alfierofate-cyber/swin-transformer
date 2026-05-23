import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

res = model.invoke(input="你是谁呀能做什么？")

print(res)