from openai import OpenAI
#①.获取client对象，OpenAI类对象

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
#2. 调用模型
response = client.chat.completions.create(
    model= "qwen3-max",
    messages= [
{"role": "system", "content": "你是一个Python编程专家，并且不说废话简单回答"},
{"role": "assistant", "content": "好的，我是编程专家，并且话不多，你要问什么？"},
{"role": "user", "content": "写个扫雷的程序"}
],
stream=True

)
#3. 输出结果
#print(response.choices[0].message.content)
for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)