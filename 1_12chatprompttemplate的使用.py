from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，能够将{input_language}翻译成{output_language}。"),
    MessagesPlaceholder("history"),
    ("human", "{text}"),
])

history_data = [
    ("human", "The weather is nice today."),
    ("ai", "今天天气很好。"),
    ("human", "I love programming."),
    ("ai", "我喜欢编程。"),
    ("human", "Machine learning is changing the world."),
    ("ai", "机器学习正在改变世界。"),
]

# 构建 chain: prompt -> 模型
model = ChatTongyi(model="qwen-max")
chain = chat_prompt_template | model

result = chain.invoke({
    "history": history_data,
    "input_language": "English",
    "output_language": "Chinese",
    "text": "Modern GPUs feature specialized hardware for low-precision floating-point arithmetic to accelerate compute-intensive workloads that do not require high numerical accuracy, such as those from artificial intelligence. However, despite the significant gains in computational throughput, memory bandwidth utilization, and energy efficiency, integrating low-precision formats into scientific applications remains difficult. We introduce Kernel Float, a header-only C++ library that simplifies the development of portable mixed-precision GPU kernels. Kernel Float provides a generic vector type, a unified interface for common mathematical operations, and fast approximations for low-precision transcendental functions that lack native hardware support. To demonstrate the potential of mixed-precision computing unlocked by our library, we integrated Kernel Float into nine GPU kernels from various domains. Our evaluation on Nvidia A100 and AMD MI250X GPUs shows performance improvements of up to 12x over double precision, while reducing source code length by up to 50% compared to handwritten kernels and having negligible runtime overhead. Our results further show that mixed-precision performance depends not only on choosing appropriate data types, but also on tuning traditional optimization parameters (e.g., block size and vector width) and, when relevant, even domain-specific parameters.",
})
print(result.content)
