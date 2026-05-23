from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

@tool(description="获取股价，传入股票名称，返回字符串信息")
def get_price(name:str) -> str:
    return f"{name}的股价是100元"

@tool(description="获取股票信息，传入股票名称，返回字符串信息")
def get_info(name:str) -> str:
    return f"{name}的股票信息是：市盈率10，市值100亿"



agent = create_agent(
    model = ChatTongyi(model = "qwen3-max"),
    tools = [get_info,get_price],
    system_prompt = "你是一个有用的助手，可以回答股票相关的问题，记住请告知我思考过程，让我知道你为什么调用某个工具"
)

for chunk in agent.stream(
    input={"messages":[
        {"role":"user","content":"请告诉我一下腾讯的股价和股票信息"}
    ]},
    stream_mode="values"
):
    latest_message = chunk['messages'][-1]
    
    if latest_message.content:
        print(type(latest_message.content).__name__,latest_message.content)

    try:
        if latest_message.tool_calls:
            print(f"工具调用: { [tc['name'] for tc in latest_message.tool_calls] }")    

    except AttributeError as e:
        pass
