from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from utils.logger_handler import logger


# 模块级标志位，用于动态切换提示词（由 fill_context_for_report 工具触发）
_report_mode = False


def set_report_mode():
    """将标志位设为 True，使 Agent 在下一次调用模型时使用报告生成提示词"""
    global _report_mode
    _report_mode = True


class ReactAgent:
    """ReAct Agent 核心类：大模型 + 工具列表 + 循环推理"""

    def __init__(self):
        # 注册 Agent 可使用的所有工具
        self.tools = [
            rag_summarize,           # RAG 知识库检索
            get_weather,             # 查天气
            get_user_location,       # 获取用户所在城市
            get_user_id,             # 获取用户 ID
            get_current_month,       # 获取当前月份
            fetch_external_data,     # 查询用户历史使用数据
            fill_context_for_report, # 触发切换为报告生成模式
        ]

        # 创建 ReAct Agent：模型负责思考和决策，工具负责执行具体操作
        self.agent = create_react_agent(
            model=chat_model,                    # 通义千问对话模型
            tools=self.tools,                    # Agent 可调用的工具
            state_modifier=self._state_modifier, # 每次调用模型前修改状态（注入系统提示词）
        )

    def _state_modifier(self, state):
        """每次模型推理前调用，根据当前模式动态选择系统提示词"""
        global _report_mode
        if _report_mode:
            # 报告生成模式：使用报告专用提示词
            system_prompt = load_report_prompts()
        else:
            # 普通模式：使用客服提示词
            system_prompt = load_system_prompts()

        # 把系统提示词插到消息列表最前面
        messages = state.get("messages", [])
        system_msg = SystemMessage(content=system_prompt)
        return [system_msg] + list(messages)

    def execute_stream(self, query: str):
        """
        流式执行 Agent 推理循环，逐步 yield 模型输出内容。
        整个过程：用户提问 → 模型思考 → 调工具 → 再思考 → ... → 最终回答
        """
        global _report_mode
        _report_mode = False  # 每次新查询重置模式，防止上次的状态残留

        logger.info(f"[log_before_model]收到用户查询：{query}")

        # 构造输入消息
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 流式获取 Agent 每一步的输出（包括工具调用和最终回答）
        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            messages = chunk.get("messages", [])
            if not messages:
                continue

            # 取最新一条消息（可能是模型回复、工具调用、或工具返回结果）
            latest_message = messages[-1]

            # 如果模型决定调用工具，记录日志
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    logger.info(f"[tool monitor]执行工具：{tc['name']}")
                    logger.info(f"[tool monitor]传入参数：{tc['args']}")

            # 如果是工具返回的结果，记录日志
            if latest_message.type == "tool":
                logger.info(f"[tool monitor]工具执行完成")

            # 有文本内容就 yield 给调用方（最终会显示在前端）
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
