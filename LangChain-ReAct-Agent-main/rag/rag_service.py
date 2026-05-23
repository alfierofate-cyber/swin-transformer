
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复

整体流程：
1. 用户输入问题
2. 通过向量检索器从知识库中召回相关文档
3. 将问题和检索到的参考资料拼接后送入 LLM
4. LLM 基于上下文生成总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


class RagSummarizeService:
    """RAG 总结服务：负责检索增强生成（Retrieval-Augmented Generation）的完整流程"""

    def __init__(self):
        # 初始化向量存储服务，用于管理和查询向量化的文档
        self.vector_store = VectorStoreService()
        # 从向量存储中获取检索器，用于根据查询召回相关文档
        self.retriever = self.vector_store.get_retriever()
        # 加载 RAG 专用的 prompt 模板文本
        self.prompt_text = load_rag_prompts()
        # 将文本转换为 LangChain 的 PromptTemplate 对象
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 使用工厂方法获取的聊天模型实例
        self.model = chat_model
        # 构建 LangChain 链：prompt -> 模型 -> 输出解析
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        构建 LCEL 链（LangChain Expression Language）
        流程：PromptTemplate 填充变量 -> ChatModel 生成回复 -> StrOutputParser 提取文本
        """
        return self.prompt_template | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        """
        根据用户查询从向量数据库中检索相��文档
        :param query: 用户的问题
        :return: 与问题语义相关的文档列表
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        RAG 总结的主流程方法
        :param query: 用户的问题
        :return: 模型基于检索到的参考资料生成的总结回复
        """
        # 第一步：从向量库中检索与问题相关的文档
        context_docs = self.retriever_docs(query)

        # 第二步：将检索到的文档格式化为上下文字符串
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            # 每条参考资料包含文档内容和元数据（如来源、页码等）
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 第三步：将用户问题和拼接好的上下文一起传入链，生成最终回复
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()
    # 测试：查询小户型适合的扫地机器人推荐
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
