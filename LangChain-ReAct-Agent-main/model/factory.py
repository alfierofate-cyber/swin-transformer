from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


# 抽象工厂基类，定义统一接口，所有模型工厂都要实现 generator() 方法
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


# 对话模型工厂，负责创建通义千问聊天模型（用于理解问题、生成回答）
class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        # 从配置文件读取模型名（如 qwen3-max），创建对话模型实例
        return ChatTongyi(model=rag_conf["chat_model_name"])


# 向量化模型工厂，负责创建文本嵌入模型（用于把文字转成向量，做相似度检索）
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 从配置文件读取模型名（如 text-embedding-v4），创建向量化模型实例
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 创建全局模型实例，其他模块直接导入使用即可
chat_model = ChatModelFactory().generator()    # 对话模型，用于和用户聊天
embed_model = EmbeddingsFactory().generator()  # 向量模型，用于文档检索
