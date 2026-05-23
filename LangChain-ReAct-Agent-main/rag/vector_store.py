from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf

from model.factory import embed_model

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger

import os


class VectorStoreService:
    """向量数据库服务：负责文档入库和相似度检索"""

    def __init__(self):
        # 初始化 Chroma 向量数据库连接
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],      # 集合名称
            embedding_function=embed_model,                      # 用哪个模型把文字转向量
            persist_directory=chroma_conf["persist_directory"],  # 数据持久化到本地的目录
        )

        # 文档分块器：把长文档切成小段，便于精确检索
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],       # 每块最大 200 字符
            chunk_overlap=chroma_conf["chunk_overlap"], # 相邻块重叠 20 字符，避免语义断裂
            separators=chroma_conf["separators"],       # 优先按段落、句号等自然边界切分
            length_function=len,
        )

    def get_retriever(self):
        """返回检索器，查询时会找出最相关的 k 条文档片段"""
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        扫描数据文件夹，把 txt/pdf 文件转为向量存入数据库。
        用 MD5 做去重：同一个文件内容没变就不会重复入库。
        """

        def check_md5_hex(md5_for_check: str):
            """检查该文件的 MD5 是否已记录过（已入库则返回 True）"""
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # MD5 记录文件不存在，创建空文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # 已处理过，跳过

                return False            # 未处理过，需要入库

        def save_md5_hex(md5_for_check: str):
            """将已入库文件的 MD5 追加记录，下次不再重复处理"""
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """根据文件后缀选择对应的加载器，返回文档对象列表"""
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        # 获取数据目录下所有允许类型的文件路径
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 逐个文件处理
        for path in allowed_files_path:
            # 计算文件 MD5，用于判断是否已入库
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                # 1. 读取文件内容为 Document 对象
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # 2. 把文档切成小块
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 3. 分批写入向量库（DashScope API 限制每次最多 10 条）
                batch_size = 10
                for i in range(0, len(split_document), batch_size):
                    batch = split_document[i:i + batch_size]
                    self.vector_store.add_documents(batch)

                # 4. 记录 MD5，标记该文件已处理
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info=True 会记录详细的报错堆栈
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    # 加载数据目录下的文档到向量库
    vs.load_document()

    # 测试检索：查询"迷路"相关的文档片段
    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)
