from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

#Chroma向量数据库（轻量级）
#

vector_store = Chroma(
    collection_name="test",
    embedding_function=DashScopeEmbeddings(),
    persist_directory="./chroma_db"
)

loader = CSVLoader(file_path="./data/info.csv",
                   source_column="source",
                   encoding="utf-8",
                   csv_args={"delimiter": ","}
                 )  # csv_args 传递给 csv.reader 的参数

documents = loader.load()  # 加载文档

# 向量存储的新增、删除、检索
vector_store.add_documents(
    documents = documents,
    ids = ["id" + str(i) for i in range(1, len(documents) + 1)]
    )  # 新增文档

#删除 传入[id1, id2]
vector_store.delete(ids=["id1", "id2"])

#检索 传入query和top_k
result = vector_store.similarity_search(
    "python是不是简单易学呀", 
    2
)

print(result)

