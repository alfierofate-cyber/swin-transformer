# test_agent 使用说明

`test_agent` 目录里放的是一个知识库问答示例项目：

- 使用 Streamlit 提供网页界面
- 使用 LangChain 组织 RAG 问答流程
- 使用 Chroma 在本地保存向量数据库
- 使用阿里云 DashScope 的 `text-embedding-v4` 做文本向量化
- 使用通义千问 `qwen3-max` 做聊天回答

实际代码目录是：

```text
test_agent/KnowledgeBase-RAG-LLM-System-master/
```

## 目录结构

```text
KnowledgeBase-RAG-LLM-System-master/
├── app_upload.py          # 知识库上传页面，只支持 txt 文件
├── app_chat.py            # 智能客服聊天页面
├── knowledge_base.py      # 文本切分、向量化、写入 Chroma
├── rag.py                 # RAG 检索问答链
├── vector_stores.py       # Chroma 向量库封装
├── file_history_store.py  # 本地聊天历史保存
├── config_data.py         # 模型名、向量库路径、切分参数配置
├── requirements.txt       # 依赖列表
├── assets/                # 示例 txt 文档和演示图片
├── chroma_db/             # 本地向量数据库
└── chat_history/          # 本地聊天记录
```

## 运行前准备

先进入项目目录。这个步骤很重要，因为代码里使用了 `./chroma_db`、`./chat_history` 这样的相对路径。

```bash
cd /Users/yidansu/Desktop/something/practice/swin-transformer/test_agent/KnowledgeBase-RAG-LLM-System-master
```

当前电脑的 Anaconda Python 已经安装过主要依赖。如果换到新环境，可以安装：

```bash
/Users/yidansu/anaconda3/bin/python -m pip install streamlit langchain-chroma langchain-community langchain-text-splitters dashscope chromadb
```

还需要配置 DashScope API Key，否则调用通义千问和 embedding 时会报认证错误：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
```

如果想每次打开终端都自动生效，可以把上面这一行加入 `~/.zshrc`。

## 启动知识库上传页面

```bash
/Users/yidansu/anaconda3/bin/python -m streamlit run app_upload.py --server.port 8501
```

浏览器打开：

```text
http://localhost:8501
```

页面会让你上传 `.txt` 文件。上传后，程序会：

1. 读取 txt 文本
2. 按 `config_data.py` 的规则切分文本
3. 调用 `text-embedding-v4` 生成向量
4. 写入本地 `chroma_db/`
5. 用 `md5.text` 记录文件内容指纹，避免重复入库

可以先用 `assets/` 里面的示例文件测试，例如：

```text
assets/尺码推荐.txt
assets/洗涤养护.txt
assets/颜色推荐.txt
```

## 启动聊天页面

另开一个终端，同样先进入项目目录：

```bash
cd /Users/yidansu/Desktop/something/practice/swin-transformer/test_agent/KnowledgeBase-RAG-LLM-System-master
```

启动聊天页面：

```bash
/Users/yidansu/anaconda3/bin/python -m streamlit run app_chat.py --server.port 8502
```

浏览器打开：

```text
http://localhost:8502
```

聊天时，程序会先从 `chroma_db/` 检索相关文档片段，再把检索结果和历史对话一起交给 `qwen3-max` 生成回答。

## 常用配置

主要配置在 `config_data.py`：

```python
collection_name = "rag"
persist_directory = "./chroma_db"
chunk_size = 1000
chunk_overlap = 100
similarity_threshold = 1
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"
```

常改的地方：

- `chunk_size`：每个文本块最大长度
- `chunk_overlap`：相邻文本块重叠长度
- `similarity_threshold`：每次检索返回的文档片段数量
- `chat_model_name`：聊天模型
- `embedding_model_name`：向量模型

## 直接运行脚本测试

测试写入知识库：

```bash
/Users/yidansu/anaconda3/bin/python knowledge_base.py
```

测试 RAG 问答：

```bash
/Users/yidansu/anaconda3/bin/python rag.py
```

实际使用时更推荐通过 Streamlit 页面运行。

## 重置本地数据

如果想清空向量库和上传去重记录，可以先停止 Streamlit，然后在项目目录下执行：

```bash
rm -rf chroma_db md5.text
```

如果只想清空聊天历史：

```bash
rm -rf chat_history/user_001
```

下一次运行时，程序会重新创建这些文件或目录。

## 常见问题

### No module named 'dashscope'

说明当前 Python 环境没有安装 DashScope SDK：

```bash
/Users/yidansu/anaconda3/bin/python -m pip install dashscope
```

### API Key 或认证错误

检查是否已经配置：

```bash
echo $DASHSCOPE_API_KEY
```

如果没有输出，就重新执行：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
```

### VS Code 有黄色波浪线，但终端能运行

通常是 VS Code 选择的 Python 解释器和终端运行的解释器不一致。这个项目当前使用的是：

```text
/Users/yidansu/anaconda3/bin/python
```

### 问答没有参考资料

先打开 `app_upload.py` 上传 txt 文档，确认知识已经写入 `chroma_db/`，再打开 `app_chat.py` 提问。

