import os,json
from langchain_core.messages import message_to_dict,messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Sequence
from langchain_core.messages import BaseMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


#message_to_dict:将消息对象转换为字典
#messages_from_dict:将字典转换为消息对象列表
#AIMessage、SystemMessage\HumanMessage等消息对象都继承自BaseChatMessage类，BaseChatMessage类是所有消息对象的基类，提供了消息对象的基本属性和方法

class FileChatMessageHistory(BaseChatMessageHistory): 
    def __init__(self, session_id, storage_path):
        self.session_id = session_id #会话id，代表一个会话的标识，程序会根据这个session_id自动管理对应的历史消息
        self.storage_path = storage_path #存储路径，历史消息会保存在这个路径下的一个文件中，文件名可以根据session_id来命名，确保不同会话的历史消息保存在不同的文件中
        os.makedirs(self.storage_path, exist_ok=True)
        legacy_dir_path = os.path.join(self.storage_path, self.session_id)

        # 兼容旧写法：如果历史上把 session_id 建成了目录，就在目录里继续存 json 文件
        if os.path.isdir(legacy_dir_path):
            self.file_path = os.path.join(legacy_dir_path, "messages.json")
        else:
            self.file_path = os.path.join(self.storage_path, self.session_id + ".json")

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        #Sequence序列 类似于list或者tuple，表示消息对象的列表
        all_messages =  list(self.messages)#已有的消息列表
        all_messages.extend(messages)#将新的消息添加到已有的消息列表中

        #将数据同步写入到本地文件中
        #类对象写入文件--》一堆二进制
        #为了方便，可以将BaseMessage对象转换为字典对象，再将字典对象转换为json字符串，最后将json字符串写入文件中
        #官方message_to_dict:单个消息对象（BaseMessage）转换为字典）
        #new_messages = []
        #for message in all_messages:
        #    d = message_to_dict(message)
        #    new_messages.append(d)

        new_messages = [message_to_dict(message) for message in all_messages]#列表推导式，效果同上句
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)
    @property #@property装饰器将一个方法转换为属性，使得调用这个方法时不需要加括号，可以直接通过属性的方式访问
    def messages(self) -> List[BaseMessage]:
        #当前文件内：list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []


    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)


model = ChatTongyi(
    model="qwen3-max",
)

#prompt = PromptTemplate.from_template("你需要根据会话历史回应用户的问题，历史会话如下：{history}，请根据历史会话内容回答用户的问题：{input}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回应用户的问题，历史会话如下："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答如下问题：{input}")
    ]
)

str_parser = StrOutputParser()

def print_prompt(full_prompt):
    print("="*20, full_prompt, "="*20)
    return full_prompt

base_chain = prompt | print_prompt | model | str_parser


#key就是session,value就是InMemoryChatMessageHistory对象
#实现通过会话id保存InMemoryChatMessageHistory类对象

#实现通过会话id获取InMemoryChatMessageHistory类对象
def get_history(session_id):
    return FileChatMessageHistory(session_id, storage_path="./chat_history")

#创建一个新的链，对原有链增强功能：自动附加历史消息

conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
#固定格式，添加langchain的配置，为当前程序配置所属的session_id
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }#session_id可以是任意字符串，代表一个会话的标识，程序会根据这个session_id自动管理对应的历史消息
    }
    # res = conversation_chain.invoke({ "input": "小明有2个猫"}, session_config)
    # print("第一次执行：",res)

    # res = conversation_chain.invoke({ "input": "小刚有2个猫"}, session_config)
    # print("第二次执行：",res)

    res = conversation_chain.invoke({ "input": "总共有几个宠物"}, session_config)
    print("第三次执行：",res)
