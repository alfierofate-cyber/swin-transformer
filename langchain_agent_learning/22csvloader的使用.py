from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="./data/iris.csv",
                   csv_args={"delimiter": "|",
                             "quotechar": '"',
                             "fieldnames": ["1", "2", "3", "4", "5"]},
                 )  # csv_args 传递给 csv.reader 的参数

# documents = loader.load()  

# for document in documents:
#     print(type(document),document)

#懒加载 .lazy_load() 迭代器[Document]
for document in loader.lazy_load():
    print(type(document),document)

