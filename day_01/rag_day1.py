import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import WebBaseLoader, TextLoader

# 3. 生成 Embedding 的向量 Embed
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

if os.path.exists("../datas/vectorstore"):
    print("Loading vectorstore from datas/vectorstore")
    vectorstore = FAISS.load_local("../datas/vectorstore", embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new vectorstore")
    # 1. 加载数据 Load
    loader = TextLoader("../datas/古文.md", encoding="utf-8")
    text = loader.load()

    # 2. 切分文档 Split（Chunking）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=20)
    docs = text_splitter.split_text(text[0].page_content)

    # Save chunks
    if not os.path.exists("../datas/chunks"):
        os.makedirs("../datas/chunks")
    for i, doc in enumerate(docs):
        with open(f"../datas/chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(doc)

    print(f"Saved {len(docs)} chunks to datas/chunks")

    # 4. 建立向量库 + 检索 Retrieve
    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local("../datas/vectorstore")
    print("Saved vectorstore to datas/vectorstore")

retriever = vectorstore.as_retriever()

# 5. 用上下文生成答案 Generate

prompt = """
  你是一个 AI 助手，你需要根据提供的上下文回答问题。
  请勿使用任何外部信息。
  请使用中文回答。
  上下文: {context}
  问题: {question}
"""

query = "长江赋总结一下，哪句写的最好？"

temp_docs = retriever.invoke(query)

print(len(temp_docs))

context = "\n".join([doc.page_content for doc in temp_docs])

print(context)

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.3,
)

result = llm.invoke(prompt.format(context=context, question=query))

result.pretty_print()
