from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 加载数据 Load
text = """
LangChain 是构建由 LLMs 驱动的代理和应用程序的最简单方法。只需不到 10 行代码，您就可以连接到 OpenAI、Anthropic、Google 和 更多 。LangChain 提供了预构建的代理架构和模型集成，帮助您快速入门，并无缝地将 LLMs 融入您的代理和应用程序中。
如果您想快速构建代理和自主应用程序，我们建议您使用 LangChain。当您有更高级的需求，需要结合确定性和代理工作流、重度自定义以及精确控制延迟时，请使用 LangGraph，我们的低级代理编排框架和运行时。
LangChain 代理 是基于 LangGraph 构建的，以提供持久执行、流式处理、人机协作、持久性等功能。您不需要了解 LangGraph 就可以基本使用 LangChain 代理。
"""

# 2. 切分文档 Split（Chunking）


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_text(text)

# print("切分后的 chunks：", docs)

# 3. 生成 Embedding 的向量 Embed

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

# 4. 建立向量库 + 检索 Retrieve
vectorstore = FAISS.from_texts(docs, embeddings)

retriever = vectorstore.as_retriever()

# 5. 用上下文生成答案 Generate

prompt = """
  你是一个 AI 助手，你需要根据提供的上下文回答问题。
  请勿使用任何外部信息。
  请使用中文回答。
  上下文: {context}
  问题: {question}
"""

query = "什么是 LangChain?"

temp_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in temp_docs])


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.3,
)

result = llm.invoke(prompt.format(context=context, question=query))

result.pretty_print()



# ================================== Ai Message ==================================

# LangChain 是构建由大型语言模型（LLMs）驱动的代理和应用程序的最简单方法。它的特点包括：

# - 只需不到 10 行代码即可连接到 OpenAI、Anthropic、Google 等多家提供商。
# - 提供预构建的代理架构和模型集成，帮助你快速入门，并将 LLMs 无缝融入你的代理和应用程序。
# - LangChain 代理是基于 LangGraph 构建的，具备持久执行、流式处理、人机协作、持久性等能力；但你不需要了解 LangGraph 也能基本使用 LangChain 代理。
# - 如果你想快速构建代理和自主应用，我们建议使用 LangChain。
# - 当你有更高级的需求，需要结合确定性和代理工作流、重度自定义以及对延迟的精确控制时，文段中有对应的建议（未给出后续具体信息）。
