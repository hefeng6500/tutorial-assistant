import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader

def load_data(file_path):
    """Load data from a file."""
    loader = Docx2txtLoader(file_path)
    text = loader.load()
    return text

def split_text(text, chunk_size=1200, chunk_overlap=60):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text)
    return docs

# 根据商业计划书常见标题格式自动切分
def header_based_split(text):
    """
    根据商业计划书常见标题格式自动切分：
    例如：1. xxx / 2.1 xxx / 3.5 SWOT 分析
    """

    # Python 内置的正则表达式（Regular Expression）模块
    import re

    # 模式：数字开头，后面跟句点或空格
    pattern = r"(?:\n|^)(\d{1,2}\.\d{0,2}.*?)\n"

    sections = re.split(pattern, text)
    chunks = []

    # sections 格式类似：[前导文本, 标题1, 内容1, 标题2, 内容2 ...]
    if len(sections) <= 1: 
        return [text]  # 没识别到标题，返回原文

    current_title = None

    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        content = sections[i+1].strip()
        chunk = f"{title}\n{content}"
        chunks.append(chunk)

    return chunks

# Sliding Window Chunker（窗口式），适合长段叙事类文档，解决 Recursive“句子断层”的问题，检索更稳定，召回率更高
def sliding_window_split(text, window_size=800, overlap=200):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (window_size - overlap)

    return chunks


def save_chunks(docs, output_dir="../datas/chunks"):
    """Save chunks to files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, doc in enumerate(docs):
        with open(os.path.join(output_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(doc)
    print(f"Saved {len(docs)} chunks to {output_dir}")

def get_embeddings():
    """Initialize OpenAI embeddings."""
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

def get_vectorstore(docs, embeddings, persist_dir="../datas/vectorstore"):
    """Load or create/save vector store."""
    if os.path.exists(persist_dir):
        print(f"Loading vectorstore from {persist_dir}")
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vectorstore")
        vectorstore = FAISS.from_texts(docs, embeddings)
        vectorstore.save_local(persist_dir)
        print(f"Saved vectorstore to {persist_dir}")
    return vectorstore

def query_llm(retriever, query):
    """Query the LLM with retrieved context."""
    prompt = """
      你是一个 AI 助手，你需要根据提供的上下文回答问题。
      请勿使用任何外部信息。
      请使用中文回答。
      上下文: {context}
      问题: {question}

      请根据上下文回答问题，如果上下文无法回答，请不要胡编乱造。请返回"上下文无法回答"。
    """
    
    temp_docs = retriever.invoke(query)
    print(f"Retrieved {len(temp_docs)} documents")
    
    context = "\n".join([doc.page_content for doc in temp_docs])
    # print(f"Context preview: {context[:200]}...")
    
    llm = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0,
    )
    
    result = llm.invoke(prompt.format(context=context, question=query))
    return result

def main():
    # Configuration
    data_file = "../datas/商业计划书.docx"
    chunks_dir = "../datas/chunks"
    vectorstore_dir = "../datas/vectorstore"
    query = "帮我计算一下，代理亿道的产品的成本？"

    # 1. Load Data (Only needed if vectorstore doesn't exist, but for simplicity we load to split if needed)
    # Optimization: Check if vectorstore exists first to avoid loading/splitting if not needed?
    # The original logic loaded/split only if vectorstore didn't exist. Let's preserve that logic.

    embeddings = get_embeddings()
    
    if os.path.exists(vectorstore_dir):
        vectorstore = get_vectorstore(None, embeddings, vectorstore_dir)
    else:
        text = load_data(data_file)

        # 2. Split Text
        # docs = split_text(text[0].page_content)

        # 根据标题切分
        # docs = header_based_split(text[0].page_content)

        # 窗口式切分
        docs = sliding_window_split(text[0].page_content)
        save_chunks(docs, chunks_dir)

        # 3. embedding
        vectorstore = get_vectorstore(docs, embeddings, vectorstore_dir)

    # 4. Retrieve
    retriever = vectorstore.as_retriever()

    # 5. Generate
    result = query_llm(retriever, query)
    result.pretty_print()

if __name__ == "__main__":
    main()
