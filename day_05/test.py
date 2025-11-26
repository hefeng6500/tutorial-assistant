from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

chunks = ["苹果是一种水果", "香蕉富含钾元素", "橙子很酸"]

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
)

text = "今天天气真好！"
vector = embeddings.embed_query(text)

print(f"文本: {text}")
print(f"向量维度: {len(vector)}")
print(f"前10个值: {vector[:10]}")
