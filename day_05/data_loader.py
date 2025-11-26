import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    @staticmethod
    def load_docx(path):
        loader = Docx2txtLoader(path)
        docs = loader.load()
        return docs[0].page_content

    @staticmethod
    def recursive_split(text, chunk_size=600, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    @staticmethod
    def header_based_split(text):
        pattern = r"(?:\n|^)(\d{1,2}(?:\.\d{1,2})*\s+[^\n]+)\n"
        parts = re.split(pattern, text)
        if len(parts) <= 1:
            return [text]
        chunks = []
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            chunks.append(f"{title}\n{content}")
        return chunks

    @staticmethod
    def sliding_window_split(text, window_size=800, overlap=200):
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + window_size)
            chunks.append(text[start:end])
            start += (window_size - overlap)
        return chunks
