# retriever_hub.py
"""
RetrieverHub: minimal wrapper around existing retrieval implementations.

接口：
    hub = RetrieverHub(vectorstore, bm25_retriever, chunks)
    candidates = hub.retrieve(processed_query, top_n=20, mode="hybrid")
返回：
    list of (idx, score)  —— 与现有 HybridRetriever/ BM25 保持相同格式
注意：processed_query 可以是字符串或你 Day5 的 ProcessedQuery 对象（带 .rewrites）
"""

from typing import List, Tuple, Union
from retrieval import BM25Retriever, VectorRetriever, HybridRetriever

class RetrieverHub:
    def __init__(self, vectorstore, bm25_retriever, chunks):
        """
        vectorstore: VectorRetriever.build_or_load(...) 返回的对象
        bm25_retriever: BM25Retriever(chunks) 实例
        chunks: list of chunk texts
        """
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.chunks = chunks

    def _handle_rewrites(self, rewrites: List[str], top_n: int, alpha: float, mode: str):
        """
        For a list of rewrite queries, run the appropriate retrieval strategy and
        merge results (take max score per doc).
        """
        candidate_scores = {}
        for q in rewrites:
            if mode == "bm25":
                ranks = self.bm25_retriever.retrieve(q, top_n=top_n)
                # ranks is list of (idx, score)
                for idx, sc in ranks:
                    if idx is None:
                        continue
                    if idx not in candidate_scores or sc > candidate_scores[idx]:
                        candidate_scores[idx] = float(sc)
            elif mode == "vector":
                try:
                    vs_results = self.vectorstore.similarity_search_with_score(q, k=top_n)
                    for doc, score in vs_results:
                        text = doc.page_content
                        # best-effort map to chunks
                        try:
                            idx = self.chunks.index(text)
                        except ValueError:
                            idx = None
                            for j, c in enumerate(self.chunks):
                                if text.strip() and text.strip()[:50] in c:
                                    idx = j
                                    break
                        if idx is None:
                            continue
                        if idx not in candidate_scores or float(score) > candidate_scores[idx]:
                            candidate_scores[idx] = float(score)
                except Exception:
                    # fallback: use vectorstore.as_retriever
                    retr = self.vectorstore.as_retriever(search_kwargs={"k": top_n})
                    docs = retr.get_relevant_documents(q)
                    for d in docs:
                        try:
                            idx = self.chunks.index(d.page_content)
                        except ValueError:
                            idx = None
                            for j,c in enumerate(self.chunks):
                                if d.page_content.strip() and d.page_content.strip()[:50] in c:
                                    idx = j
                                    break
                        if idx is not None:
                            candidate_scores[idx] = max(candidate_scores.get(idx, 0.0), 1.0)
            else:
                # hybrid
                merged = HybridRetriever.retrieve(self.vectorstore, self.bm25_retriever, self.chunks, q, top_n=top_n, alpha=alpha)
                for idx, sc in merged:
                    if idx is None:
                        continue
                    if idx not in candidate_scores or sc > candidate_scores[idx]:
                        candidate_scores[idx] = float(sc)

        merged_list = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return merged_list

    def retrieve(self, processed_query: Union[str, object], top_n: int = 20, mode: str = "hybrid", alpha: float = 0.8):
        """
        processed_query: either a string or ProcessedQuery (with .rewrites)
        mode: "hybrid" (default), "bm25", "vector"
        """
        if isinstance(processed_query, str):
            rewrites = [processed_query]
        else:
            rewrites = getattr(processed_query, "rewrites", None) or [getattr(processed_query, "original", str(processed_query))]

        return self._handle_rewrites(rewrites, top_n=top_n, alpha=alpha, mode=mode)
