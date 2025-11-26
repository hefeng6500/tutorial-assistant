import os
import re
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

try:
    import jieba

    HAVE_JIEBA = True
except Exception:
    HAVE_JIEBA = False


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [self._simple_tokenize(c) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _simple_tokenize(self, text):
        if HAVE_JIEBA:
            return list(jieba.cut_for_search(text))
        # fallback: split non-word chars and whitespace
        s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
        toks = [t for t in s.split() if t.strip()]
        return toks

    def retrieve(self, query, top_n=20):
        q_tokens = self._simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)  # list of scores per doc
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return ranked  # list of (idx, score)

    @staticmethod
    def normalize_scores(score_list):
        if not score_list:
            return []
        vals = [s for (_, s) in score_list]
        minv, maxv = min(vals), max(vals)
        if maxv - minv < 1e-12:
            # all equal
            return [(i, 1.0) for (i, s) in score_list]
        normed = [(i, (s - minv) / (maxv - minv)) for (i, s) in score_list]
        return normed


class VectorRetriever:
    @staticmethod
    def build_or_load(chunks, embeddings, persist_dir):
        if os.path.exists(persist_dir):
            try:
                vs = FAISS.load_local(
                    persist_dir, embeddings, allow_dangerous_deserialization=True
                )
                print(f"Loaded vectorstore from {persist_dir}")
                return vs
            except Exception as e:
                print("Failed to load existing vectorstore, rebuilding. Error:", e)
        vs = FAISS.from_texts(chunks, embeddings)
        vs.save_local(persist_dir)
        print(f"Built and saved vectorstore to {persist_dir}")
        return vs


class HybridRetriever:
    @staticmethod
    def retrieve(vectorstore, bm25_retriever, chunks, query, top_n=20, alpha=0.8):
        # 1) vector top N (get ids and vector scores)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_n})
        # LangChain v1 style: get_relevant_documents returns docs but not scores.
        # Instead we use vectorstore.similarity_search_with_score to get scores if available.
        try:
            vs_results = vectorstore.similarity_search_with_score(query, k=top_n)
            # vs_results: list of (Document, score)
            # But we need doc indices relative to chunks. We'll map by matching doc content:
            vec_candidates = []
            for doc, score in vs_results:
                text = doc.page_content
                # find index in chunks (best-effort)
                try:
                    idx = chunks.index(text)
                except ValueError:
                    # fallback: match by prefix
                    idx = None
                    for j, c in enumerate(chunks):
                        if text.strip() and text.strip()[:50] in c:
                            idx = j
                            break
                if idx is None:
                    continue
                vec_candidates.append((idx, float(score)))
        except Exception as e:
            # fallback to basic retriever without scores
            top_docs = retriever.invoke(query)
            vec_candidates = []
            for i, d in enumerate(top_docs):
                # try to find idx
                try:
                    idx = chunks.index(d.page_content)
                except ValueError:
                    idx = None
                    for j, c in enumerate(chunks):
                        if d.page_content.strip() and d.page_content.strip()[:50] in c:
                            idx = j
                            break
                if idx is not None:
                    vec_candidates.append((idx, 1.0))  # uniform score
        # normalize vector scores
        vec_norm = BM25Retriever.normalize_scores(vec_candidates)

        # 2) bm25 top N
        bm25_ranks = bm25_retriever.retrieve(
            query, top_n=top_n
        )  # returns list (idx, score)
        bm25_norm = BM25Retriever.normalize_scores(bm25_ranks)

        # 3) merge: create dict of idx -> combined score
        scores = {}
        for idx, s in vec_norm:
            scores[idx] = scores.get(idx, 0.0) + alpha * s
        for idx, s in bm25_norm:
            scores[idx] = scores.get(idx, 0.0) + (1 - alpha) * s

        # 4) sort and return top candidates
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return merged  # list of (idx, combined_score)
