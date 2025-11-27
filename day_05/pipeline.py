# pipeline.py
import os
import glob
import json
import re
from config import CHUNKS_DIR_TEMPLATE, VECTORSTORE_DIR_TEMPLATE, OUTPUT_DIR, TOP_K_VECTOR, ALPHA, FINAL_K
from retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from query_processor import QueryProcessor

class RetrievalPipeline:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def run_query(self, query, vectorstore, bm25_retriever, chunks, multi_q=3, top_n=TOP_K_VECTOR, alpha=ALPHA):
        # 1) multi-query rewrite
        # 使用通用 QueryProcessor 进行 rewrite 与分类（Day05 要求的修改）
        # 0) Query processing (use QueryProcessor) - already present
        qp = QueryProcessor.process(query, self.llm_service, n_variants=multi_q)
        rewrites = qp.rewrites
        print("Query category:", qp.category)
        print("Rewrites:", rewrites)

        # Choose dynamic alpha mapping based on category
        category_to_alpha = {
            "numeric": 0.4,   # number-focused -> more BM25
            "definition": 0.8,
            "entity": 0.6,
            "open": 0.5
        }
        alpha_used = category_to_alpha.get(qp.category, alpha)
        print(f"Using alpha={alpha_used} for category={qp.category}")

        # 2) for each rewrite, run hybrid_retrieve and collect candidate indices with scores
        candidate_scores = {}
        for rq in rewrites:
            merged = HybridRetriever.retrieve(vectorstore, bm25_retriever, chunks, rq, top_n=top_n, alpha=alpha_used)
            ...
        # after building top_candidates...
        print("Candidates (pre-rerank):", top_candidates[:10])

        # 3) rerank top_candidates by LLM -> use rerank_v2
        try:
            reranked = self.llm_service.rerank_v2(query, top_candidates, chunks, top_k=FINAL_K)
        except Exception as e:
            print("rerank_v2 failed, falling back to original rerank. Err:", e)
            reranked = self.llm_service.rerank(query, top_candidates, chunks, top_k=FINAL_K)

        used_ids = [f"chunk_{idx}" for idx,_ in reranked]
        print("Reranked top:", reranked)
        return reranked, used_ids  # list of (idx, score_llm)

    def run_experiment(self, method_name, chunks, embeddings, queries):
        print("\n=== Running method:", method_name)
        chunks_dir = CHUNKS_DIR_TEMPLATE.format(method=method_name)
        vs_dir = VECTORSTORE_DIR_TEMPLATE.format(method=method_name)

        # ensure chunks saved
        if not os.path.exists(chunks_dir) or len(glob.glob(os.path.join(chunks_dir, "*.txt"))) == 0:
            # save chunks
            os.makedirs(chunks_dir, exist_ok=True)
            for i, c in enumerate(chunks):
                with open(os.path.join(chunks_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(c)

        # build/load vectorstore
        vectorstore = VectorRetriever.build_or_load(chunks, embeddings, vs_dir)
        # build bm25
        bm25_retriever = BM25Retriever(chunks)

        results = []
        for q in queries:
            print("\nQuery:", q)
            reranked, used_ids = self.run_query(q, vectorstore, bm25_retriever, chunks, multi_q=3, top_n=TOP_K_VECTOR, alpha=ALPHA)
            
            # generate answer
            parsed, raw_llm, _ = self.llm_service.generate_answer(q, reranked, chunks)
            
            results.append({
                "method": method_name,
                "query": q,
                "answer": parsed.get("answer", ""),
                "used_sources": used_ids,
                "raw_llm": raw_llm
            })
            # save per query
            safe_q = re.sub(r"[^\w\u4e00-\u9fff]+", "_", q)[:60]
            outpath = os.path.join(OUTPUT_DIR, f"res_{method_name}_{safe_q}.json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(results[-1], f, ensure_ascii=False, indent=2)
            print("Saved result to", outpath)
        return results
