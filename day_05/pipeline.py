# pipeline.py
import os
import glob
import json
import re
from config import CHUNKS_DIR_TEMPLATE, VECTORSTORE_DIR_TEMPLATE, OUTPUT_DIR, TOP_K_VECTOR, ALPHA, FINAL_K
from retrieval import BM25Retriever, VectorRetriever, HybridRetriever

class RetrievalPipeline:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def run_query(self, query, vectorstore, bm25_retriever, chunks, multi_q=3, top_n=TOP_K_VECTOR, alpha=ALPHA):
        # 1) multi-query rewrite
        rewrites = self.llm_service.rewrite_query(query, n_variants=multi_q)
        rewrites = list(dict.fromkeys([r.strip() for r in rewrites if r.strip()]))[:multi_q]
        print("Rewrites:", rewrites)

        # Optionally: you may classify query and pick dynamic alpha here (if you have a QueryProcessor)
        # For backward compatibility we use provided alpha param.
        alpha_used = alpha
        print(f"Using alpha={alpha_used}")

        # 2) for each rewrite, run hybrid_retrieve and collect candidate indices with scores
        candidate_scores = {}  # idx -> max_score
        for rq in rewrites:
            merged = HybridRetriever.retrieve(vectorstore, bm25_retriever, chunks, rq, top_n=top_n, alpha=alpha_used)
            for idx, sc in merged:
                if idx is None:
                    continue
                if idx not in candidate_scores or sc > candidate_scores[idx]:
                    candidate_scores[idx] = sc
        merged_list = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = merged_list[:top_n]
        print("Candidates (pre-rerank):", top_candidates[:10])

        # 3) rerank top_candidates by LLM
        try:
            reranked = self.llm_service.rerank_v2(query, top_candidates, chunks, top_k=FINAL_K)
        except Exception as e:
            print("rerank_v2 failed, falling back to original rerank. Err:", e)
            reranked = self.llm_service.rerank(query, top_candidates, chunks, top_k=FINAL_K)

        used_ids = [f"chunk_{idx}" for idx, _ in reranked]
        print("Reranked top:", reranked)

        # --------------------------
        # Day8: Context Optimization
        # --------------------------
        try:
            from context_optimizer import ContextOptimizer
            context_optimizer = ContextOptimizer()
            # optimized_context is a string (ready to pass to generate_answer)
            optimized_context = context_optimizer.process(query, reranked, chunks)
        except Exception as _ctx_e:
            # fallback: pass reranked (list) to generate_answer for backward compatibility
            optimized_context = reranked

        # 4) generate answer (accepts optimized_context string or reranked list)
        parsed, raw_llm, final_context = self.llm_service.generate_answer(query, optimized_context, chunks)

        return reranked, used_ids, parsed, raw_llm, final_context

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
            reranked, used_ids, parsed, raw_llm, final_context = self.run_query(q, vectorstore, bm25_retriever, chunks, multi_q=3, top_n=TOP_K_VECTOR, alpha=ALPHA)
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
