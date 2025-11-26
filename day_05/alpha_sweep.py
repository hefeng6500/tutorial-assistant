# alpha_sweep.py
"""
Alpha sweep runner:
- Iterates over a list of alpha values
- For each alpha, uses your existing RetrievalPipeline to run experiments on a chosen chunking method
- Saves per-alpha outputs into OUTPUT_DIR/alpha_{alpha}/
- Produces a CSV summary with basic metrics (answer_len, num_source_chunks) for quick comparison.

Usage:
    python alpha_sweep.py
You can edit PARAMETERS below to change methods/queries/alphas.
"""

import os
import json
import re
import shutil
import csv
import glob
from config import OUTPUT_DIR, DOCX_PATH, CHUNKS_ROOT, CHUNKS_DIR_TEMPLATE, VECTORSTORE_DIR_TEMPLATE, TOP_K_VECTOR, FINAL_K
from data_loader import DocumentProcessor
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from llm_service import LLMService
from pipeline import RetrievalPipeline
from retrieval import BM25Retriever, VectorRetriever
from retriever_hub import RetrieverHub

# === PARAMETERS: adjust if needed ===
ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0]
METHOD = "header"   # run sweep on a single method (header/recursive/sliding). Set to "header" or others.
QUERIES = [
    "退出策略是什么？",
    "目标市场细分包括哪些？",
    "主要竞争对手有哪些？",
    "首年预计收入是多少？",
    "公司为何有竞争优势？"
]
EMBED_MODEL = "text-embedding-v4"
LLM_MODEL = os.getenv("LLM_MODEL") or "qwen3-max"
TEMPERATURE = 0.0

def run_alpha_sweep():
    full_text = DocumentProcessor.load_docx(DOCX_PATH)
    methods = {
        "recursive": lambda txt: DocumentProcessor.recursive_split(txt, chunk_size=600, chunk_overlap=100),
        "header": DocumentProcessor.header_based_split,
        "sliding": lambda txt: DocumentProcessor.sliding_window_split(txt, window_size=800, overlap=200),
    }
    splitter = methods[METHOD]
    chunks_dir = CHUNKS_DIR_TEMPLATE.format(method=METHOD)
    if os.path.exists(chunks_dir) and len(glob.glob(os.path.join(chunks_dir, "*.txt"))) > 0:
        chunks = [open(p, "r", encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(chunks_dir, "*.txt")))]
    else:
        chunks = splitter(full_text)
        os.makedirs(chunks_dir, exist_ok=True)
        for i,c in enumerate(chunks):
            with open(os.path.join(chunks_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(c)

    embeddings = DashScopeEmbeddings(model=EMBED_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                     base_url=os.getenv("DASHSCOPE_BASE_URL"), api_key=os.getenv("DASHSCOPE_API_KEY"))
    llm_service = LLMService(llm)

    # build base vectorstore once (we will reload it per alpha via VectorRetriever.build_or_load)
    vs_dir_template = VECTORSTORE_DIR_TEMPLATE.format(method=METHOD)
    vectorstore = VectorRetriever.build_or_load(chunks, embeddings, vs_dir_template)

    bm25_retriever = BM25Retriever(chunks)
    hub = RetrieverHub(vectorstore, bm25_retriever, chunks)
    pipeline = RetrievalPipeline(llm_service)

    results_summary = []
    for alpha in ALPHAS:
        print(f"\n=== Running alpha={alpha} ===")
        out_dir_alpha = os.path.join(OUTPUT_DIR, f"alpha_{alpha}")
        # clear old outputs for this alpha
        if os.path.exists(out_dir_alpha):
            shutil.rmtree(out_dir_alpha)
        os.makedirs(out_dir_alpha, exist_ok=True)

        # For each query, run a single retrieval + rerank + generate using current alpha
        for q in QUERIES:
            # use hub to get candidates (top TOP_K_VECTOR)
            candidates = hub.retrieve(q, top_n=TOP_K_VECTOR, mode="hybrid", alpha=alpha)
            # We reuse pipeline.rerank + pipeline.generate_answer via llm_service
            try:
                reranked = llm_service.rerank(q, candidates, chunks, top_k=FINAL_K)
            except Exception:
                reranked = candidates[:FINAL_K]
            parsed, raw_llm, _ = llm_service.generate_answer(q, reranked, chunks)

            # save per query
            safe_q = re.sub(r"[^\w\u4e00-\u9fff]+", "_", q)[:60]
            outpath = os.path.join(out_dir_alpha, f"res_alpha_{alpha}_{safe_q}.json")
            rec = {
                "alpha": alpha,
                "query": q,
                "candidates_pre_rerank": [(int(i), float(s)) for (i,s) in candidates[:TOP_K_VECTOR]],
                "reranked": [(int(i), float(s)) for (i,s) in reranked],
                "answer": parsed.get("answer","") if isinstance(parsed, dict) else str(parsed),
                "raw_llm": raw_llm
            }
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            results_summary.append({
                "alpha": alpha,
                "query": q,

                # JSON stringify all complex fields
                "candidates_pre_rerank": json.dumps(rec["candidates_pre_rerank"], ensure_ascii=False),
                "reranked": json.dumps(rec["reranked"], ensure_ascii=False),
                "answer": rec["answer"],
                "raw_llm": rec["raw_llm"],
                
                # optional sources (if your generator outputs them)
                "used_sources": json.dumps(parsed.get("used_sources", []), ensure_ascii=False)
                                if isinstance(parsed, dict) else "[]",

                # numeric metrics
                "num_candidates": len(rec["candidates_pre_rerank"]),
                "answer_len": len(rec["answer"]),
            })

    # write summary csv
    csv_path = os.path.join(OUTPUT_DIR, f"alpha_sweep_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "query",
                "candidates_pre_rerank",
                "reranked",
                "answer",
                "raw_llm",
                "used_sources",
                "num_candidates",
                "answer_len"
            ]
        )
        writer.writeheader()
        for row in results_summary:
            writer.writerow(row)

    print("Alpha sweep done. Summary:", csv_path)



if __name__ == "__main__":
    run_alpha_sweep()
