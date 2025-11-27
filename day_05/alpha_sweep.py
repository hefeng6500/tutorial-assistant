# alpha_sweep.py
import os
import shutil
import csv
import glob
import json
import re
from config import OUTPUT_DIR, DOCX_PATH, CHUNKS_ROOT, CHUNKS_DIR_TEMPLATE, VECTORSTORE_DIR_TEMPLATE, TOP_K_VECTOR, FINAL_K
from data_loader import DocumentProcessor
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from llm_service import LLMService
from pipeline import RetrievalPipeline
from retrieval import BM25Retriever, VectorRetriever
from retriever_hub import RetrieverHub
from results_exporter import write_longform_excel, write_per_method_csv, write_summary_csv

# === PARAMETERS: adjust if needed ===
ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0]
METHODS = ["header", "recursive", "sliding"]   # 多个 method
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
WRITE_PER_METHOD_CSV = True  # also write per-method CSVs
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "alpha_sweep_results.xlsx")
OUTPUT_PER_METHOD_DIR = os.path.join(OUTPUT_DIR, "alpha_per_method_csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "alpha_sweep_summary.csv")

def run_alpha_sweep():
    full_text = DocumentProcessor.load_docx(DOCX_PATH)

    embeddings = DashScopeEmbeddings(model=EMBED_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                     base_url=os.getenv("DASHSCOPE_BASE_URL"), api_key=os.getenv("DASHSCOPE_API_KEY"))
    llm_service = LLMService(llm)

    longform_rows = []  # list of dicts (each is one candidate row)

    for METHOD in METHODS:
        print(f"\n=== Preparing method: {METHOD} ===")
        methods_map = {
            "recursive": lambda txt: DocumentProcessor.recursive_split(txt, chunk_size=600, chunk_overlap=100),
            "header": DocumentProcessor.header_based_split,
            "sliding": lambda txt: DocumentProcessor.sliding_window_split(txt, window_size=800, overlap=200),
        }
        splitter = methods_map.get(METHOD)
        if not splitter:
            print("Unknown method:", METHOD)
            continue

        chunks_dir = CHUNKS_DIR_TEMPLATE.format(method=METHOD)
        if os.path.exists(chunks_dir) and len(glob.glob(os.path.join(chunks_dir, "*.txt"))) > 0:
            chunks = [open(p, "r", encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(chunks_dir, "*.txt")))]
        else:
            chunks = splitter(full_text)
            os.makedirs(chunks_dir, exist_ok=True)
            for i, c in enumerate(chunks):
                with open(os.path.join(chunks_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(c)

        # build vectorstore once per method
        vs_dir_template = VECTORSTORE_DIR_TEMPLATE.format(method=METHOD)
        vectorstore = VectorRetriever.build_or_load(chunks, embeddings, vs_dir_template)

        bm25_retriever = BM25Retriever(chunks)
        hub = RetrieverHub(vectorstore, bm25_retriever, chunks)

        for alpha in ALPHAS:
            print(f"\n--- alpha={alpha} method={METHOD} ---")
            # create alpha-specific folder
            out_dir_alpha = os.path.join(OUTPUT_DIR, f"{METHOD}_alpha_{alpha}")
            if os.path.exists(out_dir_alpha):
                shutil.rmtree(out_dir_alpha)
            os.makedirs(out_dir_alpha, exist_ok=True)

            for q in QUERIES:
                # retrieve candidates via hub (returns list of (idx, score))
                candidates = hub.retrieve(q, top_n=TOP_K_VECTOR, mode="hybrid", alpha=alpha)
                # run rerank (llm) on these candidates
                try:
                    reranked = llm_service.rerank(q, candidates, chunks, top_k=FINAL_K)
                except Exception:
                    reranked = candidates[:FINAL_K]

                # create maps for rerank_score lookups
                rerank_map = {int(idx): float(score) for idx, score in ([(int(i), float(s)) for (i, s) in (reranked or [])])}

                # generate answer once using reranked (top FINAL_K)
                parsed, raw_llm, _ = llm_service.generate_answer(q, reranked, chunks)
                answer_text = parsed.get("answer","") if isinstance(parsed, dict) else str(parsed)
                used_sources = parsed.get("used_sources", []) if isinstance(parsed, dict) else []

                # save json per query for traceability
                safe_q = re.sub(r"[^\w\u4e00-\u9fff]+", "_", q)[:60]
                outpath = os.path.join(out_dir_alpha, f"res_{METHOD}_alpha_{alpha}_{safe_q}.json")
                rec = {
                    "alpha": alpha,
                    "method": METHOD,
                    "query": q,
                    "candidates_pre_rerank": [(int(i), float(s)) for (i,s) in candidates[:TOP_K_VECTOR]],
                    "reranked": [(int(i), float(s)) for (i,s) in (reranked or [])],
                    "answer": answer_text,
                    "raw_llm": raw_llm,
                    "used_sources": used_sources
                }
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

                # Convert candidates -> longform rows (one row per candidate)
                for rank, (cidx, cscore) in enumerate(rec["candidates_pre_rerank"], start=1):
                    # get rerank score if available
                    rscore = rerank_map.get(int(cidx), None)
                    longform_rows.append({
                        "alpha": alpha,
                        "method": METHOD,
                        "query": q,
                        "candidate_rank": rank,
                        "candidate_idx": int(cidx),
                        "candidate_score": float(cscore),
                        "rerank_score": float(rscore) if rscore is not None else None,
                        "answer": rec["answer"],
                        "raw_llm": rec["raw_llm"],
                        "used_sources": json.dumps(rec.get("used_sources", []), ensure_ascii=False)
                    })

    # After all methods/alphas/queries -> export results
    # write Excel with a sheet per method + combined sheet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_longform_excel(longform_rows, OUTPUT_EXCEL, sheet_by="method")

    # optional: write per-method CSVs
    if WRITE_PER_METHOD_CSV:
        write_per_method_csv(longform_rows, OUTPUT_PER_METHOD_DIR)

    # summary CSV (compact)
    write_summary_csv(longform_rows, SUMMARY_CSV)

    print("Alpha sweep finished. Excel:", OUTPUT_EXCEL, "Per-method CSV dir:", OUTPUT_PER_METHOD_DIR, "Summary:", SUMMARY_CSV)

if __name__ == "__main__":
    run_alpha_sweep()
