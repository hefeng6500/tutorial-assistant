import os, glob, json, csv, re, math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS

# BM25 lib
from rank_bm25 import BM25Okapi
# optional jieba tokenizer for Chinese (recommended)
try:
    import jieba
    HAVE_JIEBA = True
except Exception:
    HAVE_JIEBA = False

# ---------------- config ----------------
DOCX_PATH = "../datas/商业计划书.docx"
CHUNKS_ROOT = "../datas"
VECTORSTORE_DIR_TEMPLATE = os.path.join(CHUNKS_ROOT, "vectorstore_{method}")
CHUNKS_DIR_TEMPLATE = os.path.join(CHUNKS_ROOT, "chunks_{method}")
OUTPUT_DIR = "../datas/outputs_day4_v2"

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5-nano"
TEMPERATURE = 0.0

TOP_K_VECTOR = 20   # vector top N before rerank
FINAL_K = 5         # after rerank
ALPHA = 0.8         # hybrid weight: final = alpha * vector_score + (1-alpha) * bm25_score (after norm)
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_ROOT, exist_ok=True)

# ---------- reuse your loader/splitter ----------
def load_docx(path=DOCX_PATH):
    loader = Docx2txtLoader(path)
    docs = loader.load()
    return docs[0].page_content

def recursive_split(text, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

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

def sliding_window_split(text, window_size=800, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + window_size)
        chunks.append(text[start:end])
        start += (window_size - overlap)
    return chunks

# ---------- simple tokenizer (for BM25) ----------
def simple_tokenize(text):
    if HAVE_JIEBA:
        return list(jieba.cut_for_search(text))
    # fallback: split non-word chars and whitespace
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    toks = [t for t in s.split() if t.strip()]
    return toks

# ---------- BM25 index builder ----------
def build_bm25_index(chunks):
    tokenized = [simple_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

# ---------- BM25 retrieve ----------
def bm25_retrieve(bm25, tokenized_corpus, query, top_n=20):
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)  # list of scores per doc
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
    return ranked  # list of (idx, score)

# ---------- helper: normalize scores to [0,1] ----------
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

# ---------- build/load faiss from chunks ----------
def build_or_load_faiss(chunks, embeddings, persist_dir):
    if os.path.exists(persist_dir):
        try:
            vs = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded vectorstore from {persist_dir}")
            return vs
        except Exception as e:
            print("Failed to load existing vectorstore, rebuilding. Error:", e)
    vs = FAISS.from_texts(chunks, embeddings)
    vs.save_local(persist_dir)
    print(f"Built and saved vectorstore to {persist_dir}")
    return vs

# ---------- multi-query rewrite using LLM ----------
def multi_query_rewrite(llm, query, n_variants=3):
    """
    Use LLM to produce few alternative reformulations of query.
    Returns list[str]
    """
    prompt = f"""
你是一个查询重写助手。给出一个用户问题，把它改写成 {n_variants} 个不同但语义相近的检索查询，适合用于文档检索（不增加外部事实）。
问题：{query}

要求：
- 返回 JSON 数组，如 ["改写1", "改写2", ...]
- 不要包含额外文本
"""
    raw = llm.invoke(prompt)
    text = raw.text if hasattr(raw, "text") else str(raw)
    # try extract JSON array
    try:
        arr = json.loads(text.strip())
        if isinstance(arr, list):
            return arr
    except Exception:
        # fallback: naive split by newline
        lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
        # dedupe
        out = []
        for l in lines:
            if l and l not in out:
                out.append(l)
        return out[:n_variants]
    return arr[:n_variants]

# ---------- hybrid retrieve: combine vector and bm25 scores ----------
def hybrid_retrieve(vectorstore, bm25, chunks, query, top_n=TOP_K_VECTOR, alpha=ALPHA):
    # 1) vector top N (get ids and vector scores)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_n})
    # LangChain v1 style: get_relevant_documents returns docs but not scores.
    # Instead we use vectorstore.similarity_search_with_score to get scores if available.
    try:
        vs_results = vectorstore.similarity_search_with_score(query, k=top_n)
        # vs_results: list of (Document, score)
        vec_list = [(i, score) for i, (_, score) in enumerate(vs_results)]  # placeholder indices will be remapped
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
        top_docs = retriever.get_relevant_documents(query)
        vec_candidates = []
        for i, d in enumerate(top_docs):
            # try to find idx
            try:
                idx = chunks.index(d.page_content)
            except ValueError:
                idx = None
                for j,c in enumerate(chunks):
                    if d.page_content.strip() and d.page_content.strip()[:50] in c:
                        idx = j
                        break
            if idx is not None:
                vec_candidates.append((idx, 1.0))  # uniform score
    # normalize vector scores
    vec_norm = normalize_scores(vec_candidates)

    # 2) bm25 top N
    bm25_ranks = bm25_retrieve(bm25, None, query, top_n=top_n)  # returns list (idx, score)
    bm25_norm = normalize_scores(bm25_ranks)

    # 3) merge: create dict of idx -> combined score
    scores = {}
    for idx, s in vec_norm:
        scores[idx] = scores.get(idx, 0.0) + alpha * s
    for idx, s in bm25_norm:
        scores[idx] = scores.get(idx, 0.0) + (1 - alpha) * s

    # 4) sort and return top candidates
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return merged  # list of (idx, combined_score)

# ---------- rerank by LLM: score candidates 0-100 and return top K ----------
def rerank_by_llm(llm, query, candidates, chunks, top_k=FINAL_K):
    """
    candidates: list of (idx, score)
    chunks: full chunk list
    returns: list of (idx, score_llm)
    """
    # prepare a prompt containing all candidates labeled
    parts = []
    for rank, (idx, sc) in enumerate(candidates):
        text = chunks[idx]
        parts.append({"id": f"chunk_{idx}", "text": text[:1500]})  # truncate long
    # build JSON-like list for prompt
    ctx = "\n\n---\n\n".join([f"## chunk_{p['id'].split('_')[-1]}\n{p['text']}" for p in parts])

    prompt = f"""
你是一个严谨的评估助手。请根据提供的上下文片段为下列问题评估每个片段的**相关性评分（0-100）**，评分越高代表片段越有助于回答问题。
要求返回 JSON 数组，格式为 [{{"chunk_id":"chunk_X","score":NN}}, ...]，数组项按原始候选顺序给出，不要返回额外文本。

问题：{query}

候选片段（已截断到前1500字符）：
{ctx}
"""
    raw = llm.invoke(prompt)
    text = raw.text if hasattr(raw, "text") else str(raw)
    # try extract JSON array
    scores = []
    try:
        arr = json.loads(text.strip())
        if isinstance(arr, list):
            for item in arr:
                cid = item.get("chunk_id")
                score = float(item.get("score", 0))
                # parse idx
                m = re.search(r"chunk_(\d+)", cid)
                idx = int(m.group(1)) if m else None
                scores.append((idx, score))
    except Exception:
        # fallback: naive heuristic - ask LLM to produce per-chunk lines? but for robustness:
        # If parse fails, we will assign uniform decreasing scores
        for i, (idx, sc) in enumerate(candidates):
            scores.append((idx, max(0, 100 - i*5)))
    # sort by score desc and return top_k
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return scores_sorted

# ---------- high-level pipeline: multi-query + hybrid + rerank ----------
def pipeline_query(llm, vectorstore, bm25, chunks, query, multi_q=3, top_n=TOP_K_VECTOR, alpha=ALPHA):
    # 1) multi-query rewrite
    rewrites = multi_query_rewrite(llm, query, n_variants=multi_q)
    # dedupe
    rewrites = list(dict.fromkeys([r.strip() for r in rewrites if r.strip()]))[:multi_q]
    print("Rewrites:", rewrites)

    # 2) for each rewrite, run hybrid_retrieve and collect candidate indices with scores
    candidate_scores = {}  # idx -> max_score
    all_candidates = []
    for rq in rewrites:
        merged = hybrid_retrieve(vectorstore, bm25, chunks, rq, top_n=top_n, alpha=alpha)
        for idx, sc in merged:
            if idx is None:
                continue
            if idx not in candidate_scores or sc > candidate_scores[idx]:
                candidate_scores[idx] = sc
            all_candidates.append((idx, sc))
    # convert to list sorted
    merged_list = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    # take top top_n unique candidates
    top_candidates = merged_list[:top_n]
    print("Candidates (pre-rerank):", top_candidates[:10])

    # 3) rerank top_candidates by LLM
    reranked = rerank_by_llm(llm, query, top_candidates, chunks, top_k=FINAL_K)
    used_ids = [f"chunk_{idx}" for idx,_ in reranked]
    print("Reranked top:", reranked)
    return reranked, used_ids  # list of (idx, score_llm)

# ---------- core: run experiments for a given chunking method ----------
def run_for_method(method_name, chunks, embeddings, queries, llm):
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
    vectorstore = build_or_load_faiss(chunks, embeddings, vs_dir)
    # build bm25
    bm25, tokenized = build_bm25_index(chunks)

    results = []
    for q in queries:
        print("\nQuery:", q)
        reranked, used_ids = pipeline_query(llm, vectorstore, bm25, chunks, q, multi_q=3, top_n=TOP_K_VECTOR, alpha=ALPHA)
        # prepare output record
        used = []
        for idx, score in reranked:
            used.append({"chunk_id": f"chunk_{idx}", "text": chunks[idx][:1000]})
        # simple answer generation by LLM using top FINAL_K chunks
        # build prompt forcing JSON answer
        ctx_parts = []
        for idx, _ in reranked:
            ctx_parts.append(f"## chunk_{idx}\n{chunks[idx]}")
        ctx = "\n\n---\n\n".join(ctx_parts)

        # 真实来源（由 pipeline 决定，不是模型决定）
        true_source_ids = [f"chunk_{idx}" for idx, _ in reranked]

        gen_prompt = f"""
你是一个严谨的 AI 助手。仅基于下面的上下文片段回答问题（不得使用外部信息）。如果上下文无法回答，请返回 "上下文无法回答"。
返回 JSON：{{"answer":"...", "used_sources":["chunk_X", ...]}} 严格返回 JSON，不要有多余文本。

上下文：
{ctx}

问题：{q}
"""
        raw = llm.invoke(gen_prompt)
        txt = raw.text if hasattr(raw, "text") else str(raw)
        # attempt to extract JSON
        try:
            parsed = json.loads(re.search(r"\{.*\}", txt, flags=re.S).group(0))
        except Exception:
            parsed = {"answer": txt.strip(), "used_sources": []}
        results.append({
            "method": method_name,
            "query": q,
            "answer": parsed.get("answer", ""),
            "used_sources": used_ids,
            "raw_llm": txt
        })
        # save per query
        safe_q = re.sub(r"[^\w\u4e00-\u9fff]+", "_", q)[:60]
        outpath = os.path.join(OUTPUT_DIR, f"res_{method_name}_{safe_q}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(results[-1], f, ensure_ascii=False, indent=2)
        print("Saved result to", outpath)
    return results

# ---------- main ----------
def main():
    full_text = load_docx(DOCX_PATH)
    print("Loaded document length:", len(full_text))

    methods = {
        "recursive": lambda txt: recursive_split(txt, chunk_size=600, chunk_overlap=100),
        "header": header_based_split,
        "sliding": lambda txt: sliding_window_split(txt, window_size=800, overlap=200)
    }

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

    queries = [
        "退出策略是什么？",
        "目标市场细分包括哪些？",
        "主要竞争对手有哪些？",
        "首年预计收入是多少？",
        "公司为何有竞争优势？"
    ]

    summary = []
    for method, splitter in methods.items():
        # get chunks
        chunks_dir = CHUNKS_DIR_TEMPLATE.format(method=method)
        if os.path.exists(chunks_dir) and len(glob.glob(os.path.join(chunks_dir, "*.txt"))) > 0:
            chunks = [open(p, "r", encoding="utf-8").read() for p in sorted(glob.glob(os.path.join(chunks_dir, "*.txt")))]
        else:
            chunks = splitter(full_text)
        res = run_for_method(method, chunks, embeddings, queries, llm)
        for r in res:
            summary.append({
                "method": r["method"],
                "query": r["query"],
                "answer_len": len(r["answer"]),
                "num_source_chunks": len(r["used_sources"]) if isinstance(r["used_sources"], list) else 0
            })

    # save summary csv
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "query", "answer_len", "num_source_chunks"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print("All done. Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
