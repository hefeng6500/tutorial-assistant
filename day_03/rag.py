import os, json, csv, glob, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS

DOCX_PATH = "../datas/商业计划书.docx"   # 你上传的商业计划书（保留此路径）
OUTPUT_DIR = "../datas/outputs_day3_v1"
CHUNKS_ROOT = "../datas"
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5-nano"   # or "gpt-5-nano"
TEMPERATURE = 0.0
TOP_K = 6  # 从向量库中取多少个 chunk 作为上下文

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_ROOT, exist_ok=True)

# ---------- 文本加载 ----------
def load_docx(path):
    loader = Docx2txtLoader(path)
    docs = loader.load()
    return docs[0].page_content

# ---------- 三种切分器 ----------
def recursive_split(text, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def header_based_split(text):
    # 仅用于带有明显数字标题的商业文档
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

def save_chunks_to_dir(chunks, dirpath):
    os.makedirs(dirpath, exist_ok=True)
    for i, c in enumerate(chunks):
        with open(os.path.join(dirpath, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(c)

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

def force_json_prompt(context_chunks, question):
    # context_chunks: list of tuples (chunk_id, chunk_text)
    # 我们把上下文按分隔符拼接，并用 chunk id 标注
    ctx_parts = []
    for cid, ctext in context_chunks:
        ctx_parts.append(f"## {cid}\n{ctext}")
    ctx = "\n\n---\n\n".join(ctx_parts)

    prompt = prompt = f"""
你是一个严谨的 AI 助手。下面给出若干【上下文片段】和一个问题。
请严格**只依据上下文片段**回答问题，不允许使用外部知识、常识或推断。

如果上下文无法回答，请将 answer 字段设为 "上下文无法回答"，并确保 used_sources 为一个空数组。

回答要求：
1. **只返回一个合法 JSON 对象**，不要输出任何额外解释、前缀或后缀。
2. JSON 结构如下：
{{
  "answer": "<用中文简洁回答>",
  "used_sources": [
    {{
      "chunk_id": "chunk_x",
      "text": "<被引用的原文内容>"
    }}
  ]
}}
3. used_sources 中必须只包含你实际引用过的片段，并附上对应的原文内容。
4. 不得编造不存在的 chunk，也不得修改原文。

下面是上下文片段（已按 chunk_id 标注）：

{ctx}

问题：{question}
"""

    return prompt

# ---------- 简单重合率计算（词级） ----------
def source_overlap_ratio(answer, sources_text):
    def tokenize(s):
        # 去除标点，只保留中英文单词和中文，分词，去重
        s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
        tokens = [t for t in s.split() if t.strip()]
        return set(tokens)
    a_tokens = tokenize(answer)
    s_tokens = tokenize(sources_text)
    if not a_tokens:
        return 0.0
    overlap = len(a_tokens & s_tokens)
    return overlap / max(1, len(a_tokens))

# ---------- 主流程 ----------
def main():
    full_text = load_docx(DOCX_PATH)
    print("Loaded document length:", len(full_text))

    methods = {
        "recursive": lambda txt: recursive_split(txt, chunk_size=600, chunk_overlap=100),
        "header": header_based_split,
        "sliding": lambda txt: sliding_window_split(txt, window_size=800, overlap=200)
    }

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # 你要评估的问题（可修改或扩展）
    queries = [
        "退出策略是什么？",
        "目标市场细分包括哪些？",
        "主要竞争对手有哪些？",
        "首年预计收入是多少？",
        "公司为何有竞争优势？"
    ]

    summary_rows = []

    for method_name, splitter in methods.items():
        print("\n\n========== METHOD:", method_name, "==========")
        chunks_dir = os.path.join(CHUNKS_ROOT, f"chunks_{method_name}")
        vs_dir = os.path.join(CHUNKS_ROOT, f"vectorstore_{method_name}")

        # 1) 切分（如果没切就切）
        txt_files = sorted(glob.glob(os.path.join(chunks_dir, "*.txt")))
        if not os.path.exists(chunks_dir) or len(txt_files) == 0:
            print("Splitting by", method_name)
            chunks = splitter(full_text)
            save_chunks_to_dir(chunks, chunks_dir)
        else:
            print("Chunks already exist:", chunks_dir)
            chunks = [open(p, "r", encoding="utf-8").read() for p in txt_files]

        print(f"{method_name} chunk count:", len(chunks))

        # 2) 向量库
        vectorstore = build_or_load_faiss(chunks, embeddings, vs_dir)

        # 3) 对每个 query 手动检索 + 拼 context + 调用 LLM
        for q in queries:
            print(f"\n[{method_name}] Query -> {q}")
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            # v1 风格： 获取相关文档
            top_docs = retriever.invoke(q)  # 返回 Document 对象列表
            # 把前 K 个 chunk 转成 (id, text)
            context_chunks = []
            used_chunk_texts = []
            for i, d in enumerate(top_docs):
                cid = f"chunk_{i}"
                ctext = d.page_content
                context_chunks.append((cid, ctext))
                used_chunk_texts.append(ctext)

            prompt = force_json_prompt(context_chunks, q)

            llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

            try:
                raw_output = llm.invoke(prompt)

                # print("raw_output")
                # raw_output 可能是一个 LLMResult 对象 or a string depending on binding;
                # 我们尽量把它转成 str
                if hasattr(raw_output, "text"):
                    output_text = raw_output.text
                else:
                    output_text = str(raw_output)
            except Exception as e:
                print("LLM call failed:", e)
                output_text = ""

            # 尝试把 LLM 返回的 JSON 解析出来
            parsed = None
            try:
                # 有时 LLM 返回多余的文本或包含 ```json```，我们尽量抽取第一个 JSON 对象
                json_text = extract_first_json(output_text)
                parsed = json.loads(json_text)
            except Exception as e:
                # 解析失败时，降级处理：把原文放到 answer 字段
                parsed = {"answer": output_text.strip(), "used_sources": []}

            concat_sources = "\n".join(used_chunk_texts)
            overlap = source_overlap_ratio(parsed.get("answer", ""), concat_sources)

            result_record = {
                "method": method_name,
                "query": q,
                "answer": parsed.get("answer", ""),
                "used_sources": parsed.get("used_sources", []),
                "num_source_chunks": len(used_chunk_texts),
                "answer_len": len(parsed.get("answer", "")),
                "source_overlap_ratio": overlap,
                "raw_llm": output_text[:2000]  # 截断保存原始回复（便于排查）
            }

            # append summary row
            summary_rows.append({
                "method": result_record["method"],
                "query": result_record["query"],
                "answer_len": result_record["answer_len"],
                "num_source_chunks": result_record["num_source_chunks"],
                "source_overlap_ratio": result_record["source_overlap_ratio"]
            })

            # save per-query file
            safe_name = q.replace(" ", "_").replace("?", "")
            outpath = os.path.join(OUTPUT_DIR, f"res_{method_name}_{safe_name}.json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(result_record, f, ensure_ascii=False, indent=2)

            print("Saved result to", outpath)
            print("Answer preview:", result_record["answer"][:300])
            print("Overlap ratio:", overlap)

    # 保存 summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "query", "answer_len", "num_source_chunks", "source_overlap_ratio"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\nAll done. Outputs saved to:", OUTPUT_DIR)
    print("Summary CSV:", csv_path)

# ---------- 辅助：从 LLM 长文本抽取第一个 JSON 对象 ----------
def extract_first_json(s):
    # 找第一个 '{' 开始的位置并向后寻找对应的 '}'（粗鲁但常用）
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    # 尝试逐字符匹配大括号
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    # 如果没有闭合，抛异常
    raise ValueError("No complete JSON object found")

if __name__ == "__main__":
    main()
