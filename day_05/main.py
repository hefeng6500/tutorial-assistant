import os
import glob
import csv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from config import (
    DOCX_PATH,
    EMBED_MODEL,
    LLM_MODEL,
    TEMPERATURE,
    CHUNKS_DIR_TEMPLATE,
    OUTPUT_DIR,
)
from data_loader import DocumentProcessor
from llm_service import LLMService
from pipeline import RetrievalPipeline
from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    full_text = DocumentProcessor.load_docx(DOCX_PATH)
    print("Loaded document length:", len(full_text))

    methods = {
        "recursive": lambda txt: DocumentProcessor.recursive_split(
            txt, chunk_size=600, chunk_overlap=100
        ),
        # "header": DocumentProcessor.header_based_split,
        # "sliding": lambda txt: DocumentProcessor.sliding_window_split(
        #     txt, window_size=800, overlap=200
        # ),
    }

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    llm_service = LLMService(llm)
    pipeline = RetrievalPipeline(llm_service)

    queries = [
        "退出策略是什么？",
        "目标市场细分包括哪些？",
        "主要竞争对手有哪些？",
        "首年预计收入是多少？",
        "公司为何有竞争优势？",
    ]

    summary = []
    for method, splitter in methods.items():
        # get chunks
        chunks_dir = CHUNKS_DIR_TEMPLATE.format(method=method)
        if (
            os.path.exists(chunks_dir)
            and len(glob.glob(os.path.join(chunks_dir, "*.txt"))) > 0
        ):
            chunks = [
                open(p, "r", encoding="utf-8").read()
                for p in sorted(glob.glob(os.path.join(chunks_dir, "*.txt")))
            ]
        else:
            chunks = splitter(full_text)
        res = pipeline.run_experiment(method, chunks, embeddings, queries)
        for r in res:
            summary.append(
                {
                    "method": r["method"],
                    "query": r["query"],
                    "answer_len": len(r["answer"]),
                    "num_source_chunks": (
                        len(r["used_sources"])
                        if isinstance(r["used_sources"], list)
                        else 0
                    ),
                }
            )

    # save summary csv
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "query", "answer_len", "num_source_chunks"]
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print("All done. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
