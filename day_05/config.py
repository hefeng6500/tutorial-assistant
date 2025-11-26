import os

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
DOCX_PATH = "../datas/商业计划书.docx"
CHUNKS_ROOT = "../datas"
VECTORSTORE_DIR_TEMPLATE = os.path.join(CHUNKS_ROOT, "vectorstore_{method}")
CHUNKS_DIR_TEMPLATE = os.path.join(CHUNKS_ROOT, "chunks_{method}")
OUTPUT_DIR = "../datas/outputs_day5_v1"

# Models
EMBED_MODEL = "text-embedding-v4"
LLM_MODEL = "qwen3-max"
TEMPERATURE = 0.0

# Retrieval Parameters
TOP_K_VECTOR = 20  # vector top N before rerank
FINAL_K = 5  # after rerank
ALPHA = 0.8  # hybrid weight: final = alpha * vector_score + (1-alpha) * bm25_score (after norm)

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_ROOT, exist_ok=True)
