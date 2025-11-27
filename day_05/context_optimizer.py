# context_optimizer.py
import re
import numpy as np

class ContextOptimizer:
    def __init__(self,
                 min_chunk_len=30,
                 max_sentences=5,
                 relevance_threshold=0.2):
        """
        min_chunk_len: 去掉太短的 chunk（标题类）
        max_sentences: 裁剪后每个 chunk 最多取多少句
        relevance_threshold: 低于该语义相似度的句子不纳入上下文
        """
        self.min_chunk_len = min_chunk_len
        self.max_sentences = max_sentences
        self.relevance_threshold = relevance_threshold

    # ----------------------------------------------------
    # Step 1: 去重（Deduplicate）
    # ----------------------------------------------------
    def dedupe(self, idx_list):
        seen = set()
        new_list = []
        for idx in idx_list:
            if idx not in seen:
                seen.add(idx)
                new_list.append(idx)
        return new_list

    # ----------------------------------------------------
    # Step 2: 去噪（Noise Filter）
    # ----------------------------------------------------
    def is_noise_chunk(self, text):
        if len(text.strip()) < self.min_chunk_len:
            return True
        # 去掉纯符号、目录项、页脚
        if re.fullmatch(r"[\W_]+", text.strip()):
            return True
        if "版权所有" in text or "免责声明" in text:
            return True
        return False

    # ----------------------------------------------------
    # Step 3: 裁剪 chunk（Smart Snippet Extraction）
    # ----------------------------------------------------
    def extract_relevant_sentences(self, query, chunk):
        """
        利用简单词重叠（或者 TF-IDF overlap）过滤句子
        Day9 会升级为 LLM scoring 版本
        """
        sentences = re.split(r"[。！？\n]", chunk)
        q_tokens = set(query)
        scored = []

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            # overlap score（非常轻量的启发式）
            overlap = len(set(s) & q_tokens) / (len(s) + 1)
            scored.append((overlap, s))

        # 排序 + 截断
        scored = sorted(scored, key=lambda x: x[0], reverse=True)
        keep = [s for (sc, s) in scored[: self.max_sentences]]
        keep = [k for k in keep if k.strip()]
        return "。".join(keep)

    # ----------------------------------------------------
    # Step 4: 合并上下文（最终传给 LLM）
    # ----------------------------------------------------
    def merge(self, chosen_chunks):
        """将多个 snippet 合并成最终上下文"""
        merged = []
        for idx, snippet in chosen_chunks:
            merged.append(f"## chunk_{idx}\n{snippet}")
        return "\n\n---\n\n".join(merged)

    # ----------------------------------------------------
    # pipeline entry
    # ----------------------------------------------------
    def process(self, query, reranked, chunks):
        """
        reranked: list[(idx, score_llm)]
        chunks: full chunk list
        """
        # Step 1: 去重
        idxs = [idx for (idx, _) in reranked]
        idxs = self.dedupe(idxs)

        chosen = []
        for idx in idxs:
            chunk = chunks[idx]
            # Step 2: 去噪
            if self.is_noise_chunk(chunk):
                continue

            # Step 3: 裁剪 chunk 语句
            snippet = self.extract_relevant_sentences(query, chunk)
            if not snippet:
                continue

            chosen.append((idx, snippet))

        # Step 4: 合并
        final_context = self.merge(chosen)
        return final_context
