import json
import re

class LLMService:
    def __init__(self, llm):
        self.llm = llm

    def rewrite_query(self, query, n_variants=3):
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
        raw = self.llm.invoke(prompt)
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

    def rerank(self, query, candidates, chunks, top_k=5):
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
        raw = self.llm.invoke(prompt)
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

    def rerank_v2(self, query, candidates, chunks, top_k=5, score_scale=100):
        """
        更严格的 reranker：
        - candidates: list[(idx, score)] （idx 为 int）
        - chunks: list of chunk texts
        - top_k: 返回 top_k (idx, score)
        - score_scale: 0-100 scale (integer or float)
        返回: list[(idx, score_float)] 按 score 降序
        解析：要求 LLM 返回严格的 JSON array: [{"chunk_id":"chunk_12","score":92.5}, ...]
        失败回退策略：尝试解析每行 "chunk_X: N" 格式；最终回退为 simple decreasing by original score。
        """
        # Prepare candidate contexts (truncate to avoid huge prompts)
        parts = []
        for (idx, orig_score) in candidates:
            txt = chunks[int(idx)]
            parts.append({"id": f"chunk_{int(idx)}", "text": txt[:1500]})

        ctx = "\n\n---\n\n".join([f"## {p['id']}\n{p['text']}" for p in parts])

        prompt = f"""
你是一个严格的相关性评估助手。目标：为下面的问题与候选片段分别给出相关性得分（数值越高越相关），数值范围为 0 到 {score_scale}。
要求严格只返回一个 JSON 数组，格式如下：
[{{"chunk_id":"chunk_12","score":92.5}}, ...]
数组项顺序不限，但每个对象必须包含 chunk_id（格式 chunk_N）和 score（数值，允许小数）。
不要返回任何其它非 JSON 的文本。

问题：{query}

候选片段（已截断到前1500字符）：
{ctx}
"""
        raw = self.llm.invoke(prompt)
        text = raw.text if hasattr(raw, "text") else str(raw)

        # try strict JSON parse
        parsed_scores = []
        try:
            j = json.loads(re.search(r"\[.*\]", text, flags=re.S).group(0))
            if isinstance(j, list):
                for item in j:
                    cid = item.get("chunk_id") if isinstance(item, dict) else None
                    score = item.get("score") if isinstance(item, dict) else None
                    if cid and score is not None:
                        m = re.search(r"chunk_(\d+)", cid)
                        idx = int(m.group(1)) if m else None
                        if idx is not None:
                            parsed_scores.append((idx, float(score)))
        except Exception:
            # fallback parse: try to extract lines like chunk_12: 92.5
            parsed_scores = []
            for line in text.splitlines():
                m = re.search(r"(?:chunk_)?(\d+)[\s:,-]+(\d+(?:\.\d+)?)", line)
                if m:
                    idx = int(m.group(1))
                    score = float(m.group(2))
                    parsed_scores.append((idx, score))
        # If parsing failed, fallback to using original candidate scores (normalized)
        if not parsed_scores:
            # normalize incoming candidates score to 0-1 then multiply
            vals = [s for (_, s) in candidates]
            if vals:
                mn, mx = min(vals), max(vals)
                if mx - mn < 1e-12:
                    parsed_scores = [(int(i), float(score_scale)) for (i, s) in candidates]
                else:
                    parsed_scores = [
                        (int(i), (float(s) - mn) / (mx - mn) * score_scale) for (i, s) in candidates
                    ]
        # sort and return top_k
        parsed_scores_sorted = sorted(parsed_scores, key=lambda x: x[1], reverse=True)[:top_k]
        return parsed_scores_sorted


    def generate_answer(self, query, reranked, chunks):
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

问题：{query}
"""
        raw = self.llm.invoke(gen_prompt)
        txt = raw.text if hasattr(raw, "text") else str(raw)
        # attempt to extract JSON
        try:
            parsed = json.loads(re.search(r"\{.*\}", txt, flags=re.S).group(0))
        except Exception:
            parsed = {"answer": txt.strip(), "used_sources": []}
        return parsed, txt, true_source_ids
