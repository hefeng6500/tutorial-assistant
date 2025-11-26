# query_processor.py
import re
from typing import List

class ProcessedQuery:
    """
    RAG 中的上帝对象：标准化后的查询结构。
    original: 原始查询
    rewrites: LLM 多路改写后的查询（提升召回）
    category: Query 类型（用于未来高级检索策略）
    """
    def __init__(self, original: str, rewrites: List[str], category: str):
        self.original = original
        self.rewrites = rewrites
        self.category = category


class QueryProcessor:
    """
    通用 Query Processor，不包含文档规则：
    - Normalize（规范化）
    - Classify（轻量分类）
    - MultiQuery rewrite（LLM）
    """

    # ------------------------
    # Step 1: 归一化
    # ------------------------
    @staticmethod
    def normalize(q: str) -> str:
        if not q:
            return ""
        s = q.strip()
        s = re.sub(r"\s+", " ", s)   # 多空白缩减
        return s

    # ------------------------
    # Step 2: 轻量 Query 分类（通用）
    # ------------------------
    @staticmethod
    def classify(q: str) -> str:
        ql = q.lower()

        # 数字类问题
        if re.search(r"(多少|几|多少年|多少万|收入|预计|金额|规模|营收)", q):
            return "numeric"

        # 定义类
        if re.search(r"(什么是|定义|含义|解释)", q):
            return "definition"

        # 实体类
        if re.search(r"(谁|哪家|竞争|对手|公司|品牌|主体)", q):
            return "entity"

        # 默认类别
        return "open"

    # ------------------------
    # Step 3: MultiQuery 重写
    # ------------------------
    @staticmethod
    def _rewrite_with_llm(llm_service, q: str, n=3) -> List[str]:
        try:
            rewrites = llm_service.rewrite_query(q, n_variants=n)
            cleaned = [QueryProcessor.normalize(r) for r in rewrites if r.strip()]
            return cleaned
        except Exception:
            return []

    # ------------------------
    # Step 4: 查询流程 orchestrator
    # ------------------------
    @classmethod
    def process(cls, raw_q: str, llm_service, n_variants: int = 3) -> ProcessedQuery:
        # Normalize
        q0 = cls.normalize(raw_q)

        # Classify
        category = cls.classify(q0)

        # Rewrite
        rewrites = cls._rewrite_with_llm(llm_service, q0, n=n_variants)

        # 保证 original 在 rewrites 的首位
        final_rewrites = []
        seen = set()
        for r in [q0] + rewrites:
            if r not in seen:
                seen.add(r)
                final_rewrites.append(r)

        return ProcessedQuery(
            original=q0,
            rewrites=final_rewrites,
            category=category
        )
