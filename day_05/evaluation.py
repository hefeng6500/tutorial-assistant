# evaluation.py
from typing import List, Dict

def recall_at_k(preds: List[int], gold: List[int], k: int) -> float:
    """
    preds: list of predicted chunk indices (ordered)
    gold: list of gold chunk indices
    returns: recall@k (fraction of gold covered in top-k)
    """
    topk = preds[:k]
    if not gold:
        return 0.0
    hit = sum(1 for g in gold if g in topk)
    return hit / len(gold)

def precision_at_k(preds: List[int], gold: List[int], k: int) -> float:
    topk = preds[:k]
    if not topk:
        return 0.0
    hit = sum(1 for p in topk if p in gold)
    return hit / len(topk)

# Example usage:
# gold_map = {"首年预计收入是多少？":[4], "目标市场细分包括哪些？":[1]}
# preds = [10,4,8,3] -> recall_at_k(preds, gold_map[q], 20)
