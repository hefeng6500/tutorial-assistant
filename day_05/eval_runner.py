# eval_runner.py
"""
Evaluation runner for alpha_sweep / per-method res_*.json outputs.

Usage:
    python eval_runner.py --gold path/to/gold_map.json
Options:
    --output_dir (default from config.OUTPUT_DIR)
    --methods (comma separated, default: recursive,header,sliding)
    --recall_k (default 20)
    --prec_k (default 5)
    --chunks_root (optional override)
"""

import os
import glob
import json
import argparse
import csv
from collections import defaultdict
from config import OUTPUT_DIR, CHUNKS_DIR_TEMPLATE

def load_gold_map(path):
    """
    gold_map.json format:
    {
      "首年预计收入是多少？": [
        "预计首年收入达5000万元",
        "预计首年收入 8000 万元"   # 可以有多个变体
      ],
      "目标市场细分包括哪些？": [
        "制造业（50%）： 自动化工厂，需要工控机和工业平板。",
        "物流与仓储（20%）： 三防手持机用于库存管理。"
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        gm = json.load(f)
    return gm

def list_res_files(output_dir, method):
    """
    Find res JSON files for a method in OUTPUT_DIR (alpha subdirs or direct files).
    Matches patterns like:
      {OUTPUT_DIR}/{method}_alpha_*/*.json
      {OUTPUT_DIR}/res_{method}_*.json
    """
    files = []
    # look for method_alpha directories
    pattern1 = os.path.join(output_dir, f"{method}_alpha_*", f"res_{method}_*.json")
    files += glob.glob(pattern1)
    # also look for direct res files
    pattern2 = os.path.join(output_dir, f"res_{method}_*.json")
    files += glob.glob(pattern2)
    # dedupe and sort
    files = sorted(list(set(files)))
    return files

def load_chunks_for_method(method):
    chunks_dir = CHUNKS_DIR_TEMPLATE.format(method=method)
    if not os.path.exists(chunks_dir):
        print(f"[WARN] chunks dir not found for method {method}: {chunks_dir}")
        return []
    files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.txt")))
    chunks = [open(p, "r", encoding="utf-8").read() for p in files]
    return chunks

def find_gold_chunk_ids_for_query(chunks, gold_snippets):
    """
    For a list of gold_snippets (strings), find chunk indices that contain any snippet.
    Return set of indices.
    """
    matched = set()
    for i, c in enumerate(chunks):
        text = c
        for snip in gold_snippets:
            if not snip:
                continue
            if snip in text:
                matched.add(i)
    return sorted(list(matched))

def recall_at_k(preds, gold_ids, k):
    if not gold_ids:
        return 0.0
    topk = preds[:k]
    hit = sum(1 for g in gold_ids if g in topk)
    return hit / len(gold_ids)

def precision_at_k(preds, gold_ids, k):
    topk = preds[:k]
    if not topk:
        return 0.0
    hit = sum(1 for p in topk if p in gold_ids)
    return hit / len(topk)

def parse_res_json(path):
    """
    Expect res JSON to contain at least:
    - alpha (optional)
    - method (optional)
    - query
    - candidates_pre_rerank : list of [idx, score]
    - reranked (optional)
    - answer, raw_llm, used_sources ...
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            rec = json.load(f)
        except Exception:
            # try to extract last JSON object in file
            s = open(path, "r", encoding="utf-8").read()
            m = None
            try:
                import re
                jtxt = re.search(r"\{.*\}", s, flags=re.S).group(0)
                rec = json.loads(jtxt)
            except Exception:
                print(f"[ERR] failed parse {path}")
                rec = {}
    return rec

def run_eval(gold_map_path, methods, recall_k=20, prec_k=5, output_dir=OUTPUT_DIR, out_prefix="eval"):
    gold_map = load_gold_map(gold_map_path)
    methods = [m.strip() for m in methods.split(",")]

    # prepare outputs
    per_file_rows = []
    aggregate = defaultdict(list)  # key (method,alpha,query) -> list of (recall,prec)

    for method in methods:
        print(f"\n[INFO] processing method: {method}")
        chunks = load_chunks_for_method(method)
        if not chunks:
            print(f"[WARN] no chunks for method {method}, skipping.")
            continue

        res_files = list_res_files(output_dir, method)
        if not res_files:
            print(f"[WARN] no res files found for method {method} in {output_dir}")
            continue

        # pre-compute gold mapping for this method
        gold_map_ids = {}
        for q, snippets in gold_map.items():
            ids = find_gold_chunk_ids_for_query(chunks, snippets)
            gold_map_ids[q] = ids

        for rf in res_files:
            rec = parse_res_json(rf)
            q = rec.get("query") or rec.get("question") or ""
            alpha = rec.get("alpha")
            # candidates_pre_rerank might be list of [idx,score]
            cand = rec.get("candidates_pre_rerank") or rec.get("candidates") or []
            cand_ids = [int(i) for (i, s) in cand] if cand else []
            gold_ids = gold_map_ids.get(q, [])
            matched = bool(gold_ids)
            r_at_k = recall_at_k(cand_ids, gold_ids, recall_k)
            p_at_k = precision_at_k(cand_ids, gold_ids, prec_k)

            per_file_rows.append({
                "res_path": rf,
                "method": method,
                "alpha": alpha,
                "query": q,
                "gold_matched": matched,
                "gold_ids": json.dumps(gold_ids, ensure_ascii=False),
                "num_candidates": len(cand_ids),
                f"recall@{recall_k}": r_at_k,
                f"precision@{prec_k}": p_at_k
            })
            aggregate[(method, alpha, q)].append((r_at_k, p_at_k))

    # write per-file CSV
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)
    per_file_csv = os.path.join(out_dir, f"{out_prefix}_per_file.csv")
    with open(per_file_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["res_path", "method", "alpha", "query", "gold_matched", "gold_ids", "num_candidates",
                      f"recall@{recall_k}", f"precision@{prec_k}"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_file_rows:
            writer.writerow(row)
    print(f"[INFO] wrote per-file results to {per_file_csv}")

    # write aggregated summary by (method,alpha,query)
    summary_rows = []
    for key, vals in aggregate.items():
        method, alpha, q = key
        recalls = [v[0] for v in vals]
        precs = [v[1] for v in vals]
        summary_rows.append({
            "method": method,
            "alpha": alpha,
            "query": q,
            "count_files": len(vals),
            f"mean_recall@{recall_k}": sum(recalls)/len(recalls) if recalls else 0.0,
            f"mean_precision@{prec_k}": sum(precs)/len(precs) if precs else 0.0
        })
    agg_csv = os.path.join(out_dir, f"{out_prefix}_summary_by_method_alpha.csv")
    with open(agg_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["method", "alpha", "query", "count_files", f"mean_recall@{recall_k}", f"mean_precision@{prec_k}"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[INFO] wrote aggregated summary to {agg_csv}")

    return per_file_csv, agg_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="path to gold_map.json")
    parser.add_argument("--methods", default="recursive,header,sliding", help="comma sep methods")
    parser.add_argument("--recall_k", type=int, default=20)
    parser.add_argument("--prec_k", type=int, default=5)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    run_eval(args.gold, args.methods, recall_k=args.recall_k, prec_k=args.prec_k, output_dir=args.output_dir)
