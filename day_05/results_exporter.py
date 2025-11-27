# results_exporter.py
import os
import json
import pandas as pd

def write_longform_excel(rows, output_excel_path, sheet_by="method"):
    """
    rows: list of dicts, each dict is a long-form record with at least keys:
        - alpha, method, query, candidate_rank, candidate_idx, candidate_score,
          rerank_score (optional), answer, raw_llm, used_sources (JSON-string or list)
    output_excel_path: path to .xlsx to write
    sheet_by: group rows into separate sheets by this key (default "method")
    """
    if not rows:
        raise ValueError("rows empty")

    df = pd.DataFrame(rows)

    # ensure columns order (some may be missing)
    cols = [
        "alpha", "method", "query",
        "candidate_rank", "candidate_idx", "candidate_score", "rerank_score",
        "answer", "raw_llm", "used_sources"
    ]
    # keep only existing cols in order
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]

    # group by sheet key
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        for key, group in df.groupby(sheet_by):
            # sort by alpha, query, candidate_rank
            group_sorted = group.sort_values(by=["alpha", "query", "candidate_rank"], ascending=[True, True, True])
            group_sorted.to_excel(writer, sheet_name=str(key)[:31], index=False, columns=cols)
        # also write a combined sheet
        df_sorted = df.sort_values(by=["method", "alpha", "query", "candidate_rank"])
        df_sorted.to_excel(writer, sheet_name="ALL", index=False, columns=cols)

def write_per_method_csv(rows, output_dir):
    """
    Write one CSV per method (long-form rows).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for method, group in df.groupby("method"):
        outpath = os.path.join(output_dir, f"{method}_longform.csv")
        group_sorted = group.sort_values(by=["alpha", "query", "candidate_rank"])
        group_sorted.to_csv(outpath, index=False, encoding="utf-8-sig")

def write_summary_csv(rows, output_path):
    """
    Write a compact summary CSV (one row per alpha x method x query),
    with some numeric metrics for quick scanning.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # compute simple metrics per alpha,method,query
    summary = df.groupby(["alpha", "method", "query"]).agg(
        num_candidates=("candidate_idx", "count"),
        avg_candidate_score=("candidate_score", "mean"),
        answer_len=("answer", lambda s: int(s.dropna().astype(str).map(len).mean()))
    ).reset_index()
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
