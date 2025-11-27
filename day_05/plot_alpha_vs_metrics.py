# plot_alpha_vs_metrics.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# 配置：把路径改成你的文件路径（默认按你上传的路径）
SUMMARY_CSV = "../datas/outputs_day5_v1/eval_summary_by_method_alpha.csv"  # 如果不在当前目录，填绝对/相对路径
OUTDIR = "./analysis_day7"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")

# 自动检测 recall / precision 列名
rec_col = next((c for c in df.columns if 'mean_recall' in c), None)
prec_col = next((c for c in df.columns if 'mean_precision' in c), None)
if rec_col is None or prec_col is None:
    # 尝试更宽松的匹配
    rec_col = next((c for c in df.columns if 'recall' in c.lower()), None)
    prec_col = next((c for c in df.columns if 'precision' in c.lower()), None)

if rec_col is None or prec_col is None:
    raise SystemExit(f"无法找到 recall/precision 列。当前 summary CSV 的列为: {df.columns.tolist()}")

# 确保 alpha 是数值
df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')

# 对每个 method 绘图
methods = df['method'].unique().tolist()
for method in methods:
    sub = df[df['method'] == method]
    if sub.empty:
        continue
    agg = sub.groupby('alpha')[[rec_col, prec_col]].mean().reset_index().sort_values('alpha')

    plt.figure(figsize=(8,4))
    plt.plot(agg['alpha'], agg[rec_col], marker='o', label='mean_recall')
    plt.plot(agg['alpha'], agg[prec_col], marker='s', label='mean_precision')
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title(f'{method} — mean recall & precision vs alpha')
    plt.xticks(agg['alpha'].unique())
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    out_path = os.path.join(OUTDIR, f"{method}_recall_precision_vs_alpha.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved:", out_path)

print("All done. Check the images in", OUTDIR)
