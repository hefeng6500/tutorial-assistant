# Tutorial Assistant


## Get Started

```shell
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate      # Linux/macOS
# 或
venv\Scripts\activate         # Windows
```

```shell
# 安装依赖
pip install -r requirements.txt
```

## Day02

商业计划书 13 页

| Chunk  Strategy                                              | query                                | Answer                                                       | 答案 |
| ------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------ | ---- |
| RecursiveCharacterTextSplitter<br />chunk_size: 1200<br />chunk_overlap: 60 | 代理亿道的三防设备的退出策略？       | 上下文无法回答                                               | ❌    |
| header_based_split                                           | 代理亿道的三防设备的退出策略？       | 退出策略是在5年内通过IPO或被收购实现退出。                   | ✅    |
| header_based_split                                           | 帮我计算一下，代理亿道的产品的成本？ | 上下文无法回答                                               | ❌    |
| Sliding Window Chunker                                       | 帮我计算一下，代理亿道的产品的成本？ | 据上下文，代理亿道的产品成本分为两部分：可变成本（采购）和固定成本（租金等）。在财务表中：<br/>- 采购成本（成本项）= 收入 × 60%<br/>- 固定成本 = 200 万元/年<br/><br/>因此：<br/>- 2026 年<br/>  - 收入：8000 万元<br/>  - 采购成本（成本）：4800 万元<br/>  - 固定成本：200 万元<br/>  - 总成本：6800 万元<br/><br/>- 2027 年<br/>  - 收入：12000 万元<br/>  - 采购成本（成本）：7200 万元<br/>  - 固定成本：200 万元<br/>  - 总成本：9200 万元<br/><br/>- 2028 年<br/>  - 收入：18000 万元<br/>  - 采购成本（成本）：10800 万元<br/>  - 固定成本：200 万元<br/>  - 总成本：12800 万元<br/><br/>如仅列出采购成本（成本项），三年的数值为：4800 万元、7200 万元、10800 万元。 |      |





```
# 2.1. Recursive
recursive_chunks = split_text(full_text)
save_chunks(recursive_chunks, "datas/chunks_recursive")

# 2.2. Header-based
header_chunks = header_based_split(full_text)
save_chunks(header_chunks, "datas/chunks_header")

# 2.3. Sliding Window
sliding_chunks = sliding_window_split(full_text)
save_chunks(sliding_chunks, "datas/chunks_sliding")

print("Chunking done!")

```

```
Saved 6 chunks to datas/chunks_recursive
Saved 85 chunks to datas/chunks_header
Saved 11 chunks to datas/chunks_sliding
Chunking done!

=== RECURSIVE ===
Chunks: 6
Average length: 1104.33
Example chunk:
风险分析与缓解措施 …………………………………………………… 67
11.1 主要风险 ……………………………………………………………….. 67
11.2 风险缓解策略 ………………………………………………………… 69


附录 …………………………………………………………………………… 71
12.1 支持文件 ……………………………………………………………….. 71
12.2 参考文献 ……………………………………………………………….. 72

（注：页码为估计值，基于标准A4纸张、12号字体、1.5倍行距渲染时的大约页数，总计约40页。实际渲染可能略有差异。）



1. 执行摘...

=== HEADER ===
Chunks: 85
Average length: 70.45
Example chunk:
3.3 目标市场细分
制造业（50%）： 自动化工厂，需要工控机和工业平板。

物流与仓储（20%）： 三防手持机用于库存管理。

能源与矿业（15%）： 加固设备耐极端环境。

医疗与公共服务（15%）： 移动终端。

客户类型：中小企业（70%）、大型企业（30%）。...

=== SLIDING ===
Chunks: 11
Average length: 771.55
Example chunk:
 33
6.3 定价策略 ………………………………………………………………… 35
6.4 推广活动 ………………………………………………………………… 36
6.5 客户关系管理 …………………………………………………………. 37


运营计划 ……………………………………………………………………… 39
7.1 运营流程 ………………………………………………………………… 39
7.2 设施与设备 …………………………………………………………….. 41
7.3 人力资源 ………………………………………………………………… 42
7.4 质量控制 ……………………………………………………...
```

