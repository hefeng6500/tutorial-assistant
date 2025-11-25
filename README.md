# Tutorial Assistant


## Get Started

```shell
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate      # Linux/macOS
# 或
source venv/Scripts/activate         # Windows
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

## Day03

基于商业计划书，对比评估三种文本切分方法（recursive、header、sliding）在 RAG 问答中的表现。

### 评估指标
- **源重叠率**（source_overlap_ratio）：答案与源文本的词汇重叠比例，衡量答案是否基于上下文
- **答案长度**（answer_len）：生成答案的字符数
- **源chunk数量**（num_source_chunks）：检索到的上下文片段数量

### 测试结果

| 问题 | Recursive | Header | Sliding | 最佳方法 |
|------|-----------|--------|---------|----------|
| 退出策略是什么？ | 0.0 | **1.0** | **1.0** | Header/Sliding |
| 目标市场细分包括哪些？ | 0.88 | 0.92 | **1.0** | Sliding |
| 主要竞争对手有哪些？ | 0.83 | 0.83 | 0.83 | 相同 |
| 首年预计收入是多少？ | 0.0 | 0.0 | 0.0 | 均需改进 |
| 公司为何有竞争优势？ | 0.0 | **0.38** | 0.0 | Header |

### 核心发现

1. **Header 和 Sliding 在结构化问题上表现更优**：退出策略、目标市场等结构化信息，header/sliding 的源重叠率达到 1.0，答案更贴近原文。

2. **Recursive 表现不稳定**：在部分问题上源重叠率为 0，可能存在答案生成时未充分引用上下文的情况。

3. **数值类问题普遍表现较差**：所有方法在"首年预计收入"问题上源重叠率均为 0，可能因数值信息分散或检索不准确。

4. **Header 方法在语义理解上略胜一筹**：在"竞争优势"这类需要综合理解的问题上，header 方法（0.38）优于其他方法（0.0）。

### 结论

对于结构化商业文档，**header-based 和 sliding window 切分方法更适合 RAG 应用**，能更好地保持语义完整性和检索准确性。

## Day04

# 🟦 ① Header 法的整体表现（结构化文档的最优选择）

### ✔ 1. “公司为何有竞争优势？”

答案 **完全正确**（高质量）
引用 chunk 完整、有逻辑，来源可靠。
（）

→ **header 在结构化问题上表现极佳**

---

### ✔ 2. “目标市场细分包括哪些？”

答案非常准确，与文档一致。
（）

→ **正确率高**

---

### ✔ 3. “首年预计收入是多少？”

答案：“5000万元”
引用 chunk_19 避免了 recursive 中那种视角混乱。
（）

→ **语义仍有视角歧义（Day 5 要解决）**
但结果本身是命中内容的。

---

### ❌ 4. “退出策略是什么？”

回答为：“上下文无法回答”
（）

但是文档中存在退出策略

→ 这是 **BM25 + 向量召回失败** 或 **Rerank 阶段错误过滤**。

造成原因：

* 退出策略在文档后部（章节 9.8）
* Header chunk 切分后，退出策略可能变成非常小的单独 chunk
* Hybrid 检索的 `top_n=20` 中该 chunk 未进入候选
* LLM reranker 可能认为“退出策略”并不与某些 chunk 强相关（评分偏低）

**结论：可通过 MultiQuery + 加大 top_n 解决**

---

### ❌ 5. “主要竞争对手有哪些？”

回答为：“上下文无法回答”
（）

但文档有明确竞争分析（4.1）

推测：

* “竞争对手”段落被切分为多个 mid-chunk
* Hybrid 检索分数低，被 Reranker 滤掉了
* MultiQuery 重写方式未包含“竞争者/industry competitor”等关键词

**结论：需要 Query Rewriting（改善语义覆盖）**

---

### ✔ Header 总评

| 项目      | 评价                |
| ------- | ----------------- |
| 检索召回    | 中上                |
| 回答质量    | 高                 |
| 结构化内容召回 | 优秀                |
| 非结构化小节  | Occasionally miss |
| 语义视角问题  | 有待处理（Day 5）       |

👉 **Header 明显是你这份文档的最佳切分方式，但某些段落需要强化检索增广（Day 5 做）。**

---

# 🟦 ② Recursive 法（表现最不稳定）

### 📌 优势

* 某些答案居然非常完整（例如 competitive advantage）
  （但常常靠 LLM 自行补全→危险）

### ❌ 明显缺陷

#### “公司为何有竞争优势？”

答案极长（几百字），包含多个推测、扩写
（）

说明：

* Reranker 未有效过滤噪声
* LLM 在 chunk 不够结构化条件下强行编造连接语

#### “目标市场细分”

内容是对的（）

但 recursive 的命中是“靠运气”，因为切分粒度小。

#### “首年收入”

“5000万元”
（）

但这仍存在视角歧义。

#### “退出策略”、“竞争对手”

都是准确回答
（, ）

但 recursive 自身 chunk 很碎 → **Reranker 有时错误地依赖噪声 chunk**

### Recursive 总评：

❌ 不稳定
❌ 高幻觉风险
❌ 不适合商业计划书
✔ 可作为多路检索的一种补充（Day 5 可做 Multi-Retriever Ensemble）

结论：**不建议用作主检索**。

---

# 🟦 ③ Sliding Window 法（整体中等）

### ❌ “公司为何有竞争优势？”

回答：“上下文无法回答”
（）

→ 说明 sliding window chunk 与 query 匹配不够精确
→ Chunk 之间语义跨度大，Reranker 去掉了相关 chunk

---

### ✔ “目标市场细分”

答案非常正确
（）

Sliding 的优势就在于这类大段落。

---

### ✔ “首年预计收入”

答案 “5000万元”
（）

和 header 一样的问题（视角歧义）。

---

### ✔ “退出策略”

这次找到了正确段落（）

说明：

* Sliding + top20 特别适合找“长段落的中部小节”

---

### ✔ “主要竞争对手”

答案完全正确
（）

→ Sliding window 在“篇幅较大的段落”中非常稳。

---

### Sliding 总评

| 项目      | 评价          |
| ------- | ----------- |
| 大段落召回   | 强           |
| 短段落召回   | 偏弱（可能 miss） |
| 视角问题    | 存在          |
| 回答长度稳定性 | 中等          |

结论：
✔ 适合内容密度高、跨段落信息相关的文档
❌ 对短标题结构不如 header 稳定

---

# 🟩 最终综合总结（非常重要）

## 结论一：**Header 是最佳主检索策略；Sliding 作为补充策略；Recursive 仅作为兜底。**

你的 Day 4 数据已经完全证明：

### **📌 主检索器 = Header-based + Hybrid（BM25 + Vector）**

### **📌 辅助检索器 = Sliding-window（多段落召回补充）**

### **📌 不推荐单独使用 Recursive（容易 hallucination）**

---

## 结论二：你的 Day 4 pipeline 正常工作，但仍有 3 个系统性问题

这些不是代码 bug，而是所有企业级 RAG 都必须处理的问题：

### ① **问题语义歧义（最典型）**

例如：

> “首年预计收入是多少？”

文档中有多个“公司”概念：

* 华道科技
* 亿道信息
* 市场规模预测
  → 模型无法分清你问的是谁。

**这必须在 Day 5 做 Query Disambiguation。**

---

### ② **短段落（如退出策略）容易被滤掉**

原因：

* Header chunk 过短
* Reranker 的评分机制会把它误判为“不重要”
* MultiQuery 重写得不够精准

---

### ③ **信息密度不均导致 chunk 选择偏差**

Sliding 大 chunk → 更可能被 Reranker 选中
Header 小 chunk → 可能被忽略
→ 你需要 Multi-Retriever Fusion（Day 5）

---

# 🟩 Day 4 总体评分（工程级）

| 指标         | 分数（10）                         | 评论                        |
| ---------- | ------------------------------ | ------------------------- |
| 向量检索       | 8                              | 稳定有效                      |
| BM25 检索    | 7                              | 对结构标题敏感，但 chunk 短时可能 miss |
| Hybrid     | 8                              | 召回率提升明显                   |
| Reranker   | 7                              | 对短 chunk 不稳定              |
| MultiQuery | 6                              | 有改善，但重写质量需优化              |
| 回答质量       | 8（Header/Sliding）；4（Recursive） | 取决于 chunk                 |
| 可用性        | 8                              | 已可进入企业级调优阶段               |

整体 Day 4 已达到：

# 🎉 **“可上线前调优阶段”的水平**



