# HelloAgents 0.2.0 — RAGTool 完整框架与实现分析

## 目录

1. [什么是 RAG](#1-什么是-rag)
2. [RAGTool 整体架构](#2-ragtool-整体架构)
3. [核心数据流：从文档到答案](#3-核心数据流从文档到答案)
4. [阶段一：文档解析与转换](#4-阶段一文档解析与转换)
5. [阶段二：智能分块](#5-阶段二智能分块)
6. [阶段三：嵌入向量生成与存储](#6-阶段三嵌入向量生成与存储)
7. [阶段四：检索与召回](#7-阶段四检索与召回)
8. [阶段五：排序与融合](#8-阶段五排序与融合)
9. [阶段六：上下文组装与 LLM 生成](#9-阶段六上下文组装与-llm-生成)
10. [RAGTool 类完整实现](#10-ragtool-类完整实现)
11. [RAG Pipeline 工厂函数](#11-rag-pipeline-工厂函数)
12. [多租户命名空间隔离](#12-多租户命名空间隔离)
13. [与 Agent 的集成方式](#13-与-agent-的集成方式)
14. [完整使用示例](#14-完整使用示例)
15. [配置参考](#15-配置参考)

---

## 1. 什么是 RAG

**RAG（Retrieval-Augmented Generation，检索增强生成）** 是一种将**信息检索**与**大模型生成**相结合的技术：

```
传统 LLM：     用户提问 → LLM 直接回答（依赖训练数据，可能过时/幻觉）
RAG 增强：     用户提问 → 检索知识库 → 将相关内容注入提示词 → LLM 基于真实知识回答
```

**核心价值：** 让大模型的回答基于你自己的文档/知识库，而不是凭空编造。

---

## 2. RAGTool 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAGTool (工具层)                                │
│  name="rag", 6种操作: add_document, add_text, ask, search, stats, clear │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline (管道层)                              │
│  create_rag_pipeline() 返回的函数字典：                                  │
│  ┌──────────────────┐  ┌──────────┐  ┌────────────────┐  ┌──────────┐  │
│  │ add_documents()  │  │ search() │  │search_advanced()│  │get_stats()│ │
│  └────────┬─────────┘  └────┬─────┘  └───────┬────────┘  └──────────┘  │
│           │                 │                 │                          │
│  ┌────────▼──────────────────▼─────────────────▼──────────────────────┐  │
│  │                    核心处理函数                                     │  │
│  │  load_and_chunk_texts()  - 文档加载与分块                          │  │
│  │  index_chunks()          - 嵌入生成与向量存储                      │  │
│  │  search_vectors()        - 基础向量检索                            │  │
│  │  search_vectors_expanded() - 高级检索（MQE + HyDE）               │  │
│  │  rank()                  - 混合排序                                │  │
│  │  merge_snippets_grouped() - 上下文合并与引用                       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   ┌─────────────────┐  ┌───────────────┐   ┌──────────────┐
   │  MarkItDown      │  │ Embedding     │   │ Qdrant       │
   │  (文档转换)       │  │ (嵌入模型)    │   │ (向量数据库)  │
   │  40+格式支持      │  │ DashScope/    │   │ HNSW索引     │
   │  PDF增强处理      │  │ Transformers/ │   │ Payload过滤  │
   └─────────────────┘  │ TF-IDF        │   └──────────────┘
                         └───────────────┘
```

---

## 3. 核心数据流：从文档到答案

### 写入流（文档摄入）

```
文档文件 (PDF/Word/Excel/图片/...)
    │
    ▼
[1] MarkItDown 转换 → Markdown 文本
    │
    ▼
[2] 段落拆分（保留标题层级）→ 段落列表 [{content, heading_path, start, end}]
    │
    ▼
[3] Token 级分块（CJK 感知）→ 分块列表 [{content, heading_path, start, end}]
    │
    ▼
[4] 去重（content_hash）+ 元数据标注 → 最终分块
    │
    ▼
[5] Markdown 预处理（去标记）→ 清洁文本
    │
    ▼
[6] 嵌入模型编码 → 384维向量
    │
    ▼
[7] Qdrant 批量 Upsert → 持久化存储
```

### 读取流（问答）

```
用户问题: "什么是机器学习？"
    │
    ▼
[1] 查询嵌入 → 384维向量
    │
    ├─→ [可选] MQE 查询扩展 → LLM 生成 2 个等价查询
    ├─→ [可选] HyDE 假设文档 → LLM 生成假想答案段落
    │
    ▼
[2] Qdrant 向量检索 → 候选分块 (Top-K × 4 池)
    │
    ▼
[3] 图信号计算（同文档密度 + 邻近度加权）
    │
    ▼
[4] 混合排序: final = 0.7 × vector_score + 0.3 × graph_score
    │
    ▼
[5] 上下文组装: 按文档分组 → 保持阅读顺序 → 添加引用标记
    │
    ▼
[6] Prompt 构建: System Prompt + User Prompt（问题 + 上下文）
    │
    ▼
[7] LLM 生成答案
    │
    ▼
[8] 格式化输出: 答案 + 参考来源 + 性能指标
```

---

## 4. 阶段一：文档解析与转换

**源码位置：** `pipeline.py:49-196`

### 4.1 支持的文件格式（40+种）

| 类别 | 格式 |
|------|------|
| **文档** | PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX |
| **文本** | TXT, MD, CSV, JSON, XML, HTML, HTM |
| **图片** | JPG, JPEG, PNG, GIF, BMP, TIFF, WebP（OCR + 元数据） |
| **音频** | MP3, WAV, M4A, AAC, FLAC, OGG（转录） |
| **档案** | ZIP, TAR, GZ, RAR |
| **代码** | PY, JS, TS, Java, C++, C, H, CSS, SCSS |
| **配置** | LOG, CONF, INI, CFG, YAML, YML, TOML |

### 4.2 通用转换流程

```python
def _convert_to_markdown(path: str) -> str:
    """万能文档转 Markdown"""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pdf':
        return _enhanced_pdf_processing(path)  # PDF 专用增强处理

    # 其他格式使用 MarkItDown
    md_instance = MarkItDown()
    result = md_instance.convert(path)
    return result.text_content
```

### 4.3 PDF 增强处理

PDF 文件有特殊的后处理流程：

```python
def _enhanced_pdf_processing(path):
    # 1. MarkItDown 提取原始文本
    raw_text = MarkItDown().convert(path).text_content

    # 2. 后处理清理
    cleaned_text = _post_process_pdf_text(raw_text)
    return cleaned_text

def _post_process_pdf_text(text):
    # 1. 按行清理
    #    - 移除单字符噪音行
    #    - 移除纯数字行（页码）
    #    - 移除页眉页脚噪音

    # 2. 智能合并短行
    #    - 短于 60 字符的行 → 尝试与下一行合并
    #    - 合并条件：不是标题、不是冒号结尾

    # 3. 重新组织段落
    #    - 标题行、冒号结尾行、长句 → 新段落
    #    - 用双换行分隔段落
```

### 4.4 回退机制

```
MarkItDown 可用？
    ├── 是 → 使用 MarkItDown 转换
    └── 否 → _fallback_text_reader()
                ├── 尝试 UTF-8 编码读取
                └── 尝试 Latin-1 编码读取
```

---

## 5. 阶段二：智能分块

**源码位置：** `pipeline.py:207-389`

### 5.1 CJK 感知的 Token 估算

```python
def _approx_token_len(text: str) -> int:
    """近似 Token 计数"""
    cjk = sum(1 for ch in text if _is_cjk(ch))     # CJK 字符：每字 = 1 token
    non_cjk_tokens = len([t for t in text.split()])  # 非 CJK：按空白分词
    return cjk + non_cjk_tokens
```

### 5.2 段落拆分（保留标题层级）

```python
def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """按段落和标题层级拆分"""
    # 维护标题栈 heading_stack
    # 遇到 # 标题 → 更新栈层级
    # 遇到空行 → 结束当前段落
    # 每个段落记录：
    return [{
        "content": "段落文本内容",
        "heading_path": "一级标题 > 二级标题 > 三级标题",  # 标题路径
        "start": 0,     # 字符偏移量起始
        "end": 150       # 字符偏移量结束
    }, ...]
```

**示例：**

```markdown
# 机器学习
## 监督学习
决策树是一种常用的监督学习算法...

## 无监督学习
K-Means聚类是一种无监督学习方法...
```

**拆分结果：**

```python
[
    {"content": "决策树是...", "heading_path": "机器学习 > 监督学习", "start": 30, "end": 80},
    {"content": "K-Means是...", "heading_path": "机器学习 > 无监督学习", "start": 95, "end": 150}
]
```

### 5.3 Token 级分块（带重叠）

```python
def _chunk_paragraphs(paragraphs, chunk_tokens=800, overlap_tokens=100):
    """将段落按 Token 限制分块"""
    # 策略：
    # 1. 逐个段落累加，直到达到 chunk_tokens 上限
    # 2. 不会在段落中间切断（保持段落完整）
    # 3. 生成块后，保留尾部 overlap_tokens 的段落作为下一块的开头
    #    → 确保上下文连续性

    # 伪代码：
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"])

        if 当前块 token 数 + p_tokens <= chunk_tokens:
            加入当前块
        else:
            输出当前块
            # 重叠：从尾部保留 overlap_tokens
            从当前块尾部截取重叠段落作为新块的开头
```

### 5.4 去重与元数据标注

```python
def load_and_chunk_texts(paths, chunk_size=800, chunk_overlap=100, namespace=None):
    seen_hashes = set()

    for path in paths:
        markdown_text = _convert_to_markdown(path)          # 转 Markdown
        lang = _detect_lang(markdown_text)                   # 检测语言
        doc_id = md5(f"{path}|{len(markdown_text)}")         # 文档 ID

        paragraphs = _split_paragraphs_with_headings(markdown_text)
        chunks = _chunk_paragraphs(paragraphs, chunk_tokens=chunk_size, overlap_tokens=chunk_overlap)

        for chunk in chunks:
            content_hash = md5(chunk["content"])
            if content_hash in seen_hashes:
                continue  # 跳过重复内容
            seen_hashes.add(content_hash)

            chunk_id = md5(f"{doc_id}|{start}|{end}|{content_hash}")
            # 输出分块，附带完整元数据：
            # source_path, file_ext, doc_id, lang, start, end,
            # content_hash, namespace, heading_path, format
```

---

## 6. 阶段三：嵌入向量生成与存储

**源码位置：** `pipeline.py:426-632`, `embedding.py`

### 6.1 Markdown 预处理（嵌入前）

```python
def _preprocess_markdown_for_embedding(text: str) -> str:
    """去除 Markdown 标记，保留语义内容"""
    text = re.sub(r'^#{1,6}\s+', '', text)           # 去掉 # 标题符号
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 去链接保文字
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)   # 去粗体
    text = re.sub(r'\*([^*]+)\*', r'\1', text)        # 去斜体
    text = re.sub(r'`([^`]+)`', r'\1', text)          # 去行内代码
    text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)  # 去代码块
    return text.strip()
```

### 6.2 嵌入模型（三级回退）

```
优先级 1: DashScope（阿里云 text-embedding-v3）
    ├── REST 模式：OpenAI 兼容接口
    └── SDK 模式：DashScope 官方 SDK

优先级 2: LocalTransformerEmbedding（sentence-transformers）
    └── 默认模型：all-MiniLM-L6-v2，384 维

优先级 3: TFIDFEmbedding（scikit-learn TF-IDF）
    └── 无深度学习依赖的轻量回退
```

### 6.3 批量编码与存储

```python
def index_chunks(store, chunks, rag_namespace="default", batch_size=64):
    embedder = get_text_embedder()
    dimension = get_dimension(384)  # 默认 384 维

    # 1. 预处理所有文本
    processed_texts = [_preprocess_markdown_for_embedding(c["content"]) for c in chunks]

    # 2. 分批编码
    vecs = []
    for i in range(0, len(processed_texts), batch_size):
        batch = processed_texts[i:i+batch_size]
        try:
            batch_vecs = embedder.encode(batch)
            # numpy → List[float] 标准化
            vecs.extend(normalize(batch_vecs))
        except:
            # 重试：拆成更小的批次（每次 8 条）
            for small_batch in split(batch, 8):
                time.sleep(2)  # 避免频率限制
                try:
                    small_vecs = embedder.encode(small_batch)
                    vecs.extend(normalize(small_vecs))
                except:
                    vecs.extend([zero_vector] * len(small_batch))  # 最终回退

    # 3. 准备元数据（打上 RAG 标签）
    metas = []
    for chunk in chunks:
        meta = {
            "memory_id": chunk["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",        # RAG 标识
            "content": chunk["content"],        # 原始 Markdown 内容
            "data_source": "rag_pipeline",      # 数据来源标识
            "rag_namespace": rag_namespace,     # 命名空间
            "is_rag_data": True,                # RAG 数据标记
            **chunk["metadata"]                 # 合并分块元数据
        }
        metas.append(meta)

    # 4. 批量写入 Qdrant
    store.add_vectors(vectors=vecs, metadata=metas, ids=[c["id"] for c in chunks])
```

---

## 7. 阶段四：检索与召回

**源码位置：** `pipeline.py:635-805`

### 7.1 查询嵌入

```python
def embed_query(query: str) -> List[float]:
    """将用户查询转为向量"""
    embedder = get_text_embedder()
    vec = embedder.encode(query)
    # 标准化为 List[float]，维度校验，必要时补零或截断
    return normalized_vec  # 384 维
```

### 7.2 基础向量检索

```python
def search_vectors(store, query, top_k=8, rag_namespace=None, score_threshold=None):
    qv = embed_query(query)

    # 构建过滤条件：只检索 RAG 数据
    where = {
        "memory_type": "rag_chunk",
        "is_rag_data": True,
        "data_source": "rag_pipeline"
    }
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    return store.search_similar(
        query_vector=qv,
        limit=top_k,
        score_threshold=score_threshold,
        where=where
    )
```

### 7.3 高级检索：MQE + HyDE 查询扩展

```python
def search_vectors_expanded(store, query, top_k=8, enable_mqe=False, enable_hyde=False, ...):
    expansions = [query]  # 原始查询

    # MQE（Multi-Query Expansion）：LLM 生成语义等价查询
    if enable_mqe:
        # Prompt: "生成语义等价或互补的多样化查询"
        # 例如 "什么是机器学习" → ["机器学习的定义", "ML概念解释"]
        expansions.extend(_prompt_mqe(query, n=2))

    # HyDE（Hypothetical Document Embeddings）：LLM 生成假想答案
    if enable_hyde:
        # Prompt: "根据问题写一段可能的答案性段落"
        # 例如 "什么是机器学习" → "机器学习是人工智能的分支，通过数据训练..."
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)

    # 为每个扩展查询执行检索
    pool = top_k * 4  # 4 倍候选池
    per_expansion = pool // len(expansions)

    all_results = {}
    for q in expansions:
        qv = embed_query(q)
        hits = store.search_similar(query_vector=qv, limit=per_expansion, where=where)
        for hit in hits:
            mid = hit["metadata"]["memory_id"]
            # 保留最高分
            if mid not in all_results or hit["score"] > all_results[mid]["score"]:
                all_results[mid] = hit

    # 按分数排序，返回 Top-K
    return sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]
```

**MQE 查询扩展示意：**

```
原始查询: "Python的发展历史"
   │
   ├─→ MQE 扩展 1: "Python编程语言的起源和版本演变"
   ├─→ MQE 扩展 2: "Python语言历年重大更新"
   │
   ├─→ 分别检索 Qdrant
   │
   └─→ 合并去重，保留最高分 → 最终候选集
```

**HyDE 假设文档示意：**

```
原始查询: "什么是RAG技术"
   │
   ├─→ HyDE 生成: "RAG是检索增强生成的缩写，它结合了信息检索系统
   │              和大语言模型，通过先检索相关文档再生成回答的方式..."
   │
   └─→ 用这段假想答案的向量去检索 → 可能匹配到更精确的知识块
```

---

## 8. 阶段五：排序与融合

**源码位置：** `pipeline.py:831-1107`

### 8.1 图信号计算

```python
def compute_graph_signals_from_pool(vector_hits, same_doc_weight=1.0,
                                     proximity_weight=1.0, proximity_window_chars=1600):
    """基于文档结构计算图信号"""

    # 按文档分组
    by_doc = group_by(vector_hits, key="doc_id")

    for doc_id, chunks in by_doc.items():
        chunks.sort(by="start")  # 按位置排序

        # 同文档密度分数
        density = len(chunks) / max_count  # 该文档的块数 / 最大块数

        # 邻近度分数（双指针扫描）
        for i, chunk in enumerate(chunks):
            prox_acc = 0.0
            # 向左扫描
            for j in range(i-1, -1, -1):
                dist = abs(chunk.start - chunks[j].start)
                if dist > 1600: break  # 超过窗口
                prox_acc += 1.0 - (dist / 1600.0)
            # 向右扫描
            for j in range(i+1, len(chunks)):
                dist = abs(chunk.start - chunks[j].start)
                if dist > 1600: break
                prox_acc += 1.0 - (dist / 1600.0)

            # 综合信号
            signal = same_doc_weight * density + proximity_weight * prox_acc

    # 归一化到 [0, 1]
```

**设计意图：** 如果一个文档有多个块被检索命中（密度高），且这些块在原文中位置相近（邻近度高），说明这个文档与查询高度相关，应该被提升排名。

### 8.2 混合排序

```python
def rank(vector_hits, graph_signals, w_vector=0.7, w_graph=0.3):
    """向量分数 + 图信号 混合排序"""
    for hit in vector_hits:
        mid = hit["metadata"]["memory_id"]
        v = hit["score"]                    # 向量相似度
        g = graph_signals.get(mid, 0.0)     # 图信号
        final_score = 0.7 * v + 0.3 * g     # 加权融合
    # 按 final_score 降序排序
```

### 8.3 Cross-Encoder 重排序（可选）

```python
def rerank_with_cross_encoder(query, items, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """使用交叉编码器精排"""
    ce = CrossEncoder(model_name)
    pairs = [[query, item["content"]] for item in items]
    scores = ce.predict(pairs)  # 对每个 (query, doc) 对计算相关性
    # 按 rerank_score 重新排序
```

### 8.4 邻居扩展

```python
def expand_neighbors_from_pool(selected, pool, neighbors=1, max_additions=5):
    """扩展已选块的前后邻居"""
    # 对每个已选块，在同文档的候选池中查找位置上相邻的块
    # 左右各扩展 neighbors 个位置
    # 最多添加 max_additions 个新块
```

### 8.5 块压缩

```python
def compress_ranked_items(ranked_items, max_per_doc=2, join_gap=200):
    """压缩排序结果"""
    # 1. 每个文档最多保留 max_per_doc 个块
    # 2. 如果同文档中两个块的位置差 <= join_gap 字符，合并为一个块
    # → 减少冗余，保证上下文简洁
```

---

## 9. 阶段六：上下文组装与 LLM 生成

**源码位置：** `rag_tool.py:383-554`

### 9.1 _ask() 方法：核心问答流程

```python
def _ask(self, question=None, query=None, limit=5, enable_advanced_search=True,
         include_citations=True, max_chars=1200, namespace=None, **kwargs):

    user_question = question or query

    # ===== 1. 检索相关内容 =====
    pipeline = self._get_pipeline(namespace)
    if enable_advanced_search:
        results = pipeline["search_advanced"](
            query=user_question, top_k=limit,
            enable_mqe=True, enable_hyde=True)
    else:
        results = pipeline["search"](query=user_question, top_k=limit)

    if not results:
        return "抱歉，知识库中没有找到相关信息。"

    # ===== 2. 整理上下文 =====
    context_parts = []
    citations = []
    for i, result in enumerate(results):
        content = result["metadata"]["content"].strip()
        source = result["metadata"]["source_path"]
        score = result["score"]

        cleaned = self._clean_content_for_context(content)  # 去空白，截取300字
        context_parts.append(f"片段 {i+1}：{cleaned}")
        citations.append({"index": i+1, "source": basename(source), "score": score})

    context = "\n\n".join(context_parts)
    if len(context) > max_chars:
        context = self._smart_truncate_context(context, max_chars)

    # ===== 3. 构建 Prompt =====
    system_prompt = self._build_system_prompt()
    user_prompt = self._build_user_prompt(user_question, context)

    # ===== 4. 调用 LLM =====
    answer = self.llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ])

    # ===== 5. 格式化输出 =====
    return self._format_final_answer(question, answer, citations, search_time, llm_time, avg_score)
```

### 9.2 System Prompt

```python
def _build_system_prompt(self):
    return """你是一个专业的知识助手，具备以下能力：
1. 📖 精准理解：仔细理解用户问题的核心意图
2. 🎯 可信回答：严格基于提供的上下文信息回答，不编造内容
3. 🔍 信息整合：从多个片段中提取关键信息，形成完整答案
4. 💡 清晰表达：用简洁明了的语言回答，适当使用结构化格式
5. 🚫 诚实表达：如果上下文不足以回答问题，请坦诚说明

回答格式要求：
• 直接回答核心问题
• 必要时使用要点或步骤
• 引用关键原文时使用引号
• 避免重复和冗余"""
```

### 9.3 User Prompt

```python
def _build_user_prompt(self, question, context):
    return f"""请基于以下上下文信息回答问题：

【问题】{question}

【相关上下文】
{context}

【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。"""
```

### 9.4 输出格式

```
🤖 **智能问答结果**

机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。
主要包括三种类型：
1. 监督学习 - 使用标注数据训练
2. 无监督学习 - 发现数据中的隐藏模式
3. 强化学习 - 通过奖惩信号学习策略

📚 **参考来源**
🟢 [1] ml_basics.md (相似度: 0.892)
🟡 [2] ai_overview.md (相似度: 0.756)

⚡ 检索: 245ms | 生成: 1234ms | 平均相似度: 0.824
```

**相似度颜色编码：**
- 🟢 绿色：> 0.8（高度相关）
- 🟡 黄色：0.6 - 0.8（中度相关）
- 🔵 蓝色：< 0.6（低度相关）

---

## 10. RAGTool 类完整实现

**源码位置：** `hello_agents/tools/builtin/rag_tool.py`

### 10.1 构造函数

```python
class RAGTool(Tool):
    def __init__(self,
        knowledge_base_path="./knowledge_base",   # 本地目录（临时文件）
        qdrant_url=None,                           # Qdrant 云 URL
        qdrant_api_key=None,                       # Qdrant API Key
        collection_name="rag_knowledge_base",      # Qdrant 集合名
        rag_namespace="default"                    # 默认命名空间
    ):
        super().__init__(name="rag", description="RAG工具 - ...")
        self._pipelines = {}           # 命名空间 → Pipeline 映射
        self.llm = HelloAgentsLLM()    # LLM 实例用于答案生成
        # 初始化默认命名空间的 Pipeline
        self._pipelines[rag_namespace] = create_rag_pipeline(...)
```

### 10.2 支持的 6 种操作

| 操作 | 方法 | 说明 |
|------|------|------|
| `add_document` | `_add_document()` | 添加文件到知识库（支持 40+ 格式） |
| `add_text` | `_add_text()` | 添加纯文本到知识库 |
| `ask` | `_ask()` | **核心：检索 + LLM 问答** |
| `search` | `_search()` | 纯检索（不调用 LLM 生成） |
| `stats` | `_get_stats()` | 知识库统计信息 |
| `clear` | `_clear_knowledge_base()` | 清空知识库（需 confirm=true） |

### 10.3 参数定义

```python
def get_parameters(self):
    return [
        ToolParameter(name="action",    type="string",  required=True,  description="操作类型"),
        ToolParameter(name="file_path", type="string",  required=False, description="文档路径"),
        ToolParameter(name="text",      type="string",  required=False, description="文本内容"),
        ToolParameter(name="question",  type="string",  required=False, description="用户问题"),
        ToolParameter(name="query",     type="string",  required=False, description="搜索查询"),
        ToolParameter(name="namespace", type="string",  required=False, default="default"),
        ToolParameter(name="limit",     type="integer", required=False, default=5),
        ToolParameter(name="include_citations", type="boolean", required=False, default=True),
    ]
```

### 10.4 默认参数

```python
defaults = {
    "namespace": "default",
    "limit": 5,                    # 返回 Top-5 结果
    "include_citations": True,     # 包含引用来源
    "enable_advanced_search": True, # 启用 MQE + HyDE
    "max_chars": 1200,             # 上下文最大字符数
    "min_score": 0.1,              # 最低相似度阈值
    "chunk_size": 800,             # 分块大小（Token）
    "chunk_overlap": 100           # 分块重叠（Token）
}
```

### 10.5 便捷方法

```python
# 简化调用
rag_tool.add_document("document.pdf")
rag_tool.add_text("Python是一种编程语言...")
rag_tool.ask("什么是Python？")
rag_tool.search("编程语言")

# 批量操作
rag_tool.add_documents_batch(["doc1.pdf", "doc2.pdf", "doc3.docx"])
rag_tool.add_texts_batch(["文本1", "文本2"], document_ids=["id1", "id2"])
```

---

## 11. RAG Pipeline 工厂函数

**源码位置：** `pipeline.py:1130-1207`

```python
def create_rag_pipeline(qdrant_url=None, qdrant_api_key=None,
                         collection_name="hello_agents_rag_vectors",
                         rag_namespace="default") -> Dict[str, Any]:
    """创建完整的 RAG 管道，返回函数字典"""

    # 1. 创建 Qdrant 向量存储
    dimension = get_dimension(384)
    store = QdrantVectorStore(
        url=qdrant_url, api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_size=dimension, distance="cosine"
    )

    # 2. 定义管道函数（闭包绑定 store 和 namespace）
    def add_documents(file_paths, chunk_size=800, chunk_overlap=100):
        chunks = load_and_chunk_texts(file_paths, chunk_size, chunk_overlap, namespace=rag_namespace)
        index_chunks(store=store, chunks=chunks, rag_namespace=rag_namespace)
        return len(chunks)

    def search(query, top_k=8, score_threshold=None):
        return search_vectors(store=store, query=query, top_k=top_k,
                              rag_namespace=rag_namespace, score_threshold=score_threshold)

    def search_advanced(query, top_k=8, enable_mqe=False, enable_hyde=False, score_threshold=None):
        return search_vectors_expanded(store=store, query=query, top_k=top_k,
                                       rag_namespace=rag_namespace,
                                       enable_mqe=enable_mqe, enable_hyde=enable_hyde,
                                       score_threshold=score_threshold)

    def get_stats():
        return store.get_collection_stats()

    # 3. 返回管道字典
    return {
        "store": store,               # Qdrant 实例
        "namespace": rag_namespace,    # 命名空间
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats
    }
```

**设计特点：** Pipeline 是一组**闭包函数**，通过捕获 `store` 和 `rag_namespace` 变量，使得调用者无需关心底层存储细节。

---

## 12. 多租户命名空间隔离

```python
class RAGTool:
    def __init__(self, ...):
        self._pipelines: Dict[str, Dict] = {}  # 命名空间 → Pipeline

    def _get_pipeline(self, namespace=None):
        """懒加载：按需创建命名空间 Pipeline"""
        target_ns = namespace or self.rag_namespace
        if target_ns not in self._pipelines:
            self._pipelines[target_ns] = create_rag_pipeline(
                ..., rag_namespace=target_ns
            )
        return self._pipelines[target_ns]
```

**隔离机制：** 同一个 Qdrant 集合中，通过 `rag_namespace` 字段的 Payload 过滤实现逻辑隔离。

```
Qdrant Collection: "rag_knowledge_base"
├── rag_namespace="project_a"   → 项目 A 的文档
├── rag_namespace="project_b"   → 项目 B 的文档
└── rag_namespace="default"     → 默认知识库
```

---

## 13. 与 Agent 的集成方式

### 13.1 注册与调用

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import RAGTool

# 1. 创建
rag_tool = RAGTool(knowledge_base_path="./kb", collection_name="my_kb")

# 2. 注册到工具注册表
registry = ToolRegistry()
registry.register_tool(rag_tool)

# 3. 绑定到 Agent
agent = SimpleAgent(name="知识助手", llm=HelloAgentsLLM(), tool_registry=registry)

# 4. Agent 使用
response = agent.run("RAG技术是什么？")
```

### 13.2 Agent 自动调用流程

```
用户: "什么是机器学习？"
  │
  ▼
Agent 构建 System Prompt（包含 rag 工具描述）
  │
  ▼
LLM 判断需要查询知识库，输出:
  "让我查一下。[TOOL_CALL:rag:action=ask,question=什么是机器学习]"
  │
  ▼
Agent 解析 → _parse_tool_calls()
  │
  ▼
Agent 执行 → RAGTool.run({action: "ask", question: "什么是机器学习"})
  │
  ├─→ 检索 Qdrant（MQE + HyDE）
  ├─→ 组装上下文
  ├─→ LLM 生成答案
  └─→ 返回: "🤖 智能问答结果\n\n机器学习是..."
  │
  ▼
Agent 将结果反馈给 LLM
  │
  ▼
LLM 基于工具结果生成最终回答
```

---

## 14. 完整使用示例

### 14.1 基本使用

```python
from hello_agents.tools import RAGTool

# 初始化
rag = RAGTool(
    knowledge_base_path="./knowledge_base",
    collection_name="my_project_kb",
    rag_namespace="default"
)

# === 添加文档 ===
# 添加 PDF
rag.execute("add_document", file_path="report.pdf")

# 添加文本
rag.execute("add_text",
    text="Python是由Guido van Rossum于1991年首次发布的高级编程语言。",
    document_id="python_intro")

# 批量添加
rag.add_documents_batch(["doc1.pdf", "doc2.docx", "data.xlsx"])
rag.add_texts_batch(
    texts=["文本1内容", "文本2内容"],
    document_ids=["text_1", "text_2"]
)

# === 搜索 ===
result = rag.execute("search",
    query="Python编程语言",
    limit=3,
    min_score=0.1)
print(result)

# === 智能问答（核心功能）===
answer = rag.execute("ask",
    question="Python是什么时候发布的？",
    include_citations=True,
    enable_advanced_search=True)
print(answer)

# === 统计 ===
stats = rag.execute("stats")
print(stats)

# === 清空 ===
rag.execute("clear", confirm=True)
```

### 14.2 多命名空间使用

```python
rag = RAGTool(collection_name="multi_project_kb")

# 项目 A 的知识库
rag.execute("add_text",
    text="项目A使用React和TypeScript...",
    namespace="project_a")

# 项目 B 的知识库
rag.execute("add_text",
    text="项目B使用Django和Python...",
    namespace="project_b")

# 在项目 A 的范围内搜索
result_a = rag.execute("ask",
    question="用了什么前端框架？",
    namespace="project_a")

# 在项目 B 的范围内搜索
result_b = rag.execute("ask",
    question="后端技术栈是什么？",
    namespace="project_b")
```

### 14.3 与 Agent 集成

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import RAGTool

# 初始化
llm = HelloAgentsLLM()
rag_tool = RAGTool(collection_name="agent_kb", rag_namespace="test")

# 注册
registry = ToolRegistry()
registry.register_tool(rag_tool)

# 先导入知识
rag_tool.execute("add_text",
    text="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。",
    document_id="ml_basics")

# 创建 Agent
agent = SimpleAgent(
    name="知识助手", llm=llm, tool_registry=registry,
    system_prompt="你是一个有知识库的智能助手，可以使用 rag 工具检索和回答问题。"
)

# 对话（Agent 自动调用 RAGTool）
response = agent.run("请问什么是机器学习？")
print(response)
```

---

## 15. 配置参考

### 环境变量

```bash
# Qdrant 向量数据库
QDRANT_URL=https://your-qdrant-instance.cloud       # Qdrant 云服务 URL
QDRANT_API_KEY=your_api_key                           # Qdrant API 密钥
QDRANT_COLLECTION=hello_agents_rag_vectors            # 集合名称

# Qdrant HNSW 调优
QDRANT_HNSW_M=32                 # HNSW 图度数（越大越准，越慢）
QDRANT_HNSW_EF_CONSTRUCT=256    # 构建时搜索深度
QDRANT_SEARCH_EF=128            # 查询时搜索深度
QDRANT_SEARCH_EXACT=0           # 是否精确搜索（0=HNSW, 1=暴力）

# 嵌入模型
EMBED_MODEL_TYPE=local           # local | dashscope | tfidf
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBED_API_KEY=your_embedding_api_key
EMBED_BASE_URL=https://...       # DashScope REST 模式的 base URL

# 模型下载镜像
HF_ENDPOINT=https://hf-mirror.com
```

### RAGTool 参数

```python
RAGTool(
    knowledge_base_path="./knowledge_base",  # 本地临时目录
    qdrant_url=None,                         # 覆盖环境变量
    qdrant_api_key=None,                     # 覆盖环境变量
    collection_name="rag_knowledge_base",    # Qdrant 集合名
    rag_namespace="default"                  # 默认命名空间
)
```

### 分块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 800 | 每块最大 Token 数 |
| `chunk_overlap` | 100 | 块间重叠 Token 数 |
| `max_chars` | 1200 | 注入 LLM 的上下文最大字符数 |

### 检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `limit` | 5 | 返回 Top-K 结果 |
| `min_score` | 0.1 | 最低相似度阈值 |
| `enable_advanced_search` | True | 是否启用 MQE + HyDE |
| `include_citations` | True | 是否包含引用来源 |
| `candidate_pool_multiplier` | 4 | 候选池倍数（内部） |

---

## 总结

HelloAgents 0.2.0 的 RAGTool 是一个**完整的端到端 RAG 系统**：

| 阶段 | 核心技术 | 关键函数 |
|------|---------|---------|
| **文档解析** | MarkItDown（40+格式）+ PDF 增强 | `_convert_to_markdown()` |
| **智能分块** | CJK 感知 Token 分块 + 标题层级保持 | `_chunk_paragraphs()` |
| **嵌入存储** | 三级回退嵌入 + Qdrant HNSW 索引 | `index_chunks()` |
| **智能检索** | 基础向量搜索 + MQE/HyDE 查询扩展 | `search_vectors_expanded()` |
| **混合排序** | 向量 × 0.7 + 图信号 × 0.3 + 可选 CrossEncoder | `rank()` |
| **上下文组装** | 文档分组 + 阅读顺序 + 智能截断 + 引用 | `merge_snippets_grouped()` |
| **答案生成** | System + User Prompt → LLM → 格式化输出 | `_ask()` |
| **多租户** | 命名空间隔离 + 懒加载 Pipeline | `_get_pipeline()` |
