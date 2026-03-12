# HelloAgents 0.2.0 — MemoryTool 与大模型交互全流程分析

## 目录

1. [交互架构总览](#1-交互架构总览)
2. [核心交互机制：Prompt-Based Tool Calling](#2-核心交互机制prompt-based-tool-calling)
3. [完整交互生命周期](#3-完整交互生命周期)
4. [WorkingMemory 完整工作流程](#4-workingmemory-完整工作流程)
5. [EpisodicMemory 完整工作流程](#5-episodicmemory-完整工作流程)
6. [SemanticMemory 完整工作流程](#6-semanticmemory-完整工作流程)
7. [PerceptualMemory 完整工作流程](#7-perceptualmemory-完整工作流程)
8. [记忆检索与大模型的闭环](#8-记忆检索与大模型的闭环)
9. [多轮工具调用（ReAct 模式）](#9-多轮工具调用react-模式)
10. [自动对话记录机制](#10-自动对话记录机制)
11. [关键设计决策分析](#11-关键设计决策分析)

---

## 1. 交互架构总览

HelloAgents 0.2.0 采用 **Prompt-Based Tool Calling（基于提示词的工具调用）** 模式，而非 OpenAI 原生的 `tools` / `function_calling` 参数。这意味着：

- 工具定义以**自然语言文本**注入到 System Prompt 中
- LLM 以**文本标记** `[TOOL_CALL:...]` 的形式输出工具调用
- Agent 层通过**正则解析**提取工具调用并执行
- 工具结果以**新的 user 消息**形式反馈给 LLM

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        完整交互闭环                                      │
│                                                                          │
│   用户输入                                                               │
│      │                                                                   │
│      ▼                                                                   │
│   ┌──────────────────────────────────────┐                              │
│   │  SimpleAgent.run()                   │                              │
│   │  ┌─────────────────────────────────┐ │                              │
│   │  │ 1. 构建 System Prompt          │ │  ← 注入工具描述文本           │
│   │  │    （包含工具说明和调用格式）    │ │                              │
│   │  └─────────────┬───────────────────┘ │                              │
│   │                ▼                     │                              │
│   │  ┌─────────────────────────────────┐ │                              │
│   │  │ 2. 调用 LLM                    │ │  → messages[] 发送给大模型    │
│   │  │    llm.invoke(messages)         │ │  ← 返回包含 [TOOL_CALL] 的文本│
│   │  └─────────────┬───────────────────┘ │                              │
│   │                ▼                     │                              │
│   │  ┌─────────────────────────────────┐ │                              │
│   │  │ 3. 解析工具调用                │ │  正则: \[TOOL_CALL:...\]      │
│   │  │    _parse_tool_calls(response)  │ │                              │
│   │  └─────────────┬───────────────────┘ │                              │
│   │                ▼                     │                              │
│   │  ┌─────────────────────────────────┐ │                              │
│   │  │ 4. 执行工具                    │ │                              │
│   │  │    MemoryTool.run(param_dict)   │ │  → 调用 MemoryManager        │
│   │  │                                 │ │  ← 返回操作结果字符串        │
│   │  └─────────────┬───────────────────┘ │                              │
│   │                ▼                     │                              │
│   │  ┌─────────────────────────────────┐ │                              │
│   │  │ 5. 将结果反馈给 LLM           │ │  ← 作为 "user" 消息注入      │
│   │  │    "工具执行结果：..."         │ │                              │
│   │  └─────────────┬───────────────────┘ │                              │
│   │                ▼                     │                              │
│   │  ┌─────────────────────────────────┐ │                              │
│   │  │ 6. LLM 再次生成                │ │  基于工具结果生成最终回答     │
│   │  │    （最终回答，无工具调用）      │ │                              │
│   │  └─────────────────────────────────┘ │                              │
│   └──────────────────────────────────────┘                              │
│      │                                                                   │
│      ▼                                                                   │
│   返回最终回答给用户                                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心交互机制：Prompt-Based Tool Calling

### 2.1 System Prompt 构建

**源码位置：** `simple_agent.py:41-63`

当 SimpleAgent 启用了工具调用，它会在 System Prompt 中附加工具描述和调用格式说明：

```python
def _get_enhanced_system_prompt(self) -> str:
    base_prompt = self.system_prompt or "你是一个有用的AI助手。"

    # 获取工具描述
    tools_description = self.tool_registry.get_tools_description()
    # 输出示例: "- memory: 记忆工具 - 可以存储和检索对话历史、知识和经验"

    tools_section = "\n\n## 可用工具\n"
    tools_section += "你可以使用以下工具来帮助回答问题：\n"
    tools_section += tools_description + "\n"

    tools_section += "\n## 工具调用格式\n"
    tools_section += "当需要使用工具时，请使用以下格式：\n"
    tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
    tools_section += "例如：`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
    tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

    return base_prompt + tools_section
```

**实际发送给 LLM 的 System Prompt 示例：**

```
你是一个有记忆能力的智能助手。

## 可用工具
你可以使用以下工具来帮助回答问题：
- memory: 记忆工具 - 可以存储和检索对话历史、知识和经验

## 工具调用格式
当需要使用工具时，请使用以下格式：
`[TOOL_CALL:{tool_name}:{parameters}]`
例如：`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`

工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。
```

### 2.2 LLM 输出解析

**源码位置：** `simple_agent.py:65-78`

Agent 使用正则表达式从 LLM 输出中提取工具调用：

```python
def _parse_tool_calls(self, text: str) -> list:
    pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
    matches = re.findall(pattern, text)

    tool_calls = []
    for tool_name, parameters in matches:
        tool_calls.append({
            'tool_name': tool_name.strip(),
            'parameters': parameters.strip(),
            'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
        })
    return tool_calls
```

**LLM 输出示例：**

```
好的，我帮你记住。[TOOL_CALL:memory:action=add,content=用户叫张三,memory_type=semantic,importance=0.8]
```

**解析结果：**

```python
[{
    'tool_name': 'memory',
    'parameters': 'action=add,content=用户叫张三,memory_type=semantic,importance=0.8',
    'original': '[TOOL_CALL:memory:action=add,content=用户叫张三,memory_type=semantic,importance=0.8]'
}]
```

### 2.3 智能参数解析

**源码位置：** `simple_agent.py:101-159`

SimpleAgent 实现了一套智能的参数解析机制，支持多种 LLM 输出格式：

```python
def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
    param_dict = {}

    if '=' in parameters:
        if ',' in parameters:
            # 多参数: "action=add,content=张三,memory_type=semantic"
            pairs = parameters.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    param_dict[key.strip()] = value.strip()
        else:
            # 单参数: "query=Python"
            key, value = parameters.split('=', 1)
            param_dict[key.strip()] = value.strip()

        # 智能推断 action（如果 LLM 没有明确指定）
        if 'action' not in param_dict:
            param_dict = self._infer_action(tool_name, param_dict)
    else:
        # 直接传入值: "Python编程" → 推断为搜索
        param_dict = self._infer_simple_parameters(tool_name, parameters)

    return param_dict
```

**智能 Action 推断（专门为 memory 工具设计）：**

```python
def _infer_action(self, tool_name, param_dict):
    if tool_name == 'memory':
        if 'recall' in param_dict:       # recall=xxx → search
            param_dict['action'] = 'search'
            param_dict['query'] = param_dict.pop('recall')
        elif 'store' in param_dict:      # store=xxx → add
            param_dict['action'] = 'add'
            param_dict['content'] = param_dict.pop('store')
        elif 'query' in param_dict:      # 有 query → search
            param_dict['action'] = 'search'
        elif 'content' in param_dict:    # 有 content → add
            param_dict['action'] = 'add'
    return param_dict
```

**支持的参数格式示例：**

| LLM 输出 | 解析结果 |
|----------|---------|
| `[TOOL_CALL:memory:action=add,content=张三,memory_type=semantic]` | `{action: add, content: 张三, memory_type: semantic}` |
| `[TOOL_CALL:memory:query=用户信息]` | `{action: search, query: 用户信息}` |
| `[TOOL_CALL:memory:recall=张三]` | `{action: search, query: 张三}` |
| `[TOOL_CALL:memory:store=张三是开发者]` | `{action: add, content: 张三是开发者}` |
| `[TOOL_CALL:memory:Python编程]` | `{action: search, query: Python编程}` |

### 2.4 工具执行与结果回传

**源码位置：** `simple_agent.py:80-99` 和 `simple_agent.py:198-228`

```python
# 执行工具
def _execute_tool_call(self, tool_name, parameters):
    tool = self.tool_registry.get_tool(tool_name)  # 获取 MemoryTool 实例
    param_dict = self._parse_tool_parameters(tool_name, parameters)
    result = tool.run(param_dict)  # MemoryTool.run() 返回结果字符串
    return f"🔧 工具 {tool_name} 执行结果：\n{result}"

# 主循环中：将结果作为新消息反馈给 LLM
messages.append({"role": "assistant", "content": clean_response})  # 去掉 [TOOL_CALL] 标记
messages.append({
    "role": "user",
    "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"
})
# 再次调用 LLM（带有工具结果上下文）
response = self.llm.invoke(messages)
```

**关键设计：** 工具结果以 `role: "user"` 的消息注入，而不是 `role: "tool"`，因为使用的是文本解析模式而非 OpenAI 原生工具调用。

---

## 3. 完整交互生命周期

### 示例场景："记住我叫张三"

以下是从用户输入到最终回答的完整执行链路：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 1：用户输入
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户: "记住我叫张三"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 2：SimpleAgent 构建 messages
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

messages = [
    {
        "role": "system",
        "content": "你是一个有记忆能力的智能助手。\n\n## 可用工具\n...\n## 工具调用格式\n..."
    },
    // ... 历史消息 ...
    {
        "role": "user",
        "content": "记住我叫张三"
    }
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 3：第一次 LLM 调用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

response = llm.invoke(messages)
→ LLM 返回：
  "好的，我来帮你记住。[TOOL_CALL:memory:action=add,content=用户叫张三,memory_type=semantic,importance=0.8]"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 4：解析工具调用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tool_calls = _parse_tool_calls(response)
→ [{
      tool_name: "memory",
      parameters: "action=add,content=用户叫张三,memory_type=semantic,importance=0.8"
  }]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 5：参数解析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

param_dict = _parse_tool_parameters("memory", "action=add,content=用户叫张三,...")
→ {
      "action": "add",
      "content": "用户叫张三",
      "memory_type": "semantic",
      "importance": "0.8"
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 6：MemoryTool.run() 执行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MemoryTool.run(param_dict)
  → validate_parameters()   ✅
  → execute("add", content="用户叫张三", memory_type="semantic", importance=0.8)
  → _add_memory(...)
    → MemoryManager.add_memory(...)
      → 创建 MemoryItem(id=UUID, content="用户叫张三", ...)
      → memory_types["semantic"].add(memory_item)
        → 嵌入向量生成 → Qdrant 存储
        → 实体提取 → Neo4j 存储
        → 内存缓存更新
  → 返回: "✅ 记忆已添加 (ID: abc12345...)"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 7：结果反馈给 LLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

messages.append({
    "role": "assistant",
    "content": "好的，我来帮你记住。"   // 去掉 [TOOL_CALL] 标记
})
messages.append({
    "role": "user",
    "content": "工具执行结果：\n🔧 工具 memory 执行结果：\n✅ 记忆已添加 (ID: abc12345...)\n\n请基于这些结果给出完整的回答。"
})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 8：第二次 LLM 调用（最终回答）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

response = llm.invoke(messages)
→ LLM 返回（无 [TOOL_CALL]）：
  "好的，我已经记住你叫张三了。以后有什么需要帮忙的，随时告诉我！"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段 9：保存历史 & 返回
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

history.append(Message("记住我叫张三", "user"))
history.append(Message("好的，我已经记住你叫张三了。...", "assistant"))
return final_response
```

---

## 4. WorkingMemory 完整工作流程

### 认知定位
**工作记忆** = 人类的短期记忆，容量有限，快速存取，会自动过期。

### 写入流程

```
用户: "我正在讨论项目架构"
  │
  ▼
LLM → [TOOL_CALL:memory:action=add,content=当前正在讨论项目架构]
  │
  ▼
SimpleAgent._parse_tool_parameters()
→ {action: "add", content: "当前正在讨论项目架构"}
  │  （注意：未指定 memory_type，默认为 "working"）
  ▼
MemoryTool.run(param_dict)
→ MemoryTool.execute("add", content="当前正在讨论项目架构")
→ MemoryTool._add_memory(content="当前正在讨论项目架构", memory_type="working", importance=0.5)
  │
  ▼
MemoryManager.add_memory(content, memory_type="working", importance=0.5, auto_classify=False)
→ 创建 MemoryItem(id=UUID, content="当前正在讨论项目架构",
                   memory_type="working", importance=0.5, timestamp=now)
  │
  ▼
WorkingMemory.add(memory_item)
  │
  ├─→ 1. _expire_old_memories()          // 清除 TTL 过期记忆（120分钟）
  │     cutoff = now - 120分钟
  │     移除所有 timestamp < cutoff 的记忆
  │     重建优先级堆
  │
  ├─→ 2. _calculate_priority(memory)      // 计算优先级
  │     priority = importance * time_decay
  │     time_decay = 0.95^(hours/6)       // 指数衰减
  │
  ├─→ 3. heapq.heappush(heap, (-priority, timestamp, memory))  // 入堆
  │     self.memories.append(memory)       // 入列表
  │
  ├─→ 4. current_tokens += len(content.split())  // 更新 token 计数
  │
  └─→ 5. _enforce_capacity_limits()       // 强制容量限制
        while len(memories) > 10:          // 超过10条 → 淘汰最低优先级
            _remove_lowest_priority_memory()
        while current_tokens > 2000:       // 超过2000 tokens → 淘汰
            _remove_lowest_priority_memory()
```

**存储位置：** 纯内存（Python List + heapq），不持久化

### 读取流程

```
用户: "我们之前在讨论什么？"
  │
  ▼
LLM → [TOOL_CALL:memory:action=search,query=之前讨论的内容]
  │
  ▼
MemoryTool._search_memory(query="之前讨论的内容", limit=5)
  │
  ▼
MemoryManager.retrieve_memories(query="之前讨论的内容", memory_types=None)
  │  （memory_types=None → 查询所有启用的类型）
  │
  ▼
WorkingMemory.retrieve(query="之前讨论的内容", limit=5)
  │
  ├─→ 1. _expire_old_memories()                // 先清除过期
  │
  ├─→ 2. 过滤: active_memories (未标记 forgotten)
  │
  ├─→ 3. TF-IDF 向量搜索（尝试）
  │     from sklearn.feature_extraction.text import TfidfVectorizer
  │     documents = [query] + [m.content for m in memories]
  │     tfidf_matrix = TfidfVectorizer().fit_transform(documents)
  │     similarities = cosine_similarity(query_vector, doc_vectors)
  │     → vector_scores = {memory_id: similarity_score}
  │     如果失败 → vector_scores = {} （回退到纯关键词）
  │
  ├─→ 4. 关键词匹配
  │     if query in content:
  │         keyword_score = len(query) / len(content)
  │     else:
  │         keyword_score = |query_words ∩ content_words| / |query_words ∪ content_words| * 0.8
  │
  ├─→ 5. 混合评分
  │     if vector_score > 0:
  │         base_relevance = vector_score * 0.7 + keyword_score * 0.3
  │     else:
  │         base_relevance = keyword_score
  │
  ├─→ 6. 时间衰减
  │     time_decay = 0.95^(hours_elapsed / 6)    // 最小保持 0.1
  │     base_relevance *= time_decay
  │
  ├─→ 7. 重要性加权
  │     importance_weight = 0.8 + (importance * 0.4)   // 范围 [0.8, 1.2]
  │     final_score = base_relevance * importance_weight
  │
  └─→ 8. 按 final_score 降序排序，返回 Top-5
```

**返回给 LLM 的格式：**

```
🔍 找到 2 条相关记忆:
1. [工作记忆] 当前正在讨论项目架构 (重要性: 0.50)
2. [工作记忆] 用户提到使用微服务架构 (重要性: 0.60)
```

---

## 5. EpisodicMemory 完整工作流程

### 认知定位
**情景记忆** = 人类的自传体记忆，记录"什么时候发生了什么"。

### 写入流程

```
用户: "记住今天我们讨论了数据库优化方案"
  │
  ▼
LLM → [TOOL_CALL:memory:action=add,content=今天讨论了数据库优化方案,memory_type=episodic,importance=0.7]
  │
  ▼
MemoryTool._add_memory(content="今天讨论了数据库优化方案", memory_type="episodic", importance=0.7)
  │
  ├─→ 注入 metadata: { session_id: "session_20260312_143022", timestamp: "2026-03-12T14:30:22" }
  │
  ▼
MemoryManager.add_memory(content, memory_type="episodic", importance=0.7, auto_classify=False)
→ 创建 MemoryItem(id=UUID, content="今天讨论了数据库优化方案",
                   memory_type="episodic", importance=0.7)
  │
  ▼
EpisodicMemory.add(memory_item)
  │
  ├─→ 1. 创建 Episode 对象（内存缓存）
  │     Episode(episode_id=UUID, user_id, session_id, timestamp, content, context, importance)
  │     self.episodes.append(episode)
  │     self.sessions[session_id].append(episode_id)
  │
  ├─→ 2. SQLite 权威存储（持久化）
  │     doc_store.add_memory(
  │         memory_id=UUID,
  │         content="今天讨论了数据库优化方案",
  │         memory_type="episodic",
  │         timestamp=1741758622,       // Unix 时间戳
  │         importance=0.7,
  │         properties={session_id, context, outcome, participants, tags}
  │     )
  │
  └─→ 3. Qdrant 向量索引
        embedding = embedder.encode("今天讨论了数据库优化方案")  // 384 维向量
        vector_store.add_vectors(
            vectors=[embedding],
            metadata=[{memory_id, user_id, memory_type, importance, session_id, content}],
            ids=[UUID]
        )
        // 如果 Qdrant 入库失败，不影响权威存储
```

**三层存储架构：**

```
┌──────────────────────────────┐
│ 内存缓存 (Episode List)      │  ← 快速访问
│ episodes: [Episode, ...]     │
├──────────────────────────────┤
│ SQLite (权威存储)             │  ← 持久化，完整数据
│ memory.db                    │
├──────────────────────────────┤
│ Qdrant (向量索引)             │  ← 语义检索
│ hello_agents_vectors         │
└──────────────────────────────┘
```

### 读取流程

```
用户: "我们之前讨论过什么数据库相关的事情？"
  │
  ▼
LLM → [TOOL_CALL:memory:action=search,query=数据库相关讨论]
  │
  ▼
EpisodicMemory.retrieve(query="数据库相关讨论", limit=5)
  │
  ├─→ 1. 结构化过滤（如果有 time_range / importance_threshold）
  │     docs = doc_store.search_memories(memory_type="episodic", ...)
  │     candidate_ids = {d["memory_id"] for d in docs}
  │
  ├─→ 2. Qdrant 向量检索
  │     query_vec = embedder.encode("数据库相关讨论")
  │     hits = vector_store.search_similar(
  │         query_vector=query_vec,
  │         limit=25,                    // 5 * 5 或至少 20
  │         where={memory_type: "episodic", user_id: ...}
  │     )
  │
  ├─→ 3. 对每个 hit 做综合评分
  │     for hit in hits:
  │         doc = doc_store.get_memory(hit.memory_id)    // 从权威库获取完整数据
  │
  │         vec_score = hit.score                         // Qdrant 相似度分数
  │         age_days = (now - doc.timestamp) / 86400
  │         recency_score = 1.0 / (1.0 + age_days)       // 近因效应
  │         importance = doc.importance
  │
  │         // 评分公式:
  │         base_relevance = vec_score * 0.8 + recency_score * 0.2
  │         importance_weight = 0.8 + (importance * 0.4)  // [0.8, 1.2]
  │         final_score = base_relevance * importance_weight
  │
  ├─→ 4. 回退：如果 Qdrant 无结果
  │     for episode in episodes:
  │         if "数据库" in episode.content.lower():
  │             keyword_score = 0.5
  │             base_relevance = keyword_score * 0.8 + recency_score * 0.2
  │             final_score = base_relevance * importance_weight
  │
  └─→ 5. 按 final_score 降序排序，返回 Top-5 的 MemoryItem
```

**返回给 LLM 的格式：**

```
🔍 找到 1 条相关记忆:
1. [情景记忆] 今天讨论了数据库优化方案 (重要性: 0.70)
```

---

## 6. SemanticMemory 完整工作流程

### 认知定位
**语义记忆** = 人类的知识记忆，存储"什么是什么"的抽象概念和关系。

### 写入流程

```
用户: "记住：Python是一种动态类型的编程语言，由Guido van Rossum创建"
  │
  ▼
LLM → [TOOL_CALL:memory:action=add,content=Python是一种动态类型的编程语言由Guido van Rossum创建,memory_type=semantic,importance=0.9]
  │
  ▼
SemanticMemory.add(memory_item)
  │
  ├─→ 1. 生成文本嵌入（384维向量）
  │     embedding = embedding_model.encode("Python是一种动态类型的编程语言...")
  │     memory_embeddings[memory_id] = embedding
  │
  ├─→ 2. 提取实体（spaCy NER）
  │     entities = _extract_entities(content)
  │     → [Entity(name="Python", type="PRODUCT"),
  │        Entity(name="Guido van Rossum", type="PERSON"),
  │        Entity(name="动态类型", type="CONCEPT")]
  │
  ├─→ 3. 提取关系
  │     relations = _extract_relations(content, entities)
  │     → [Relation(from="Python", to="动态类型", type="has_property"),
  │        Relation(from="Guido van Rossum", to="Python", type="created")]
  │
  ├─→ 4. 存储到 Neo4j 知识图谱
  │     for entity in entities:
  │         _add_entity_to_graph(entity, memory_item)
  │         // CREATE (n:Entity {id, name, type, ...})
  │     for relation in relations:
  │         _add_relation_to_graph(relation, memory_item)
  │         // CREATE (a)-[r:RELATION_TYPE]->(b)
  │
  ├─→ 5. 存储到 Qdrant 向量数据库
  │     vector_store.add_vectors(
  │         vectors=[embedding.tolist()],
  │         metadata=[{memory_id, user_id, content, memory_type,
  │                    timestamp, importance, entities, entity_count, relation_count}],
  │         ids=[memory_id]
  │     )
  │
  └─→ 6. 更新内存缓存
        memory_item.metadata["entities"] = [e.entity_id for e in entities]
        memory_item.metadata["relations"] = ["Python-has_property-动态类型", ...]
        semantic_memories.append(memory_item)
```

**双重存储架构：**

```
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ Qdrant (向量检索)             │     │ Neo4j (知识图谱)              │
│                              │     │                              │
│ 向量: [0.12, -0.34, ...]    │     │ (Python)──has_property──►    │
│ 元数据: {content, entities}  │     │         (动态类型)            │
│                              │     │ (Guido)──created──►(Python) │
└──────────────────────────────┘     └──────────────────────────────┘
          │                                      │
          └──────────┐          ┌────────────────┘
                     ▼          ▼
              ┌─────────────────────┐
              │   混合检索 & 排序    │
              └─────────────────────┘
```

### 读取流程

```
用户: "Python是什么？"
  │
  ▼
LLM → [TOOL_CALL:memory:action=search,query=Python是什么]
  │
  ▼
SemanticMemory.retrieve(query="Python是什么", limit=5)
  │
  ├─→ 1. 向量检索（Qdrant）
  │     query_embedding = embedding_model.encode("Python是什么")
  │     vector_results = vector_store.search_similar(
  │         query_vector=query_embedding.tolist(),
  │         limit=10,
  │         where={memory_type: "semantic"}
  │     )
  │     → 返回: [{id, score: 0.87, metadata: {content, ...}}, ...]
  │
  ├─→ 2. 图检索（Neo4j）
  │     query_entities = _extract_entities("Python是什么")
  │     → [Entity(name="Python")]
  │     related_entities = graph_store.find_related_entities("Python")
  │     → ["动态类型", "Guido van Rossum"]
  │     related_memory_ids = graph_store.get_entity_memory_ids(...)
  │     → 返回关联的 memory_id 列表及分数
  │
  ├─→ 3. 混合排序
  │     _combine_and_rank_results(vector_results, graph_results)
  │     // 对于每个结果:
  │     //   combined_score = vector_score * 0.6 + graph_score * 0.4
  │     // 去重，按 combined_score 排序
  │
  ├─→ 4. Softmax 概率计算
  │     scores = [r.combined_score for r in results]
  │     probs = softmax(scores)    // 归一化为概率分布
  │
  └─→ 5. 过滤已遗忘记忆，构建 MemoryItem（附带分数和概率）
        MemoryItem(
            id, content, memory_type="semantic",
            metadata={
                combined_score: 0.82,
                vector_score: 0.87,
                graph_score: 0.74,
                probability: 0.45
            }
        )
```

**返回给 LLM 的格式：**

```
🔍 找到 1 条相关记忆:
1. [语义记忆] Python是一种动态类型的编程语言，由Guido van Rossum创建 (重要性: 0.90)
```

---

## 7. PerceptualMemory 完整工作流程

### 认知定位
**感知记忆** = 人类的多模态记忆，处理视觉、听觉等多种感知信息。

### 写入流程（以图片为例）

```
用户: "记住这张架构图" (提供 file_path: /path/to/architecture.png)
  │
  ▼
LLM → [TOOL_CALL:memory:action=add,content=系统架构图,memory_type=perceptual,file_path=/path/to/architecture.png]
  │
  ▼
MemoryTool._add_memory(content="系统架构图", memory_type="perceptual",
                       file_path="/path/to/architecture.png")
  │
  ├─→ 1. 推断模态: _infer_modality("architecture.png") → "image"
  │
  ├─→ 2. 注入 metadata: {modality: "image", raw_data: "/path/to/architecture.png"}
  │
  ▼
PerceptualMemory.add(memory_item)
  │
  ├─→ 1. 编码感知数据
  │     _encode_perception(raw_data="/path/to/architecture.png", modality="image", id)
  │     │
  │     ├── 尝试 CLIP 编码（如果 transformers 可用）
  │     │   image = Image.open("/path/to/architecture.png")
  │     │   inputs = clip_processor(images=image, return_tensors="pt")
  │     │   feats = clip_model.get_image_features(**inputs)
  │     │   → 512 维向量
  │     │
  │     └── 回退: 哈希编码（如果 CLIP 不可用）
  │         data_bytes = read("/path/to/architecture.png")
  │         hex_str = sha256(data_bytes).hexdigest()
  │         → 确定性 384 维随机向量（基于哈希种子）
  │
  ├─→ 2. 创建 Perception 对象
  │     Perception(perception_id, data="/path/to/architecture.png",
  │                modality="image", encoding=[...], data_hash=md5)
  │
  ├─→ 3. 缓存与模态索引
  │     perceptions[perception_id] = perception
  │     modality_index["image"].append(perception_id)
  │
  ├─→ 4. SQLite 权威入库
  │     doc_store.add_memory(memory_id, content="系统架构图",
  │         memory_type="perceptual", properties={perception_id, modality, ...})
  │
  └─→ 5. Qdrant 向量入库（image 模态专用集合）
        vector_stores["image"].add_vectors(
            vectors=[encoding],
            metadata=[{memory_id, modality: "image", content, ...}],
            ids=[memory_id]
        )
```

**按模态分集合的 Qdrant 架构：**

```
Qdrant
├── hello_agents_vectors_perceptual_text   (384 维)
├── hello_agents_vectors_perceptual_image  (512 维 CLIP / 384 维降级)
└── hello_agents_vectors_perceptual_audio  (512 维 CLAP / 384 维降级)
```

### 读取流程

```
用户: "找一下之前的架构图"
  │
  ▼
LLM → [TOOL_CALL:memory:action=search,query=架构图,memory_type=perceptual]
  │
  ▼
PerceptualMemory.retrieve(query="架构图", limit=5,
                          query_modality="text", target_modality=None)
  │
  ├─→ 1. 编码查询
  │     qvec = _encode_data("架构图", "text")
  │     → text_embedder.encode("架构图") → 384 维向量
  │
  ├─→ 2. 向量检索（在 text 模态集合中搜索）
  │     // 注意：跨模态搜索（文本搜图片）受限于编码器能力
  │     store = vector_stores["text"]  // 如果 target_modality 未指定
  │     hits = store.search_similar(query_vector=qvec, limit=25,
  │                                  where={memory_type: "perceptual"})
  │
  ├─→ 3. 融合排序（与 Episodic 相同的评分公式）
  │     base_relevance = vec_score * 0.8 + recency_score * 0.2
  │     importance_weight = 0.8 + (importance * 0.4)
  │     final_score = base_relevance * importance_weight
  │
  └─→ 4. 回退：内存缓存关键词匹配
        for m in perceptual_memories:
            if "架构图" in m.content: ...
```

---

## 8. 记忆检索与大模型的闭环

### 检索结果如何影响 LLM 的回答

以"我叫什么名字？"为例：

```
━━━ 第1轮 LLM 调用 ━━━

messages = [
    {role: "system", content: "你是一个有记忆能力的...（含工具描述）"},
    {role: "user", content: "我叫什么名字？"}
]

LLM 输出:
"让我帮你查一下。[TOOL_CALL:memory:action=search,query=用户名字]"

━━━ 工具执行 ━━━

MemoryTool.execute("search", query="用户名字")
→ MemoryManager.retrieve_memories("用户名字")
  → SemanticMemory.retrieve("用户名字")
    → Qdrant 向量搜索: 匹配到 "用户叫张三" (score=0.82)
    → Neo4j 图搜索: 实体 "张三" 关联的记忆
  → WorkingMemory.retrieve("用户名字")
    → TF-IDF + 关键词匹配
→ 按 importance 排序，返回结果

返回:
"🔍 找到 1 条相关记忆:
1. [语义记忆] 用户叫张三 (重要性: 0.80)"

━━━ 第2轮 LLM 调用 ━━━

messages = [
    {role: "system", content: "你是一个有记忆能力的..."},
    {role: "user", content: "我叫什么名字？"},
    {role: "assistant", content: "让我帮你查一下。"},
    {role: "user", content: "工具执行结果：\n🔧 工具 memory 执行结果：
     \n🔍 找到 1 条相关记忆:\n1. [语义记忆] 用户叫张三 (重要性: 0.80)
     \n\n请基于这些结果给出完整的回答。"}
]

LLM 输出（无工具调用）:
"你叫张三！这是我之前记住的。还有什么需要帮忙的吗？"
```

**关键点：** LLM 通过"工具执行结果"消息获得记忆内容，然后基于这些信息生成回答。记忆数据以**文本**形式注入 LLM 上下文，LLM 以自然语言理解并整合这些信息。

---

## 9. 多轮工具调用（ReAct 模式）

SimpleAgent 支持最多 `max_tool_iterations=3` 轮工具调用。

### 复杂场景示例

```
用户: "查一下我的个人信息，然后把今天的讨论要点记录下来"
  │
  ▼
━━━ 迭代 1 ━━━
LLM: "好的，先帮你查一下个人信息。
      [TOOL_CALL:memory:action=search,query=用户个人信息]"

执行 → 返回: "🔍 找到 1 条: 用户叫张三，Python开发者"
反馈给 LLM

━━━ 迭代 2 ━━━
LLM: "找到了你的信息。现在记录今天的讨论要点。
      [TOOL_CALL:memory:action=add,content=今天讨论了数据库优化和微服务架构,memory_type=episodic,importance=0.8]"

执行 → 返回: "✅ 记忆已添加 (ID: abc123...)"
反馈给 LLM

━━━ 迭代 3（最终回答）━━━
LLM: "已经完成了！你的个人信息：张三，Python开发者。
      今天的讨论要点已经记录：数据库优化和微服务架构。"
```

**源码逻辑（`simple_agent.py:198-228`）：**

```python
while current_iteration < max_tool_iterations:
    response = self.llm.invoke(messages)
    tool_calls = self._parse_tool_calls(response)

    if tool_calls:
        # 执行工具，收集结果
        for call in tool_calls:
            result = self._execute_tool_call(call['tool_name'], call['parameters'])
            tool_results.append(result)

        # 将结果反馈给 LLM
        messages.append({"role": "assistant", "content": clean_response})
        messages.append({"role": "user", "content": f"工具执行结果：\n{results}\n\n请基于这些结果给出完整的回答。"})

        current_iteration += 1
        continue

    # 无工具调用 → 最终回答
    final_response = response
    break
```

---

## 10. 自动对话记录机制

MemoryTool 提供了 `auto_record_conversation()` 方法，可以在 Agent 外部调用，自动将对话记录到记忆中：

**源码位置：** `memory_tool.py:313-346`

```python
def auto_record_conversation(self, user_input: str, agent_response: str):
    self.conversation_count += 1

    # 1. 用户输入 → 工作记忆（重要性 0.6）
    self._add_memory(
        content=f"用户: {user_input}",
        memory_type="working",
        importance=0.6,
        type="user_input",
        conversation_id=self.conversation_count
    )

    # 2. Agent 响应 → 工作记忆（重要性 0.7）
    self._add_memory(
        content=f"助手: {agent_response}",
        memory_type="working",
        importance=0.7,
        type="agent_response",
        conversation_id=self.conversation_count
    )

    # 3. 如果是重要对话 → 额外存入情景记忆（重要性 0.8）
    if len(agent_response) > 100 or "重要" in user_input or "记住" in user_input:
        self._add_memory(
            content=f"对话 - 用户: {user_input}\n助手: {agent_response}",
            memory_type="episodic",
            importance=0.8,
            type="interaction",
            conversation_id=self.conversation_count
        )
```

**重要对话的判定条件：**
- Agent 回复超过 100 字符
- 用户输入包含"重要"
- 用户输入包含"记住"

**使用方式（在 Agent 外部手动调用）：**

```python
# Agent 运行后，手动记录
response = agent.run("记住我叫张三")
memory_tool.auto_record_conversation("记住我叫张三", response)
```

---

## 11. 关键设计决策分析

### 11.1 为什么用 Prompt-Based 而不是 OpenAI 原生 Function Calling？

| 方面 | Prompt-Based（HelloAgents 采用） | OpenAI Function Calling |
|------|:---:|:---:|
| **LLM 兼容性** | 兼容任何 LLM（DashScope、本地模型等） | 仅支持 OpenAI/兼容 API |
| **实现复杂度** | 低（文本解析） | 中（需要处理 tool_calls 结构） |
| **参数类型安全** | 弱（全部为字符串） | 强（JSON Schema 约束） |
| **调用可靠性** | 依赖 LLM 输出格式一致性 | 结构化保证 |
| **可调试性** | 高（人可直接阅读） | 中 |

**HelloAgents 的选择理由：** 框架定位为兼容多种 LLM（阿里 DashScope、本地部署等），Prompt-Based 方案是最大公约数。

### 11.2 为什么工具结果以 "user" 角色注入而非 "tool" 角色？

```python
# HelloAgents 的做法：
messages.append({"role": "user", "content": f"工具执行结果：\n{tool_results}"})
```

因为标准的 `role: "tool"` 是 OpenAI 原生 Function Calling 的一部分，而 HelloAgents 使用的是文本解析模式。在这种模式下：

- LLM 不知道有 "tool" 角色的消息格式
- 将结果作为 "user" 消息注入，LLM 可以自然地理解和回应
- 附加的 "请基于这些结果给出完整的回答。" 提示词引导 LLM 整合结果

### 11.3 智能参数推断的价值

```python
# LLM 可能输出简化格式，Agent 自动推断：
[TOOL_CALL:memory:recall=张三]        → {action: "search", query: "张三"}
[TOOL_CALL:memory:store=用户是开发者]  → {action: "add", content: "用户是开发者"}
[TOOL_CALL:memory:Python编程]         → {action: "search", query: "Python编程"}
```

这种设计提高了系统的**容错性**——即使 LLM 没有完美遵循参数格式，Agent 也能正确理解意图并执行。

### 11.4 四种记忆的存储策略对比

```
WorkingMemory   ─── 纯内存 ─── 不持久化
                    │
                    │  巩固（importance ≥ 0.7）
                    ▼
EpisodicMemory  ─── SQLite + Qdrant ─── 持久化
                    │
                    │  知识提取
                    ▼
SemanticMemory  ─── Qdrant + Neo4j ─── 持久化 + 知识图谱

PerceptualMemory ── SQLite + Qdrant(按模态) ─── 持久化 + 多模态
```

### 11.5 评分公式统一设计

所有记忆类型（除 Working）使用统一的评分框架：

```
base_relevance = vector_score × 0.8 + recency_score × 0.2
importance_weight = 0.8 + (importance × 0.4)    // [0.8, 1.2]
final_score = base_relevance × importance_weight
```

**设计意图：**
- 相似度（vector_score）是主要因素（80%权重）
- 时间近因（recency_score）提供一定偏好（20%权重）
- 重要性（importance）作为**乘法因子**而非加法，避免低重要性记忆因高相似度而排名过高

Working Memory 使用不同的公式，增加了时间衰减：

```
hybrid_score = vector_score × 0.7 + keyword_score × 0.3
hybrid_score *= time_decay(0.95^(hours/6))
final_score = hybrid_score × importance_weight
```

---

## 总结

HelloAgents 0.2.0 的 MemoryTool 与大模型交互的核心机制可以归纳为：

1. **Prompt 注入**：工具描述以文本形式写入 System Prompt，教会 LLM 如何调用
2. **文本标记**：LLM 通过 `[TOOL_CALL:name:params]` 标记表达工具调用意图
3. **正则解析**：Agent 层通过正则提取标记，智能解析参数（支持多种格式）
4. **路由执行**：MemoryTool.run() → execute() → 具体操作方法 → MemoryManager → 各记忆类型
5. **结果回传**：工具结果作为 user 消息注入对话，LLM 基于结果生成最终回答
6. **ReAct 循环**：支持最多 3 轮工具调用，实现多步推理和操作

每种记忆类型在这个流程中的角色：
- **WorkingMemory**：快速上下文管理，TF-IDF + 关键词检索，自动过期
- **EpisodicMemory**：事件持久化，SQLite 权威 + Qdrant 向量检索，支持回退
- **SemanticMemory**：知识管理，Qdrant 向量 + Neo4j 图 混合检索，实体关系推理
- **PerceptualMemory**：多模态处理，按模态分 Qdrant 集合，CLIP/CLAP 优雅降级
