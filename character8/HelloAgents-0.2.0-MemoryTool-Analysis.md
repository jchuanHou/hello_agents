# HelloAgents 0.2.0 MemoryTool 实现分析

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [分层架构详解](#2-分层架构详解)
3. [基础层：Tool 抽象基类](#3-基础层tool-抽象基类)
4. [工具层：MemoryTool 实现](#4-工具层memorytool-实现)
5. [管理层：MemoryManager 核心协调器](#5-管理层memorymanager-核心协调器)
6. [数据模型层：MemoryItem 与 MemoryConfig](#6-数据模型层memoryitem-与-memoryconfig)
7. [四种记忆类型实现](#7-四种记忆类型实现)
   - 7.1 [WorkingMemory（工作记忆）](#71-workingmemory工作记忆)
   - 7.2 [EpisodicMemory（情景记忆）](#72-episodicmemory情景记忆)
   - 7.3 [SemanticMemory（语义记忆）](#73-semanticmemory语义记忆)
   - 7.4 [PerceptualMemory（感知记忆）](#74-perceptualmemory感知记忆)
8. [存储后端](#8-存储后端)
9. [嵌入向量系统](#9-嵌入向量系统)
10. [工具注册与执行机制](#10-工具注册与执行机制)
11. [Agent 集成方式](#11-agent-集成方式)
12. [遗忘与巩固机制](#12-遗忘与巩固机制)
13. [完整使用示例](#13-完整使用示例)
14. [记忆类型特性对比](#14-记忆类型特性对比)
15. [配置参考](#15-配置参考)

---

## 1. 整体架构概览

HelloAgents 0.2.0 的 MemoryTool 是一个**模拟人类认知记忆系统**的工具实现，采用分层架构设计，将记忆的存储、检索、遗忘、巩固等操作抽象为统一的工具接口，供 Agent 调用。

整个系统的核心设计理念：

- **仿生设计**：借鉴认知科学中的记忆分类（工作记忆、情景记忆、语义记忆、感知记忆）
- **分层解耦**：工具接口 → 记忆管理器 → 记忆类型 → 存储后端，各层职责清晰
- **混合检索**：结合向量语义搜索、关键词匹配、知识图谱推理等多种检索策略
- **优雅降级**：当高级组件（如 Qdrant、Neo4j）不可用时，自动回退到轻量方案

**架构总览图：**

```
┌──────────────────────────────────────────────────────────┐
│              APPLICATION LAYER（应用层）                    │
│    SimpleAgent 通过 ToolRegistry 调用 MemoryTool          │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              TOOL LAYER（工具层）                           │
│    MemoryTool：统一接口，9 种操作，参数解析与路由            │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              MANAGER LAYER（管理层）                        │
│    MemoryManager：协调四种记忆类型，自动分类，重要性计算      │
└────────┬───────────┬───────────┬───────────┬─────────────┘
         │           │           │           │
    ┌────▼───┐  ┌────▼───┐  ┌───▼────┐  ┌───▼──────┐
    │Working │  │Episodic│  │Semantic│  │Perceptual│
    │Memory  │  │Memory  │  │Memory  │  │Memory    │
    └────┬───┘  └────┬───┘  └───┬────┘  └───┬──────┘
         │           │          │            │
┌────────▼───────────▼──────────▼────────────▼─────────────┐
│              STORAGE LAYER（存储层）                        │
│    SQLite（权威存储）│ Qdrant（向量检索）│ Neo4j（知识图谱） │
└──────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────┐
│              EMBEDDING LAYER（嵌入层）                      │
│    SentenceTransformers(384维) / DashScope / TF-IDF       │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 分层架构详解

### 各层职责

| 层级 | 核心组件 | 职责 |
|------|---------|------|
| **应用层** | SimpleAgent + ToolRegistry | Agent 通过工具注册表发现和调用 MemoryTool |
| **工具层** | MemoryTool | 统一接口、参数定义、操作路由、结果格式化 |
| **管理层** | MemoryManager | 协调四种记忆类型，自动分类，重要性计算，遗忘/巩固策略 |
| **记忆类型层** | Working/Episodic/Semantic/Perceptual | 各类型记忆的具体存储与检索逻辑 |
| **存储层** | SQLite/Qdrant/Neo4j | 持久化存储、向量索引、知识图谱 |
| **嵌入层** | SentenceTransformers 等 | 将文本/多模态数据转为向量表示 |

### 调用链路

```
用户输入 → Agent.run()
         → ToolRegistry.execute_tool("memory", input)
         → MemoryTool.run(parameters)
         → MemoryTool.execute(action, **kwargs)
         → MemoryManager.add_memory() / retrieve_memories() / ...
         → WorkingMemory.add() / EpisodicMemory.retrieve() / ...
         → SQLiteStore / QdrantStore / Neo4jStore
```

---

## 3. 基础层：Tool 抽象基类

**文件路径：** `hello_agents/tools/base.py`

所有工具（包括 MemoryTool）都继承自 `Tool` 抽象基类：

```python
class Tool(ABC):
    """所有工具的抽象基类"""

    def __init__(self, name: str, description: str):
        self.name = name              # 工具名称，用于注册和查找
        self.description = description # 工具描述，用于 LLM 理解工具用途

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具，接收参数字典，返回结果字符串"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """定义工具的输入参数列表"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证必需参数是否已提供"""
        required = [p for p in self.get_parameters() if p.required]
        return all(p.name in parameters for p in required)

    def to_dict(self) -> Dict[str, Any]:
        """将工具转为字典格式（用于构建 LLM 提示词）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.get_parameters()]
        }
```

**关键设计点：**

- `run()` 是工具的执行入口，所有子类必须实现
- `get_parameters()` 声明工具的参数 schema，LLM 据此生成正确的调用参数
- `to_dict()` 用于将工具描述注入到 LLM 的系统提示词中

---

## 4. 工具层：MemoryTool 实现

**文件路径：** `hello_agents/tools/builtin/memory_tool.py`（约 454 行）

### 4.1 构造函数

```python
class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memory_types: List[str] = None
    ):
        super().__init__(
            name="memory",
            description="记忆管理工具，支持记忆的增删改查、搜索、遗忘和巩固操作"
        )
        self.memory_manager = MemoryManager(
            config=memory_config,
            user_id=user_id,
            enable_working="working" in (memory_types or ["working", "episodic", "semantic"]),
            enable_episodic="episodic" in (memory_types or ["working", "episodic", "semantic"]),
            enable_semantic="semantic" in (memory_types or ["working", "episodic", "semantic"]),
            enable_perceptual="perceptual" in (memory_types or [])
        )
        self.current_session_id = str(uuid.uuid4())
        self.conversation_count = 0
```

**设计要点：**

- 默认启用 working、episodic、semantic 三种记忆，perceptual 默认关闭
- 内部创建 `MemoryManager` 实例来管理所有记忆操作
- 维护 session_id 用于追踪当前会话

### 4.2 参数定义

MemoryTool 定义了 11 个输入参数：

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `action` | string | **是** | 操作类型：add/search/summary/stats/update/remove/forget/consolidate/clear_all |
| `content` | string | 否 | 记忆内容（用于 add/update） |
| `query` | string | 否 | 搜索查询（用于 search） |
| `memory_type` | string | 否 | 记忆类型：working/episodic/semantic/perceptual |
| `importance` | float | 否 | 重要性评分 (0.0-1.0) |
| `limit` | int | 否 | 搜索结果数量限制 |
| `memory_id` | string | 否 | 目标记忆 ID（用于 update/remove） |
| `file_path` | string | 否 | 文件路径（用于感知记忆） |
| `modality` | string | 否 | 数据模态：text/image/audio |
| `strategy` | string | 否 | 遗忘策略 |
| `threshold` | float | 否 | 遗忘阈值 |

### 4.3 支持的 9 种操作

```python
def execute(self, action: str, **kwargs) -> str:
    """根据 action 路由到对应的处理方法"""
    action_map = {
        "add":         self._add_memory,
        "search":      self._search_memory,
        "summary":     self._get_summary,
        "stats":       self._get_stats,
        "update":      self._update_memory,
        "remove":      self._remove_memory,
        "forget":      self._forget,
        "consolidate": self._consolidate,
        "clear_all":   self._clear_all,
    }
    handler = action_map.get(action)
    return handler(**kwargs)
```

### 4.4 核心操作方法

**添加记忆 `_add_memory()`：**

```python
def _add_memory(self, content, memory_type="working", importance=None, **metadata):
    # 1. 创建 MemoryItem
    # 2. 如果是 perceptual 类型，自动推断 modality（从文件扩展名）
    # 3. 调用 manager.add_memory()
    # 4. 返回记忆 ID
    memory_id = self.memory_manager.add_memory(
        content=content,
        memory_type=memory_type,
        importance=importance,
        metadata=metadata
    )
    return f"记忆已添加，ID: {memory_id}"
```

**搜索记忆 `_search_memory()`：**

```python
def _search_memory(self, query, limit=5, memory_type=None):
    # 1. 调用 manager.retrieve_memories()
    # 2. 格式化搜索结果（包含类型标签、重要性、时间戳）
    results = self.memory_manager.retrieve_memories(
        query=query,
        memory_types=[memory_type] if memory_type else None,
        limit=limit
    )
    # 返回格式化的结果字符串
```

**获取摘要 `_get_summary()`：**

```python
def _get_summary(self, limit=10):
    # 1. 从所有记忆类型中获取记忆
    # 2. 去重、按重要性排序
    # 3. 返回统计信息 + Top-N 记忆摘要
```

### 4.5 便捷方法

MemoryTool 还提供了几个高层便捷方法，简化常见操作：

```python
def add_knowledge(self, content, importance=0.9):
    """快速添加知识到语义记忆（高重要性）"""
    return self.execute("add", content=content,
                        memory_type="semantic", importance=importance)

def get_context_for_query(self, query, limit=3) -> str:
    """为查询检索相关记忆上下文（用于注入 LLM 提示词）"""
    return self.execute("search", query=query, limit=limit)

def auto_record_conversation(self, user_input, agent_response):
    """自动记录对话轮次到情景记忆"""
    self.conversation_count += 1
    content = f"用户: {user_input}\n助手: {agent_response}"
    self.execute("add", content=content, memory_type="episodic",
                 importance=0.6, session_id=self.current_session_id)

def clear_session(self):
    """清除当前会话和工作记忆"""

def forget_old_memories(self, max_age_days=30):
    """遗忘超过指定天数的旧记忆"""
```

---

## 5. 管理层：MemoryManager 核心协调器

**文件路径：** `hello_agents/memory/manager.py`（约 343 行）

MemoryManager 是整个记忆系统的**中央协调器**，管理四种记忆类型的创建、路由和协调。

### 5.1 构造函数

```python
class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        # 根据开关创建对应的记忆类型实例
        self.memories = {}
        if enable_working:
            self.memories["working"] = WorkingMemory(config=self.config)
        if enable_episodic:
            self.memories["episodic"] = EpisodicMemory(config=self.config)
        if enable_semantic:
            self.memories["semantic"] = SemanticMemory(config=self.config)
        if enable_perceptual:
            self.memories["perceptual"] = PerceptualMemory(config=self.config)
```

### 5.2 自动分类逻辑

当用户没有指定 `memory_type` 时，MemoryManager 会自动根据内容特征分类：

```python
def _classify_memory_type(self, content: str, metadata: Optional[Dict]) -> str:
    """基于内容特征自动判断记忆类型"""

    # 优先使用 metadata 中的显式指定
    if metadata and "type" in metadata:
        return metadata["type"]

    # 情景记忆特征词：昨天, 今天, 上次, 刚才, 那次...
    episodic_keywords = ["昨天", "今天", "上次", "刚才", "那次", "记得"]
    if any(kw in content for kw in episodic_keywords):
        return "episodic"

    # 语义记忆特征词：定义, 概念, 规则, 知识, 原理...
    semantic_keywords = ["定义", "概念", "规则", "知识", "原理", "是指"]
    if any(kw in content for kw in semantic_keywords):
        return "semantic"

    # 默认归入工作记忆
    return "working"
```

### 5.3 重要性计算

```python
def _calculate_importance(self, content: str, metadata: Optional[Dict]) -> float:
    """基于内容和元数据计算记忆重要性"""
    base = 0.5

    # 内容长度加分
    if len(content) > 100:
        base += 0.1

    # 重要关键词加分
    important_keywords = ["重要", "关键", "必须", "核心", "紧急"]
    if any(kw in content for kw in important_keywords):
        base += 0.2

    # metadata 中的优先级
    if metadata:
        if metadata.get("priority") == "high":
            base += 0.3
        elif metadata.get("priority") == "low":
            base -= 0.2

    return max(0.0, min(1.0, base))  # 钳位到 [0.0, 1.0]
```

### 5.4 核心方法

**添加记忆：**

```python
def add_memory(self, content, memory_type="working", importance=None,
               metadata=None, auto_classify=True) -> str:
    # 1. 自动分类（如果启用且未指定类型）
    if auto_classify and memory_type == "working":
        memory_type = self._classify_memory_type(content, metadata)

    # 2. 自动计算重要性（如果未指定）
    if importance is None:
        importance = self._calculate_importance(content, metadata)

    # 3. 创建 MemoryItem
    memory_item = MemoryItem(
        id=str(uuid.uuid4()),
        content=content,
        memory_type=memory_type,
        user_id=self.user_id,
        timestamp=datetime.now(),
        importance=importance,
        metadata=metadata or {}
    )

    # 4. 路由到对应的记忆类型处理器
    self.memories[memory_type].add(memory_item)
    return memory_item.id
```

**检索记忆：**

```python
def retrieve_memories(self, query, memory_types=None, limit=10,
                      min_importance=0.0, time_range=None) -> List[MemoryItem]:
    results = []

    # 查询所有指定的记忆类型（默认查询全部）
    target_types = memory_types or list(self.memories.keys())
    for mem_type in target_types:
        if mem_type in self.memories:
            type_results = self.memories[mem_type].retrieve(
                query=query, limit=limit
            )
            results.extend(type_results)

    # 按重要性降序排序，返回 Top-N
    results.sort(key=lambda x: x.importance, reverse=True)
    return results[:limit]
```

**巩固记忆：**

```python
def consolidate_memories(self, from_type="working", to_type="episodic",
                         importance_threshold=0.7) -> int:
    """将高重要性的短期记忆转移到长期记忆"""
    source = self.memories.get(from_type)
    target = self.memories.get(to_type)

    # 获取所有超过阈值的记忆
    high_importance = [m for m in source.get_all()
                       if m.importance >= importance_threshold]

    count = 0
    for memory in high_importance:
        # 重要性提升 10%
        memory.importance = min(1.0, memory.importance * 1.1)
        memory.memory_type = to_type
        target.add(memory)
        source.remove(memory.id)
        count += 1

    return count
```

---

## 6. 数据模型层：MemoryItem 与 MemoryConfig

**文件路径：** `hello_agents/memory/base.py`（约 171 行）

### 6.1 MemoryItem 数据结构

```python
class MemoryItem(BaseModel):
    """记忆条目的统一数据结构"""
    id: str                          # UUID，唯一标识
    content: str                     # 记忆文本内容
    memory_type: str                 # "working" | "episodic" | "semantic" | "perceptual"
    user_id: str                     # 所属用户 ID
    timestamp: datetime              # 创建时间戳
    importance: float = 0.5          # 重要性评分 [0.0, 1.0]
    metadata: Dict[str, Any] = {}    # 附加元数据（上下文、标签等）
```

### 6.2 MemoryConfig 配置

```python
class MemoryConfig(BaseModel):
    """记忆系统全局配置"""
    storage_path: str = "./memory_data"              # 存储路径
    max_capacity: int = 100                           # 最大记忆容量
    importance_threshold: float = 0.1                 # 重要性最低阈值
    decay_factor: float = 0.95                        # 时间衰减因子

    # 工作记忆专用配置
    working_memory_capacity: int = 10                 # 工作记忆容量上限
    working_memory_tokens: int = 2000                 # 工作记忆 token 上限
    working_memory_ttl_minutes: int = 120             # 工作记忆 TTL（分钟）

    # 感知记忆模态
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]
```

### 6.3 BaseMemory 抽象基类

```python
class BaseMemory(ABC):
    """所有记忆类型的抽象基类"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆，返回 memory_id"""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索相关记忆"""

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict = None) -> bool:
        """更新记忆"""

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆"""

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在"""

    @abstractmethod
    def clear(self):
        """清空所有记忆"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""

    def _generate_id(self) -> str:
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float) -> float:
        """根据内容计算重要性"""
```

---

## 7. 四种记忆类型实现

### 7.1 WorkingMemory（工作记忆）

**文件路径：** `hello_agents/memory/types/working.py`（约 412 行）

**认知模型对应：** 人类的短期记忆 / 工作记忆，容量有限，快速存取

**特点：**

| 特性 | 说明 |
|------|------|
| 存储方式 | 纯内存（不持久化） |
| 容量限制 | 默认 10 条 |
| Token 限制 | 默认 2000 tokens |
| TTL | 默认 120 分钟自动过期 |
| 时间衰减 | 重要性随时间指数衰减 (0.95^(hours/6)) |

**检索策略 - 混合评分：**

```python
def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
    # 1. 清除过期记忆（TTL 检查）
    self._expire_old_memories()

    # 2. 尝试 TF-IDF 向量相似度搜索
    vector_scores = self._vector_search(query)

    # 3. 关键词匹配（回退方案）
    keyword_scores = self._keyword_search(query)

    # 4. 应用时间衰减因子
    for memory in self.memories:
        hours_elapsed = (now - memory.timestamp).total_seconds() / 3600
        time_decay = self.config.decay_factor ** (hours_elapsed / 6)
        # ...

    # 5. 混合评分 = 向量分数 * 0.7 + 关键词分数 * 0.3
    hybrid_score = vector_score * 0.7 + keyword_score * 0.3

    # 6. 最终得分 = 混合评分 * 重要性权重
    final_score = hybrid_score * importance_weight

    return sorted_results[:limit]
```

**遗忘策略：**

```python
def forget(self, strategy: str, threshold: float, max_age_days: int) -> int:
    if strategy == "importance_based":
        # 移除重要性低于阈值的记忆
        to_remove = [m for m in self.memories if m.importance < threshold]
    elif strategy == "time_based":
        # 移除超过指定天数的记忆
        cutoff = datetime.now() - timedelta(days=max_age_days)
        to_remove = [m for m in self.memories if m.timestamp < cutoff]
    elif strategy == "capacity_based":
        # 只保留重要性最高的 Top-N
        sorted_memories = sorted(self.memories, key=lambda m: m.importance)
        to_remove = sorted_memories[:len(sorted_memories) - self.config.working_memory_capacity]
```

**上下文摘要：**

```python
def get_context_summary(self, max_length: int = 500) -> str:
    """生成工作记忆摘要（用于注入 LLM 提示词）"""
    # 按重要性排序，截取前 N 条，格式化为文本
```

---

### 7.2 EpisodicMemory（情景记忆）

**文件路径：** `hello_agents/memory/types/episodic.py`（约 599 行）

**认知模型对应：** 人类的自传体记忆，记录特定事件和交互

**特点：**

| 特性 | 说明 |
|------|------|
| 存储方式 | SQLite（权威存储）+ Qdrant（向量索引）+ 内存缓存 |
| 持久化 | 是 |
| 容量限制 | 无硬限制 |
| 上下文 | 丰富的上下文信息（会话ID、事件结果等） |

**Episode 数据结构：**

```python
class Episode:
    episode_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    content: str
    context: Dict[str, Any]       # 丰富的上下文信息
    outcome: Optional[str]         # 事件结果
    importance: float
```

**存储策略 - 三层存储：**

```
写入路径：Episode → SQLite (权威) → Qdrant (向量索引) → 内存缓存
读取路径：向量搜索 (Qdrant) → 回退到 SQLite 关键词搜索
```

**检索策略：**

```python
def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
    # 1. 结构化过滤：time_range, importance_threshold, session_id
    # 2. 向量搜索：通过 Qdrant 查找语义相似的 Episode
    # 3. 评分计算：
    #    base_relevance = vec_score * 0.8 + recency_score * 0.2
    #    final_score = base_relevance * importance_weight
    # 4. 如果 Qdrant 不可用，回退到 SQLite 关键词匹配
    # 5. 过滤已遗忘的 Episode
```

**独有功能：**

```python
def find_patterns(self, user_id=None, min_frequency=2) -> List[Dict]:
    """模式发现：发现重复出现的关键词和上下文模式"""

def get_timeline(self, user_id=None, limit=50) -> List[Dict]:
    """时间线视图：按时间顺序展示 Episode"""

def get_session_episodes(self, session_id) -> List[Episode]:
    """获取某个会话的所有 Episode"""
```

---

### 7.3 SemanticMemory（语义记忆）

**文件路径：** `hello_agents/memory/types/semantic.py`

**认知模型对应：** 人类的知识记忆，存储抽象概念和关系

**特点：**

| 特性 | 说明 |
|------|------|
| 存储方式 | Qdrant（向量检索）+ Neo4j（知识图谱）+ 内存缓存 |
| 持久化 | 是 |
| 实体提取 | 自动从文本中提取实体（基于 spaCy NER） |
| 关系建模 | 实体间关系以知识图谱形式存储 |

**核心数据结构：**

```python
class Entity:
    entity_id: str
    name: str
    entity_type: str        # PERSON, ORG, PRODUCT, SKILL, CONCEPT
    description: str
    properties: Dict[str, Any]
    frequency: int          # 出现次数

class Relation:
    from_entity: str
    to_entity: str
    relation_type: str      # "knows", "is_a", "works_at", "belongs_to"
```

**检索策略 - 混合搜索：**

```python
def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
    # 1. 向量搜索（Qdrant）：基于语义相似度
    vector_results = self.qdrant_store.search_similar(query_embedding, limit)

    # 2. 图搜索（Neo4j）：基于实体关系
    graph_results = self.neo4j_store.find_related(query_entities)

    # 3. 混合评分：
    #    combined_score = vector_score * 0.6 + graph_score * 0.4

    # 4. 去重、排序、返回
```

**独有功能：**

- 自动实体提取（NER）
- 关系推理
- 知识图谱可视化
- 跨概念推理

---

### 7.4 PerceptualMemory（感知记忆）

**文件路径：** `hello_agents/memory/types/perceptual.py`（约 710 行）

**认知模型对应：** 人类的感知记忆，处理多模态信息

**特点：**

| 特性 | 说明 |
|------|------|
| 存储方式 | SQLite + Qdrant（按模态分集合）+ 内存缓存 |
| 支持模态 | text, image, audio, video |
| 编码策略 | 多模态编码，支持优雅降级 |
| 去重 | 基于数据哈希的去重 |

**Perception 数据结构：**

```python
class Perception:
    perception_id: str
    data: Any                   # 原始数据或文件路径
    modality: str              # "text", "image", "audio", "video"
    encoding: List[float]      # 向量嵌入
    metadata: Dict[str, Any]
    timestamp: datetime
    data_hash: str             # 数据哈希（用于去重）
```

**编码策略（优雅降级）：**

```
Text:   sentence-transformers (384维)
Image:  CLIP (如果可用) → 基于哈希的向量 (回退方案)
Audio:  CLAP (如果可用) → 基于哈希的向量 (回退方案)
```

每种模态在 Qdrant 中有独立的 collection，支持模态内和跨模态搜索：

```python
def cross_modal_search(self, query, query_modality, target_modality=None):
    """跨模态搜索：例如用文本描述搜索相关图片"""

def get_by_modality(self, modality, limit=10):
    """按模态类型获取记忆"""
```

---

## 8. 存储后端

### 8.1 SQLiteDocumentStore

**文件路径：** `hello_agents/memory/storage/document_store.py`

**职责：** 权威数据存储，保存所有记忆的完整元数据

```python
class SQLiteDocumentStore:
    def add_memory(self, memory_id, user_id, content, memory_type,
                   timestamp, importance, properties)
    def get_memory(self, memory_id) -> Dict
    def search_memories(self, user_id, memory_type, start_time, end_time,
                       importance_threshold, limit) -> List[Dict]
    def update_memory(self, memory_id, content, importance, properties)
    def delete_memory(self, memory_id)  # 硬删除
    def get_database_stats() -> Dict
```

### 8.2 QdrantVectorStore

**文件路径：** `hello_agents/memory/storage/qdrant_store.py`

**职责：** 向量索引，支持语义相似度搜索

```python
class QdrantVectorStore:
    def __init__(self):
        # 使用 QdrantConnectionManager 管理连接（避免重复连接）

    def add_vectors(self, vectors, ids, metadata)
    def search_similar(self, query_vector, limit, filters) -> List[Dict]
    def delete_memories(self, memory_ids)
    def get_collection_stats() -> Dict
```

**特性：**
- 使用连接管理器避免重复连接
- 支持按模态创建独立集合
- 元数据与向量一起存储

### 8.3 Neo4jStore（语义记忆专用）

**文件路径：** `hello_agents/memory/storage/neo4j_store.py`

**职责：** 知识图谱存储，支持实体关系建模和图遍历

```
图结构：
  节点（Node）= 实体（Entity）+ 属性
  边（Edge）  = 关系（Relation）+ 类型

支持图遍历和路径推理
```

---

## 9. 嵌入向量系统

**文件路径：** `hello_agents/memory/embedding.py`

统一的嵌入接口，支持多种后端和自动回退：

```python
def get_text_embedder() -> EmbeddingModel:
    """获取文本嵌入模型（按优先级选择）"""
    # 优先级：
    # 1. DashScope（如果 EMBED_MODEL_TYPE="dashscope" 且有 API Key）
    # 2. LocalTransformer（sentence-transformers，默认）
    # 3. TF-IDF（轻量回退方案）

def get_dimension(embedder=None) -> int:
    """获取嵌入维度，默认 384"""
```

**三种嵌入实现：**

| 实现 | 维度 | 说明 |
|------|------|------|
| `LocalTransformerEmbedding` | 384 | sentence-transformers (all-MiniLM-L6-v2)，默认方案 |
| `DashScopeEmbedding` | 可变 | 阿里云 text-embedding-v3，需要 API Key |
| `TFIDFEmbedding` | 可变 | 基于 TF-IDF 的轻量方案，最后的回退 |

---

## 10. 工具注册与执行机制

**文件路径：** `hello_agents/tools/registry.py`（约 137 行）

```python
class ToolRegistry:
    """工具注册表，管理工具的注册、查找和执行"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}        # Tool 对象
        self._functions: dict[str, dict] = {}    # 简单函数

    def register_tool(self, tool: Tool):
        """注册 Tool 实例"""
        self._tools[tool.name] = tool

    def register_function(self, name, description, func):
        """注册简单函数为工具"""
        self._functions[name] = {"description": description, "func": func}

    def execute_tool(self, name: str, input_text: str) -> str:
        """按名称执行工具"""
        if name in self._tools:
            tool = self._tools[name]
            return tool.run({"input": input_text})
        elif name in self._functions:
            return self._functions[name]["func"](input_text)

    def get_tools_description(self) -> str:
        """获取格式化的工具列表（注入 LLM 提示词）"""
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys()) + list(self._functions.keys())
```

**注册与调用流程：**

```
1. 创建 MemoryTool 实例
2. 注册到 ToolRegistry
3. Agent 在 LLM 提示词中包含工具描述
4. LLM 决定调用 "memory" 工具并生成参数
5. Agent 解析 LLM 输出，调用 ToolRegistry.execute_tool("memory", ...)
6. ToolRegistry 路由到 MemoryTool.run()
7. MemoryTool 执行操作并返回结果
8. 结果注入回 LLM 上下文
```

---

## 11. Agent 集成方式

**文件路径：** `hello_agents/core/agent.py`

### Agent 基类

```python
class Agent(ABC):
    def __init__(self, name: str, llm: HelloAgentsLLM,
                 system_prompt: Optional[str] = None, ...):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """执行 Agent"""
```

### SimpleAgent 使用 MemoryTool

```python
# 完整集成示例
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool

# 1. 创建 LLM
llm = HelloAgentsLLM()

# 2. 创建 MemoryTool
memory_tool = MemoryTool(
    user_id="user123",
    memory_types=["working", "episodic", "semantic"]
)

# 3. 注册工具
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)

# 4. 创建 Agent
agent = SimpleAgent(
    name="智能助手",
    llm=llm,
    tool_registry=tool_registry,
    system_prompt="你是一个有记忆能力的智能助手，可以使用 memory 工具来记录和检索信息。"
)

# 5. 对话
response = agent.run("记住我叫张三，是一名 Python 开发者")
# Agent 内部：LLM 判断需要调用 memory 工具 → 自动添加到语义记忆

response = agent.run("我叫什么名字？")
# Agent 内部：LLM 判断需要搜索记忆 → 检索到 "张三" → 回答
```

---

## 12. 遗忘与巩固机制

### 12.1 遗忘策略

```python
def forget_memories(self, strategy="importance_based", threshold=0.1,
                    max_age_days=30) -> int:
```

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `importance_based` | 删除重要性低于阈值的记忆 | 清理低价值信息 |
| `time_based` | 删除超过指定天数的记忆 | 定期清理旧数据 |
| `capacity_based` | 只保留 Top-N 条记忆 | 控制存储容量 |

### 12.2 巩固机制

巩固是将**短期记忆**（Working）转移到**长期记忆**（Episodic/Semantic）的过程：

```
巩固流程：
1. 从 WorkingMemory 中筛选重要性 >= 阈值的记忆
2. 将记忆的重要性提升 10%（min(1.0, importance * 1.1)）
3. 修改 memory_type 为目标类型
4. 添加到目标记忆类型
5. 从 WorkingMemory 中移除

效果：高价值的短期记忆被"固化"为长期记忆，不会随时间衰减或过期丢失
```

### 12.3 WorkingMemory 的时间衰减

```python
# 时间衰减公式
time_decay = decay_factor ** (hours_elapsed / 6)

# 示例（decay_factor = 0.95）：
# 6小时后：  0.95^1 = 0.950
# 12小时后： 0.95^2 = 0.903
# 24小时后： 0.95^4 = 0.815
# 48小时后： 0.95^8 = 0.663
```

---

## 13. 完整使用示例

### 基本使用

```python
from hello_agents.tools import MemoryTool
from hello_agents.memory.base import MemoryConfig

# 创建配置
config = MemoryConfig(
    storage_path="./my_memory_data",
    working_memory_capacity=20,
    working_memory_ttl_minutes=60
)

# 创建 MemoryTool
tool = MemoryTool(
    user_id="user_001",
    memory_config=config,
    memory_types=["working", "episodic", "semantic", "perceptual"]
)

# === 添加记忆 ===
# 添加到工作记忆
tool.execute("add", content="当前正在讨论项目架构")

# 添加到语义记忆（知识）
tool.execute("add", content="Python 是一种动态类型语言",
             memory_type="semantic", importance=0.9)

# 添加到情景记忆（事件）
tool.execute("add", content="用户今天询问了数据库优化方案",
             memory_type="episodic", importance=0.7)

# === 搜索记忆 ===
result = tool.execute("search", query="Python 编程语言", limit=5)
print(result)

# === 获取统计 ===
stats = tool.execute("stats")
print(stats)

# === 巩固记忆 ===
# 将工作记忆中重要的内容转移到情景记忆
count = tool.execute("consolidate", from_type="working",
                     to_type="episodic", importance_threshold=0.6)

# === 遗忘旧记忆 ===
tool.execute("forget", strategy="time_based", max_age_days=30)

# === 便捷方法 ===
tool.add_knowledge("机器学习是人工智能的一个分支")
context = tool.get_context_for_query("什么是深度学习？")
tool.auto_record_conversation("你好", "你好！有什么可以帮助你的？")
```

### 与 Agent 集成

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool

llm = HelloAgentsLLM()
memory_tool = MemoryTool(user_id="user123")

registry = ToolRegistry()
registry.register_tool(memory_tool)

agent = SimpleAgent(
    name="记忆助手",
    llm=llm,
    tool_registry=registry,
    system_prompt="""你是一个有记忆能力的智能助手。
    你可以使用 memory 工具来：
    - 记住用户告诉你的信息（action: add）
    - 回忆之前的信息（action: search）
    - 查看记忆摘要（action: summary）
    当用户让你记住什么，或问你之前的事情时，主动使用记忆工具。"""
)

# 多轮对话
agent.run("记住我叫小明，今年25岁，是一名前端工程师")
agent.run("我最近在学习 React 框架")
agent.run("我叫什么？在学什么？")  # Agent 会自动搜索记忆并回答
```

---

## 14. 记忆类型特性对比

| 特性 | WorkingMemory | EpisodicMemory | SemanticMemory | PerceptualMemory |
|------|:---:|:---:|:---:|:---:|
| **存储方式** | 纯内存 | SQLite+Qdrant | Qdrant+Neo4j | SQLite+Qdrant |
| **是否持久化** | 否 | 是 | 是 | 是 |
| **容量限制** | 10 条 | 无限制 | 无限制 | 无限制 |
| **TTL 过期** | 120 分钟 | 无 | 无 | 无 |
| **时间衰减** | 是 (0.95) | 近因权重 | 无 | 无 |
| **检索方式** | TF-IDF + 关键词 | 向量 + SQLite回退 | 向量 + 图搜索 | 向量（按模态） |
| **上下文丰富度** | 最少 | 丰富（会话/结果） | 关系型（实体/关系） | 模态元数据 |
| **支持模态** | 仅文本 | 仅文本 | 仅文本 | 多模态 |
| **模式发现** | 否 | 是 | 否 | 否 |
| **跨模态搜索** | 否 | 否 | 否 | 是 |

---

## 15. 配置参考

### MemoryConfig 参数

```python
MemoryConfig(
    storage_path="./memory_data",
    max_capacity=100,
    importance_threshold=0.1,
    decay_factor=0.95,
    working_memory_capacity=10,
    working_memory_tokens=2000,
    working_memory_ttl_minutes=120,
    perceptual_memory_modalities=["text", "image", "audio", "video"]
)
```

### 环境变量

```bash
# 嵌入模型配置
EMBED_MODEL_TYPE=local                                    # 或 "dashscope"
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2   # 模型名称

# Qdrant 向量数据库
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=hello_agents_vectors

# Neo4j 知识图谱（语义记忆需要）
NEO4J_URI=bolt://localhost:7687
NEO4J_AUTH=neo4j/password
```

---

## 总结

HelloAgents 0.2.0 的 MemoryTool 是一个**架构精良、层次分明**的记忆系统实现：

1. **分层设计**：Tool → Manager → MemoryTypes → Storage → Embedding，各层职责清晰，接口明确
2. **仿生记忆**：四种记忆类型对应认知科学的记忆分类，各有不同的存储策略和生命周期
3. **混合检索**：结合向量语义搜索、关键词匹配、知识图谱推理等多种策略，提升检索质量
4. **优雅降级**：从 DashScope → SentenceTransformers → TF-IDF，从 Qdrant → SQLite，确保系统在不同环境下都能运行
5. **自动化智能**：自动记忆分类、自动重要性计算、自动时间衰减、自动巩固等机制减少手动管理负担
6. **灵活扩展**：通过抽象基类和配置化设计，可以方便地添加新的记忆类型、存储后端或嵌入模型
