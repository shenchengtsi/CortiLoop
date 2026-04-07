# CortiLoop vs Hindsight vs mem0 vs OpenViking 详细对比报告

> 四种 Agent 记忆系统的设计哲学、架构差异和工程取舍的全面对比。

---

## 1. 一句话定位

| 系统 | 核心隐喻 | 一句话定位 |
|------|---------|-----------|
| **CortiLoop** | 大脑完整生命周期 | 仿生记忆引擎：注意力→编码→巩固→检索→关联→遗忘→再巩固，7 层全覆盖 |
| **Hindsight** | 大脑分层 | 仿生记忆引擎：retain/recall/reflect 三操作，LongMemEval SOTA |
| **mem0** | 便利贴 | 通用记忆层：极简接入 (3 行代码)，20+ 向量存储，生态最广 |
| **OpenViking** | 文件系统 | 上下文数据库：用 FS 范式统一管理记忆 + 资源 + 技能，L0/L1/L2 按需加载 |

---

## 2. 基础信息

| 维度 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| **语言** | Python | Python + TS + Rust | Python | Go + C++ + Rust |
| **License** | MIT | MIT | Apache-2.0 | **AGPL-3.0** |
| **Stars** | 新项目 | 7,659 | 52,124 | 21,374 |
| **数据库** | SQLite（零依赖） | PostgreSQL（嵌入式） | 20+ 向量存储可选 | VikingDB / 本地 |
| **LLM 依赖** | OpenAI / Anthropic | OpenAI / Anthropic / 本地 | 10+ 提供商 | 豆包 / OpenAI |
| **部署复杂度** | `pip install` | Docker / Helm | `pip install` | 编译 (Go+C+++Rust) |
| **商业友好度** | ★★★★★ MIT | ★★★★★ MIT | ★★★★ Apache | ★ AGPL 封锁 |

---

## 3. 设计哲学深度对比

### 3.1 记忆写入哲学

```
CortiLoop:  输入 → [注意力门控：值得记吗？] → 事实提取 → 永久写入
Hindsight:  输入 → 事实提取 → 永久写入（无门控，全量记录）
mem0:       输入 → 事实提取 → ADD/UPDATE/DELETE（破坏性改写）
OpenViking: 输入 → write 到 URI 路径 → commit 后可检索
```

**关键差异：注意力门控**

| | CortiLoop | Hindsight | mem0 | OpenViking |
|---|---|---|---|---|
| 写入前筛选 | **是** — 5 维度重要性评分 | 否 — 依赖 LLM 提取质量 | 否 — 全量记录 | 否 — 手动决定 |
| 纠错信号加权 | **最高权重 (0.30)** | 无 | 无 | 无 |
| 新奇度评估 | **是** — 与现有记忆对比 | 无 | 无 | 无 |

**为什么这很重要**：mem0 的 97.8% 噪声问题（Issue #4573，10134 条记忆仅 2.2% 有价值）本质上就是因为缺乏写入前筛选。CortiLoop 通过注意力门控在写入前过滤噪声，从源头解决。

### 3.2 记忆进化哲学

这是四个系统最根本的分歧。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        记忆进化光谱                                  │
│                                                                     │
│  ← 永不改写                                     彻底覆盖 →          │
│                                                                     │
│  CortiLoop          Hindsight          OpenViking        mem0       │
│  ┃                  ┃                  ┃                 ┃          │
│  ┃ 原始 units 只读  ┃ 原始 units 只读  ┃ 覆盖写入        ┃ LLM 决策 │
│  ┃ Observations 可  ┃ Observations 可  ┃ 但有版本控制     ┃ ADD/     │
│  ┃ 更新但保留       ┃ 更新但保留       ┃                 ┃ UPDATE/  │
│  ┃ 完整 history[]  ┃ 完整历史         ┃                 ┃ DELETE   │
│  ┃ + 冲突检测       ┃                  ┃                 ┃ 直接改写 │
│  ┃ + 来源标记       ┃                  ┃                 ┃          │
└─────────────────────────────────────────────────────────────────────┘
```

| 操作 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 旧事实保留？ | **永久保留**（memory_units 只读） | **永久保留** | **被覆盖或删除** | 覆盖（可有版本） |
| 知识如何进化？ | Observation 更新 + history[] | Observation 更新 | 直接改写原文 | 重写文件 |
| 能追溯历史？ | **是** — history[] + conflicts 表 | **是** — 原始 units 可查 | 仅 history.db 日志 | 需版本控制 |
| 冲突检测？ | **是** — LLM 检测 + 3 种决议 | 否 | 否 | 否 |
| 来源标记？ | **是** — user_said vs llm_inferred | 否 | 否 | 否 |

**生物学视角**：人脑的再巩固是"更新"而非"替换"——检索记忆时进入不稳定窗口，整合新信息后重新稳定化，但原始痕迹不会被抹除。CortiLoop 和 Hindsight 符合这一原则，mem0 的破坏性更新不符合。

### 3.3 记忆层级哲学

```
CortiLoop (4 层 + 图谱):
┌─────────────────────────────────────┐
│ Mental Models (λ=0.01)              │  ← reflect() 生成
│ Observations (λ=0.03)              │  ← 突触巩固生成
│ Memory Units (λ=0.1)               │  ← 即时编码
│ Procedural Memories (λ=0.005)      │  ← 重复模式检测
│ ─── Association Graph (赫布动态) ───│  ← 贯穿所有层
└─────────────────────────────────────┘

Hindsight (3 层 + 图谱):
┌─────────────────────────────────────┐
│ Mental Models                       │  ← reflect 生成
│ Observations                        │  ← Consolidation 生成
│ Memory Units                        │  ← retain 写入
│ ─── 实体/时序/因果图谱 ──────────── │
└─────────────────────────────────────┘

mem0 (1 层):
┌─────────────────────────────────────┐
│ Facts (扁平事实列表)                 │  ← 全部在同一层
│ + 可选 graph_store                  │
└─────────────────────────────────────┘

OpenViking (3 层 × 3 类):
┌─────────────────────────────────────┐
│ L0: 极简摘要 (几十 tokens)          │
│ L1: 中等详情 (几百 tokens)          │
│ L2: 完整内容 (任意大小)             │
│ × memory / resource / skill        │
└─────────────────────────────────────┘
```

| 分层能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|---------|-----------|-----------|------|------------|
| 记忆层级 | 4 层 (units→obs→models→procedural) | 3 层 (units→obs→models) | **1 层 (扁平)** | 3 层 (L0→L1→L2) |
| 层级含义 | 抽象度递增 | 抽象度递增 | N/A | **详情度递增** |
| 程序记忆 | **是** — 重复模式检测→固化 | 否 | 部分 — 有 procedural 概念 | 部分 — skill 管理 |
| Token 效率 | 中等 (按 top_k 裁剪) | 中等 (token budget) | 低 (全量返回) | **最高** (L0/L1 按需) |

**关键差异**：OpenViking 的 L0/L1/L2 是"详情粒度"维度（同一事物的不同精度），而 CortiLoop/Hindsight 的分层是"抽象度"维度（从原始事实到归纳知识到心智模型）。两种设计解决不同问题。

---

## 4. 七大能力维度逐项对比

### 4.1 编码（写入）

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 注意力门控 | ✅ 5 维度评分 | ❌ | ❌ | ❌ |
| LLM 事实提取 | ✅ | ✅ (Phase 1) | ✅ | ❌ (手动/模板) |
| 实体解析 | ✅ (LLM 提取) | ✅ (pg_trgm 模糊匹配) | ✅ (基础) | ✅ (NER) |
| Embedding | ✅ (批量) | ✅ (本地模型可选) | ✅ | ✅ |
| 编码上下文保存 | ✅ (task, entities, mood, session) | 部分 (namespace) | ❌ | 部分 (URI 路径) |
| 写入后即可检索 | ✅ | ✅ | ✅ | ❌ (需 commit) |
| 写入成本 | 1 次 LLM + 1 次 embed | 1 次 LLM + 1 次 embed | 1 次 LLM + 1 次 embed | 0-1 次 LLM |

### 4.2 巩固

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 突触巩固 (即时) | ✅ units→observations | ✅ Consolidation Worker | ❌ | ❌ |
| 系统巩固 (批量) | ✅ reflect() | ✅ reflect | ❌ | ❌ |
| 程序记忆习得 | ✅ 重复模式检测 | ❌ | ❌ | ❌ |
| 心智模型生成 | ✅ | ✅ (Agentic loop) | ❌ | ❌ |
| 巩固调度 | ✅ 手动 + 定时 | 每次 retain 后异步 | N/A | N/A |
| 单维度约束 | ✅ | ✅ | N/A | N/A |

### 4.3 检索

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 语义向量检索 | ✅ (numpy cosine) | ✅ (HNSW pgvector) | ✅ | ✅ |
| 关键词检索 | ✅ (LIKE) | ✅ (BM25 tsvector) | ❌ | ✅ |
| 图谱遍历 | ✅ (扩散激活) | ✅ (LinkExpansion) | 部分 (graph_store) | ❌ |
| 时序检索 | ✅ (时间意图提取) | ✅ (LLM 时间解析) | ❌ | ❌ |
| 融合排序 | ✅ RRF (k=60) | ✅ RRF (k=60) | 单路 | 自有排序 |
| 重排序 | ❌ | ✅ Cross-encoder | ❌ | ❌ |
| 检索后强化 | ✅ access_count++ | ❌ | ❌ | ❌ |
| 赫布共检索强化 | ✅ edge.weight++ | ❌ | ❌ | ❌ |

**检索对比总结**：

```
检索路数:
  CortiLoop:  4 路 (语义 + 关键词 + 图谱扩散 + 时序)
  Hindsight:  4 路 (语义 + BM25 + 图谱 + 时序)       ← 最成熟
  mem0:       1 路 (语义向量)
  OpenViking: 2 路 (语义 + 关键词)

独特检索能力:
  CortiLoop:  扩散激活（联想检索）+ 检索后赫布强化
  Hindsight:  Cross-encoder 重排序（精度最高）
  mem0:       无
  OpenViking: L0 快速扫描（Token 效率最高）
```

### 4.4 关联 (知识图谱)

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 知识图谱 | ✅ | ✅ | 部分 (可选) | ❌ |
| 边类型 | 4 种 (共现/时序/因果/语义) | 3 种 (时序/语义/因果) | 关系提取 | N/A |
| 动态权重 | ✅ **赫布学习** (共活强化,衰减弱化) | ❌ (静态边) | ❌ | N/A |
| 扩散激活 | ✅ (多跳遍历 + 距离衰减) | ✅ (LinkExpansion) | ❌ | N/A |

**关键差异**：CortiLoop 的图谱边权重是**动态的**——共同检索的记忆之间连接增强，长期不激活则衰减。这模仿了赫布学习的核心原则。Hindsight 的图谱边是静态的，创建后权重不变。

### 4.5 遗忘

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 被动衰减 | ✅ **艾宾浩斯曲线** (R=e^{-λt}) | ❌ | ❌ | ❌ |
| 差异化衰减速率 | ✅ (episodic > semantic > procedural) | ❌ | ❌ | ❌ |
| 间隔重复效应 | ✅ (检索重置衰减 + boost) | ❌ | ❌ | ❌ |
| 主动去重 | ✅ (cosine > 0.92 → archive) | ❌ | ✅ (UPDATE 合并) | ❌ |
| 容量管理 | ✅ (超阈值 → archive 最弱) | ❌ | ❌ | ❌ |
| 状态转换 | ✅ ACTIVE → ARCHIVE → COLD | ❌ (永久 ACTIVE) | DELETE (直接删除) | ❌ |

**这是 CortiLoop 相对所有竞品的最大差异化能力**。

```
随时间推移的记忆系统行为:

CortiLoop:  活跃记忆稳定增长 → 不用的自然衰减归档 → 总量可控
Hindsight:  记忆单调递增 → 无上限 → 最终检索性能退化
mem0:       记忆单调递增 + 噪声累积 → 97.8% 噪声 → 检索严重退化
OpenViking: 手动管理 → 依赖用户自觉整理
```

### 4.6 再巩固 (记忆更新)

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 冲突自动检测 | ✅ LLM 检测 | ❌ | ❌ | ❌ |
| 更新决议类型 | ✅ supersede/merge/coexist | 覆盖式 UPDATE | ADD/UPDATE/DELETE | 覆盖写入 |
| 变更历史保留 | ✅ history[] 完整记录 | ✅ 原始 units 可查 | 仅 history.db 日志 | 需版本控制 |
| 来源标记 | ✅ user_said vs llm_inferred | ❌ | ❌ | ❌ |
| 防幻觉污染 | ✅ 低置信度推断标记 | ❌ | ❌ | ❌ |

### 4.7 管理范围

| 能力 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 记忆管理 | ✅ | ✅ | ✅ | ✅ |
| 文档/资源管理 | ❌ | ❌ | ❌ | **✅** |
| 技能/工具管理 | ❌ | ❌ | ❌ | **✅** |
| 多模态 (图/音/视频) | ❌ | ❌ | ❌ | **✅** |
| Bot 集成 (飞书/钉钉) | ❌ | ❌ | ❌ | **✅** |

**OpenViking 独特定位**：它不只是记忆系统，而是 Agent 的"操作系统存储层"。如果需要统一管理记忆+资源+技能，OpenViking 是唯一选择（但受 AGPL 限制）。

---

## 5. 架构差异图

```
                    CortiLoop                              Hindsight
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  AttentionGate → Encoder         │  │  LLM Extraction → Phase 1/2/3   │
│       ↓                          │  │       ↓                          │
│  SQLite (memory_units) ←── 只读  │  │  PostgreSQL (memory_units)←只读 │
│       ↓                          │  │       ↓                          │
│  SynapticConsolidator ──async──→ │  │  Consolidation Worker ──async──→│
│       ↓                          │  │       ↓                          │
│  Observations (可更新+history)   │  │  Observations (可更新)           │
│       ↓                          │  │       ↓                          │
│  SystemsConsolidator ──periodic→ │  │  Reflect (Agentic loop)         │
│       ↓                          │  │       ↓                          │
│  Mental Models + Procedurals     │  │  Mental Models                   │
│       ↓                          │  │       ↓                          │
│  AssociationGraph (赫布动态)     │  │  Knowledge Graph (静态)          │
│       ↓                          │  │       ↓                          │
│  MultiProbe (4路) + RRF         │  │  4路 gather + RRF + CrossEnc    │
│       ↓                          │  │       ↓                          │
│  DecayManager + Pruner ──sweep→  │  │  (无遗忘机制)                    │
│       ↓                          │  │                                  │
│  Reconsolidator (冲突检测)       │  │  (无冲突检测)                    │
└──────────────────────────────────┘  └──────────────────────────────────┘

                    mem0                               OpenViking
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  LLM Extraction                  │  │  write(URI, content)             │
│       ↓                          │  │       ↓                          │
│  LLM Decision: ADD/UPDATE/DELETE │  │  SemanticDagExecutor             │
│       ↓                          │  │  (L0/L1/L2 自动生成)             │
│  Vector Store (扁平)             │  │       ↓                          │
│       ↓                          │  │  commit()                        │
│  (无巩固)                        │  │       ↓                          │
│  (无图谱，或可选 graph_store)     │  │  VikingDB / 本地存储             │
│       ↓                          │  │       ↓                          │
│  语义检索 (单路)                 │  │  语义 + 关键词检索               │
│       ↓                          │  │  L0 快速扫描 → L1/L2 按需       │
│  (无遗忘)                        │  │       ↓                          │
│  (无冲突检测)                    │  │  (无遗忘)                        │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

---

## 6. 仿生维度覆盖雷达图

```
                    注意力门控
                        │
                   5 ── ● ── CortiLoop
                  4 ──╱   ╲
                 3 ─╱       ╲
          记忆    2╱    ·     ╲  分层
          衰减   1╱      ·    ╲  巩固
                 ╱        ·    ╲
                ●─────────·─────●
               ╱ ╲        ·  ╱ ╲
              ╱   ╲       · ╱   ╲
             ╱     ╲      ·╱     ╲
            ╱       ╲     ╱       ╲
           ●─────────●──●─────────●
         再巩固     关联图谱     多路检索
          冲突检测              


   CortiLoop  ████████  全面覆盖 (7/7 维度)
   Hindsight  ██████    强覆盖 (5/7: 编码+巩固+图谱+检索+分层)
   mem0       ██        弱覆盖 (1/7: 仅基础编码)
   OpenViking ███       部分覆盖 (2/7: 分层存储+编码)


各维度得分:

维度          CortiLoop  Hindsight  mem0  OpenViking
───────────────────────────────────────────────────
注意力门控      5          0         0       0
分层巩固        5          5         0       0
多路检索        4          5         1       2
关联图谱        5          4         1       0
再巩固/冲突     5          2         0       0
记忆衰减        5          0         0       0
程序记忆        4          0         1       2
───────────────────────────────────────────────────
仿生总分       33         16         3       4
```

---

## 7. 性能与工程取舍

### 7.1 写入延迟

```
单次 retain 操作:

CortiLoop:  注意力评分(本地) + LLM提取(~1s) + embed(~0.3s) + 巩固(~1s) = ~2.5s
Hindsight:  LLM提取(~1s) + embed(~0.3s) + 巩固(异步,不阻塞) = ~1.5s
mem0:       LLM提取(~1s) + embed(~0.3s) + LLM决策(~1s) = ~2.3s
OpenViking: write(~0.1s) + commit + DAG(异步) = ~0.3s (无LLM时)

注: CortiLoop 的注意力门控可能直接跳过 (score < 0.2)，此时延迟 = ~0.01s
```

### 7.2 检索延迟

```
单次 recall 操作:

CortiLoop:  embed(~0.3s) + 4路SQLite查询(~0.1s) + RRF(本地) = ~0.5s
Hindsight:  embed(~0.3s) + 4路PostgreSQL(~0.1s) + CrossEnc(~0.5s) = ~1.0s
mem0:       embed(~0.3s) + 向量搜索(~0.1s) = ~0.4s
OpenViking: L0扫描(~0.1s) + embed(~0.3s) = ~0.4s
```

### 7.3 存储效率

| 维度 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 每条记忆存储 | ~2 KB (content + embedding blob) | ~3 KB (PG row + index) | ~1.5 KB | ~0.5-5 KB (L0/L1/L2) |
| 外部依赖 | SQLite (内置) | PostgreSQL | 需外部向量 DB | 需编译 Go/C++ |
| 冷启动 | `pip install` | `docker compose up` | `pip install` | 编译 30+ 分钟 |

### 7.4 规模上限

| 维度 | CortiLoop | Hindsight | mem0 | OpenViking |
|------|-----------|-----------|------|------------|
| 向量搜索 | numpy 暴力扫描 (~100K 可用) | HNSW pgvector (百万级) | 取决于后端 (亿级) | VikingDB (百万级) |
| 图谱查询 | SQLite 遍历 (~10K 边可用) | PostgreSQL 图谱 (~100K) | 可选 Neo4j | N/A |
| 瓶颈 | **向量搜索 (numpy)** | 磁盘 I/O | 向量 DB 成本 | 编译/部署 |

**CortiLoop 的规模限制**：当前使用 numpy 暴力计算 cosine similarity，适合 10 万级记忆。超过此规模需要替换为 pgvector/Faiss/usearch。这是有意的设计取舍：零依赖 vs 规模性能。

---

## 8. 适用场景决策

```
你的场景是什么？
│
├── 需要管理记忆+资源+技能（全栈 Agent OS）？
│   └── 是 → OpenViking（但注意 AGPL）
│
├── 需要已验证的 SOTA 精度 + 生产环境稳定性？
│   └── 是 → Hindsight（MIT, Helm chart, benchmark 验证）
│
├── 需要最快 MVP + 已有向量基础设施？
│   └── 是 → mem0（3 行代码, 20+ 向量存储）
│
├── 需要记忆系统自动管理质量（噪声过滤+遗忘+衰减）？
│   └── 是 → CortiLoop
│
├── 需要完整的仿生记忆生命周期研究平台？
│   └── 是 → CortiLoop
│
├── 需要记忆与知识图谱深度融合 + 联想检索？
│   └── 是 → CortiLoop（赫布动态图谱 + 扩散激活）
│
└── 需要商业 SaaS + 最广泛 LLM/向量存储兼容？
    └── mem0 或 Hindsight
```

### 场景速查表

| 场景 | 首选 | 次选 | 原因 |
|------|------|------|------|
| 个人 AI 助手（长期记忆） | **CortiLoop** | Hindsight | 注意力门控 + 遗忘 = 长期可持续 |
| 商业 SaaS 产品 | Hindsight | mem0 | MIT + SOTA + 生产就绪 |
| 快速原型 / MVP | mem0 | CortiLoop | 最简接入 |
| 研究型 Agent 记忆 | **CortiLoop** | Hindsight | 仿生完整度最高 + 可调参数多 |
| 企业知识管理平台 | OpenViking | Hindsight | 多模态 + 资源 + 技能统一管理 |
| 高频写入（IoT/日志） | mem0 | OpenViking | 低延迟写入 |
| 需要联想/创造性检索 | **CortiLoop** | Hindsight | 扩散激活 → 查"Alice"能联想到"ProjectX" |
| 10 万+ 记忆规模 | Hindsight | mem0 | pgvector HNSW |
| nanobot 集成 | **CortiLoop** | Hindsight | 原生 Python + MCP 双模式 |
| openclaw 集成 | **CortiLoop** | Hindsight | MCP + skill manifest |

---

## 9. CortiLoop 相对竞品的独特价值总结

### 只有 CortiLoop 有的能力

| 能力 | 价值 | 竞品现状 |
|------|------|---------|
| **注意力门控** | 从源头过滤噪声，解决 mem0 97.8% 问题 | 所有竞品都没有 |
| **艾宾浩斯衰减** | 未使用记忆自然退化，防止检索退化 | 所有竞品都没有 |
| **间隔重复效应** | 常用记忆越来越容易找到 | 所有竞品都没有 |
| **赫布动态图谱** | 关联随使用自然进化 | Hindsight 有图谱但静态 |
| **冲突检测 + 3 种决议** | supersede/merge/coexist，防止信息丢失 | 所有竞品都没有 |
| **来源标记** | 区分"用户说的"vs"LLM 推断的"，防幻觉污染 | 所有竞品都没有 |
| **程序记忆自动习得** | 从重复模式中自动提取工作流 | 所有竞品都没有 |
| **差异化衰减速率** | 情景记忆快衰、语义慢衰、程序极慢 | 所有竞品都没有 |

### CortiLoop 相对不足的地方

| 不足 | 影响 | 弥补方案 |
|------|------|---------|
| 向量搜索用 numpy 暴力扫描 | 10 万+记忆时变慢 | 替换为 Faiss/usearch |
| 无 Cross-encoder 重排序 | 检索精度略低于 Hindsight | 可添加重排步骤 |
| 新项目，无生产验证 | 稳定性未知 | 渐进式采用 |
| 不管理资源/技能 | 不如 OpenViking 全面 | 专注做好记忆这一件事 |
| 绑定 OpenAI/Anthropic | LLM 选择受限 | 扩展 LLM 适配器 |

---

## 10. 演进路线建议

```
Phase 1 — 当前 (v0.1)
  ✅ 7 层仿生架构完整实现
  ✅ MCP + nanobot + openclaw 适配器
  ✅ SQLite 零依赖存储

Phase 2 — 规模化 (v0.2)
  · 替换 numpy → Faiss/usearch 向量索引
  · 添加 Cross-encoder 重排序
  · 支持更多 LLM (Ollama 本地模型)
  · 异步巩固改为真正的 background worker

Phase 3 — 生产化 (v0.3)
  · PostgreSQL 存储后端（可选）
  · Benchmark: LongMemEval 对标 Hindsight
  · 多租户 + 认证
  · 可视化面板 (记忆图谱浏览)

Phase 4 — 差异化 (v1.0)
  · 情绪记忆调制 (杏仁核层)
  · 工作记忆动态管理 (context window 自动加载/卸载)
  · 图式加速巩固 (新实体自动继承默认关联)
  · 联邦记忆 (多 Agent 记忆共享)
```

---

*对比报告日期：2026-04-07*
*基于源码精读 + wiki 深度分析 + CortiLoop 实现*
