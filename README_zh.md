# CortiLoop

**仿生 Agent 记忆引擎** — 模拟人类大脑记忆的完整生命周期。

[English Version / 英文文档](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()

> 可集成 [nanobot](https://github.com/HKUDS/nanobot)、[openclaw](https://github.com/openclaw/openclaw) 及任何 MCP 兼容的 Agent 框架。

---

## 为什么选择 CortiLoop？

大多数 Agent 记忆系统只是简单的键值存储或 RAG。而真正的大脑通过 **编码、巩固、检索、关联、遗忘和再巩固** 来管理记忆 —— 一套完整的生命周期，让知识在长期使用中保持准确、相关且可管理。

CortiLoop 实现了这套完整的生命周期：

| 问题 | 大脑的解决方案 | CortiLoop 的实现 |
|------|--------------|-----------------|
| 噪声泛滥 | 前额叶注意力门控 | 5 维重要性评分 — 纠错和显式标记获得最高权重 |
| 知识过时 | 再巩固窗口 | 冲突检测 + 取代/合并/共存三种决议 |
| 检索退化 | 遗忘曲线 | 艾宾浩斯衰减，按记忆层级差异化衰减速率 |
| 碎片化召回 | 模式补全 (CA3) | 4 路多探针搜索 + 倒数排名融合 (RRF) |
| 缺乏关联 | 赫布学习 | 动态知识图谱 + 扩散激活 |
| 信息过载 | 睡眠巩固 | 后台 worker 周期性深度巩固 + 修剪 |

## 架构

```
Agent 输入 → [注意力门控] → [编码器] → [海马体存储]
                                              │
                              ┌───────────────┤
                              ↓               ↓
                      [突触巩固]          [关联图谱]
                      (事实→观察)         (赫布边)
                              │
                              ↓ (周期性)
                      [系统巩固]
                      (心智模型、程序记忆检测)
                              │
      [多探针检索] ←──────────┘
      (语义+关键词+图谱+时间 → RRF 融合)
                              │
                      [再巩固]          [遗忘]
                      (冲突检测)        (衰减+修剪)
```

### 7 层仿生架构

| 层 | 大脑类比 | 功能 |
|----|---------|------|
| **注意力门控** | 前额叶 + 多巴胺新奇信号 | 评估重要性；编码前过滤噪声 |
| **编码器** | 海马体编码 + 实体绑定 | 通过 LLM 提取结构化事实、实体、嵌入向量 |
| **巩固** | 睡眠驱动的海马体→新皮层转移 | 突触巩固（即时）+ 系统巩固（深度/周期性）|
| **关联** | 赫布学习 + 扩散激活 | 知识图谱：共现/时序/因果/语义边 |
| **检索** | CA3 模式补全 + 多模态融合 | 4 路搜索 + RRF + 可选 cross-encoder 重排序 |
| **遗忘** | 艾宾浩斯曲线 + 小胶质细胞修剪 | 强度衰减、去重、容量管理 |
| **再巩固** | 记忆去稳定 + 再稳定 | 冲突检测、安全更新、历史保留 |

## 功能特性

### 核心 (v0.1)
- 7 层仿生记忆生命周期
- MCP 服务器 + nanobot 插件 + openclaw 技能
- SQLite 零依赖存储
- 中英文双语注意力门控

### 规模化 (v0.2)
- 可插拔向量索引（usearch HNSW / numpy 回退）
- Ollama 本地 LLM 支持（完全离线）
- litellm 万能适配器（100+ LLM 提供商）
- Cross-encoder 重排序
- 后台巩固 worker

### 生产化 (v0.3)
- PostgreSQL + pgvector 存储后端
- 多租户认证（API key → 命名空间隔离）
- LongMemEval 基准测试（5 维度、13 个测试用例）
- Web 可视化面板（D3.js 知识图谱 + 仪表盘）
- `BaseStore` 抽象，支持自定义存储后端
- **48 个测试** 全部通过

## 快速开始

```bash
pip install cortiloop

# 可选后端：
pip install cortiloop[usearch]     # HNSW 向量索引
pip install cortiloop[postgres]    # PostgreSQL + pgvector
pip install cortiloop[all]         # 全部安装
```

### Python API

```python
import asyncio
from cortiloop import CortiLoop, CortiLoopConfig

async def main():
    loop = CortiLoop(CortiLoopConfig(db_path="memory.db"))

    # 存储（注意力门控自动过滤噪声）
    await loop.retain("Alice 是 ProjectX 的产品经理，使用 React + TypeScript")
    await loop.retain("好的")  # 被注意力门控过滤

    # 检索（多探针融合）
    results = await loop.recall("Alice 的项目是什么？")
    for r in results:
        print(f"[{r['type']}] {r['content']} (score: {r['score']:.3f})")

    # 深度巩固
    await loop.reflect()
    loop.close()

asyncio.run(main())
```

### MCP 服务器

```bash
export OPENAI_API_KEY=sk-...
cortiloop-mcp
```

### 使用 Ollama（完全本地，无需 API key）

```python
config = CortiLoopConfig(db_path="memory.db")
config.llm.provider = "ollama"
config.llm.model = "llama3.1"
config.llm.embedding_model = "nomic-embed-text"
config.llm.embedding_dim = 768
loop = CortiLoop(config)
```

### 使用 PostgreSQL（生产级规模）

```bash
pip install cortiloop[postgres]
```

```python
config = CortiLoopConfig(
    db_path="postgresql://user:pass@localhost:5432/cortiloop",
    storage_backend="postgres",  # 原生 pgvector HNSW
)
loop = CortiLoop(config)
```

### 可视化面板

```bash
cortiloop-viz --db cortiloop.db --port 8765
# 打开 http://localhost:8765
```

功能：力导向知识图谱、统计仪表盘、记忆时间线、衰减曲线图。

### 基准测试

```bash
cortiloop-bench --provider openai --model gpt-4o-mini
```

评测 5 个维度：信息提取、时间推理、知识更新、关联检索、多会话推理。

## 集成方式

### nanobot

```json
{
  "mcp": {
    "servers": {
      "cortiloop": {
        "command": "python",
        "args": ["-m", "cortiloop.adapters.mcp_server"],
        "env": { "CORTILOOP_DB_PATH": "~/.nanobot/cortiloop.db" }
      }
    }
  }
}
```

### openclaw

```json
{
  "cortiloop": {
    "command": "python",
    "args": ["-m", "cortiloop.adapters.mcp_server"],
    "env": { "CORTILOOP_DB_PATH": "~/.openclaw/cortiloop.db" }
  }
}
```

### nanobot 直接插件（Python）

```python
from cortiloop.adapters.nanobot_plugin import NanobotMemoryPlugin

memory = NanobotMemoryPlugin({"db_path": "memory.db"})
await memory.on_user_message("我喜欢 TypeScript 严格模式")
context = await memory.on_before_response("写一个 React 组件")
# context 包含相关记忆，注入到 prompt 中
```

## MCP 工具

| 工具 | 描述 |
|------|------|
| `cortiloop_retain` | 通过注意力门控将文本存入长期记忆 |
| `cortiloop_recall` | 多探针检索 + RRF 融合 |
| `cortiloop_reflect` | 深度巩固周期（程序记忆检测 + 衰减 + 修剪）|
| `cortiloop_stats` | 记忆系统统计信息 |

## 配置

参见 [config.example.yaml](config.example.yaml) 了解所有选项。

```yaml
storage_backend: "sqlite"       # "sqlite" | "postgres"
vector_backend: "auto"          # "auto" | "numpy" | "usearch"

llm:
  provider: "openai"            # "openai" | "anthropic" | "ollama" | "litellm"

attention_gate:
  threshold: 0.2
  weights:
    correction: 0.30            # 最强信号
    novelty: 0.25
    explicit_mark: 0.20

retrieval:
  rerank_enabled: false         # cross-encoder 重排序
  rerank_top_k: 50

decay:
  episodic_rate: 0.1            # 快：对话细节
  semantic_rate: 0.03           # 中：提取的知识
  procedural_rate: 0.005        # 慢：学到的习惯

auth:
  enabled: false
  api_keys: {}                  # key → namespace 映射
```

## 设计原则

1. **不是所有事都值得记住** — 注意力门控过滤噪声
2. **快写慢炼** — 即时编码 + 异步巩固
3. **积累而非覆盖** — 原始事实不可变；观察可演化
4. **用进废退** — 检索增强记忆；不用则衰退
5. **遗忘是特性** — 主动修剪防止检索退化
6. **部分线索，完整回忆** — 多探针搜索最大化召回
7. **一起激活的神经元连在一起** — 赫布图谱增强
8. **安全更新，绝不删除原始数据** — 再巩固保留完整历史

## 项目结构

```
cortiloop/
├── encoding/          # 注意力门控 + LLM 编码器
├── consolidation/     # 突触巩固（即时）+ 系统巩固（深度）
├── retrieval/         # 多探针检索 + RRF + 重排序
├── association/       # 赫布知识图谱
├── forgetting/        # 艾宾浩斯衰减 + 修剪器
├── reconsolidation/   # 冲突检测 + 安全更新
├── storage/           # BaseStore ABC + SQLite + PostgreSQL
├── llm/               # LLM 抽象层（OpenAI/Anthropic/Ollama/litellm）
├── workers/           # 后台巩固 worker
├── adapters/          # MCP 服务器 + nanobot 插件 + openclaw 技能
├── viz/               # Web 可视化面板
└── auth.py            # 多租户认证
benchmarks/
└── longmemeval.py     # LongMemEval 基准测试
```

## 开发

```bash
git clone https://github.com/shenchengtsi/CortiLoop.git
cd CortiLoop
pip install -e ".[dev]"
pytest  # 48 个测试
```

## 许可证

MIT
