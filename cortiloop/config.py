"""Configuration for CortiLoop memory engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai" | "anthropic" | "ollama" | "litellm"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    api_key: str = ""  # read from env if empty
    base_url: str = ""  # custom endpoint (e.g. Ollama: http://localhost:11434/v1)
    rerank_model: str = ""  # optional dedicated reranker (e.g. "rerank-english-v3.0")
    headers: dict[str, str] = field(default_factory=dict)  # custom HTTP headers (e.g. User-Agent override)


@dataclass
class AttentionGateConfig:
    """Thresholds for the attention gate (what's worth remembering)."""
    enabled: bool = True
    threshold: float = 0.2  # below this → skip
    weights: dict[str, float] = field(default_factory=lambda: {
        "novelty": 0.25,
        "correction": 0.30,
        "explicit_mark": 0.20,
        "emotional_intensity": 0.10,
        "task_relevance": 0.15,
    })


@dataclass
class DecayConfig:
    """Ebbinghaus-inspired decay parameters."""
    episodic_rate: float = 0.1    # fast decay
    semantic_rate: float = 0.03   # moderate decay
    procedural_rate: float = 0.005  # very slow decay
    access_boost: float = 0.3     # log(1 + access_count) multiplier
    archive_threshold: float = 0.3
    cold_threshold: float = 0.1


@dataclass
class ConsolidationConfig:
    """Background consolidation worker settings."""
    synaptic_enabled: bool = True
    systems_enabled: bool = True
    systems_interval_seconds: int = 3600  # deep consolidation every hour
    max_units_per_batch: int = 50
    observation_single_dimension: bool = True


@dataclass
class RetrievalConfig:
    """Multi-probe retrieval settings."""
    semantic_weight: float = 0.4
    keyword_weight: float = 0.2
    graph_weight: float = 0.25
    temporal_weight: float = 0.15
    max_results: int = 20
    spreading_activation_hops: int = 2
    spreading_decay_factor: float = 0.5
    rerank_enabled: bool = False  # enable cross-encoder reranking
    rerank_top_k: int = 50  # how many candidates to rerank


@dataclass
class ForgettingConfig:
    """Active forgetting / pruning settings."""
    enabled: bool = True
    prune_interval_seconds: int = 7200  # every 2 hours
    merge_similarity_threshold: float = 0.92
    max_memory_units: int = 100000
    max_observations: int = 10000


@dataclass
class AuthConfig:
    """Multi-tenant authentication settings."""
    enabled: bool = False
    api_keys: dict[str, str] = field(default_factory=dict)  # key → namespace mapping
    admin_key: str = ""  # admin key for cross-namespace operations


@dataclass
class CortiLoopConfig:
    """Top-level configuration for the CortiLoop engine."""
    db_path: str = "cortiloop.db"
    namespace: str = "default"  # multi-tenant isolation
    storage_backend: str = "sqlite"  # "sqlite" | "postgres"
    vector_backend: str = "auto"  # "auto" | "numpy" | "usearch" (sqlite only; postgres uses pgvector)

    llm: LLMConfig = field(default_factory=LLMConfig)
    attention_gate: AttentionGateConfig = field(default_factory=AttentionGateConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    @classmethod
    def from_dict(cls, d: dict) -> CortiLoopConfig:
        cfg = cls()
        for section_name in ("llm", "attention_gate", "decay", "consolidation", "retrieval", "forgetting", "auth"):
            if section_name in d:
                section_cls = type(getattr(cfg, section_name))
                setattr(cfg, section_name, section_cls(**d[section_name]))
        for key in ("db_path", "namespace", "storage_backend", "vector_backend"):
            if key in d:
                setattr(cfg, key, d[key])
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> CortiLoopConfig:
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})
