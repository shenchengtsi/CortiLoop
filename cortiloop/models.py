"""Core data models for CortiLoop memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    USER_SAID = "user_said"
    LLM_INFERRED = "llm_inferred"
    SYSTEM = "system"


class MemoryState(str, Enum):
    ACTIVE = "active"
    ARCHIVE = "archive"
    COLD = "cold"


class EdgeType(str, Enum):
    CO_OCCURRENCE = "co_occurrence"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    SEMANTIC = "semantic"


class MemoryTier(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class EncodingContext:
    """Context captured at encoding time (for encoding-specificity matching)."""
    task: str = ""
    entities: list[str] = field(default_factory=list)
    mood: str = ""
    session_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryUnit:
    """
    Hippocampal layer: raw memory unit. Immutable after creation.
    Corresponds to a single fact extracted from agent interaction.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source_type: SourceType = SourceType.USER_SAID
    importance_score: float = 0.5
    encoding_context: EncodingContext = field(default_factory=EncodingContext)
    entities: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Strength & decay (Ebbinghaus)
    base_strength: float = 1.0
    decay_rate: float = 0.1  # episodic decays faster
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    state: MemoryState = MemoryState.ACTIVE
    tier: MemoryTier = MemoryTier.EPISODIC


@dataclass
class Observation:
    """
    Cortical layer: abstracted knowledge from consolidation.
    Can be updated but preserves change history.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dimension: str = ""  # single-dimension constraint
    content: str = ""
    confidence: float = 1.0
    version: int = 1
    source_unit_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Strength & decay (semantic decays slower)
    base_strength: float = 1.0
    decay_rate: float = 0.03
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    state: MemoryState = MemoryState.ACTIVE
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProceduralMemory:
    """
    Basal ganglia layer: learned patterns, workflows, preferences.
    Acquired through repetition, decays extremely slowly.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern: str = ""  # trigger pattern description
    procedure: str = ""  # what to do
    entities: list[str] = field(default_factory=list)
    acquisition_count: int = 1
    confidence: float = 0.3  # increases with repetition
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    # Procedural memories decay extremely slowly
    base_strength: float = 1.0
    decay_rate: float = 0.005
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    state: MemoryState = MemoryState.ACTIVE


@dataclass
class MemoryEdge:
    """
    Association layer edge. Weight updated via Hebbian rule.
    """
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.CO_OCCURRENCE
    weight: float = 1.0
    co_activation_count: int = 1
    last_co_activated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkingMemorySlot:
    """A slot in working memory buffer — loaded from long-term store."""
    memory_id: str = ""
    memory_type: str = ""  # "unit", "observation", "procedural"
    content: str = ""
    relevance_score: float = 0.0
    loaded_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictRecord:
    """Tracks conflicts detected during reconsolidation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    old_memory_id: str = ""
    new_memory_id: str = ""
    dimension: str = ""
    old_value: str = ""
    new_value: str = ""
    resolution: str = ""  # "superseded", "coexist", "pending"
    created_at: datetime = field(default_factory=datetime.now)
