"""
Abstract base class for CortiLoop storage backends.

All storage implementations (SQLite, PostgreSQL, etc.) must implement this interface.
This enables swapping backends without changing any business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cortiloop.models import (
    ConflictRecord,
    EdgeType,
    MemoryEdge,
    MemoryState,
    MemoryUnit,
    Observation,
    ProceduralMemory,
)


class BaseStore(ABC):
    """Abstract storage backend for CortiLoop memory engine."""

    # ── MemoryUnit CRUD ──

    @abstractmethod
    def insert_unit(self, unit: MemoryUnit) -> None: ...

    @abstractmethod
    def get_unit(self, unit_id: str) -> MemoryUnit | None: ...

    @abstractmethod
    def get_active_units(self, limit: int = 1000) -> list[MemoryUnit]: ...

    @abstractmethod
    def get_recent_units(self, limit: int = 50) -> list[MemoryUnit]: ...

    @abstractmethod
    def search_units_by_vector(self, query_emb: list[float], top_k: int = 20) -> list[tuple[MemoryUnit, float]]: ...

    @abstractmethod
    def search_units_by_keyword(self, keyword: str, limit: int = 20) -> list[MemoryUnit]: ...

    @abstractmethod
    def search_units_by_entity(self, entity: str, limit: int = 50) -> list[MemoryUnit]: ...

    @abstractmethod
    def update_unit_access(self, unit_id: str) -> None: ...

    @abstractmethod
    def update_unit_state(self, unit_id: str, state: MemoryState) -> None: ...

    @abstractmethod
    def count_units(self) -> int: ...

    # ── Observation CRUD ──

    @abstractmethod
    def insert_observation(self, obs: Observation) -> None: ...

    @abstractmethod
    def get_observation(self, obs_id: str) -> Observation | None: ...

    @abstractmethod
    def get_active_observations(self, limit: int = 500) -> list[Observation]: ...

    @abstractmethod
    def search_observations_by_vector(self, query_emb: list[float], top_k: int = 20) -> list[tuple[Observation, float]]: ...

    @abstractmethod
    def search_observations_by_dimension(self, dimension: str) -> list[Observation]: ...

    @abstractmethod
    def update_observation_access(self, obs_id: str) -> None: ...

    @abstractmethod
    def count_observations(self) -> int: ...

    # ── ProceduralMemory CRUD ──

    @abstractmethod
    def insert_procedural(self, pm: ProceduralMemory) -> None: ...

    @abstractmethod
    def get_active_procedurals(self, limit: int = 100) -> list[ProceduralMemory]: ...

    @abstractmethod
    def search_procedurals_by_vector(self, query_emb: list[float], top_k: int = 5) -> list[tuple[ProceduralMemory, float]]: ...

    # ── Edge CRUD ──

    @abstractmethod
    def upsert_edge(self, edge: MemoryEdge) -> None: ...

    @abstractmethod
    def get_edges_from(self, source_id: str) -> list[MemoryEdge]: ...

    @abstractmethod
    def get_edges_to(self, target_id: str) -> list[MemoryEdge]: ...

    @abstractmethod
    def get_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> MemoryEdge | None: ...

    # ── Conflict CRUD ──

    @abstractmethod
    def insert_conflict(self, conflict: ConflictRecord) -> None: ...

    # ── Bulk / Maintenance ──

    @abstractmethod
    def get_all_active_units_for_decay(self) -> list[tuple[str, float, float, Any, int]]: ...

    @abstractmethod
    def get_all_active_observations_for_decay(self) -> list[tuple[str, float, float, Any, int]]: ...

    # ── Lifecycle ──

    @abstractmethod
    def close(self) -> None: ...
