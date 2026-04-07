"""
Pruner — active forgetting via conflict resolution, deduplication, and compression.

Brain analogy:
- Microglia-mediated synaptic pruning (complement-dependent elimination)
- Neurogenesis-induced forgetting (new neurons disrupt old patterns)
- Capacity-driven compression (information overload → force consolidation)
"""

from __future__ import annotations

import numpy as np
from datetime import datetime

from cortiloop.config import ForgettingConfig
from cortiloop.models import MemoryState
from cortiloop.storage.base_store import BaseStore


class Pruner:
    """
    Active forgetting: deduplication, conflict resolution, capacity management.
    Runs periodically as a background worker.
    """

    def __init__(self, config: ForgettingConfig, store: BaseStore):
        self.config = config
        self.store = store

    def run_pruning_cycle(self):
        """Execute a full pruning cycle."""
        if not self.config.enabled:
            return

        self._deduplicate_units()
        self._capacity_check()

    def _deduplicate_units(self):
        """
        Merge highly similar memory units.
        Brain analogy: microglia pruning redundant synapses.
        """
        units = self.store.get_active_units(limit=2000)
        if len(units) < 2:
            return

        # Build embedding matrix for active units with embeddings
        indexed_units = [(u, u.embedding) for u in units if u.embedding]
        if len(indexed_units) < 2:
            return

        embeddings = np.array([e for _, e in indexed_units], dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Cosine similarity matrix
        sim_matrix = normalized @ normalized.T

        # Find duplicate pairs above threshold
        to_archive = set()
        for i in range(len(indexed_units)):
            if indexed_units[i][0].id in to_archive:
                continue
            for j in range(i + 1, len(indexed_units)):
                if indexed_units[j][0].id in to_archive:
                    continue
                if sim_matrix[i, j] >= self.config.merge_similarity_threshold:
                    # Keep the one with higher access count / more recent
                    unit_i, unit_j = indexed_units[i][0], indexed_units[j][0]
                    victim = unit_j if unit_i.access_count >= unit_j.access_count else unit_i
                    to_archive.add(victim.id)

        # Archive duplicates
        for uid in to_archive:
            self.store.update_unit_state(uid, MemoryState.ARCHIVE)

    def _capacity_check(self):
        """
        When memory exceeds capacity, force archival of weakest memories.
        Brain analogy: neurogenesis-induced forgetting — new memories
        naturally push out old patterns.
        """
        count = self.store.count_units()
        if count <= self.config.max_memory_units:
            return

        excess = count - self.config.max_memory_units
        # Archive the oldest, least-accessed active units
        rows = self.store.conn.execute(
            f"""SELECT id FROM memory_units_{self.store.config.namespace}
                WHERE state='active'
                ORDER BY access_count ASC, last_accessed ASC
                LIMIT ?""",
            (excess,),
        ).fetchall()

        for (uid,) in rows:
            self.store.update_unit_state(uid, MemoryState.ARCHIVE)
