"""
Decay Manager — Ebbinghaus-inspired memory strength computation.

Brain analogy:
- Forgetting curve: R = e^(-t/S)
- Spaced repetition: each retrieval resets decay and increases base strength
- Differential decay rates: episodic > semantic > procedural
"""

from __future__ import annotations

import math
from datetime import datetime

from cortiloop.config import DecayConfig
from cortiloop.models import MemoryState
from cortiloop.storage.base_store import BaseStore


class DecayManager:
    """Computes memory strength and manages state transitions based on decay."""

    def __init__(self, config: DecayConfig, store: BaseStore):
        self.config = config
        self.store = store

    @staticmethod
    def compute_strength(
        base_strength: float,
        decay_rate: float,
        last_accessed: datetime,
        access_count: int,
        now: datetime | None = None,
    ) -> float:
        """
        Compute current memory strength using Ebbinghaus-inspired formula.

        strength = base_strength * e^(-decay_rate * elapsed_days) * spaced_boost
        where spaced_boost = 1 + access_boost * log(1 + access_count)
        """
        now = now or datetime.now()
        elapsed_seconds = (now - last_accessed).total_seconds()
        elapsed_days = max(elapsed_seconds / 86400, 0)

        decay = math.exp(-decay_rate * elapsed_days)
        spaced_boost = 1.0 + 0.3 * math.log1p(access_count)

        return base_strength * decay * spaced_boost

    def evaluate_state(self, strength: float) -> MemoryState:
        """Determine memory state based on current strength."""
        if strength >= self.config.archive_threshold:
            return MemoryState.ACTIVE
        elif strength >= self.config.cold_threshold:
            return MemoryState.ARCHIVE
        else:
            return MemoryState.COLD

    def run_decay_sweep(self):
        """
        Sweep all active memories, update states based on current strength.
        Brain analogy: gradual synaptic weakening over time without reactivation.
        """
        now = datetime.now()

        # Sweep memory units
        for uid, decay_rate, base_strength, last_accessed, access_count in self.store.get_all_active_units_for_decay():
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            strength = self.compute_strength(base_strength, decay_rate, last_accessed, access_count, now)
            new_state = self.evaluate_state(strength)
            if new_state != MemoryState.ACTIVE:
                self.store.update_unit_state(uid, new_state)

        # Sweep observations
        for oid, decay_rate, base_strength, last_accessed, access_count in self.store.get_all_active_observations_for_decay():
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            strength = self.compute_strength(base_strength, decay_rate, last_accessed, access_count, now)
            new_state = self.evaluate_state(strength)
            if new_state != MemoryState.ACTIVE:
                # For observations, just archive (don't cold-store)
                if new_state == MemoryState.COLD:
                    new_state = MemoryState.ARCHIVE
                self.store.conn.execute(
                    f"UPDATE observations_{self.store.config.namespace} SET state=? WHERE id=?",
                    (new_state.value, oid),
                )
        self.store.conn.commit()
