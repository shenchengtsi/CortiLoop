"""
CortiLoop — Main engine facade.

Wires together all 7 bioinspired layers into a simple 3-method API:
  retain(text)   — encode + store + consolidate
  recall(query)  — retrieve + associate
  reflect()      — deep consolidation + maintenance

This maps to the Retain/Recall/Reflect pattern while adding
attention gating, forgetting, and reconsolidation under the hood.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cortiloop.association.graph import AssociationGraph
from cortiloop.config import CortiLoopConfig
from cortiloop.consolidation.synaptic import SynapticConsolidator
from cortiloop.consolidation.systems import SystemsConsolidator
from cortiloop.encoding.attention_gate import AttentionGate
from cortiloop.encoding.encoder import Encoder
from cortiloop.forgetting.decay import DecayManager
from cortiloop.forgetting.pruner import Pruner
from cortiloop.llm.client import LLMClient
from cortiloop.models import MemoryUnit, Observation
from cortiloop.reconsolidation.updater import Reconsolidator
from cortiloop.retrieval.multi_probe import MultiProbeRetriever
from cortiloop.storage.base_store import BaseStore
from cortiloop.workers.consolidation_worker import ConsolidationWorker

logger = logging.getLogger("cortiloop")


class CortiLoop:
    """
    Bioinspired Agent Memory Engine.

    Usage:
        loop = CortiLoop(config)
        await loop.retain("User said something important")
        results = await loop.recall("What did the user say?")
        await loop.reflect()  # deep consolidation
    """

    def __init__(self, config: CortiLoopConfig | None = None):
        self.config = config or CortiLoopConfig()

        # Infrastructure
        self.llm = LLMClient(self.config.llm)
        self.store: BaseStore = self._create_store()

        # 7 Bioinspired Layers
        self.attention_gate = AttentionGate(self.config.attention_gate, self.llm)
        self.encoder = Encoder(self.config, self.llm)
        self.graph = AssociationGraph(self.store)
        self.retriever = MultiProbeRetriever(
            self.config.retrieval, self.store, self.llm, self.graph,
        )
        self.synaptic = SynapticConsolidator(
            self.config.consolidation, self.store, self.llm,
        )
        self.systems = SystemsConsolidator(
            self.config.consolidation, self.store, self.llm,
        )
        self.decay = DecayManager(self.config.decay, self.store)
        self.pruner = Pruner(self.config.forgetting, self.store)
        self.reconsolidator = Reconsolidator(self.store, self.llm)

        # Background worker
        self.worker = ConsolidationWorker(
            self,
            interval_seconds=self.config.consolidation.systems_interval_seconds,
        )

    def _create_store(self) -> BaseStore:
        """Factory: create storage backend based on config."""
        backend = self.config.storage_backend
        if backend == "postgres":
            from cortiloop.storage.postgres_store import PostgresStore
            logger.info("Using PostgreSQL storage backend")
            return PostgresStore(self.config)
        else:
            from cortiloop.storage.sqlite_store import SQLiteStore
            logger.info("Using SQLite storage backend")
            return SQLiteStore(self.config)

    # ── Public API ──

    async def retain(
        self,
        text: str,
        session_id: str = "",
        task_context: str = "",
        source_type: str = "user_said",
    ) -> dict[str, Any]:
        """
        Encode input into memory. Full pipeline:
        1. Attention gate (importance scoring)
        2. Encoding (fact extraction + entity resolution)
        3. Storage (write to hippocampal layer)
        4. Association (build graph edges)
        5. Synaptic consolidation (async: units → observations)
        6. Reconsolidation (conflict detection + safe update)

        Returns: {"stored": int, "importance": float, "skipped": bool}
        """
        # Step 1: Attention gate
        entity_count = self.store.count_units()
        importance = await self.attention_gate.score(
            text, entity_count, task_context,
        )

        if not self.attention_gate.passes(importance):
            logger.debug("Attention gate filtered: importance=%.2f < threshold", importance)
            return {"stored": 0, "importance": importance, "skipped": True}

        # Step 2: Encoding
        units = await self.encoder.encode(
            text, importance, session_id, task_context,
        )
        if not units:
            return {"stored": 0, "importance": importance, "skipped": True}

        # Step 3: Storage
        for unit in units:
            self.store.insert_unit(unit)

        # Step 4: Association (Hebbian linking)
        self.graph.link_co_occurring(units)

        # Step 5: Synaptic consolidation (async)
        try:
            await self.synaptic.consolidate(units)
        except Exception as e:
            logger.warning("Synaptic consolidation failed: %s", e)

        # Step 6: Reconsolidation check
        try:
            if units[0].embedding:
                related_obs = [
                    obs for obs, sim
                    in self.store.search_observations_by_vector(units[0].embedding, 5)
                    if sim > 0.6
                ]
                if related_obs:
                    conflicts = await self.reconsolidator.check_and_update(units, related_obs)
                    if conflicts:
                        logger.info("Reconsolidation: %d conflicts detected", len(conflicts))
        except Exception as e:
            logger.warning("Reconsolidation check failed: %s", e)

        return {
            "stored": len(units),
            "importance": importance,
            "skipped": False,
            "unit_ids": [u.id for u in units],
            "entities": list({e for u in units for e in u.entities}),
        }

    async def recall(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories relevant to query. Full pipeline:
        1. Multi-probe retrieval (semantic + keyword + graph + temporal)
        2. RRF fusion + ranking
        3. Testing effect (retrieval strengthens accessed memories)
        4. Hebbian co-retrieval strengthening

        Returns: list of {"id", "type", "content", "score", "entities"}
        """
        return await self.retriever.recall(query, top_k)

    async def reflect(self) -> dict[str, Any]:
        """
        Deep consolidation cycle ("sleep mode"). Runs:
        1. Systems consolidation (procedural detection + mental models)
        2. Decay sweep (Ebbinghaus curve state transitions)
        3. Pruning (dedup + capacity management)

        Call periodically or during low-activity periods.
        Returns: {"decayed", "pruned", "models_generated"}
        """
        results: dict[str, Any] = {}

        # Systems consolidation
        try:
            await self.systems.run_deep_consolidation()
            results["consolidation"] = "ok"
        except Exception as e:
            logger.warning("Systems consolidation failed: %s", e)
            results["consolidation"] = f"error: {e}"

        # Decay sweep
        try:
            self.decay.run_decay_sweep()
            results["decay_sweep"] = "ok"
        except Exception as e:
            logger.warning("Decay sweep failed: %s", e)
            results["decay_sweep"] = f"error: {e}"

        # Pruning
        try:
            self.pruner.run_pruning_cycle()
            results["pruning"] = "ok"
        except Exception as e:
            logger.warning("Pruning failed: %s", e)
            results["pruning"] = f"error: {e}"

        return results

    # ── Background Worker ──

    def start_worker(self) -> None:
        """Start background consolidation worker (periodic reflect cycles)."""
        self.worker.start()

    async def stop_worker(self) -> None:
        """Stop background consolidation worker."""
        await self.worker.stop()

    # ── Utility ──

    async def stats(self) -> dict[str, int]:
        """Get memory system statistics."""
        return {
            "memory_units": self.store.count_units(),
            "observations": self.store.count_observations(),
            "namespace": self.config.namespace,
            "worker_running": self.worker.is_running,
        }

    async def aclose(self):
        """Async cleanup — stops worker and closes store."""
        await self.worker.stop()
        self.store.close()

    def close(self):
        """Sync cleanup — closes store (use aclose() if worker is running)."""
        self.store.close()
