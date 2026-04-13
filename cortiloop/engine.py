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
from datetime import datetime
from typing import Any

from cortiloop.association.graph import AssociationGraph
from cortiloop.config import CortiLoopConfig
from cortiloop.consolidation.synaptic import SynapticConsolidator
from cortiloop.consolidation.systems import SystemsConsolidator
from cortiloop.encoding.attention_gate import AttentionGate
from cortiloop.encoding.encoder import Encoder
from cortiloop.forgetting.decay import DecayManager
from cortiloop.forgetting.pruner import Pruner
from cortiloop.llm.protocol import Embedder, MemoryLLM, Reranker
from cortiloop.models import MemoryUnit, Observation
from cortiloop.reconsolidation.updater import Reconsolidator
from cortiloop.retrieval.multi_probe import MultiProbeRetriever
from cortiloop.storage.base_store import BaseStore
from cortiloop.workers.consolidation_worker import ConsolidationWorker

logger = logging.getLogger("cortiloop")


class _FallbackEmbedder:
    """Tries primary embedder first; on failure, lazily creates and uses fallback.

    Solves the problem where LLMClient implements the Embedder protocol but the
    actual API provider doesn't support embedding (e.g. doubao coding endpoint).
    """

    def __init__(self, primary: Embedder, fallback_factory):
        self._primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: Embedder | None = None
        self._use_fallback = False

    def _get_fallback(self) -> Embedder:
        if self._fallback is None:
            self._fallback = self._fallback_factory()
            logger.info("Switched to fallback embedder: %s", type(self._fallback).__name__)
        return self._fallback

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._use_fallback:
            return await self._get_fallback().embed(texts)
        try:
            return await self._primary.embed(texts)
        except Exception as e:
            logger.warning("LLM embedding failed (%s), switching to fallback", e)
            self._use_fallback = True
            return await self._get_fallback().embed(texts)

    async def embed_one(self, text: str) -> list[float]:
        if self._use_fallback:
            return await self._get_fallback().embed_one(text)
        try:
            return await self._primary.embed_one(text)
        except Exception as e:
            logger.warning("LLM embedding failed (%s), switching to fallback", e)
            self._use_fallback = True
            return await self._get_fallback().embed_one(text)


class CortiLoop:
    """
    Bioinspired Agent Memory Engine.

    Usage with Agent's LLM (recommended — only chat completion needed):
        loop = CortiLoop(llm=agent.llm)
        await loop.retain("User said something important")
        results = await loop.recall("What did the user say?")

    Optionally provide a dedicated embedder:
        loop = CortiLoop(llm=agent.llm, embedder=my_embedder)

    Usage standalone (requires LLM config):
        loop = CortiLoop(config=CortiLoopConfig(...))
    """

    def __init__(
        self,
        config: CortiLoopConfig | None = None,
        llm: MemoryLLM | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.config = config or CortiLoopConfig()

        # Chat LLM — use provided or create from config
        if llm is not None:
            self.llm: MemoryLLM = llm
        else:
            from cortiloop.llm.client import LLMClient
            self.llm = LLMClient(self.config.llm)

        # Embedder — use provided, or detect from llm, or auto-select best available
        if embedder is not None:
            self.embedder: Embedder = embedder
        elif isinstance(self.llm, Embedder):
            # LLM implements Embedder protocol — wrap with fallback in case provider
            # doesn't actually support embedding API (e.g. doubao coding endpoint)
            self.embedder = _FallbackEmbedder(self.llm, self._create_default_embedder)  # type: ignore
        else:
            self.embedder = self._create_default_embedder()

        # Reranker — use provided, or detect from llm, or auto-select best available
        if reranker is not None:
            self.reranker: Reranker = reranker
        elif isinstance(self.llm, Reranker):
            self.reranker = self.llm
        else:
            self.reranker = self._create_default_reranker()

        self.store: BaseStore = self._create_store()

        # 7 Bioinspired Layers
        self.attention_gate = AttentionGate(self.config.attention_gate, self.llm)
        self.encoder = Encoder(self.config, self.llm, self.embedder)
        self.graph = AssociationGraph(self.store)
        self.retriever = MultiProbeRetriever(
            self.config.retrieval, self.store, self.embedder, self.reranker, self.graph,
        )
        self.synaptic = SynapticConsolidator(
            self.config.consolidation, self.store, self.llm, self.embedder,
        )
        self.systems = SystemsConsolidator(
            self.config.consolidation, self.store, self.llm, self.embedder,
        )
        self.decay = DecayManager(self.config.decay, self.store)
        self.pruner = Pruner(self.config.forgetting, self.store)
        self.reconsolidator = Reconsolidator(self.store, self.llm, self.embedder)

        # Background worker
        self.worker = ConsolidationWorker(
            self,
            interval_seconds=self.config.consolidation.systems_interval_seconds,
        )

    def _create_default_embedder(self) -> Embedder:
        """Auto-select best available embedder: sentence-transformers > hash fallback."""
        try:
            import sentence_transformers  # noqa: F401 — availability check
            from cortiloop.llm.local_embedder import LocalEmbedder
            logger.info("Using local sentence-transformers embedder")
            return LocalEmbedder()
        except ImportError:
            logger.info("sentence-transformers not available, using hash-based embedding")
            from cortiloop.llm.builtin_embedder import BuiltinEmbedder
            return BuiltinEmbedder(dim=self.config.llm.embedding_dim)

    def _create_default_reranker(self) -> Reranker:
        """Auto-select best available reranker: cross-encoder > word-overlap fallback."""
        try:
            import sentence_transformers  # noqa: F401 — availability check
            from cortiloop.llm.local_embedder import LocalReranker
            logger.info("Using local sentence-transformers reranker")
            return LocalReranker()
        except ImportError:
            logger.info("sentence-transformers not available, using word-overlap reranking")
            from cortiloop.llm.builtin_embedder import BuiltinReranker
            return BuiltinReranker()

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
        session_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Encode input into memory. Full pipeline:
        1. Attention gate (importance scoring)
        2. Encoding (fact extraction + entity resolution)
        3. Storage (write to hippocampal layer)
        4. Association (build graph edges)
        5. Synaptic consolidation (async: units → observations)
        6. Reconsolidation (conflict detection + safe update)

        Args:
            session_timestamp: When the conversation originally happened
                (as opposed to created_at which is always "now").

        Returns: {"stored": int, "importance": float, "skipped": bool}
        """
        # Step 1: Attention gate
        entity_count = self.store.count_units()
        try:
            importance = await self.attention_gate.score(
                text, entity_count, task_context,
            )
        except Exception as e:
            logger.warning("Attention gate failed, using default importance: %s", e)
            importance = 0.5

        if not self.attention_gate.passes(importance):
            logger.debug("Attention gate filtered: importance=%.2f < threshold", importance)
            return {"stored": 0, "importance": importance, "skipped": True}

        # Step 2: Encoding
        try:
            units = await self.encoder.encode(
                text, importance, session_id, task_context,
            )
        except Exception as e:
            logger.warning("Encoding failed: %s", e)
            return {"stored": 0, "importance": importance, "skipped": True, "error": str(e)}
        if not units:
            return {"stored": 0, "importance": importance, "skipped": True}

        # Stamp session_timestamp on all units so recall can reason about recency
        if session_timestamp:
            for unit in units:
                unit.session_timestamp = session_timestamp

        # Step 3: Storage
        try:
            for unit in units:
                self.store.insert_unit(unit)
        except Exception as e:
            logger.warning("Storage failed: %s", e)
            return {"stored": 0, "importance": importance, "skipped": True, "error": str(e)}

        # Step 4: Association (Hebbian linking)
        self.graph.link_co_occurring(units)

        # Step 5: Synaptic consolidation (async)
        try:
            await self.synaptic.consolidate(units)
        except Exception as e:
            logger.warning("Synaptic consolidation failed: %s", e)

        # Step 6: Reconsolidation check — scan all units for conflicts
        try:
            seen_obs: set[str] = set()
            related_obs: list = []
            for unit in units:
                if not unit.embedding:
                    continue
                for obs, sim in self.store.search_observations_by_vector(unit.embedding, 5):
                    if obs.id not in seen_obs and sim > 0.6:
                        seen_obs.add(obs.id)
                        related_obs.append(obs)
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
        try:
            return await self.retriever.recall(query, top_k)
        except Exception as e:
            logger.warning("Recall failed: %s", e)
            return []

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
