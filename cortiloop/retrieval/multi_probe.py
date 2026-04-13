"""
Multi-Probe Retriever — pattern completion via multi-route fusion.

Brain analogy:
- CA3 pattern completion (partial cue → full memory)
- 4-route parallel retrieval: semantic, keyword, graph, temporal
- Reciprocal Rank Fusion for merging results
- Testing effect: retrieval strengthens accessed memories
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from cortiloop.association.graph import AssociationGraph
from cortiloop.config import RetrievalConfig
from cortiloop.llm.protocol import Embedder, Reranker
from cortiloop.models import MemoryUnit, Observation, ProceduralMemory
from cortiloop.storage.base_store import BaseStore

logger = logging.getLogger("cortiloop.retrieval")


class MultiProbeRetriever:
    """
    Multi-route retrieval with Reciprocal Rank Fusion + optional cross-encoder reranking.
    Mimics hippocampal pattern completion via diverse signal fusion.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        store: BaseStore,
        embedder: Embedder,
        reranker: Reranker,
        graph: AssociationGraph,
    ):
        self.config = config
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.graph = graph

    async def recall(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Main recall entry point. Returns ranked list of memories.
        Each result: {"id", "type", "content", "score", "entities"}
        """
        top_k = top_k or self.config.max_results
        query_emb = await self.embedder.embed_one(query)

        # Route 1: Semantic search (vector ANN)
        semantic_results = self._semantic_search(query_emb, top_k * 2)

        # Route 2: Keyword search (BM25-like)
        keyword_results = self._keyword_search(query, top_k * 2)

        # Route 3: Graph traversal (spreading activation)
        graph_results = self._graph_search(semantic_results, top_k * 2)

        # Route 4: Temporal search
        temporal_results = self._temporal_search(query, top_k)

        # Reciprocal Rank Fusion
        fused = self._rrf_fuse(
            [
                (semantic_results, self.config.semantic_weight),
                (keyword_results, self.config.keyword_weight),
                (graph_results, self.config.graph_weight),
                (temporal_results, self.config.temporal_weight),
            ],
            k=60,
        )

        # Cross-encoder reranking (optional)
        if self.config.rerank_enabled and len(fused) > 0:
            rerank_candidates = fused[:self.config.rerank_top_k]
            try:
                reranked = await self._rerank(query, rerank_candidates)
                results = reranked[:top_k]
            except Exception as e:
                logger.warning("Reranking failed, using RRF order: %s", e)
                results = fused[:top_k]
        else:
            results = fused[:top_k]

        # Testing effect: strengthen accessed memories
        accessed_ids = [r["id"] for r in results]
        for rid in accessed_ids:
            self.store.update_unit_access(rid)
        self.graph.strengthen_on_retrieval(accessed_ids)

        return results

    async def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-encoder reranking via LLM."""
        documents = [c["content"] for c in candidates]
        scored = await self.reranker.rerank(query, documents, top_k=len(candidates))

        reranked = []
        for orig_idx, score in scored:
            if orig_idx < len(candidates):
                entry = candidates[orig_idx].copy()
                entry["score"] = score
                reranked.append(entry)
        return reranked

    def _semantic_search(self, query_emb: list[float], limit: int) -> list[dict]:
        results = []

        # Search memory units
        for unit, sim in self.store.search_units_by_vector(query_emb, limit):
            entry = {
                "id": unit.id, "type": "unit", "content": unit.content,
                "score": sim, "entities": unit.entities,
            }
            if unit.session_timestamp:
                entry["session_timestamp"] = unit.session_timestamp.strftime("%Y-%m-%d")
            results.append(entry)

        # Search observations
        for obs, sim in self.store.search_observations_by_vector(query_emb, limit):
            results.append({
                "id": obs.id, "type": "observation", "content": obs.content,
                "score": sim, "entities": obs.entities,
            })

        # Search procedural memories
        for pm, sim in self.store.search_procedurals_by_vector(query_emb, limit // 4):
            results.append({
                "id": pm.id, "type": "procedural",
                "content": f"[Pattern: {pm.pattern}] {pm.procedure}",
                "score": sim, "entities": pm.entities,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _keyword_search(self, query: str, limit: int) -> list[dict]:
        # Extract keywords (simple: split on whitespace and punctuation)
        keywords = [w for w in re.split(r'\W+', query) if len(w) > 1]
        results = []
        seen = set()

        for kw in keywords[:5]:  # limit keyword count
            for unit in self.store.search_units_by_keyword(kw, limit):
                if unit.id not in seen:
                    seen.add(unit.id)
                    entry = {
                        "id": unit.id, "type": "unit", "content": unit.content,
                        "score": 1.0, "entities": unit.entities,
                    }
                    if unit.session_timestamp:
                        entry["session_timestamp"] = unit.session_timestamp.strftime("%Y-%m-%d")
                    results.append(entry)

        return results[:limit]

    def _graph_search(self, seed_results: list[dict], limit: int) -> list[dict]:
        """Use top semantic results as seeds for spreading activation."""
        seed_ids = [r["id"] for r in seed_results[:5]]
        if not seed_ids:
            return []

        activations = self.graph.spreading_activation(
            seed_ids,
            max_hops=self.config.spreading_activation_hops,
            decay_factor=self.config.spreading_decay_factor,
        )

        # Convert activated node IDs back to memory content
        results = []
        for node_id, activation in sorted(activations.items(), key=lambda x: x[1], reverse=True):
            if node_id in {r["id"] for r in seed_results}:
                continue  # skip seeds (already in results)
            unit = self.store.get_unit(node_id)
            if unit:
                entry = {
                    "id": unit.id, "type": "unit", "content": unit.content,
                    "score": activation, "entities": unit.entities,
                }
                if unit.session_timestamp:
                    entry["session_timestamp"] = unit.session_timestamp.strftime("%Y-%m-%d")
                results.append(entry)
            if len(results) >= limit:
                break

        return results

    def _temporal_search(self, query: str, limit: int) -> list[dict]:
        """Extract temporal intent and filter by time range."""
        time_range = self._extract_time_range(query)
        if not time_range:
            return []

        start, end = time_range
        results = []
        for unit in self.store.get_active_units(limit * 5):
            if start <= unit.created_at <= end:
                results.append({
                    "id": unit.id, "type": "unit", "content": unit.content,
                    "score": 1.0, "entities": unit.entities,
                })
        return results[:limit]

    @staticmethod
    def _extract_time_range(query: str) -> tuple[datetime, datetime] | None:
        """Simple time range extraction from query text."""
        now = datetime.now()
        lower = query.lower()

        patterns = {
            r"今天|today": (now.replace(hour=0, minute=0, second=0), now),
            r"昨天|yesterday": (
                (now - timedelta(days=1)).replace(hour=0, minute=0, second=0),
                now.replace(hour=0, minute=0, second=0),
            ),
            r"这周|this week": (now - timedelta(days=now.weekday()), now),
            r"上周|last week": (
                now - timedelta(days=now.weekday() + 7),
                now - timedelta(days=now.weekday()),
            ),
            r"这个月|this month": (now.replace(day=1, hour=0, minute=0, second=0), now),
            r"最近|recently": (now - timedelta(days=7), now),
        }

        for pattern, time_range in patterns.items():
            if re.search(pattern, lower):
                return time_range
        return None

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[tuple[list[dict], float]],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion — merge multiple ranked lists into one.
        score(d) = Σ weight_i / (k + rank_i(d))
        """
        scores: dict[str, float] = {}
        item_map: dict[str, dict] = {}

        for ranked_list, weight in ranked_lists:
            for rank, item in enumerate(ranked_list):
                mid = item["id"]
                rrf_score = weight / (k + rank + 1)
                scores[mid] = scores.get(mid, 0.0) + rrf_score
                if mid not in item_map:
                    item_map[mid] = item

        # Sort by fused score
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        results = []
        for mid in sorted_ids:
            entry = item_map[mid].copy()
            entry["score"] = scores[mid]
            results.append(entry)
        return results
