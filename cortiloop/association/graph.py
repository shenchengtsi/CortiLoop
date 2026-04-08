"""
Association Graph — Hebbian-weighted knowledge graph.

Brain analogy: "Neurons that fire together wire together."
Co-occurrence strengthens edges; disuse weakens them.
Spreading activation traverses the graph for retrieval.
"""

from __future__ import annotations

import math
from datetime import datetime

from cortiloop.models import EdgeType, MemoryEdge, MemoryUnit
from cortiloop.storage.base_store import BaseStore

LEARNING_RATE = 0.2
DECAY_RATE = 0.01  # edge weight decay per day


class AssociationGraph:
    """Manages the association layer — edges between memory entities."""

    def __init__(self, store: BaseStore):
        self.store = store

    def link_co_occurring(self, units: list[MemoryUnit]):
        """
        Create/strengthen co-occurrence edges between all entities
        that appear in the same encoding batch. (Hebbian learning)
        """
        all_entity_ids: list[tuple[str, str]] = []
        for unit in units:
            for entity in unit.entities:
                all_entity_ids.append((unit.id, entity))

        # Link entities that co-occur
        entities = list({e for _, e in all_entity_ids})
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                self._strengthen_edge(e1, e2, EdgeType.CO_OCCURRENCE)

        # Link units temporally (sequential encoding)
        for i in range(len(units) - 1):
            self._strengthen_edge(
                units[i].id, units[i + 1].id, EdgeType.TEMPORAL
            )

    def _strengthen_edge(self, source: str, target: str, edge_type: EdgeType):
        """Apply Hebbian strengthening to an edge."""
        existing = self.store.get_edge(source, target, edge_type)
        now = datetime.now()

        if existing:
            # Time-based decay since last co-activation
            elapsed_days = (now - existing.last_co_activated).total_seconds() / 86400
            decayed_weight = existing.weight * math.exp(-DECAY_RATE * elapsed_days)

            # Hebbian update: co-activation strengthens
            new_weight = decayed_weight + LEARNING_RATE
            existing.weight = min(new_weight, 10.0)  # cap
            existing.co_activation_count += 1
            existing.last_co_activated = now
            self.store.upsert_edge(existing)
        else:
            edge = MemoryEdge(
                source_id=source,
                target_id=target,
                edge_type=edge_type,
                weight=1.0,
                co_activation_count=1,
                last_co_activated=now,
                created_at=now,
            )
            self.store.upsert_edge(edge)

    def strengthen_on_retrieval(self, memory_ids: list[str]):
        """
        When multiple memories are co-retrieved, strengthen their links.
        (Testing effect: retrieval itself strengthens memory.)
        """
        for i, id1 in enumerate(memory_ids):
            for id2 in memory_ids[i + 1:]:
                self._strengthen_edge(id1, id2, EdgeType.CO_OCCURRENCE)

    def spreading_activation(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        decay_factor: float = 0.5,
    ) -> dict[str, float]:
        """
        Spread activation from seed nodes through the graph.
        Returns {node_id: activation_score}.

        Brain analogy: Collins & Loftus spreading activation —
        activation radiates from a node and decays with distance.
        """
        activations: dict[str, float] = {}
        for sid in seed_ids:
            activations[sid] = 1.0

        frontier = list(seed_ids)
        for hop in range(max_hops):
            next_frontier = []
            hop_decay = decay_factor ** (hop + 1)

            for node_id in frontier:
                edges = self.store.get_edges_from(node_id)
                edges += self.store.get_edges_to(node_id)

                for edge in edges:
                    neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
                    activation = edge.weight * hop_decay
                    if neighbor in activations:
                        activations[neighbor] = max(activations[neighbor], activation)
                    else:
                        activations[neighbor] = activation
                        next_frontier.append(neighbor)

            frontier = next_frontier

        return activations
