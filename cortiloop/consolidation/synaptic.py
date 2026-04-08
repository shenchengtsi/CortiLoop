"""
Synaptic Consolidation — immediate fact→observation abstraction.

Brain analogy: Protein synthesis stabilizes synaptic changes (minutes-hours).
Triggered after each retain, runs asynchronously.
Each Observation tracks a single dimension (single cortical column).
"""

from __future__ import annotations

from datetime import datetime

from cortiloop.config import ConsolidationConfig
from cortiloop.llm.protocol import Embedder, MemoryLLM
from cortiloop.models import MemoryUnit, Observation
from cortiloop.storage.base_store import BaseStore

_CONSOLIDATION_PROMPT = """You are a memory consolidation system. Given new facts and existing observations,
decide whether to CREATE a new observation or UPDATE an existing one.

Rules:
- Each observation tracks exactly ONE dimension (one person, one topic, one metric)
- Never combine unrelated topics into a single observation
- When updating, integrate new information into the existing text without losing prior facts
- Do NOT do math or counting — state what happened, not computed totals

Return JSON:
{
  "actions": [
    {
      "type": "create" or "update",
      "observation_id": "existing ID if update, empty if create",
      "dimension": "what this observation tracks (e.g. 'Alice:role', 'ProjectX:status')",
      "content": "the complete observation text incorporating new information",
      "source_unit_ids": ["ids of memory units this is based on"],
      "entities": ["entity1", "entity2"]
    }
  ]
}

If no observation is warranted, return {"actions": []}."""


class SynapticConsolidator:
    """
    Immediate consolidation: memory_units → Observations.
    Runs after each retain operation (async).
    """

    def __init__(self, config: ConsolidationConfig, store: BaseStore, llm: MemoryLLM, embedder: Embedder):
        self.config = config
        self.store = store
        self.llm = llm
        self.embedder = embedder

    async def consolidate(self, new_units: list[MemoryUnit]):
        """Process new memory units into Observations."""
        if not self.config.synaptic_enabled or not new_units:
            return

        # Gather relevant existing observations
        entities = set()
        for u in new_units:
            entities.update(u.entities)

        existing_obs = []
        for entity in list(entities)[:10]:
            for obs in self.store.search_observations_by_dimension(entity):
                if obs.id not in {o.id for o in existing_obs}:
                    existing_obs.append(obs)

        # Also search by vector similarity for related observations
        if new_units[0].embedding:
            for obs, sim in self.store.search_observations_by_vector(new_units[0].embedding, 5):
                if obs.id not in {o.id for o in existing_obs} and sim > 0.7:
                    existing_obs.append(obs)

        # Build context for LLM
        new_facts_text = "\n".join(
            f"[{u.id}] {u.content} (entities: {', '.join(u.entities)})"
            for u in new_units
        )
        existing_obs_text = "\n".join(
            f"[{o.id}] dimension={o.dimension}: {o.content}"
            for o in existing_obs
        ) or "(no existing observations)"

        user_msg = f"""NEW FACTS:
{new_facts_text}

EXISTING OBSERVATIONS:
{existing_obs_text}"""

        result = await self.llm.complete_json(
            system=_CONSOLIDATION_PROMPT,
            user=user_msg,
        )

        now = datetime.now()
        for action in result.get("actions", []):
            if action["type"] == "create":
                embedding = await self.embedder.embed_one(action["content"])
                obs = Observation(
                    dimension=action.get("dimension", ""),
                    content=action["content"],
                    source_unit_ids=action.get("source_unit_ids", []),
                    entities=action.get("entities", []),
                    embedding=embedding,
                    created_at=now,
                    updated_at=now,
                    last_accessed=now,
                    decay_rate=self.store.config.decay.semantic_rate,
                )
                self.store.insert_observation(obs)

            elif action["type"] == "update":
                obs_id = action.get("observation_id", "")
                existing = self.store.get_observation(obs_id) if obs_id else None
                if existing:
                    # Preserve history (reconsolidation trace)
                    existing.history.append({
                        "version": existing.version,
                        "content": existing.content,
                        "updated_at": existing.updated_at.isoformat(),
                    })
                    existing.content = action["content"]
                    existing.version += 1
                    existing.updated_at = now
                    existing.source_unit_ids.extend(action.get("source_unit_ids", []))
                    existing.entities = list(set(existing.entities + action.get("entities", [])))
                    existing.embedding = await self.embedder.embed_one(action["content"])
                    self.store.insert_observation(existing)
