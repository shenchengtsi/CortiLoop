"""
Systems Consolidation — deep cross-session integration.

Brain analogy: Sleep-driven hippocampus→neocortex transfer.
Runs periodically in "sleep mode" — batch processing for:
1. Cross-session observation merging
2. Episodic→semantic transformation (gist extraction)
3. Procedural memory detection (repeated patterns)
4. Mental model generation (Reflect)
"""

from __future__ import annotations

from datetime import datetime

from cortiloop.config import ConsolidationConfig
from cortiloop.llm.protocol import MemoryLLM
from cortiloop.models import Observation, ProceduralMemory
from cortiloop.storage.base_store import BaseStore

_REFLECT_PROMPT = """You are a deep reflection system. Given a set of observations about a topic,
generate a high-level mental model — a concise synthesis that captures the essential patterns.

Return JSON:
{
  "mental_model": "A concise paragraph synthesizing the key patterns and insights",
  "dimension": "what this model covers (e.g. 'user:work_style', 'project:architecture')",
  "entities": ["key entities"],
  "source_observation_ids": ["obs IDs used"]
}

If the observations are too sparse or unrelated for meaningful synthesis, return {"mental_model": ""}."""

_PROCEDURAL_DETECT_PROMPT = """You are a pattern detection system. Given recent memory units,
identify any REPEATED behavioral patterns that could be procedural memories (learned habits/workflows).

A procedural memory must:
1. Have been observed at least 2-3 times
2. Follow a consistent trigger→action pattern
3. Be specific enough to be actionable

Return JSON:
{
  "patterns": [
    {
      "pattern": "trigger/situation description",
      "procedure": "the repeated action/workflow",
      "entities": ["entity1"],
      "confidence": 0.3 to 1.0
    }
  ]
}

Return {"patterns": []} if no clear repeated patterns found."""


class SystemsConsolidator:
    """
    Deep consolidation: runs periodically to perform cross-session integration.
    The "sleep cycle" of the memory system.
    """

    def __init__(self, config: ConsolidationConfig, store: BaseStore, llm: MemoryLLM):
        self.config = config
        self.store = store
        self.llm = llm

    async def run_deep_consolidation(self):
        """Full deep consolidation cycle. Call periodically."""
        if not self.config.systems_enabled:
            return

        await self._detect_procedural_memories()
        await self._generate_mental_models()

    async def _detect_procedural_memories(self):
        """
        Scan recent memory units for repeated patterns.
        Brain analogy: basal ganglia gradually acquiring motor programs through repetition.
        """
        recent_units = self.store.get_recent_units(limit=self.config.max_units_per_batch)
        if len(recent_units) < 5:
            return

        units_text = "\n".join(
            f"[{u.created_at.isoformat()[:10]}] {u.content}"
            for u in recent_units
        )

        result = await self.llm.complete_json(
            system=_PROCEDURAL_DETECT_PROMPT,
            user=units_text,
        )

        now = datetime.utcnow()
        existing_procs = self.store.get_active_procedurals()

        for pat in result.get("patterns", []):
            if not pat.get("pattern") or not pat.get("procedure"):
                continue

            # Check if similar procedural memory exists
            matched = False
            for ep in existing_procs:
                if self._is_similar_pattern(ep.pattern, pat["pattern"]):
                    # Strengthen existing procedural memory
                    ep.acquisition_count += 1
                    ep.confidence = min(ep.confidence + 0.15, 1.0)
                    ep.last_accessed = now
                    self.store.insert_procedural(ep)
                    matched = True
                    break

            if not matched and pat.get("confidence", 0) >= 0.3:
                embedding = await self.llm.embed_one(pat["pattern"] + " " + pat["procedure"])
                pm = ProceduralMemory(
                    pattern=pat["pattern"],
                    procedure=pat["procedure"],
                    entities=pat.get("entities", []),
                    confidence=pat.get("confidence", 0.3),
                    embedding=embedding,
                    created_at=now,
                    last_accessed=now,
                )
                self.store.insert_procedural(pm)

    async def _generate_mental_models(self):
        """
        Generate high-level mental models from clusters of observations.
        Brain analogy: REM sleep abstracting gist from detailed episodes.
        """
        observations = self.store.get_active_observations(limit=100)
        if len(observations) < 5:
            return

        # Group observations by entity overlap
        entity_groups: dict[str, list[Observation]] = {}
        for obs in observations:
            for entity in obs.entities[:3]:  # limit to top entities
                entity_groups.setdefault(entity, []).append(obs)

        # Generate mental model for groups with enough observations
        for entity, obs_list in entity_groups.items():
            if len(obs_list) < 3:
                continue

            obs_text = "\n".join(
                f"[{o.id}] {o.dimension}: {o.content}"
                for o in obs_list[:15]
            )

            result = await self.llm.complete_json(
                system=_REFLECT_PROMPT,
                user=f"OBSERVATIONS about '{entity}':\n{obs_text}",
            )

            model_text = result.get("mental_model", "")
            if not model_text:
                continue

            # Store as a high-level observation with low decay rate
            now = datetime.utcnow()
            embedding = await self.llm.embed_one(model_text)
            mental_model = Observation(
                dimension=result.get("dimension", f"{entity}:mental_model"),
                content=model_text,
                source_unit_ids=result.get("source_observation_ids", []),
                entities=result.get("entities", [entity]),
                embedding=embedding,
                created_at=now,
                updated_at=now,
                last_accessed=now,
                decay_rate=0.01,  # mental models decay very slowly
                confidence=0.8,
            )
            self.store.insert_observation(mental_model)

    @staticmethod
    def _is_similar_pattern(a: str, b: str) -> bool:
        """Simple word-overlap similarity check for procedural patterns."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        return overlap / min(len(words_a), len(words_b)) > 0.5
