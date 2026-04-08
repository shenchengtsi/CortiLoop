"""
Encoder — extracts structured facts + entities from raw input.

Brain analogy: Hippocampal encoding — binding multi-modal cortical representations
into a unified memory trace with entity links.
"""

from __future__ import annotations

from datetime import datetime

from cortiloop.config import CortiLoopConfig
from cortiloop.llm.protocol import Embedder, MemoryLLM
from cortiloop.models import EncodingContext, MemoryUnit, SourceType

_EXTRACT_PROMPT = """You are a memory extraction system. Given a conversation message, extract structured facts.

Return JSON:
{
  "facts": [
    {
      "content": "one atomic fact in a single sentence",
      "entities": ["entity1", "entity2"],
      "source_type": "user_said" or "llm_inferred"
    }
  ],
  "skip": true/false  // true if the message contains no memorable information (greetings, acknowledgements)
}

Rules:
- Each fact must be a single, atomic statement
- Entity names should be normalized (consistent casing, no abbreviations)
- source_type is "user_said" for facts directly stated by the user, "llm_inferred" for inferences
- Set skip=true for pure greetings, "ok", "thanks", acknowledgements with no factual content
- Prefer extracting fewer, higher-quality facts over many low-quality ones"""


class Encoder:
    """Extracts structured memory units from raw text input."""

    def __init__(self, config: CortiLoopConfig, llm: MemoryLLM, embedder: Embedder):
        self.config = config
        self.llm = llm
        self.embedder = embedder

    async def encode(
        self,
        text: str,
        importance_score: float,
        session_id: str = "",
        task_context: str = "",
    ) -> list[MemoryUnit]:
        """
        Extract memory units from input text.
        Returns empty list if nothing worth remembering.
        """
        result = await self.llm.complete_json(
            system=_EXTRACT_PROMPT,
            user=text,
        )

        if result.get("skip", False):
            return []

        facts = result.get("facts", [])
        if not facts:
            return []

        # Batch embed all fact contents
        contents = [f["content"] for f in facts]
        embeddings = await self.embedder.embed(contents)

        units = []
        now = datetime.now()
        for i, fact in enumerate(facts):
            entities = [e.strip() for e in fact.get("entities", []) if e.strip()]
            source = SourceType.USER_SAID
            if fact.get("source_type") == "llm_inferred":
                source = SourceType.LLM_INFERRED

            unit = MemoryUnit(
                content=fact["content"],
                source_type=source,
                importance_score=importance_score,
                encoding_context=EncodingContext(
                    task=task_context,
                    entities=entities,
                    session_id=session_id,
                ),
                entities=entities,
                embedding=embeddings[i] if i < len(embeddings) else [],
                created_at=now,
                decay_rate=self.config.decay.episodic_rate,
                last_accessed=now,
            )
            units.append(unit)

        return units
