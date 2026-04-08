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

_EXTRACT_PROMPT = """You are a memory extraction system. Extract the most important facts from the input.

Return JSON:
{
  "facts": [
    {
      "content": "a complete, self-contained factual statement",
      "entities": ["Person Name", "Project Name", "Organization"],
      "source_type": "user_said"
    }
  ],
  "skip": true/false
}

Rules:
1. QUALITY over quantity: extract at most 5 facts. Merge related details into one fact instead of splitting.
   BAD:  "CortiLoop uses SQLite" + "CortiLoop uses Python" (two fragments)
   GOOD: "CortiLoop is a Python-based memory engine using SQLite storage" (one complete fact)
2. Each fact must be self-contained — understandable without reading the original text.
   BAD:  "py/updater.py" or "核心特色：7层生物记忆" (fragments, not self-contained)
   GOOD: "CortiLoop's core feature is a 7-layer bioinspired memory lifecycle"
3. Entity rules:
   - Only extract proper nouns: person names, project names, organizations, technologies
   - Use FULL names: "Hindsight" not "Hindsigh", "Cross-encoder" not "Cross", "Mental Models" not "Mental M"
   - Do NOT extract common words as entities: "我正", "项目托管", "Issue" are NOT entities
   - Chinese proper nouns should stay in Chinese: "字节跳动", "火山引擎"
4. source_type: "user_said" for explicit statements, "llm_inferred" for conclusions you derive
5. skip=true for: greetings, "ok/thanks", commands, pure code snippets, section headers like "【标题】"
6. For long documents or comparison tables: summarize key conclusions, don't extract every row"""


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
