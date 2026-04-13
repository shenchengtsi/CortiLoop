"""
Encoder — extracts structured facts + entities from raw input.

Brain analogy: Hippocampal encoding — binding multi-modal cortical representations
into a unified memory trace with entity links.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from cortiloop.config import CortiLoopConfig
from cortiloop.llm.protocol import Embedder, MemoryLLM
from cortiloop.models import EncodingContext, MemoryUnit, SourceType

logger = logging.getLogger("cortiloop.encoding")

# Chunking parameters: split long multi-turn conversations so that casual
# mentions (dates, prices, device names) are not drowned by the dominant topic.
_CHUNK_MAX_TURNS = 4        # max conversation turns per chunk
_CHUNK_CHAR_THRESHOLD = 1500  # only chunk texts longer than this

_TURN_BOUNDARY = re.compile(r"^(user|assistant|system):\s", re.MULTILINE | re.IGNORECASE)

_EXTRACT_PROMPT = """You are a memory extraction system. Extract important facts from the input.

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
1. Extract up to 7 facts. Merge closely related details into one fact instead of splitting.
   BAD:  "CortiLoop uses SQLite" + "CortiLoop uses Python" (two fragments)
   GOOD: "CortiLoop is a Python-based memory engine using SQLite storage" (one complete fact)
2. Each fact must be self-contained — understandable without reading the original text.
   BAD:  "py/updater.py" or "核心特色：7层生物记忆" (fragments, not self-contained)
   GOOD: "CortiLoop's core feature is a 7-layer bioinspired memory lifecycle"
3. ALWAYS extract facts containing specific details, even if they appear as casual mentions ("By the way", "also", "recently"):
   - Numbers, dates, prices, quantities ("on February 20th", "$350,000", "7 shirts")
   - Product/brand names ("Samsung Galaxy S22", "Dell XPS 13")
   - Place names ("Target", "Best Buy at the mall")
   - Named items ("playlist called Summer Vibes", "Anker PowerCore 20000")
4. Extract facts from BOTH user and assistant messages equally.
   Assistant details (recommendations, tables, stories, specific names mentioned) are just as important.
5. Extract implicit user preferences and habits inferred from the conversation:
   BAD:  skip "I've been experimenting with turbinado sugar"
   GOOD: "User prefers baking with turbinado sugar for richer flavor"
   BAD:  skip "I enjoy playing blues guitar"
   GOOD: "User is interested in blues guitar, relevant for music recommendations"
6. Entity rules:
   - Only extract proper nouns: person names, project names, organizations, technologies
   - Use FULL names: "Hindsight" not "Hindsigh", "Cross-encoder" not "Cross", "Mental Models" not "Mental M"
   - Do NOT extract common words as entities: "我正", "项目托管", "Issue" are NOT entities
   - Chinese proper nouns should stay in Chinese: "字节跳动", "火山引擎"
7. source_type: "user_said" for explicit statements, "llm_inferred" for conclusions/preferences you derive
8. skip=true for: greetings, "ok/thanks", commands, pure code snippets, section headers like "【标题】"
9. For long documents or comparison tables: summarize key conclusions, don't extract every row"""


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

        Long multi-turn conversations are automatically chunked so that
        casual mentions (dates, devices, prices) are not drowned by the
        dominant topic of the conversation.
        """
        chunks = self._chunk_conversation(text)

        # Extract facts from each chunk independently
        all_facts: list[dict] = []
        for chunk in chunks:
            result = await self.llm.complete_json(
                system=_EXTRACT_PROMPT,
                user=chunk,
            )
            if not result.get("skip", False):
                all_facts.extend(result.get("facts", []))

        if not all_facts:
            return []

        # Batch embed all fact contents
        contents = [f["content"] for f in all_facts]
        embeddings = await self.embedder.embed(contents)

        units = []
        now = datetime.now()
        for i, fact in enumerate(all_facts):
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

    # ── Chunking ──

    @staticmethod
    def _chunk_conversation(
        text: str,
        max_turns: int = _CHUNK_MAX_TURNS,
        char_threshold: int = _CHUNK_CHAR_THRESHOLD,
    ) -> list[str]:
        """Split long multi-turn text into smaller chunks.

        Only activates when *both* conditions are met:
        1. The text is longer than ``char_threshold``
        2. The text contains more than ``max_turns`` conversation turns

        Each chunk contains up to ``max_turns`` turns so the LLM fact-extractor
        can focus on a manageable amount of content.
        """
        if len(text) < char_threshold:
            return [text]

        # Find turn boundaries (lines starting with "user:", "assistant:", etc.)
        boundaries = [m.start() for m in _TURN_BOUNDARY.finditer(text)]

        if len(boundaries) <= max_turns:
            return [text]

        chunks = []
        for i in range(0, len(boundaries), max_turns):
            start = boundaries[i]
            end = boundaries[i + max_turns] if i + max_turns < len(boundaries) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

        logger.debug(
            "Chunked %d-char text into %d chunks (%d turns each)",
            len(text), len(chunks), max_turns,
        )
        return chunks if chunks else [text]
