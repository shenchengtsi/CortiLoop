"""
Local LLM client — deterministic, zero-dependency, no API key needed.

Uses rule-based fact extraction + hash-based embeddings for offline operation.
Useful for benchmarking, testing, and environments without API access.

Brain analogy: like running cognitive tasks on the brainstem alone —
crude but functional pattern matching without the full cortex.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


class LocalLLMClient:
    """
    Drop-in replacement for LLMClient that works entirely offline.
    Uses deterministic algorithms instead of LLM API calls.
    """

    def __init__(self, embedding_dim: int = 256):
        self._dim = embedding_dim

    async def complete(self, system: str, user: str, response_format: str = "json") -> str:
        """Route to appropriate handler based on system prompt content."""
        lower_sys = system.lower()

        if "memory extraction" in lower_sys or "extract" in lower_sys:
            return json.dumps(self._extract_facts(user))
        elif "consolidation" in lower_sys:
            return json.dumps(self._consolidate(user))
        elif "conflict" in lower_sys:
            return json.dumps(self._detect_conflict(user))
        elif "reflection" in lower_sys or "mental model" in lower_sys:
            return json.dumps(self._reflect(user))
        elif "procedural" in lower_sys or "pattern" in lower_sys:
            return json.dumps(self._detect_patterns(user))
        elif "relevance" in lower_sys and "scorer" in lower_sys:
            return json.dumps(self._score_relevance(user))
        else:
            return json.dumps({"result": "ok"})

    async def complete_json(self, system: str, user: str) -> dict:
        text = await self.complete(system, user, response_format="json")
        return json.loads(text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic hash-based embeddings."""
        return [self._hash_embed(t) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        return self._hash_embed(text)

    async def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Rerank by word overlap similarity."""
        q_words = set(self._tokenize(query))
        scored = []
        for i, doc in enumerate(documents):
            d_words = set(self._tokenize(doc))
            overlap = len(q_words & d_words)
            score = overlap / max(len(q_words | d_words), 1)
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Hash-based embedding ──

    def _hash_embed(self, text: str) -> list[float]:
        """
        Generate a deterministic embedding from text using character n-gram hashing.
        Similar texts produce similar vectors (locality-sensitive).
        """
        vec = [0.0] * self._dim

        # Character n-grams (n=3) for language-agnostic similarity
        text_lower = text.lower().strip()
        ngrams = []
        for n in (2, 3, 4):
            for i in range(len(text_lower) - n + 1):
                ngrams.append(text_lower[i:i + n])

        # Also add whole words
        words = self._tokenize(text)
        ngrams.extend(words)

        for gram in ngrams:
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    # ── Rule-based fact extraction ──

    def _extract_facts(self, text: str) -> dict:
        """Extract facts using sentence splitting + entity detection."""
        text = text.strip()
        if not text or len(text) < 3:
            return {"facts": [], "skip": True}

        # Skip greetings and acknowledgements
        skip_patterns = [
            r"^(ok|okay|好的|嗯|hmm|thanks|谢谢|hello|hi|你好|bye|再见)\s*[.!？。]*$",
        ]
        for pat in skip_patterns:
            if re.match(pat, text.strip(), re.IGNORECASE):
                return {"facts": [], "skip": True}

        # Split into sentences — preserve version numbers like "v2.0"
        # Use lookbehind to avoid splitting on decimal points in versions
        sentences = re.split(r'(?<!\d)[.。!！?？;；\n]+|(?<=\d\.)\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        # If splitting produced nothing useful, use the whole text
        if not sentences and len(text.strip()) > 3:
            sentences = [text.strip()]

        if not sentences:
            return {"facts": [], "skip": True}

        facts = []
        for sent in sentences:
            entities = self._extract_entities(sent)
            source_type = "user_said"

            # Detect corrections as higher-priority facts
            if re.search(r'(actually|不对|应该是|wrong|correction|纠正)', sent, re.IGNORECASE):
                source_type = "user_said"

            facts.append({
                "content": sent,
                "entities": entities,
                "source_type": source_type,
            })

        return {"facts": facts, "skip": False}

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities using capitalization + CJK name patterns."""
        entities = []

        # Capitalized words (English names, proper nouns)
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
        # Filter common sentence starters
        stop = {"The", "This", "That", "These", "Those", "What", "When",
                "Where", "How", "Why", "It", "He", "She", "They", "We",
                "My", "Your", "His", "Her", "Its", "Our", "Their",
                "But", "And", "Or", "So", "If", "For", "With", "From",
                "Not", "No", "Yes", "Also", "Just", "Actually",
                "Session", "Meeting", "Recently", "Set", "Each", "Every"}
        for cap in caps:
            if cap not in stop and len(cap) > 1:
                entities.append(cap)

        # Version numbers (v1.0, v2.0)
        versions = re.findall(r'(v\d+\.\d+)', text)
        entities.extend(versions)

        # Tech terms (React, TypeScript, Python, etc.)
        tech_pattern = r'\b(React|Vue|Angular|TypeScript|JavaScript|Python|Go|Rust|Node|Docker|PostgreSQL|Redis|GraphQL|REST|API|SDK|CLI|MCP|HNSW)\b'
        techs = re.findall(tech_pattern, text, re.IGNORECASE)
        entities.extend(techs)

        # CJK names (2-3 char sequences that look like names)
        cjk_names = re.findall(r'([\u4e00-\u9fff]{2,4})(?:是|在|说|做|负责|管理|使用)', text)
        entities.extend(cjk_names)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)
        return unique

    # ── Rule-based consolidation ──

    def _consolidate(self, text: str) -> dict:
        """Decide CREATE or UPDATE for consolidation."""
        # Simple: always create new observation
        return {
            "action": "CREATE",
            "dimension": "general",
            "content": text[:200] if len(text) > 200 else text,
        }

    def _detect_conflict(self, text: str) -> dict:
        """Detect conflicts between old and new info."""
        lower = text.lower()
        if any(w in lower for w in ["actually", "不对", "wrong", "correction", "should be", "应该是", "switched", "changed"]):
            return {"conflict_type": "correction", "resolution": "supersede"}
        if any(w in lower for w in ["also", "还", "另外", "additionally"]):
            return {"conflict_type": "supplement", "resolution": "coexist"}
        return {"conflict_type": "none", "resolution": "none"}

    def _reflect(self, text: str) -> dict:
        """Generate mental model from observations."""
        return {
            "mental_model": text[:300],
            "confidence": 0.7,
        }

    def _detect_patterns(self, text: str) -> dict:
        """Detect procedural patterns."""
        return {"patterns": [], "confidence": 0.5}

    def _score_relevance(self, text: str) -> dict:
        """Score document relevance (for reranking fallback)."""
        # Parse the user message to find query and documents
        lines = text.split("\n")
        scores = []
        idx = 0
        for line in lines:
            if line.strip().startswith("[") and "]" in line:
                scores.append({"index": idx, "score": 0.5})
                idx += 1
        return {"scores": scores}

    # ── Helpers ──

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization for both English and CJK."""
        # English words
        words = re.findall(r'[a-zA-Z]+', text.lower())
        # CJK characters (each char is a token)
        cjk = re.findall(r'[\u4e00-\u9fff]', text)
        return words + cjk
