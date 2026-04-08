"""
Attention Gate — decides what's worth remembering.

Brain analogy: Prefrontal cortex attention filtering + amygdala emotional modulation
+ dopaminergic novelty signal.
"""

from __future__ import annotations

import re

from cortiloop.config import AttentionGateConfig
from cortiloop.llm.protocol import MemoryLLM


# Signals that indicate correction / prediction error → strong encoding
_CORRECTION_PATTERNS = [
    r"不对", r"错了", r"应该是", r"不是.*而是", r"纠正", r"更正",
    r"no[,.]?\s*(it|that)'s", r"actually", r"correction", r"wrong",
    r"should be", r"not .* but",
]

# Explicit memory request signals
_EXPLICIT_PATTERNS = [
    r"记住", r"别忘了", r"请记", r"remember", r"don't forget", r"keep in mind",
    r"note that", r"important:",
]


class AttentionGate:
    """
    Evaluates whether input is worth encoding into long-term memory.
    Returns an importance score [0, 1].

    High scores for: corrections (prediction error), explicit requests,
    novel information, emotional content.
    Low scores for: greetings, repetition, phatic communication.
    """

    def __init__(self, config: AttentionGateConfig, llm: MemoryLLM):
        self.config = config
        self.llm = llm

    async def score(
        self,
        text: str,
        existing_entity_count: int = 0,
        task_context: str = "",
    ) -> float:
        """
        Compute importance score for input text.
        Uses rule-based signals + optional LLM scoring.
        """
        if not self.config.enabled:
            return 1.0  # pass everything through

        scores: dict[str, float] = {}

        # Correction signal (prediction error → dopamine burst → strong encoding)
        scores["correction"] = self._detect_correction(text)

        # Explicit memory request
        scores["explicit_mark"] = self._detect_explicit_request(text)

        # Novelty (simple proxy: new entities / length ratio)
        scores["novelty"] = self._estimate_novelty(text, existing_entity_count)

        # Emotional intensity (keyword-based fast estimate)
        scores["emotional_intensity"] = self._estimate_emotion(text)

        # Task relevance
        scores["task_relevance"] = self._estimate_task_relevance(text, task_context)

        # Strong signals (correction, explicit) should pass the gate on their own
        # even if other dimensions score low. Use max(strong_signal, weighted_sum).
        strong_signal = max(scores.get("correction", 0), scores.get("explicit_mark", 0))

        # Weighted combination of all dimensions
        weighted_sum = sum(
            scores[k] * self.config.weights.get(k, 0.0)
            for k in scores
        )

        total = max(strong_signal * 0.8, weighted_sum)
        return min(max(total, 0.0), 1.0)

    def passes(self, score: float) -> bool:
        return score >= self.config.threshold

    # ── Signal detectors ──

    @staticmethod
    def _detect_correction(text: str) -> float:
        for pattern in _CORRECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        return 0.0

    @staticmethod
    def _detect_explicit_request(text: str) -> float:
        for pattern in _EXPLICIT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        return 0.0

    @staticmethod
    def _estimate_novelty(text: str, existing_entity_count: int) -> float:
        # Longer, substantive text is more likely novel
        # Use character count for CJK-friendly estimation
        char_count = len(text.strip())
        if char_count < 5:
            return 0.1
        # 20+ chars = 0.4+, 50+ chars = 1.0 — most real sentences qualify
        length_score = min(char_count / 50, 1.0)
        # If few existing entities, everything is novel
        entity_bonus = 1.0 if existing_entity_count < 10 else 0.6
        return min(length_score * entity_bonus, 1.0)

    @staticmethod
    def _estimate_emotion(text: str) -> float:
        emotion_words = [
            "!", "？！", "太", "非常", "极其", "amazing", "terrible", "love",
            "hate", "excited", "frustrated", "angry", "happy", "worried",
            "urgent", "critical", "紧急", "严重", "崩溃", "完美",
        ]
        count = sum(1 for w in emotion_words if w.lower() in text.lower())
        return min(count / 3, 1.0)

    @staticmethod
    def _estimate_task_relevance(text: str, task_context: str) -> float:
        if not task_context:
            return 0.5  # neutral when no context
        # Simple word overlap
        text_words = set(text.lower().split())
        task_words = set(task_context.lower().split())
        if not task_words:
            return 0.5
        overlap = len(text_words & task_words)
        return min(overlap / max(len(task_words), 1), 1.0)
