"""
CortiLoop — Bioinspired Agent Memory Engine

A memory system for AI agents modeled after human brain mechanisms:
encoding, consolidation, storage, retrieval, association, forgetting, reconsolidation.

Designed as a plugin for nanobot, openclaw, or any MCP-compatible agent framework.
"""

__version__ = "0.4.7"

from cortiloop.engine import CortiLoop
from cortiloop.llm.protocol import Embedder, MemoryLLM, Reranker
from cortiloop.models import MemoryUnit, Observation, ProceduralMemory, MemoryEdge
from cortiloop.config import CortiLoopConfig

__all__ = [
    "CortiLoop",
    "CortiLoopConfig",
    "MemoryLLM",
    "Embedder",
    "Reranker",
    "MemoryUnit",
    "Observation",
    "ProceduralMemory",
    "MemoryEdge",
]
