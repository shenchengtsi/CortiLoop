"""
Reconsolidation — safe memory update protocol.

Brain analogy:
- Retrieved memories enter a labile state (destabilization)
- New info can be integrated during the reconsolidation window
- Re-stabilization requires new protein synthesis
- Risk: false memories can be implanted during this window

Design: original memory_units are NEVER modified (immutable hippocampal trace).
Only Observations can be updated, with full history preserved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cortiloop.llm.client import LLMClient
from cortiloop.models import ConflictRecord, MemoryUnit, Observation, SourceType
from cortiloop.storage.base_store import BaseStore

_CONFLICT_DETECT_PROMPT = """You are a memory conflict detection system. Given a NEW fact and an EXISTING observation,
determine if they conflict.

Return JSON:
{
  "conflicts": true/false,
  "conflict_type": "correction" | "contradiction" | "update" | "supplement" | "none",
  "resolution": "supersede" | "coexist" | "merge",
  "merged_content": "if resolution is merge, provide the merged observation text",
  "confidence": 0.0 to 1.0,
  "explanation": "brief reason"
}

Rules:
- "correction": user explicitly corrects prior information → supersede old
- "contradiction": two incompatible facts without clear winner → coexist
- "update": new info updates/refines old info (e.g. status change) → merge
- "supplement": new info adds detail without conflict → merge
- "none": no relationship → no action"""


class Reconsolidator:
    """
    Handles memory update with conflict detection and safe resolution.
    Ensures immutability of raw memory_units while allowing Observation evolution.
    """

    def __init__(self, store: BaseStore, llm: LLMClient):
        self.store = store
        self.llm = llm

    async def check_and_update(
        self,
        new_units: list[MemoryUnit],
        retrieved_obs: list[Observation],
    ) -> list[ConflictRecord]:
        """
        Check new memory units against retrieved observations for conflicts.
        Apply safe updates following the reconsolidation protocol.

        Returns list of detected conflicts for logging/audit.
        """
        conflicts: list[ConflictRecord] = []

        for unit in new_units:
            for obs in retrieved_obs:
                # Skip if no entity overlap
                if not set(unit.entities) & set(obs.entities):
                    continue

                conflict = await self._detect_conflict(unit, obs)
                if conflict:
                    conflicts.append(conflict)
                    await self._apply_resolution(conflict, unit, obs)

        return conflicts

    async def _detect_conflict(
        self,
        new_unit: MemoryUnit,
        existing_obs: Observation,
    ) -> ConflictRecord | None:
        """Use LLM to detect if new fact conflicts with existing observation."""
        result = await self.llm.complete_json(
            system=_CONFLICT_DETECT_PROMPT,
            user=f"""NEW FACT: {new_unit.content}
EXISTING OBSERVATION [{existing_obs.dimension}]: {existing_obs.content}""",
        )

        if not result.get("conflicts", False):
            return None

        return ConflictRecord(
            old_memory_id=existing_obs.id,
            new_memory_id=new_unit.id,
            dimension=existing_obs.dimension,
            old_value=existing_obs.content,
            new_value=new_unit.content,
            resolution=result.get("resolution", "coexist"),
            created_at=datetime.utcnow(),
        )

    async def _apply_resolution(
        self,
        conflict: ConflictRecord,
        new_unit: MemoryUnit,
        existing_obs: Observation,
    ):
        """Apply the conflict resolution."""
        now = datetime.utcnow()

        if conflict.resolution == "supersede":
            # User correction: new info replaces old
            # But we NEVER delete — we update the observation with history
            existing_obs.history.append({
                "version": existing_obs.version,
                "content": existing_obs.content,
                "updated_at": existing_obs.updated_at.isoformat(),
                "superseded_by": new_unit.id,
            })
            existing_obs.content = new_unit.content
            existing_obs.version += 1
            existing_obs.updated_at = now
            existing_obs.source_unit_ids.append(new_unit.id)
            existing_obs.embedding = await self.llm.embed_one(new_unit.content)

            # Higher confidence for user corrections
            if new_unit.source_type == SourceType.USER_SAID:
                existing_obs.confidence = min(existing_obs.confidence + 0.1, 1.0)
            self.store.insert_observation(existing_obs)

        elif conflict.resolution == "merge":
            # Generate merged content via LLM
            merged = await self.llm.complete_json(
                system="Merge these two pieces of information into one coherent observation. Return JSON: {\"content\": \"merged text\"}",
                user=f"OLD: {existing_obs.content}\nNEW: {new_unit.content}",
            )
            merged_content = merged.get("content", new_unit.content)

            existing_obs.history.append({
                "version": existing_obs.version,
                "content": existing_obs.content,
                "updated_at": existing_obs.updated_at.isoformat(),
                "merged_with": new_unit.id,
            })
            existing_obs.content = merged_content
            existing_obs.version += 1
            existing_obs.updated_at = now
            existing_obs.source_unit_ids.append(new_unit.id)
            existing_obs.embedding = await self.llm.embed_one(merged_content)
            self.store.insert_observation(existing_obs)

        elif conflict.resolution == "coexist":
            # Cannot resolve — record conflict for future disambiguation
            pass

        # Always log the conflict
        self.store.insert_conflict(conflict)
