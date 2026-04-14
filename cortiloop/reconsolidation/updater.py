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

import numpy as np

from cortiloop.llm.protocol import Embedder, MemoryLLM
from cortiloop.models import ConflictRecord, MemoryUnit, Observation, SourceType
from cortiloop.storage.base_store import BaseStore

_CONFLICT_DETECT_PROMPT = """You are a memory conflict detection system. Given a NEW fact and an EXISTING observation,
determine if they conflict.

Return JSON:
{
  "conflicts": true/false,
  "conflict_type": "correction" | "contradiction" | "update" | "supplement" | "none",
  "resolution": "supersede" | "coexist" | "merge",
  "merged_content": "if resolution is merge or supersede, provide the updated observation text",
  "confidence": 0.0 to 1.0,
  "explanation": "brief reason"
}

Rules:
- "correction": user explicitly corrects prior information → supersede old
- "update": a numeric value, status, count, or factual detail about the SAME subject changes → supersede old
- "contradiction": two incompatible facts about DIFFERENT subjects → coexist
- "supplement": new info adds detail without conflict → merge
- "none": no relationship → no action

IMPORTANT: When the NEW fact changes a number, amount, frequency, status, or state for the same
subject as the EXISTING observation, this is ALWAYS "update" with resolution "supersede".
"coexist" should ONLY be used when the two facts describe genuinely DIFFERENT subjects or topics.

Examples:
- EXISTING: "User is pre-approved for $350,000 mortgage from Wells Fargo"
  NEW: "User is now pre-approved for $400,000 mortgage from Wells Fargo"
  → conflicts: true, conflict_type: "update", resolution: "supersede"

- EXISTING: "User's to-watch list has 20 titles"
  NEW: "User added 5 more shows, to-watch list now has 25 titles"
  → conflicts: true, conflict_type: "update", resolution: "supersede"

- EXISTING: "Mom uses paper grocery lists"
  NEW: "Mom now uses the same grocery list app as the user"
  → conflicts: true, conflict_type: "update", resolution: "supersede"

- EXISTING: "User's 5K personal best is 27:12"
  NEW: "User ran a charity 5K in 25:50, a new personal best"
  → conflicts: true, conflict_type: "correction", resolution: "supersede"

- EXISTING: "User leads a team of 4 engineers"
  NEW: "User's team grew, now leading 5 engineers"
  → conflicts: true, conflict_type: "update", resolution: "supersede"

- EXISTING: "User has been using Fitbit Charge 3 for 6 months"
  NEW: "User has been using Fitbit Charge 3 for 9 months now"
  → conflicts: true, conflict_type: "update", resolution: "supersede"

- EXISTING: "User attended 3 bereavement support group sessions"
  NEW: "User has now attended 5 bereavement support group sessions total"
  → conflicts: true, conflict_type: "update", resolution: "supersede\""""


class Reconsolidator:
    """
    Handles memory update with conflict detection and safe resolution.
    Ensures immutability of raw memory_units while allowing Observation evolution.
    """

    def __init__(self, store: BaseStore, llm: MemoryLLM, embedder: Embedder):
        self.store = store
        self.llm = llm
        self.embedder = embedder

    async def check_and_update(
        self,
        new_units: list[MemoryUnit],
        retrieved_obs: list[Observation],
    ) -> list[ConflictRecord]:
        """
        Check new memory units against retrieved observations for conflicts.
        Apply safe updates following the reconsolidation protocol.

        Trigger condition: entity overlap OR vector similarity >= 0.7.
        This ensures numeric/status updates are caught even when entity names
        don't match exactly (e.g. "5K run" vs "charity 5K").

        Returns list of detected conflicts for logging/audit.
        """
        conflicts: list[ConflictRecord] = []

        for unit in new_units:
            for obs in retrieved_obs:
                # Trigger conflict detection if entities overlap OR vectors are similar
                entity_overlap = bool(set(unit.entities) & set(obs.entities))
                vec_sim = self._cosine_sim(unit.embedding, obs.embedding)
                if not entity_overlap and vec_sim < 0.7:
                    continue

                conflict = await self._detect_conflict(unit, obs)
                if conflict:
                    conflicts.append(conflict)
                    await self._apply_resolution(conflict, unit, obs)

        return conflicts

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

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
            created_at=datetime.now(),
        )

    async def _apply_resolution(
        self,
        conflict: ConflictRecord,
        new_unit: MemoryUnit,
        existing_obs: Observation,
    ):
        """Apply the conflict resolution."""
        now = datetime.now()

        if conflict.resolution == "supersede":
            # User correction / update: new info replaces old
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
            existing_obs.embedding = await self.embedder.embed_one(new_unit.content)

            # Propagate session_timestamp from the newer unit
            if new_unit.session_timestamp:
                existing_obs.session_timestamp = new_unit.session_timestamp

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
            existing_obs.embedding = await self.embedder.embed_one(merged_content)

            # Propagate session_timestamp from the newer unit
            if new_unit.session_timestamp:
                existing_obs.session_timestamp = new_unit.session_timestamp

            self.store.insert_observation(existing_obs)

        elif conflict.resolution == "coexist":
            # Cannot resolve — record conflict for future disambiguation
            pass

        # Always log the conflict
        self.store.insert_conflict(conflict)
