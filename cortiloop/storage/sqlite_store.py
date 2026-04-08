"""SQLite-based storage backend with pluggable vector index."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

import numpy as np

from cortiloop.config import CortiLoopConfig
from cortiloop.models import (
    ConflictRecord,
    EdgeType,
    MemoryEdge,
    MemoryState,
    MemoryTier,
    MemoryUnit,
    Observation,
    ProceduralMemory,
    SourceType,
    EncodingContext,
)
from cortiloop.storage.base_store import BaseStore
from cortiloop.storage.vector_index import VectorIndex, create_vector_index

logger = logging.getLogger("cortiloop.store")


def _adapt_datetime(dt: datetime) -> str:
    return dt.isoformat()


def _convert_datetime(s: bytes) -> datetime:
    return datetime.fromisoformat(s.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)


class SQLiteStore(BaseStore):
    """Storage using SQLite for persistence + pluggable vector index for ANN search."""

    def __init__(self, config: CortiLoopConfig):
        self.config = config
        self.conn = sqlite3.connect(
            config.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

        # Vector indices (one per memory type)
        backend = config.vector_backend
        dim = config.llm.embedding_dim
        self._unit_index: VectorIndex = create_vector_index(dim, backend)
        self._obs_index: VectorIndex = create_vector_index(dim, backend)
        self._proc_index: VectorIndex = create_vector_index(dim, backend)
        self._build_indices()

    def _create_tables(self):
        ns = self.config.namespace
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS memory_units_{ns} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                importance_score REAL,
                encoding_context TEXT,
                entities TEXT,
                embedding BLOB,
                created_at timestamp,
                base_strength REAL DEFAULT 1.0,
                decay_rate REAL DEFAULT 0.1,
                last_accessed timestamp,
                access_count INTEGER DEFAULT 0,
                state TEXT DEFAULT 'active',
                tier TEXT DEFAULT 'episodic'
            );

            CREATE TABLE IF NOT EXISTS observations_{ns} (
                id TEXT PRIMARY KEY,
                dimension TEXT,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                version INTEGER DEFAULT 1,
                source_unit_ids TEXT,
                entities TEXT,
                embedding BLOB,
                created_at timestamp,
                updated_at timestamp,
                base_strength REAL DEFAULT 1.0,
                decay_rate REAL DEFAULT 0.03,
                last_accessed timestamp,
                access_count INTEGER DEFAULT 0,
                state TEXT DEFAULT 'active',
                history TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS procedural_memories_{ns} (
                id TEXT PRIMARY KEY,
                pattern TEXT,
                procedure TEXT,
                entities TEXT,
                acquisition_count INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.3,
                embedding BLOB,
                created_at timestamp,
                base_strength REAL DEFAULT 1.0,
                decay_rate REAL DEFAULT 0.005,
                last_accessed timestamp,
                access_count INTEGER DEFAULT 0,
                state TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS edges_{ns} (
                source_id TEXT,
                target_id TEXT,
                edge_type TEXT,
                weight REAL DEFAULT 1.0,
                co_activation_count INTEGER DEFAULT 1,
                last_co_activated timestamp,
                created_at timestamp,
                PRIMARY KEY (source_id, target_id, edge_type)
            );

            CREATE TABLE IF NOT EXISTS conflicts_{ns} (
                id TEXT PRIMARY KEY,
                old_memory_id TEXT,
                new_memory_id TEXT,
                dimension TEXT,
                old_value TEXT,
                new_value TEXT,
                resolution TEXT,
                created_at timestamp
            );

            CREATE INDEX IF NOT EXISTS idx_mu_state_{ns}
                ON memory_units_{ns}(state);
            CREATE INDEX IF NOT EXISTS idx_mu_entities_{ns}
                ON memory_units_{ns}(entities);
            CREATE INDEX IF NOT EXISTS idx_obs_state_{ns}
                ON observations_{ns}(state);
            CREATE INDEX IF NOT EXISTS idx_obs_dimension_{ns}
                ON observations_{ns}(dimension);
            CREATE INDEX IF NOT EXISTS idx_edges_source_{ns}
                ON edges_{ns}(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target_{ns}
                ON edges_{ns}(target_id);
        """)
        self.conn.commit()

    # ── helpers ──

    def _t(self, table: str) -> str:
        return f"{table}_{self.config.namespace}"

    @staticmethod
    def _encode_embedding(emb: list[float]) -> bytes | None:
        if not emb:
            return None
        return np.array(emb, dtype=np.float32).tobytes()

    @staticmethod
    def _decode_embedding(blob: bytes | None) -> list[float]:
        if not blob:
            return []
        return np.frombuffer(blob, dtype=np.float32).tolist()

    @staticmethod
    def _cosine_sim(a: list[float], b: bytes | None) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.frombuffer(b, dtype=np.float32)
        dot = np.dot(va, vb)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    def _build_indices(self):
        """Load existing embeddings into in-memory vector indices on startup."""
        ns = self.config.namespace
        # Units
        rows = self.conn.execute(
            f"SELECT id, embedding FROM memory_units_{ns} WHERE state='active' AND embedding IS NOT NULL"
        ).fetchall()
        items = []
        for row_id, blob in rows:
            if blob:
                items.append((row_id, self._decode_embedding(blob)))
        if items:
            self._unit_index.bulk_add(items)
            logger.info("Loaded %d unit vectors into index", len(items))

        # Observations
        rows = self.conn.execute(
            f"SELECT id, embedding FROM observations_{ns} WHERE state='active' AND embedding IS NOT NULL"
        ).fetchall()
        items = []
        for row_id, blob in rows:
            if blob:
                items.append((row_id, self._decode_embedding(blob)))
        if items:
            self._obs_index.bulk_add(items)
            logger.info("Loaded %d observation vectors into index", len(items))

        # Procedurals
        rows = self.conn.execute(
            f"SELECT id, embedding FROM procedural_memories_{ns} WHERE state='active' AND embedding IS NOT NULL"
        ).fetchall()
        items = []
        for row_id, blob in rows:
            if blob:
                items.append((row_id, self._decode_embedding(blob)))
        if items:
            self._proc_index.bulk_add(items)
            logger.info("Loaded %d procedural vectors into index", len(items))

    # ── MemoryUnit CRUD ──

    def insert_unit(self, unit: MemoryUnit):
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self._t('memory_units')} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                unit.id, unit.content, unit.source_type.value, unit.importance_score,
                json.dumps(unit.encoding_context.__dict__), json.dumps(unit.entities),
                self._encode_embedding(unit.embedding), unit.created_at,
                unit.base_strength, unit.decay_rate, unit.last_accessed,
                unit.access_count, unit.state.value, unit.tier.value,
            ),
        )
        self.conn.commit()
        # Update vector index
        if unit.embedding:
            self._unit_index.add(unit.id, unit.embedding)

    def get_unit(self, unit_id: str) -> MemoryUnit | None:
        row = self.conn.execute(
            f"SELECT * FROM {self._t('memory_units')} WHERE id=?", (unit_id,)
        ).fetchone()
        return self._row_to_unit(row) if row else None

    def get_active_units(self, limit: int = 1000) -> list[MemoryUnit]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('memory_units')} WHERE state='active' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def get_recent_units(self, limit: int = 50) -> list[MemoryUnit]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('memory_units')} WHERE state='active' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def search_units_by_vector(self, query_emb: list[float], top_k: int = 20) -> list[tuple[MemoryUnit, float]]:
        # Use vector index for fast ANN search
        hits = self._unit_index.search(query_emb, top_k)
        results = []
        for uid, sim in hits:
            unit = self.get_unit(uid)
            if unit and unit.state == MemoryState.ACTIVE:
                results.append((unit, sim))
        return results

    def search_units_by_keyword(self, keyword: str, limit: int = 20) -> list[MemoryUnit]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('memory_units')} WHERE state='active' AND content LIKE ? LIMIT ?",
            (f"%{keyword}%", limit),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def search_units_by_entity(self, entity: str, limit: int = 50) -> list[MemoryUnit]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('memory_units')} WHERE state='active' AND entities LIKE ? LIMIT ?",
            (f'%"{entity}"%', limit),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def update_unit_access(self, unit_id: str):
        now = datetime.now()
        self.conn.execute(
            f"UPDATE {self._t('memory_units')} SET last_accessed=?, access_count=access_count+1 WHERE id=?",
            (now, unit_id),
        )
        self.conn.commit()

    def update_unit_state(self, unit_id: str, state: MemoryState):
        self.conn.execute(
            f"UPDATE {self._t('memory_units')} SET state=? WHERE id=?",
            (state.value, unit_id),
        )
        self.conn.commit()
        # Remove from index if no longer active
        if state != MemoryState.ACTIVE:
            self._unit_index.remove(unit_id)

    def count_units(self) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {self._t('memory_units')}").fetchone()
        return row[0] if row else 0

    def _row_to_unit(self, r) -> MemoryUnit:
        ctx_dict = json.loads(r[4]) if r[4] else {}
        return MemoryUnit(
            id=r[0], content=r[1], source_type=SourceType(r[2]),
            importance_score=r[3], encoding_context=EncodingContext(**ctx_dict),
            entities=json.loads(r[5]) if r[5] else [],
            embedding=self._decode_embedding(r[6]),
            created_at=r[7], base_strength=r[8], decay_rate=r[9],
            last_accessed=r[10], access_count=r[11],
            state=MemoryState(r[12]), tier=MemoryTier(r[13]),
        )

    # ── Observation CRUD ──

    def insert_observation(self, obs: Observation):
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self._t('observations')} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                obs.id, obs.dimension, obs.content, obs.confidence, obs.version,
                json.dumps(obs.source_unit_ids), json.dumps(obs.entities),
                self._encode_embedding(obs.embedding),
                obs.created_at, obs.updated_at,
                obs.base_strength, obs.decay_rate, obs.last_accessed,
                obs.access_count, obs.state.value, json.dumps(obs.history),
            ),
        )
        self.conn.commit()
        if obs.embedding:
            self._obs_index.add(obs.id, obs.embedding)

    def get_observation(self, obs_id: str) -> Observation | None:
        row = self.conn.execute(
            f"SELECT * FROM {self._t('observations')} WHERE id=?", (obs_id,)
        ).fetchone()
        return self._row_to_observation(row) if row else None

    def get_active_observations(self, limit: int = 500) -> list[Observation]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('observations')} WHERE state='active' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def search_observations_by_vector(self, query_emb: list[float], top_k: int = 20) -> list[tuple[Observation, float]]:
        hits = self._obs_index.search(query_emb, top_k)
        results = []
        for oid, sim in hits:
            obs = self.get_observation(oid)
            if obs and obs.state == MemoryState.ACTIVE:
                results.append((obs, sim))
        return results

    def search_observations_by_dimension(self, dimension: str) -> list[Observation]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('observations')} WHERE state='active' AND dimension=?",
            (dimension,),
        ).fetchall()
        return [self._row_to_observation(r) for r in rows]

    def update_observation_access(self, obs_id: str):
        now = datetime.now()
        self.conn.execute(
            f"UPDATE {self._t('observations')} SET last_accessed=?, access_count=access_count+1 WHERE id=?",
            (now, obs_id),
        )
        self.conn.commit()

    def count_observations(self) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {self._t('observations')}").fetchone()
        return row[0] if row else 0

    def _row_to_observation(self, r) -> Observation:
        return Observation(
            id=r[0], dimension=r[1], content=r[2], confidence=r[3], version=r[4],
            source_unit_ids=json.loads(r[5]) if r[5] else [],
            entities=json.loads(r[6]) if r[6] else [],
            embedding=self._decode_embedding(r[7]),
            created_at=r[8], updated_at=r[9],
            base_strength=r[10], decay_rate=r[11], last_accessed=r[12],
            access_count=r[13], state=MemoryState(r[14]),
            history=json.loads(r[15]) if r[15] else [],
        )

    # ── ProceduralMemory CRUD ──

    def insert_procedural(self, pm: ProceduralMemory):
        self.conn.execute(
            f"INSERT OR REPLACE INTO {self._t('procedural_memories')} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                pm.id, pm.pattern, pm.procedure, json.dumps(pm.entities),
                pm.acquisition_count, pm.confidence,
                self._encode_embedding(pm.embedding),
                pm.created_at, pm.base_strength, pm.decay_rate,
                pm.last_accessed, pm.access_count, pm.state.value,
            ),
        )
        self.conn.commit()
        if pm.embedding:
            self._proc_index.add(pm.id, pm.embedding)

    def get_active_procedurals(self, limit: int = 100) -> list[ProceduralMemory]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('procedural_memories')} WHERE state='active' ORDER BY confidence DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_procedural(r) for r in rows]

    def search_procedurals_by_vector(self, query_emb: list[float], top_k: int = 5) -> list[tuple[ProceduralMemory, float]]:
        hits = self._proc_index.search(query_emb, top_k)
        results = []
        for pid, sim in hits:
            pm = self._get_procedural(pid)
            if pm and pm.state == MemoryState.ACTIVE:
                results.append((pm, sim))
        return results

    def _get_procedural(self, proc_id: str) -> ProceduralMemory | None:
        row = self.conn.execute(
            f"SELECT * FROM {self._t('procedural_memories')} WHERE id=?", (proc_id,)
        ).fetchone()
        return self._row_to_procedural(row) if row else None

    def _row_to_procedural(self, r) -> ProceduralMemory:
        return ProceduralMemory(
            id=r[0], pattern=r[1], procedure=r[2],
            entities=json.loads(r[3]) if r[3] else [],
            acquisition_count=r[4], confidence=r[5],
            embedding=self._decode_embedding(r[6]),
            created_at=r[7], base_strength=r[8], decay_rate=r[9],
            last_accessed=r[10], access_count=r[11],
            state=MemoryState(r[12]),
        )

    # ── Edge CRUD ──

    def upsert_edge(self, edge: MemoryEdge):
        self.conn.execute(
            f"""INSERT INTO {self._t('edges')} VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(source_id, target_id, edge_type) DO UPDATE SET
                    weight=excluded.weight,
                    co_activation_count=excluded.co_activation_count,
                    last_co_activated=excluded.last_co_activated""",
            (
                edge.source_id, edge.target_id, edge.edge_type.value,
                edge.weight, edge.co_activation_count,
                edge.last_co_activated, edge.created_at,
            ),
        )
        self.conn.commit()

    def get_edges_from(self, source_id: str) -> list[MemoryEdge]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('edges')} WHERE source_id=? ORDER BY weight DESC",
            (source_id,),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(self, target_id: str) -> list[MemoryEdge]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t('edges')} WHERE target_id=? ORDER BY weight DESC",
            (target_id,),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> MemoryEdge | None:
        row = self.conn.execute(
            f"SELECT * FROM {self._t('edges')} WHERE source_id=? AND target_id=? AND edge_type=?",
            (source_id, target_id, edge_type.value),
        ).fetchone()
        return self._row_to_edge(row) if row else None

    def _row_to_edge(self, r) -> MemoryEdge:
        return MemoryEdge(
            source_id=r[0], target_id=r[1], edge_type=EdgeType(r[2]),
            weight=r[3], co_activation_count=r[4],
            last_co_activated=r[5], created_at=r[6],
        )

    # ── Conflict CRUD ──

    def insert_conflict(self, conflict: ConflictRecord):
        self.conn.execute(
            f"INSERT INTO {self._t('conflicts')} VALUES (?,?,?,?,?,?,?,?)",
            (
                conflict.id, conflict.old_memory_id, conflict.new_memory_id,
                conflict.dimension, conflict.old_value, conflict.new_value,
                conflict.resolution, conflict.created_at,
            ),
        )
        self.conn.commit()

    # ── Bulk / Maintenance ──

    def get_all_active_units_for_decay(self) -> list[tuple[str, float, float, str, int]]:
        """Returns (id, decay_rate, base_strength, last_accessed_iso, access_count) for active units."""
        rows = self.conn.execute(
            f"SELECT id, decay_rate, base_strength, last_accessed, access_count FROM {self._t('memory_units')} WHERE state='active'"
        ).fetchall()
        return rows

    def get_all_active_observations_for_decay(self) -> list[tuple[str, float, float, str, int]]:
        rows = self.conn.execute(
            f"SELECT id, decay_rate, base_strength, last_accessed, access_count FROM {self._t('observations')} WHERE state='active'"
        ).fetchall()
        return rows

    def close(self):
        self.conn.close()
