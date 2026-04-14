"""
PostgreSQL storage backend with pgvector for native vector similarity search.

Requires:
  pip install psycopg[binary] pgvector
  PostgreSQL with pgvector extension enabled.

Advantages over SQLite:
- Native HNSW/IVFFlat vector index via pgvector (millions of vectors)
- Concurrent read/write (no single-writer lock)
- Production-grade: replication, backup, monitoring
- Full-text search via tsvector (better than LIKE)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

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

logger = logging.getLogger("cortiloop.postgres")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PostgresStore(BaseStore):
    """
    PostgreSQL + pgvector storage backend.

    Uses native vector similarity operators for ANN search,
    eliminating the need for in-memory indices.
    """

    def __init__(self, config: CortiLoopConfig):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError:
            raise ImportError(
                "PostgreSQL backend requires psycopg and pgvector. "
                "Install with: pip install 'psycopg[binary]' pgvector"
            )

        self.config = config
        self._dim = config.llm.embedding_dim
        self._ns = config.namespace

        self.conn = psycopg.connect(config.db_path)  # db_path = connection string
        self.conn.autocommit = True
        register_vector(self.conn)

        self._ensure_extension()
        self._create_tables()

    def _ensure_extension(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def _create_tables(self):
        ns = self._ns
        dim = self._dim
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS memory_units_{ns} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    importance_score REAL,
                    encoding_context JSONB,
                    entities JSONB,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ,
                    session_timestamp TIMESTAMPTZ,
                    base_strength REAL DEFAULT 1.0,
                    decay_rate REAL DEFAULT 0.1,
                    last_accessed TIMESTAMPTZ,
                    access_count INTEGER DEFAULT 0,
                    state TEXT DEFAULT 'active',
                    tier TEXT DEFAULT 'episodic'
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS observations_{ns} (
                    id TEXT PRIMARY KEY,
                    dimension TEXT,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    version INTEGER DEFAULT 1,
                    source_unit_ids JSONB,
                    entities JSONB,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ,
                    session_timestamp TIMESTAMPTZ,
                    base_strength REAL DEFAULT 1.0,
                    decay_rate REAL DEFAULT 0.03,
                    last_accessed TIMESTAMPTZ,
                    access_count INTEGER DEFAULT 0,
                    state TEXT DEFAULT 'active',
                    history JSONB DEFAULT '[]'::jsonb
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS procedural_memories_{ns} (
                    id TEXT PRIMARY KEY,
                    pattern TEXT,
                    procedure TEXT,
                    entities JSONB,
                    acquisition_count INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.3,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ,
                    base_strength REAL DEFAULT 1.0,
                    decay_rate REAL DEFAULT 0.005,
                    last_accessed TIMESTAMPTZ,
                    access_count INTEGER DEFAULT 0,
                    state TEXT DEFAULT 'active'
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS edges_{ns} (
                    source_id TEXT,
                    target_id TEXT,
                    edge_type TEXT,
                    weight REAL DEFAULT 1.0,
                    co_activation_count INTEGER DEFAULT 1,
                    last_co_activated TIMESTAMPTZ,
                    created_at TIMESTAMPTZ,
                    PRIMARY KEY (source_id, target_id, edge_type)
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS conflicts_{ns} (
                    id TEXT PRIMARY KEY,
                    old_memory_id TEXT,
                    new_memory_id TEXT,
                    dimension TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    resolution TEXT,
                    created_at TIMESTAMPTZ
                )
            """)

            # Indices
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_mu_state_{ns} ON memory_units_{ns}(state)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_mu_entities_{ns} ON memory_units_{ns} USING gin(entities)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_obs_state_{ns} ON observations_{ns}(state)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_obs_dim_{ns} ON observations_{ns}(dimension)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_edges_src_{ns} ON edges_{ns}(source_id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_edges_tgt_{ns} ON edges_{ns}(target_id)"
            )

            # pgvector HNSW indices for fast ANN search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_mu_vec_{ns}
                ON memory_units_{ns} USING hnsw (embedding vector_cosine_ops)
                WHERE state = 'active'
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_obs_vec_{ns}
                ON observations_{ns} USING hnsw (embedding vector_cosine_ops)
                WHERE state = 'active'
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_proc_vec_{ns}
                ON procedural_memories_{ns} USING hnsw (embedding vector_cosine_ops)
                WHERE state = 'active'
            """)

        # ── Schema migration: add session_timestamp to existing tables ──
        self._migrate_add_column(
            f"memory_units_{ns}", "session_timestamp", "TIMESTAMPTZ"
        )
        self._migrate_add_column(
            f"observations_{ns}", "session_timestamp", "TIMESTAMPTZ"
        )

    def _migrate_add_column(self, table: str, column: str, col_type: str):
        """Add a column to an existing table if it doesn't exist yet."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
                (table, column),
            )
            if not cur.fetchone():
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                logger.info("Migrated %s: added column %s", table, column)

    def _t(self, table: str) -> str:
        return f"{table}_{self._ns}"

    # Canonical column order — all SELECTs use these instead of SELECT *
    _UNIT_COLS = (
        "id, content, source_type, importance_score, encoding_context, "
        "entities, embedding, created_at, session_timestamp, base_strength, "
        "decay_rate, last_accessed, access_count, state, tier"
    )
    _OBS_COLS = (
        "id, dimension, content, confidence, version, source_unit_ids, "
        "entities, embedding, created_at, updated_at, session_timestamp, "
        "base_strength, decay_rate, last_accessed, access_count, state, history"
    )
    _PROC_COLS = (
        "id, pattern, procedure, entities, acquisition_count, confidence, "
        "embedding, created_at, base_strength, decay_rate, last_accessed, "
        "access_count, state"
    )
    _EDGE_COLS = (
        "source_id, target_id, edge_type, weight, co_activation_count, "
        "last_co_activated, created_at"
    )

    # ── MemoryUnit CRUD ──

    def insert_unit(self, unit: MemoryUnit) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {self._t("memory_units")}
                    (id, content, source_type, importance_score, encoding_context,
                     entities, embedding, created_at, session_timestamp, base_strength,
                     decay_rate, last_accessed, access_count, state, tier)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        content=EXCLUDED.content, embedding=EXCLUDED.embedding,
                        state=EXCLUDED.state""",
                (
                    unit.id,
                    unit.content,
                    unit.source_type.value,
                    unit.importance_score,
                    json.dumps(unit.encoding_context.__dict__),
                    json.dumps(unit.entities),
                    unit.embedding if unit.embedding else None,
                    unit.created_at,
                    unit.session_timestamp,
                    unit.base_strength,
                    unit.decay_rate,
                    unit.last_accessed,
                    unit.access_count,
                    unit.state.value,
                    unit.tier.value,
                ),
            )

    def get_unit(self, unit_id: str) -> MemoryUnit | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._UNIT_COLS} FROM {self._t('memory_units')} WHERE id=%s",
                (unit_id,),
            )
            row = cur.fetchone()
            return self._row_to_unit(row) if row else None

    def get_active_units(self, limit: int = 1000) -> list[MemoryUnit]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._UNIT_COLS} FROM {self._t('memory_units')} WHERE state='active' ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [self._row_to_unit(r) for r in cur.fetchall()]

    def get_recent_units(self, limit: int = 50) -> list[MemoryUnit]:
        return self.get_active_units(limit)

    def search_units_by_vector(
        self, query_emb: list[float], top_k: int = 20
    ) -> list[tuple[MemoryUnit, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT {self._UNIT_COLS}, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self._t("memory_units")}
                    WHERE state='active' AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (query_emb, query_emb, top_k),
            )
            results = []
            for row in cur.fetchall():
                sim = row[-1]  # similarity is last column
                unit = self._row_to_unit(row[:-1])
                results.append((unit, float(sim)))
            return results

    def search_units_by_keyword(
        self, keyword: str, limit: int = 20
    ) -> list[MemoryUnit]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._UNIT_COLS} FROM {self._t('memory_units')} WHERE state='active' AND content ILIKE %s LIMIT %s",
                (f"%{keyword}%", limit),
            )
            return [self._row_to_unit(r) for r in cur.fetchall()]

    def search_units_by_entity(self, entity: str, limit: int = 50) -> list[MemoryUnit]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._UNIT_COLS} FROM {self._t('memory_units')} WHERE state='active' AND entities ? %s LIMIT %s",
                (entity, limit),
            )
            return [self._row_to_unit(r) for r in cur.fetchall()]

    def update_unit_access(self, unit_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"UPDATE {self._t('memory_units')} SET last_accessed=%s, access_count=access_count+1 WHERE id=%s",
                (_utcnow(), unit_id),
            )

    def update_unit_state(self, unit_id: str, state: MemoryState) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"UPDATE {self._t('memory_units')} SET state=%s WHERE id=%s",
                (state.value, unit_id),
            )

    def count_units(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._t('memory_units')}")
            return cur.fetchone()[0]

    def _row_to_unit(self, r) -> MemoryUnit:
        ctx_raw = r[4]
        ctx_dict = json.loads(ctx_raw) if isinstance(ctx_raw, str) else (ctx_raw or {})
        entities_raw = r[5]
        entities = (
            json.loads(entities_raw)
            if isinstance(entities_raw, str)
            else (entities_raw or [])
        )
        embedding = list(r[6]) if r[6] is not None else []
        return MemoryUnit(
            id=r[0],
            content=r[1],
            source_type=SourceType(r[2]),
            importance_score=r[3],
            encoding_context=EncodingContext(**ctx_dict),
            entities=entities,
            embedding=embedding,
            created_at=r[7],
            session_timestamp=r[8],
            base_strength=r[9],
            decay_rate=r[10],
            last_accessed=r[11],
            access_count=r[12],
            state=MemoryState(r[13]),
            tier=MemoryTier(r[14]),
        )

    # ── Observation CRUD ──

    def insert_observation(self, obs: Observation) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {self._t("observations")}
                    (id, dimension, content, confidence, version, source_unit_ids,
                     entities, embedding, created_at, updated_at, session_timestamp,
                     base_strength, decay_rate, last_accessed, access_count, state, history)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        content=EXCLUDED.content, confidence=EXCLUDED.confidence,
                        version=EXCLUDED.version, embedding=EXCLUDED.embedding,
                        updated_at=EXCLUDED.updated_at, session_timestamp=EXCLUDED.session_timestamp,
                        history=EXCLUDED.history, state=EXCLUDED.state""",
                (
                    obs.id,
                    obs.dimension,
                    obs.content,
                    obs.confidence,
                    obs.version,
                    json.dumps(obs.source_unit_ids),
                    json.dumps(obs.entities),
                    obs.embedding if obs.embedding else None,
                    obs.created_at,
                    obs.updated_at,
                    obs.session_timestamp,
                    obs.base_strength,
                    obs.decay_rate,
                    obs.last_accessed,
                    obs.access_count,
                    obs.state.value,
                    json.dumps(obs.history),
                ),
            )

    def get_observation(self, obs_id: str) -> Observation | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._OBS_COLS} FROM {self._t('observations')} WHERE id=%s",
                (obs_id,),
            )
            row = cur.fetchone()
            return self._row_to_observation(row) if row else None

    def get_active_observations(self, limit: int = 500) -> list[Observation]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._OBS_COLS} FROM {self._t('observations')} WHERE state='active' ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
            return [self._row_to_observation(r) for r in cur.fetchall()]

    def search_observations_by_vector(
        self, query_emb: list[float], top_k: int = 20
    ) -> list[tuple[Observation, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT {self._OBS_COLS}, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self._t("observations")}
                    WHERE state='active' AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (query_emb, query_emb, top_k),
            )
            results = []
            for row in cur.fetchall():
                sim = row[-1]
                obs = self._row_to_observation(row[:-1])
                results.append((obs, float(sim)))
            return results

    def search_observations_by_dimension(self, dimension: str) -> list[Observation]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._OBS_COLS} FROM {self._t('observations')} WHERE state='active' AND dimension=%s",
                (dimension,),
            )
            return [self._row_to_observation(r) for r in cur.fetchall()]

    def update_observation_access(self, obs_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"UPDATE {self._t('observations')} SET last_accessed=%s, access_count=access_count+1 WHERE id=%s",
                (_utcnow(), obs_id),
            )

    def count_observations(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._t('observations')}")
            return cur.fetchone()[0]

    def _row_to_observation(self, r) -> Observation:
        source_ids = json.loads(r[5]) if isinstance(r[5], str) else (r[5] or [])
        entities = json.loads(r[6]) if isinstance(r[6], str) else (r[6] or [])
        embedding = list(r[7]) if r[7] is not None else []
        history = json.loads(r[16]) if isinstance(r[16], str) else (r[16] or [])
        return Observation(
            id=r[0],
            dimension=r[1],
            content=r[2],
            confidence=r[3],
            version=r[4],
            source_unit_ids=source_ids,
            entities=entities,
            embedding=embedding,
            created_at=r[8],
            updated_at=r[9],
            session_timestamp=r[10],
            base_strength=r[11],
            decay_rate=r[12],
            last_accessed=r[13],
            access_count=r[14],
            state=MemoryState(r[15]),
            history=history,
        )

    # ── ProceduralMemory CRUD ──

    def insert_procedural(self, pm: ProceduralMemory) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {self._t("procedural_memories")}
                    (id, pattern, procedure, entities, acquisition_count, confidence,
                     embedding, created_at, base_strength, decay_rate,
                     last_accessed, access_count, state)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        confidence=EXCLUDED.confidence,
                        acquisition_count=EXCLUDED.acquisition_count,
                        embedding=EXCLUDED.embedding""",
                (
                    pm.id,
                    pm.pattern,
                    pm.procedure,
                    json.dumps(pm.entities),
                    pm.acquisition_count,
                    pm.confidence,
                    pm.embedding if pm.embedding else None,
                    pm.created_at,
                    pm.base_strength,
                    pm.decay_rate,
                    pm.last_accessed,
                    pm.access_count,
                    pm.state.value,
                ),
            )

    def get_active_procedurals(self, limit: int = 100) -> list[ProceduralMemory]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._PROC_COLS} FROM {self._t('procedural_memories')} WHERE state='active' ORDER BY confidence DESC LIMIT %s",
                (limit,),
            )
            return [self._row_to_procedural(r) for r in cur.fetchall()]

    def search_procedurals_by_vector(
        self, query_emb: list[float], top_k: int = 5
    ) -> list[tuple[ProceduralMemory, float]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT {self._PROC_COLS}, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self._t("procedural_memories")}
                    WHERE state='active' AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (query_emb, query_emb, top_k),
            )
            results = []
            for row in cur.fetchall():
                sim = row[-1]
                pm = self._row_to_procedural(row[:-1])
                results.append((pm, float(sim)))
            return results

    def _row_to_procedural(self, r) -> ProceduralMemory:
        entities = json.loads(r[3]) if isinstance(r[3], str) else (r[3] or [])
        embedding = list(r[6]) if r[6] is not None else []
        return ProceduralMemory(
            id=r[0],
            pattern=r[1],
            procedure=r[2],
            entities=entities,
            acquisition_count=r[4],
            confidence=r[5],
            embedding=embedding,
            created_at=r[7],
            base_strength=r[8],
            decay_rate=r[9],
            last_accessed=r[10],
            access_count=r[11],
            state=MemoryState(r[12]),
        )

    # ── Edge CRUD ──

    def upsert_edge(self, edge: MemoryEdge) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {self._t("edges")}
                    (source_id, target_id, edge_type, weight, co_activation_count,
                     last_co_activated, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                        weight=EXCLUDED.weight,
                        co_activation_count=EXCLUDED.co_activation_count,
                        last_co_activated=EXCLUDED.last_co_activated""",
                (
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type.value,
                    edge.weight,
                    edge.co_activation_count,
                    edge.last_co_activated,
                    edge.created_at,
                ),
            )

    def get_edges_from(self, source_id: str) -> list[MemoryEdge]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._EDGE_COLS} FROM {self._t('edges')} WHERE source_id=%s ORDER BY weight DESC",
                (source_id,),
            )
            return [self._row_to_edge(r) for r in cur.fetchall()]

    def get_edges_to(self, target_id: str) -> list[MemoryEdge]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._EDGE_COLS} FROM {self._t('edges')} WHERE target_id=%s ORDER BY weight DESC",
                (target_id,),
            )
            return [self._row_to_edge(r) for r in cur.fetchall()]

    def get_edge(
        self, source_id: str, target_id: str, edge_type: EdgeType
    ) -> MemoryEdge | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._EDGE_COLS} FROM {self._t('edges')} WHERE source_id=%s AND target_id=%s AND edge_type=%s",
                (source_id, target_id, edge_type.value),
            )
            row = cur.fetchone()
            return self._row_to_edge(row) if row else None

    def _row_to_edge(self, r) -> MemoryEdge:
        return MemoryEdge(
            source_id=r[0],
            target_id=r[1],
            edge_type=EdgeType(r[2]),
            weight=r[3],
            co_activation_count=r[4],
            last_co_activated=r[5],
            created_at=r[6],
        )

    # ── Conflict CRUD ──

    def insert_conflict(self, conflict: ConflictRecord) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._t('conflicts')} VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    conflict.id,
                    conflict.old_memory_id,
                    conflict.new_memory_id,
                    conflict.dimension,
                    conflict.old_value,
                    conflict.new_value,
                    conflict.resolution,
                    conflict.created_at,
                ),
            )

    # ── Bulk / Maintenance ──

    def get_all_active_units_for_decay(
        self,
    ) -> list[tuple[str, float, float, Any, int]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, decay_rate, base_strength, last_accessed, access_count FROM {self._t('memory_units')} WHERE state='active'"
            )
            return cur.fetchall()

    def get_all_active_observations_for_decay(
        self,
    ) -> list[tuple[str, float, float, Any, int]]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, decay_rate, base_strength, last_accessed, access_count FROM {self._t('observations')} WHERE state='active'"
            )
            return cur.fetchall()

    def close(self) -> None:
        self.conn.close()
