"""Tests for the SQLite storage backend."""

import os
import pytest
import tempfile
from datetime import datetime

from cortiloop.config import CortiLoopConfig
from cortiloop.models import (
    EdgeType, EncodingContext, MemoryEdge, MemoryState, MemoryUnit, Observation, SourceType,
)
from cortiloop.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    config = CortiLoopConfig(db_path=tmp.name, namespace="test")
    s = SQLiteStore(config)
    yield s
    s.close()
    os.unlink(tmp.name)


def test_insert_and_get_unit(store):
    unit = MemoryUnit(
        content="Alice is a PM",
        source_type=SourceType.USER_SAID,
        importance_score=0.8,
        entities=["Alice"],
    )
    store.insert_unit(unit)
    got = store.get_unit(unit.id)
    assert got is not None
    assert got.content == "Alice is a PM"
    assert got.entities == ["Alice"]


def test_keyword_search(store):
    store.insert_unit(MemoryUnit(content="React project uses TypeScript", entities=["React"]))
    store.insert_unit(MemoryUnit(content="Python is great", entities=["Python"]))
    results = store.search_units_by_keyword("TypeScript")
    assert len(results) == 1
    assert "TypeScript" in results[0].content


def test_entity_search(store):
    store.insert_unit(MemoryUnit(content="Alice manages ProjectX", entities=["Alice", "ProjectX"]))
    store.insert_unit(MemoryUnit(content="Bob writes code", entities=["Bob"]))
    results = store.search_units_by_entity("Alice")
    assert len(results) == 1


def test_unit_state_update(store):
    unit = MemoryUnit(content="test", entities=[])
    store.insert_unit(unit)
    store.update_unit_state(unit.id, MemoryState.ARCHIVE)
    got = store.get_unit(unit.id)
    assert got.state == MemoryState.ARCHIVE


def test_observation_crud(store):
    obs = Observation(
        dimension="Alice:role",
        content="Alice is the PM of ProjectX",
        entities=["Alice", "ProjectX"],
    )
    store.insert_observation(obs)
    got = store.get_observation(obs.id)
    assert got is not None
    assert got.dimension == "Alice:role"


def test_edge_upsert_and_query(store):
    edge = MemoryEdge(
        source_id="a", target_id="b",
        edge_type=EdgeType.CO_OCCURRENCE, weight=1.0,
    )
    store.upsert_edge(edge)
    edges = store.get_edges_from("a")
    assert len(edges) == 1
    assert edges[0].target_id == "b"

    # Upsert again with higher weight
    edge.weight = 2.0
    store.upsert_edge(edge)
    edges = store.get_edges_from("a")
    assert edges[0].weight == 2.0


def test_count(store):
    assert store.count_units() == 0
    store.insert_unit(MemoryUnit(content="a", entities=[]))
    store.insert_unit(MemoryUnit(content="b", entities=[]))
    assert store.count_units() == 2
