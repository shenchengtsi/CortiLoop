"""Tests for the pluggable vector index abstraction."""

import numpy as np
import pytest

from cortiloop.storage.vector_index import NumpyIndex, create_vector_index


def _random_vec(dim=128, seed=None):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_add_and_search():
    idx = NumpyIndex()
    v1 = _random_vec(seed=1)
    v2 = _random_vec(seed=2)
    idx.add("a", v1)
    idx.add("b", v2)

    results = idx.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"  # v1 should be most similar to itself
    assert results[0][1] > 0.99


def test_remove():
    idx = NumpyIndex()
    idx.add("a", _random_vec(seed=1))
    idx.add("b", _random_vec(seed=2))
    assert idx.count() == 2

    idx.remove("a")
    assert idx.count() == 1

    results = idx.search(_random_vec(seed=1), top_k=5)
    assert len(results) == 1
    assert results[0][0] == "b"


def test_update_existing_key():
    idx = NumpyIndex()
    v1 = _random_vec(seed=1)
    v2 = _random_vec(seed=2)

    idx.add("a", v1)
    idx.add("a", v2)  # update same key
    assert idx.count() == 1

    results = idx.search(v2, top_k=1)
    assert results[0][0] == "a"
    assert results[0][1] > 0.99


def test_clear():
    idx = NumpyIndex()
    for i in range(10):
        idx.add(f"v{i}", _random_vec(seed=i))
    assert idx.count() == 10

    idx.clear()
    assert idx.count() == 0
    assert idx.search(_random_vec(seed=0), top_k=5) == []


def test_bulk_add():
    idx = NumpyIndex()
    items = [(f"v{i}", _random_vec(seed=i)) for i in range(50)]
    idx.bulk_add(items)
    assert idx.count() == 50

    results = idx.search(_random_vec(seed=0), top_k=3)
    assert len(results) == 3
    assert results[0][0] == "v0"


def test_empty_search():
    idx = NumpyIndex()
    results = idx.search(_random_vec(seed=1), top_k=5)
    assert results == []


def test_create_factory_numpy():
    idx = create_vector_index(dim=128, backend="numpy")
    assert isinstance(idx, NumpyIndex)


def test_create_factory_auto_fallback():
    """auto should fall back to numpy when usearch is not installed."""
    idx = create_vector_index(dim=128, backend="auto")
    # Should be either NumpyIndex or UsearchIndex depending on environment
    assert idx.count() == 0


def test_search_accuracy():
    """Verify that nearest neighbor search returns correct ranking."""
    idx = NumpyIndex()
    base = _random_vec(seed=42)

    # Create vectors at known distances
    base_np = np.array(base, dtype=np.float32)
    close = (base_np + 0.1 * np.random.RandomState(1).randn(128).astype(np.float32))
    close = (close / np.linalg.norm(close)).tolist()
    far = _random_vec(seed=99)

    idx.add("close", close)
    idx.add("far", far)

    results = idx.search(base, top_k=2)
    assert results[0][0] == "close"  # close vector should rank first
