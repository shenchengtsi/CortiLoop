"""Tests for the background consolidation worker."""

import asyncio
import pytest

from cortiloop.workers.consolidation_worker import ConsolidationWorker


class MockEngine:
    """Minimal mock of CortiLoop for worker testing."""

    def __init__(self):
        self.reflect_count = 0

    async def reflect(self):
        self.reflect_count += 1
        return {"consolidation": "ok", "decay_sweep": "ok", "pruning": "ok"}


@pytest.mark.asyncio
async def test_worker_start_stop():
    engine = MockEngine()
    worker = ConsolidationWorker(engine, interval_seconds=3600)

    assert not worker.is_running
    worker.start()
    assert worker.is_running

    await worker.stop()
    assert not worker.is_running


@pytest.mark.asyncio
async def test_worker_run_once():
    engine = MockEngine()
    worker = ConsolidationWorker(engine, interval_seconds=3600)

    result = await worker.run_once()
    assert result["consolidation"] == "ok"
    assert engine.reflect_count == 1


@pytest.mark.asyncio
async def test_worker_periodic_execution():
    """Worker should call reflect() after each interval."""
    engine = MockEngine()
    worker = ConsolidationWorker(engine, interval_seconds=1)  # 1 second for test

    worker.start()
    await asyncio.sleep(1.5)  # wait for at least one cycle
    await worker.stop()

    assert engine.reflect_count >= 1


@pytest.mark.asyncio
async def test_worker_double_start():
    engine = MockEngine()
    worker = ConsolidationWorker(engine, interval_seconds=3600)

    worker.start()
    worker.start()  # should not error
    assert worker.is_running

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_stop_without_start():
    engine = MockEngine()
    worker = ConsolidationWorker(engine, interval_seconds=3600)

    await worker.stop()  # should not error
    assert not worker.is_running
