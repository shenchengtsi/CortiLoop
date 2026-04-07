"""
Background consolidation worker — asyncio-based periodic "sleep" cycle.

Brain analogy: The brain consolidates memories during sleep via
hippocampus→neocortex replay. This worker runs periodic consolidation
cycles (systems consolidation + decay + pruning) in the background.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortiloop.engine import CortiLoop

logger = logging.getLogger("cortiloop.worker")


class ConsolidationWorker:
    """
    Async background worker that periodically runs reflect().

    Usage:
        worker = ConsolidationWorker(engine, interval_seconds=3600)
        worker.start()   # non-blocking, creates background task
        ...
        await worker.stop()  # graceful shutdown
    """

    def __init__(self, engine: CortiLoop, interval_seconds: int = 3600):
        self.engine = engine
        self.interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the background consolidation loop."""
        if self.is_running:
            logger.warning("Consolidation worker already running")
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())
        logger.info(
            "Consolidation worker started (interval=%ds)", self.interval
        )

    async def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Consolidation worker stopped")

    async def _run_loop(self) -> None:
        """Main loop: sleep → reflect → repeat."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            logger.info("Consolidation cycle starting...")
            try:
                result = await self.engine.reflect()
                logger.info("Consolidation cycle complete: %s", result)
            except Exception as e:
                logger.error("Consolidation cycle failed: %s", e)

    async def run_once(self) -> dict:
        """Run a single consolidation cycle immediately (for testing / manual trigger)."""
        logger.info("Manual consolidation cycle triggered")
        return await self.engine.reflect()
