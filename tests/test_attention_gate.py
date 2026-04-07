"""Tests for the attention gate (importance scoring)."""

import pytest
from unittest.mock import AsyncMock

from cortiloop.config import AttentionGateConfig
from cortiloop.encoding.attention_gate import AttentionGate


@pytest.fixture
def gate():
    config = AttentionGateConfig()
    llm = AsyncMock()
    return AttentionGate(config, llm)


@pytest.mark.asyncio
async def test_correction_gets_high_score(gate):
    score = await gate.score("不对，应该是 TypeScript 而不是 JavaScript")
    assert score > 0.5


@pytest.mark.asyncio
async def test_explicit_request_gets_high_score(gate):
    score = await gate.score("记住：我的 API key 在 .env 文件里")
    assert score > 0.4


@pytest.mark.asyncio
async def test_greeting_gets_low_score(gate):
    score = await gate.score("你好")
    assert score < 0.3


@pytest.mark.asyncio
async def test_ok_gets_low_score(gate):
    score = await gate.score("ok")
    assert score < 0.3


@pytest.mark.asyncio
async def test_substantive_content_passes(gate):
    score = await gate.score(
        "我们的 React 项目使用 TypeScript strict mode，数据库是 PostgreSQL 14",
        existing_entity_count=5,
    )
    assert gate.passes(score)


@pytest.mark.asyncio
async def test_disabled_gate_passes_all(gate):
    gate.config.enabled = False
    score = await gate.score("ok")
    assert score == 1.0
    assert gate.passes(score)
