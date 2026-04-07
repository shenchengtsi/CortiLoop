"""Tests for the Ebbinghaus decay model."""

import pytest
from datetime import datetime, timedelta

from cortiloop.config import DecayConfig
from cortiloop.forgetting.decay import DecayManager
from cortiloop.models import MemoryState


@pytest.fixture
def decay():
    config = DecayConfig()
    return DecayManager(config, None)  # store not needed for compute


def test_fresh_memory_full_strength(decay):
    now = datetime.utcnow()
    strength = decay.compute_strength(1.0, 0.1, now, 0, now)
    assert abs(strength - 1.0) < 0.01


def test_decay_after_one_day(decay):
    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)
    strength = decay.compute_strength(1.0, 0.1, yesterday, 0, now)
    # e^(-0.1 * 1) ≈ 0.905
    assert 0.85 < strength < 0.95


def test_decay_after_one_week(decay):
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    strength = decay.compute_strength(1.0, 0.1, week_ago, 0, now)
    # e^(-0.1 * 7) ≈ 0.497
    assert 0.4 < strength < 0.55


def test_access_count_boosts_strength(decay):
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    # No accesses
    s0 = decay.compute_strength(1.0, 0.1, week_ago, 0, now)
    # 10 accesses
    s10 = decay.compute_strength(1.0, 0.1, week_ago, 10, now)
    assert s10 > s0


def test_procedural_decays_slowly(decay):
    now = datetime.utcnow()
    month_ago = now - timedelta(days=30)
    # Episodic rate
    s_episodic = decay.compute_strength(1.0, 0.1, month_ago, 0, now)
    # Procedural rate
    s_procedural = decay.compute_strength(1.0, 0.005, month_ago, 0, now)
    assert s_procedural > s_episodic


def test_state_evaluation(decay):
    assert decay.evaluate_state(0.8) == MemoryState.ACTIVE
    assert decay.evaluate_state(0.2) == MemoryState.ARCHIVE
    assert decay.evaluate_state(0.05) == MemoryState.COLD
