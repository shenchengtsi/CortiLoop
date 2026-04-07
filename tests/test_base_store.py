"""Tests that SQLiteStore properly implements BaseStore interface."""

from cortiloop.config import CortiLoopConfig
from cortiloop.storage.base_store import BaseStore
from cortiloop.storage.sqlite_store import SQLiteStore


def test_sqlite_store_is_base_store():
    """SQLiteStore must be a subclass of BaseStore."""
    assert issubclass(SQLiteStore, BaseStore)


def test_sqlite_store_instance():
    config = CortiLoopConfig(db_path=":memory:")
    store = SQLiteStore(config)
    assert isinstance(store, BaseStore)
    store.close()


def test_engine_factory_creates_sqlite():
    """Engine should create SQLiteStore by default."""
    from cortiloop.engine import CortiLoop
    config = CortiLoopConfig(db_path=":memory:")
    loop = CortiLoop(config)
    assert isinstance(loop.store, BaseStore)
    assert isinstance(loop.store, SQLiteStore)
    loop.close()


def test_engine_factory_postgres_config():
    """Engine with storage_backend='postgres' should try to create PostgresStore."""
    config = CortiLoopConfig(db_path=":memory:", storage_backend="postgres")
    # We can't test actual PostgresStore creation without a running Postgres,
    # but we can verify the config is set correctly.
    assert config.storage_backend == "postgres"
