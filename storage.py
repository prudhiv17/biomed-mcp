import os
import sqlite3
import threading
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent

# Anchored to the project root: a bare load_dotenv() searches upward from the
# working directory, which is the client's, not ours.
load_dotenv(_ROOT / ".env")

_SCHEMA_PATH = _ROOT / "schema.sql"
_connection: sqlite3.Connection | None = None
_init_lock = threading.Lock()


def db_path() -> Path:
    # MCP clients launch the server from their own working directory, so a relative
    # DB_PATH has to anchor to the project root rather than to os.getcwd().
    raw = Path(os.getenv("DB_PATH", "data/app.db"))
    return raw if raw.is_absolute() else _ROOT / raw


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cache reads and writes are dispatched through asyncio.to_thread, so the
    # connection is touched from more than one thread.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    # Cache lookups run concurrently in worker threads; without the lock two of them
    # race to apply the schema and one loses with "database is locked".
    global _connection
    if _connection is None:
        with _init_lock:
            if _connection is None:
                _connection = init_db(connect())
    return _connection


def close_db() -> None:
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
