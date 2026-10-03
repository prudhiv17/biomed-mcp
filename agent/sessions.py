import time
import uuid
from typing import Any

from pydantic import BaseModel

from storage import get_db


class Turn(BaseModel):
    id: int
    role: str
    content: str
    tool_name: str | None = None
    tokens: int | None = None


def create_session() -> str:
    session_id = uuid.uuid4().hex[:12]
    now = int(time.time())
    db = get_db()
    db.execute(
        "INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
        (session_id, now, now),
    )
    db.commit()
    return session_id


def ensure_session(session_id: str | None) -> str:
    if session_id is None:
        return create_session()
    row = get_db().execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return session_id if row else create_session()


def add_turn(
    session_id: str,
    role: str,
    content: str,
    tool_name: str | None = None,
    tokens: int | None = None,
) -> int:
    db = get_db()
    cursor = db.execute(
        "INSERT INTO turns (session_id, role, content, tool_name, tokens, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, tool_name, tokens, int(time.time())),
    )
    db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (int(time.time()), session_id))
    db.commit()
    return int(cursor.lastrowid or 0)


def turns(session_id: str, include_compacted: bool = False) -> list[Turn]:
    clause = "" if include_compacted else " AND compacted = 0"
    rows = get_db().execute(
        f"SELECT id, role, content, tool_name, tokens FROM turns"
        f" WHERE session_id = ?{clause} ORDER BY id",
        (session_id,),
    ).fetchall()
    return [Turn(**dict(row)) for row in rows]


def mark_compacted(turn_ids: list[int]) -> None:
    if not turn_ids:
        return
    db = get_db()
    placeholders = ",".join("?" * len(turn_ids))
    db.execute(f"UPDATE turns SET compacted = 1 WHERE id IN ({placeholders})", turn_ids)
    db.commit()


def summary(session_id: str) -> str | None:
    row = get_db().execute("SELECT summary FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return row["summary"] if row else None


def set_summary(session_id: str, text: str) -> None:
    db = get_db()
    db.execute("UPDATE sessions SET summary = ? WHERE id = ?", (text, session_id))
    db.commit()


def add_tokens(session_id: str, count: int) -> None:
    db = get_db()
    db.execute(
        "UPDATE sessions SET token_total = token_total + ? WHERE id = ?", (count, session_id)
    )
    db.commit()


def token_total(session_id: str) -> int:
    row = get_db().execute(
        "SELECT token_total FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return int(row["token_total"]) if row else 0


def list_sessions(limit: int = 25) -> list[dict[str, Any]]:
    rows = get_db().execute(
        "SELECT s.id, s.created_at, s.updated_at, s.token_total,"
        " (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turn_count"
        " FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def first_question(session_id: str) -> str:
    row = get_db().execute(
        "SELECT content FROM turns WHERE session_id = ? AND role = 'user' ORDER BY id LIMIT 1",
        (session_id,),
    ).fetchone()
    return row["content"] if row else "(empty session)"


def log_call(
    session_id: str | None,
    tool: str,
    status: str,
    latency_ms: int,
    tokens_in: int = 0,
    tokens_out: int = 0,
    args_hash: str | None = None,
) -> None:
    db = get_db()
    db.execute(
        "INSERT INTO audit_log (session_id, tool, args_hash, status, latency_ms, tokens_in,"
        " tokens_out, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, tool, args_hash, status, latency_ms, tokens_in, tokens_out, int(time.time())),
    )
    db.commit()


def recent_calls(limit: int = 50) -> list[dict[str, Any]]:
    rows = get_db().execute(
        "SELECT session_id, tool, status, latency_ms, tokens_in, tokens_out, created_at"
        " FROM audit_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]
