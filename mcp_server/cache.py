import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from storage import get_db


def default_ttl() -> int:
    return int(os.getenv("CACHE_TTL_SECONDS", "86400"))


def cache_key(source: str, params: Mapping[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return f"{source}:{hashlib.sha256(blob.encode()).hexdigest()}"


def _read(key: str) -> Any | None:
    row = get_db().execute(
        "SELECT payload, created_at, ttl_seconds FROM api_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if row is None or time.time() - row["created_at"] > row["ttl_seconds"]:
        return None
    return json.loads(row["payload"])


def _write(key: str, payload: Any, ttl: int) -> None:
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO api_cache (cache_key, payload, created_at, ttl_seconds)"
        " VALUES (?, ?, ?, ?)",
        (key, json.dumps(payload), int(time.time()), ttl),
    )
    db.commit()


async def cached(
    source: str,
    params: Mapping[str, Any],
    fetch: Callable[[], Awaitable[Any]],
    ttl: int | None = None,
) -> tuple[Any, bool]:
    key = cache_key(source, params)
    hit = await asyncio.to_thread(_read, key)
    if hit is not None:
        return hit, True

    payload = await fetch()
    await asyncio.to_thread(_write, key, payload, ttl if ttl is not None else default_ttl())
    return payload, False
