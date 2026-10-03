import asyncio
import random
import time
from collections.abc import Awaitable, Callable

import httpx

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                await asyncio.sleep(self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval


async def with_retry(
    send: Callable[[], Awaitable[httpx.Response]],
    attempts: int = 3,
    base_delay: float = 0.5,
) -> httpx.Response:
    response = await send()
    for attempt in range(attempts - 1):
        if response.status_code not in RETRY_STATUS:
            return response
        # Jitter stops concurrent callers retrying in lockstep after a shared 429.
        await asyncio.sleep(base_delay * 2**attempt * (0.5 + random.random()))
        response = await send()
    return response
