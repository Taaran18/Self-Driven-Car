import time
from collections import deque
from threading import Lock

from fastapi import status

from app.core.errors import AppError


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._last_sweep = time.monotonic()

    def hit(self, key: str, limit: int, window_seconds: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > window_seconds:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])) + 1)
                raise AppError(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "rate_limited",
                    f"Too many attempts. Wait {retry_after} seconds and try again.",
                    {"retry_after": retry_after},
                )
            bucket.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        stale = [k for k, v in self._hits.items() if not v or now - v[-1] > 3600]
        for key in stale:
            del self._hits[key]


rate_limiter = RateLimiter()
