"""ProxyPool: pull and rotate Webshare.io residential proxies.

Why this exists. Some sites (Cloudflare, Qrator, DDoS-Guard etc.) rate-limit
or block the whole sequence of probes against a single domain when they come
from the same source IP, even though our throughput is gentle by anti-DDoS
standards. Rotating each request through a distinct upstream proxy makes
those bursts look like unrelated organic traffic and dramatically reduces
false-positive blocks.

Design constraints (single-user / single-VPS service):
- No persistent infrastructure besides a tiny on-disk cache file.
- Best-effort: if the Webshare API is down or the key is missing we keep
  serving requests directly so the service is never *worse* off than today.
- Per-request rotation: every call to `random_proxy()` returns a fresh pick
  uniformly from the currently healthy pool (or None when nothing's available).

Health tracking is intentionally minimal: when the crawler reports a failure
that smells like the proxy itself (timeout, connect refused, TLS error),
`mark_bad()` puts the proxy in a short cooldown window (default 5 min).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

WEBSHARE_LIST_URL = "https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page_size=100"
WEBSHARE_TIMEOUT_SECONDS = 15.0
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes
DEFAULT_REFRESH_INTERVAL_SECONDS = 3600  # 1 hour


@dataclass
class Proxy:
    """A single upstream proxy from Webshare.

    `url` is the httpx-compatible scheme/auth form
    ``http://user:pass@ip:port`` that we hand straight to AsyncClient.
    """

    id: str
    address: str
    port: int
    username: str
    password: str
    country: str = ""
    city: str = ""
    valid: bool = True

    @property
    def url(self) -> str:
        return f"http://{self.username}:{self.password}@{self.address}:{self.port}"

    @property
    def label(self) -> str:
        return f"{self.address}:{self.port} [{self.country or '??'}]"

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "address": self.address,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "country": self.country,
            "city": self.city,
            "valid": self.valid,
        }

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> "Proxy":
        return cls(
            id=str(item.get("id") or ""),
            address=str(item.get("proxy_address") or ""),
            port=int(item.get("port") or 0),
            username=str(item.get("username") or ""),
            password=str(item.get("password") or ""),
            country=str(item.get("country_code") or ""),
            city=str(item.get("city_name") or ""),
            valid=bool(item.get("valid", True)),
        )

    @classmethod
    def from_cache(cls, item: dict[str, Any]) -> "Proxy":
        return cls(
            id=str(item.get("id") or ""),
            address=str(item.get("address") or ""),
            port=int(item.get("port") or 0),
            username=str(item.get("username") or ""),
            password=str(item.get("password") or ""),
            country=str(item.get("country") or ""),
            city=str(item.get("city") or ""),
            valid=bool(item.get("valid", True)),
        )


class ProxyPool:
    """In-memory pool with on-disk cache + per-proxy cooldown windows.

    Thread-safety: not relevant — we're inside a single asyncio event loop and
    every mutating method acquires `self._lock`. The cooldown table is a tiny
    dict, so we don't bother with a more elaborate structure.
    """

    def __init__(
        self,
        api_key: str,
        cache_path: Path,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self._api_key = api_key.strip()
        self._cache_path = cache_path
        self._cooldown_seconds = max(30, int(cooldown_seconds))
        self._refresh_interval = max(60, int(refresh_interval_seconds))
        self._proxies: list[Proxy] = []
        self._bad_until: dict[str, float] = {}  # proxy.url -> monotonic deadline
        self._lock = asyncio.Lock()
        self._last_refresh_ts: float | None = None
        self._refresh_task: asyncio.Task | None = None

    # --- public API ---------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return bool(self._api_key)

    def size(self) -> int:
        return len(self._proxies)

    def healthy_size(self) -> int:
        now = time.monotonic()
        return sum(1 for p in self._proxies if self._bad_until.get(p.url, 0.0) <= now)

    def random_proxy(self) -> Proxy | None:
        """Uniformly pick one currently-healthy proxy, or None if all banned."""
        now = time.monotonic()
        healthy = [p for p in self._proxies if self._bad_until.get(p.url, 0.0) <= now]
        if not healthy:
            return None
        return random.choice(healthy)

    def mark_bad(self, proxy_url: str, cooldown_seconds: int | None = None) -> None:
        """Quarantine a misbehaving proxy for a short window."""
        if not proxy_url:
            return
        seconds = self._cooldown_seconds if cooldown_seconds is None else max(10, int(cooldown_seconds))
        self._bad_until[proxy_url] = time.monotonic() + seconds

    def status(self) -> dict[str, Any]:
        """Diagnostic snapshot for /healthz, SSE log, or the UI."""
        return {
            "enabled": self.is_enabled,
            "total": self.size(),
            "healthy": self.healthy_size(),
            "last_refresh_age_seconds": (
                None
                if self._last_refresh_ts is None
                else int(time.monotonic() - self._last_refresh_ts)
            ),
            "refresh_interval_seconds": self._refresh_interval,
        }

    async def refresh(self) -> int:
        """Re-pull the proxy list from Webshare and atomically swap.

        Returns the number of proxies after refresh. On any error the
        previous list is kept and a warning is logged.
        """
        if not self.is_enabled:
            return 0
        try:
            new_proxies = await self._fetch_from_api()
        except Exception as exc:
            logger.warning("ProxyPool: Webshare refresh failed: %s: %s", type(exc).__name__, exc)
            return self.size()

        async with self._lock:
            self._proxies = new_proxies
            self._last_refresh_ts = time.monotonic()
            # Cooldown entries pointing at proxies no longer in the pool
            # are useless — let them GC.
            live_urls = {p.url for p in new_proxies}
            self._bad_until = {u: t for u, t in self._bad_until.items() if u in live_urls}
        self._save_cache_safely()
        return len(new_proxies)

    async def load(self) -> int:
        """Best-effort hydration on service start.

        Order of fallbacks: live API -> on-disk cache -> empty.
        """
        if self.is_enabled:
            try:
                fetched = await self._fetch_from_api()
                async with self._lock:
                    self._proxies = fetched
                    self._last_refresh_ts = time.monotonic()
                self._save_cache_safely()
                return len(fetched)
            except Exception as exc:
                logger.warning(
                    "ProxyPool: initial Webshare load failed (%s: %s); trying cache",
                    type(exc).__name__,
                    exc,
                )
        # Fallback to cache
        cached = self._load_cache_safely()
        async with self._lock:
            self._proxies = cached
        return len(cached)

    def start_background_refresh(self) -> None:
        """Spawn a periodic refresh task tied to the running event loop."""
        if self._refresh_task and not self._refresh_task.done():
            return
        if not self.is_enabled:
            return

        async def _loop() -> None:
            while True:
                try:
                    await asyncio.sleep(self._refresh_interval)
                    await self.refresh()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("ProxyPool: background refresh error: %s: %s", type(exc).__name__, exc)

        self._refresh_task = asyncio.create_task(_loop(), name="proxy-pool-refresh")

    def stop_background_refresh(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

    # --- internals ----------------------------------------------------------

    async def _fetch_from_api(self) -> list[Proxy]:
        headers = {"Authorization": f"Token {self._api_key}"}
        timeout = httpx.Timeout(WEBSHARE_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(WEBSHARE_LIST_URL, headers=headers)
            response.raise_for_status()
            payload = response.json()
        items = payload.get("results") or []
        out: list[Proxy] = []
        for item in items:
            try:
                p = Proxy.from_api(item)
            except Exception:
                continue
            if not p.address or not p.port or not p.username or not p.password:
                continue
            if not p.valid:
                continue
            out.append(p)
        return out

    def _load_cache_safely(self) -> list[Proxy]:
        try:
            if not self._cache_path.exists():
                return []
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            items = raw.get("proxies") or []
            return [Proxy.from_cache(it) for it in items if isinstance(it, dict)]
        except Exception as exc:
            logger.warning("ProxyPool: cache read failed: %s: %s", type(exc).__name__, exc)
            return []

    def _save_cache_safely(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            payload = {
                "saved_at": time.time(),
                "proxies": [p.to_cache_dict() for p in self._proxies],
            }
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self._cache_path)
        except Exception as exc:
            logger.warning("ProxyPool: cache write failed: %s: %s", type(exc).__name__, exc)


# Module-level singleton; bound by app/main.py during lifespan.
_pool: ProxyPool | None = None


def get_pool() -> ProxyPool | None:
    return _pool


def set_pool(pool: ProxyPool | None) -> None:
    global _pool
    _pool = pool
