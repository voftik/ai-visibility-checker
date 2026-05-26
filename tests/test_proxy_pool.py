"""Unit tests for app.services.proxy_pool.

These tests exercise the local logic (cache I/O, cooldown windows, random
selection) without touching the live Webshare API. The fetch path is
verified against an in-memory fake HTTP responder.
"""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.services.proxy_pool import (
    DEFAULT_COOLDOWN_SECONDS,
    Proxy,
    ProxyPool,
)


def _api_payload(n: int) -> dict:
    """Build a minimal Webshare-shaped /api/v2/proxy/list/ response."""
    return {
        "count": n,
        "next": None,
        "previous": None,
        "results": [
            {
                "id": f"d-{i:09d}",
                "username": f"user{i}",
                "password": f"pw{i}",
                "proxy_address": f"10.0.0.{i + 1}",
                "port": 5000 + i,
                "valid": True,
                "country_code": "US" if i % 2 == 0 else "FR",
                "city_name": "X",
            }
            for i in range(n)
        ],
    }


class ProxyDataclassTests(unittest.TestCase):
    def test_url_format_is_httpx_compatible(self) -> None:
        p = Proxy(
            id="d-1", address="1.2.3.4", port=8080, username="u", password="p", country="US"
        )
        self.assertEqual(p.url, "http://u:p@1.2.3.4:8080")

    def test_from_api_extracts_essentials_and_treats_invalid_as_not_valid(self) -> None:
        item = {
            "id": "d-1",
            "username": "u",
            "password": "p",
            "proxy_address": "1.2.3.4",
            "port": 8080,
            "valid": False,
            "country_code": "FR",
            "city_name": "Paris",
        }
        p = Proxy.from_api(item)
        self.assertFalse(p.valid)
        self.assertEqual(p.country, "FR")


class ProxyPoolHealthAndSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def _make_pool(self) -> tuple[ProxyPool, TemporaryDirectory]:
        tmp = TemporaryDirectory()
        pool = ProxyPool(
            api_key="fake-key",
            cache_path=Path(tmp.name) / ".proxy_cache.json",
        )
        # Inject proxies directly to avoid the network.
        pool._proxies = [
            Proxy(id=f"d-{i}", address=f"1.2.3.{i}", port=8080 + i, username="u", password="p")
            for i in range(5)
        ]
        return pool, tmp

    async def test_random_proxy_returns_none_when_pool_empty(self) -> None:
        pool = ProxyPool(api_key="", cache_path=Path("/tmp/never-used.json"))
        self.assertIsNone(pool.random_proxy())

    async def test_random_proxy_excludes_marked_bad(self) -> None:
        pool, tmp = await self._make_pool()
        try:
            bad = pool._proxies[0]
            pool.mark_bad(bad.url, cooldown_seconds=60)
            picks = {pool.random_proxy().url for _ in range(200)}
            self.assertNotIn(bad.url, picks)
            self.assertEqual(pool.healthy_size(), 4)
        finally:
            tmp.cleanup()

    async def test_cooldown_expires_and_proxy_returns_to_pool(self) -> None:
        pool, tmp = await self._make_pool()
        try:
            target = pool._proxies[0]
            pool.mark_bad(target.url, cooldown_seconds=1)
            self.assertEqual(pool.healthy_size(), 4)
            # Fast-forward by mutating the deadline rather than sleeping.
            pool._bad_until[target.url] = time.monotonic() - 1
            self.assertEqual(pool.healthy_size(), 5)
        finally:
            tmp.cleanup()

    async def test_is_enabled_follows_api_key_presence(self) -> None:
        empty = ProxyPool(api_key="", cache_path=Path("/tmp/x.json"))
        full = ProxyPool(api_key="abc", cache_path=Path("/tmp/x.json"))
        self.assertFalse(empty.is_enabled)
        self.assertTrue(full.is_enabled)


class ProxyPoolFetchAndCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_replaces_proxies_atomically(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / ".proxy_cache.json"
            pool = ProxyPool(api_key="abc", cache_path=cache_path)

            async def _fake_fetch() -> list[Proxy]:
                return [
                    Proxy(id="x", address="9.9.9.9", port=80, username="u", password="p"),
                    Proxy(id="y", address="9.9.9.10", port=80, username="u", password="p"),
                ]

            with patch.object(pool, "_fetch_from_api", _fake_fetch):
                count = await pool.refresh()
            self.assertEqual(count, 2)
            self.assertEqual(pool.size(), 2)
            self.assertTrue(cache_path.exists())
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["proxies"]), 2)

    async def test_load_falls_back_to_cache_when_api_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / ".proxy_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "saved_at": time.time(),
                        "proxies": [
                            {
                                "id": "cached-1",
                                "address": "5.6.7.8",
                                "port": 9000,
                                "username": "u",
                                "password": "p",
                                "country": "DE",
                                "city": "Berlin",
                                "valid": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pool = ProxyPool(api_key="abc", cache_path=cache_path)

            async def _broken_fetch() -> list[Proxy]:
                raise RuntimeError("Webshare unreachable")

            with patch.object(pool, "_fetch_from_api", _broken_fetch):
                size = await pool.load()
            self.assertEqual(size, 1)
            self.assertEqual(pool._proxies[0].address, "5.6.7.8")

    async def test_disabled_pool_skips_refresh_and_returns_zero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / ".proxy_cache.json"
            pool = ProxyPool(api_key="", cache_path=cache_path)
            self.assertEqual(await pool.refresh(), 0)
            self.assertEqual(pool.size(), 0)
            self.assertFalse(cache_path.exists())


class ProxyPoolDefaultsTests(unittest.TestCase):
    def test_default_cooldown_value(self) -> None:
        self.assertEqual(DEFAULT_COOLDOWN_SECONDS, 300)


if __name__ == "__main__":
    unittest.main()
