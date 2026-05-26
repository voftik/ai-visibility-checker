"""Behavioural tests for proxy visibility in SSE events.

These tests exercise the crawler's event-emission contract:
  - `probe_done` events MUST carry proxy_used / proxy_address /
    proxy_country / proxy_fallback fields.
  - The `GET ... as ... via ...` log line MUST contain the proxy's
    ip:port but never the credentials (user/password).
  - When a proxy probe fails with a transport-error class and
    PROXY_FALLBACK_DIRECT is on, the crawler MUST emit a warn-level
    log mentioning "retrying directly" AND the resulting probe_done
    event MUST have proxy_fallback=True.

We drive `run_crawl()` on a 1-domain / 1-UA config with a mocked
`_do_probe`, a mocked SessionLocal/Run, a stub proxy pool, and a
no-op analyzer. The real EventBus collects events for inspection.
"""
from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx

from app.services import crawler
from app.services.event_bus import EventBus
from app.services.proxy_pool import Proxy


def _make_proxy(
    address: str = "10.0.0.1",
    port: int = 8001,
    username: str = "secret-user",
    password: str = "super-secret-pass",
    country: str = "US",
) -> Proxy:
    return Proxy(
        id="d-1",
        address=address,
        port=port,
        username=username,
        password=password,
        country=country,
    )


class _StubPool:
    """Minimal stand-in for app.services.proxy_pool.ProxyPool."""

    is_enabled = True

    def __init__(self, proxy: Proxy | None) -> None:
        self._proxy = proxy
        self.bad_marked: list[str] = []

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "total": 1 if self._proxy else 0,
                "healthy": 1 if self._proxy else 0,
                "last_refresh_age_seconds": 0, "refresh_interval_seconds": 3600}

    def random_proxy(self) -> Proxy | None:
        return self._proxy

    def mark_bad(self, proxy_url: str, cooldown_seconds: int | None = None) -> None:
        self.bad_marked.append(proxy_url)


class _FakeRun:
    """In-memory replacement for ORM Run row."""

    def __init__(self, run_id: str, domains: list[str], ua_labels: list[str]) -> None:
        self.id = run_id
        self.config_json: dict[str, Any] = {
            "domains": domains,
            "user_agents": ua_labels,
            "concurrency": 1,
            "timeout_seconds": 5,
        }
        self.status = None
        self.progress_current = 0
        self.progress_total = 0
        self.error_message: str | None = None


class _FakeResult:
    def __init__(self, row: Any) -> None:
        self._row = row

    def scalar_one_or_none(self) -> Any:
        return self._row

    def scalar_one(self) -> Any:
        return self._row


class _FakeSession:
    """No-op async session that returns the same Run for any select()."""

    def __init__(self, run: _FakeRun) -> None:
        self._run = run

    async def execute(self, _stmt: Any) -> _FakeResult:
        return _FakeResult(self._run)

    def add(self, _obj: Any) -> None:  # pragma: no cover - swallowed by run
        pass

    async def commit(self) -> None:
        pass

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        pass


def _session_factory(run: _FakeRun):
    """Build a SessionLocal() replacement bound to a specific fake run."""

    def _factory() -> _FakeSession:
        return _FakeSession(run)

    return _factory


@asynccontextmanager
async def _drive_crawl(
    *,
    domain: str,
    ua_label: str,
    proxy: Proxy | None,
    probe_results: list[crawler.ProbeResult],
):
    """Run run_crawl() with everything heavy stubbed out, yield captured events."""

    run = _FakeRun(run_id="test-run-vis", domains=[domain], ua_labels=[ua_label])
    pool = _StubPool(proxy)

    # Sequential probe results: first call returns probe_results[0],
    # second call (the fallback) returns probe_results[1] if present.
    call_state = {"n": 0}

    async def _fake_do_probe(**_kwargs: Any) -> crawler.ProbeResult:
        idx = call_state["n"]
        call_state["n"] += 1
        if idx < len(probe_results):
            return probe_results[idx]
        return probe_results[-1]

    async def _noop_analyze(_run_id: str) -> None:
        # crawler.run_crawl always calls analyze_run at the end; we don't
        # want the LLM pipeline firing in a unit test.
        from app.services.event_bus import bus
        bus.publish(_run_id, {"type": "final", "status": "completed"})

    test_bus = EventBus()
    captured: list[dict[str, Any]] = []

    def _capture(run_id: str, event: dict[str, Any]) -> None:
        captured.append(event)
        # Forward to the real bus mechanics so finalisation paths still work.
        EventBus.publish(test_bus, run_id, event)

    with (
        patch.object(crawler, "SessionLocal", _session_factory(run)),
        patch.object(crawler, "get_pool", lambda: pool),
        patch.object(crawler, "_do_probe", _fake_do_probe),
        patch.object(crawler, "analyze_run", _noop_analyze),
        patch.object(crawler.bus, "publish", _capture),
    ):
        await crawler.run_crawl(run.id)

    yield SimpleNamespace(events=captured, pool=pool, run=run)


def _ok_probe() -> crawler.ProbeResult:
    pr = crawler.ProbeResult()
    pr.http_status = 200
    pr.response_size_bytes = 1500
    pr.ttfb_ms = 50
    pr.total_time_ms = 80
    pr.tls_ok = True
    pr.final_url = "https://example.com/"
    pr.response_headers = {"content-type": "text/html"}
    pr.body_sample = "<html><body><p>hi</p></body></html>"
    pr.body_bytes = pr.body_sample.encode("utf-8")
    pr.body_looks_empty = False
    pr.full_text = pr.body_sample
    pr.content_type = "text/html"
    return pr


def _proxy_error_probe() -> crawler.ProbeResult:
    pr = crawler.ProbeResult()
    pr.tls_ok = None
    pr.error_class = "connect_timeout"
    pr.error_message = "timed out"
    pr.total_time_ms = 5000
    return pr


class ProbeDoneProxyFieldsTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_done_event_includes_proxy_fields(self) -> None:
        """probe_done MUST carry proxy_used/address/country/fallback."""
        proxy = _make_proxy(address="10.0.0.1", port=8001, country="US")

        # robots.txt probe (1st) + main_page probe (2nd) — both succeed.
        results = [_ok_probe(), _ok_probe()]
        async with _drive_crawl(
            domain="example.com",
            ua_label="GPTBot",
            proxy=proxy,
            probe_results=results,
        ) as ctx:
            pass

        probe_done = [e for e in ctx.events if e.get("type") == "probe_done"]
        self.assertGreaterEqual(len(probe_done), 1)
        main_done = [
            e for e in probe_done if e.get("user_agent_label") == "GPTBot"
        ]
        self.assertEqual(len(main_done), 1)

        event = main_done[0]
        self.assertIn("proxy_used", event)
        self.assertIn("proxy_address", event)
        self.assertIn("proxy_country", event)
        self.assertIn("proxy_fallback", event)

        self.assertTrue(event["proxy_used"])
        self.assertEqual(event["proxy_address"], "10.0.0.1:8001")
        self.assertEqual(event["proxy_country"], "US")
        self.assertFalse(event["proxy_fallback"])


class ProxyLogCredentialSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_log_message_contains_proxy_ip_no_credentials(self) -> None:
        """`GET ... as ... via ip:port [CC]` MUST not leak user:pass."""
        proxy = _make_proxy(
            address="10.0.0.42",
            port=8042,
            username="webshare-user",
            password="webshare-PASSWORD!",
            country="DE",
        )
        results = [_ok_probe(), _ok_probe()]
        async with _drive_crawl(
            domain="example.com",
            ua_label="ClaudeBot",
            proxy=proxy,
            probe_results=results,
        ) as ctx:
            pass

        get_logs = [
            e for e in ctx.events
            if e.get("type") == "log"
            and isinstance(e.get("message"), str)
            and e["message"].startswith("GET ")
        ]
        self.assertGreaterEqual(len(get_logs), 1)

        for event in get_logs:
            msg = event["message"]
            # Positive: must mention the proxy IP+port.
            self.assertIn("10.0.0.42:8042", msg)
            self.assertIn("via", msg)
            # Negative: credentials MUST NOT appear anywhere.
            self.assertNotIn("webshare-user", msg)
            self.assertNotIn("webshare-PASSWORD!", msg)
            self.assertNotIn("password", msg.lower())
            # The full httpx-style "http://user:pass@host" form must
            # not have leaked verbatim.
            self.assertNotIn("@10.0.0.42", msg)


class ProxyFallbackEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_fallback_emits_warn_and_marks_event(self) -> None:
        """Proxy transport-error → warn log + probe_done.proxy_fallback=True."""
        proxy = _make_proxy(address="10.0.0.7", port=8007, country="FR")

        # robots.txt: proxy errors out → fallback direct succeeds.
        # main_page: proxy errors out → fallback direct succeeds.
        # Order of _do_probe calls: robots(proxy)→robots(direct)→
        #   main(proxy)→main(direct).
        results = [
            _proxy_error_probe(),
            _ok_probe(),
            _proxy_error_probe(),
            _ok_probe(),
        ]
        async with _drive_crawl(
            domain="example.com",
            ua_label="GPTBot",
            proxy=proxy,
            probe_results=results,
        ) as ctx:
            pass

        warn_logs = [
            e for e in ctx.events
            if e.get("type") == "log"
            and e.get("level") == "warn"
            and isinstance(e.get("message"), str)
            and "retrying directly" in e["message"]
        ]
        self.assertGreaterEqual(
            len(warn_logs),
            1,
            f"expected at least one 'retrying directly' warn log; "
            f"got events: {ctx.events}",
        )
        # The fallback warn line MUST reference the proxy by ip:port,
        # never by credentials.
        for event in warn_logs:
            msg = event["message"]
            self.assertIn("10.0.0.7:8007", msg)
            self.assertNotIn(proxy.password, msg)
            self.assertNotIn(proxy.username, msg)

        # And every probe_done in this scenario was a fallback.
        probe_done = [e for e in ctx.events if e.get("type") == "probe_done"]
        self.assertGreaterEqual(len(probe_done), 1)
        for event in probe_done:
            self.assertTrue(
                event.get("proxy_fallback"),
                f"expected proxy_fallback=True on {event}",
            )

        # Proxy should have been quarantined (mark_bad called).
        self.assertGreaterEqual(len(ctx.pool.bad_marked), 1)


if __name__ == "__main__":
    unittest.main()
