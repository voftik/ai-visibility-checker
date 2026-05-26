"""Crawler: probes (domain, user_agent) pairs over async httpx and persists
DomainProbe + RobotsRule rows; publishes phase/log/progress events to event_bus.

Pipeline per Run:
  1. Load Run, parse config_json.
  2. phase_change crawling_started; progress_total = D * (UAs + 1).
  3. asyncio.Semaphore(concurrency) over a flat list of probes:
        per-domain robots.txt probe (UA="robots-fetcher")  +
        per-(domain, ua) main_page probe.
  4. Each probe writes a DomainProbe row; robots.txt success additionally
     writes RobotsRule rows via the parser.
  5. phase_change crawling_done -> Run.status = completed (analyzer arrives
     in the next phase). On top-level failure: status=failed + final event.
"""
from __future__ import annotations

import asyncio
import re
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from sqlalchemy import select

from app.db import SessionLocal
from app.models import DomainProbe, ProbeType, RobotsRule, Run, RunStatus
from app.services.analyzer import analyze_run
from app.services.content_extractor import extract_text_signals
from app.services.event_bus import bus
from app.services.protections import detect_protections
from app.services.robots_parser import parse_robots, parse_robots_unavailable

USER_AGENT_STRINGS: dict[str, str] = {
    "GPTBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; GPTBot/1.2; +https://openai.com/gptbot",
    "OAI-SearchBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; OAI-SearchBot/1.0; +https://openai.com/searchbot",
    "ChatGPT-User": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; ChatGPT-User/1.0; +https://openai.com/bot",
    "ClaudeBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; ClaudeBot/1.0; +claudebot@anthropic.com",
    "anthropic-ai": "anthropic-ai",
    "Claude-Web": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; Claude-Web/1.0; +http://www.anthropic.com",
    "PerplexityBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; PerplexityBot/1.0; +https://perplexity.ai/perplexitybot",
    "Perplexity-User": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; Perplexity-User/1.0; +https://perplexity.ai/perplexity-user",
    "DeepSeekBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; DeepSeekBot/1.0; +https://deepseek.com",
    "DeepSeek-User": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; DeepSeek-User/1.0",
    "Chrome-control": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "empty-ua": "",
    "robots-fetcher": "ai-visibility-checker/1.0 research bot",
}

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

BODY_SAMPLE_BYTES = 4096
HEADERS_BUDGET_BYTES = 8 * 1024
BODY_LOOKS_EMPTY_THRESHOLD = 1500
# Minimum interval between two requests to the same domain. With column-major
# scheduling alone the gap is already a few seconds on 10+ domain runs, but
# this floor makes the behaviour predictable on tiny runs (e.g. 2 domains × 6
# UAs) where pure scheduling can still come close to 0.
MIN_GAP_PER_DOMAIN_SECONDS = 1.0
_BODY_CONTENT_RX = re.compile(r"<(article|main|section|p|h1|h2)[\s>]", re.I)


@dataclass
class ProbeResult:
    http_status: int | None = None
    response_size_bytes: int | None = None
    ttfb_ms: int | None = None
    total_time_ms: int | None = None
    tls_ok: bool | None = None
    final_url: str | None = None
    redirect_chain: list[dict[str, Any]] | None = None
    response_headers: dict[str, Any] | None = None
    body_sample: str | None = None
    body_bytes: bytes | None = None
    body_looks_empty: bool = False
    full_text: str = ""           # decoded body (or empty for binary)
    content_type: str | None = None
    error_class: str | None = None
    error_message: str | None = None


def _classify_error(exc: BaseException) -> tuple[str, str]:
    msg = str(exc)[:500]
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout", msg
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout", msg
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout", msg
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout", msg
    if isinstance(exc, httpx.RemoteProtocolError):
        return "protocol_error", msg
    if isinstance(exc, httpx.TooManyRedirects):
        return "redirect_loop", msg
    if isinstance(exc, httpx.ConnectError):
        s = msg.lower()
        if any(k in s for k in ("ssl", "certificate", "tls")):
            return "tls_error", msg
        if "name or service not known" in s or "nodename nor servname" in s:
            return "dns_fail", msg
        return "connection_refused", msg
    if isinstance(exc, socket.gaierror) or "name or service not known" in msg.lower():
        return "dns_fail", msg
    return "other", msg


def _looks_binary(body: bytes, content_type: str | None) -> bool:
    if content_type:
        ct = content_type.lower()
        if ct.startswith(("image/", "audio/", "video/", "application/octet-stream", "application/pdf")):
            return True
    head = body[:200]
    if not head:
        return False
    nontext = sum(1 for b in head if b < 9 or (13 < b < 32))
    return nontext / max(1, len(head)) > 0.1


def _trim_headers(headers: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    used = 0
    truncated = False
    for k, v in headers.items():
        if k.lower() == "set-cookie":
            continue
        size = len(k) + len(v) + 4
        if used + size > HEADERS_BUDGET_BYTES:
            truncated = True
            break
        out[k] = v
        used += size
    if truncated:
        out["_truncated"] = True
    return out


def _decode_body(body: bytes, content_type: str | None) -> str:
    enc = "utf-8"
    if content_type and "charset=" in content_type.lower():
        m = re.search(r"charset=([^;\s]+)", content_type, re.I)
        if m:
            enc = m.group(1).strip().strip('"').strip("'") or "utf-8"
    try:
        return body.decode(enc, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


async def _do_probe(
    url: str,
    user_agent: str,
    timeout_seconds: int,
    concurrency: int,
) -> ProbeResult:
    result = ProbeResult()
    redirects: list[dict[str, Any]] = []

    async def _on_response(resp: httpx.Response) -> None:
        if 300 <= resp.status_code < 400:
            redirects.append(
                {
                    "status": resp.status_code,
                    "url": str(resp.request.url),
                    "location": resp.headers.get("location"),
                }
            )

    headers = {"User-Agent": user_agent, **DEFAULT_HEADERS}
    if not user_agent:
        headers.pop("User-Agent")

    timeout = httpx.Timeout(connect=5.0, read=float(timeout_seconds), write=5.0, pool=5.0)
    limits = httpx.Limits(
        max_connections=max(2, concurrency * 2),
        max_keepalive_connections=max(1, concurrency),
    )

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            http2=True,
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            verify=True,
            event_hooks={"response": [_on_response]},
        ) as client:
            response = await client.get(url, headers=headers)
            elapsed_ms = int(response.elapsed.total_seconds() * 1000)
            result.ttfb_ms = elapsed_ms  # httpx: time until response is "ready"; treat as TTFB proxy
            result.http_status = response.status_code
            result.final_url = str(response.url)
            result.response_headers = _trim_headers(dict(response.headers))
            result.tls_ok = True

            body_chunks: list[bytes] = []
            collected = 0
            async for chunk in response.aiter_bytes():
                body_chunks.append(chunk)
                collected += len(chunk)
                if collected >= 512 * 1024:
                    break  # cap at 512KB to avoid pulling huge pages
            await response.aclose()
            body = b"".join(body_chunks)
            result.response_size_bytes = len(body)
            result.body_bytes = body

            ct = response.headers.get("content-type")
            result.content_type = ct
            if _looks_binary(body, ct):
                result.body_sample = f"<binary content, content-type={ct or 'unknown'}>"
            else:
                text = _decode_body(body[:BODY_SAMPLE_BYTES * 2], ct)
                result.body_sample = text[:BODY_SAMPLE_BYTES]

            full_text = "" if _looks_binary(body, ct) else _decode_body(body, ct)
            result.full_text = full_text
            if len(body) < BODY_LOOKS_EMPTY_THRESHOLD:
                result.body_looks_empty = not bool(_BODY_CONTENT_RX.search(full_text))
            else:
                result.body_looks_empty = False

            if response.history:
                redirects = [
                    {
                        "status": h.status_code,
                        "url": str(h.request.url),
                        "location": h.headers.get("location"),
                    }
                    for h in response.history
                ]
            result.redirect_chain = redirects or None
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        cls, msg = _classify_error(exc)
        result.error_class = cls
        result.error_message = msg
        if cls == "tls_error":
            result.tls_ok = False
        # other errors leave tls_ok = None
    finally:
        result.total_time_ms = int((time.monotonic() - started) * 1000)

    return result


def _summarize_probe(label: str, markers: list[str], probe: ProbeResult) -> str:
    if probe.error_class:
        return f"{label} → ERROR {probe.error_class}"
    status = probe.http_status
    size_kb = (probe.response_size_bytes or 0) / 1024
    if markers:
        first = markers[0]
        return f"{label} → {first} ({status})"
    return f"{label} → {status} ({size_kb:.0f}KB, {probe.total_time_ms}ms)"


@dataclass
class _Job:
    domain: str
    user_agent_label: str
    user_agent_string: str
    target_url: str
    probe_type: ProbeType


def normalize_domain(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""

    if "://" in s:
        parsed = urlparse(s)
        host = parsed.netloc
    else:
        host = s

    if "@" in host:
        host = host.rsplit("@", 1)[1]
    for sep in ("/", "?", "#"):
        host = host.split(sep, 1)[0]
    if ":" in host and host.count(":") == 1:
        host = host.split(":", 1)[0]
    host = host.strip().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_domains(domains: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in domains:
        domain = normalize_domain(raw)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
    return out


def _build_jobs(domains: list[str], ua_labels: list[str]) -> list[_Job]:
    """Schedule jobs column-major: one full pass over all domains per UA.

    Old (row-major) order hit each domain N+1 times in a row, which trips
    rate-limiters and antibot stacks. Column-major schedules robots.txt for
    every domain first, then main_page with UA #1 for every domain, then UA
    #2 for every domain, and so on. Combined with a per-domain throttle in
    the worker, the gap between two requests to the same host is at least
    `len(domains) * (probe_time + small_overhead)`, which on a 10+ domain
    run is several seconds — enough to not look like a flood.
    """
    norm = normalize_domains(domains)
    if not norm:
        return []
    jobs: list[_Job] = []

    # Pass 0: robots.txt for every domain
    robots_ua = USER_AGENT_STRINGS["robots-fetcher"]
    for d_clean in norm:
        jobs.append(
            _Job(
                domain=d_clean,
                user_agent_label="robots-fetcher",
                user_agent_string=robots_ua,
                target_url=f"https://{d_clean}/robots.txt",
                probe_type=ProbeType.robots_txt,
            )
        )

    # Passes 1..K: main_page with each UA, in column-major order
    for label in ua_labels:
        ua_str = USER_AGENT_STRINGS.get(label)
        if ua_str is None:
            continue
        for d_clean in norm:
            jobs.append(
                _Job(
                    domain=d_clean,
                    user_agent_label=label,
                    user_agent_string=ua_str,
                    target_url=f"https://{d_clean}/",
                    probe_type=ProbeType.main_page,
                )
            )
    return jobs


async def run_crawl(run_id: str) -> None:
    try:
        async with SessionLocal() as session:
            run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
            if run is None:
                bus.publish(run_id, {"type": "log", "level": "error", "message": "run not found"})
                bus.publish(run_id, {"type": "final", "status": "failed"})
                return
            cfg = dict(run.config_json or {})
            domains: list[str] = list(cfg.get("domains") or [])
            ua_labels: list[str] = list(cfg.get("user_agents") or [])
            concurrency: int = int(cfg.get("concurrency") or 8)
            timeout_seconds: int = int(cfg.get("timeout_seconds") or 15)

            jobs = _build_jobs(domains, ua_labels)
            run.status = RunStatus.crawling
            run.progress_current = 0
            run.progress_total = len(jobs)
            await session.commit()

        bus.publish(run_id, {"type": "phase_change", "phase": "crawling_started"})
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "info",
                "message": f"Crawling {len(domains)} domains × {len(ua_labels)} user-agents (+robots.txt)",
            },
        )
        bus.publish(run_id, {"type": "progress", "current": 0, "total": len(jobs), "phase": "crawling"})

        if not jobs:
            async with SessionLocal() as session:
                run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one()
                run.status = RunStatus.completed
                await session.commit()
            bus.publish(run_id, {"type": "phase_change", "phase": "crawling_done"})
            bus.publish(run_id, {"type": "phase_change", "phase": "completed"})
            bus.publish(run_id, {"type": "final", "status": "completed"})
            return

        sem = asyncio.Semaphore(concurrency)
        progress_lock = asyncio.Lock()
        completed_count = {"n": 0, "errors": 0}

        # Per-domain throttle. Two probes to the same host respect a minimum
        # gap, so we don't look like a flood to anti-DDoS / WAF stacks. Each
        # domain has its own asyncio.Lock to make the read-then-update of the
        # last-request timestamp atomic across concurrent workers.
        domain_locks: dict[str, asyncio.Lock] = {}
        last_request_ts: dict[str, float] = {}

        def _lock_for(d: str) -> asyncio.Lock:
            lk = domain_locks.get(d)
            if lk is None:
                lk = asyncio.Lock()
                domain_locks[d] = lk
            return lk

        async def worker(job: _Job) -> None:
            async with sem:
                # Throttle per domain. Hold the per-domain lock around the
                # sleep+stamp so a sibling worker for the same host queues up
                # behind us instead of racing the timestamp check.
                lk = _lock_for(job.domain)
                async with lk:
                    last = last_request_ts.get(job.domain)
                    now = time.monotonic()
                    if last is not None:
                        wait = MIN_GAP_PER_DOMAIN_SECONDS - (now - last)
                        if wait > 0:
                            await asyncio.sleep(wait)
                    last_request_ts[job.domain] = time.monotonic()

                bus.publish(
                    run_id,
                    {
                        "type": "log",
                        "level": "info",
                        "message": f"GET {job.target_url} as {job.user_agent_label}",
                    },
                )
                probe = await _do_probe(
                    url=job.target_url,
                    user_agent=job.user_agent_string,
                    timeout_seconds=timeout_seconds,
                    concurrency=concurrency,
                )

                body_text_for_detect = probe.body_sample if probe.body_sample and not probe.body_sample.startswith("<binary") else ""

                # Content-shape signals: only meaningful for HTML responses on
                # main_page. For robots.txt or binary errors we skip the run.
                content_signals: dict | None = None
                content_extractable_len: int | None = None
                if (
                    job.probe_type is ProbeType.main_page
                    and probe.error_class is None
                    and probe.http_status is not None
                ):
                    try:
                        content_signals = extract_text_signals(
                            probe.full_text or "",
                            probe.content_type,
                        )
                        content_extractable_len = content_signals["extractable_text_length"]
                    except Exception:
                        content_signals = None
                        content_extractable_len = None

                markers: list[str] = []
                challenge = False
                if probe.http_status is not None or probe.tls_ok is not None:
                    markers, challenge = detect_protections(
                        status=probe.http_status,
                        headers=probe.response_headers,
                        body_text=body_text_for_detect,
                        final_url=probe.final_url or job.target_url,
                        tls_ok=probe.tls_ok,
                        domain=job.domain,
                        body_looks_empty=probe.body_looks_empty,
                        probe_type=job.probe_type.value,
                        content_signals=content_signals,
                    )
                elif probe.error_class == "tls_error":
                    markers, challenge = detect_protections(
                        status=None,
                        headers=None,
                        body_text="",
                        final_url=job.target_url,
                        tls_ok=False,
                        domain=job.domain,
                        body_looks_empty=False,
                        probe_type=job.probe_type.value,
                        content_signals=None,
                    )

                # Per-probe DB write isolated in try/except: a transient lock,
                # an FK race after DELETE, or a JSON-serialisation surprise on
                # one probe must not crash the entire run. Worst case: that
                # single probe row is missing.
                try:
                    async with SessionLocal() as session:
                        db_probe = DomainProbe(
                            run_id=run_id,
                            domain=job.domain,
                            user_agent_label=job.user_agent_label,
                            user_agent_string=job.user_agent_string,
                            target_url=job.target_url,
                            probe_type=job.probe_type,
                            http_status=probe.http_status,
                            response_size_bytes=probe.response_size_bytes,
                            ttfb_ms=probe.ttfb_ms,
                            total_time_ms=probe.total_time_ms,
                            tls_ok=probe.tls_ok,
                            final_url=probe.final_url,
                            redirect_chain=probe.redirect_chain,
                            response_headers=probe.response_headers,
                            detected_protections=markers or None,
                            challenge_detected=challenge,
                            body_sample=probe.body_sample,
                            body_looks_empty=probe.body_looks_empty,
                            content_extractable_text_length=content_extractable_len,
                            content_signals=content_signals,
                            error_class=probe.error_class,
                            error_message=probe.error_message,
                        )
                        session.add(db_probe)

                        if job.probe_type is ProbeType.robots_txt:
                            if probe.error_class is None and probe.http_status == 200 and probe.body_bytes is not None:
                                try:
                                    text = probe.body_bytes.decode("utf-8", errors="replace")
                                except Exception:
                                    text = ""
                                for bot, rule, raw in parse_robots(text):
                                    session.add(
                                        RobotsRule(
                                            run_id=run_id,
                                            domain=job.domain,
                                            bot_name=bot,
                                            rule=rule,
                                            raw_directives=raw or None,
                                        )
                                    )
                            else:
                                err = probe.error_class or (
                                    f"http_{probe.http_status}" if probe.http_status else "unknown"
                                )
                                for bot, rule, raw in parse_robots_unavailable(err):
                                    session.add(
                                        RobotsRule(
                                            run_id=run_id,
                                            domain=job.domain,
                                            bot_name=bot,
                                            rule=rule,
                                            raw_directives=raw,
                                        )
                                    )
                        await session.commit()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    bus.publish(run_id, {
                        "type": "log",
                        "level": "warn",
                        "message": (
                            f"DB write failed for {job.domain} [{job.user_agent_label}]: "
                            f"{type(exc).__name__}: {exc}"
                        )[:300],
                    })

                async with progress_lock:
                    completed_count["n"] += 1
                    if probe.error_class:
                        completed_count["errors"] += 1
                    n = completed_count["n"]
                    try:
                        async with SessionLocal() as session:
                            run = (
                                await session.execute(select(Run).where(Run.id == run_id))
                            ).scalar_one_or_none()
                            if run is None:
                                # Run was deleted while in-flight; nothing to update.
                                return
                            run.progress_current = n
                            await session.commit()
                    except asyncio.CancelledError:
                        raise
                    except BaseException:
                        # Progress-counter update is best-effort. Don't kill the run.
                        pass

                summary = _summarize_probe(job.user_agent_label, markers, probe)
                bus.publish(
                    run_id,
                    {
                        "type": "probe_done",
                        "domain": job.domain,
                        "user_agent_label": job.user_agent_label,
                        "http_status": probe.http_status,
                        "summary": summary,
                    },
                )
                bus.publish(
                    run_id,
                    {"type": "progress", "current": completed_count["n"], "total": len(jobs), "phase": "crawling"},
                )
                if probe.error_class:
                    bus.publish(
                        run_id,
                        {
                            "type": "log",
                            "level": "warn",
                            "message": f"{job.domain} [{job.user_agent_label}] {probe.error_class}: {probe.error_message[:200] if probe.error_message else ''}",
                        },
                    )

        await asyncio.gather(*(worker(j) for j in jobs), return_exceptions=False)

        bus.publish(run_id, {"type": "phase_change", "phase": "crawling_done"})
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "info",
                "message": f"Crawled {completed_count['n']} probes, {completed_count['errors']} errors",
            },
        )

        # Hand off to the analyzer (cross-probe pass + 4-step LLM pipeline).
        # analyze_run handles its own status transitions and final event.
        await analyze_run(run_id)

    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            bus.publish(run_id, {"type": "log", "level": "error", "message": "run cancelled"})
            bus.publish(run_id, {"type": "phase_change", "phase": "failed"})
            bus.publish(run_id, {"type": "final", "status": "failed"})
            raise
        try:
            async with SessionLocal() as session:
                run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
                if run is not None:
                    run.status = RunStatus.failed
                    run.error_message = f"{type(exc).__name__}: {exc}"[:1000]
                    await session.commit()
        except Exception:
            pass
        bus.publish(
            run_id,
            {"type": "log", "level": "error", "message": f"crawler error: {type(exc).__name__}: {exc}"},
        )
        bus.publish(run_id, {"type": "phase_change", "phase": "failed"})
        bus.publish(run_id, {"type": "final", "status": "failed"})
