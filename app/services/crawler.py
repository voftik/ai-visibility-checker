"""Resumable bounded-corpus crawler for the AIV audit.

The public product accepts one domain.  Discovery may inspect a broad
same-domain navigation/sitemap frontier, but the expensive page and crawler-UA
matrix is always bounded to an evidence-backed representative corpus.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
import shlex
import socket
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urljoin, urlparse, urlunparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select, update

from app.config import settings
from app.db import SessionLocal
from app.models import (
    DomainProbe,
    ProbeType,
    RobotsRule,
    Run,
    RunArtifact,
    RunStatus,
    SitePage,
)
from app.services.content_extractor import extract_text_signals
from app.services.progress import fail_run, update_progress
from app.services.protections import detect_protections
from app.services.proxy_pool import Proxy, get_pool
from app.services.robots_parser import parse_robots, parse_robots_unavailable
from app.services.run_lease import assert_run_lease, bind_run_lease
from app.services.site_preview import capture_site_preview

logger = logging.getLogger(__name__)

USER_AGENT_STRINGS: dict[str, str] = {
    "GPTBot": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "GPTBot/1.2; +https://openai.com/gptbot"
    ),
    "OAI-SearchBot": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "OAI-SearchBot/1.0; +https://openai.com/searchbot"
    ),
    "ChatGPT-User": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "ChatGPT-User/1.0; +https://openai.com/bot"
    ),
    "ClaudeBot": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "ClaudeBot/1.0; +claudebot@anthropic.com"
    ),
    "PerplexityBot": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "PerplexityBot/1.0; +https://perplexity.ai/perplexitybot"
    ),
    "Perplexity-User": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "Perplexity-User/1.0; +https://perplexity.ai/perplexity-user"
    ),
    # DeepSeek does not publish a crawler specification. This observed token
    # is kept as a lower-confidence diagnostic identity, never as proof of
    # DeepSeek's production retrieval behaviour.
    "DeepSeekBot": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; "
        "DeepSeekBot/1.0; +https://deepseek.com"
    ),
    "Googlebot-desktop": (
        "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; "
        "Googlebot/2.1; +http://www.google.com/bot.html) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Google-Agent-desktop": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko; compatible; Google-Agent; "
        "+https://developers.google.com/crawling/docs/crawlers-fetchers/google-agent) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Chrome-control": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "robots-fetcher": "RW+ AIV audit/2.0",
}

AUDIT_USER_AGENTS: tuple[str, ...] = (
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    "ClaudeBot",
    "PerplexityBot",
    "Perplexity-User",
    "Googlebot-desktop",
    "Google-Agent-desktop",
    "DeepSeekBot",
    "Chrome-control",
)

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
# This is a memory-to-disk spool threshold, not a response-size limit.  The
# crawler reads until EOF and lets httpx's connect/read deadlines be the
# physical safety envelope.  Long but meaningful pages must never become a
# silently accepted prefix merely because they crossed an arbitrary byte cap.
LEGACY_BODY_LIMIT_BYTES = 768 * 1024
BODY_STREAM_SPOOL_BYTES = 4 * 1024 * 1024
BODY_READ_POLICY_VERSION = "aiv-body-read-policy-unbounded-eof-v2"
BODY_SAMPLE_BYTES = 4096
HEADERS_BUDGET_BYTES = 8 * 1024
MIN_GAP_PER_DOMAIN_SECONDS = 0.65
SITE_PAGE_MANIFEST_KEY = "site_page_manifest"
SITE_PAGE_MANIFEST_VERSION = "aiv-2026-08-26-site-page-manifest-v7"
SITE_PAGE_EVIDENCE_VERSION = "aiv-site-page-evidence-v1"
SITEMAP_DISCOVERY_VERSION = "aiv-sitemap-graph-v2"
TECHNICAL_MATRIX_ARTIFACT_KEY = "technical_matrix_receipt"
TECHNICAL_MATRIX_VERSION = "aiv-technical-matrix-v1"
CRAWL_ADMISSION_VERSION = "aiv-crawl-admission-v2"
PAGE_SCOPE = "bounded_representative_v2"
PAGE_RELEVANCE_ARTIFACT_KEY = "page_candidate_relevance"
COMMERCIAL_RELEVANCE_POLICY_VERSION = "aiv-commercial-relevance-v1"
COMMERCIAL_RELEVANCE_RECEIPT_VERSION = "aiv-commercial-relevance-receipt-v1"
AUDIT_PAGE_MIN = 6
AUDIT_PAGE_DEFAULT = 8
AUDIT_PAGE_HARD_MAX = 10
COMMERCIAL_RELEVANCE_THRESHOLD = 40
MAX_DISCOVERY_OCCURRENCES_PER_URL = 8
MAX_DISCOVERY_TEXT_CHARS = 420
RENDERED_NAVIGATION_WORKER_PROTOCOL = "aiv-site-preview-worker-v2"
MAX_RENDERED_NAVIGATION_WORKER_OUTPUT_BYTES = 3 * 1024 * 1024
MAX_RENDERED_NAVIGATION_WORKER_ANCHORS = 1_000
SITEMAP_MAX_DOCUMENTS = 64
SITEMAP_DISCOVERY_DEADLINE_SECONDS = 75.0
SITEMAP_MAX_FETCH_ATTEMPTS = 3
TECHNICAL_MAX_PROBE_ATTEMPTS = 3
SITE_PAGE_MAX_PROBE_ATTEMPTS = 3
_DOMAIN_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_BODY_CONTENT_RX = re.compile(r"<(article|main|section|p|h1|h2)[\s>]", re.I)
_TERMINAL_PROBE_ERRORS = frozenset(
    {"cross_domain_redirect", "redirect_loop", "unsafe_target"}
)
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429})


class SitemapFrontierIncomplete(RuntimeError):
    """A retryable sitemap node is checkpointed but not yet complete.

    The coordinator treats this as a durable continuation boundary, not as a
    terminal crawl failure. Analysis must never consume the partial frontier.
    """


class SitePageCorpusIncomplete(RuntimeError):
    """Selected browser pages are not yet a complete usable corpus."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class TechnicalMatrixIncomplete(RuntimeError):
    """At least one selected-page / crawler-UA observation is retryable."""


class CrawlAdmissionIncomplete(RuntimeError):
    """The analyzer cannot yet prove both crawl admission receipts."""


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
    body_truncated: bool = False
    body_looks_empty: bool = False
    full_text: str = ""
    content_type: str | None = None
    error_class: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _Job:
    domain: str
    user_agent_label: str
    user_agent_string: str
    target_url: str
    probe_type: ProbeType
    page_kind: str | None = None


def _body_read_policy(probe: ProbeResult) -> dict[str, Any]:
    """Describe the code-owned terminal policy for the stored observation."""

    return {
        "version": BODY_READ_POLICY_VERSION,
        "response_size_limit_bytes": None,
        "spool_to_disk_after_bytes": BODY_STREAM_SPOOL_BYTES,
        "result": "transport_incomplete" if probe.body_truncated else "complete_eof",
        # An incomplete stream is never terminal/reusable.  A continuation
        # retries it instead of blessing a prefix as the whole document.
        "terminal": not probe.body_truncated,
    }


def _current_complete_eof_policy(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("version") == BODY_READ_POLICY_VERSION
        and value.get("response_size_limit_bytes") is None
        and value.get("result") == "complete_eof"
        and value.get("terminal") is True
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _bounded_page_limit(value: Any = None) -> int:
    """Clamp the audited corpus, never the discovered candidate frontier."""

    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = int(settings.AUDIT_PAGE_LIMIT or AUDIT_PAGE_DEFAULT)
    if requested <= 0:
        requested = AUDIT_PAGE_DEFAULT
    return max(AUDIT_PAGE_MIN, min(AUDIT_PAGE_HARD_MAX, requested))


def _selected_page_records(
    pages: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    return [
        {"ordinal": index, "url": url, "page_kind": page_kind}
        for index, (url, page_kind) in enumerate(pages)
    ]


def _selected_pages_sha256(pages: list[tuple[str, str]]) -> str:
    return _json_sha256(_selected_page_records(pages))


def _normalised_user_agents(
    user_agents: list[str] | tuple[str, ...] | None,
) -> list[str]:
    return list(
        dict.fromkeys(
            label
            for label in (user_agents or AUDIT_USER_AGENTS)
            if label in USER_AGENT_STRINGS and label != "robots-fetcher"
        )
    )


def normalize_domain(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").strip().strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    if not host or len(host) > 253 or "." not in host:
        return ""
    try:
        ipaddress.ip_address(host)
        return ""
    except ValueError:
        pass
    labels = host.split(".")
    if any(not _DOMAIN_LABEL.fullmatch(label) for label in labels):
        return ""
    if labels[-1].isdigit() or len(labels[-1]) < 2:
        return ""
    return host


def normalize_domains(domains: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in domains:
        domain = normalize_domain(raw)
        if domain and domain not in seen:
            seen.add(domain)
            output.append(domain)
    return output


async def _validate_public_host(host: str) -> None:
    normalized = host.strip("[]").lower()
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(
        ".local"
    ):
        raise ValueError("unsafe_target")
    try:
        direct_ip = ipaddress.ip_address(normalized)
        addresses = [direct_ip]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(
                normalized,
                None,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise socket.gaierror(str(exc)) from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except ValueError:
                continue
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("unsafe_target")


async def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("unsafe_target")
    if parsed.username or parsed.password:
        raise ValueError("unsafe_target")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("unsafe_target")
    await _validate_public_host(parsed.hostname)


def _classify_error(exc: BaseException) -> tuple[str, str]:
    message = str(exc)[:500]
    if isinstance(exc, ValueError) and message == "unsafe_target":
        return "unsafe_target", message
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout", message
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout", message
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout", message
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout", message
    if isinstance(exc, httpx.RemoteProtocolError):
        return "protocol_error", message
    if isinstance(exc, httpx.TooManyRedirects):
        return "redirect_loop", message
    if isinstance(exc, httpx.ConnectError):
        lowered = message.lower()
        if any(token in lowered for token in ("ssl", "certificate", "tls")):
            return "tls_error", message
        if "name or service not known" in lowered or "nodename nor servname" in lowered:
            return "dns_fail", message
        return "connection_refused", message
    if isinstance(exc, socket.gaierror):
        return "dns_fail", message
    return "other", message


def _looks_binary(body: bytes, content_type: str | None) -> bool:
    if content_type and content_type.lower().startswith(
        ("image/", "audio/", "video/", "application/octet-stream", "application/pdf")
    ):
        return True
    head = body[:200]
    if not head:
        return False
    non_text = sum(1 for byte in head if byte < 9 or 13 < byte < 32)
    return non_text / len(head) > 0.1


def _trim_headers(headers: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    used = 0
    for key, value in headers.items():
        if key.lower() in {"set-cookie", "authorization", "proxy-authorization"}:
            continue
        size = len(key) + len(value) + 4
        if used + size > HEADERS_BUDGET_BYTES:
            output["_truncated"] = True
            break
        output[key] = value
        used += size
    return output


def _decode_body(body: bytes, content_type: str | None) -> str:
    encoding = "utf-8"
    if content_type and "charset=" in content_type.lower():
        match = re.search(r"charset=([^;\s]+)", content_type, re.I)
        if match:
            encoding = match.group(1).strip("\"'") or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


async def _do_probe(
    url: str,
    user_agent: str,
    timeout_seconds: int,
    concurrency: int,
    proxy_url: str | None = None,
) -> ProbeResult:
    result = ProbeResult()
    redirects: list[dict[str, Any]] = []
    headers = {"User-Agent": user_agent, **DEFAULT_HEADERS}
    timeout = httpx.Timeout(
        connect=7.0,
        read=float(timeout_seconds),
        write=7.0,
        pool=7.0,
    )
    limits = httpx.Limits(
        max_connections=max(2, concurrency * 2),
        max_keepalive_connections=max(1, concurrency),
    )
    started = time.monotonic()
    current_url = url
    try:
        async with httpx.AsyncClient(
            http2=True,
            follow_redirects=False,
            timeout=timeout,
            limits=limits,
            verify=True,
            proxy=proxy_url,
        ) as client:
            for _ in range(9):
                await _validate_public_url(current_url)
                request = client.build_request("GET", current_url, headers=headers)
                response_started = time.monotonic()
                response = await client.send(request, stream=True)
                result.ttfb_ms = int((time.monotonic() - response_started) * 1000)
                if response.is_redirect:
                    location = response.headers.get("location")
                    redirects.append(
                        {
                            "status": response.status_code,
                            "url": current_url,
                            "location": location,
                        }
                    )
                    await response.aclose()
                    if not location:
                        break
                    current_url = urljoin(current_url, location)
                    continue

                # SpooledTemporaryFile keeps ordinary responses cheap and
                # moves large bodies to disk without changing their bytes or
                # stopping at a product-defined length.  The stream is only
                # complete after EOF.
                with tempfile.SpooledTemporaryFile(
                    max_size=BODY_STREAM_SPOOL_BYTES,
                    mode="w+b",
                ) as body_stream:
                    async for chunk in response.aiter_bytes():
                        body_stream.write(chunk)
                    body_stream.seek(0)
                    body = body_stream.read()
                await response.aclose()
                content_type = response.headers.get("content-type")
                result.http_status = response.status_code
                result.final_url = str(response.url)
                result.response_headers = _trim_headers(dict(response.headers))
                result.tls_ok = True
                result.body_bytes = body
                result.response_size_bytes = len(body)
                result.content_type = content_type
                if _looks_binary(body, content_type):
                    result.body_sample = (
                        f"<binary content, content-type={content_type or 'unknown'}>"
                    )
                else:
                    result.full_text = _decode_body(body, content_type)
                    result.body_sample = result.full_text[:BODY_SAMPLE_BYTES]
                result.body_looks_empty = bool(
                    len(body) < 1500 and not _BODY_CONTENT_RX.search(result.full_text)
                )
                break
            else:
                raise httpx.TooManyRedirects(
                    "redirect limit exceeded",
                    request=httpx.Request("GET", current_url),
                )
            result.redirect_chain = redirects or None
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        error_class, message = _classify_error(exc)
        result.error_class = error_class
        result.error_message = message
        if error_class == "tls_error":
            result.tls_ok = False
    finally:
        result.total_time_ms = int((time.monotonic() - started) * 1000)
    return result


async def _probe_with_transport(
    *,
    url: str,
    user_agent: str,
    timeout_seconds: int,
    concurrency: int,
) -> tuple[ProbeResult, dict[str, Any], Proxy | None]:
    pool = get_pool()
    proxy = pool.random_proxy() if pool is not None else None
    proxy_url = proxy.url if proxy is not None else None
    transport = {
        "proxy_used": proxy is not None,
        "country": proxy.country if proxy is not None else None,
        "fallback_direct": False,
    }
    result = await _do_probe(
        url=url,
        user_agent=user_agent,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        proxy_url=proxy_url,
    )
    retryable = {
        "connect_timeout",
        "connection_refused",
        "tls_error",
        "pool_timeout",
        "protocol_error",
    }
    if (
        proxy is not None
        and result.error_class in retryable
        and settings.PROXY_FALLBACK_DIRECT
    ):
        if pool is not None:
            pool.mark_bad(proxy.url)
        result = await _do_probe(
            url=url,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
            proxy_url=None,
        )
        transport["fallback_direct"] = True
    target_domain = normalize_domain(urlparse(url).hostname or "")
    final_domain = normalize_domain(urlparse(result.final_url or "").hostname or "")
    if (
        result.error_class is None
        and target_domain
        and final_domain
        and final_domain != target_domain
    ):
        # Keep the redirect evidence, but never classify or score another
        # domain's body as if it belonged to the site entered by the user.
        result.error_class = "cross_domain_redirect"
        result.error_message = (
            f"redirected outside requested domain: {target_domain} -> {final_domain}"
        )
        result.full_text = ""
        result.body_bytes = None
        result.body_sample = ""
        result.body_looks_empty = True
    return result, transport, proxy


def _page_kind(url: str) -> str:
    path = unquote(urlparse(url).path).casefold().strip("/")
    if not path:
        return "home"
    segments = {segment for segment in path.split("/") if segment}
    if segments.intersection(
        {
            "success",
            "thanks",
            "thank-you",
            "thankyou",
            "submitted",
            "confirmation",
            "confirmed",
            "privacy",
            "terms",
            "cookies",
            "cookie-policy",
            "legal",
            "policy",
            "login",
            "signin",
            "auth",
            "cart",
            "checkout",
        }
    ):
        return "utility"
    rules = (
        ("pricing", ("price", "pricing", "tarif", "тариф", "стоим", "цены")),
        (
            "product",
            (
                "product",
                "service",
                "solution",
                "catalog",
                "direction",
                "offer",
                "industry",
                "expertise",
                "услуг",
                "продукт",
                "решени",
                "направлен",
                "отрасл",
            ),
        ),
        ("about", ("about", "company", "o-nas", "компан", "о-нас")),
        (
            "content",
            ("blog", "article", "news", "case", "media", "стат", "новост", "кейс"),
        ),
        ("faq", ("faq", "help", "question", "support", "вопрос", "помощ")),
        ("contact", ("contact", "kontakty", "контакт")),
    )
    for kind, tokens in rules:
        if any(token in path for token in tokens):
            return kind
    return "other"


def _canonical_candidate(url: str, domain: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    candidate_domain = normalize_domain(parsed.hostname)
    if candidate_domain != domain:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    lowered = unquote(path).casefold()
    segments = {segment for segment in lowered.strip("/").split("/") if segment}
    if segments.intersection(
        {
            "success",
            "thanks",
            "thank-you",
            "thankyou",
            "submitted",
            "confirmation",
            "confirmed",
            "privacy",
            "terms",
            "cookies",
            "cookie-policy",
            "legal",
            "policy",
        }
    ):
        return None
    if any(
        token in lowered
        for token in (
            "/login",
            "/signin",
            "/auth",
            "/cart",
            "/checkout",
            "/privacy",
            "/terms",
            "/cookie",
            "/legal",
            "/policy",
            "/personal-data",
            "/soglasie",
            "/agreement",
            "/wp-admin",
        )
    ):
        return None
    if re.search(r"\.(pdf|jpe?g|png|gif|webp|svg|zip|xml|json)$", lowered):
        return None
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def _compact_discovery_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", unescape(str(value or ""))).strip()
    return text[:MAX_DISCOVERY_TEXT_CHARS]


def _discovery_occurrence(
    *,
    source: str,
    anchor_text: Any = "",
    title: Any = "",
    primary_snippet: Any = "",
) -> dict[str, str]:
    return {
        "source": str(source),
        "anchor_text": _compact_discovery_text(anchor_text),
        "title": _compact_discovery_text(title),
        "primary_snippet": _compact_discovery_text(primary_snippet),
    }


def _candidate_evidence_record(
    url: str,
    *,
    source: str,
    anchor_text: Any = "",
    title: Any = "",
    primary_snippet: Any = "",
) -> dict[str, Any]:
    return {
        "url": url,
        "discoveries": [
            _discovery_occurrence(
                source=source,
                anchor_text=anchor_text,
                title=title,
                primary_snippet=primary_snippet,
            )
        ],
    }


def _canonical_candidate_evidence(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        url = value
        raw_discoveries: list[Any] = []
    elif isinstance(value, dict):
        url = value.get("url")
        raw_discoveries = (
            value.get("discoveries")
            if isinstance(value.get("discoveries"), list)
            else []
        )
    else:
        return None
    if not isinstance(url, str) or not url:
        return None

    discoveries: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_discoveries:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "unknown")
        occurrence = _discovery_occurrence(
            source=source,
            anchor_text=raw.get("anchor_text"),
            title=raw.get("title"),
            primary_snippet=raw.get("primary_snippet"),
        )
        digest = _json_sha256(occurrence)
        if digest in seen:
            continue
        seen.add(digest)
        discoveries.append(occurrence)
        if len(discoveries) >= MAX_DISCOVERY_OCCURRENCES_PER_URL:
            break
    if not discoveries:
        discoveries = [_discovery_occurrence(source="url_frontier")]
    return {"url": url, "discoveries": discoveries}


def _merge_candidate_evidence(values: list[Any]) -> list[dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for value in values:
        record = _canonical_candidate_evidence(value)
        if record is None:
            continue
        url = str(record["url"])
        target = by_url.setdefault(url, {"url": url, "discoveries": []})
        existing = {_json_sha256(item) for item in target["discoveries"]}
        for occurrence in record["discoveries"]:
            digest = _json_sha256(occurrence)
            if digest in existing:
                continue
            target["discoveries"].append(occurrence)
            existing.add(digest)
            if len(target["discoveries"]) >= MAX_DISCOVERY_OCCURRENCES_PER_URL:
                break
    return list(by_url.values())


def _anchor_primary_snippet(anchor: Any) -> str:
    container = anchor.find_parent(["li", "article", "section"])
    if container is None:
        container = anchor.parent
    return _compact_discovery_text(
        container.get_text(" ", strip=True) if container is not None else ""
    )


def _link_candidates_from_html(
    html: str,
    base_url: str,
    domain: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html or "", "html.parser")
    records: list[dict[str, Any]] = []
    for anchor in soup.find_all("a", href=True):
        candidate = _canonical_candidate(urljoin(base_url, anchor["href"]), domain)
        if candidate is None:
            continue
        records.append(
            _candidate_evidence_record(
                candidate,
                source="server_navigation",
                anchor_text=anchor.get_text(" ", strip=True),
                title=anchor.get("title") or anchor.get("aria-label") or "",
                primary_snippet=_anchor_primary_snippet(anchor),
            )
        )
    return _merge_candidate_evidence(records)


def _links_from_html(html: str, base_url: str, domain: str) -> list[str]:
    """Compatibility URL view over source-grounded navigation evidence."""

    return [
        str(record["url"])
        for record in _link_candidates_from_html(html, base_url, domain)
    ]


def _rendered_candidate_evidence(
    rendered_anchors: Any,
    *,
    domain: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in rendered_anchors if isinstance(rendered_anchors, list) else []:
        if isinstance(raw, str):
            href = raw
            anchor_text = title = primary_snippet = ""
        elif isinstance(raw, dict):
            href = str(raw.get("href") or "")
            anchor_text = raw.get("anchor_text")
            title = raw.get("title")
            primary_snippet = raw.get("primary_snippet")
        else:
            continue
        candidate = _canonical_candidate(href, domain)
        if candidate is None:
            continue
        records.append(
            _candidate_evidence_record(
                candidate,
                source="rendered_navigation",
                anchor_text=anchor_text,
                title=title,
                primary_snippet=primary_snippet,
            )
        )
    return _merge_candidate_evidence(records)


async def _rendered_navigation_worker(
    url: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = str(settings.SITE_PREVIEW_WORKER_COMMAND or "").strip()
    if not command:
        raise RuntimeError("rendered_navigation_worker_not_configured")
    process = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        "--mode",
        "navigation",
        "--url",
        url,
        "--timeout",
        str(timeout_seconds),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds + 8,
        )
    except asyncio.CancelledError:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError("rendered_navigation_worker_timeout")
    if len(stdout) > MAX_RENDERED_NAVIGATION_WORKER_OUTPUT_BYTES:
        raise ValueError("rendered_navigation_worker_output_too_large")
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = stdout.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail[:1000] or "rendered_navigation_worker_failed")
    result = json.loads(stdout.decode("utf-8"))
    anchors = result.get("anchors") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("ok") is not True
        or result.get("mode") != "navigation"
        or result.get("worker_protocol_version") != RENDERED_NAVIGATION_WORKER_PROTOCOL
        or not isinstance(anchors, list)
        or len(anchors) > MAX_RENDERED_NAVIGATION_WORKER_ANCHORS
        or result.get("anchor_count") != len(anchors)
        or type(result.get("truncated")) is not bool
    ):
        raise ValueError("rendered_navigation_worker_invalid_response")
    return result


async def _links_from_rendered_homepage(
    url: str,
    *,
    domain: str,
    timeout_seconds: int,
) -> tuple[list[str], dict[str, Any]]:
    """Best-effort rendered navigation discovery for JS-only homepages.

    It is deliberately discovery-only: the selected corpus is still fetched
    and persisted through the normal lossless HTTP path. Production uses the
    same isolated Playwright worker as the first-screen preview, but through a
    versioned navigation mode that returns bounded link evidence, not an image.
    """

    try:
        if str(settings.SITE_PREVIEW_WORKER_COMMAND or "").strip():
            result = await _rendered_navigation_worker(
                url,
                timeout_seconds=timeout_seconds,
            )
            final_url = str(result.get("final_url") or url)
            await _validate_public_url(final_url)
            if normalize_domain(urlparse(final_url).hostname or "") != normalize_domain(
                domain
            ):
                raise ValueError("rendered_navigation_cross_domain_redirect")
            evidence = _rendered_candidate_evidence(
                result.get("anchors"),
                domain=domain,
            )
            links = [str(record["url"]) for record in evidence]
            state = "bounded" if result.get("truncated") is True else "completed"
            return links, {
                "state": state,
                "candidate_count": len(links),
                "candidate_urls_sha256": _json_sha256(links),
                "candidate_evidence": evidence,
                "candidate_evidence_sha256": _json_sha256(evidence),
                "worker_protocol_version": RENDERED_NAVIGATION_WORKER_PROTOCOL,
                "worker_anchor_count": result.get("anchor_count"),
                "worker_output_truncated": bool(result.get("truncated")),
            }

        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright

        await _validate_public_url(url)
        timeout_ms = max(5_000, min(20_000, int(timeout_seconds * 1000)))
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    service_workers="block",
                    accept_downloads=False,
                    reduced_motion="reduce",
                )

                async def guarded_route(route: Any) -> None:
                    request = route.request
                    if request.resource_type in {"media", "font"}:
                        await route.abort("blockedbyclient")
                        return
                    try:
                        await _validate_public_url(request.url)
                    except (OSError, ValueError):
                        await route.abort("blockedbyclient")
                        return
                    await route.continue_()

                await context.route("**/*", guarded_route)
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                await _validate_public_url(page.url)
                try:
                    await page.wait_for_load_state("networkidle", timeout=3_500)
                except PlaywrightTimeoutError:
                    pass
                rendered_anchors = await page.eval_on_selector_all(
                    "a[href]",
                    """elements => elements.map(element => {
                        const container = element.closest('li, article, section')
                            || element.parentElement;
                        return {
                            href: element.href,
                            anchor_text: (element.innerText || element.textContent || ''),
                            title: element.getAttribute('title')
                                || element.getAttribute('aria-label') || '',
                            primary_snippet: container
                                ? (container.innerText || container.textContent || '')
                                : ''
                        };
                    })""",
                )
                evidence = _rendered_candidate_evidence(
                    rendered_anchors,
                    domain=domain,
                )
                links = [str(record["url"]) for record in evidence]
                return links, {
                    "state": "completed",
                    "candidate_count": len(links),
                    "candidate_urls_sha256": _json_sha256(links),
                    "candidate_evidence": evidence,
                    "candidate_evidence_sha256": _json_sha256(evidence),
                }
            finally:
                await browser.close()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return [], {
            "state": "failed_non_blocking",
            "candidate_count": 0,
            "error_class": type(exc).__name__,
            "error": str(exc)[:300],
        }


def _links_from_sitemap(xml: str, base_url: str, domain: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    try:
        root = ElementTree.fromstring(xml or "")
        locations = [
            str(element.text or "").strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1].lower() == "loc"
        ]
    except ElementTree.ParseError:
        locations = [
            unescape(value).strip()
            for value in re.findall(
                r"<loc\b[^>]*>(.*?)</loc>",
                xml or "",
                re.IGNORECASE | re.DOTALL,
            )
        ]
    for location in locations:
        candidate = _canonical_candidate(urljoin(base_url, location), domain)
        if candidate and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def _sitemap_document(
    xml: str,
) -> tuple[str | None, list[str], str | None]:
    """Parse one sitemap document without applying a location-count cap."""

    try:
        root = ElementTree.fromstring(xml or "")
    except ElementTree.ParseError as exc:
        return None, [], f"{type(exc).__name__}: {exc}"
    kind = root.tag.rsplit("}", 1)[-1].lower()
    locations = [
        str(element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1].lower() == "loc"
        and str(element.text or "").strip()
    ]
    if kind not in {"urlset", "sitemapindex"}:
        return kind, locations, f"unsupported sitemap root: {kind or 'empty'}"
    return kind, locations, None


def _canonical_sitemap_candidate(
    url: str,
    *,
    domain: str,
) -> str | None:
    """Canonicalise same-domain sitemap URLs separately from page URLs."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if normalize_domain(parsed.hostname) != domain:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


async def _discover_sitemap_graph(
    *,
    root_urls: list[str],
    supplied_probes: dict[str, ProbeResult] | None = None,
    domain: str,
    timeout_seconds: int,
    concurrency: int,
    prior_manifest: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Walk a broad sitemap graph under explicit time/document safety bounds.

    Page selection remains independent and capped at ten. Successful document
    receipts are checkpointed, so a transient child failure resumes without
    refetching already parsed parents.
    """

    queue: deque[tuple[str, ProbeResult | None]] = deque(
        (url, (supplied_probes or {}).get(url)) for url in dict.fromkeys(root_urls)
    )
    visited: set[str] = set()
    page_candidates: list[str] = []
    page_seen: set[str] = set()
    receipts: list[dict[str, Any]] = []
    parse_failure_count = 0
    fetch_failure_count = 0
    excluded_sitemap_count = 0
    complete = True
    resume_required = False
    bounded = False
    started_at = time.monotonic()
    prior_attempts: dict[str, int] = {}
    for item in (prior_manifest or {}).get("documents") or []:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            try:
                attempts = max(1, int(item.get("attempt") or 1))
            except (TypeError, ValueError):
                attempts = 1
            prior_attempts[item["url"]] = max(
                attempts,
                prior_attempts.get(item["url"], 0),
            )
    prior_receipts = {
        str(item.get("url") or ""): item
        for item in (prior_manifest or {}).get("documents") or []
        if isinstance(item, dict)
        and item.get("http_status") == 200
        and item.get("body_truncated") is False
        and item.get("parse_error") is None
        and item.get("kind") in {"urlset", "sitemapindex"}
    }

    def receipt_manifest(*, terminal_complete: bool) -> dict[str, Any]:
        snapshot = {
            "version": SITEMAP_DISCOVERY_VERSION,
            "root_urls": list(dict.fromkeys(root_urls)),
            "documents": list(receipts),
            "document_count": len(receipts),
            "visited_urls_sha256": hashlib.sha256(
                json.dumps(
                    [item["url"] for item in receipts],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "candidate_count": len(page_candidates),
            "candidate_urls_sha256": hashlib.sha256(
                json.dumps(
                    page_candidates,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "parse_failure_count": parse_failure_count,
            "fetch_failure_count": fetch_failure_count,
            "excluded_sitemap_count": excluded_sitemap_count,
            "complete": bool(terminal_complete),
            "resume_required": bool(resume_required),
            "bounded": bool(bounded),
            "max_documents": SITEMAP_MAX_DOCUMENTS,
            "deadline_seconds": SITEMAP_DISCOVERY_DEADLINE_SECONDS,
            "max_fetch_attempts": SITEMAP_MAX_FETCH_ATTEMPTS,
        }
        snapshot["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return snapshot

    async def save_checkpoint() -> None:
        if checkpoint is not None:
            await checkpoint(receipt_manifest(terminal_complete=False))

    while queue:
        if len(receipts) >= SITEMAP_MAX_DOCUMENTS:
            complete = False
            bounded = True
            break
        if time.monotonic() - started_at >= SITEMAP_DISCOVERY_DEADLINE_SECONDS:
            complete = False
            resume_required = True
            bounded = True
            await save_checkpoint()
            break
        sitemap_url, supplied_probe = queue.popleft()
        if sitemap_url in visited:
            continue
        visited.add(sitemap_url)
        prior_receipt = prior_receipts.get(sitemap_url)
        if supplied_probe is None and prior_receipt is not None:
            receipt = {
                **dict(prior_receipt),
                "resumed_from_receipt": True,
            }
            receipts.append(receipt)
            if receipt.get("kind") == "sitemapindex":
                for child in receipt.get("child_sitemap_urls") or []:
                    if isinstance(child, str) and child not in visited:
                        queue.append((child, None))
            else:
                for candidate in receipt.get("page_candidates") or []:
                    if isinstance(candidate, str) and candidate not in page_seen:
                        page_seen.add(candidate)
                        page_candidates.append(candidate)
            await save_checkpoint()
            continue
        probe = supplied_probe
        if probe is None:
            probe, _, _ = await _probe_with_transport(
                url=sitemap_url,
                user_agent=USER_AGENT_STRINGS["robots-fetcher"],
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
            )
        receipt: dict[str, Any] = {
            "url": sitemap_url,
            "attempt": prior_attempts.get(sitemap_url, 0) + 1,
            "http_status": probe.http_status,
            "body_sha256": hashlib.sha256(probe.body_bytes or b"").hexdigest(),
            "body_utf8_chars": len(probe.full_text),
            "body_truncated": bool(probe.body_truncated),
        }
        if probe.http_status != 200 or probe.body_truncated:
            complete = False
            fetch_failure_count += 1
            retryable_fetch = bool(
                probe.body_truncated
                or probe.http_status is None
                or probe.http_status in _TRANSIENT_HTTP_STATUSES
                or (
                    isinstance(probe.http_status, int)
                    and 500 <= probe.http_status < 600
                )
            )
            attempts = int(receipt["attempt"])
            retryable_fetch = bool(
                retryable_fetch and attempts < SITEMAP_MAX_FETCH_ATTEMPTS
            )
            resume_required = resume_required or retryable_fetch
            receipt.update(
                {
                    "kind": None,
                    "location_count": None,
                    "parse_error": probe.error_class or "incomplete sitemap response",
                    "retryable": retryable_fetch,
                    "terminal_after_attempts": bool(
                        not retryable_fetch and attempts >= SITEMAP_MAX_FETCH_ATTEMPTS
                    ),
                }
            )
            receipts.append(receipt)
            await save_checkpoint()
            continue
        kind, locations, parse_error = _sitemap_document(probe.full_text)
        receipt.update(
            {
                "kind": kind,
                "location_count": len(locations),
                "parse_error": parse_error,
                "retryable": False,
            }
        )
        receipts.append(receipt)
        if parse_error is not None:
            complete = False
            parse_failure_count += 1
            await save_checkpoint()
            continue
        if kind == "sitemapindex":
            child_sitemaps: list[str] = []
            for location in locations:
                child = _canonical_sitemap_candidate(
                    urljoin(sitemap_url, location),
                    domain=domain,
                )
                if child is None:
                    excluded_sitemap_count += 1
                    complete = False
                    continue
                child_sitemaps.append(child)
                if child not in visited:
                    queue.append((child, None))
            receipt["child_sitemap_urls"] = list(dict.fromkeys(child_sitemaps))
            receipt["page_candidates"] = []
            await save_checkpoint()
            continue
        document_candidates: list[str] = []
        for location in locations:
            candidate = _canonical_candidate(
                urljoin(sitemap_url, location),
                domain,
            )
            if candidate and candidate not in page_seen:
                page_seen.add(candidate)
                page_candidates.append(candidate)
                document_candidates.append(candidate)
        receipt["child_sitemap_urls"] = []
        receipt["page_candidates"] = document_candidates
        await save_checkpoint()

    final_manifest = receipt_manifest(terminal_complete=complete)
    return {
        "candidates": page_candidates,
        "manifest": final_manifest,
    }


def _sitemap_urls_from_robots(
    robots_text: str,
    *,
    robots_url: str,
    domain: str,
) -> list[str]:
    """Return every same-domain global or block-level Sitemap directive."""

    output: list[str] = []
    seen: set[str] = set()
    for raw_line in (robots_text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        key, separator, value = line.partition(":")
        if not separator or key.strip().casefold() != "sitemap":
            continue
        candidate = _canonical_sitemap_candidate(
            urljoin(robots_url, value.strip()),
            domain=domain,
        )
        if candidate and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def _commercial_cluster(url: str) -> str:
    segments = [
        segment
        for segment in unquote(urlparse(url).path).casefold().strip("/").split("/")
        if segment
    ]
    if not segments:
        return "home"
    generic = {
        "product",
        "products",
        "service",
        "services",
        "solution",
        "solutions",
        "direction",
        "directions",
        "catalog",
        "uslugi",
        "услуги",
        "продукты",
        "решения",
        "направления",
    }
    if segments[0] in generic and len(segments) > 1:
        return segments[1]
    return segments[0]


_COMMERCIAL_OFFER_RE = re.compile(
    r"\b(?:"
    r"services?|products?|solutions?|platforms?|software|packages?|"
    r"subscriptions?|consult(?:ing|ation)?|agency|studio|treatments?|courses?|"
    r"услуг\w*|продукт\w*|решени\w*|платформ\w*|сервис\w*|пакет\w*|"
    r"консультац\w*|агентств\w*|студи\w*|курс\w*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)
_COMMERCIAL_ACTION_RE = re.compile(
    r"\b(?:"
    r"buy|book|order|subscribe|request\s+(?:a\s+)?(?:demo|quote)|get\s+(?:a\s+)?quote|"
    r"contact\s+sales|start\s+(?:a\s+)?trial|"
    r"куп\w*|заказ\w*|запис\w*|заброниров\w*|остав\w*\s+заяв\w*|"
    r"получ\w*\s+(?:демо|расч[её]т|предложен\w*)"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)
_GENERIC_LINK_CTA_RE = re.compile(
    r"^(?:"
    r"learn\s+more|read\s+more|view|open|explore|details?|"
    r"подробнее|узнать\s+больше|смотреть|открыть|перейти"
    r")[\s.!?→›»]*$",
    re.IGNORECASE | re.UNICODE,
)
_PRICING_RE = re.compile(
    r"\b(?:price|prices|pricing|plans?|tariffs?|cost|from\s+[$€£₽]|"
    r"цен\w*|тариф\w*|стоим\w*|от\s+\d[\d\s]*(?:₽|руб))\b",
    re.IGNORECASE | re.UNICODE,
)
_CONTENT_RE = re.compile(
    r"\b(?:blog|article|news|case\s+stud(?:y|ies)|guide|insights?|trends?|"
    r"блог\w*|стат\w*|новост\w*|кейс\w*|гайд\w*|тренд\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
_ABOUT_RE = re.compile(
    r"\b(?:about\s+us|our\s+company|company|team|о\s+нас|компан\w*|команд\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
_FAQ_RE = re.compile(
    r"\b(?:faq|help|support|questions?|помощ\w*|вопрос\w*)\b",
    re.IGNORECASE | re.UNICODE,
)
_CONTACT_RE = re.compile(
    r"\b(?:contacts?|get\s+in\s+touch|контакт\w*|связат\w*)\b",
    re.IGNORECASE | re.UNICODE,
)


def _candidate_field_text(record: dict[str, Any], field: str) -> str:
    return " ".join(
        str(occurrence.get(field) or "")
        for occurrence in record.get("discoveries") or []
        if isinstance(occurrence, dict)
    )


def _evidence_page_kind(record: dict[str, Any]) -> str:
    url = str(record.get("url") or "")
    url_kind = _page_kind(url)
    if url_kind != "other":
        return url_kind
    anchor_text = _candidate_field_text(record, "anchor_text")
    title = _candidate_field_text(record, "title")
    primary_snippet = _candidate_field_text(record, "primary_snippet")
    direct_evidence = " ".join((anchor_text, title)).strip()
    contextual_evidence_allowed = bool(
        not direct_evidence or _GENERIC_LINK_CTA_RE.fullmatch(direct_evidence)
    )
    evidence = " ".join(
        (
            direct_evidence,
            primary_snippet if contextual_evidence_allowed else "",
        )
    ).strip()
    has_action = _COMMERCIAL_ACTION_RE.search(evidence) is not None
    if _PRICING_RE.search(evidence):
        return "pricing"
    if _CONTENT_RE.search(evidence) and not has_action:
        return "content"
    if _COMMERCIAL_OFFER_RE.search(evidence) or has_action:
        return "product"
    if _FAQ_RE.search(evidence):
        return "faq"
    if _CONTACT_RE.search(evidence):
        return "contact"
    if _ABOUT_RE.search(evidence):
        return "about"
    return "other"


def _commercial_relevance(record: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_candidate_evidence(record)
    if canonical is None:
        raise ValueError("Candidate evidence is invalid")
    url = str(canonical["url"])
    page_kind = _evidence_page_kind(canonical)
    components: list[dict[str, Any]] = []

    def add(signal: str, points: int, evidence: str = "") -> None:
        components.append(
            {
                "signal": signal,
                "points": points,
                "evidence": _compact_discovery_text(evidence),
            }
        )

    if page_kind == "product":
        add("page_kind_product", 45, url)
    elif page_kind == "pricing":
        add("page_kind_pricing", 50, url)

    anchor_text = _candidate_field_text(canonical, "anchor_text")
    title = _candidate_field_text(canonical, "title")
    snippet = _candidate_field_text(canonical, "primary_snippet")
    for field_name, value, points in (
        ("anchor_offer_language", anchor_text, 20),
        ("title_offer_language", title, 18),
        ("snippet_offer_language", snippet, 10),
    ):
        if _COMMERCIAL_OFFER_RE.search(value):
            add(field_name, points, value)
    combined = " ".join((anchor_text, title, snippet))
    if _COMMERCIAL_ACTION_RE.search(combined):
        add("commercial_action", 20, combined)
    if _PRICING_RE.search(combined):
        add("pricing_language", 20, combined)
    if page_kind == "content" and not _COMMERCIAL_ACTION_RE.search(combined):
        add("editorial_context", -20, combined)

    score = max(0, min(100, sum(int(item["points"]) for item in components)))
    evidence_payload = {
        "url": url,
        "discoveries": canonical["discoveries"],
    }
    return {
        **evidence_payload,
        "page_kind": page_kind,
        "commercial_relevance_score": score,
        "commercially_relevant": bool(
            page_kind in {"product", "pricing"}
            and score >= COMMERCIAL_RELEVANCE_THRESHOLD
        ),
        "score_components": components,
        "evidence_sha256": _json_sha256(evidence_payload),
    }


def _candidate_sort_key(url: str) -> tuple[Any, ...]:
    kind = _page_kind(url)
    priorities = {
        "product": 0,
        "pricing": 1,
        "about": 2,
        "content": 3,
        "faq": 4,
        "contact": 5,
        "other": 6,
    }
    path = unquote(urlparse(url).path).casefold()
    commercial_tokens = (
        "service",
        "product",
        "solution",
        "direction",
        "catalog",
        "offer",
        "услуг",
        "продукт",
        "решени",
        "направлен",
    )
    return (
        priorities.get(kind, 99),
        0 if any(token in path for token in commercial_tokens) else 1,
        path.count("/"),
        len(path),
        url,
    )


def _candidate_record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    url = str(record["url"])
    kind = str(record["page_kind"])
    priorities = {
        "product": 0,
        "pricing": 1,
        "about": 2,
        "content": 3,
        "faq": 4,
        "contact": 5,
        "other": 6,
    }
    path = unquote(urlparse(url).path).casefold()
    return (
        0 if record.get("commercially_relevant") is True else 1,
        priorities.get(kind, 99),
        -int(record.get("commercial_relevance_score") or 0),
        path.count("/"),
        len(path),
        url,
    )


def _ranked_candidate_records(
    homepage_url: str,
    candidates: list[Any],
) -> list[dict[str, Any]]:
    merged = _merge_candidate_evidence(candidates)
    ranked = [
        _commercial_relevance(record)
        for record in merged
        if record.get("url") != homepage_url
        and _page_kind(str(record.get("url") or "")) != "utility"
    ]
    return sorted(ranked, key=_candidate_record_sort_key)


def _ranked_semantic_candidates(
    homepage_url: str,
    candidates: list[Any],
) -> list[tuple[str, str]]:
    """Rank eligible URLs while retaining more candidates than we will audit."""

    return [
        (str(record["url"]), str(record["page_kind"]))
        for record in _ranked_candidate_records(homepage_url, candidates)
    ]


def _semantic_frontier_urls(
    homepage_url: str,
    candidates: list[Any],
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Choose an ordered, diverse 6..10-page audit corpus.

    The function may return fewer pages only when the eligible discovered
    frontier itself is smaller.  It never pads the corpus with legal,
    confirmation or other utility URLs.
    """

    target = _bounded_page_limit(limit)
    ranked = _ranked_semantic_candidates(homepage_url, candidates)
    selected: list[tuple[str, str]] = [(homepage_url, "home")]
    selected_urls = {homepage_url}
    used_clusters: set[str] = set()

    # Commercial detail pages carry most of the offer evidence.  Prefer
    # distinct product/service clusters before spending slots on company copy.
    commercial_budget = max(4, target - 3)
    for url, kind in ranked:
        if kind not in {"product", "pricing"}:
            continue
        cluster = _commercial_cluster(url)
        if cluster in used_clusters:
            continue
        selected.append((url, kind))
        selected_urls.add(url)
        used_clusters.add(cluster)
        if len(selected) - 1 >= commercial_budget or len(selected) >= target:
            break

    # Add diverse supporting evidence, at most one first-choice page per kind.
    for support_kind in ("about", "content", "faq", "contact"):
        if len(selected) >= target:
            break
        choice = next(
            (
                (url, kind)
                for url, kind in ranked
                if kind == support_kind and url not in selected_urls
            ),
            None,
        )
        if choice is not None:
            selected.append(choice)
            selected_urls.add(choice[0])

    # Fill remaining slots by the same deterministic commercial-first rank.
    for url, kind in ranked:
        if len(selected) >= target:
            break
        if url in selected_urls:
            continue
        selected.append((url, kind))
        selected_urls.add(url)
    return selected


def _candidate_attempt(
    page: tuple[str, str],
    *,
    outcome: str,
    page_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if outcome not in {"usable", "retryable", "unusable"}:
        raise ValueError("Candidate attempt outcome is invalid")
    return {
        "url": page[0],
        "page_kind": page[1],
        "outcome": outcome,
        "page_receipt_sha256": page_receipt_sha256,
    }


def _selection_relevance_receipt(
    *,
    homepage_url: str,
    candidates: list[Any],
    target: int,
    proposed: list[tuple[str, str]],
    attempts: list[dict[str, Any]],
    selected: list[tuple[str, str]],
) -> dict[str, Any]:
    bounded_target = _bounded_page_limit(target)
    ranked = _ranked_candidate_records(homepage_url, candidates)
    canonical_attempts = [
        {
            "ordinal": ordinal,
            "url": str(value.get("url") or ""),
            "page_kind": str(value.get("page_kind") or ""),
            "outcome": str(value.get("outcome") or ""),
            "page_receipt_sha256": (
                str(value.get("page_receipt_sha256"))
                if value.get("page_receipt_sha256")
                else None
            ),
        }
        for ordinal, value in enumerate(attempts)
    ]
    attempted_urls = {str(item.get("url") or "") for item in canonical_attempts}
    selection_exhausted = len(attempted_urls) >= 1 + len(ranked)
    retryable_attempt_count = sum(
        1 for item in canonical_attempts if item.get("outcome") == "retryable"
    )
    body = {
        "version": COMMERCIAL_RELEVANCE_RECEIPT_VERSION,
        "policy_version": COMMERCIAL_RELEVANCE_POLICY_VERSION,
        "homepage_url": homepage_url,
        "target_page_count": bounded_target,
        "commercial_relevance_threshold": COMMERCIAL_RELEVANCE_THRESHOLD,
        "candidate_count": len(ranked),
        "candidates": ranked,
        "proposed_pages": _selected_page_records(proposed),
        "attempts": canonical_attempts,
        "attempt_count": len(canonical_attempts),
        "selection_exhausted": selection_exhausted,
        "retryable_attempt_count": retryable_attempt_count,
        "terminal_exhaustion": bool(
            selection_exhausted and retryable_attempt_count == 0
        ),
        "selected_pages": _selected_page_records(selected),
        "selected_pages_sha256": _selected_pages_sha256(selected),
        "complete": True,
    }
    return {**body, "receipt_sha256": _json_sha256(body)}


def _selection_relevance_projection(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    evidence_by_url = {
        str(item.get("url") or ""): item
        for item in receipt.get("candidates") or []
        if isinstance(item, dict)
    }
    selected_evidence: list[dict[str, Any]] = []
    for page in receipt.get("selected_pages") or []:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        evidence = evidence_by_url.get(url)
        selected_evidence.append(
            {
                "ordinal": page.get("ordinal"),
                "url": url,
                "page_kind": page.get("page_kind"),
                "commercial_relevance_score": (
                    evidence.get("commercial_relevance_score")
                    if evidence is not None
                    else None
                ),
                "commercially_relevant": bool(
                    evidence is not None
                    and evidence.get("commercially_relevant") is True
                ),
                "evidence_sha256": (
                    evidence.get("evidence_sha256") if evidence is not None else None
                ),
            }
        )
    body = {
        "version": COMMERCIAL_RELEVANCE_RECEIPT_VERSION,
        "policy_version": COMMERCIAL_RELEVANCE_POLICY_VERSION,
        "artifact_key": PAGE_RELEVANCE_ARTIFACT_KEY,
        "artifact_sha256": receipt.get("receipt_sha256"),
        "candidate_count": receipt.get("candidate_count"),
        "attempt_count": receipt.get("attempt_count"),
        "selection_exhausted": receipt.get("selection_exhausted"),
        "terminal_exhaustion": receipt.get("terminal_exhaustion"),
        "selected_pages_sha256": receipt.get("selected_pages_sha256"),
        "selected_evidence": selected_evidence,
        "selected_evidence_count": len(selected_evidence),
    }
    return {**body, "projection_sha256": _json_sha256(body)}


def _selection_relevance_receipt_is_valid(
    receipt: Any,
    *,
    domain: str,
) -> bool:
    if not isinstance(receipt, dict):
        return False
    body = dict(receipt)
    receipt_sha256 = body.pop("receipt_sha256", None)
    if receipt_sha256 != _json_sha256(body):
        return False
    if (
        body.get("version") != COMMERCIAL_RELEVANCE_RECEIPT_VERSION
        or body.get("policy_version") != COMMERCIAL_RELEVANCE_POLICY_VERSION
        or body.get("commercial_relevance_threshold") != COMMERCIAL_RELEVANCE_THRESHOLD
        or body.get("complete") is not True
    ):
        return False
    homepage_url = body.get("homepage_url")
    if not isinstance(homepage_url, str):
        return False
    normalized_domain = normalize_domain(domain)
    if normalize_domain(urlparse(homepage_url).hostname or "") != normalized_domain:
        return False
    raw_candidates = body.get("candidates")
    if not isinstance(raw_candidates, list):
        return False
    candidate_inputs = [
        {
            "url": value.get("url"),
            "discoveries": value.get("discoveries"),
        }
        for value in raw_candidates
        if isinstance(value, dict)
    ]
    if len(candidate_inputs) != len(raw_candidates):
        return False
    recomputed = _ranked_candidate_records(homepage_url, candidate_inputs)
    if recomputed != raw_candidates or body.get("candidate_count") != len(recomputed):
        return False
    if any(
        normalize_domain(urlparse(str(item.get("url") or "")).hostname or "")
        != normalized_domain
        for item in recomputed
    ):
        return False
    target = body.get("target_page_count")
    if type(target) is not int or target != _bounded_page_limit(target):
        return False
    expected_proposed = _semantic_frontier_urls(
        homepage_url,
        candidate_inputs,
        target,
    )
    if body.get("proposed_pages") != _selected_page_records(expected_proposed):
        return False
    attempts = body.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return False
    normalized_attempts: list[tuple[str, str, str]] = []
    for ordinal, attempt in enumerate(attempts):
        if (
            not isinstance(attempt, dict)
            or attempt.get("ordinal") != ordinal
            or attempt.get("outcome") not in {"usable", "retryable", "unusable"}
        ):
            return False
        normalized_attempts.append(
            (
                str(attempt.get("url") or ""),
                str(attempt.get("page_kind") or ""),
                str(attempt.get("outcome")),
            )
        )
    proposed_pairs = [(url, kind) for url, kind in expected_proposed]
    if [
        pair[:2] for pair in normalized_attempts[: len(proposed_pairs)]
    ] != proposed_pairs:
        return False
    ranked_pairs = [(str(item["url"]), str(item["page_kind"])) for item in recomputed]
    remaining_pairs = [pair for pair in ranked_pairs if pair not in proposed_pairs]
    extra_pairs = [pair[:2] for pair in normalized_attempts[len(proposed_pairs) :]]
    if extra_pairs != remaining_pairs[: len(extra_pairs)]:
        return False
    attempted_urls = {pair[0] for pair in normalized_attempts}
    selection_exhausted = len(attempted_urls) >= 1 + len(recomputed)
    retryable_attempt_count = sum(
        1 for pair in normalized_attempts if pair[2] == "retryable"
    )
    if (
        body.get("attempt_count") != len(normalized_attempts)
        or body.get("selection_exhausted") is not selection_exhausted
        or body.get("retryable_attempt_count") != retryable_attempt_count
        or body.get("terminal_exhaustion")
        is not bool(selection_exhausted and retryable_attempt_count == 0)
    ):
        return False
    selected = [pair[:2] for pair in normalized_attempts if pair[2] == "usable"][
        :target
    ]
    if body.get("selected_pages") != _selected_page_records(selected):
        return False
    return body.get("selected_pages_sha256") == _selected_pages_sha256(selected)


def _selection_relevance_projection_is_valid(
    projection: Any,
    *,
    pages: list[tuple[str, str]],
) -> bool:
    if not isinstance(projection, dict):
        return False
    body = dict(projection)
    projection_sha256 = body.pop("projection_sha256", None)
    if projection_sha256 != _json_sha256(body):
        return False
    if (
        body.get("version") != COMMERCIAL_RELEVANCE_RECEIPT_VERSION
        or body.get("policy_version") != COMMERCIAL_RELEVANCE_POLICY_VERSION
        or body.get("artifact_key") != PAGE_RELEVANCE_ARTIFACT_KEY
        or body.get("selected_pages_sha256") != _selected_pages_sha256(pages)
        or not isinstance(body.get("artifact_sha256"), str)
        or type(body.get("attempt_count")) is not int
        or type(body.get("selection_exhausted")) is not bool
        or type(body.get("terminal_exhaustion")) is not bool
    ):
        return False
    selected_evidence = body.get("selected_evidence")
    if not isinstance(selected_evidence, list) or len(selected_evidence) != len(pages):
        return False
    if [
        (item.get("ordinal"), item.get("url"), item.get("page_kind"))
        for item in selected_evidence
        if isinstance(item, dict)
    ] != [(ordinal, url, page_kind) for ordinal, (url, page_kind) in enumerate(pages)]:
        return False
    return body.get("selected_evidence_count") == len(pages)


async def _store_site_page(
    run_id: str,
    *,
    url: str,
    page_kind: str,
    probe: ProbeResult,
) -> None:
    signals = extract_text_signals(probe.full_text, probe.content_type)
    signals["_body_truncated"] = probe.body_truncated
    signals["_body_read_policy"] = _body_read_policy(probe)
    signals["_probe_error_class"] = probe.error_class
    signals["_probe_final_url"] = probe.final_url
    signals["_source_body_sha256"] = hashlib.sha256(
        probe.body_bytes
        if probe.body_bytes is not None
        else probe.full_text.encode("utf-8")
    ).hexdigest()
    signals["structured_data_complete"] = bool(
        not probe.body_truncated
        and signals.get("structured_data_complete") is not False
    )
    if probe.body_truncated:
        # A prefix can prove that a signal exists, but its absence cannot prove
        # anything about the unseen remainder. Never publish definitive
        # rendering or schema conclusions from an incomplete HTML document.
        signals["render_strategy"] = "unknown"
        signals["render_strategy_confidence"] = "low"
        signals["structured_data_complete"] = False
    async with SessionLocal() as session:
        existing = (
            await session.execute(
                select(SitePage).where(
                    SitePage.run_id == run_id,
                    SitePage.url == url,
                )
            )
        ).scalar_one_or_none()
        page = existing or SitePage(run_id=run_id, url=url)
        previous_signals = (
            dict(existing.content_signals or {}) if existing is not None else {}
        )
        previous_attempts = previous_signals.get("_probe_attempts", 0)
        try:
            probe_attempts = max(0, int(previous_attempts)) + 1
        except (TypeError, ValueError):
            probe_attempts = 1
        page.canonical_url = signals.get("canonical_url")
        page.page_kind = page_kind
        page.http_status = probe.http_status
        page.title = signals.get("title")
        page.meta_description = signals.get("meta_description")
        page.main_text = signals.get("main_text_excerpt") or ""
        page.text_length = int(signals.get("main_content_length") or 0)
        page.content_signals = {
            key: value
            for key, value in signals.items()
            if key not in {"main_text_excerpt", "visible_text_excerpt"}
        }
        page.content_signals["_probe_attempts"] = probe_attempts
        page.content_signals["_probe_attempt_limit"] = SITE_PAGE_MAX_PROBE_ATTEMPTS
        if existing is None:
            session.add(page)
        await session.commit()


def _site_page_is_usable(page: SitePage, *, legacy: bool = False) -> bool:
    signals = dict(page.content_signals or {})
    if signals.get("_body_truncated") is True:
        return False
    if not legacy and not _current_complete_eof_policy(
        signals.get("_body_read_policy")
    ):
        return False
    if not isinstance(page.http_status, int) or not 200 <= page.http_status < 400:
        return False
    return bool(
        str(page.main_text or "").strip()
        or str(page.title or "").strip()
        or str(page.meta_description or "").strip()
        or int(page.text_length or 0) > 0
    )


def _site_page_is_retryable(page: SitePage) -> bool:
    signals = dict(page.content_signals or {})
    error_class = signals.get("_probe_error_class")
    try:
        attempts = max(0, int(signals.get("_probe_attempts") or 0))
    except (TypeError, ValueError):
        attempts = 0
    if attempts >= SITE_PAGE_MAX_PROBE_ATTEMPTS:
        return False
    if error_class in _TERMINAL_PROBE_ERRORS:
        return False
    if signals.get("_body_truncated") is True:
        return True
    if error_class and error_class not in _TERMINAL_PROBE_ERRORS:
        return True
    status = page.http_status
    return bool(
        status is None
        or status in _TRANSIENT_HTTP_STATUSES
        or (isinstance(status, int) and 500 <= status < 600)
    )


def site_page_evidence_sha256(
    page: SitePage,
    *,
    url: str,
    page_kind: str,
) -> str:
    """Bind every SitePage field consumed by semantic or technical analysis.

    Empty strings and missing optional text have the same downstream meaning,
    so they intentionally share one canonical representation.  The complete
    content_signals object is bound because schema, rendering and identity
    decisions read different keys from it over the lifetime of a report.
    """

    signals = (
        dict(page.content_signals) if isinstance(page.content_signals, dict) else {}
    )
    payload = {
        "version": SITE_PAGE_EVIDENCE_VERSION,
        "url": url,
        "page_kind": page_kind,
        "http_status": page.http_status,
        "title": str(page.title or ""),
        "meta_description": str(page.meta_description or ""),
        "main_text": str(page.main_text or ""),
        "text_length": (
            int(page.text_length)
            if isinstance(page.text_length, int)
            and not isinstance(page.text_length, bool)
            else 0
        ),
        "content_signals": signals,
    }
    return _json_sha256(payload)


def _site_page_receipt(
    pages: list[tuple[str, str]],
    stored_pages: dict[str, SitePage],
    *,
    legacy_snapshot: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    retryable_urls: list[str] = []
    unusable_urls: list[str] = []
    for ordinal, (url, page_kind) in enumerate(pages):
        page = stored_pages.get(url)
        usable = bool(
            page is not None and _site_page_is_usable(page, legacy=legacy_snapshot)
        )
        if page is None:
            retryable_urls.append(url)
        elif not usable:
            (retryable_urls if _site_page_is_retryable(page) else unusable_urls).append(
                url
            )
        signals = dict(page.content_signals or {}) if page is not None else {}
        main_text = str(page.main_text or "") if page is not None else ""
        title = str(page.title or "") if page is not None else ""
        meta_description = str(page.meta_description or "") if page is not None else ""
        records.append(
            {
                "ordinal": ordinal,
                "url": url,
                "page_kind": page_kind,
                "http_status": page.http_status if page is not None else None,
                "text_length": int(page.text_length or 0) if page is not None else 0,
                "content_sha256": hashlib.sha256(main_text.encode("utf-8")).hexdigest(),
                "title_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
                "meta_description_sha256": hashlib.sha256(
                    meta_description.encode("utf-8")
                ).hexdigest(),
                "content_signals_sha256": _json_sha256(signals),
                "evidence_version": SITE_PAGE_EVIDENCE_VERSION,
                "evidence_sha256": (
                    site_page_evidence_sha256(
                        page,
                        url=url,
                        page_kind=page_kind,
                    )
                    if page is not None
                    else None
                ),
                "source_body_sha256": signals.get("_source_body_sha256"),
                "body_policy_version": (
                    (signals.get("_body_read_policy") or {}).get("version")
                    if isinstance(signals.get("_body_read_policy"), dict)
                    else None
                ),
                "usable": usable,
            }
        )
    payload = {
        "expected_page_count": len(pages),
        "usable_page_count": sum(1 for item in records if item["usable"]),
        "pages": records,
        "retryable_urls": retryable_urls,
        "unusable_urls": unusable_urls,
        "legacy_snapshot": bool(legacy_snapshot),
        "complete": not retryable_urls and not unusable_urls,
    }
    payload["receipt_sha256"] = _json_sha256(payload)
    return payload


def _site_page_manifest_input(
    domain: str,
    limit: int | None = None,
    *,
    legacy_snapshot: bool = False,
) -> dict[str, Any]:
    target = _bounded_page_limit(limit)
    return {
        "domain": normalize_domain(domain),
        "page_scope": PAGE_SCOPE,
        "selection_policy": {
            "target_page_count": target,
            "min_page_count": 1 if legacy_snapshot else AUDIT_PAGE_MIN,
            "max_page_count": AUDIT_PAGE_HARD_MAX,
            "commercial_relevance_policy_version": (
                "legacy_unproven"
                if legacy_snapshot
                else COMMERCIAL_RELEVANCE_POLICY_VERSION
            ),
        },
        "legacy_snapshot": bool(legacy_snapshot),
    }


def _manifest_pages(
    output_json: dict[str, Any] | list[Any] | None,
    *,
    domain: str,
) -> list[tuple[str, str]] | None:
    if not isinstance(output_json, dict):
        return None
    raw_pages = output_json.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        return None
    if output_json.get("page_scope") != PAGE_SCOPE:
        return None

    pages: list[tuple[str, str]] = []
    seen: set[str] = set()
    normalized_domain = normalize_domain(domain)
    for ordinal, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, dict):
            return None
        url = raw_page.get("url")
        page_kind = raw_page.get("page_kind")
        if (
            raw_page.get("ordinal") != ordinal
            or not isinstance(url, str)
            or not isinstance(page_kind, str)
            or page_kind == "utility"
        ):
            return None
        parsed = urlparse(url)
        page_domain = normalize_domain(parsed.hostname or "")
        if (
            parsed.scheme not in {"http", "https"}
            or page_domain != normalized_domain
            or url in seen
        ):
            return None
        seen.add(url)
        pages.append((url, page_kind))

    if pages[0][1] != "home":
        return None
    selected_count = output_json.get("selected_count")
    expected_count = output_json.get("expected_page_count")
    if (
        type(selected_count) is not int
        or selected_count != len(pages)
        or type(expected_count) is not int
        or expected_count != len(pages)
        or not 1 <= len(pages) <= AUDIT_PAGE_HARD_MAX
    ):
        return None
    if output_json.get("selected_pages_sha256") != _selected_pages_sha256(pages):
        return None
    legacy_snapshot = output_json.get("legacy_snapshot") is True
    selection_policy = output_json.get("selection_policy")
    if not isinstance(selection_policy, dict):
        return None
    expected_relevance_policy = (
        "legacy_unproven" if legacy_snapshot else COMMERCIAL_RELEVANCE_POLICY_VERSION
    )
    if (
        selection_policy.get("commercial_relevance_policy_version")
        != expected_relevance_policy
    ):
        return None
    verified_exhaustion = output_json.get("verified_exhaustion") is True
    discovered_count = output_json.get("discovered_candidate_count")
    compatibility_count = output_json.get("discovered_count")
    if (
        type(discovered_count) is not int
        or discovered_count < len(pages)
        or type(compatibility_count) is not int
        or compatibility_count != discovered_count
    ):
        return None
    discovery_state = output_json.get("discovery_state")
    coverage_state = output_json.get("coverage_state")
    if legacy_snapshot:
        if discovery_state != "legacy_snapshot" or coverage_state != "limited":
            return None
    elif discovery_state not in {
        "complete",
        "bounded",
        "terminal_partial",
    } or coverage_state != (
        "complete"
        if discovered_count == len(pages) and discovery_state == "complete"
        else "bounded"
    ):
        return None
    receipt = output_json.get("site_page_receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("expected_page_count") != len(pages)
        or receipt.get("usable_page_count") != len(pages)
        or receipt.get("complete") is not True
        or receipt.get("legacy_snapshot") is not legacy_snapshot
    ):
        return None
    receipt_copy = dict(receipt)
    receipt_digest = receipt_copy.pop("receipt_sha256", None)
    if receipt_digest != _json_sha256(receipt_copy):
        return None
    receipt_pages = receipt.get("pages")
    if not isinstance(receipt_pages, list) or len(receipt_pages) != len(pages):
        return None
    if [
        (item.get("url"), item.get("page_kind"), item.get("ordinal"))
        for item in receipt_pages
        if isinstance(item, dict)
    ] != [(url, page_kind, ordinal) for ordinal, (url, page_kind) in enumerate(pages)]:
        return None
    relevance_projection = output_json.get("commercial_relevance_receipt")
    if legacy_snapshot:
        if relevance_projection is not None:
            return None
    elif not _selection_relevance_projection_is_valid(
        relevance_projection,
        pages=pages,
    ):
        return None
    elif output_json.get("selection_exhausted") is not relevance_projection.get(
        "selection_exhausted"
    ) or verified_exhaustion is not bool(
        len(pages) < AUDIT_PAGE_MIN
        and relevance_projection.get("terminal_exhaustion") is True
    ):
        return None
    if len(pages) < AUDIT_PAGE_MIN and not (legacy_snapshot or verified_exhaustion):
        return None
    return pages


def validated_site_page_manifest_output(
    artifact: RunArtifact | None,
    *,
    domain: str,
    allow_legacy_snapshot: bool = True,
) -> dict[str, Any] | None:
    """Validate the complete v6 bounded manifest without reading the network.

    Technical reporting uses this pure helper to select its denominator.  The
    stronger async admission gate additionally re-derives SitePage and probe
    receipts from database rows before analysis begins.
    """

    if (
        artifact is None
        or artifact.status != "completed"
        or artifact.prompt_version != SITE_PAGE_MANIFEST_VERSION
        or not isinstance(artifact.input_json, dict)
        or not isinstance(artifact.output_json, dict)
    ):
        return None
    input_json = artifact.input_json
    output = artifact.output_json
    normalized_domain = normalize_domain(domain)
    if (
        not normalized_domain
        or input_json.get("domain") != normalized_domain
        or input_json.get("page_scope") != PAGE_SCOPE
        or output.get("page_scope") != PAGE_SCOPE
        or input_json.get("selection_policy") != output.get("selection_policy")
    ):
        return None
    legacy_snapshot = output.get("legacy_snapshot") is True
    if input_json.get("legacy_snapshot") is not legacy_snapshot or (
        legacy_snapshot and not allow_legacy_snapshot
    ):
        return None
    if _manifest_pages(output, domain=normalized_domain) is None:
        return None
    return dict(output)


def _relevance_artifact_matches_projection(
    artifact: RunArtifact | None,
    projection: Any,
    *,
    domain: str,
) -> bool:
    return bool(
        artifact is not None
        and artifact.status == "completed"
        and artifact.prompt_version == COMMERCIAL_RELEVANCE_POLICY_VERSION
        and isinstance(artifact.input_json, dict)
        and artifact.input_json.get("domain") == normalize_domain(domain)
        and artifact.input_json.get("policy_version")
        == COMMERCIAL_RELEVANCE_POLICY_VERSION
        and isinstance(artifact.output_json, dict)
        and _selection_relevance_receipt_is_valid(
            artifact.output_json,
            domain=domain,
        )
        and isinstance(projection, dict)
        and projection.get("artifact_key") == PAGE_RELEVANCE_ARTIFACT_KEY
        and projection.get("artifact_sha256")
        == artifact.output_json.get("receipt_sha256")
        and projection == _selection_relevance_projection(artifact.output_json)
    )


async def _save_page_relevance_artifact(
    run_id: str,
    *,
    domain: str,
    receipt: dict[str, Any],
) -> None:
    if not _selection_relevance_receipt_is_valid(receipt, domain=domain):
        raise ValueError("Commercial relevance receipt is invalid")
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == PAGE_RELEVANCE_ARTIFACT_KEY,
                )
            )
        ).scalar_one_or_none()
        if artifact is None:
            artifact = RunArtifact(
                run_id=run_id,
                stage_key="site_discovery",
                artifact_key=PAGE_RELEVANCE_ARTIFACT_KEY,
            )
            session.add(artifact)
        artifact.stage_key = "site_discovery"
        artifact.status = "completed"
        artifact.model = None
        artifact.prompt_version = COMMERCIAL_RELEVANCE_POLICY_VERSION
        artifact.input_json = {
            "domain": normalize_domain(domain),
            "policy_version": COMMERCIAL_RELEVANCE_POLICY_VERSION,
        }
        artifact.output_json = receipt
        artifact.raw_text = None
        artifact.usage_json = {
            "candidate_count": receipt.get("candidate_count"),
            "attempt_count": len(receipt.get("attempts") or []),
        }
        artifact.error_message = None
        await session.commit()


async def _matching_site_page_manifest(
    run_id: str,
    *,
    domain: str,
    limit: int | None = None,
    allow_legacy_snapshot: bool = False,
) -> tuple[dict[str, Any], list[tuple[str, str]]] | None:
    input_json = _site_page_manifest_input(domain, limit)
    async with SessionLocal() as session:
        artifacts = list(
            (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key.in_(
                            [SITE_PAGE_MANIFEST_KEY, PAGE_RELEVANCE_ARTIFACT_KEY]
                        ),
                    )
                )
            ).scalars()
        )
        by_key = {artifact.artifact_key: artifact for artifact in artifacts}
        artifact = by_key.get(SITE_PAGE_MANIFEST_KEY)
        if (
            artifact is None
            or artifact.status != "completed"
            or artifact.prompt_version != SITE_PAGE_MANIFEST_VERSION
            or (
                artifact.input_json != input_json
                and not (
                    allow_legacy_snapshot
                    and isinstance(artifact.input_json, dict)
                    and artifact.input_json
                    == _site_page_manifest_input(
                        domain,
                        limit,
                        legacy_snapshot=True,
                    )
                )
            )
            or not isinstance(artifact.output_json, dict)
        ):
            return None
        pages = _manifest_pages(
            artifact.output_json,
            domain=domain,
        )
        if pages is None:
            return None
        if artifact.output_json.get("legacy_snapshot") is not True and not (
            _relevance_artifact_matches_projection(
                by_key.get(PAGE_RELEVANCE_ARTIFACT_KEY),
                artifact.output_json.get("commercial_relevance_receipt"),
                domain=domain,
            )
        ):
            return None
        return dict(artifact.output_json), pages


async def _partial_sitemap_manifest(
    run_id: str,
    *,
    domain: str,
    limit: int | None = None,
) -> dict[str, Any] | None:
    """Load a compatible incomplete sitemap receipt for lossless resume.

    Successful sitemap documents are immutable content-addressed receipts.
    Failed or truncated documents are deliberately absent from the reusable
    subset and will be fetched again on the next continuation.
    """

    expected_input = _site_page_manifest_input(domain, limit)
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == SITE_PAGE_MANIFEST_KEY,
                )
            )
        ).scalar_one_or_none()
    if (
        artifact is None
        or artifact.status == "completed"
        or artifact.prompt_version != SITE_PAGE_MANIFEST_VERSION
        or artifact.input_json != expected_input
        or not isinstance(artifact.output_json, dict)
    ):
        return None
    sitemap_manifest = artifact.output_json.get("sitemap_discovery")
    if not isinstance(sitemap_manifest, dict):
        return None
    if sitemap_manifest.get("version") != SITEMAP_DISCOVERY_VERSION:
        return None
    return dict(sitemap_manifest)


async def _save_site_page_manifest(
    run_id: str,
    *,
    status: str,
    input_json: dict[str, Any],
    output_json: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == SITE_PAGE_MANIFEST_KEY,
                )
            )
        ).scalar_one_or_none()
        if artifact is None:
            artifact = RunArtifact(
                run_id=run_id,
                stage_key="site_discovery",
                artifact_key=SITE_PAGE_MANIFEST_KEY,
            )
            session.add(artifact)
        artifact.stage_key = "site_discovery"
        artifact.status = status
        artifact.model = None
        artifact.prompt_version = SITE_PAGE_MANIFEST_VERSION
        artifact.input_json = input_json
        artifact.output_json = output_json
        artifact.raw_text = None
        artifact.usage_json = None
        artifact.error_message = error_message[:1000] if error_message else None
        await session.commit()


def _known_discovered_page_count(
    sitemap_xml: str,
    *,
    sitemap_status: int | None,
    homepage_url: str,
    candidates: list[str],
) -> int | None:
    """Return a denominator only for a complete, bounded URL-set sitemap."""

    if sitemap_status != 200:
        return None
    try:
        root = ElementTree.fromstring(sitemap_xml or "")
    except ElementTree.ParseError:
        return None
    if root.tag.rsplit("}", 1)[-1].lower() != "urlset":
        return None
    locations = [
        element
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1].lower() == "loc"
    ]
    discovered = {
        url for url in [homepage_url, *candidates] if _page_kind(url) != "utility"
    }
    return len(discovered)


async def _reconcile_site_pages(
    run_id: str,
    pages: list[tuple[str, str]],
    *,
    timeout_seconds: int,
    concurrency: int,
    refresh_probes: dict[str, ProbeResult] | None = None,
    prune: bool = True,
    legacy_snapshot: bool = False,
) -> dict[str, Any]:
    manifest_kinds = dict(pages)
    existing_by_url: dict[str, SitePage] = {}
    async with SessionLocal() as session:
        existing_pages = list(
            (
                await session.execute(select(SitePage).where(SitePage.run_id == run_id))
            ).scalars()
        )
        for page in existing_pages:
            page_kind = manifest_kinds.get(page.url)
            if page_kind is None:
                if prune:
                    await session.delete(page)
                continue
            page.page_kind = page_kind
            existing_by_url[page.url] = page
        await session.commit()

    refresh_probes = refresh_probes or {}
    for index, (url, page_kind) in enumerate(pages):
        probe = refresh_probes.get(url)
        existing = existing_by_url.get(url)
        if (
            existing is not None
            and _site_page_is_usable(existing, legacy=legacy_snapshot)
            and probe is None
        ):
            continue
        if probe is None:
            await update_progress(
                run_id,
                stage="site_discovery",
                percent=min(9, 3 + index),
                detail=f"Читаем разделы сайта: {index + 1} из {len(pages)}.",
                eta_seconds=max(1200, 1800 - index * 45),
            )
            if index:
                await asyncio.sleep(MIN_GAP_PER_DOMAIN_SECONDS)
            probe, _, _ = await _probe_with_transport(
                url=url,
                user_agent=USER_AGENT_STRINGS["Chrome-control"],
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
            )
        await _store_site_page(
            run_id,
            url=url,
            page_kind=page_kind,
            probe=probe,
        )
    async with SessionLocal() as session:
        stored = {
            page.url: page
            for page in (
                await session.execute(
                    select(SitePage).where(
                        SitePage.run_id == run_id,
                        SitePage.url.in_([url for url, _kind in pages]),
                    )
                )
            ).scalars()
        }
    return _site_page_receipt(
        pages,
        stored,
        legacy_snapshot=legacy_snapshot,
    )


async def _prune_site_pages(
    run_id: str,
    pages: list[tuple[str, str]],
) -> None:
    selected_urls = {url for url, _kind in pages}
    async with SessionLocal() as session:
        existing = list(
            (
                await session.execute(select(SitePage).where(SitePage.run_id == run_id))
            ).scalars()
        )
        for page in existing:
            if page.url not in selected_urls:
                await session.delete(page)
        await session.commit()


async def discover_site_pages(
    run_id: str,
    domain: str,
    *,
    limit: int | None = None,
    timeout_seconds: int,
    concurrency: int,
) -> list[tuple[str, str]]:
    target = _bounded_page_limit(limit)
    manifest_input = _site_page_manifest_input(domain, target)
    cached_sitemap_manifest: dict[str, Any] | None = None
    manifest = await _matching_site_page_manifest(
        run_id,
        domain=domain,
        limit=target,
    )
    if manifest is not None:
        output_json, selected = manifest
        receipt = await _reconcile_site_pages(
            run_id,
            selected,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        if receipt.get("complete") is True:
            refreshed_output = {**output_json, "site_page_receipt": receipt}
            await _save_site_page_manifest(
                run_id,
                status="completed",
                input_json=manifest_input,
                output_json=refreshed_output,
            )
            return selected
        cached_sitemap_manifest = (
            dict(output_json.get("sitemap_discovery"))
            if isinstance(output_json.get("sitemap_discovery"), dict)
            else None
        )
        await _save_site_page_manifest(
            run_id,
            status="running",
            input_json=manifest_input,
            output_json={**output_json, "site_page_receipt": receipt},
        )

    prior_sitemap_manifest = cached_sitemap_manifest or (
        await _partial_sitemap_manifest(
            run_id,
            domain=domain,
            limit=target,
        )
    )

    await update_progress(
        run_id,
        stage="site_discovery",
        percent=2,
        detail="Открываем главную страницу и ищем ключевые разделы.",
        eta_seconds=1800,
        status=RunStatus.crawling,
    )
    await _save_site_page_manifest(
        run_id,
        status="running",
        input_json=manifest_input,
        output_json=(
            {"sitemap_discovery": prior_sitemap_manifest}
            if prior_sitemap_manifest is not None
            else None
        ),
    )
    try:
        homepage_url = f"https://{domain}/"
        homepage_probe, _, _ = await _probe_with_transport(
            url=homepage_url,
            user_agent=USER_AGENT_STRINGS["Chrome-control"],
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        if homepage_probe.error_class in {"tls_error", "connection_refused"}:
            http_url = f"http://{domain}/"
            fallback, _, _ = await _probe_with_transport(
                url=http_url,
                user_agent=USER_AGENT_STRINGS["Chrome-control"],
                timeout_seconds=timeout_seconds,
                concurrency=concurrency,
            )
            if fallback.http_status is not None:
                homepage_probe = fallback
                homepage_url = http_url
        if homepage_probe.final_url:
            homepage_url = homepage_probe.final_url

        server_candidate_evidence = _link_candidates_from_html(
            homepage_probe.full_text,
            homepage_url,
            domain,
        )
        server_candidates = [str(record["url"]) for record in server_candidate_evidence]
        rendered_candidates: list[str] = []
        rendered_navigation: dict[str, Any] = {
            "state": "not_needed",
            "candidate_count": 0,
        }
        homepage_signals = extract_text_signals(
            homepage_probe.full_text,
            homepage_probe.content_type,
        )
        homepage_is_client_shell = (
            homepage_signals.get("render_strategy") == "client_rendered_shell"
        )
        if homepage_is_client_shell:
            (
                rendered_candidates,
                rendered_navigation,
            ) = await _links_from_rendered_homepage(
                homepage_url,
                domain=domain,
                timeout_seconds=min(timeout_seconds, 20),
            )
        raw_rendered_evidence = rendered_navigation.get("candidate_evidence")
        if raw_rendered_evidence is not None:
            if not isinstance(raw_rendered_evidence, list) or rendered_navigation.get(
                "candidate_evidence_sha256"
            ) != _json_sha256(raw_rendered_evidence):
                raise ValueError("Rendered navigation evidence receipt mismatch")
            rendered_candidate_evidence = _merge_candidate_evidence(
                raw_rendered_evidence
            )
        else:
            # Compatibility for a test double or an older in-process provider.
            # URL-only rendered candidates remain eligible, but cannot acquire
            # a commercial score without source text.
            rendered_candidate_evidence = _merge_candidate_evidence(
                [
                    _candidate_evidence_record(
                        candidate,
                        source="rendered_navigation_url_only",
                    )
                    for candidate in rendered_candidates
                ]
            )
        candidate_evidence = _merge_candidate_evidence(
            [*server_candidate_evidence, *rendered_candidate_evidence]
        )
        robots_url = urljoin(homepage_url, "/robots.txt")
        robots_probe, _, _ = await _probe_with_transport(
            url=robots_url,
            user_agent=USER_AGENT_STRINGS["robots-fetcher"],
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        advertised_sitemaps = (
            _sitemap_urls_from_robots(
                robots_probe.full_text,
                robots_url=robots_url,
                domain=domain,
            )
            if robots_probe.http_status == 200 and not robots_probe.body_truncated
            else []
        )
        sitemap_roots = advertised_sitemaps or [urljoin(homepage_url, "/sitemap.xml")]

        async def checkpoint_sitemap_graph(
            partial_manifest: dict[str, Any],
        ) -> None:
            await _save_site_page_manifest(
                run_id,
                status="running",
                input_json=manifest_input,
                output_json={"sitemap_discovery": partial_manifest},
            )

        sitemap_discovery = await _discover_sitemap_graph(
            root_urls=sitemap_roots,
            domain=domain,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
            prior_manifest=prior_sitemap_manifest,
            checkpoint=checkpoint_sitemap_graph,
        )
        sitemap_candidate_evidence = [
            _candidate_evidence_record(candidate, source="sitemap")
            for candidate in sitemap_discovery["candidates"]
        ]
        candidate_evidence = _merge_candidate_evidence(
            [*candidate_evidence, *sitemap_candidate_evidence]
        )
        candidate_urls = [str(record["url"]) for record in candidate_evidence]
        sitemap_manifest = sitemap_discovery["manifest"]
        rendered_navigation_manifest = {
            key: value
            for key, value in rendered_navigation.items()
            if key != "candidate_evidence"
        }
        discovery_receipts = {
            "sitemap_discovery": sitemap_manifest,
            "rendered_navigation_discovery": rendered_navigation_manifest,
            "candidate_frontier": {
                "candidate_count": len(candidate_urls),
                "candidate_urls_sha256": _json_sha256(candidate_urls),
                "candidate_evidence_artifact_key": PAGE_RELEVANCE_ARTIFACT_KEY,
            },
            "server_navigation_discovery": {
                "candidate_count": len(server_candidates),
                "candidate_urls_sha256": _json_sha256(server_candidates),
                "candidate_evidence_sha256": _json_sha256(server_candidate_evidence),
            },
            "robots_sitemap_discovery": {
                "robots_url": robots_url,
                "robots_http_status": robots_probe.http_status,
                "robots_body_truncated": bool(robots_probe.body_truncated),
                "advertised_sitemap_count": len(advertised_sitemaps),
                "advertised_sitemap_urls": advertised_sitemaps,
                "used_default_sitemap": not bool(advertised_sitemaps),
            },
        }
        if sitemap_manifest.get("resume_required") is True:
            await _save_site_page_manifest(
                run_id,
                status="running",
                input_json=manifest_input,
                output_json=discovery_receipts,
            )
            raise SitemapFrontierIncomplete(
                "Retryable sitemap frontier is durably checkpointed: "
                f"documents={sitemap_manifest.get('document_count')}, "
                f"fetch_failures={sitemap_manifest.get('fetch_failure_count')}"
            )
    except SitemapFrontierIncomplete:
        # Keep the exact running checkpoint. A generic failed artifact would
        # destroy the resumable graph and could later bless a partial frontier.
        raise
    except Exception as exc:
        await _save_site_page_manifest(
            run_id,
            status="failed",
            input_json=manifest_input,
            error_message=str(exc),
        )
        raise

    ranked = _ranked_semantic_candidates(homepage_url, candidate_evidence)
    proposed = _semantic_frontier_urls(homepage_url, candidate_evidence, target)
    attempted_urls = {url for url, _kind in proposed}
    receipt = await _reconcile_site_pages(
        run_id,
        proposed,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        refresh_probes={homepage_url: homepage_probe},
        prune=False,
    )
    usable = [
        (item["url"], item["page_kind"])
        for item in receipt.get("pages") or []
        if item.get("usable") is True
    ]
    attempts: list[dict[str, Any]] = []
    initial_receipts = {
        str(item.get("url") or ""): item
        for item in receipt.get("pages") or []
        if isinstance(item, dict)
    }
    retryable_initial = set(receipt.get("retryable_urls") or [])
    for page in proposed:
        item = initial_receipts.get(page[0])
        outcome = (
            "usable"
            if item is not None and item.get("usable") is True
            else "retryable"
            if page[0] in retryable_initial
            else "unusable"
        )
        attempts.append(
            _candidate_attempt(
                page,
                outcome=outcome,
                page_receipt_sha256=(_json_sha256(item) if item is not None else None),
            )
        )
    if not usable or usable[0] != (homepage_url, "home"):
        if homepage_url in (receipt.get("retryable_urls") or []):
            raise SitePageCorpusIncomplete(
                "Homepage did not yet materialise as a usable SitePage"
            )
        raise RuntimeError("Homepage is terminally unusable for semantic analysis")

    # A terminally broken candidate must not shrink an otherwise rich site.
    # Try ranked replacements without ever crossing the configured target.
    retryable_candidate_urls = set(receipt.get("retryable_urls") or [])
    for candidate in ranked:
        if len(usable) >= target:
            break
        if candidate[0] in attempted_urls:
            continue
        attempted_urls.add(candidate[0])
        candidate_receipt = await _reconcile_site_pages(
            run_id,
            [candidate],
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
            prune=False,
        )
        item = (candidate_receipt.get("pages") or [{}])[0]
        if item.get("usable") is True:
            usable.append(candidate)
            outcome = "usable"
        elif candidate_receipt.get("retryable_urls"):
            retryable_candidate_urls.add(candidate[0])
            outcome = "retryable"
        else:
            outcome = "unusable"
        attempts.append(
            _candidate_attempt(
                candidate,
                outcome=outcome,
                page_receipt_sha256=(
                    _json_sha256(item) if isinstance(item, dict) else None
                ),
            )
        )

    selected = usable[:target]
    final_receipt = await _reconcile_site_pages(
        run_id,
        selected,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        prune=True,
    )
    if final_receipt.get("complete") is not True:
        raise SitePageCorpusIncomplete(
            "Selected SitePage corpus is not completely materialised"
        )
    eligible_count = 1 + len(ranked)
    verified_exhaustion = bool(
        len(selected) < AUDIT_PAGE_MIN
        and len(attempted_urls) >= eligible_count
        and not retryable_candidate_urls
    )
    if len(selected) < AUDIT_PAGE_MIN and not verified_exhaustion:
        raise SitePageCorpusIncomplete(
            f"Only {len(selected)} usable pages; minimum is {AUDIT_PAGE_MIN}",
            retryable=bool(retryable_candidate_urls),
        )
    relevance_receipt = _selection_relevance_receipt(
        homepage_url=homepage_url,
        candidates=candidate_evidence,
        target=target,
        proposed=proposed,
        attempts=attempts,
        selected=selected,
    )
    await _save_page_relevance_artifact(
        run_id,
        domain=domain,
        receipt=relevance_receipt,
    )
    relevance_projection = _selection_relevance_projection(relevance_receipt)
    selected_digest = _selected_pages_sha256(selected)
    rendered_navigation_complete = bool(
        not homepage_is_client_shell or rendered_navigation.get("state") == "completed"
    )
    discovery_state = (
        "terminal_partial"
        if not rendered_navigation_complete
        else "complete"
        if sitemap_manifest.get("complete") is True
        else "bounded"
        if sitemap_manifest.get("bounded") is True
        else "terminal_partial"
    )
    coverage_state = (
        "complete"
        if eligible_count == len(selected) and discovery_state == "complete"
        else "bounded"
    )
    manifest_output = {
        "pages": _selected_page_records(selected),
        "expected_page_count": len(selected),
        "selected_count": len(selected),
        "selected_pages_sha256": selected_digest,
        "discovered_candidate_count": max(eligible_count, len(selected)),
        # Compatibility denominator for existing report code. It denotes the
        # discovered candidate frontier, never a site-wide page count.
        "discovered_count": max(eligible_count, len(selected)),
        "page_scope": PAGE_SCOPE,
        "selection_policy": manifest_input["selection_policy"],
        "selection_exhausted": len(attempted_urls) >= eligible_count,
        "verified_exhaustion": verified_exhaustion,
        "legacy_snapshot": False,
        "discovery_state": discovery_state,
        "coverage_state": coverage_state,
        "site_page_receipt": final_receipt,
        "commercial_relevance_receipt": relevance_projection,
        **discovery_receipts,
    }
    # Completion is intentionally the final write, after DB reconciliation and
    # receipt construction. An interrupted run therefore cannot expose a
    # manifest whose pages were never materialised.
    await _save_site_page_manifest(
        run_id,
        status="completed",
        input_json=manifest_input,
        output_json=manifest_output,
    )
    return selected


def _build_jobs(
    domains: list[str],
    ua_labels: list[str],
    page_urls: list[tuple[str, str]] | None = None,
) -> list[_Job]:
    normalized = normalize_domains(domains)
    if not normalized:
        return []
    jobs: list[_Job] = []
    for domain in normalized:
        robots_url = (
            urljoin(page_urls[0][0], "/robots.txt")
            if page_urls
            else f"https://{domain}/robots.txt"
        )
        jobs.append(
            _Job(
                domain=domain,
                user_agent_label="robots-fetcher",
                user_agent_string=USER_AGENT_STRINGS["robots-fetcher"],
                target_url=robots_url,
                probe_type=ProbeType.robots_txt,
            )
        )

    for label in ua_labels:
        user_agent = USER_AGENT_STRINGS.get(label)
        if user_agent is None or label == "robots-fetcher":
            continue
        for domain in normalized:
            targets = page_urls or [(f"https://{domain}/", "home")]
            for url, page_kind in targets:
                jobs.append(
                    _Job(
                        domain=domain,
                        user_agent_label=label,
                        user_agent_string=user_agent,
                        target_url=url,
                        probe_type=ProbeType.main_page,
                        page_kind=page_kind,
                    )
                )
    return jobs


async def _completed_probe_keys(run_id: str) -> set[tuple[str, str, str]]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(
                DomainProbe.target_url,
                DomainProbe.user_agent_label,
                DomainProbe.probe_type,
                DomainProbe.error_class,
                DomainProbe.http_status,
                DomainProbe.id,
                DomainProbe.content_signals,
                DomainProbe.response_size_bytes,
            )
            .where(DomainProbe.run_id == run_id)
            .order_by(DomainProbe.id)
        )
        latest: dict[
            tuple[str, str, str],
            tuple[str | None, int | None, bool, Any],
        ] = {}
        attempts: dict[tuple[str, str, str], int] = {}
        for (
            url,
            label,
            probe_type,
            error_class,
            http_status,
            _probe_id,
            content_signals,
            response_size_bytes,
        ) in result.all():
            body_read_policy = (content_signals or {}).get("_body_read_policy")
            key = (url, label, probe_type.value)
            attempts[key] = attempts.get(key, 0) + 1
            latest[key] = (
                error_class,
                http_status,
                bool((content_signals or {}).get("_body_truncated"))
                or (
                    body_read_policy is None
                    and response_size_bytes == LEGACY_BODY_LIMIT_BYTES
                ),
                body_read_policy,
            )
        return {
            key
            for key, (
                error_class,
                http_status,
                body_truncated,
                body_read_policy,
            ) in latest.items()
            if _probe_result_is_reusable(
                error_class,
                http_status,
                body_truncated=body_truncated,
                body_read_policy=body_read_policy,
            )
            or attempts.get(key, 0) >= TECHNICAL_MAX_PROBE_ATTEMPTS
        }


def _probe_result_is_reusable(
    error_class: str | None,
    http_status: int | None,
    *,
    body_truncated: bool = False,
    body_read_policy: Any = None,
) -> bool:
    if body_truncated:
        return False
    if error_class:
        return error_class in _TERMINAL_PROBE_ERRORS
    if http_status is None:
        return False
    if not _current_complete_eof_policy(body_read_policy):
        return False
    return http_status not in _TRANSIENT_HTTP_STATUSES and not 500 <= http_status < 600


def _technical_matrix_input(
    domain: str,
    pages: list[tuple[str, str]],
    user_agents: list[str] | tuple[str, ...] | None,
    *,
    legacy_snapshot: bool = False,
) -> dict[str, Any]:
    labels = _normalised_user_agents(user_agents)
    return {
        "domain": normalize_domain(domain),
        "selected_pages_sha256": _selected_pages_sha256(pages),
        "user_agents": labels,
        "user_agents_sha256": _json_sha256(labels),
        "expected_cell_count": len(pages) * len(labels),
        "legacy_snapshot": bool(legacy_snapshot),
    }


def _technical_cell_state(
    probe: DomainProbe | None,
    *,
    attempt_count: int = 0,
    legacy_snapshot: bool = False,
) -> str:
    if probe is None:
        return "missing"
    signals = dict(probe.content_signals or {})
    if signals.get("_body_truncated") is True:
        return (
            "terminal_blocked"
            if attempt_count >= TECHNICAL_MAX_PROBE_ATTEMPTS
            else "retryable"
        )
    if probe.error_class:
        if probe.error_class in _TERMINAL_PROBE_ERRORS:
            return "terminal_blocked"
        return (
            "terminal_blocked"
            if attempt_count >= TECHNICAL_MAX_PROBE_ATTEMPTS
            else "retryable"
        )
    status = probe.http_status
    if status is None:
        return (
            "terminal_blocked"
            if attempt_count >= TECHNICAL_MAX_PROBE_ATTEMPTS
            else "retryable"
        )
    if status in _TRANSIENT_HTTP_STATUSES or 500 <= status < 600:
        return (
            "terminal_blocked"
            if attempt_count >= TECHNICAL_MAX_PROBE_ATTEMPTS
            else "retryable"
        )
    policy = signals.get("_body_read_policy")
    if not legacy_snapshot and not _current_complete_eof_policy(policy):
        return (
            "terminal_blocked"
            if attempt_count >= TECHNICAL_MAX_PROBE_ATTEMPTS
            else "retryable"
        )
    return "success" if 200 <= status < 400 else "terminal_blocked"


def _technical_matrix_receipt(
    pages: list[tuple[str, str]],
    user_agents: list[str],
    latest: dict[tuple[str, str], DomainProbe],
    *,
    attempt_counts: dict[tuple[str, str], int] | None = None,
    legacy_snapshot: bool = False,
) -> dict[str, Any]:
    attempt_counts = attempt_counts or {}
    cells: list[dict[str, Any]] = []
    for page_ordinal, (url, page_kind) in enumerate(pages):
        for ua_ordinal, label in enumerate(user_agents):
            probe = latest.get((url, label))
            state = _technical_cell_state(
                probe,
                attempt_count=attempt_counts.get((url, label), 0),
                legacy_snapshot=legacy_snapshot,
            )
            cells.append(
                {
                    "ordinal": len(cells),
                    "page_ordinal": page_ordinal,
                    "ua_ordinal": ua_ordinal,
                    "url": url,
                    "page_kind": page_kind,
                    "user_agent": label,
                    "probe_id": probe.id if probe is not None else None,
                    "attempt_count": attempt_counts.get((url, label), 0),
                    "http_status": probe.http_status if probe is not None else None,
                    "error_class": probe.error_class if probe is not None else None,
                    "state": state,
                    "terminal_reason": (
                        "retry_exhausted"
                        if state == "terminal_blocked"
                        and probe is not None
                        and probe.error_class not in _TERMINAL_PROBE_ERRORS
                        and (
                            probe.http_status is None
                            or probe.http_status in _TRANSIENT_HTTP_STATUSES
                            or (
                                isinstance(probe.http_status, int)
                                and 500 <= probe.http_status < 600
                            )
                        )
                        else None
                    ),
                }
            )
    terminal = [
        item for item in cells if item["state"] in {"success", "terminal_blocked"}
    ]
    payload = {
        "selected_pages_sha256": _selected_pages_sha256(pages),
        "user_agents_sha256": _json_sha256(user_agents),
        "expected_cell_count": len(cells),
        "terminal_cell_count": len(terminal),
        "success_cell_count": sum(1 for item in cells if item["state"] == "success"),
        "terminal_blocked_cell_count": sum(
            1 for item in cells if item["state"] == "terminal_blocked"
        ),
        "retryable_cell_count": sum(
            1 for item in cells if item["state"] == "retryable"
        ),
        "missing_cell_count": sum(1 for item in cells if item["state"] == "missing"),
        "cells": cells,
        "legacy_snapshot": bool(legacy_snapshot),
        "complete": len(terminal) == len(cells) and bool(cells),
    }
    payload["receipt_sha256"] = _json_sha256(payload)
    return payload


async def _latest_main_probes(
    run_id: str,
    pages: list[tuple[str, str]],
    user_agents: list[str],
) -> dict[tuple[str, str], DomainProbe]:
    urls = [url for url, _kind in pages]
    if not urls or not user_agents:
        return {}
    async with SessionLocal() as session:
        probes = list(
            (
                await session.execute(
                    select(DomainProbe)
                    .where(
                        DomainProbe.run_id == run_id,
                        DomainProbe.probe_type == ProbeType.main_page,
                        DomainProbe.target_url.in_(urls),
                        DomainProbe.user_agent_label.in_(user_agents),
                    )
                    .order_by(DomainProbe.id)
                )
            ).scalars()
        )
    latest: dict[tuple[str, str], DomainProbe] = {}
    for probe in probes:
        latest[(probe.target_url, probe.user_agent_label)] = probe
    return latest


async def _main_probe_attempt_counts(
    run_id: str,
    pages: list[tuple[str, str]],
    user_agents: list[str],
) -> dict[tuple[str, str], int]:
    urls = [url for url, _kind in pages]
    if not urls or not user_agents:
        return {}
    async with SessionLocal() as session:
        rows = list(
            await session.execute(
                select(
                    DomainProbe.target_url,
                    DomainProbe.user_agent_label,
                ).where(
                    DomainProbe.run_id == run_id,
                    DomainProbe.probe_type == ProbeType.main_page,
                    DomainProbe.target_url.in_(urls),
                    DomainProbe.user_agent_label.in_(user_agents),
                )
            )
        )
    counts: dict[tuple[str, str], int] = {}
    for url, label in rows:
        counts[(url, label)] = counts.get((url, label), 0) + 1
    return counts


async def _save_technical_matrix_artifact(
    run_id: str,
    *,
    status: str,
    input_json: dict[str, Any],
    output_json: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    async with SessionLocal() as session:
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == TECHNICAL_MATRIX_ARTIFACT_KEY,
                )
            )
        ).scalar_one_or_none()
        if artifact is None:
            artifact = RunArtifact(
                run_id=run_id,
                stage_key="technical_access",
                artifact_key=TECHNICAL_MATRIX_ARTIFACT_KEY,
            )
            session.add(artifact)
        artifact.stage_key = "technical_access"
        artifact.status = status
        artifact.model = None
        artifact.prompt_version = TECHNICAL_MATRIX_VERSION
        artifact.input_json = input_json
        artifact.output_json = output_json
        artifact.raw_text = None
        artifact.usage_json = None
        artifact.error_message = error_message[:1000] if error_message else None
        await session.commit()


async def save_technical_matrix_receipt(
    run_id: str,
    *,
    domain: str,
    pages: list[tuple[str, str]],
    user_agents: list[str] | tuple[str, ...] | None,
    legacy_snapshot: bool = False,
) -> dict[str, Any]:
    labels = _normalised_user_agents(user_agents)
    input_json = _technical_matrix_input(
        domain,
        pages,
        labels,
        legacy_snapshot=legacy_snapshot,
    )
    latest = await _latest_main_probes(run_id, pages, labels)
    attempt_counts = await _main_probe_attempt_counts(run_id, pages, labels)
    receipt = _technical_matrix_receipt(
        pages,
        labels,
        latest,
        attempt_counts=attempt_counts,
        legacy_snapshot=legacy_snapshot,
    )
    await _save_technical_matrix_artifact(
        run_id,
        status="completed" if receipt["complete"] else "running",
        input_json=input_json,
        output_json=receipt,
    )
    return receipt


def _receipt_digest_is_valid(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    copy = dict(payload)
    digest = copy.pop("receipt_sha256", None)
    return isinstance(digest, str) and digest == _json_sha256(copy)


async def require_crawl_admission(
    run_id: str,
    *,
    domain: str,
    user_agents: list[str] | tuple[str, ...] | None = None,
    allow_legacy_snapshot: bool = False,
) -> dict[str, Any]:
    """Prove the exact page corpus and terminal technical denominator.

    This is the public analyzer admission API.  It re-derives both receipts
    from current database rows, so a stale JSON artifact cannot bless deleted
    pages, changed text or a matrix with missing page/UA pairs.
    """

    normalized_domain = normalize_domain(domain)
    async with SessionLocal() as session:
        artifacts = {
            artifact.artifact_key: artifact
            for artifact in (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key.in_(
                            [
                                SITE_PAGE_MANIFEST_KEY,
                                PAGE_RELEVANCE_ARTIFACT_KEY,
                                TECHNICAL_MATRIX_ARTIFACT_KEY,
                            ]
                        ),
                    )
                )
            ).scalars()
        }
    manifest = artifacts.get(SITE_PAGE_MANIFEST_KEY)
    relevance = artifacts.get(PAGE_RELEVANCE_ARTIFACT_KEY)
    matrix = artifacts.get(TECHNICAL_MATRIX_ARTIFACT_KEY)
    if (
        manifest is None
        or manifest.status != "completed"
        or manifest.prompt_version != SITE_PAGE_MANIFEST_VERSION
        or not isinstance(manifest.input_json, dict)
        or manifest.input_json.get("domain") != normalized_domain
        or manifest.input_json.get("page_scope") != PAGE_SCOPE
        or not isinstance(manifest.output_json, dict)
    ):
        raise CrawlAdmissionIncomplete("site_page_manifest is not admissible")
    legacy_snapshot = bool(manifest.output_json.get("legacy_snapshot") is True)
    if legacy_snapshot and not allow_legacy_snapshot:
        raise CrawlAdmissionIncomplete("legacy snapshot requires explicit admission")
    pages = _manifest_pages(manifest.output_json, domain=normalized_domain)
    if pages is None:
        raise CrawlAdmissionIncomplete("site_page_manifest contract mismatch")
    if not legacy_snapshot and not _relevance_artifact_matches_projection(
        relevance,
        manifest.output_json.get("commercial_relevance_receipt"),
        domain=normalized_domain,
    ):
        raise CrawlAdmissionIncomplete(
            "commercial relevance receipt is not bound to manifest"
        )
    async with SessionLocal() as session:
        stored_pages = {
            page.url: page
            for page in (
                await session.execute(
                    select(SitePage).where(
                        SitePage.run_id == run_id,
                        SitePage.url.in_([url for url, _kind in pages]),
                    )
                )
            ).scalars()
        }
    current_page_receipt = _site_page_receipt(
        pages,
        stored_pages,
        legacy_snapshot=legacy_snapshot,
    )
    saved_page_receipt = manifest.output_json.get("site_page_receipt")
    if (
        current_page_receipt.get("complete") is not True
        or not _receipt_digest_is_valid(saved_page_receipt)
        or current_page_receipt.get("receipt_sha256")
        != saved_page_receipt.get("receipt_sha256")
    ):
        raise CrawlAdmissionIncomplete("SitePage rows no longer match manifest")
    if (
        matrix is None
        or matrix.status != "completed"
        or matrix.prompt_version != TECHNICAL_MATRIX_VERSION
        or not isinstance(matrix.input_json, dict)
        or not isinstance(matrix.output_json, dict)
        or matrix.input_json.get("domain") != normalized_domain
        or matrix.input_json.get("selected_pages_sha256")
        != manifest.output_json.get("selected_pages_sha256")
        or matrix.input_json.get("legacy_snapshot") is not legacy_snapshot
    ):
        raise CrawlAdmissionIncomplete("technical matrix is not bound to manifest")
    saved_labels = matrix.input_json.get("user_agents")
    if not isinstance(saved_labels, list) or not saved_labels:
        raise CrawlAdmissionIncomplete("technical matrix has no configured UAs")
    labels = _normalised_user_agents(saved_labels)
    if labels != saved_labels:
        raise CrawlAdmissionIncomplete("technical matrix UA list is invalid")
    if user_agents is not None and labels != _normalised_user_agents(user_agents):
        raise CrawlAdmissionIncomplete("technical matrix UA configuration changed")
    latest = await _latest_main_probes(run_id, pages, labels)
    attempt_counts = await _main_probe_attempt_counts(run_id, pages, labels)
    current_matrix = _technical_matrix_receipt(
        pages,
        labels,
        latest,
        attempt_counts=attempt_counts,
        legacy_snapshot=legacy_snapshot,
    )
    if (
        current_matrix.get("complete") is not True
        or not _receipt_digest_is_valid(matrix.output_json)
        or current_matrix.get("receipt_sha256")
        != matrix.output_json.get("receipt_sha256")
    ):
        raise CrawlAdmissionIncomplete(
            "technical matrix contains missing, retryable or changed cells"
        )
    admission = {
        "version": CRAWL_ADMISSION_VERSION,
        "run_id": run_id,
        "domain": normalized_domain,
        "page_scope": PAGE_SCOPE,
        "legacy_snapshot": legacy_snapshot,
        "coverage_state": manifest.output_json.get("coverage_state"),
        "page_count": len(pages),
        "selected_pages_sha256": manifest.output_json.get("selected_pages_sha256"),
        "site_page_receipt_sha256": current_page_receipt["receipt_sha256"],
        "commercial_relevance_receipt_sha256": (
            None if legacy_snapshot else relevance.output_json.get("receipt_sha256")
        ),
        "user_agents": labels,
        "user_agents_sha256": current_matrix["user_agents_sha256"],
        "expected_cell_count": current_matrix["expected_cell_count"],
        "terminal_cell_count": current_matrix["terminal_cell_count"],
        "technical_matrix_receipt_sha256": current_matrix["receipt_sha256"],
    }
    admission["admission_sha256"] = _json_sha256(admission)
    return admission


async def bootstrap_legacy_crawl_admission(
    run_id: str,
    *,
    domain: str,
    user_agents: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Bind an immutable saved 1..10-page snapshot without any network I/O.

    Intended only for reprocessing previously completed answers.  Missing
    SitePages or probe cells are never fabricated and make the bootstrap fail.
    """

    normalized_domain = normalize_domain(domain)
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        stored = list(
            (
                await session.execute(
                    select(SitePage)
                    .where(SitePage.run_id == run_id)
                    .order_by(SitePage.id)
                )
            ).scalars()
        )
        observed_labels = list(
            (
                await session.execute(
                    select(DomainProbe.user_agent_label)
                    .where(
                        DomainProbe.run_id == run_id,
                        DomainProbe.probe_type == ProbeType.main_page,
                    )
                    .distinct()
                )
            ).scalars()
        )
    # Resolve page kind once.  For a legacy snapshot, persisted metadata is
    # the historical truth: applying today's URL rules to eligibility as well
    # as the manifest could silently remove a page that contributed to the
    # saved downstream evidence (for example, an old content page at
    # /privacy).  The current classifier is only a missing-metadata fallback.
    stored_with_kinds = [
        (
            page,
            str(page.page_kind or "").strip() or _page_kind(page.url),
        )
        for page in stored
    ]
    eligible = [
        (page, effective_kind)
        for page, effective_kind in stored_with_kinds
        if effective_kind != "utility"
        and normalize_domain(urlparse(page.url).hostname or "") == normalized_domain
        and _site_page_is_usable(page, legacy=True)
    ]
    if not 1 <= len(eligible) <= AUDIT_PAGE_HARD_MAX:
        raise CrawlAdmissionIncomplete(
            "legacy snapshot must contain 1..10 usable same-domain pages"
        )
    eligible.sort(
        key=lambda item: (
            0 if item[1] == "home" else 1,
            _candidate_sort_key(item[0].url),
            item[0].id,
        )
    )
    if eligible[0][1] != "home":
        raise CrawlAdmissionIncomplete("legacy snapshot has no usable homepage")
    pages = [(page.url, effective_kind) for page, effective_kind in eligible]
    page_map = {page.url: page for page, _effective_kind in eligible}
    page_receipt = _site_page_receipt(
        pages,
        page_map,
        legacy_snapshot=True,
    )
    if page_receipt.get("complete") is not True:
        raise CrawlAdmissionIncomplete("legacy SitePage corpus is incomplete")
    labels_source: list[str] | tuple[str, ...] | None = user_agents
    if labels_source is None and run is not None:
        configured = (run.config_json or {}).get("user_agents")
        if isinstance(configured, list) and configured:
            labels_source = configured
    if labels_source is None:
        observed = set(observed_labels)
        labels_source = [label for label in AUDIT_USER_AGENTS if label in observed]
    labels = _normalised_user_agents(labels_source)
    if not labels:
        raise CrawlAdmissionIncomplete("legacy snapshot has no observed crawler UAs")
    manifest_input = _site_page_manifest_input(
        normalized_domain,
        len(pages),
        legacy_snapshot=True,
    )
    manifest_output = {
        "pages": _selected_page_records(pages),
        "expected_page_count": len(pages),
        "selected_count": len(pages),
        "selected_pages_sha256": _selected_pages_sha256(pages),
        "discovered_candidate_count": len(pages),
        "discovered_count": len(pages),
        "page_scope": PAGE_SCOPE,
        "selection_policy": manifest_input["selection_policy"],
        "selection_exhausted": False,
        "verified_exhaustion": True,
        "legacy_snapshot": True,
        "discovery_state": "legacy_snapshot",
        "coverage_state": "limited",
        "site_page_receipt": page_receipt,
    }
    await _save_site_page_manifest(
        run_id,
        status="completed",
        input_json=manifest_input,
        output_json=manifest_output,
    )
    matrix_receipt = await save_technical_matrix_receipt(
        run_id,
        domain=normalized_domain,
        pages=pages,
        user_agents=labels,
        legacy_snapshot=True,
    )
    if matrix_receipt.get("complete") is not True:
        raise CrawlAdmissionIncomplete(
            "legacy technical matrix has missing or retryable cells"
        )
    return await require_crawl_admission(
        run_id,
        domain=normalized_domain,
        user_agents=labels,
        allow_legacy_snapshot=True,
    )


async def _persist_probe(
    run_id: str,
    job: _Job,
    probe: ProbeResult,
    transport: dict[str, Any],
) -> None:
    await assert_run_lease(run_id)
    signals: dict[str, Any] | None = None
    extractable_length: int | None = None
    if job.probe_type is ProbeType.main_page and probe.http_status is not None:
        signals = extract_text_signals(probe.full_text, probe.content_type)
        extractable_length = int(signals.get("extractable_text_length") or 0)
        main_excerpt = str(signals.pop("main_text_excerpt", ""))
        signals["content_fingerprint"] = (
            hashlib.sha256(main_excerpt.encode("utf-8")).hexdigest()
            if main_excerpt
            else None
        )
        signals["main_text_excerpt"] = main_excerpt[:2400]
        signals["visible_text_excerpt"] = str(
            signals.get("visible_text_excerpt") or ""
        )[:1200]
    if signals is None:
        signals = {}
    signals["_body_truncated"] = probe.body_truncated
    signals["_body_read_policy"] = _body_read_policy(probe)
    signals["structured_data_complete"] = bool(
        not probe.body_truncated
        and signals.get("structured_data_complete") is not False
    )
    signals["_transport"] = transport

    markers, challenge = detect_protections(
        status=probe.http_status,
        headers=probe.response_headers,
        body_text=probe.full_text,
        final_url=probe.final_url or job.target_url,
        tls_ok=probe.tls_ok,
        domain=job.domain,
        body_looks_empty=probe.body_looks_empty,
        probe_type=job.probe_type.value,
        content_signals=signals,
    )
    async with SessionLocal() as session:
        session.add(
            DomainProbe(
                run_id=run_id,
                domain=job.domain,
                user_agent_label=job.user_agent_label,
                user_agent_string=job.user_agent_string,
                target_url=job.target_url,
                page_kind=job.page_kind,
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
                content_extractable_text_length=extractable_length,
                content_signals=signals,
                error_class=probe.error_class,
                error_message=probe.error_message,
            )
        )
        if job.probe_type is ProbeType.robots_txt:
            rules = (
                parse_robots(probe.full_text)
                if probe.error_class is None and probe.http_status == 200
                else parse_robots_unavailable(
                    probe.error_class
                    or (f"http_{probe.http_status}" if probe.http_status else "unknown")
                )
            )
            for bot, rule, raw in rules:
                session.add(
                    RobotsRule(
                        run_id=run_id,
                        domain=job.domain,
                        bot_name=bot,
                        rule=rule,
                        raw_directives=raw or None,
                    )
                )
        await session.commit()


async def _run_crawl_impl(
    run_id: str,
    *,
    lease_owner: str | None = None,
) -> None:
    try:
        async with SessionLocal() as session:
            claimed_at = datetime.now(timezone.utc)
            claim_conditions = [
                Run.id == run_id,
                Run.status == RunStatus.pending,
            ]
            if lease_owner is not None:
                claim_conditions.extend(
                    [
                        Run.execution_slot == 1,
                        Run.lease_owner == lease_owner,
                    ]
                )
            claimed = await session.execute(
                update(Run)
                .where(*claim_conditions)
                .values(
                    status=RunStatus.crawling,
                    state_revision=Run.state_revision + 1,
                    state_changed_at=claimed_at,
                )
            )
            await session.commit()
            if claimed.rowcount != 1:
                return
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                return
            config = dict(run.config_json or {})
            domain = run.domain or normalize_domain(
                str((config.get("domains") or [""])[0])
            )
            if not domain:
                await fail_run(
                    run_id,
                    "Не удалось распознать домен. Создайте новую проверку.",
                )
                return
            run.domain = domain
            await session.commit()

        concurrency = max(1, min(12, int(config.get("concurrency") or 6)))
        timeout_seconds = max(5, min(60, int(config.get("timeout_seconds") or 20)))
        page_limit = _bounded_page_limit(config.get("page_limit"))
        user_agents = _normalised_user_agents(config.get("user_agents"))

        pages = await discover_site_pages(
            run_id,
            domain,
            limit=page_limit,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        preview_task = asyncio.create_task(
            capture_site_preview(
                run_id,
                domain=domain,
                source_url=pages[0][0],
                validate_url=_validate_public_url,
            ),
            name=f"aiv-site-preview-{run_id}",
        )
        try:
            await update_progress(
                run_id,
                stage="technical_access",
                percent=10,
                detail="Сравниваем ответы сайта для поисковых и ИИ-краулеров.",
                eta_seconds=1500,
                status=RunStatus.crawling,
            )
            jobs = _build_jobs([domain], user_agents, pages)
            completed_keys = await _completed_probe_keys(run_id)
            pending_jobs = [
                job
                for job in jobs
                if (
                    job.target_url,
                    job.user_agent_label,
                    job.probe_type.value,
                )
                not in completed_keys
            ]
            already_done = len(jobs) - len(pending_jobs)
            semaphore = asyncio.Semaphore(concurrency)
            domain_lock = asyncio.Lock()
            last_request = 0.0
            progress_lock = asyncio.Lock()
            done_count = already_done

            async def worker(job: _Job) -> None:
                nonlocal last_request, done_count
                async with semaphore:
                    async with domain_lock:
                        delay = MIN_GAP_PER_DOMAIN_SECONDS - (
                            time.monotonic() - last_request
                        )
                        if delay > 0:
                            await asyncio.sleep(delay)
                        last_request = time.monotonic()
                    probe, transport, _ = await _probe_with_transport(
                        url=job.target_url,
                        user_agent=job.user_agent_string,
                        timeout_seconds=timeout_seconds,
                        concurrency=concurrency,
                    )
                    await _persist_probe(run_id, job, probe, transport)
                    async with progress_lock:
                        done_count += 1
                        ratio = done_count / max(1, len(jobs))
                        percent = 10 + round(ratio * 17)
                        eta = max(1000, int(1500 - ratio * 420))
                        detail = (
                            "Проверяем, совпадает ли содержимое для обычного браузера "
                            f"и ИИ-краулеров: {round(ratio * 100)}%."
                        )
                        await update_progress(
                            run_id,
                            stage="technical_access",
                            percent=percent,
                            detail=detail,
                            eta_seconds=eta,
                        )

            await asyncio.gather(*(worker(job) for job in pending_jobs))
            await preview_task
        finally:
            if not preview_task.done():
                preview_task.cancel()
            await asyncio.gather(preview_task, return_exceptions=True)
        matrix_receipt = await save_technical_matrix_receipt(
            run_id,
            domain=domain,
            pages=pages,
            user_agents=user_agents,
        )
        if matrix_receipt.get("complete") is not True:
            raise TechnicalMatrixIncomplete(
                "Technical page/UA matrix still has "
                f"{matrix_receipt.get('retryable_cell_count')} retryable and "
                f"{matrix_receipt.get('missing_cell_count')} missing cells"
            )
        await require_crawl_admission(
            run_id,
            domain=domain,
            user_agents=user_agents,
        )
        await update_progress(
            run_id,
            stage="technical_access",
            percent=28,
            detail="Технический аудит завершён. Переходим к смыслу сайта.",
            eta_seconds=1080,
        )
        # Local import keeps the admission API importable from analyzer.py
        # without a crawler/analyzer module cycle.
        from app.services.analyzer import analyze_run

        await analyze_run(run_id)
    except asyncio.CancelledError:
        raise
    except (SitemapFrontierIncomplete, TechnicalMatrixIncomplete) as exc:
        logger.warning(
            "Crawl evidence for run %s is incomplete; continuation queued: %s",
            run_id,
            exc,
        )
        await update_progress(
            run_id,
            stage="site_discovery",
            percent=2,
            detail=(
                "Часть технических наблюдений временно недоступна. "
                "Проверка продолжится с сохранённого места."
            ),
            eta_seconds=None,
            status=RunStatus.crawling,
        )
        # Returning with an active run lets RunCoordinator release the lease
        # back to pending and start the next attempt from the saved receipts.
        return
    except SitePageCorpusIncomplete as exc:
        if not exc.retryable:
            raise
        logger.warning(
            "Page corpus for run %s is incomplete; continuation queued: %s",
            run_id,
            exc,
        )
        await update_progress(
            run_id,
            stage="site_discovery",
            percent=4,
            detail=(
                "Не все выбранные страницы прочитались с первого раза. "
                "Проверка продолжится с сохранённого места."
            ),
            eta_seconds=None,
            status=RunStatus.crawling,
        )
        return
    except Exception:
        logger.exception("AIV crawl failed for run %s", run_id)
        await fail_run(
            run_id,
            "Проверка прервалась на этапе чтения сайта. Повторите попытку: "
            "сохранённые результаты останутся на месте.",
        )


async def run_crawl(
    run_id: str,
    *,
    lease_owner: str | None = None,
) -> None:
    """Execute a run and propagate its durable owner to every child task."""

    if lease_owner is None:
        await _run_crawl_impl(run_id)
        return
    with bind_run_lease(run_id, lease_owner):
        await _run_crawl_impl(run_id, lease_owner=lease_owner)
