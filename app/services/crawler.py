"""Representative-page crawler for the AIV audit.

The public product accepts one domain. Internally we discover a small set of
representative pages, then request each page with a fixed panel of crawler
identities. Raw transport details stay in the database; the browser receives
only stage-level progress.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
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
from app.services.analyzer import analyze_run
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
# Large modern marketing pages routinely exceed 1 MiB after decompression.
# Keep a bounded ceiling, but high enough that representative content is not
# silently parsed from a broken HTML tail.
LEGACY_BODY_LIMIT_BYTES = 768 * 1024
BODY_LIMIT_BYTES = 4 * 1024 * 1024
BODY_SAMPLE_BYTES = 4096
HEADERS_BUDGET_BYTES = 8 * 1024
MIN_GAP_PER_DOMAIN_SECONDS = 0.65
SITE_PAGE_MANIFEST_KEY = "site_page_manifest"
SITE_PAGE_MANIFEST_VERSION = "aiv-2026-07-30-site-page-manifest-v1"
_DOMAIN_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_BODY_CONTENT_RX = re.compile(r"<(article|main|section|p|h1|h2)[\s>]", re.I)
_TERMINAL_PROBE_ERRORS = frozenset(
    {"cross_domain_redirect", "redirect_loop", "unsafe_target"}
)
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429})


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
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".local"):
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

                chunks: list[bytes] = []
                collected = 0
                async for chunk in response.aiter_bytes():
                    remaining = BODY_LIMIT_BYTES - collected
                    if remaining <= 0:
                        result.body_truncated = True
                        break
                    if len(chunk) > remaining:
                        chunks.append(chunk[:remaining])
                        collected += remaining
                        result.body_truncated = True
                        break
                    chunks.append(chunk)
                    collected += len(chunk)
                await response.aclose()
                body = b"".join(chunks)
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
                    len(body) < 1500
                    and not _BODY_CONTENT_RX.search(result.full_text)
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
    path = urlparse(url).path.lower().strip("/")
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
        }
    ):
        return "utility"
    rules = (
        ("pricing", ("price", "pricing", "tarif", "тариф", "стоим")),
        ("product", ("product", "service", "solution", "catalog", "услуг", "продукт")),
        ("about", ("about", "company", "o-nas", "компан", "о-нас")),
        ("content", ("blog", "article", "news", "case", "media", "стат", "новост", "кейс")),
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
    lowered = path.lower()
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
            "/wp-admin",
        )
    ):
        return None
    if re.search(r"\.(pdf|jpe?g|png|gif|webp|svg|zip|xml|json)$", lowered):
        return None
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def _links_from_html(html: str, base_url: str, domain: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    output: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        candidate = _canonical_candidate(urljoin(base_url, anchor["href"]), domain)
        if candidate and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


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
    for location in locations[:600]:
        candidate = _canonical_candidate(urljoin(base_url, location), domain)
        if candidate and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def _select_representative_urls(
    homepage_url: str,
    candidates: list[str],
    limit: int,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = [(homepage_url, "home")]
    deduped = [
        url
        for url in dict.fromkeys(candidates)
        if _page_kind(url) != "utility"
    ]
    priorities = ("product", "pricing", "about", "content", "faq", "contact", "other")
    for kind in priorities:
        matches = sorted(
            (url for url in deduped if _page_kind(url) == kind),
            key=lambda item: (urlparse(item).path.count("/"), len(item)),
        )
        if matches and all(matches[0] != url for url, _ in selected):
            selected.append((matches[0], kind))
        if len(selected) >= max(1, limit):
            break
    if len(selected) < max(1, limit):
        for url in sorted(deduped, key=lambda item: (len(item), item)):
            if all(url != existing for existing, _ in selected):
                selected.append((url, _page_kind(url)))
            if len(selected) >= limit:
                break
    return selected[: max(1, limit)]


async def _store_site_page(
    run_id: str,
    *,
    url: str,
    page_kind: str,
    probe: ProbeResult,
) -> None:
    signals = extract_text_signals(probe.full_text, probe.content_type)
    signals["_body_truncated"] = probe.body_truncated
    signals["structured_data_complete"] = not probe.body_truncated
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
        if existing is None:
            session.add(page)
        await session.commit()


def _site_page_manifest_input(domain: str, limit: int) -> dict[str, Any]:
    return {
        "domain": normalize_domain(domain),
        "selection_limit": max(1, int(limit)),
    }


def _manifest_pages(
    output_json: dict[str, Any] | list[Any] | None,
    *,
    domain: str,
    limit: int,
) -> list[tuple[str, str]] | None:
    if not isinstance(output_json, dict):
        return None
    raw_pages = output_json.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        return None
    if output_json.get("selection_limit") != max(1, int(limit)):
        return None

    pages: list[tuple[str, str]] = []
    seen: set[str] = set()
    normalized_domain = normalize_domain(domain)
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            return None
        url = raw_page.get("url")
        page_kind = raw_page.get("page_kind")
        if not isinstance(url, str) or not isinstance(page_kind, str):
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
    if (
        type(selected_count) is not int
        or selected_count != len(pages)
        or len(pages) > max(1, int(limit))
    ):
        return None

    discovered_count = output_json.get("discovered_count")
    if discovered_count is not None and (
        type(discovered_count) is not int or discovered_count < len(pages)
    ):
        return None
    expected_coverage = (
        "unknown"
        if discovered_count is None
        else ("complete" if len(pages) >= discovered_count else "limited")
    )
    if output_json.get("coverage_state") != expected_coverage:
        return None
    return pages


async def _matching_site_page_manifest(
    run_id: str,
    *,
    domain: str,
    limit: int,
) -> tuple[dict[str, Any], list[tuple[str, str]]] | None:
    input_json = _site_page_manifest_input(domain, limit)
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
            or artifact.status != "completed"
            or artifact.prompt_version != SITE_PAGE_MANIFEST_VERSION
            or artifact.input_json != input_json
            or not isinstance(artifact.output_json, dict)
        ):
            return None
        pages = _manifest_pages(
            artifact.output_json,
            domain=domain,
            limit=limit,
        )
        if pages is None:
            return None
        return dict(artifact.output_json), pages


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
    if len(locations) > 600:
        return None
    discovered = {
        url
        for url in [homepage_url, *candidates]
        if _page_kind(url) != "utility"
    }
    return len(discovered)


async def _reconcile_site_pages(
    run_id: str,
    pages: list[tuple[str, str]],
    *,
    timeout_seconds: int,
    concurrency: int,
    refresh_probes: dict[str, ProbeResult] | None = None,
) -> None:
    manifest_kinds = dict(pages)
    existing_urls: set[str] = set()
    async with SessionLocal() as session:
        existing_pages = list(
            (
                await session.execute(
                    select(SitePage).where(SitePage.run_id == run_id)
                )
            ).scalars()
        )
        for page in existing_pages:
            page_kind = manifest_kinds.get(page.url)
            if page_kind is None:
                await session.delete(page)
                continue
            page.page_kind = page_kind
            existing_urls.add(page.url)
        await session.commit()

    refresh_probes = refresh_probes or {}
    for index, (url, page_kind) in enumerate(pages):
        probe = refresh_probes.get(url)
        if url in existing_urls and probe is None:
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
        existing_urls.add(url)


async def discover_site_pages(
    run_id: str,
    domain: str,
    *,
    limit: int,
    timeout_seconds: int,
    concurrency: int,
) -> list[tuple[str, str]]:
    manifest = await _matching_site_page_manifest(
        run_id,
        domain=domain,
        limit=limit,
    )
    if manifest is not None:
        _output_json, selected = manifest
        await _reconcile_site_pages(
            run_id,
            selected,
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        return selected

    await update_progress(
        run_id,
        stage="site_discovery",
        percent=2,
        detail="Открываем главную страницу и ищем ключевые разделы.",
        eta_seconds=1800,
        status=RunStatus.crawling,
    )
    manifest_input = _site_page_manifest_input(domain, limit)
    await _save_site_page_manifest(
        run_id,
        status="running",
        input_json=manifest_input,
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

        candidates = _links_from_html(homepage_probe.full_text, homepage_url, domain)
        sitemap_url = urljoin(homepage_url, "/sitemap.xml")
        sitemap_probe, _, _ = await _probe_with_transport(
            url=sitemap_url,
            user_agent=USER_AGENT_STRINGS["robots-fetcher"],
            timeout_seconds=timeout_seconds,
            concurrency=concurrency,
        )
        if sitemap_probe.http_status == 200:
            candidates.extend(
                _links_from_sitemap(sitemap_probe.full_text, sitemap_url, domain)
            )

        selected = _select_representative_urls(homepage_url, candidates, limit)
        discovered_count = _known_discovered_page_count(
            sitemap_probe.full_text,
            sitemap_status=sitemap_probe.http_status,
            homepage_url=homepage_url,
            candidates=candidates,
        )
        coverage_state = (
            "unknown"
            if discovered_count is None
            else ("complete" if len(selected) >= discovered_count else "limited")
        )
        manifest_output = {
            "pages": [
                {"url": url, "page_kind": page_kind}
                for url, page_kind in selected
            ],
            "discovered_count": discovered_count,
            "selected_count": len(selected),
            "selection_limit": max(1, int(limit)),
            "coverage_state": coverage_state,
        }
        await _save_site_page_manifest(
            run_id,
            status="completed",
            input_json=manifest_input,
            output_json=manifest_output,
        )
    except Exception as exc:
        await _save_site_page_manifest(
            run_id,
            status="failed",
            input_json=manifest_input,
            error_message=str(exc),
        )
        raise

    await _reconcile_site_pages(
        run_id,
        selected,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        refresh_probes={selected[0][0]: homepage_probe},
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
            ).where(DomainProbe.run_id == run_id)
            .order_by(DomainProbe.id)
        )
        latest: dict[
            tuple[str, str, str],
            tuple[str | None, int | None, bool],
        ] = {}
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
            latest[(url, label, probe_type.value)] = (
                error_class,
                http_status,
                bool((content_signals or {}).get("_body_truncated"))
                or response_size_bytes == LEGACY_BODY_LIMIT_BYTES,
            )
        return {
            key
            for key, (error_class, http_status, body_truncated) in latest.items()
            if _probe_result_is_reusable(
                error_class,
                http_status,
                body_truncated=body_truncated,
            )
        }


def _probe_result_is_reusable(
    error_class: str | None,
    http_status: int | None,
    *,
    body_truncated: bool = False,
) -> bool:
    if body_truncated:
        return False
    if error_class:
        return error_class in _TERMINAL_PROBE_ERRORS
    if http_status is None:
        return False
    return (
        http_status not in _TRANSIENT_HTTP_STATUSES
        and not 500 <= http_status < 600
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
    signals["structured_data_complete"] = not probe.body_truncated
    signals["_transport"] = transport

    body_for_detection = probe.full_text[:32_000]
    markers, challenge = detect_protections(
        status=probe.http_status,
        headers=probe.response_headers,
        body_text=body_for_detection,
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
        page_limit = max(1, min(8, int(config.get("page_limit") or 6)))
        user_agents = [
            label
            for label in (config.get("user_agents") or AUDIT_USER_AGENTS)
            if label in USER_AGENT_STRINGS and label != "robots-fetcher"
        ]

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
        await update_progress(
            run_id,
            stage="technical_access",
            percent=28,
            detail="Технический аудит завершён. Переходим к смыслу сайта.",
            eta_seconds=1080,
        )
        await analyze_run(run_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("AIV crawl failed for run %s", run_id)
        await fail_run(
            run_id,
            "Проверка прервалась на этапе чтения сайта. Повторите попытку — "
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
