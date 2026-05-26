"""LLM analyzer: cross-probe ua-conditional-block pass + 4-step OpenRouter pipeline.

Runs after the crawler in the same background task. Pipeline:
  1. apply_ua_conditional_block(): mark AI-bot probes blocked while Chrome OK.
  2. _build_dataset_text(): structured ground-truth text passed to every step.
  3. step2 LLM (structural JSON facts).
  4. step3 LLM (categorical Markdown analysis).
  5. step4 LLM (AI visibility implications Markdown).
  6. step5 LLM (final Markdown report).

Outputs:
  Run.analysis_markdown          = step5 final report.
  Run.config_json["intermediate_analysis"] = {dataset_meta, step2, step3, step4, step5}.

Any failure transitions the run to status=failed with error_message; raw probes
and robots rules stay intact for inspection.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from app.config import settings
from app.db import SessionLocal
from app.models import DomainProbe, ProbeType, RobotsRule, Run, RunStatus
from app.services.event_bus import bus

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_TIMEOUT_SECONDS = 600.0
LLM_CONNECT_TIMEOUT_SECONDS = 15.0
LLM_MAX_TOKENS_PER_STEP = 32000
LLM_MAX_TOKENS_TRUNCATION_RETRY = 64000
LLM_TEMPERATURE = 0.3
LLM_NETWORK_RETRY_SLEEPS = (5.0, 15.0)  # sleeps before attempt 2 and 3
LLM_NETWORK_RETRY_EXC = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.PoolTimeout,
)

AI_BOT_LABELS: set[str] = {
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    "ClaudeBot",
    "anthropic-ai",
    "Claude-Web",
    "PerplexityBot",
    "Perplexity-User",
    "DeepSeekBot",
    "DeepSeek-User",
    "Googlebot-smartphone",
    "Googlebot-desktop",
    "GoogleOther-mobile",
    "GoogleOther-desktop",
    "Google-Agent-mobile",
    "Google-Agent-desktop",
    "Google-NotebookLM",
    "Google-CloudVertexBot",
    "GoogleAgent-URLContext",
    "Gemini-Deep-Research",
}
CONTROL_LABEL = "Chrome-control"
UA_BLOCK_STATUS_CODES = {401, 403, 429, 451}

# Switch dataset to compact mode (raw rows only for "interesting" domains) when
# the full form exceeds this many characters. ~3.5 chars per token: 100K chars
# is roughly 30K tokens, well below the model context.
DATASET_COMPACT_CHAR_THRESHOLD = 100_000
DATASET_HARD_CHAR_LIMIT = 250_000
DATASET_INITIAL_DETAIL_DOMAINS = 40

# Robots blocks worth surfacing in the dataset (others stay collapsed).
ROBOTS_BOTS_OF_INTEREST: tuple[str, ...] = (
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    "ClaudeBot",
    "anthropic-ai",
    "Claude-Web",
    "PerplexityBot",
    "Perplexity-User",
    "DeepSeekBot",
    "DeepSeek-User",
    "Google-Extended",
    "Googlebot",
    "GoogleOther",
    "Google-Agent",
    "Google-NotebookLM",
    "Google-CloudVertexBot",
    "Gemini-Deep-Research",
    "Applebot-Extended",
    "CCBot",
    "Bytespider",
    "*",
)

# Stable order for main_page probes inside a domain block.
_UA_DISPLAY_ORDER: tuple[str, ...] = (
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    "ClaudeBot",
    "anthropic-ai",
    "Claude-Web",
    "PerplexityBot",
    "Perplexity-User",
    "DeepSeekBot",
    "DeepSeek-User",
    "Googlebot-smartphone",
    "Googlebot-desktop",
    "GoogleOther-mobile",
    "GoogleOther-desktop",
    "Google-Agent-mobile",
    "Google-Agent-desktop",
    "Google-NotebookLM",
    "Google-CloudVertexBot",
    "GoogleAgent-URLContext",
    "Gemini-Deep-Research",
    "Chrome-control",
    "empty-ua",
)


# --- Cross-probe pass -------------------------------------------------------


async def apply_ua_conditional_block(run_id: str) -> int:
    """Mark AI-bot probes with `ua-conditional-block` whenever the Chrome-control
    UA got 2xx on the same domain but the AI-bot UA got 401/403/429/451.

    Returns the number of probes that received the new marker.
    """
    added = 0
    async with SessionLocal() as session:
        result = await session.execute(
            select(DomainProbe).where(
                DomainProbe.run_id == run_id,
                DomainProbe.probe_type == ProbeType.main_page,
            )
        )
        probes = list(result.scalars().all())
        by_domain: dict[str, list[DomainProbe]] = {}
        for p in probes:
            by_domain.setdefault(p.domain, []).append(p)

        for group in by_domain.values():
            chrome_ok = any(
                p.user_agent_label == CONTROL_LABEL
                and p.http_status is not None
                and 200 <= p.http_status < 300
                for p in group
            )
            if not chrome_ok:
                continue
            for p in group:
                if p.user_agent_label not in AI_BOT_LABELS:
                    continue
                if p.http_status not in UA_BLOCK_STATUS_CODES:
                    continue
                markers = list(p.detected_protections or [])
                if "ua-conditional-block" in markers:
                    continue
                markers.append("ua-conditional-block")
                p.detected_protections = markers
                flag_modified(p, "detected_protections")
                added += 1
        await session.commit()
    return added


# --- Dataset text builder ---------------------------------------------------


def _format_robots_for_domain(rules: list[RobotsRule]) -> list[str]:
    by_bot = {r.bot_name: r for r in rules}
    out: list[str] = []
    for bot in ROBOTS_BOTS_OF_INTEREST:
        rule = by_bot.get(bot)
        if rule is None:
            continue
        if rule.rule == "partial" and rule.raw_directives:
            tail: list[str] = []
            for line in rule.raw_directives.splitlines():
                stripped = line.strip()
                if not stripped or stripped.lower().startswith("user-agent"):
                    continue
                tail.append(stripped)
                if len(tail) >= 4:
                    break
            joined = "; ".join(tail)
            out.append(f"  {bot}: partial - {joined}")
        else:
            out.append(f"  {bot}: {rule.rule}")
    return out


_SIGNAL_FLAG_KEYS = (
    ("looks_like_spa_shell", "spa-shell"),
    ("looks_like_redirect_shell", "redirect-shell"),
    ("looks_like_login_wall", "login-wall"),
    ("looks_like_captcha_page", "captcha-page"),
    ("looks_like_error_page", "error-page"),
    ("looks_like_geo_block", "geo-block"),
    ("looks_disproportionate_wrapper", "wrapper-only"),
)


def _signals_compact(cs: dict | None) -> list[str]:
    """Compact list of true flags + language hint; '-' if nothing useful."""
    if not cs:
        return ["-"]
    out: list[str] = []
    for key, short in _SIGNAL_FLAG_KEYS:
        if cs.get(key):
            out.append(short)
    lang = cs.get("primary_language_guess")
    if lang:
        out.append(lang)
    return out or ["-"]


def _format_main_page_probe(p: DomainProbe) -> str:
    label = p.user_agent_label
    if p.error_class:
        msg = (p.error_message or "")[:80].replace("\n", " ")
        return f"  {label:<14} -> ERROR {p.error_class}: {msg}"
    status = p.http_status if p.http_status is not None else "?"
    size_kb = (p.response_size_bytes or 0) / 1024
    elapsed = p.total_time_ms or 0
    markers = p.detected_protections or []
    markers_str = f"[{', '.join(markers)}]" if markers else "[]"
    text_len = p.content_extractable_text_length
    text_str = "?" if text_len is None else str(text_len)
    signals = _signals_compact(p.content_signals)
    signals_str = f"[{', '.join(signals)}]"
    extras: list[str] = []
    if p.challenge_detected:
        extras.append("challenge")
    if p.body_looks_empty:
        extras.append("body_empty")
    extras_str = (", " + ", ".join(extras)) if extras else ""
    return (
        f"  {label:<14} -> {status}, body={size_kb:.0f}KB, text={text_str}, "
        f"signals={signals_str}, {elapsed}ms, {markers_str}{extras_str}"
    )


def _domain_is_interesting(probes: list[DomainProbe], rules: list[RobotsRule]) -> bool:
    if any(p.error_class for p in probes):
        return True
    if any(p.challenge_detected for p in probes):
        return True
    for p in probes:
        markers = p.detected_protections or []
        if "ua-conditional-block" in markers:
            return True
        if markers:
            return True
    if any(r.bot_name != "*" and r.rule in ("disallow_all", "partial") for r in rules):
        return True
    main_statuses = {
        p.http_status for p in probes if p.probe_type == ProbeType.main_page and p.http_status is not None
    }
    if any(s >= 400 or s < 200 for s in main_statuses):
        return True
    return False


def _domain_interest_rank(probes: list[DomainProbe], rules: list[RobotsRule]) -> int:
    score = 0
    for p in probes:
        markers = p.detected_protections or []
        if p.error_class:
            score += 5
        if p.challenge_detected:
            score += 8
        if "ua-conditional-block" in markers:
            score += 10
        if markers:
            score += 3
        if p.probe_type == ProbeType.main_page and p.http_status is not None:
            if p.http_status >= 400:
                score += 4
            elif p.http_status < 200:
                score += 2
        if p.tls_ok is False:
            score += 6
        if p.body_looks_empty:
            score += 3
    for r in rules:
        if r.bot_name != "*" and r.rule == "disallow_all":
            score += 6
        elif r.bot_name != "*" and r.rule == "partial":
            score += 3
    return score


def _compact_counts(items: list[str], *, limit: int = 6) -> str:
    if not items:
        return "-"
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    parts = [
        f"{k}:{v}"
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    ]
    if len(counts) > limit:
        parts.append(f"+{len(counts) - limit} more")
    return ", ".join(parts)


def _domain_summary_line(domain: str, probes: list[DomainProbe], rules: list[RobotsRule]) -> str:
    main_probes = [p for p in probes if p.probe_type == ProbeType.main_page]
    statuses = sorted({str(p.http_status) for p in main_probes if p.http_status is not None})
    errors = [p.error_class for p in main_probes if p.error_class]
    markers = [m for p in main_probes for m in (p.detected_protections or [])]
    text_lens = [
        p.content_extractable_text_length
        for p in main_probes
        if p.user_agent_label in AI_BOT_LABELS and p.content_extractable_text_length is not None
    ]
    max_text = max(text_lens) if text_lens else None
    robot_blocks = [
        f"{r.bot_name}:{r.rule}"
        for r in rules
        if r.bot_name in ROBOTS_BOTS_OF_INTEREST and r.rule in ("disallow_all", "partial")
    ]
    tls_failures = [p.user_agent_label for p in probes if p.tls_ok is False]
    challenges = sum(1 for p in main_probes if p.challenge_detected)
    empty_bodies = sum(1 for p in main_probes if p.body_looks_empty)
    return (
        f"- {domain}: statuses={statuses or ['?']}; "
        f"errors={_compact_counts([e for e in errors if e])}; "
        f"markers={_compact_counts(markers)}; "
        f"max_ai_text={'?' if max_text is None else max_text}; "
        f"robots={_compact_counts(robot_blocks, limit=4)}; "
        f"challenges={challenges}; empty_bodies={empty_bodies}; "
        f"tls_failures={_compact_counts(tls_failures, limit=4)}"
    )


def _domain_block(
    domain: str,
    probes: list[DomainProbe],
    rules: list[RobotsRule],
    *,
    compact_uninteresting: bool,
) -> list[str]:
    main_probes = [p for p in probes if p.probe_type == ProbeType.main_page]
    robots_probe = next((p for p in probes if p.probe_type == ProbeType.robots_txt), None)

    if compact_uninteresting and not _domain_is_interesting(probes, rules):
        statuses = sorted({p.http_status for p in main_probes if p.http_status is not None})
        return [
            f"=== DOMAIN: {domain} === (compact: all UAs OK, robots permissive; statuses={statuses})"
        ]

    out = [f"=== DOMAIN: {domain} ==="]

    if robots_probe is None:
        out.append("robots.txt: not probed")
    elif robots_probe.error_class:
        out.append(f"robots.txt: ERROR {robots_probe.error_class}")
    else:
        out.append(
            f"robots.txt: HTTP {robots_probe.http_status}, "
            f"{robots_probe.response_size_bytes or 0} bytes"
        )
    out.extend(_format_robots_for_domain(rules))

    if main_probes:
        # content_summary across AI-bot probes (Chrome-control excluded — we
        # care about what AI sees, not what humans see).
        ai_probes = [p for p in main_probes if p.user_agent_label in AI_BOT_LABELS]
        if ai_probes:
            text_lens = [
                p.content_extractable_text_length
                for p in ai_probes
                if p.content_extractable_text_length is not None
            ]
            max_text = max(text_lens) if text_lens else None
            pattern_counter: dict[str, int] = {}
            lang_counter: dict[str, int] = {}
            for p in ai_probes:
                cs = p.content_signals or {}
                for key, short in _SIGNAL_FLAG_KEYS:
                    if cs.get(key):
                        pattern_counter[short] = pattern_counter.get(short, 0) + 1
                lg = cs.get("primary_language_guess") if cs else None
                if lg:
                    lang_counter[lg] = lang_counter.get(lg, 0) + 1
            dominant = (
                max(pattern_counter.items(), key=lambda x: x[1])[0]
                if pattern_counter
                else "-"
            )
            language = (
                max(lang_counter.items(), key=lambda x: x[1])[0]
                if lang_counter
                else "-"
            )
            max_text_str = "?" if max_text is None else str(max_text)
            out.append(
                f"content_summary: max_text={max_text_str} chars across all AI probes, "
                f"dominant_pattern={dominant}, language={language}"
            )

        out.append("main_page probes:")
        order_map = {ua: i for i, ua in enumerate(_UA_DISPLAY_ORDER)}
        main_probes_sorted = sorted(main_probes, key=lambda p: order_map.get(p.user_agent_label, 999))
        for p in main_probes_sorted:
            out.append(_format_main_page_probe(p))

    final_urls = sorted({p.final_url for p in main_probes if p.final_url})
    if len(final_urls) > 1:
        out.append("final_url variations: " + ", ".join(final_urls[:3]))
    elif len(final_urls) == 1 and final_urls[0] != f"https://{domain}/":
        out.append(f"final_url: {final_urls[0]}")

    tls_failures = sorted({p.user_agent_label for p in probes if p.tls_ok is False})
    if tls_failures:
        out.append(f"tls_failures: {', '.join(tls_failures)}")

    return out


async def _build_dataset_text(run_id: str) -> tuple[str, dict[str, Any]]:
    """Returns (dataset_text, meta) where meta carries counts and chosen mode."""
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run is None:
            raise RuntimeError(f"run {run_id} disappeared before dataset_text build")
        cfg = dict(run.config_json or {})
        domains_in_cfg: list[str] = list(cfg.get("domains") or [])
        ua_labels: list[str] = list(cfg.get("user_agents") or [])
        concurrency = cfg.get("concurrency")
        timeout_seconds = cfg.get("timeout_seconds")
        source_breakdown = cfg.get("source_breakdown") or None

        probes = list(
            (await session.execute(select(DomainProbe).where(DomainProbe.run_id == run_id)))
            .scalars()
            .all()
        )
        robots_rules = list(
            (await session.execute(select(RobotsRule).where(RobotsRule.run_id == run_id)))
            .scalars()
            .all()
        )

    domains: list[str] = []
    seen: set[str] = set()
    for d in domains_in_cfg:
        d = d.strip()
        if d and d not in seen:
            domains.append(d)
            seen.add(d)
    for p in probes:
        if p.domain not in seen:
            domains.append(p.domain)
            seen.add(p.domain)

    probes_by_domain: dict[str, list[DomainProbe]] = {d: [] for d in domains}
    for p in probes:
        probes_by_domain.setdefault(p.domain, []).append(p)
    rules_by_domain: dict[str, list[RobotsRule]] = {d: [] for d in domains}
    for r in robots_rules:
        rules_by_domain.setdefault(r.domain, []).append(r)

    header = [
        f"RUN: {run_id}",
        (
            f"DOMAINS: {len(domains)}, USER_AGENTS: {ua_labels}, "
            f"CONCURRENCY: {concurrency}, TIMEOUT: {timeout_seconds}s"
        ),
    ]
    if source_breakdown:
        s1 = source_breakdown.get("set1_selected", 0)
        s2 = source_breakdown.get("set2_selected", 0)
        custom = source_breakdown.get("custom", 0)
        dups = source_breakdown.get("deduplicated_removed", 0)
        header.extend([
            "",
            "SOURCE BREAKDOWN:",
            f"  Set 1 (Manual research baseline): {s1} domains",
            f"  Set 2 (Research corpus sources): {s2} domains",
            f"  Custom: {custom} domains",
            f"  Deduplicated: {dups} duplicates removed",
        ])
    header.append("")

    def render(*, compact: bool) -> str:
        body: list[str] = []
        for d in domains:
            body.extend(
                _domain_block(
                    d,
                    probes_by_domain.get(d, []),
                    rules_by_domain.get(d, []),
                    compact_uninteresting=compact,
                )
            )
            body.append("")
        return "\n".join(header + body)

    ranked_domains = sorted(
        domains,
        key=lambda d: _domain_interest_rank(
            probes_by_domain.get(d, []),
            rules_by_domain.get(d, []),
        ),
        reverse=True,
    )

    def render_hard_compact(*, detail_limit: int) -> str:
        detailed = set(ranked_domains[:detail_limit])
        body: list[str] = [
            "DATASET MODE: hard_compact",
            (
                "All domains are preserved in the summary index below. "
                f"Detailed per-UA blocks are included for the top {len(detailed)} "
                "highest-risk domains only to keep the LLM prompt within budget."
            ),
            "",
            "DOMAIN SUMMARY INDEX:",
        ]
        for d in domains:
            body.append(
                _domain_summary_line(
                    d,
                    probes_by_domain.get(d, []),
                    rules_by_domain.get(d, []),
                )
            )
        if detailed:
            body.extend(["", "DETAILED DOMAIN BLOCKS:"])
            for d in ranked_domains:
                if d not in detailed:
                    continue
                body.extend(
                    _domain_block(
                        d,
                        probes_by_domain.get(d, []),
                        rules_by_domain.get(d, []),
                        compact_uninteresting=True,
                    )
                )
                body.append("")
        return "\n".join(header + body)

    text = render(compact=False)
    mode = "full"
    if len(text) > DATASET_COMPACT_CHAR_THRESHOLD:
        text = render(compact=True)
        mode = "compact"
    detail_domains: int | None = None
    if len(text) > DATASET_HARD_CHAR_LIMIT:
        mode = "hard_compact"
        detail_limit = min(DATASET_INITIAL_DETAIL_DOMAINS, len(domains))
        while detail_limit >= 0:
            candidate = render_hard_compact(detail_limit=detail_limit)
            if len(candidate) <= DATASET_HARD_CHAR_LIMIT or detail_limit == 0:
                text = candidate
                detail_domains = detail_limit
                break
            detail_limit = detail_limit // 2

    meta = {
        "mode": mode,
        "char_count": len(text),
        "domains": len(domains),
        "probes": len(probes),
    }
    if detail_domains is not None:
        meta["detail_domains"] = detail_domains
    return text, meta


# --- LLM call wrapper -------------------------------------------------------


async def _http_post_llm(
    run_id: str,
    payload: dict[str, Any],
    step_label: str,
) -> dict[str, Any]:
    """Single OpenRouter POST with 3-attempt retry on transient network errors."""
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ai-visibility-checker",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(LLM_TIMEOUT_SECONDS, connect=LLM_CONNECT_TIMEOUT_SECONDS)
    last_exc: BaseException | None = None
    for attempt in range(1, 4):
        bus.publish(run_id, {
            "type": "log",
            "level": "info",
            "message": f"LLM call attempt {attempt}/3 for {step_label}",
        })
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            if resp.status_code >= 400:
                # not a transient network problem - do not retry
                raise RuntimeError(
                    f"OpenRouter HTTP {resp.status_code} on {step_label}: {resp.text[:500]}"
                )
            return resp.json()
        except LLM_NETWORK_RETRY_EXC as exc:
            last_exc = exc
            bus.publish(run_id, {
                "type": "log",
                "level": "warn",
                "message": (
                    f"LLM network error on {step_label} attempt {attempt}/3: "
                    f"{type(exc).__name__}: {exc}"
                ),
            })
            if attempt < 3:
                sleep_for = LLM_NETWORK_RETRY_SLEEPS[attempt - 1]
                await asyncio.sleep(sleep_for)
                continue
            raise
    # unreachable, but satisfy typing
    raise last_exc if last_exc else RuntimeError("unreachable")


async def _call_llm(
    run_id: str,
    messages: list[dict[str, str]],
    step_label: str,
) -> tuple[str, dict[str, Any]]:
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    bus.publish(
        run_id,
        {"type": "log", "level": "info", "message": f"LLM step: {step_label} starting"},
    )
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS_PER_STEP,
    }
    data = await _http_post_llm(run_id, payload, step_label)

    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"OpenRouter returned no content for {step_label}: {str(data)[:300]}"
        ) from exc

    usage = data.get("usage") or {}
    finish_reason = choice.get("finish_reason")

    # If the model hit max_tokens, retry once with a much larger budget.
    if finish_reason == "length":
        ct = usage.get("completion_tokens")
        bus.publish(run_id, {
            "type": "log",
            "level": "warn",
            "message": (
                f"LLM response truncated by max_tokens at step {step_label} "
                f"(finish_reason=length, completion_tokens={ct}); "
                f"retrying with max_tokens={LLM_MAX_TOKENS_TRUNCATION_RETRY}"
            ),
        })
        payload2 = {**payload, "max_tokens": LLM_MAX_TOKENS_TRUNCATION_RETRY}
        data = await _http_post_llm(run_id, payload2, step_label + "-extended")
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"OpenRouter returned no content for {step_label}-extended: {str(data)[:300]}"
            ) from exc
        usage = data.get("usage") or usage
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            bus.publish(run_id, {
                "type": "log",
                "level": "warn",
                "message": (
                    f"Step {step_label} still truncated after extended retry; "
                    f"using partial output"
                ),
            })

    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    suffix = f" (prompt={pt}, completion={ct})" if pt is not None else ""
    bus.publish(
        run_id,
        {"type": "log", "level": "info", "message": f"LLM step: {step_label} done{suffix}"},
    )
    return content, usage


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n?```\s*$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


# --- Step prompts -----------------------------------------------------------

_STEP2_SYSTEM = (
    "Ты технический аналитик, специализирующийся на инфраструктуре веб-краулинга и AI-видимости брендов. "
    "Тебе передан результат проверки списка доменов несколькими User-Agent-ами, имитирующими краулеры "
    "LLM-провайдеров (OpenAI, Anthropic, Perplexity). Твоя задача - выделить структурные паттерны в данных. "
    "Не делай выводов про бренды и не пиши итоговый отчёт - только инвентаризация фактов."
)

_STEP2_USER_TEMPLATE = """Проанализируй данные ниже. Выдай СТРОГО валидный JSON со следующими полями:

{{
  "total_domains": int,
  "total_probes": int,
  "domains_with_full_access": [list of domains where all UAs got 2xx],
  "domains_with_total_block": [list of domains where all UAs failed or got 4xx/5xx],
  "domains_with_ua_conditional": [list where Chrome OK but at least one AI-bot blocked],
  "robots_explicit_ai_blocks": [
    {{"domain": "...", "bots_blocked": ["GPTBot", "ClaudeBot"], "rule": "disallow_all|partial"}}
  ],
  "waf_distribution": {{"cloudflare": N, "qrator": N, "ddos-guard": N}},
  "challenge_pages_count": int,
  "tls_failures": [list of domains with tls_ok=false],
  "russian_specific_patterns": {{
    "yandex_smartcaptcha": [domains],
    "qrator": [domains],
    "variti": [domains],
    "russian_tls_block": [domains],
    "geo_interstitial": [domains]
  }},
  "notable_outliers": [
    {{"domain": "...", "what": "краткое описание странности"}}
  ]
}}

Возвращай ТОЛЬКО JSON, без обёрток ```json``` и без поясняющего текста.

Данные:
{dataset_text}
"""


_STEP3_SYSTEM = (
    "Ты аналитик AI-видимости. Тебе передан результат структурного анализа набора доменов и сырые данные. "
    "Твоя задача - сгруппировать домены по функциональным категориям (e-commerce, government, business "
    "media, state media, IT/UGC, financial, и т.д. - категории определяй сам, по характеру доменов в "
    "выборке) и описать паттерны доступности по каждой категории. Опиши не цифры, а явления."
)

_STEP3_USER_TEMPLATE = """На основе данных ниже:

1. Раздели домены на функциональные категории. Не больше 8 категорий, в каждой минимум 1 домен.
2. По каждой категории опиши:
   - какой паттерн доступности преобладает (полностью открыто / частично закрыто WAF / системно блокирует AI / иное)
   - есть ли расхождения между UA внутри категории
   - есть ли что-то категорийно-специфичное (например, госсайты часто без WAF, e-commerce - с агрессивным антибот-стеком)
3. Отдельно выдели 3-5 самых интересных кросс-доменных наблюдений - это могут быть unexpected findings, противоречия с очевидными ожиданиями, технологические паттерны.
4. Если в выборке преобладают российские домены - особо проанализируй специфику их инфраструктуры защиты (Yandex SmartCaptcha, Qrator, Variti, Минцифры-сертификаты).

При группировке доменов по категориям учитывай маркетинговую релевантность, не только функциональную природу:
- Госпорталы могут разделяться на "открытые госуслуги" (gosuslugi, nalog, mos) и "недоступные с зарубежного IP" (kremlin, duma) - это две разные категории с точки зрения AI visibility, хотя обе формально government.
- Экосистема Яндекса заслуживает отдельной категории - у неё специфичный паттерн SSO-redirect и SmartCaptcha, который нигде больше не встречается, и она обрабатывает огромный объём российского контента (поисковые сниппеты, Дзен-публикации, маркет-листинги, Кинопоиск). Невидимость этой категории - самый большой блок в общей картине российской AI visibility.
- E-commerce и классифайды стоит объединять только если поведение похоже. Если у Avito и Ozon оно разное - разнеси в две группы.
- Не объединяй "деловые СМИ" и "общественные СМИ" если их паттерн доступности существенно различается. Лучше две маленькие категории чем одна неинформативная.
- Думай также о том, какой тип брендов и контента типично размещается на ресурсах этой категории - это поможет тебе делать осмысленные маркетинговые выводы по категории. Деловые СМИ - это PR корпоративных брендов, экспертиза. Маркетплейсы - это товарные карточки и бренды-производители. Дзен и Habr - это авторская и B2B-экспертиза. Госпорталы - это присутствие государственных и регулируемых тем. Каждая категория обслуживает разные маркетинговые задачи разных типов брендов, и эффект от её невидимости в AI ощущается разными игроками.

В каждом блоке про категорию (даже на этом промежуточном шаге) кратко упоминай, какие типы брендов и контента присутствуют на этих ресурсах - это даст шагу 5 материал для развёрнутых маркетинговых выводов.

Контекст исследования. Замеры идут с зарубежного IP. Это позиция AI-краулеров глобальных провайдеров, пользователя под VPN и эмигрантской аудитории. Часть наблюдаемых эффектов - российская анти-ботовая защита, часть - геоблокировка с российской стороны, часть - TLS-проблемы из-за сертификатов Минцифры за пределами РФ. Удерживай эту рамку.

Стилистические запреты, которые здесь же необходимо соблюдать: не используй конструкций "Это не X, это Y", "Это не просто X, это Y", "не X, а Y" как универсального риторического приёма, а также фраз "стоит отметить", "важно подчеркнуть", "в современном ландшафте", "ключевой вывод заключается". Противопоставлять идеи можно прямым языком, без шаблонной рамки.

Верни ответ в свободной форме (Markdown), но со чёткой структурой по категориям. Это промежуточный материал, не финальный отчёт. Будь конкретен, опирайся на данные, не делай общих заявлений без подкрепления.

Структурный анализ:
{step2_json}

Сырые данные:
{dataset_text}
"""


_STEP4_SYSTEM = """Ты отраслевой аналитик, который пишет про доступность брендов, продуктов и контента в ответах LLM-систем (ChatGPT, Claude, Perplexity, Gemini, DeepSeek). Финальный читатель твоего материала - маркетолог, CMO, PR-директор, бренд-менеджер, digital-стратег. Это не техническая аудитория. Они понимают SEO, охваты, медиапланирование, бренд-метрики, performance, PR, но не обязаны знать HTTP-коды, robots.txt, SPA, SSO, JavaScript rendering, retrieval, crawler, bot user-agent и подобные термины.

Главное правило: технические детали допустимы, но при первом упоминании ты сразу объясняешь термин простым языком и привязываешь его к бизнес-риску. Дальше можешь использовать термин аккуратно, без повторного разжёвывания.

Контекст исследования. Замеры идут с зарубежного IP (Frankfurt). Это та же точка наблюдения, что и у:
- AI-краулеров OpenAI, Anthropic, Perplexity (их дата-центры в США/Европе);
- пользователя за VPN или эмигрантской аудитории;
- глобальных агрегаторов и retrieval-индексов, на которые опираются LLM при ответе.

Поэтому когда мы фиксируем, что российский ресурс возвращает пустоту, ошибку или редирект на авторизацию для AI-ботов, это не локальная аномалия - это та картина, которую видят все эти системы. Часть наблюдаемых эффектов - результат российской анти-ботовой инфраструктуры (Qrator, Variti, Yandex SmartCaptcha), часть - геоблокировки с российской стороны для зарубежных IP, часть - TLS-проблемы из-за сертификационных цепочек Минцифры, не доверяемых за пределами РФ. В отчёте эту рамку нужно держать постоянно: мы оцениваем эффекты блокировок российских ресурсов на их доступность для VPN, эмигрантского, зарубежного трафика и для глобальных AI-систем в текущей политической и инфраструктурной ситуации.

Главное аналитическое правило: AI-видимость оценивается не по формальному HTTP-коду, а по фактической доступности извлекаемого текста. Если сайт отдаёт код 200 (формально успешный ответ), но текста на странице меньше 500 символов - для AI-ответов это эквивалентно недоступности.

Объясняй для CMO следующие сценарии прямо и просто:
- Код 200 без текста: сервер формально ответил, но страница оказалась оболочкой без статьи или карточки товара. Причина - либо контент рисуется JavaScript уже в браузере (SPA-архитектура), либо это редирект на единый вход (SSO) с просьбой авторизоваться, либо это страница капчи. Для AI-системы эта страница пустая.
- Код 200 с гео-блок-баннером: сервер вернул 200 OK, но в теле страница "Доступ ограничен / отключите VPN / недоступно в вашем регионе" вместо контента. Маркер `content-geo-block` или `geo-interstitial`. Это однозначный blocked для AI: ни один retrieval-fetcher не увидит ничего, кроме текста баннера. Российские госпорталы (gosuslugi.ru, mos.ru, nalog.gov.ru) и часть медиа используют этот паттерн против зарубежного трафика. Для глобальных AI-помощников это полная невидимость, потому что их retrieval-инфраструктура физически работает с не-российских IP. Не путай с "ограниченной видимостью": это нулевая видимость с обнадёживающим HTTP-кодом.
- Код 200 с непропорционально большим телом и крошечным извлекаемым текстом: маркер `wrapper-only` (а часто рядом и `geo-block`). Признак: тело >= 3KB, но извлекаемого текста меньше 300 символов. Это структурный отпечаток баннера-обёртки, и его достаточно сам по себе чтобы считать домен функционально невидимым - даже если конкретные слова баннера не совпали со словарём (другая локализация, другой текст). Не игнорируй такие домены и не отмечай их "high" только потому что body_size большой.
- Код 403/401/498 с большим телом: технический отказ от защиты сайта (WAF) - бот не получил контент.
- Код 200 с обширным текстом (несколько тысяч символов извлекаемого текста) и без content-флагов вроде geo-block/spa-shell/login-wall/captcha: настоящая доступность. Такой источник может попадать и в тренировочные корпуса будущих моделей, и в retrieval-ответы AI-помощников.

Три уровня AI-видимости:
- high: значимый объём извлекаемого текста (минимум 3000-5000 символов на типичной странице) во всех AI-агентах. Контент реально доступен retrieval-слою (тому слою, через который AI-помощник цитирует источник пользователю в реальном времени) и тренировочным корпусам.
- partial: ограниченная или асимметричная доступность. Часть AI-агентов получает полный контент, часть - оболочку. Или robots.txt разрешает, а WAF режет конкретного бота. Или верхний слой открыт, а глубокие страницы - нет.
- blocked: AI-агенты не получают значимого текста. Сюда относятся: явные отказы (403, капча, таймаут); функциональные пустоты (код 200 с почти нулевым извлекаемым текстом - SPA-shell, redirect-shell, login-wall); и **геоблоки с HTTP 200** - страница `Доступ ограничен / отключите VPN / недоступно в вашем регионе`, помеченная маркером `content-geo-block` или `geo-interstitial`. Geo-block - это самый коварный случай: для пользователя из России сайт работает, для AI-инфраструктуры (которая всегда сидит на не-российских IP) сайт фактически закрыт.

Ни один формально-успешный HTTP-код сам по себе не делает домен high. Если все десять AI-агентов получили 200, но максимум извлекаемого текста - 87 символов, это blocked.

Сценарии использования, которые нужно держать в голове и переводить на язык маркетинга:

- Тренировочный краулинг (тренировочный обход интернета AI-компаниями для будущих моделей). Происходит редко - раз в 12-18 месяцев. Если домен отдаёт пустую оболочку - в новые версии GPT, Claude, Gemini, DeepSeek контент уйдёт как шум, отфильтруется и не попадёт в "знания" модели. Эффект на бренд: модель не будет ничего знать про бренд из этого источника.
- Retrieval / web search (обход в реальном времени, когда пользователь задал AI-помощнику вопрос). Происходит каждый день при каждом запросе. От доступности здесь зависит, будет ли LLM цитировать домен как источник под ответом пользователю прямо сейчас. Эффект на бренд: видим ли он в текущих ответах ChatGPT, Claude, Perplexity и DeepSeek.

Отдельная заметка про DeepSeek. Это китайский AI-сервис, у которого нет официальной публичной документации по своим краулерам (в отличие от OpenAI, Anthropic и Perplexity, которые свои боты документируют). Имена DeepSeekBot и DeepSeek-User в утилите идентифицированы по block-листам сайтов и обзорам поведения DeepSeek R1, не подтверждены вендором. Это менее достоверная имитация, чем для других провайдеров. Однако многие сайты явно блокируют именно эти имена в robots.txt и в правилах WAF, и эту реакцию можно фиксировать. В отчёте упоминай DeepSeek наравне с другими провайдерами, но при первом упоминании одной фразой обозначь его статус: "китайский AI-сервис, чьи краулеры (по сигналам block-листов - DeepSeekBot и DeepSeek-User) не имеют официальной документации".
- Эффект замещения. Если первоисточник недоступен AI-агентам, LLM-системы цитируют доступные альтернативы - международные СМИ, агрегаторы, архивы, релокационные медиа, вторичные пересказы. Эффект на бренд: про него и его рынок будут говорить чужие голоса.

Каждое техническое наблюдение должно быть переведено в маркетинговое последствие. Готовые соответствия, которыми можно пользоваться:
- Бот не получает текст -> публикация может не попасть в AI-ответы.
- Страница уходит на авторизацию -> официальный источник проигрывает вторичным пересказам.
- Контент появляется только после JavaScript -> LLM видит пустую оболочку вместо материала.
- Площадка не блокирует AI-ботов в robots.txt -> проблема не в политике запрета, а в инфраструктуре.
- Антибот-защита срабатывает на AI-агента -> защита от скраперов одновременно обрезает AI-видимость бренда.
- Все AI-агенты на домене получают 403 -> площадку нельзя считать каналом для попадания в AI-ответы.
- Только часть AI-агентов получает контент -> аудитория Claude и аудитория ChatGPT могут видеть про бренд разные картины.

Думай о маркетологе и CMO как о финальном читателе. Они принимают решения про бюджеты, выбор площадок для PR, размещение экспертизы, медиапланирование. Твой анализ должен давать им управленческие основания для этих решений, а не технический лог.

Кроме систематического анализа по всем доменам, отдельно выдели 5-8 самых "ярких" находок из данных текущего прогона - таких, которые удивляют, противоречат здравому смыслу, или представляют собой паттерн, который читатель не предположил бы заранее. Категории ярких находок, которые стоит искать в данных:

- Парадоксы декларации и поведения (явный allow в robots.txt при фактической невидимости из-за технической архитектуры сайта)
- Обратные дискриминации (AI получает больше контента чем браузер по extractable_text_length)
- UA-асимметрии (один AI-вендор пропускается, другой режется). Анализируй ВСЕ пять провайдеров: OpenAI, Anthropic, Perplexity, DeepSeek, Google - не только западную тройку. Если DeepSeek был в прогоне и его поведение существенно отличается от других вендоров (в любую сторону) - это самостоятельная яркая находка. Аналогично для Google: если данные прогона показывают разное поведение между Googlebot (Search/AI Overviews surface) и Google-Agent / Google-NotebookLM / Google-CloudVertexBot (user-triggered и enterprise RAG surfaces) - это отдельная яркая находка, потому что для бренда Search-видимость и Gemini-grounding-видимость могут различаться
- Кейсы где формальный 200 OK скрывает функциональную блокировку
- Кейсы где формальная блокировка оказывается просто геофильтром, а не анти-AI политикой
- Категориальные неожиданности: категория, которую читатель ожидает видеть в одном состоянии, оказывается в другом

Все находки должны опираться на конкретные домены, конкретные цифры (extractable_text_length, http_status, body_size) и конкретные паттерны из dataset_text. Не выдумывай находки, которых нет в данных. Если данных хватает только на 5 ярких находок - значит будет 5, не натягивай до 8.

Эти находки должны быть готовы для использования в открывающем блоке шага 5 как материал для секции "Главные парадоксы". Не растворяй их в категориальном анализе - выноси отдельно."""

_STEP4_USER_TEMPLATE = """На основе предыдущих этапов анализа сделай следующее:

1. Для каждой категории доменов из шага 3 опиши, какой эффект на AI visibility создаёт наблюдаемый паттерн доступности. Различай три уровня:
   - Эффект на тренировочный краулинг (попадёт ли контент в датасет следующих базовых моделей)
   - Эффект на retrieval-augmented ответы (будет ли LLM цитировать этот домен через web search tool)
   - Эффект замещения (если домен недоступен - какие источники с высокой вероятностью займут его место в ответах)

2. Идентифицируй домены с самым высоким риском "невидимости" в LLM-ответах и объясни механизм.

3. Идентифицируй домены, которые наоборот хорошо позиционированы для AI visibility, и почему.

4. Если в выборке есть российские домены - отдельно опиши, как российская специфика (антибот-инфраструктура, отсутствие AI opt-out в robots.txt, геоблокировки) трансформирует AI visibility российских брендов в глобальных LLM. Будь конкретен.

5. Дай 3-5 практических рекомендаций для владельцев доменов из выборки, которые хотят улучшить свою видимость для AI-краулеров.

Это всё ещё промежуточный материал - следующий шаг соберёт всё в финальный отчёт. Здесь сосредоточься на содержательной глубине, не на формате.

Предыдущий анализ:
{step3_markdown}

Структурные факты:
{step2_json}
"""


_STEP5_SYSTEM = """Ты пишешь финальный аналитический отчёт по результатам проверки доступности набора доменов для AI-систем. Это деловой материал для маркетингового читателя.

Целевая аудитория - CMO, PR-директор, бренд-менеджер, контент-стратег, digital-руководитель. Они понимают SEO, медиапланирование, охваты, бренд-метрики, performance, PR. Они не обязаны знать HTTP-коды, robots.txt, SPA, SSO, JavaScript rendering, retrieval, crawler, bot user-agent. Не пиши для технического аудитора - пиши для управленца, который хочет понять, что эти данные значат для его маркетинга, бренда и каналов.

Важно про учёт AI-провайдеров. В пайплайне утилиты могут быть включены user-agents ПЯТИ AI-провайдеров:

1. **OpenAI** (GPTBot, OAI-SearchBot, ChatGPT-User) - поисково-индексационные и user-triggered боты ChatGPT.
2. **Anthropic** (ClaudeBot, anthropic-ai, Claude-Web) - краулеры Claude.
3. **Perplexity** (PerplexityBot, Perplexity-User) - Perplexity search и user-triggered.
4. **DeepSeek** (DeepSeekBot, DeepSeek-User) - наблюдения по UA-строке, не подтверждены официально DeepSeek (см. ниже).
5. **Google** - Google не имеет одного "Gemini-краулера". Это четыре отдельных контура видимости:
   - **Search / AI Overviews / AI Mode** (Googlebot-smartphone, Googlebot-desktop). От этого зависит, попадёт ли страница в выдачу Google и в Gemini-grounding через Search.
   - **Gemini training & grounding control** (Google-Extended - это robots.txt-токен, не HTTP-User-Agent). Управляет, можно ли использовать уже crawled-контент сайта для обучения Gemini и grounding'а в Gemini Apps / Vertex AI.
   - **User-triggered AI-агенты** (Google-Agent-mobile, Google-Agent-desktop, Google-NotebookLM). Это запросы, инициированные пользователем (Project Mariner, NotebookLM-sources).
   - **Enterprise / RAG** (Google-CloudVertexBot). Корпоративные RAG-сценарии на Google Cloud Agent Search / Vertex AI.
   - Дополнительно могут попадаться UA `GoogleAgent-URLContext` и `Gemini-Deep-Research` - это НАБЛЮДАЕМЫЕ строки, не задокументированные Google официально. Если они попали в прогон - упоминай с пометкой "наблюдаемая, не подтверждена Google-документацией".

Не все провайдеры могут быть включены в каждом конкретном прогоне. Но если в данных прогона есть проба с user_agent_label из любой провайдерской группы - в отчёте должны быть сделаны выводы про этого провайдера наравне с остальными.

Особый статус DeepSeek: при описании поведения DeepSeek всегда добавляй короткую оговорку "наблюдения по UA-строке, не подтверждены официально DeepSeek". Это нужно один раз при первом существенном упоминании DeepSeek в отчёте, дальше можно не повторять.

Если в прогоне DeepSeek-агенты не были включены - просто не упоминай DeepSeek. Если Google-агенты не были включены - не упоминай Google. Молчание здесь - правильное поведение.

Принципиальное требование. Все наблюдения, парадоксы, асимметрии, цифры, имена доменов в отчёте должны быть выведены из реальных полей dataset_text текущего прогона. Промпт не подсказывает тебе "правильных ответов" про конкретные домены: имена в любых примерах ниже - это формат, не данные. Если паттерн, описанный в промпте как "категория парадокса", не подтверждается данными прогона - не выдумывай его, переходи к следующему. Лучше короткий честный отчёт, чем длинный с придуманными находками.

Контекст исследования. Все замеры идут с зарубежного IP. Это та же позиция, с которой сайт видят AI-краулеры OpenAI, Anthropic, Perplexity, пользователи под VPN, эмигрантская аудитория и глобальные агрегаторы. Часть наблюдаемых эффектов - российская анти-ботовая защита (Qrator, Variti, Yandex SmartCaptcha), часть - геоблокировка с российской стороны для зарубежных IP, часть - TLS-проблемы из-за сертификатов Минцифры, не признаваемых за пределами РФ. В отчёте удерживай эту рамку: мы оцениваем эффекты блокировок российских ресурсов на их доступность для VPN, эмигрантского и зарубежного трафика и для глобальных AI-систем в текущей политической и инфраструктурной обстановке.

Стиль. Деловой аналитический отчёт для маркетинговой аудитории:
- без рекламной подачи;
- уверенный, но не мотивационный;
- с авторской позицией;
- без канцелярита;
- без академичности;
- без презентационности в духе McKinsey/Bain;
- без чрезмерно гладких LLM-формулировок;
- без длинных объяснительных эссе там, где нужен управленческий вывод;
- без декоративных метафор;
- без эмоциональных преувеличений без данных.

Текст должен звучать так, будто его написал сильный отраслевой аналитик, который понимает и маркетинг, и техническую инфраструктуру, но пишет для CMO.

Заголовки.
Заголовки должны быть сухими и утилитарными. Они не должны звучать как название статьи в медиа, твит или презентационный слоган. В заголовке нельзя использовать необъяснённые технические термины: 200 OK, SPA, SSO, robots.txt, crawler, retrieval, HTTP. Если термин нужен - вводи его внутри текста с пояснением, а в заголовок выноси бизнес-смысл.

Плохие заголовки:
- "AI visibility российского веба: когда 200 OK не означает присутствия"
- "Невидимый интернет: почему бренды исчезают из ответов AI"
- "Новая реальность AI-поиска"
- "Когда сайт есть, но его нет"

Хорошие заголовки:
- "Проблема AI-видимости российских площадок ввиду блокировок"
- "Почему публикации на российских площадках могут не попадать в ответы LLM"
- "Технические причины снижения AI-видимости брендов"
- "Риск для маркетинга: площадка видима пользователям, но не видима AI-системам"
- "Как инфраструктура сайта влияет на присутствие бренда в AI-ответах"

Объяснение технических терминов.
При первом упоминании любого технического термина дай короткое человеческое пояснение в той же фразе, через тире или скобки. Дальше можешь использовать термин аккуратно, без повторного разжёвывания.

Образцы пояснений (используй похожий уровень детальности):
- 200 OK - технический код, который означает, что сервер формально отдал страницу;
- robots.txt - файл с правилами для поисковых и AI-ботов;
- SPA - сайт, где основной контент часто дорисовывается уже в браузере пользователя;
- SSO - единый вход через аккаунт, который может увести анонимного посетителя на страницу авторизации;
- retrieval - слой поиска источников, из которых LLM берёт факты и ссылки для ответа;
- crawler - бот, который автоматически обходит страницы и извлекает из них данные;
- WAF - защитный фильтр перед сайтом, отбивающий автоматические запросы;
- captcha - проверка "вы человек?", которая отдаётся вместо контента.

Жёсткие авторские формулировки сохраняй.
Не сглаживай текст до нейтральной корпоративной воды. Сильные управленческие фразы оставляй, если они отражают данные.

Желательны фразы такого типа:
- "Это системная особенность российского веба."
- "Никто не проектировал, и поэтому никто и не чинит."
- "Публикация на Дзене - это инвестиция в актив, которого для LLM не существует."
- "В ответах ChatGPT, Claude и Perplexity про этот бренд будут говорить другие голоса - в основном западные и эмигрантские."
- "Площадка может быть видимой для людей и невидимой для AI."
- "Маркетинг может покупать охват, который не превращается в AI-присутствие."

Не подменяй такие формулировки на стерильные:
- "может наблюдаться снижение эффективности";
- "имеет место ограниченная доступность";
- "следует учитывать потенциальные риски";
- "наблюдается неоднозначная ситуация".

Запрещённые LLM-паттерны. Эти конструкции делают текст узнаваемо машинным, не используй их совсем:
- "Это не X, это Y."
- "Это не просто X, это Y."
- "Речь не только о X, речь о Y."
- "не X, а Y" как универсальный риторический приём.
- "Важно понимать, что..."
- "Стоит отметить, что..."
- "В современном цифровом ландшафте..."
- "В условиях стремительного развития..."
- "Это особенно важно, потому что..."
- "Ключевой вывод заключается в том, что..."
- "Данный кейс демонстрирует..."
- "Можно выделить три ключевых фактора..."
- "В конечном счёте..."
- "Таким образом..." в начале финального абзаца.

Запрет на конструкцию "Это не X, это Y" не означает запрет на противопоставления. Противопоставлять идеи можно и нужно, но прямым аналитическим языком, без шаблонной рамки.
- Плохо: "Это не проблема SEO, это проблема AI visibility."
- Хорошо: "SEO-метрики здесь не помогают. Площадка может быть сильной в поиске и слабой как источник для AI-ответов."
- Плохо: "Это не артефакт измерения. Это системная особенность российского веба."
- Хорошо: "Повторяемость результата на нескольких крупных доменах указывает на системную особенность российского веба."
- Плохо: "Это не политика издателей, а побочный эффект инфраструктуры."
- Хорошо: "Российские издатели редко блокируют AI-ботов напрямую. Невидимость чаще возникает из-за инфраструктуры: антибот-защиты, клиентского рендера и авторизации."

Также не используй маркетинговую корпоративную лексику в нейтральном смысле: "ландшафт", "экосистема" (только если речь буквально про экосистему компании), "вызовы", "трансформация", "парадигма", "ключевые драйверы", "синергия", "возможности роста", "новая реальность".

Дополнительно к этому запрещены публицистические обороты:
- "сообщение фрагментируется"
- "новый класс риска"
- "новое поле", "новая реальность", "новый слой"
- "управление распределено между отделами"
- "до недавнего времени не существовал в брифах"
- любые формулировки вида "Х стал новым Y"
- любые формулировки с "пока это так"

Это маркеры лекторского или публицистического стиля. Профессиональный аналитический отчёт пишется без них. Если хочется обозначить новизну явления - описывай его конкретными признаками, а не словами "новый" и "впервые".

Отдельно: формулировки уровня "Это системная особенность российского веба", "Никто не проектировал, и поэтому никто и не чинит", "Публикация на Дзене - это инвестиция в актив, которого для LLM не существует", "Будут говорить другие голоса - в основном западные и эмигрантские" - это сильные авторские формулировки и они РАЗРЕШЕНЫ. Не путай их с публицистическим штампом. Их использовать нужно ровно тогда, когда они отражают реальную картину данных, а не как декоративный элемент. Не подменяй стерильным "наблюдается ограниченная доступность".

Финальный абзац отчёта НЕ начинается с "любопытная деталь", "интересно отметить", "примечательно". Начинай с конкретного наблюдения сразу.

Запрещены эмодзи, длинные тире и среднее тире. Используй обычный дефис -. Символы — и – не использовать.

Markdown-разметка как часть стиля. Ты пишешь не сплошной текст, а размеченный документ для маркетингового читателя. В каждом разделе:
- 1-3 управленческих тезиса выделяй жирным `**...**` (короткие фразы по 5-15 слов).
- Самые сильные авторские наблюдения - в одну блок-цитату через `>` (одна на раздел максимум, 4-7 на весь отчёт).
- Между абзацами одна пустая строка, между разделами две.
- Абзацы средней длины: 3-5 предложений. Не сваливай 8 предложений в один абзац, не дроби на однострочники.
- Списки только там, где данные действительно списочные.

Тон.
Сохраняй: прямоту, управленческую жёсткость, причинно-следственную логику, конкретику, риск-ориентированность, авторский голос.
Избегай: универсальных фраз без предмета, длинных симметричных предложений, одинакового ритма абзацев, чрезмерного числа списков, частых "во-первых / во-вторых / в-третьих", декоративных метафор.

Глоссарий технических меток в данных.
В исходных данных встретятся ярлыки User-Agent. При упоминании в отчёте не вставляй их сырыми. Объясняй один раз и дальше используй короткое описание:
- Chrome-control - контрольный запрос обычным браузером, нужен как точка отсчёта для сравнения с AI-ботами. В тексте можно писать "контрольный запрос обычным браузером", "не-AI клиент", "браузерный baseline".
- empty-ua - запрос без User-Agent, проверяет блокировки, не привязанные к конкретному боту.
- GPTBot, OAI-SearchBot, ChatGPT-User - агенты OpenAI. GPTBot ходит ради тренировочного обхода для будущих моделей; OAI-SearchBot и ChatGPT-User - обходят страницы в момент ответа пользователю в ChatGPT.
- ClaudeBot, anthropic-ai, Claude-Web - агенты Anthropic, тренировочный и web-search.
- PerplexityBot, Perplexity-User - агенты Perplexity, тренировочный и runtime.
- Googlebot-smartphone, Googlebot-desktop - основной краулер Google для Search и AI Overviews / AI Mode (мобильная и десктоп-версии). Через него страница попадает в выдачу Google и в Gemini-grounding через Search.
- GoogleOther-mobile, GoogleOther-desktop - вспомогательные краулеры Google для внутренних исследований и продуктов (не Search-индексация).
- Google-Agent-mobile, Google-Agent-desktop - user-triggered AI-агенты Google (Project Mariner и подобные). Ходят в момент действия пользователя в продуктах Google.
- Google-NotebookLM - агент NotebookLM, забирает источники, добавленные пользователем в notebook. Также user-triggered.
- Google-CloudVertexBot - корпоративный RAG-агент для сценариев Google Cloud Agent Search / Vertex AI. Enterprise-поверхность.
- GoogleAgent-URLContext, Gemini-Deep-Research - наблюдаемые UA-строки, не задокументированные Google официально. Упоминай с этой пометкой.

Привязка к данным. Каждое утверждение про конкретный домен опирай на конкретные числа из исходных данных: длина извлекаемого текста, размер тела ответа, факт allow/disallow в robots.txt, статус-код. Не округляй и не фантазируй цифры.

Геоблок с HTTP 200 - частая ошибка интерпретации. Если в маркерах пробы есть `content-geo-block` или `geo-interstitial`, это означает что сервер отдал HTTP 200 со страницей "Доступ ограничен / отключите VPN / недоступно в вашем регионе". Размер тела может быть и 9KB, но реального текста там 100-300 символов - сам баннер. AI visibility таких доменов - blocked, не high и не partial. Глобальная retrieval-инфраструктура LLM-сервисов (OpenAI, Anthropic, Perplexity, DeepSeek) физически работает с не-российских IP, для них этот баннер - и есть весь сайт. Российские госпорталы (gosuslugi.ru, mos.ru, nalog.gov.ru, часть финансовых регуляторов) и часть медиа массово используют этот паттерн. Не пиши "доступ есть с ограничениями" - пиши "функционально невидим для AI с зарубежного IP, retrieval получает только баннер".

Чек-лист самопроверки. Прежде чем выдавать финальный текст, мысленно пройдись по нему и убедись:
- Заголовок отчёта понятен CMO без технической подготовки.
- В заголовке нет необъяснённых технических терминов.
- Первый абзац говорит о бизнес-риске, а не о технической детали. Он не начинается с технического термина.
- Каждый технический термин объяснён при первом употреблении.
- Нет конструкций "Это не X, это Y" и аналогичных шаблонных антитез.
- Нет фраз "важно отметить", "в современном ландшафте", "ключевой вывод заключается в том, что", "таким образом" в начале финального абзаца.
- Сильные авторские формулировки сохранены, не стерилизованы.
- Текст не похож на SEO-статью или LinkedIn-пост.
- В каждом разделе понятно, что это значит для маркетинга, CMO, PR или digital.
- Выводы не шире, чем позволяют данные."""


_STEP5_USER_TEMPLATE = """Напиши финальный отчёт. Это деловой аналитический материал для маркетингового читателя (CMO, бренд-директор, PR, контент-стратег, digital-руководитель). Не для технического аудитора.

Заголовок отчёта.
Сухой и утилитарный. Не используй необъяснённых технических терминов в заголовке (200 OK, SPA, SSO, robots.txt, retrieval, crawler, HTTP). Заголовок несёт бизнес-смысл, а не выглядит как название статьи в медиа. Ориентир по форме: "Проблема AI-видимости российских площадок ввиду блокировок", "Технические причины снижения AI-видимости брендов", "Как инфраструктура сайтов влияет на присутствие брендов в AI-ответах".

Структура. Используй следующий порядок разделов. Заголовки разделов формулируй сам, в стиле "сухие и утилитарные", не копируй буквально нумерацию из этого списка - переформулируй живо.

I. Открывающий блок (3-5 абзацев, без заголовка). Это первое что читает CMO, и именно по этому блоку он решает, дочитывать или нет. Никаких общих слов, никакой подготовки контекста, никакой "проблематизации". Только конкретные находки из данных текущего прогона.

Структура открывающего блока:

Первый абзац - главная цифра и её немедленная интерпретация для маркетинга. Сколько именно доменов в выборке функционально невидимы для AI несмотря на формальный успех HTTP. Перечисли поимённо первые несколько самых заметных - это "ловушка маркетолога", который видит в аналитике "сайт работает", а на деле AI не получает контента. Назови конкретные бренды и темы, которые из-за этого выпадают из ответов LLM. Не общими словами "часть доменов", а перечислением имён доменов с цифрами по их extractable_text_length из dataset_text. Все имена доменов и цифры берутся из переданных тебе данных, не из общих знаний.

Второй абзац - главные парадоксы выборки, по 1-2 строки на каждый. Парадокс это случай, когда наблюдаемое поведение домена противоречит ожиданиям, которые читатель сформировал бы по формальным признакам. Найди такие парадоксы в данных текущего прогона. Возможные категории парадоксов: расхождение декларации robots.txt и фактического поведения; обратная дискриминация (AI получает больше контента чем обычный браузер); рассинхронизация политик (robots.txt одно, WAF другое); неожиданная открытость или неожиданная закрытость для конкретной категории. Не перечисляй парадоксы которых нет в данных - если паттерн не подтверждается прогоном, не выдумывай.

Третий абзац - асимметрия между AI-вендорами в данных текущего прогона. Если в выборке есть домены, которые отвечают разным AI-вендорам по-разному (по extractable_text_length, по http_status, по signals) - это отдельная важная находка, потому что означает разную картину российского рынка в ответах разных LLM-сервисов. Перечисли такие домены поимённо с конкретикой "вендор X получает Y, вендор Z получает W". Если асимметрии в данных нет - не натягивай её.

Сразу после третьего абзаца, до четвёртого, разместить СВОДНУЮ ТАБЛИЦУ по всем доменам (она ниже описана как раздел II). Сама таблица не получает отдельного заголовка раздела - читается как иллюстрация к открывающему блоку.

Четвёртый абзац (опционально, если данные дают материал) - географическая правда. Часть наблюдаемых блокировок может быть не анти-AI политикой, а IP-фильтрацией всего зарубежного трафика. Найди в данных домены, у которых Chrome-baseline получил тот же отказ что и AI-боты (одинаковые статусы, одинаковые тела, одинаковые таймауты). Это маркер геоблока, не анти-AI. Перечисли такие домены и сделай оговорку: для российской аудитории, обращающейся к LLM с российского IP, картина может быть другой, но retrieval-слой LLM-сервисов всё равно работает с не-российских IP, поэтому для конечного пользователя в РФ функциональный результат идентичен.

Пятый абзац - рамка дальнейшего отчёта. Одна-две строки про то, как организован остальной материал и что в нём искать. Не больше.

Никаких "в данном отчёте мы рассмотрим", никаких "AI visibility становится новым полем", никаких "пользователь видит публикацию, поисковик её индексирует, но". Это все известные читателю общие места, и они занимают место настоящих наблюдений.

II. Сводная таблица по доменам (вставляется внутри открывающего блока, между третьим и четвёртым абзацами; см. инструкцию выше). Колонки в порядке:
- Домен
- Категория
- Главный сигнал доступности (короткая фраза, отражающая фактическую доступность контента, а не сухой код. Пример: "код 200 всем AI, но 87 символов извлекаемого текста" вместо "200 OK всем AI")
- Web fetch (полный / оболочка / отказ от защиты / капча / авторизация / таймаут / ошибка)
- robots.txt про AI (разрешено / запрещено / частично / не упомянут / недоступен)
- Какие AI-агенты блокируются (нет / OpenAI / Anthropic / Perplexity / DeepSeek / Google / комбинация / все). Для Google различай поверхности: Search (Googlebot), user-triggered (Google-Agent, NotebookLM), enterprise (CloudVertexBot) - если данные показывают разное поведение между ними, отметь это в строке таблицы коротко
- AI-видимость (высокая / частичная / отсутствует)

III. Главные парадоксы выборки. 4-7 коротких блоков, каждый по 2-3 предложения, по структуре "наблюдение → почему это контр-интуитивно → что это значит для маркетинга". Это сжатая выжимка самых неочевидных находок исследования из текущего прогона. Маркетолог должен прочитать этот раздел и понять, что данные дали ему неожиданные знания, которых не было до исследования.

Категории парадоксов, которые стоит искать в данных:

- Расхождение декларации в robots.txt и фактического поведения сайта при HTTP-обходе
- Обратная дискриминация: AI-боты получают больше осмысленного контента (по extractable_text_length) чем Chrome-baseline
- Рассинхрон между явной политикой в robots.txt и поведением WAF
- UA-асимметрии между AI-вендорами на одном и том же домене (включая DeepSeek, если он включён)
- Случаи когда формальный 200 OK скрывает функциональную блокировку через SPA-shell, login-wall или redirect-shell
- Случаи когда формальная блокировка по статусу оказывается общим геофильтром, а не анти-AI политикой
- Категориальные неожиданности: категория, которую маркетинг считает "обязательной площадкой", оказывается невидимой для AI; или наоборот
- Если включён DeepSeek и его поведение даёт основания для отдельной находки

Каждый парадокс - это конкретное утверждение опирающееся на конкретные домены и цифры из dataset_text. Если в данных прогона определённый тип парадокса не встретился - не выдумывай его, переходи к следующему. Минимум 4 парадокса должны быть подкреплены данными; если данные дают только 4 - значит будет 4, не натягивай до 7.

Каждый блок не должен быть абзацем-эссе, должен быть плотной аналитической репликой.

IV. Главный вывод для маркетинга (2-3 абзаца).
Здесь стратегическая формулировка для CMO. Какая часть выборки реально работает на AI-присутствие, какая нет, и как это должно влиять на распределение бюджетов на PR и контент. Объясни, почему доступность для пользователя и доступность для AI-системы - это разные вещи, и почему медиапланирование, опирающееся только на охват и SEO, перестаёт давать полную картину.

V. Разбор по категориям доменов.
Каждая категория - связный текст 4-7 абзацев. Внутри категории идёт следующая логика повествования (без подзаголовков а/б/в/г/д, это смысловая структура, а не визуальная):
- Что показывают данные на техническом уровне, простыми словами. Сразу с переводом терминов в человеческий язык.
- Что это значит для retrieval (того слоя, через который LLM подтягивает источники в ответ пользователю в реальном времени): будут ли AI-помощники цитировать ресурсы этой категории, какой текст они получают.
- Что это значит для тренировочных корпусов будущих базовых моделей: войдёт ли свежий контент категории в "знания" следующих GPT, Claude, Gemini, DeepSeek.
- Что это значит для маркетинга и бренда: бренд-эффект, контент-эффект, конкурентный эффект, эффект на performance-метрики.
- Где это уместно - короткая практическая рекомендация для маркетинговой команды: какие площадки в категории сейчас работают на AI-видимость, какие нет, какие альтернативы рассматривать.

В категориальных разрезах НЕ пересказывай содержимое robots.txt по каждому домену подробно. Достаточно одной строки про robots.txt в описании каждого домена: либо короткая характеристика правила, либо констатация отсутствия упоминаний AI-ботов. Полная картина по robots.txt - в отдельном агрегированном разделе ниже.

VI. Техническая защита и её эффект на AI-видимость.
Какие WAF-системы и анти-ботовые сервисы встретились (Cloudflare, Qrator, Variti, DDoS-Guard, Yandex SmartCaptcha и тому подобные - объясни в одной фразе, что это вообще такое). Как они влияют на доступ AI-агентов. Категориальная специфика. В конце - короткий маркетинговый вывод: выбор инфраструктуры на стороне сайта становится переменной при выборе площадок для размещения брендового контента.

VII. Раздел про robots.txt - короткий, на 2-3 абзаца максимум. Не пересказывай поведение каждого домена снова - оно уже описано в категориальных разрезах. В этом разделе только агрегированные наблюдения по данным текущего прогона:

- Сколько доменов из выборки имеют явные правила для AI-ботов (любые, посчитай по dataset_text)
- Сколько имеют согласованную политику (декларация и поведение совпадают)
- Сколько имеют рассинхрон (декларация одна, реальность другая)
- Что это говорит о состоянии "осознанности" работы с AI-краулерами в выборке
- Если DeepSeek был в прогоне: упоминается ли DeepSeekBot хоть где-то в robots.txt выборки

Если в выборке текущего прогона нашлись сайты с осмысленной allow-политикой и/или с осмысленной disallow-политикой - назови их поимённо как примеры стратегий. Не повторяй описание их поведения, только сошлись на разделы где оно описано. Если таких сайтов в выборке не оказалось - просто констатируй и переходи дальше, не натягивай.

Никакого "проверка одного robots.txt бесполезна" - это интуитивно понятно после прочтения предыдущих разделов и не требует отдельной фразы.

VIII. Эффекты на AI-видимость и эффект замещения (центральная часть отчёта, 5-8 абзацев).
Здесь синтез:
- Тренировочный обход интернета: какая часть контента из выборки уйдёт в "знания" будущих моделей, а какая нет.
- Ответы AI-помощников в реальном времени: какие домены LLM сможет процитировать пользователю прямо сейчас.

Раздел про эффект замещения. Начинай его с явного дисклеймера в одну строку: "Эффект замещения - это не результат прямого замера ответов LLM, а гипотеза, основанная на том, какие источники физически доступны retrieval-слою в данных нашего исследования. Реальные ответы ChatGPT, Claude, Perplexity и DeepSeek (если он был в прогоне) на конкретные запросы про каждый из этих доменов могут отличаться и требуют отдельного измерения."

После дисклеймера развивай гипотезу на основе данных текущего прогона. Для каждой большой категории недоступных доменов сформулируй: какие альтернативные источники существуют в принципе и могут заместить эти домены в retrieval-слое LLM. Выводы про то, "кто кого замещает", формулируй в условном наклонении или через "вероятно". Это не подрывает аналитическую ценность раздела, но возвращает интеллектуальную честность.

Сильная авторская формула "будут говорить другие голоса - в основном западные и эмигрантские" - разрешена и желательна, когда она отражает реальные данные прогона (например, когда крупный российский первоисточник недоступен AI-агентам, а его тематика покрывается доступными западными или эмигрантскими медиа). Используй её именно как точную формулировку наблюдения, не как декорацию. Параллельно в этом разделе нормально использовать и нейтральные технические формы: "в retrieval-результатах будут доминировать сторонние источники", "первичные источники замещаются вторичными".

Если в прогоне был включён DeepSeek и его поведение даёт основания для отдельной гипотезы (например, доступ к доменам, недоступным для других вендоров, или наоборот - закрытость там, где западные AI проходят) - сформулируй её как отдельную линию замещения. Это полезный угол: китайская AI-инфраструктура может видеть российский рунет иначе чем американская. Только если эту разницу подтверждают данные прогона.

Асимметрия между AI-вендорами. Если в данных видно, что один домен пускает один вендор, но режет другой - аудитория разных AI-помощников будет видеть про этот бренд разные картины. Опиши это конкретно по доменам и цифрам, без публицистической рамки.

IX. Практические выводы по ролям.
Короткие подразделы по каждой роли:
- Для CMO и бренд-директора: какие KPI начинают иметь значение, какие старые медиа-стратегии перестают работать в полную силу.
- Для контент-стратега: где размещать материалы, чтобы они попадали в AI-ответы; каких форматов избегать (SPA-сайты, контент за логином, контент за капчей); как пересмотреть выбор площадок для гостевых публикаций и экспертизы.
- Для performance-маркетолога: какие домены теряют ценность как источники переходов из AI-помощников и какой новый канал появляется.
- Для PR и коммуникаций: какие площадки для пресс-релизов и комментариев экспертов работают на AI-видимость, какие нет.
- Для владельца сайта или площадки: что нужно поменять в инфраструктуре, чтобы стать AI-видимым (или наоборот - как осознанно закрыться).

X. Методологические ограничения. Одна точка наблюдения с зарубежного IP. Без выполнения JavaScript на стороне клиента (это сделано намеренно: AI-агенты сами не выполняют JS на момент 2026 года). Снимок одной точки во времени. Главная страница не репрезентирует сайт целиком. Не видно, что в итоге попадает в финальные тренировочные датасеты после фильтрации. Не видно, как это пересекается с retrieval-индексами Bing и Google.

Отдельная методологическая оговорка - про геофильтрацию. Часть наблюдаемых блокировок относятся не к политике сайта в отношении AI-ботов, а к общей IP-фильтрации зарубежного трафика. Найди в данных текущего прогона домены с одинаковым отказом и AI-ботам, и Chrome-baseline (одинаковые статусы, тела, таймауты) и перечисли их. Для российской аудитории, обращающейся к ChatGPT, Claude, Perplexity или DeepSeek с российского IP, retrieval-слой LLM-сервиса по этим доменам может работать иначе - поскольку сами LLM-сервисы (OpenAI, Anthropic, Perplexity физически расположены вне РФ; DeepSeek в Китае), retrieval всё равно идёт с не-российского IP. То есть для конечного пользователя в России различие между "сайт блокирует AI" и "сайт блокирует весь зарубежный трафик" функционально незаметно. Но для интерпретации данных и для рекомендаций владельцам сайтов это разные ситуации.

Дополнительная оговорка про DeepSeek (только если он был в прогоне). DeepSeek - китайский AI-сервис без официальной документации своих краулеров. Имена DeepSeekBot и DeepSeek-User используются на основании наблюдений сообщества и block-листов. Это означает: (а) реальный DeepSeek может ходить и под другими именами или вовсе с обычным браузерным User-Agent; (б) сайты, не упоминающие DeepSeekBot в robots.txt, всё равно могут блокировать настоящий DeepSeek-трафик через WAF по другим признакам; (в) география DeepSeek - Китай, а не США/Европа, поэтому даже если сайт пропускает GPTBot и ClaudeBot, для настоящего DeepSeek картина может быть иной из-за маршрутизации трафика. Наша утилита измеряет реакцию сайта на UA-строку, не на реальный трафик DeepSeek-инфраструктуры. Это нижняя граница оценки.

XI. Финальный абзац - одно конкретное наблюдение, которое должно остаться в голове читателя. Не вывод-обобщение, не призыв, не "в заключение". Это парадокс или наблюдение второго порядка, вытекающее из всех предыдущих данных, но проявляющееся только если посмотреть на картину целиком. Наблюдение должно опираться на данные текущего прогона, а не быть универсальной мудростью.

Хороший характер финального наблюдения: парадокс структуры (а не данных), который объясняет почему ситуация сложилась именно такой; неочевидное следствие из совокупности находок; наблюдение про распределение ответственности или про слепые зоны принятия решений.

Плохой характер финального наблюдения: общая фраза вида "это структурная картина, требующая системного ответа"; призыв к действию; обобщение в стиле "вот так устроен мир".

Длина финала: один абзац, 4-7 предложений. Если получается длиннее - режь.

Каждое утверждение про конкретный домен опирай на конкретные числа из данных - длина извлекаемого текста, размер тела ответа, факт разрешения или запрета в robots.txt, статус-код. Не округляй и не выдумывай.

Оформление текста (Markdown).
- Заголовки разделов: уровень `##`. Подразделы внутри категорий и блоков - `###`. Не используй `#` нигде кроме самого верхнего заголовка отчёта (он уже на уровне `#`).
- Абзацы средней длины: 3-5 предложений. Не лепи 8-предложенные простыни и не дроби текст на однострочники.
- Между абзацами одна пустая строка. Между разделами две пустые строки.
- Сводная таблица - в обычном Markdown-формате с `|` и `-`. После таблицы оставь пустую строку.
- В каждом разделе и в каждой категории выделяй ключевые управленческие тезисы жирным `**...**`. Это короткие фразы по 5-15 слов, не целые предложения. На раздел 1-3 таких выделения, не больше. Если выделений слишком много - они теряют смысл.
- Самые сильные авторские наблюдения, которые имеет смысл вырвать из абзаца и оставить как самостоятельную мысль, оформляй блок-цитатой через `>` (один знак больше в начале строки). Это формат "pull-quote": фразы вроде "Маркетинг может покупать охват, который не превращается в AI-присутствие" или "Площадка может быть видимой для людей и невидимой для AI". На раздел не больше одной такой цитаты, на весь отчёт - 4-7. Не клади в blockquote обычные пояснения, только сильные финальные фразы.
- Списки используй экономно. Только там, где данные действительно списочные (роли в разделе 8, перечень WAF-вендоров). Связный текст в параграфах сильнее списков.
- Не используй `---` (горизонтальные линии) внутри отчёта.

Объём: минимум 2000 слов, верхней границы нет. Не разводи воду, но и не сокращай ради сокращения.

Перед финальной выдачей мысленно прогони текст по чек-листу:
- Заголовок понятен CMO без технической подготовки.
- В заголовке нет необъяснённых технических терминов.
- Первый абзац говорит о бизнес-риске, а не о технической детали, и не начинается с технического термина.
- Каждый технический термин объяснён при первом употреблении.
- Нет конструкций "Это не X, это Y" и аналогичных шаблонных антитез.
- Нет фраз "важно отметить", "стоит отметить", "в современном ландшафте", "ключевой вывод заключается в том, что", "таким образом" в начале финального абзаца, "данный кейс демонстрирует".
- Сильные авторские формулировки сохранены.
- Текст не похож на SEO-статью или LinkedIn-пост.
- В каждом разделе понятно, что это значит для маркетинга, CMO, PR или digital.
- Выводы не шире, чем позволяют данные.

Все материалы предыдущих этапов:

=== STRUCTURAL FACTS ===
{step2_json}

=== CATEGORICAL ANALYSIS ===
{step3_markdown}

=== AI VISIBILITY IMPLICATIONS ===
{step4_markdown}

=== RAW DATA ===
{dataset_text}
"""


# --- Step runners -----------------------------------------------------------


async def _step2_structural(run_id: str, dataset_text: str) -> dict[str, Any]:
    user = _STEP2_USER_TEMPLATE.format(dataset_text=dataset_text)
    messages = [
        {"role": "system", "content": _STEP2_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw, _ = await _call_llm(run_id, messages, "structural-analysis")
    parsed = _parse_json_loose(raw)
    if parsed is None:
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "warn",
                "message": "structural step JSON malformed - retrying with stricter prompt",
            },
        )
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Твой предыдущий ответ был не валидным JSON. Верни ТОЛЬКО валидный JSON, "
                    "без обёрток ```json```, без поясняющего текста. Только JSON."
                ),
            },
        ]
        raw, _ = await _call_llm(run_id, retry_messages, "structural-analysis-retry")
        parsed = _parse_json_loose(raw)
    if parsed is None:
        raise RuntimeError("structural step: LLM returned non-JSON twice")
    return parsed


async def _step3_categorical(
    run_id: str, dataset_text: str, step2_json: dict[str, Any]
) -> str:
    user = _STEP3_USER_TEMPLATE.format(
        step2_json=json.dumps(step2_json, ensure_ascii=False, indent=2),
        dataset_text=dataset_text,
    )
    messages = [
        {"role": "system", "content": _STEP3_SYSTEM},
        {"role": "user", "content": user},
    ]
    content, _ = await _call_llm(run_id, messages, "categorical-analysis")
    return content


async def _step4_implications(
    run_id: str, step2_json: dict[str, Any], step3_markdown: str
) -> str:
    user = _STEP4_USER_TEMPLATE.format(
        step2_json=json.dumps(step2_json, ensure_ascii=False, indent=2),
        step3_markdown=step3_markdown,
    )
    messages = [
        {"role": "system", "content": _STEP4_SYSTEM},
        {"role": "user", "content": user},
    ]
    content, _ = await _call_llm(run_id, messages, "ai-visibility-implications")
    return content


async def _step5_report(
    run_id: str,
    dataset_text: str,
    step2_json: dict[str, Any],
    step3_markdown: str,
    step4_markdown: str,
) -> str:
    user = _STEP5_USER_TEMPLATE.format(
        step2_json=json.dumps(step2_json, ensure_ascii=False, indent=2),
        step3_markdown=step3_markdown,
        step4_markdown=step4_markdown,
        dataset_text=dataset_text,
    )
    messages = [
        {"role": "system", "content": _STEP5_SYSTEM},
        {"role": "user", "content": user},
    ]
    content, _ = await _call_llm(run_id, messages, "final-report")
    return content


# --- Orchestration ----------------------------------------------------------


async def _set_progress(run_id: str, current: int, total: int) -> None:
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run is None:
            # Run was deleted while analyser was running. Bail out silently.
            return
        run.progress_current = current
        run.progress_total = total
        await session.commit()
    bus.publish(
        run_id,
        {"type": "progress", "current": current, "total": total, "phase": "analyzing"},
    )


async def analyze_run(run_id: str) -> None:
    """Run the cross-probe pass + 4-step LLM pipeline. Sets Run.status terminally."""
    current_step = 0
    try:
        added = await apply_ua_conditional_block(run_id)
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "info",
                "message": f"Cross-probe: ua-conditional-block added on {added} probe(s)",
            },
        )

        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                return  # run deleted between crawl_done and analyze_start
            run.status = RunStatus.analyzing
            run.progress_current = 0
            run.progress_total = 4
            await session.commit()
        bus.publish(run_id, {"type": "phase_change", "phase": "analyzing_started"})
        bus.publish(
            run_id, {"type": "progress", "current": 0, "total": 4, "phase": "analyzing"}
        )

        dataset_text, ds_meta = await _build_dataset_text(run_id)
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "info",
                "message": (
                    f"Dataset prepared: mode={ds_meta['mode']}, chars={ds_meta['char_count']}, "
                    f"domains={ds_meta['domains']}, probes={ds_meta['probes']}"
                ),
            },
        )

        intermediate: dict[str, Any] = {"dataset_meta": ds_meta}

        current_step = 2
        step2_json = await _step2_structural(run_id, dataset_text)
        intermediate["step2"] = step2_json
        await _set_progress(run_id, 1, 4)
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "info",
                "message": (
                    f"Structural analysis complete: "
                    f"total_domains={step2_json.get('total_domains', '?')}, "
                    f"total_probes={step2_json.get('total_probes', '?')}"
                ),
            },
        )

        current_step = 3
        step3_md = await _step3_categorical(run_id, dataset_text, step2_json)
        intermediate["step3"] = step3_md
        await _set_progress(run_id, 2, 4)
        bus.publish(
            run_id,
            {"type": "log", "level": "info", "message": "Categorical analysis complete"},
        )

        current_step = 4
        step4_md = await _step4_implications(run_id, step2_json, step3_md)
        intermediate["step4"] = step4_md
        await _set_progress(run_id, 3, 4)
        bus.publish(
            run_id,
            {"type": "log", "level": "info", "message": "AI visibility implications drafted"},
        )

        current_step = 5
        final_md = await _step5_report(run_id, dataset_text, step2_json, step3_md, step4_md)
        intermediate["step5"] = final_md

        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                # Run deleted during the final LLM call; the report is lost.
                return
            run.progress_current = 4
            run.analysis_markdown = final_md
            cfg = dict(run.config_json or {})
            cfg["intermediate_analysis"] = intermediate
            run.config_json = cfg
            flag_modified(run, "config_json")
            run.status = RunStatus.completed
            await session.commit()

        bus.publish(
            run_id, {"type": "progress", "current": 4, "total": 4, "phase": "analyzing"}
        )
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "info",
                "message": f"Final report ready: {len(final_md)} chars",
            },
        )
        bus.publish(run_id, {"type": "phase_change", "phase": "analyzing_done"})
        bus.publish(run_id, {"type": "phase_change", "phase": "completed"})
        bus.publish(run_id, {"type": "final", "status": "completed"})

    except asyncio.CancelledError:
        bus.publish(
            run_id,
            {"type": "log", "level": "error", "message": "analyzer cancelled"},
        )
        bus.publish(run_id, {"type": "phase_change", "phase": "failed"})
        bus.publish(run_id, {"type": "final", "status": "failed"})
        raise
    except BaseException as exc:
        try:
            async with SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == run_id))
                ).scalar_one_or_none()
                if run is not None:
                    run.status = RunStatus.failed
                    run.error_message = (
                        f"Analyzer failed at step {current_step}: "
                        f"{type(exc).__name__}: {exc}"
                    )[:1000]
                    await session.commit()
        except Exception:
            pass
        bus.publish(
            run_id,
            {
                "type": "log",
                "level": "error",
                "message": f"analyzer error at step {current_step}: {type(exc).__name__}: {exc}",
            },
        )
        bus.publish(run_id, {"type": "phase_change", "phase": "failed"})
        bus.publish(run_id, {"type": "final", "status": "failed"})


# Backwards-compat alias for the old stub name.
run_analysis = analyze_run
