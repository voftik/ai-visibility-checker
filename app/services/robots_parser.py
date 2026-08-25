"""Minimal robots.txt parser that preserves raw directives per user-agent block.

Unlike urllib.robotparser, this returns the exact directive lines so we can
store them in RobotsRule.raw_directives for human inspection later.

Returns a list of (bot_name, rule, raw_directives) tuples for the bots in
KNOWN_AI_BOTS plus the wildcard "*".
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

KNOWN_AI_BOTS: list[str] = [
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
    "CCBot",
    "Google-Extended",
    "Googlebot",
    "GoogleOther",
    "Google-Agent",
    "Google-NotebookLM",
    "Google-CloudVertexBot",
    "Gemini-Deep-Research",
    "Applebot-Extended",
    "Bytespider",
    "Meta-ExternalAgent",
    "*",
]


def _parse_blocks(text: str) -> list[tuple[list[str], list[tuple[str, str]]]]:
    """Split robots.txt into (user_agents, directives) blocks.

    A block starts on a User-agent line and continues until a blank line or
    a User-agent line *after* at least one non-UA directive (per RFC, sequential
    User-agent lines belong to the same block).
    """
    blocks: list[tuple[list[str], list[tuple[str, str]]]] = []
    cur_uas: list[str] = []
    cur_dirs: list[tuple[str, str]] = []
    seen_directive = False

    def flush() -> None:
        nonlocal cur_uas, cur_dirs, seen_directive
        # Sitemap is a global robots.txt directive and is valid before any
        # User-agent block. Preserve it as wildcard metadata instead of
        # silently discarding the discovery root during ``flush``.
        if cur_uas or any(key == "sitemap" for key, _value in cur_dirs):
            blocks.append((cur_uas or ["*"], cur_dirs))
        cur_uas = []
        cur_dirs = []
        seen_directive = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            flush()
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            if seen_directive:
                flush()
            cur_uas.append(value)
        elif key in ("allow", "disallow", "crawl-delay", "sitemap", "host"):
            if not cur_uas and key != "sitemap":
                continue
            cur_dirs.append((key, value))
            seen_directive = True
    flush()
    return blocks


def _classify(directives: list[tuple[str, str]]) -> str:
    non_meta = [(k, v) for k, v in directives if k in ("allow", "disallow")]
    if not non_meta:
        return "partial"
    if len(non_meta) == 1:
        key, value = non_meta[0]
        if key == "disallow" and value == "/":
            return "disallow_all"
        if key == "disallow" and value == "":
            return "allow_all"
        if key == "allow" and value == "/":
            return "allow_all"
    if all(k == "disallow" and v == "/" for k, v in non_meta):
        return "disallow_all"
    if all((k == "disallow" and v == "") or (k == "allow" and v == "/") for k, v in non_meta):
        return "allow_all"
    return "partial"


def _format_block(uas: list[str], directives: list[tuple[str, str]]) -> str:
    lines: list[str] = [f"User-agent: {ua}" for ua in uas]
    for key, value in directives:
        lines.append(f"{key.capitalize() if key != 'crawl-delay' else 'Crawl-delay'}: {value}")
    return "\n".join(lines)


def parse_robots(text: str | None) -> list[tuple[str, str, str]]:
    """Return [(bot_name, rule, raw_directives), ...] for KNOWN_AI_BOTS.

    rule ∈ {"allow_all", "disallow_all", "partial", "not_mentioned"}.
    """
    if not text:
        return [(bot, "not_mentioned", "") for bot in KNOWN_AI_BOTS]

    blocks = _parse_blocks(text)
    result: list[tuple[str, str, str]] = []
    for bot in KNOWN_AI_BOTS:
        exact_blocks = [
            block
            for block in blocks
            if any(
                user_agent.strip().casefold() == bot.casefold()
                for user_agent in block[0]
            )
        ]
        wildcard_blocks = [
            block
            for block in blocks
            if any(user_agent.strip() == "*" for user_agent in block[0])
        ]
        applicable = (
            exact_blocks
            if exact_blocks
            else wildcard_blocks
            if bot != "*"
            else wildcard_blocks
        )
        if not applicable:
            result.append((bot, "not_mentioned", ""))
            continue
        directives = [
            directive
            for _user_agents, block_directives in applicable
            for directive in block_directives
        ]
        rule = _classify(directives)
        raw = "\n\n".join(
            _format_block(user_agents, block_directives)
            for user_agents, block_directives in applicable
        )
        result.append((bot, rule, raw))
    return result


def robots_path_allowed(raw_directives: str | None, url: str) -> bool | None:
    """Evaluate applicable stored rules for one URL using longest-match."""

    raw = str(raw_directives or "").strip()
    if not raw or raw.startswith("<robots.txt unavailable:"):
        return None
    directives = [
        directive
        for _user_agents, block_directives in _parse_blocks(raw)
        for directive in block_directives
        if directive[0] in {"allow", "disallow"}
    ]
    parsed = urlsplit(url)
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"

    matches: list[tuple[int, bool]] = []
    for key, value in directives:
        pattern = value.strip()
        if key == "disallow" and not pattern:
            continue
        if not pattern:
            continue
        end_anchored = pattern.endswith("$")
        if end_anchored:
            pattern = pattern[:-1]
        expression = re.escape(pattern).replace(r"\*", ".*")
        expression = f"^{expression}{'$' if end_anchored else ''}"
        if re.search(expression, target):
            specificity = len(pattern.replace("*", ""))
            matches.append((specificity, key == "allow"))
    if not matches:
        return True
    longest = max(length for length, _allowed in matches)
    # Allow wins ties, matching RFC 9309's least restrictive rule.
    return any(allowed for length, allowed in matches if length == longest)


def parse_robots_unavailable(error_class: str | None) -> list[tuple[str, str, str]]:
    """Single wildcard entry to record when robots.txt could not be fetched."""
    note = f"<robots.txt unavailable: {error_class or 'unknown'}>"
    return [("*", "not_mentioned", note)]
