"""Protection / WAF / interstitial detection from a single HTTP probe.

Each rule is an independent predicate; multiple markers can fire at once.
Add new signatures by extending RULES.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeFacts:
    status: int | None
    headers_lower: dict[str, str]  # header name lower-cased -> value
    body_text: str
    body_text_lower: str
    final_url: str
    final_url_lower: str
    tls_ok: bool | None
    domain: str
    content_signals: dict | None = None


CHALLENGE_MARKERS = {
    "cloudflare-challenge",
    "ddos-guard-challenge",
    "qrator-challenge",
    "variti-challenge",
    "yandex-smartcaptcha",
}

KNOWN_RUSSIAN_TLS_BLOCKED = {
    "sberbank.ru",
    "vtb.ru",
    "gazprombank.ru",
}


def _has_header(facts: ProbeFacts, name: str, contains: str | None = None) -> bool:
    val = facts.headers_lower.get(name.lower())
    if val is None:
        return False
    if contains is None:
        return True
    return contains.lower() in val.lower()


def _any_header_prefix(facts: ProbeFacts, prefix: str) -> bool:
    p = prefix.lower()
    return any(h.startswith(p) for h in facts.headers_lower)


def _is_cloudflare(facts: ProbeFacts) -> bool:
    if _has_header(facts, "server", "cloudflare"):
        return True
    return "cf-ray" in facts.headers_lower or "cf-cache-status" in facts.headers_lower


def _is_qrator(facts: ProbeFacts) -> bool:
    if _has_header(facts, "server", "qrator"):
        return True
    if "qrator-id" in facts.headers_lower:
        return True
    return "qrator.net" in facts.body_text_lower


def _is_ddos_guard(facts: ProbeFacts) -> bool:
    if _has_header(facts, "server", "ddos-guard"):
        return True
    if any(h.startswith("__ddg") for h in facts.headers_lower):
        return True
    set_cookie = facts.headers_lower.get("set-cookie", "")
    if "__ddg" in set_cookie.lower():
        return True
    return "ddos-guard.net" in facts.body_text_lower


def _is_variti(facts: ProbeFacts) -> bool:
    if facts.status == 498:
        return True
    if _any_header_prefix(facts, "x-variti"):
        return True
    return "variti.io" in facts.body_text_lower


def _is_yandex_captcha(facts: ProbeFacts) -> bool:
    if "/showcaptcha?" in facts.final_url_lower or "captcha.yandex" in facts.final_url_lower:
        return True
    if "smartcaptcha" in facts.body_text_lower:
        return True
    return "x-yandex-captcha" in facts.headers_lower


def _is_akamai(facts: ProbeFacts) -> bool:
    if _has_header(facts, "server", "akamaighost"):
        return True
    return _any_header_prefix(facts, "x-akamai")


def _is_imperva(facts: ProbeFacts) -> bool:
    if "x-iinfo" in facts.headers_lower:
        return True
    return "_incapsula_resource" in facts.body_text_lower


def _is_datadome(facts: ProbeFacts) -> bool:
    if "x-datadome-cid" in facts.headers_lower:
        return True
    set_cookie = facts.headers_lower.get("set-cookie", "")
    return "datadome" in set_cookie.lower()


def _is_russian_tls_block(facts: ProbeFacts) -> bool:
    if facts.tls_ok is not False:
        return False
    return facts.domain.lower().endswith(".ru") or facts.domain.lower() in KNOWN_RUSSIAN_TLS_BLOCKED


_GEO_PHRASES = (
    # Click-through interstitials (user is invited to continue)
    "i agree",
    "я согласен",
    "continue to site",
    "accept and continue",
    "подтвердите",
    # Hard geo-blocks (user is told to leave / disable VPN)
    "отключите vpn",
    "отключите впн",
    "доступ ограничен",
    "доступ запрещ",
    "по соображениям безопасности",
    "из соображений безопасности",
    "недоступен в вашей стране",
    "недоступен в вашем регионе",
    "недоступно в вашей стране",
    "недоступно в вашем регионе",
    "не доступен в вашей стране",
    "географическим ограничен",
    "только из россии",
    "только на территории",
    "vpn detected",
    "please disable your vpn",
    "disable vpn",
    "this content is not available in your country",
    "not available in your region",
    "blocked in your country",
)


def _is_geo_interstitial(facts: ProbeFacts) -> bool:
    if facts.status != 200:
        return False
    if len(facts.body_text.encode("utf-8", errors="ignore")) >= 30 * 1024:
        return False
    return any(p in facts.body_text_lower for p in _GEO_PHRASES)


_LOGIN_FORM = re.compile(r"<input[^>]+(type=['\"]?password|name=['\"]?(login|password|email))", re.I)
_BODY_HAS_CONTENT = re.compile(r"<(article|main|section|p|h1|h2)[\s>]", re.I)


def _is_auth_required(facts: ProbeFacts) -> bool:
    if facts.status == 401:
        return True
    if facts.status == 200 and _LOGIN_FORM.search(facts.body_text):
        if not _BODY_HAS_CONTENT.search(facts.body_text):
            return True
    return False


# Composite (rely on prior simple detectors)


def _cloudflare_challenge(facts: ProbeFacts, simple: set[str]) -> bool:
    if "cloudflare" not in simple:
        return False
    if facts.status not in (403, 503):
        return False
    blob = facts.body_text_lower
    return any(s in blob for s in ("just a moment", "checking your browser", "cf-challenge", "challenge-platform"))


def _qrator_challenge(facts: ProbeFacts, simple: set[str]) -> bool:
    if "qrator" not in simple:
        return False
    if facts.status is not None and 400 <= facts.status < 600:
        return True
    title_match = re.search(r"<title[^>]*>([^<]*)</title>", facts.body_text, re.I)
    return bool(title_match and "qrator" in title_match.group(1).lower())


def _ddos_guard_challenge(facts: ProbeFacts, simple: set[str]) -> bool:
    if "ddos-guard" not in simple:
        return False
    blob = facts.body_text_lower
    if "ddos-guard" not in blob:
        return False
    return "checking" in blob or "проверка" in blob


def _variti_challenge(facts: ProbeFacts, simple: set[str]) -> bool:
    if "variti" not in simple:
        return False
    return facts.status in (498, 403, 429)


# WAF / infrastructure markers — meaningful for any HTTP response (incl. robots.txt).
WAF_RULES: list[tuple[str, Callable[[ProbeFacts], bool]]] = [
    ("cloudflare", _is_cloudflare),
    ("qrator", _is_qrator),
    ("ddos-guard", _is_ddos_guard),
    ("variti", _is_variti),
    ("akamai", _is_akamai),
    ("imperva", _is_imperva),
    ("datadome", _is_datadome),
    ("russian-tls-block", _is_russian_tls_block),
]

def _is_content_shell(facts: ProbeFacts) -> bool:
    cs = facts.content_signals or {}
    return bool(cs.get("looks_like_spa_shell") or cs.get("looks_like_redirect_shell"))


def _is_content_login_wall(facts: ProbeFacts) -> bool:
    cs = facts.content_signals or {}
    return bool(cs.get("looks_like_login_wall"))


def _is_content_geo_block(facts: ProbeFacts) -> bool:
    cs = facts.content_signals or {}
    return bool(cs.get("looks_like_geo_block"))


# Content-shape markers — only meaningful when probing an actual HTML page.
CONTENT_RULES: list[tuple[str, Callable[[ProbeFacts], bool]]] = [
    ("yandex-smartcaptcha", _is_yandex_captcha),
    ("geo-interstitial", _is_geo_interstitial),
    ("auth-required", _is_auth_required),
    ("content-shell", _is_content_shell),
    ("content-login-wall", _is_content_login_wall),
    ("content-geo-block", _is_content_geo_block),
]

COMPOSITE_RULES: list[tuple[str, Callable[[ProbeFacts, set[str]], bool]]] = [
    ("cloudflare-challenge", _cloudflare_challenge),
    ("qrator-challenge", _qrator_challenge),
    ("ddos-guard-challenge", _ddos_guard_challenge),
    ("variti-challenge", _variti_challenge),
]


def detect_protections(
    *,
    status: int | None,
    headers: dict[str, str] | None,
    body_text: str,
    final_url: str,
    tls_ok: bool | None,
    domain: str,
    body_looks_empty: bool,
    probe_type: str = "main_page",
    content_signals: dict | None = None,
) -> tuple[list[str], bool]:
    """Return (markers, challenge_detected).

    markers: ordered, deduped list of detected protection labels.
    challenge_detected: True if any marker is in CHALLENGE_MARKERS.

    For probe_type != "main_page" the content-shape rules (SPA shell, geo
    interstitial, auth wall, captcha pages) are skipped — they only make sense
    for a normal HTML page, not for robots.txt or other endpoints.
    """
    headers_lower = {k.lower(): v for k, v in (headers or {}).items()}
    facts = ProbeFacts(
        status=status,
        headers_lower=headers_lower,
        body_text=body_text or "",
        body_text_lower=(body_text or "").lower(),
        final_url=final_url or "",
        final_url_lower=(final_url or "").lower(),
        tls_ok=tls_ok,
        domain=domain,
        content_signals=content_signals,
    )

    apply_content_rules = probe_type == "main_page"

    markers: list[str] = []
    seen: set[str] = set()

    rules = list(WAF_RULES)
    if apply_content_rules:
        rules.extend(CONTENT_RULES)
    for name, predicate in rules:
        try:
            if predicate(facts) and name not in seen:
                markers.append(name)
                seen.add(name)
        except Exception:
            continue
    for name, predicate in COMPOSITE_RULES:
        try:
            if predicate(facts, seen) and name not in seen:
                markers.append(name)
                seen.add(name)
        except Exception:
            continue

    if (
        apply_content_rules
        and body_looks_empty
        and status == 200
        and "spa-empty-shell" not in seen
    ):
        markers.append("spa-empty-shell")
        seen.add("spa-empty-shell")

    challenge = any(m in CHALLENGE_MARKERS for m in markers)
    return markers, challenge
