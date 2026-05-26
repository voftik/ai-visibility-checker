"""Cheap content-shape detector.

`extract_text_signals` runs over the decoded HTML body and returns:
  - extractable_text_length: how many characters survive after stripping
    <script>/<style> blocks, all tags, and whitespace collapse. This is the
    primary "what would an AI fetcher actually see" signal.
  - tag_paragraph_count / tag_link_count: rough structural markers.
  - looks_like_*: five orthogonal heuristic flags for content shape.
  - primary_language_guess: "ru" / "en" / "mixed" / None on cyrillic ratio.

Pure regex + str.count, no HTML parser dependency. Designed to be tolerant of
broken markup; returns conservative defaults rather than raising.
"""
from __future__ import annotations

import re

_RE_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>", re.I | re.S)
_RE_STYLE = re.compile(r"<style\b[^>]*>.*?</style>", re.I | re.S)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"\s+")

_RE_P = re.compile(r"<p\b", re.I)
_RE_ARTICLE = re.compile(r"<article\b", re.I)
_RE_HEADING = re.compile(r"<h[1-3]\b", re.I)
_RE_LINK = re.compile(r"<a\s[^>]*href=", re.I)
_RE_SCRIPT_OPEN = re.compile(r"<script[\s>]", re.I)

_LOGIN_TOKENS = (
    'name="password"',
    "name='password'",
    'type="password"',
    "type='password'",
    "login-form",
    "sign-in",
    "войти",
    "логин",
)

_REDIRECT_TOKENS = (
    'meta http-equiv="refresh"',
    "meta http-equiv='refresh'",
    "window.location",
    "document.location",
)

_CAPTCHA_TOKENS = (
    "captcha",
    "smartcaptcha",
    "recaptcha",
    "проверка",
    "are you human",
    "checking your browser",
    "just a moment",
)

_ERROR_TOKENS = (
    "404",
    "403",
    "401",
    "498",
    "500",
    "forbidden",
    "not found",
    "access denied",
)

# Phrases that indicate the page is a geo-restriction notice rather than the
# real content. RU phrases first (gosuslugi/banks/regulators), then EN. These
# are intentionally distinct from captcha/error tokens — a geo-block page
# usually serves HTTP 200 with a friendly banner.
_GEO_BLOCK_TOKENS = (
    # Russian
    "отключите vpn",
    "отключите впн",
    "доступ ограничен",
    "доступ запрещ",
    "доступ к сайту ограничен",
    "по соображениям безопасности",
    "из соображений безопасности",
    "недоступен в вашей стране",
    "недоступно в вашей стране",
    "недоступен в вашем регионе",
    "недоступно в вашем регионе",
    "недоступен из-за рубежа",
    "из вашей страны",
    "из вашего региона",
    "географическим ограничен",
    "вы находитесь за пределами",
    "сервис доступен только",
    "только из россии",
    "только на территории",
    # English
    "this content is not available in your country",
    "not available in your region",
    "geo-restricted",
    "geo restricted",
    "access denied for your region",
    "service is only available",
    "blocked in your country",
    "outside your country",
    "vpn detected",
    "please disable your vpn",
    "disable vpn",
)


def _strip_to_text(body: str) -> str:
    if not body:
        return ""
    s = _RE_SCRIPT.sub(" ", body)
    s = _RE_STYLE.sub(" ", s)
    s = _RE_TAG.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s


def _language_guess(text: str) -> str | None:
    head = text[:2000]
    if len(head) < 100:
        return None
    cyr = sum(1 for ch in head if "Ѐ" <= ch <= "ӿ")
    letters = sum(1 for ch in head if ch.isalpha())
    if letters == 0:
        return None
    cyr_ratio = cyr / letters
    if cyr_ratio > 0.7:
        return "ru"
    if cyr_ratio < 0.10:
        return "en"
    return "mixed"


def extract_text_signals(body_text: str, content_type: str | None) -> dict:
    if content_type:
        ct = content_type.lower()
        if ct.startswith(("image/", "audio/", "video/", "application/octet-stream", "application/pdf")):
            return {
                "extractable_text_length": 0,
                "body_size_bytes": 0,
                "body_to_text_ratio": None,
                "tag_paragraph_count": 0,
                "tag_link_count": 0,
                "looks_like_spa_shell": False,
                "looks_like_login_wall": False,
                "looks_like_redirect_shell": False,
                "looks_like_captcha_page": False,
                "looks_like_error_page": False,
                "looks_like_geo_block": False,
                "looks_disproportionate_wrapper": False,
                "primary_language_guess": None,
            }

    body_text = body_text or ""
    body_lower = body_text.lower()
    body_len = len(body_text)

    extractable = _strip_to_text(body_text)
    extractable_len = len(extractable)

    p_count = len(_RE_P.findall(body_text))
    art_count = len(_RE_ARTICLE.findall(body_text))
    h_count = len(_RE_HEADING.findall(body_text))
    paragraph_total = p_count + art_count + h_count
    link_count = len(_RE_LINK.findall(body_text))

    script_count = len(_RE_SCRIPT_OPEN.findall(body_text))

    # "Heavy wrapper around a tiny body" - the page is large in raw bytes but
    # produces almost no readable text. This is the structural fingerprint of
    # an interstitial/banner/SPA stub regardless of which language or wording
    # the page uses. We use it as a soft signal that strengthens other
    # detectors.
    body_to_text_ratio = (body_len / extractable_len) if extractable_len else float("inf")
    looks_disproportionate = (
        body_len >= 1500               # at least 1.5KB on the wire (banners/SPAs are rarely smaller)
        and extractable_len < 300      # very little real text
        and body_to_text_ratio > 15    # at least 15 bytes per visible char
    )

    looks_like_spa_shell = (
        extractable_len < 500
        and paragraph_total == 0
        and script_count >= 3
    )

    looks_like_login_wall = (
        any(t in body_lower for t in _LOGIN_TOKENS) and extractable_len < 2000
    )

    looks_like_redirect_shell = (
        any(t in body_lower for t in _REDIRECT_TOKENS) and extractable_len < 1000
    )

    looks_like_captcha_page = any(t in body_lower for t in _CAPTCHA_TOKENS)

    looks_like_error_page = (
        extractable_len < 800 and any(t in body_lower for t in _ERROR_TOKENS)
    )

    # Geo-block: page that the server serves with HTTP 200 telling the visitor
    # they are coming from the wrong country/IP. Distinct from captcha/error —
    # the visitor is not asked to prove they are human, they are told to leave.
    #
    # Two ways to fire:
    #   1. Token match: any RU/EN geo-block phrase is present AND body is
    #      shortish (extractable_len < 4000) so a stray "vpn" mention in a
    #      real article doesn't mis-fire.
    #   2. Structural match: a large body (>= 3KB) returned almost no readable
    #      text (< 300 chars). This catches localised banners whose wording we
    #      have not enumerated. To stay specific we require either a typical
    #      banner-page link count (<= 5 anchors — interstitials are minimal)
    #      OR an extreme byte-per-char ratio plus paragraph_total <= 1.
    matched_token = any(t in body_lower for t in _GEO_BLOCK_TOKENS) and extractable_len < 4000
    matched_structural = looks_disproportionate and (
        link_count <= 5 or paragraph_total <= 1
    )
    looks_like_geo_block = matched_token or matched_structural

    return {
        "extractable_text_length": extractable_len,
        "body_size_bytes": body_len,
        "body_to_text_ratio": (
            None if not extractable_len else round(body_to_text_ratio, 1)
        ),
        "tag_paragraph_count": paragraph_total,
        "tag_link_count": link_count,
        "looks_like_spa_shell": bool(looks_like_spa_shell),
        "looks_like_login_wall": bool(looks_like_login_wall),
        "looks_like_redirect_shell": bool(looks_like_redirect_shell),
        "looks_like_captcha_page": bool(looks_like_captcha_page),
        "looks_like_error_page": bool(looks_like_error_page),
        "looks_like_geo_block": bool(looks_like_geo_block),
        "looks_disproportionate_wrapper": bool(looks_disproportionate),
        "primary_language_guess": _language_guess(extractable),
    }
