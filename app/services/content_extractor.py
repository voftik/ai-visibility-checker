"""Content extraction and render-strategy signals for crawler responses.

The important distinction is between an authentication *form* and an
authentication *wall*. A newsletter/login widget on an otherwise readable
page must never erase the page's real content.
"""
from __future__ import annotations

import json
import re
from html import unescape
from typing import Any

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")
_FRAMEWORK_MARKERS = (
    "__next_data__",
    "__nuxt__",
    "__remixcontext",
    "data-reactroot",
    "ng-version",
    'id="__next"',
    'id="app"',
    'id="root"',
)
_REDIRECT_TOKENS = (
    'http-equiv="refresh"',
    "window.location",
    "document.location",
)
_CAPTCHA_TOKENS = (
    "captcha",
    "smartcaptcha",
    "recaptcha",
    "are you human",
    "checking your browser",
    "just a moment",
)
_ERROR_TOKENS = (
    "404",
    "403",
    "401",
    "forbidden",
    "not found",
    "access denied",
    "страница не найдена",
    "доступ запрещён",
)
_AUTH_TOKENS = (
    "sign in",
    "log in",
    "login",
    "авторизация",
    "войдите в аккаунт",
    "войдите, чтобы продолжить",
    "требуется вход",
)
_GEO_BLOCK_TOKENS = (
    "отключите vpn",
    "отключите впн",
    "доступ ограничен",
    "доступ к сайту ограничен",
    "по соображениям безопасности",
    "из соображений безопасности",
    "недоступен в вашей стране",
    "недоступно в вашей стране",
    "недоступен в вашем регионе",
    "недоступно в вашем регионе",
    "только из россии",
    "this content is not available in your country",
    "not available in your region",
    "geo-restricted",
    "blocked in your country",
    "vpn detected",
    "please disable your vpn",
)
_SCHEMA_ORG_TYPE_URL = re.compile(
    r"^https?://(?:www\.)?schema\.org/([^/?#]+)/?$",
    re.I,
)


def _clean_text(value: str | None) -> str:
    return _WS.sub(" ", unescape(value or "")).strip()


def _language_guess(text: str) -> str | None:
    if len(text) < 100:
        return None
    letters = sum(1 for char in text if char.isalpha())
    if not letters:
        return None
    cyrillic = sum(1 for char in text if "Ѐ" <= char <= "ӿ")
    ratio = cyrillic / letters
    if ratio > 0.70:
        return "ru"
    if ratio < 0.10:
        return "en"
    return "mixed"


def _schema_type_name(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _SCHEMA_ORG_TYPE_URL.fullmatch(text)
    return match.group(1) if match else text


def _deduplicate_types(values: list[str]) -> list[str]:
    """Return the complete, stable type set found in the stored document.

    The crawler reads the physical HTML through EOF.  A silent type-count cap
    would still turn a complete positive observation into an incomplete one
    without leaving any provenance that this happened.
    """

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _schema_type_name(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _jsonld_observation(soup: BeautifulSoup) -> dict[str, Any]:
    """Parse every JSON-LD script and retain both evidence and failures."""

    found: list[str] = []
    errors: list[dict[str, Any]] = []
    tags = [
        tag
        for tag in soup.find_all("script")
        if str(tag.get("type") or "")
        .split(";", 1)[0]
        .strip()
        .casefold()
        == "application/ld+json"
    ]

    def visit(value: Any) -> None:
        # Iterative traversal avoids treating deeply nested but valid JSON-LD
        # as a parser failure merely because it exceeds Python's call stack.
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                item_type = current.get("@type")
                if isinstance(item_type, str):
                    found.append(item_type)
                elif isinstance(item_type, list):
                    found.extend(str(item) for item in item_type if item)
                pending.extend(reversed(list(current.values())))
            elif isinstance(current, list):
                pending.extend(reversed(current))

    parsed_count = 0
    for script_index, tag in enumerate(tags):
        raw = tag.string if isinstance(tag.string, str) else tag.get_text()
        try:
            document = json.loads(raw or "")
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "script_index": script_index,
                    "error_type": "json_decode_error",
                    "message": exc.msg,
                    "line": exc.lineno,
                    "column": exc.colno,
                    "char_offset": exc.pos,
                }
            )
            continue
        except (TypeError, ValueError) as exc:
            errors.append(
                {
                    "script_index": script_index,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        parsed_count += 1
        visit(document)

    failed_count = len(errors)
    return {
        "types": _deduplicate_types(found),
        "script_count": len(tags),
        "parsed_count": parsed_count,
        "failed_count": failed_count,
        "errors": errors,
        "state": (
            "complete"
            if failed_count == 0
            else "partial"
            if parsed_count > 0
            else "failed"
        ),
    }


def _microdata_types(soup: BeautifulSoup) -> list[str]:
    found: list[str] = []
    for tag in soup.find_all(attrs={"itemtype": True}):
        raw_value = tag.get("itemtype")
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            for token in str(value or "").split():
                match = _SCHEMA_ORG_TYPE_URL.fullmatch(token.strip())
                if match:
                    found.append(match.group(1))
    return _deduplicate_types(found)


def _structured_data_types(
    soup: BeautifulSoup,
    *,
    jsonld: dict[str, Any],
) -> list[str]:
    return _deduplicate_types(
        [
            *(jsonld.get("types") or []),
            *_microdata_types(soup),
        ]
    )


def _extract_main_text(soup: BeautifulSoup) -> tuple[str, bool]:
    candidates = soup.select("main, article, [role='main']")
    has_semantic_main = bool(candidates)
    container = max(
        candidates,
        key=lambda node: len(_clean_text(node.get_text(" ", strip=True))),
        default=soup.body or soup,
    )
    clone = BeautifulSoup(str(container), "html.parser")
    for node in clone.select(
        "script, style, noscript, template, svg, nav, header, footer, aside, "
        "dialog, form, [aria-hidden='true']"
    ):
        node.decompose()
    return _clean_text(clone.get_text(" ", strip=True)), has_semantic_main


def _empty_result() -> dict[str, Any]:
    return {
        "extractable_text_length": 0,
        "main_content_length": 0,
        "body_size_bytes": 0,
        "body_to_text_ratio": None,
        "tag_paragraph_count": 0,
        "tag_link_count": 0,
        "script_count": 0,
        "jsonld": {
            "script_count": 0,
            "parsed_count": 0,
            "failed_count": 0,
            "errors": [],
            "state": "unknown",
        },
        "title": None,
        "meta_description": None,
        "canonical_url": None,
        "structured_data_types": [],
        "structured_data_complete": False,
        "render_strategy": "unknown",
        "render_strategy_confidence": "low",
        "render_strategy_reasons": [],
        "auth_form_present": False,
        "looks_like_login_wall": False,
        "looks_like_spa_shell": False,
        "looks_like_redirect_shell": False,
        "looks_like_captcha_page": False,
        "looks_like_error_page": False,
        "looks_like_geo_block": False,
        "looks_disproportionate_wrapper": False,
        "primary_language_guess": None,
        "main_text_excerpt": "",
        "visible_text_excerpt": "",
    }


def extract_text_signals(body_text: str, content_type: str | None) -> dict[str, Any]:
    if content_type:
        media_type = content_type.lower()
        if media_type.startswith(
            ("image/", "audio/", "video/", "application/octet-stream", "application/pdf")
        ):
            return _empty_result()

    html = body_text or ""
    if not html:
        return _empty_result()

    lower = html.lower()
    soup = BeautifulSoup(html, "html.parser")
    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "") or None
    description_node = soup.find(
        "meta",
        attrs={"name": re.compile(r"^description$", re.I)},
    )
    meta_description = (
        _clean_text(description_node.get("content"))
        if description_node and description_node.get("content")
        else None
    )
    canonical_node = soup.find("link", attrs={"rel": lambda value: value and "canonical" in value})
    canonical_url = (
        str(canonical_node.get("href")).strip()
        if canonical_node and canonical_node.get("href")
        else None
    )
    jsonld = _jsonld_observation(soup)
    structured_types = _structured_data_types(soup, jsonld=jsonld)

    script_count = len(soup.find_all("script"))
    link_count = len(soup.find_all("a", href=True))
    paragraph_count = len(soup.select("p, article, h1, h2, h3"))

    visible_clone = BeautifulSoup(html, "html.parser")
    for node in visible_clone.select(
        "script, style, noscript, template, svg, [aria-hidden='true']"
    ):
        node.decompose()
    visible_text = _clean_text(visible_clone.get_text(" ", strip=True))
    main_text, has_semantic_main = _extract_main_text(soup)
    visible_length = len(visible_text)
    main_length = len(main_text)
    body_length = len(html)
    body_to_text = body_length / visible_length if visible_length else None

    password_nodes = list(soup.select("input[type='password']"))
    login_forms = [
        form
        for form in soup.find_all("form")
        if any(
            token in str(form).lower()
            for token in ("login", "sign-in", "signin", "авториза")
        )
        or form.select_one("input[type='password']") is not None
    ]
    auth_form_present = bool(password_nodes or login_forms)
    auth_controls = [*login_forms, *password_nodes]
    auth_form_in_main = any(
        node.find_parent(["main", "article"]) is not None
        or node.find_parent(attrs={"role": "main"}) is not None
        for node in auth_controls
    )
    auth_language = any(token in visible_text.lower() for token in _AUTH_TOKENS)
    meaningful_content = main_length >= 800 or (
        has_semantic_main and main_length >= 500 and paragraph_count >= 3
    ) or (
        has_semantic_main
        and main_length >= 180
        and not auth_form_in_main
    )
    looks_like_login_wall = bool(
        auth_form_present
        and auth_language
        and not meaningful_content
        and main_length < 500
        and visible_length < 1600
    )

    framework_marker = any(marker in lower for marker in _FRAMEWORK_MARKERS)
    app_root = bool(soup.select_one("#__next, #app, #root, [data-reactroot], [ng-version]"))
    if visible_length < 450 and script_count >= 3 and (framework_marker or app_root):
        render_strategy = "client_rendered_shell"
        render_confidence = "high"
        render_reasons = [
            "В исходном HTML почти нет текста.",
            "Страница содержит корневой контейнер приложения и несколько скриптов.",
        ]
    elif framework_marker and main_length >= 500:
        render_strategy = "hybrid_ssr_hydration"
        render_confidence = "high"
        render_reasons = [
            "Содержательный текст уже есть в исходном HTML.",
            "Найдены признаки последующей гидратации интерфейса.",
        ]
    elif main_length >= 500 and script_count <= 2:
        render_strategy = "static_html"
        render_confidence = "medium"
        render_reasons = ["Основной текст доступен без выполнения JavaScript."]
    elif main_length >= 500:
        render_strategy = "server_rendered"
        render_confidence = "medium"
        render_reasons = ["Основной текст присутствует в серверном HTML."]
    else:
        render_strategy = "unknown"
        render_confidence = "low"
        render_reasons = ["По исходному HTML нельзя надёжно определить способ рендеринга."]

    disproportionate = bool(
        body_length >= 1500
        and visible_length < 300
        and body_to_text is not None
        and body_to_text > 15
    )
    looks_like_spa_shell = render_strategy == "client_rendered_shell"
    looks_like_redirect = any(token in lower for token in _REDIRECT_TOKENS) and visible_length < 1000
    looks_like_captcha = any(token in lower for token in _CAPTCHA_TOKENS) and visible_length < 4000
    looks_like_error = (
        visible_length < 1000
        and any(token in f"{title or ''} {visible_text[:800]}".lower() for token in _ERROR_TOKENS)
    )
    geo_token_match = (
        any(token in visible_text.lower() for token in _GEO_BLOCK_TOKENS)
        and visible_length < 4000
    )
    # A short confirmation, checkout or application page can legitimately have
    # a heavy HTML wrapper and only one paragraph. Structure alone is not
    # evidence of a geographic block; a geo-specific marker must be present.
    # Keep the structural signal separately so the comparative probe layer can
    # still notice a suspiciously thin response without inventing its cause.
    geo_structural_match = bool(geo_token_match and disproportionate)

    return {
        "extractable_text_length": visible_length,
        "main_content_length": main_length,
        "body_size_bytes": body_length,
        "body_to_text_ratio": round(body_to_text, 1) if body_to_text else None,
        "tag_paragraph_count": paragraph_count,
        "tag_link_count": link_count,
        "script_count": script_count,
        "jsonld": {
            key: value for key, value in jsonld.items() if key != "types"
        },
        "title": title,
        "meta_description": meta_description,
        "canonical_url": canonical_url,
        "structured_data_types": structured_types,
        # A valid type found in another script remains positive evidence, but
        # one malformed JSON-LD block makes absence claims about the complete
        # machine-readable description unknown.
        "structured_data_complete": jsonld["failed_count"] == 0,
        "render_strategy": render_strategy,
        "render_strategy_confidence": render_confidence,
        "render_strategy_reasons": render_reasons,
        "auth_form_present": bool(auth_form_present),
        "looks_like_login_wall": looks_like_login_wall,
        "looks_like_spa_shell": looks_like_spa_shell,
        "looks_like_redirect_shell": bool(looks_like_redirect),
        "looks_like_captcha_page": bool(looks_like_captcha),
        "looks_like_error_page": bool(looks_like_error),
        "looks_like_geo_block": bool(geo_token_match or geo_structural_match),
        "looks_disproportionate_wrapper": disproportionate,
        "primary_language_guess": _language_guess(main_text or visible_text),
        # The caller may persist or partition these values.  Do not silently
        # turn a long page into a prefix here; transport-level body truncation
        # is tracked separately as unknown/incomplete.
        "main_text_excerpt": main_text,
        "visible_text_excerpt": visible_text,
    }
