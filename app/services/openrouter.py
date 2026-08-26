from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import inspect
import json
import math
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from app.config import settings
from app.services.long_response import (
    DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
    LONG_RESPONSE_HARNESS_VERSION,
    ResponseMode,
    STRUCTURED_CONTINUATION_VERSION,
    StructuredContinuationLedger,
    text_sha256,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGE_URL = "https://openrouter.ai/api/v1/images"
# Single-model metadata uses the singular ``/model/:author/:slug`` route.
# ``/models/:author/:slug/endpoints`` is a different endpoint and must not be
# used as a capability lookup.
MODEL_URL = "https://openrouter.ai/api/v1/model"


class OpenRouterError(RuntimeError):
    pass


class OpenRouterContextHeadroomError(OpenRouterError):
    """A serialized request leaves no room for a provider completion."""


class WebSearchPolicy(StrEnum):
    """Auditable web-access contract for one OpenRouter request."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    NATIVE_REQUIRED = "native_required"


WEB_ATTESTATION_VERSION = "aiv-openrouter-web-attestation-v1"
TRANSPORT_METADATA_VERSION = "aiv-openrouter-transport-v1"
OUTPUT_ENVELOPE_VERSION = "aiv-openrouter-output-envelope-v1"
PHYSICAL_POST_AUDIT_VERSION = "aiv-openrouter-physical-post-audit-v1"
STRUCTURED_CHECKPOINT_VERSION = "aiv-structured-checkpoint-v1"
PHYSICAL_POST_AUDIT_DELTA_VERSION = "aiv-openrouter-physical-post-audit-v2"
STRUCTURED_CHECKPOINT_DELTA_VERSION = "aiv-structured-checkpoint-v2"
STRUCTURED_AUDIT_EVENT_VERSION_ATTR = "aiv_structured_audit_event_version"
STRUCTURED_AUDIT_DELTA_CAPABILITY = "aiv-structured-audit-delta-v2"
STRUCTURED_TERMINAL_SEMANTIC_FAILURE_VERSION = (
    "aiv-structured-terminal-semantic-failure-v1"
)
REQUEST_ENVELOPE_ESTIMATE_VERSION = "aiv-request-envelope-estimate-v1"
_MODEL_ENVELOPE_TTL_SECONDS = 60 * 60
_MODEL_ENVELOPE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_REQUEST_WRAPPER_TOKEN_UPPER_BOUND = 256


def _monotonic() -> float:
    """Indirection keeps logical-deadline tests from patching asyncio's clock."""

    return time.monotonic()


class OutputTokenPolicy(StrEnum):
    """How a single provider call chooses its physical output envelope."""

    PROVIDER_DEFAULT = "provider_default"
    MODEL_MAX = "model_max_available"


@dataclass(frozen=True)
class PanelModel:
    key: str
    label: str
    model: str
    memory_model: str | None


@dataclass
class ChatResult:
    text: str
    parsed: dict[str, Any] | list[Any] | None
    citations: list[dict[str, str]]
    usage: dict[str, Any]
    annotations: list[dict[str, Any]]
    request_policy: dict[str, Any]
    web_attestation: dict[str, Any]
    router_metadata: dict[str, Any]
    transport: dict[str, Any] = field(default_factory=dict)


class OpenRouterResponseContractError(OpenRouterError):
    """HTTP succeeded, but the model response is unusable by the caller."""

    def __init__(self, message: str, *, result: ChatResult):
        super().__init__(message)
        self.result = result


class OpenRouterOutputLimitError(OpenRouterResponseContractError):
    """The provider stopped before the requested output was complete."""


class OpenRouterStructuredContinuationError(OpenRouterResponseContractError):
    """A continuable JSON document could not be reconstructed safely."""

    def __init__(
        self,
        message: str,
        *,
        result: ChatResult,
        manifest: dict[str, Any],
    ):
        super().__init__(message, result=result)
        self.manifest = manifest


class OpenRouterStructuredContinuationCancelled(asyncio.CancelledError):
    """Cancellation that still exposes the exact accepted continuation state."""

    def __init__(
        self,
        message: str,
        *,
        result: ChatResult,
        manifest: dict[str, Any],
    ):
        super().__init__(message)
        self.result = result
        self.manifest = manifest


class OpenRouterAuditCheckpointError(OpenRouterError):
    """The provider POST completed, but its durable checkpoint did not."""

    def __init__(
        self,
        message: str,
        *,
        event: dict[str, Any],
        result: ChatResult | None = None,
        manifest: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.event = event
        self.result = result
        event_manifest = event.get("manifest")
        self.manifest = dict(
            manifest
            if isinstance(manifest, dict)
            else event_manifest
            if isinstance(event_manifest, dict)
            else {}
        )


class OpenRouterPolicyError(OpenRouterError):
    """The model answered, but the response violated its web-access contract."""

    def __init__(self, message: str, *, result: ChatResult):
        super().__init__(message)
        self.result = result


@dataclass
class ImageResult:
    content: bytes
    extension: str
    media_type: str
    usage: dict[str, Any]


AuditCheckpoint = Callable[
    [dict[str, Any]],
    Awaitable[None] | None,
]


def _structured_delta_audit_enabled(
    checkpoint: AuditCheckpoint | None,
) -> bool:
    """Negotiate compact structured events without changing callback API.

    Plain callables keep receiving the original self-contained snapshot events.
    The durable structured store advertises delta-v2 explicitly on its callback,
    allowing the producer to avoid constructing a cumulative payload before the
    callback is invoked.  Unknown capability values fail closed rather than
    silently changing the audit wire format.
    """

    if checkpoint is None:
        return False
    capability = getattr(
        checkpoint,
        STRUCTURED_AUDIT_EVENT_VERSION_ATTR,
        None,
    )
    if capability is None:
        return False
    if capability != STRUCTURED_AUDIT_DELTA_CAPABILITY:
        raise OpenRouterError(
            "Unsupported structured audit event capability: "
            f"{capability!r}"
        )
    return True


def panel_models() -> tuple[PanelModel, ...]:
    return (
        PanelModel(
            "openai",
            "ChatGPT",
            settings.OPENROUTER_OPENAI_MODEL,
            settings.OPENROUTER_OPENAI_MODEL,
        ),
        PanelModel(
            "gemini",
            "Gemini",
            settings.OPENROUTER_GEMINI_MODEL,
            settings.OPENROUTER_GEMINI_MODEL,
        ),
        PanelModel(
            "perplexity",
            "Perplexity",
            settings.OPENROUTER_PERPLEXITY_MODEL,
            None,
        ),
        PanelModel(
            "deepseek",
            "DeepSeek",
            settings.OPENROUTER_DEEPSEEK_MODEL,
            settings.OPENROUTER_DEEPSEEK_MODEL,
        ),
        PanelModel(
            "claude",
            "Claude",
            settings.OPENROUTER_CLAUDE_MODEL,
            settings.OPENROUTER_CLAUDE_MODEL,
        ),
    )


def _headers() -> dict[str, str]:
    if not settings.OPENROUTER_API_KEY:
        raise OpenRouterError("OpenRouter API key is not configured")
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://aiv.rw.plus",
        "X-Title": "RW+ AI Visibility",
    }


def clear_model_envelope_cache() -> None:
    """Test/operations hook; capability drift must never require a restart."""

    _MODEL_ENVELOPE_CACHE.clear()


async def model_output_envelope(model: str) -> dict[str, Any]:
    """Resolve the current model maximum without inventing a local fallback.

    OpenRouter's model metadata is the authority.  If it is unavailable, the
    request omits a completion limit and lets the selected provider apply its
    native envelope.  A guessed number would only reintroduce the hidden cap
    this harness is meant to remove.
    """

    now = _monotonic()
    cached = _MODEL_ENVELOPE_CACHE.get(model)
    if cached is not None and cached[0] > now:
        return dict(cached[1])
    envelope: dict[str, Any] = {
        "version": OUTPUT_ENVELOPE_VERSION,
        "policy": OutputTokenPolicy.MODEL_MAX.value,
        "requested_model": model,
        "resolution": "provider_default_unresolved",
        "context_length": None,
        "max_completion_tokens": None,
    }
    try:
        timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            response = await client.get(
                f"{MODEL_URL}/{quote(model, safe='/')}",
                headers=_headers(),
            )
        if response.status_code >= 400:
            raise OpenRouterError(_response_error(response))
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            data = body if isinstance(body, dict) else {}
        top_provider = data.get("top_provider")
        if not isinstance(top_provider, dict):
            top_provider = {}
        maximum = top_provider.get("max_completion_tokens")
        context_length = data.get("context_length")
        if (
            isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and maximum > 0
        ):
            envelope["max_completion_tokens"] = maximum
            envelope["resolution"] = "openrouter_model_metadata"
        if (
            isinstance(context_length, int)
            and not isinstance(context_length, bool)
            and context_length > 0
        ):
            envelope["context_length"] = context_length
    except (AttributeError, TypeError, httpx.HTTPError, ValueError, OpenRouterError):
        # Absence of metadata is not a reason to restore an arbitrary local
        # ceiling.  The unresolved state is persisted with the response.
        pass
    _MODEL_ENVELOPE_CACHE[model] = (
        now + _MODEL_ENVELOPE_TTL_SECONDS,
        dict(envelope),
    )
    return envelope


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return ""


def _citations(message: dict[str, Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for annotation in message.get("annotations") or []:
        if not isinstance(annotation, dict):
            continue
        source = annotation.get("url_citation") or annotation
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(
            {
                "url": url,
                "title": str(source.get("title") or "").strip(),
                "content": str(source.get("content") or "").strip(),
            }
        )
    return output


def _annotations(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the complete JSON-safe response annotations for provenance."""

    output: list[dict[str, Any]] = []
    raw_annotations = message.get("annotations")
    if not isinstance(raw_annotations, list):
        return output
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            continue
        item: dict[str, Any] = {
            "type": str(annotation.get("type") or "").strip(),
        }
        source = annotation.get("url_citation")
        if isinstance(source, dict):
            citation: dict[str, Any] = {
                "url": str(source.get("url") or "").strip(),
                "title": str(source.get("title") or "").strip(),
                "content": str(source.get("content") or "").strip(),
            }
            for key in ("start_index", "end_index"):
                value = source.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    citation[key] = value
            item["url_citation"] = citation
        output.append(item)
    return output


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize the exact JSON representation used on the provider wire.

    A request receipt is useful only when its digest identifies the bytes that
    were actually sent.  Keeping this encoder shared by preflight, audit, and
    transport prevents insertion order or httpx's default JSON formatting from
    producing a different body under the same logical request.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _contains_online_variant(model: str) -> bool:
    return any(
        part.strip().casefold() == "online"
        for part in str(model).split(":")[1:]
    )


def web_request_policy(
    *,
    model: str,
    policy: WebSearchPolicy | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return request fields plus the exact policy contract persisted with usage."""

    try:
        selected = WebSearchPolicy(policy)
    except ValueError as exc:
        raise OpenRouterError(f"Unsupported web access policy: {policy}") from exc
    if _contains_online_variant(model):
        raise OpenRouterError(
            "The deprecated :online model variant is not allowed; "
            "web access must be declared by policy"
        )

    fields: dict[str, Any] = {
        # Account-level defaults may otherwise turn the deprecated web plugin
        # back on for a request that is supposed to be memory-only.
        "plugins": [{"id": "web", "enabled": False}],
    }
    mechanism = "none"
    if selected is WebSearchPolicy.REQUIRED:
        mechanism = "openrouter_server_tool"
        fields.update(
            {
                "tools": [
                    {
                        "type": "openrouter:web_search",
                        "parameters": {
                            "engine": "auto",
                        },
                    }
                ],
                # "auto", а не "required": принудительный tool_choice держится
                # на каждом витке серверного цикла инструментов, поэтому модель
                # не получает хода на текст — Anthropic закрывает обмен с
                # native_finish_reason="pause_turn" и пустым content. Факт
                # поиска гарантирует пост-проверка attest_web_response(),
                # а не принуждение на входе.
                "tool_choice": "auto",
            }
        )
    elif selected is WebSearchPolicy.FORBIDDEN:
        fields["tool_choice"] = "none"
    else:
        mechanism = "perplexity_native_search"
        if not str(model).casefold().startswith("perplexity/sonar"):
            raise OpenRouterError(
                "Native-required web policy is reserved for Perplexity Sonar"
            )

    contract = {
        "version": WEB_ATTESTATION_VERSION,
        "policy": selected.value,
        "mechanism": mechanism,
        "model": model,
        "request_fields": fields,
        "requires_url_citation": selected is not WebSearchPolicy.FORBIDDEN,
    }
    return fields, {
        **contract,
        "sha256": _stable_sha256(contract),
    }


def _usage_count(usage: dict[str, Any], key: str) -> int | None:
    server_usage = usage.get("server_tool_use")
    server_usage_details = usage.get("server_tool_use_details")
    candidates = [
        server_usage.get(key) if isinstance(server_usage, dict) else None,
        (
            server_usage_details.get(key)
            if isinstance(server_usage_details, dict)
            else None
        ),
        usage.get(key),
    ]
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, float) and value >= 0 and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _router_retrieval_signals(metadata: dict[str, Any]) -> list[str]:
    output: list[str] = []
    pipeline = metadata.get("pipeline")
    if not isinstance(pipeline, list):
        return output
    for stage in pipeline:
        if not isinstance(stage, dict):
            continue
        stage_type = str(stage.get("type") or "").strip().casefold()
        stage_name = str(stage.get("name") or "").strip().casefold()
        serialized = json.dumps(stage, ensure_ascii=False).casefold()
        if stage_type == "plugin" and stage_name in {
            "web",
            "web-search",
            "web_search",
        }:
            output.append("router_plugin_web_search")
        if stage_type == "server_tools" and any(
            token in serialized
            for token in (
                "openrouter:web_search",
                "web_search",
                "web-search",
                "openrouter:web_fetch",
                "web_fetch",
                "web-fetch",
            )
        ):
            output.append("router_server_retrieval")
    return sorted(set(output))


def attest_web_response(
    *,
    requested_model: str,
    response_model: str,
    policy: WebSearchPolicy | str,
    usage: dict[str, Any],
    annotations: list[dict[str, Any]],
    citations: list[dict[str, str]],
    router_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Verify actual retrieval behavior from OpenRouter response evidence."""

    selected = WebSearchPolicy(policy)
    web_search_requests = _usage_count(usage, "web_search_requests")
    web_fetch_requests = _usage_count(usage, "web_fetch_requests")
    router_signals = _router_retrieval_signals(router_metadata)
    citation_annotations = sum(
        bool(
            isinstance(item, dict)
            and isinstance(item.get("url_citation"), dict)
            and str((item.get("url_citation") or {}).get("url") or "").strip()
        )
        for item in annotations
    )
    violations: list[str] = []
    evidence: list[str] = []
    mechanism = "none"

    if selected is WebSearchPolicy.REQUIRED:
        mechanism = "openrouter_server_tool"
        if "router_plugin_web_search" in router_signals:
            violations.append("deprecated_web_plugin_observed")
        if not web_search_requests:
            violations.append("web_search_requests_not_confirmed")
        else:
            evidence.append("usage.server_tool_use.web_search_requests")
        if not citations or citation_annotations < 1:
            violations.append("url_citation_not_confirmed")
        else:
            evidence.append("message.annotations.url_citation")
    elif selected is WebSearchPolicy.NATIVE_REQUIRED:
        mechanism = "perplexity_native_search"
        if router_signals:
            violations.append("non_native_retrieval_stage_observed")
        requested_native = requested_model.casefold().startswith(
            "perplexity/sonar"
        )
        response_native = response_model.casefold().startswith(
            "perplexity/sonar"
        )
        if (
            not requested_native
            or not response_native
            or _contains_online_variant(response_model)
        ):
            violations.append("perplexity_native_model_not_confirmed")
        else:
            evidence.append("perplexity_sonar_response_model")
        # Sonar search is intrinsic rather than an added OpenRouter server
        # tool. Standardized URL citation annotations are its durable proof.
        if not citations or citation_annotations < 1:
            violations.append("perplexity_native_citation_not_confirmed")
        else:
            evidence.append("message.annotations.url_citation")
        if web_search_requests and web_search_requests > 0:
            evidence.append("usage.server_tool_use.web_search_requests")
    else:
        mechanism = "none"
        if web_search_requests and web_search_requests > 0:
            violations.append("web_search_used_while_forbidden")
        if web_fetch_requests and web_fetch_requests > 0:
            violations.append("web_fetch_used_while_forbidden")
        if citations or citation_annotations:
            violations.append("retrieval_annotations_while_forbidden")
        if router_signals:
            violations.append("router_retrieval_stage_while_forbidden")
        if _contains_online_variant(response_model):
            violations.append("online_variant_while_forbidden")
        if not violations:
            evidence.append("no_retrieval_telemetry")

    verified = not violations
    return {
        "version": WEB_ATTESTATION_VERSION,
        "policy": selected.value,
        "mechanism": mechanism,
        "state": "verified" if verified else "violated",
        "metric_eligible": verified,
        "web_search_requests": web_search_requests,
        "web_fetch_requests": web_fetch_requests,
        "citations_count": len(citations),
        "annotations_count": len(annotations),
        "citation_annotations_count": citation_annotations,
        "requested_model": requested_model,
        "response_model": response_model,
        "router_signals": router_signals,
        "evidence": evidence,
        "violations": violations,
    }


def _parse_strict_json_document(text: str) -> dict[str, Any] | list[Any]:
    """Parse exactly one JSON document without salvaging a valid substring."""

    def reject_non_json_constant(value: str) -> None:
        raise ValueError(f"non-JSON numeric constant: {value}")

    try:
        parsed = json.loads(
            text.strip(),
            parse_constant=reject_non_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise OpenRouterError(
            "Structured response is not exactly one complete JSON document"
        ) from exc
    if not isinstance(parsed, (dict, list)):
        raise OpenRouterError("Structured response is a JSON scalar")
    return parsed


# Часть upstream-эндпоинтов выполняет веб-поиск, но не отдаёт наружу
# url_citation-аннотации OpenRouter. Замер 2026-08-21 на anthropic/claude-opus-5:
# Anthropic и Claude Platform on AWS — 0 аннотаций при web_search_requests=3,
# Amazon Bedrock и Azure — 11-12 аннотаций. Аттестация такой ответ принять не
# может, поэтому эндпоинт запоминается на время жизни процесса, и следующие
# запросы к этой модели маршрутизируются мимо него, а не сжигают на нём попытки.
_NON_CITING_PROVIDERS: dict[str, set[str]] = {}


def _non_citing_providers(model: str) -> list[str]:
    return sorted(_NON_CITING_PROVIDERS.get(model) or ())


def _remember_non_citing_provider(model: str, provider: str) -> None:
    if provider:
        _NON_CITING_PROVIDERS.setdefault(model, set()).add(provider)


def _citation_routing_failure(attestation: dict[str, Any]) -> bool:
    """Поиск состоялся, но этот эндпоинт не отдал ни одной URL-цитаты."""

    return set(attestation.get("violations") or []) == {
        "url_citation_not_confirmed"
    } and int(attestation.get("web_search_requests") or 0) > 0


def _empty_response_error(
    choice: dict[str, Any],
    message: dict[str, Any],
) -> str:
    """Name the stop reason: bare "empty response" hides why it happened."""

    return (
        "Model returned an empty response "
        f"(finish_reason={choice.get('finish_reason') or '?'}, "
        f"native_finish_reason={choice.get('native_finish_reason') or '?'}, "
        f"tool_calls={len(message.get('tool_calls') or [])})"
    )


def _normalized_finish_reason(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _output_limit_reached(choice: dict[str, Any]) -> bool:
    reasons = {
        _normalized_finish_reason(choice.get("finish_reason")),
        _normalized_finish_reason(choice.get("native_finish_reason")),
    }
    return any(
        reason in {"length", "max_tokens", "max_output_tokens", "token_limit"}
        or "max_token" in reason
        for reason in reasons
        if reason
    )


def _transport_metadata(
    *,
    body: dict[str, Any],
    choice: dict[str, Any],
    requested_model: str,
    response_model: str,
    attempt: int,
) -> dict[str, Any]:
    """Persist transport completion separately from any semantic verdict."""

    output_limited = _output_limit_reached(choice)
    finish_reason = _normalized_finish_reason(choice.get("finish_reason"))
    native_finish_reason = _normalized_finish_reason(
        choice.get("native_finish_reason")
    )
    incomplete_native_reasons = {
        "content_filter",
        "error",
        "pause_turn",
        "safety",
        "tool_calls",
        "tool_use",
    }
    output_complete = bool(
        not output_limited
        and finish_reason == "stop"
        and native_finish_reason not in incomplete_native_reasons
    )
    if output_limited:
        incomplete_reason = "output_limit"
    elif not finish_reason:
        incomplete_reason = "missing_finish_reason"
    elif finish_reason != "stop":
        incomplete_reason = f"finish_reason:{finish_reason}"
    elif native_finish_reason in incomplete_native_reasons:
        incomplete_reason = f"native_finish_reason:{native_finish_reason}"
    else:
        incomplete_reason = None
    return {
        "version": TRANSPORT_METADATA_VERSION,
        "status": "succeeded",
        "http_status": 200,
        "attempt": attempt,
        "requested_model": requested_model,
        "response_model": response_model,
        "provider": str(body.get("provider") or "").strip(),
        "response_id": str(body.get("id") or "").strip()[:200],
        "finish_reason": str(choice.get("finish_reason") or "").strip(),
        "native_finish_reason": str(
            choice.get("native_finish_reason") or ""
        ).strip(),
        "output_complete": output_complete,
        "output_limited": output_limited,
        "output_incomplete_reason": incomplete_reason,
    }


def _policy_retry_reminder(
    policy: WebSearchPolicy,
    violations: list[str],
) -> dict[str, str]:
    """Explain the rejected contract so the retry is not a blind repeat."""

    if policy is WebSearchPolicy.FORBIDDEN:
        instruction = (
            "Предыдущий ответ отклонён: в нём остались следы веб-поиска, который "
            "в этом запросе запрещён. Ответь только по собственным знаниям — без "
            "инструментов, URL и ссылок на источники."
        )
    else:
        instruction = (
            "Предыдущий ответ отклонён: обязательный веб-поиск не подтверждён. "
            "Обязательно выполни поиск в вебе и обопри каждый внешний вывод на "
            "конкретный URL из результатов поиска."
        )
    detail = ", ".join(violations) or "unknown"
    return {
        "role": "user",
        "content": f"{instruction}\n(нарушения контракта: {detail})",
    }


def _response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        else:
            message = str(error or "")
    except Exception:
        message = ""
    return f"OpenRouter returned HTTP {response.status_code}: {message[:240]}".rstrip()


def _response_snapshot(response: Any) -> dict[str, Any]:
    """Return the complete JSON/text body of one physical provider response."""

    snapshot: dict[str, Any] = {
        "http_status": getattr(response, "status_code", None),
    }
    try:
        snapshot["body_json"] = response.json()
    except Exception as exc:
        snapshot["body_json_error"] = f"{type(exc).__name__}: {exc}"
        try:
            snapshot["body_text"] = str(response.text)
        except Exception:
            snapshot["body_text"] = None
    return snapshot


def _request_envelope_estimate(payload: dict[str, Any]) -> dict[str, Any]:
    """Conservatively bound input tokens from the exact serialized request.

    No universal tokenizer exists for an OpenRouter route that may select
    several upstream providers.  UTF-8 bytes are nevertheless a strict upper
    bound for byte-level/BPE token count (a token represents at least one
    byte).  Adding a small fixed protocol allowance makes the calculation safe
    without inventing a small completion ceiling.
    """

    serialized = _canonical_json_bytes(payload)
    return {
        "version": REQUEST_ENVELOPE_ESTIMATE_VERSION,
        "serialized_request_utf8_bytes": len(serialized),
        "protocol_token_upper_bound": _REQUEST_WRAPPER_TOKEN_UPPER_BOUND,
        "input_token_upper_bound": (
            len(serialized) + _REQUEST_WRAPPER_TOKEN_UPPER_BOUND
        ),
        "request_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _apply_model_output_headroom(
    *,
    payload: dict[str, Any],
    output_envelope: dict[str, Any],
    requested_max: int | None,
) -> tuple[int | None, dict[str, Any]]:
    """Fit physical output to provider max and remaining context headroom."""

    envelope = dict(output_envelope)
    context_length = envelope.get("context_length")
    candidates: list[int] = []
    if isinstance(requested_max, int) and not isinstance(requested_max, bool):
        candidates.append(requested_max)

    # Include the actual max field in the exact serialized request estimate.
    # Two iterations are enough for the only circular input (the digit count of
    # max_completion_tokens) to stabilize; a final estimate is persisted.
    provisional = requested_max
    estimate: dict[str, Any]
    for _ in range(3):
        candidate_payload = dict(payload)
        if isinstance(provisional, int) and provisional > 0:
            candidate_payload["max_completion_tokens"] = provisional
        estimate = _request_envelope_estimate(candidate_payload)
        if (
            isinstance(context_length, int)
            and not isinstance(context_length, bool)
            and context_length > 0
        ):
            headroom = context_length - int(estimate["input_token_upper_bound"])
            if headroom <= 0:
                raise OpenRouterContextHeadroomError(
                    "OpenRouter request has no completion headroom after the "
                    "exact serialized-request upper-bound estimate "
                    f"(context_length={context_length}, "
                    "input_token_upper_bound="
                    f"{estimate['input_token_upper_bound']})"
                )
            bounded = min([*candidates, headroom]) if candidates else headroom
        else:
            headroom = None
            bounded = min(candidates) if candidates else None
        if bounded == provisional:
            provisional = bounded
            break
        provisional = bounded

    final_payload = dict(payload)
    if isinstance(provisional, int) and provisional > 0:
        final_payload["max_completion_tokens"] = provisional
    estimate = _request_envelope_estimate(final_payload)
    headroom = (
        context_length - int(estimate["input_token_upper_bound"])
        if isinstance(context_length, int)
        and not isinstance(context_length, bool)
        and context_length > 0
        else None
    )
    envelope["request_estimate"] = estimate
    envelope["context_headroom_tokens"] = headroom
    envelope["effective_max_completion_tokens"] = provisional
    return provisional, envelope


async def _emit_audit_checkpoint(
    checkpoint: AuditCheckpoint | None,
    event: dict[str, Any],
    *,
    result: ChatResult | None = None,
) -> None:
    """Shield one append-only audit write from caller cancellation."""

    if checkpoint is None:
        return

    async def invoke() -> None:
        outcome = checkpoint(event)
        if inspect.isawaitable(outcome):
            await outcome

    task = asyncio.create_task(invoke())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        # The provider POST may already be billable.  Complete the checkpoint
        # before propagating cancellation; a second cancellation still wins.
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise OpenRouterAuditCheckpointError(
                f"OpenRouter audit checkpoint failed during cancellation: {exc}",
                event=event,
                result=result,
            ) from exc
        # The write succeeded despite caller cancellation.  Tell the transport
        # boundary not to emit a second, false ``cancelled`` provider event and
        # retain the accepted/rejected response for the structured caller.
        cancellation.audit_checkpoint_persisted = True
        cancellation.audit_event = event
        cancellation.result = result
        raise cancellation
    except Exception as exc:
        raise OpenRouterAuditCheckpointError(
            f"OpenRouter audit checkpoint failed: {exc}",
            event=event,
            result=result,
        ) from exc


def _physical_post_event(
    *,
    logical_call_id: str,
    model: str,
    attempt: int,
    status: str,
    request_payload: dict[str, Any],
    request_body: bytes,
    response_snapshot: dict[str, Any] | None,
    result: ChatResult | None,
    error: BaseException | None,
    audit_context: dict[str, Any] | None,
    delta_audit: bool = False,
) -> dict[str, Any]:
    context = dict(audit_context or {})
    common = {
        "version": (
            PHYSICAL_POST_AUDIT_DELTA_VERSION
            if delta_audit
            else PHYSICAL_POST_AUDIT_VERSION
        ),
        "event_id": uuid.uuid4().hex,
        "event_kind": "provider_post",
        "logical_call_id": logical_call_id,
        "document_id": context.get("document_id"),
        "sequence": context.get("sequence"),
        "attempt": attempt,
        "status": status,
        "model": model,
        "request_payload": request_payload,
        "request_sha256": hashlib.sha256(request_body).hexdigest(),
        "request_body_utf8_bytes": len(request_body),
        "request_body_encoding": "canonical-json-utf8-v1",
        "response": dict(response_snapshot or {}),
        "raw_text": result.text if result is not None else None,
        "citations": (
            [dict(item) for item in result.citations]
            if result is not None
            else []
        ),
        "annotations": (
            [dict(item) for item in result.annotations]
            if result is not None
            else []
        ),
        "request_policy": (
            dict(result.request_policy) if result is not None else {}
        ),
        "web_attestation": (
            dict(result.web_attestation) if result is not None else {}
        ),
        "router_metadata": (
            dict(result.router_metadata) if result is not None else {}
        ),
        "usage": dict(result.usage) if result is not None else {},
        "transport": dict(result.transport) if result is not None else {},
        "resume_contract": context.get("resume_contract"),
        "error": (
            {
                "type": type(error).__name__,
                "message": str(error),
            }
            if error is not None
            else None
        ),
    }
    if delta_audit:
        predecessor = context.get("predecessor")
        if not isinstance(predecessor, dict):
            raise OpenRouterError(
                "Structured delta audit has no compact predecessor"
            )
        common["predecessor"] = dict(predecessor)
        return common
    return {
        **common,
        "partial_text": context.get("partial_text", ""),
        "manifest": context.get("manifest"),
        "aggregate_usage": context.get("aggregate_usage", {}),
        "call_records": context.get("call_records", []),
    }


def restore_completed_chat_provider_event(
    provider_event: dict[str, Any],
    *,
    model: str,
    messages: list[dict[str, Any]],
    web_policy: WebSearchPolicy,
    temperature: float,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "aiv_response",
    reasoning_effort: str | None = None,
) -> ChatResult:
    """Restore one accepted atomic chat result without another paid POST.

    The event is validated against the expected logical request and the exact
    canonical request bytes.  User-facing evidence is reconstructed from the
    provider response body rather than trusted from a convenient cached field,
    so a torn or selectively edited receipt fails closed.
    """

    if not isinstance(provider_event, dict) or (
        provider_event.get("version") != PHYSICAL_POST_AUDIT_VERSION
        or provider_event.get("event_kind") != "provider_post"
        or provider_event.get("status") != "accepted"
    ):
        raise OpenRouterError("Atomic provider receipt is not accepted v1")
    if str(provider_event.get("model") or "") != model:
        raise OpenRouterError("Atomic provider receipt model mismatch")
    payload = provider_event.get("request_payload")
    if not isinstance(payload, dict):
        raise OpenRouterError("Atomic provider receipt has no request payload")
    body_bytes = _canonical_json_bytes(payload)
    if provider_event.get("request_sha256") != hashlib.sha256(
        body_bytes
    ).hexdigest():
        raise OpenRouterError("Atomic provider receipt request digest mismatch")
    if provider_event.get("request_body_encoding") != "canonical-json-utf8-v1":
        raise OpenRouterError("Atomic provider receipt request encoding mismatch")
    if provider_event.get("request_body_utf8_bytes") != len(body_bytes):
        raise OpenRouterError("Atomic provider receipt request size mismatch")
    if payload.get("model") != model or payload.get("messages") != messages:
        raise OpenRouterError("Atomic provider receipt request input mismatch")
    if payload.get("temperature") != temperature:
        raise OpenRouterError("Atomic provider receipt temperature mismatch")
    normalized_reasoning = (
        str(reasoning_effort).strip()
        if reasoning_effort is not None
        else None
    )
    if normalized_reasoning == "":
        normalized_reasoning = None
    expected_reasoning = (
        {"effort": normalized_reasoning, "exclude": True}
        if normalized_reasoning is not None
        else None
    )
    if expected_reasoning is None:
        if "reasoning" in payload:
            raise OpenRouterError(
                "Atomic provider receipt has unexpected reasoning policy"
            )
    elif payload.get("reasoning") != expected_reasoning:
        raise OpenRouterError("Atomic provider receipt reasoning mismatch")
    expected_response_format = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        }
        if response_schema is not None
        else None
    )
    if expected_response_format is None:
        if "response_format" in payload:
            raise OpenRouterError(
                "Atomic provider receipt has unexpected response schema"
            )
    elif payload.get("response_format") != expected_response_format:
        raise OpenRouterError("Atomic provider receipt schema mismatch")
    policy_fields, request_policy = web_request_policy(
        model=model,
        policy=web_policy,
    )
    for key, expected in policy_fields.items():
        if payload.get(key) != expected:
            raise OpenRouterError(
                "Atomic provider receipt web-policy payload mismatch"
            )

    snapshot = provider_event.get("response")
    response_body = (
        snapshot.get("body_json") if isinstance(snapshot, dict) else None
    )
    if not isinstance(response_body, dict):
        raise OpenRouterError("Atomic provider receipt has no JSON response")
    choices = response_body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise OpenRouterError("Atomic provider receipt has no response choice")
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise OpenRouterError("Atomic provider receipt message is invalid")
    text = _message_text(message)
    citations = _citations(message)
    annotations = _annotations(message)
    if not text or provider_event.get("raw_text") != text:
        raise OpenRouterError("Atomic provider receipt response text mismatch")
    if provider_event.get("citations") != citations or (
        provider_event.get("annotations") != annotations
    ):
        raise OpenRouterError("Atomic provider receipt evidence mismatch")

    raw_usage = dict(response_body.get("usage") or {})
    router_metadata = dict(response_body.get("openrouter_metadata") or {})
    response_model = str(response_body.get("model") or "").strip()
    attempt = provider_event.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise OpenRouterError("Atomic provider receipt attempt is invalid")
    transport = _transport_metadata(
        body=response_body,
        choice=choice,
        requested_model=model,
        response_model=response_model,
        attempt=attempt,
    )
    attestation = attest_web_response(
        requested_model=model,
        response_model=response_model,
        policy=web_policy,
        usage=raw_usage,
        annotations=annotations,
        citations=citations,
        router_metadata=router_metadata,
    )
    stored_usage = provider_event.get("usage")
    if not isinstance(stored_usage, dict):
        raise OpenRouterError("Atomic provider receipt usage is invalid")
    for key, value in raw_usage.items():
        if stored_usage.get(key) != value:
            raise OpenRouterError("Atomic provider receipt usage mismatch")
    expected_enriched = {
        "_aiv_request_policy": request_policy,
        "_aiv_response_annotations": annotations,
        "_aiv_router_metadata": router_metadata,
        "_aiv_web_attestation": attestation,
        "_aiv_transport": transport,
    }
    for key, value in expected_enriched.items():
        if stored_usage.get(key) != value:
            raise OpenRouterError(
                f"Atomic provider receipt {key} mismatch"
            )
    if provider_event.get("request_policy") != request_policy or (
        provider_event.get("web_attestation") != attestation
        or provider_event.get("router_metadata") != router_metadata
        or provider_event.get("transport") != transport
    ):
        raise OpenRouterError("Atomic provider receipt result metadata mismatch")
    parsed: dict[str, Any] | list[Any] | None = None
    if response_schema is not None:
        try:
            parsed = _parse_strict_json_document(text)
            Draft202012Validator(response_schema).validate(parsed)
        except (OpenRouterError, ValidationError) as exc:
            raise OpenRouterError(
                f"Atomic provider receipt structured output is invalid: {exc}"
            ) from exc
    return ChatResult(
        text=text,
        parsed=parsed,
        citations=citations,
        usage=dict(stored_usage),
        annotations=annotations,
        request_policy=request_policy,
        web_attestation=attestation,
        router_metadata=router_metadata,
        transport=transport,
    )


def _aggregate_billable_usage(
    base_usage: dict[str, Any],
    call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum provider counters without hiding the original per-POST usage."""

    aggregate = dict(base_usage)
    numeric_keys: set[str] = set()
    for record in call_records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if (
                isinstance(key, str)
                and (
                    key.endswith("_tokens")
                    or key in {"cost", "total_cost", "credits"}
                )
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                numeric_keys.add(key)
    for key in sorted(numeric_keys):
        values = [
            usage[key]
            for record in call_records
            if isinstance((usage := record.get("usage")), dict)
            and isinstance(usage.get(key), (int, float))
            and not isinstance(usage.get(key), bool)
        ]
        if values:
            total = sum(values)
            aggregate[key] = (
                int(total)
                if all(isinstance(value, int) for value in values)
                else float(total)
            )
    return aggregate


def _chat_call_record(
    result: ChatResult,
    *,
    attempt: int,
    status: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "status": status,
        "raw_text": result.text,
        "text_sha256": text_sha256(result.text),
        "text_chars": len(result.text),
        "text_utf8_bytes": len(result.text.encode("utf-8")),
        "usage": dict(result.usage),
        "transport": dict(result.transport),
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }


def _attach_chat_call_audit(
    result: ChatResult,
    call_records: list[dict[str, Any]],
) -> ChatResult:
    usage = _aggregate_billable_usage(result.usage, call_records)
    usage["_aiv_call_attempts"] = [dict(record) for record in call_records]
    return replace(result, usage=usage)


async def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "aiv_result",
    web_policy: WebSearchPolicy | str = WebSearchPolicy.FORBIDDEN,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    output_token_policy: OutputTokenPolicy | str = (
        OutputTokenPolicy.PROVIDER_DEFAULT
    ),
    temperature: float = 0.2,
    accept_output_limited: bool = False,
    retry_response_contract_errors: bool = False,
    retry_transport_errors: bool = False,
    # Backwards-compatible caller override. Application stages no longer use
    # it; new code should use max_completion_tokens or MODEL_MAX.
    max_tokens: int | None = None,
    audit_checkpoint: AuditCheckpoint | None = None,
    audit_context: dict[str, Any] | None = None,
) -> ChatResult:
    try:
        selected_output_policy = OutputTokenPolicy(output_token_policy)
    except ValueError as exc:
        raise OpenRouterError(
            f"Unsupported output token policy: {output_token_policy}"
        ) from exc
    if max_tokens is not None:
        if max_completion_tokens is not None:
            raise OpenRouterError(
                "Use either max_tokens or max_completion_tokens, not both"
            )
        max_completion_tokens = max_tokens
    if max_completion_tokens is not None and (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens <= 0
    ):
        raise OpenRouterError("max_completion_tokens must be a positive integer")
    if (
        max_completion_tokens is not None
        and selected_output_policy is OutputTokenPolicy.MODEL_MAX
    ):
        raise OpenRouterError(
            "Explicit max_completion_tokens conflicts with model_max_available"
        )
    if selected_output_policy is OutputTokenPolicy.MODEL_MAX:
        output_envelope_base = await model_output_envelope(model)
        requested_physical_max = output_envelope_base.get(
            "max_completion_tokens"
        )
    else:
        metadata = (
            await model_output_envelope(model)
            if max_completion_tokens is not None
            else {}
        )
        provider_max = metadata.get("max_completion_tokens")
        explicit_candidates = [max_completion_tokens]
        if isinstance(provider_max, int) and not isinstance(provider_max, bool):
            explicit_candidates.append(provider_max)
        requested_physical_max = (
            min(explicit_candidates)
            if max_completion_tokens is not None
            else None
        )
        output_envelope_base = {
            "version": OUTPUT_ENVELOPE_VERSION,
            "policy": selected_output_policy.value,
            "requested_model": model,
            "resolution": (
                "caller_explicit"
                if max_completion_tokens is not None
                else "provider_default"
            ),
            "context_length": metadata.get("context_length"),
            "max_completion_tokens": max_completion_tokens,
            "provider_max_completion_tokens": provider_max,
        }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    policy_fields, request_policy = web_request_policy(
        model=model,
        policy=web_policy,
    )
    # web_request_policy() уже отвергла неизвестные значения понятной ошибкой.
    request_policy_enum = WebSearchPolicy(web_policy)
    payload.update(policy_fields)
    # Маршрутизация не входит в аудируемый контракт доступа к вебу и потому
    # не попадает в request_policy: контракт описывает, что требуется, а не
    # через какой эндпоинт это удалось подтвердить. Иначе смена провайдера
    # меняла бы policy hash и обесценивала бы уже собранные ячейки панели.
    if request_policy_enum is not WebSearchPolicy.FORBIDDEN:
        avoid = _non_citing_providers(model)
        if avoid:
            payload["provider"] = {"ignore": avoid}
    if reasoning_effort:
        payload["reasoning"] = {
            "effort": reasoning_effort,
            "exclude": True,
        }
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        }

    logical_call_id = uuid.uuid4().hex
    delta_audit = _structured_delta_audit_enabled(audit_checkpoint)
    if delta_audit and (
        not isinstance(audit_context, dict)
        or not isinstance(audit_context.get("resume_contract"), dict)
        or not isinstance(audit_context.get("predecessor"), dict)
    ):
        raise OpenRouterError(
            "Structured delta audit callback requires compact audit context"
        )

    timeout = httpx.Timeout(
        connect=15.0,
        # A non-streaming provider may not yield the first response byte until
        # a long model-max generation has completed.  Use a deliberately large
        # *inactivity* deadline, independent from max tokens/output size, rather
        # than either a short hidden length cap or an infinite dead socket.
        read=max(60.0, float(settings.OPENROUTER_READ_TIMEOUT_SECONDS)),
        write=30.0,
        pool=15.0,
    )
    last_error: Exception | None = None
    call_records: list[dict[str, Any]] = []
    for attempt, delay in enumerate((0.0, 2.0, 7.0), start=1):
        if delay:
            await asyncio.sleep(delay)
        if isinstance(last_error, OpenRouterPolicyError):
            attestation = last_error.result.web_attestation
            # Когда цитат не отдал сам эндпоинт, модель ни при чём:
            # исправление — смена маршрута, уже применённая к payload.
            # Иначе идентичный повтор воспроизвёл бы то же нарушение,
            # поэтому причину отказа сообщаем модели явно. Список строится
            # от исходных messages, чтобы напоминания не накапливались.
            if not _citation_routing_failure(attestation):
                payload["messages"] = [
                    *messages,
                    _policy_retry_reminder(
                        request_policy_enum,
                        attestation.get("violations") or [],
                    ),
                ]
        resolved_max, output_envelope = _apply_model_output_headroom(
            payload=payload,
            output_envelope=output_envelope_base,
            requested_max=(
                requested_physical_max
                if isinstance(requested_physical_max, int)
                and not isinstance(requested_physical_max, bool)
                else None
            ),
        )
        attempt_payload = dict(payload)
        if isinstance(resolved_max, int) and resolved_max > 0:
            # OpenRouter deprecated max_tokens in favour of this field.  This
            # is the provider maximum or exact remaining context headroom,
            # whichever is smaller.
            attempt_payload["max_completion_tokens"] = resolved_max
        request_body = _canonical_json_bytes(attempt_payload)
        response: Any | None = None
        response_audit: dict[str, Any] | None = None
        checkpoint_emitted = False
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
                headers = _headers()
                headers["X-OpenRouter-Metadata"] = "enabled"
                headers["Content-Type"] = "application/json"
                response = await client.post(
                    CHAT_URL,
                    headers=headers,
                    content=request_body,
                )
            response_audit = _response_snapshot(response)
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                raise OpenRouterError(_response_error(response))
            if response.status_code >= 400:
                raise OpenRouterError(_response_error(response))
            body = response.json()
            if not isinstance(body, dict):
                raise OpenRouterError(
                    "OpenRouter returned a non-object response body"
                )
            choices = body.get("choices") or []
            if not choices:
                raise OpenRouterError("OpenRouter returned no choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise OpenRouterError("OpenRouter returned an invalid choice")
            message = choice.get("message") or {}
            if not isinstance(message, dict):
                raise OpenRouterError("OpenRouter returned an invalid message")
            text = _message_text(message)
            citations = _citations(message)
            annotations = _annotations(message)
            raw_usage = dict(body.get("usage") or {})
            router_metadata = dict(body.get("openrouter_metadata") or {})
            response_model = str(body.get("model") or "").strip()
            transport = _transport_metadata(
                body=body,
                choice=choice,
                requested_model=model,
                response_model=response_model,
                attempt=attempt,
            )
            attestation = attest_web_response(
                requested_model=model,
                response_model=response_model,
                policy=web_policy,
                usage=raw_usage,
                annotations=annotations,
                citations=citations,
                router_metadata=router_metadata,
            )
            usage = dict(raw_usage)
            usage["_aiv_request_policy"] = request_policy
            usage["_aiv_response_annotations"] = annotations
            usage["_aiv_router_metadata"] = router_metadata
            usage["_aiv_web_attestation"] = attestation
            usage["_aiv_transport"] = transport
            usage["_aiv_output_envelope"] = output_envelope
            result = ChatResult(
                text=text,
                parsed=None,
                citations=citations,
                usage=usage,
                annotations=annotations,
                request_policy=request_policy,
                web_attestation=attestation,
                router_metadata=router_metadata,
                transport=transport,
            )
            if transport["output_limited"] and not (
                accept_output_limited
                and response_schema is None
                and bool(text.strip())
                and attestation["metric_eligible"]
            ):
                raise OpenRouterOutputLimitError(
                    "OpenRouter response hit the output limit "
                    f"(finish_reason={transport['finish_reason'] or '?'}, "
                    "native_finish_reason="
                    f"{transport['native_finish_reason'] or '?'}, "
                    "max_completion_tokens="
                    f"{resolved_max if resolved_max is not None else 'provider_default'})",
                    result=result,
                )
            if not transport["output_complete"] and not (
                transport["output_limited"]
                and accept_output_limited
                and response_schema is None
                and bool(text.strip())
                and attestation["metric_eligible"]
            ):
                raise OpenRouterResponseContractError(
                    "OpenRouter response did not reach a complete final turn "
                    f"({transport['output_incomplete_reason'] or 'unknown'}; "
                    f"finish_reason={transport['finish_reason'] or '?'}, "
                    "native_finish_reason="
                    f"{transport['native_finish_reason'] or '?'})",
                    result=result,
                )
            if not text:
                raise OpenRouterResponseContractError(
                    _empty_response_error(choice, message),
                    result=result,
                )
            if not attestation["metric_eligible"]:
                if _citation_routing_failure(attestation):
                    _remember_non_citing_provider(
                        model,
                        str(body.get("provider") or "").strip(),
                    )
                    avoid = _non_citing_providers(model)
                    if avoid:
                        payload["provider"] = {"ignore": avoid}
                violations = ", ".join(attestation["violations"])
                raise OpenRouterPolicyError(
                    f"OpenRouter web policy attestation failed: {violations}",
                    result=result,
                )
            if response_schema is not None:
                try:
                    parsed = _parse_strict_json_document(text)
                    Draft202012Validator(response_schema).validate(parsed)
                    result = replace(result, parsed=parsed)
                except (OpenRouterError, ValidationError) as exc:
                    raise OpenRouterResponseContractError(
                        f"Structured response is unusable: {exc}",
                        result=result,
                    ) from exc
            call_records.append(
                _chat_call_record(
                    result,
                    attempt=attempt,
                    status="accepted",
                )
            )
            accepted_result = _attach_chat_call_audit(result, call_records)
            event = _physical_post_event(
                logical_call_id=logical_call_id,
                model=model,
                attempt=attempt,
                status="accepted",
                request_payload=attempt_payload,
                request_body=request_body,
                response_snapshot=response_audit,
                result=accepted_result,
                error=None,
                audit_context=audit_context,
                delta_audit=delta_audit,
            )
            await _emit_audit_checkpoint(
                audit_checkpoint,
                event,
                result=accepted_result,
            )
            checkpoint_emitted = True
            return accepted_result
        except OpenRouterAuditCheckpointError:
            raise
        except asyncio.CancelledError as exc:
            if bool(getattr(exc, "audit_checkpoint_persisted", False)):
                checkpoint_emitted = True
            if not checkpoint_emitted:
                event = _physical_post_event(
                    logical_call_id=logical_call_id,
                    model=model,
                    attempt=attempt,
                    status="cancelled",
                    request_payload=attempt_payload,
                    request_body=request_body,
                    response_snapshot=response_audit,
                    result=None,
                    error=exc,
                    audit_context=audit_context,
                    delta_audit=delta_audit,
                )
                try:
                    await _emit_audit_checkpoint(audit_checkpoint, event)
                except asyncio.CancelledError:
                    # The shielded inner write continues if the event loop is
                    # alive; cancellation semantics remain authoritative.
                    pass
                exc.audit_event = event
            raise
        except (httpx.HTTPError, ValueError, OpenRouterError) as exc:
            rejected_result = getattr(exc, "result", None)
            if isinstance(rejected_result, ChatResult):
                call_records.append(
                    _chat_call_record(
                        rejected_result,
                        attempt=attempt,
                        status="rejected",
                        error=exc,
                    )
                )
                exc.result = _attach_chat_call_audit(
                    rejected_result,
                    call_records,
                )
                rejected_result = exc.result
            status = (
                "rejected"
                if isinstance(rejected_result, ChatResult)
                else (
                    (
                        "http_error"
                        if int(getattr(response, "status_code", 0) or 0) >= 400
                        else "response_error"
                    )
                    if response is not None
                    else "transport_error"
                )
            )
            if not checkpoint_emitted:
                event = _physical_post_event(
                    logical_call_id=logical_call_id,
                    model=model,
                    attempt=attempt,
                    status=status,
                    request_payload=attempt_payload,
                    request_body=request_body,
                    response_snapshot=response_audit,
                    result=(
                        rejected_result
                        if isinstance(rejected_result, ChatResult)
                        else None
                    ),
                    error=exc,
                    audit_context=audit_context,
                    delta_audit=delta_audit,
                )
                await _emit_audit_checkpoint(
                    audit_checkpoint,
                    event,
                    result=(
                        rejected_result
                        if isinstance(rejected_result, ChatResult)
                        else None
                    ),
                )
                checkpoint_emitted = True
                exc.audit_event = event
            if isinstance(exc, OpenRouterOutputLimitError):
                raise
            if (
                isinstance(
                    exc,
                    (OpenRouterResponseContractError, OpenRouterPolicyError),
                )
                and not retry_response_contract_errors
            ):
                raise
            if not retry_transport_errors:
                raise
            last_error = exc
            if attempt >= 3:
                break
    if isinstance(
        last_error,
        (OpenRouterPolicyError, OpenRouterResponseContractError),
    ):
        raise last_error
    if last_error is not None:
        # Preserve the final physical POST event (and an attached response
        # result, when one exists).  Re-wrapping it as a fresh string-only
        # OpenRouterError used to sever the only in-process link to the full
        # HTTP/JSON audit body after the retry budget was exhausted.
        raise last_error
    raise OpenRouterError("OpenRouter request failed")


def _continuation_prompt(
    *,
    cursor: dict[str, Any],
    response_schema: dict[str, Any],
) -> dict[str, str]:
    """Build a data-delimited literal continuation request.

    A continuation is intentionally *not* a new strict-schema response: a JSON
    fragment is not a valid instance of the root schema.  Code supplies the
    sequence and exact overlap, joins the fragment, then validates the one
    assembled root document.
    """

    contract = {
        "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
        "document_id": cursor["document_id"],
        "sequence": cursor["next_sequence"],
        "previous_document_sha256": cursor["document_sha256"],
        "previous_document_chars": cursor["document_chars"],
        "required_literal_prefix": cursor["expected_overlap"],
        "required_literal_prefix_sha256": cursor[
            "expected_overlap_sha256"
        ],
        "json_prefix_state": cursor["json_prefix"],
        "root_json_schema": response_schema,
    }
    return {
        "role": "user",
        "content": (
            "Continue the SAME JSON document that was cut off by the provider's "
            "physical output limit. This is literal continuation, not a new "
            "answer. Return plain text only: no Markdown fence, explanation, "
            "labels, or fresh root object. The first characters of your response "
            "MUST equal required_literal_prefix exactly, character for character. "
            "After that prefix, emit only the characters that follow it in the "
            "same JSON document. Do not rewrite or summarize earlier content. "
            "Close the root JSON value only when the document is genuinely "
            "complete. The caller verifies the prefix, sequence, hashes, complete "
            "JSON parse, and Draft 2020-12 schema.\n\n"
            "CONTINUATION_CONTRACT_JSON:\n"
            + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
        ),
    }


def _continuation_messages(
    *,
    messages: list[dict[str, Any]],
    document_text: str,
    cursor: dict[str, Any],
    response_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Give the provider the full accepted prefix plus the exact join suffix.

    The overlap in ``cursor`` is a byte-for-byte boundary contract, not a
    substitute for semantic context.  Supplying only that suffix makes a model
    forget earlier array items, cross-field constraints, and entity identity.
    We therefore retain the entire accepted JSON prefix as the assistant turn.
    This is a single-context bridge across physical output turns, not an
    arbitrary-length document protocol: once the prefix exhausts the model's
    input context, ``chat`` fails visibly before the POST and the caller
    checkpoints the accepted ledger.  JSON is never summarized or silently
    discarded to force it through that boundary.
    """

    return [
        *messages,
        {"role": "assistant", "content": document_text},
        _continuation_prompt(
            cursor=cursor,
            response_schema=response_schema,
        ),
    ]


def _continuation_call_record(
    result: ChatResult,
    *,
    sequence: int,
) -> dict[str, Any]:
    usage = dict(result.usage)
    nested_attempts = usage.pop("_aiv_call_attempts", None)
    prior_transport_attempts: list[dict[str, Any]] = []
    if isinstance(nested_attempts, list):
        current_digest = text_sha256(result.text)
        removed_current = False
        for raw_record in reversed(nested_attempts):
            if not isinstance(raw_record, dict):
                continue
            record = dict(raw_record)
            nested_usage = record.get("usage")
            if isinstance(nested_usage, dict):
                nested_usage = dict(nested_usage)
                nested_usage.pop("_aiv_call_attempts", None)
                nested_usage.pop("_aiv_structured_continuation", None)
                record["usage"] = nested_usage
            # The accepted/rejected POST represented by ``result`` is already
            # the top-level raw_text below.  Retain only earlier transport or
            # contract attempts so every provider body appears exactly once.
            if (
                not removed_current
                and str(record.get("text_sha256") or "") == current_digest
                and str(record.get("raw_text") or "") == result.text
            ):
                removed_current = True
                continue
            prior_transport_attempts.append(record)
        prior_transport_attempts.reverse()
    return {
        "sequence": sequence,
        # Raw text is intentionally retained for every billable POST.  Hashes
        # alone are insufficient when a boundary is rejected and an operator
        # must reconstruct what the provider actually returned.
        "raw_text": result.text,
        "text_sha256": text_sha256(result.text),
        "text_chars": len(result.text),
        "text_utf8_bytes": len(result.text.encode("utf-8")),
        "transport": dict(result.transport),
        "request_policy": dict(result.request_policy),
        "web_attestation": dict(result.web_attestation),
        # This is the aggregate billing usage for the logical call.  Nested
        # chat-attempt ledgers are normalized above so large raw fragments are
        # not recursively duplicated in persisted JSON.
        "usage": usage,
        "prior_transport_attempts": prior_transport_attempts,
    }


def _aggregate_continuation_usage(
    initial_usage: dict[str, Any],
    call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate billable counters while retaining every per-call record."""
    return _aggregate_billable_usage(initial_usage, call_records)


def _structured_manifest_with_calls(
    ledger: StructuredContinuationLedger,
    *,
    call_records: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    manifest = ledger.manifest(complete=complete)
    manifest["calls"] = [dict(record) for record in call_records]
    manifest["accepted_document_text"] = ledger.text
    return manifest


def _structured_terminal_schema_failure_marker(
    ledger: StructuredContinuationLedger,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    """Seal the terminal state where complete JSON violates its schema.

    The marker is created only after independently proving both halves of the
    disposition: the accepted ledger is one complete JSON document, and that
    document fails the exact response schema.  Callers can therefore spend a
    bounded semantic attempt without inspecting exception prose or mistaking a
    partial/transport failure for a completed model decision.
    """

    parsed = _parse_strict_json_document(ledger.text)
    try:
        Draft202012Validator(response_schema).validate(parsed)
    except ValidationError as exc:
        body = {
            "version": STRUCTURED_TERMINAL_SEMANTIC_FAILURE_VERSION,
            "failure_kind": "complete_json_schema_violation",
            "terminal": True,
            "document_sha256": text_sha256(ledger.text),
            "document_chars": len(ledger.text),
            "document_utf8_bytes": len(ledger.text.encode("utf-8")),
            "response_schema_sha256": _stable_sha256(response_schema),
            "validation_path": list(exc.absolute_path),
            "schema_path": list(exc.absolute_schema_path),
            "validator": str(exc.validator or ""),
        }
    else:
        raise OpenRouterError(
            "Terminal structured semantic failure requires a schema violation"
        )
    return {**body, "marker_sha256": _stable_sha256(body)}


def _structured_terminal_non_json_failure_marker(
    ledger: StructuredContinuationLedger,
    response_schema: dict[str, Any],
    transport: dict[str, Any],
) -> dict[str, Any]:
    """Seal a complete provider turn whose assembled document is not JSON."""

    if not (
        isinstance(transport, dict)
        and transport.get("status") == "succeeded"
        and transport.get("http_status") == 200
        and transport.get("output_complete") is True
        and transport.get("output_limited") is False
    ):
        raise OpenRouterError(
            "Terminal non-JSON failure requires a complete non-limited turn"
        )
    try:
        _parse_strict_json_document(ledger.text)
    except OpenRouterError:
        body = {
            "version": STRUCTURED_TERMINAL_SEMANTIC_FAILURE_VERSION,
            "failure_kind": "complete_non_json_document",
            "terminal": True,
            "document_sha256": text_sha256(ledger.text),
            "document_chars": len(ledger.text),
            "document_utf8_bytes": len(ledger.text.encode("utf-8")),
            "response_schema_sha256": _stable_sha256(response_schema),
            "last_transport_sha256": _stable_sha256(transport),
        }
    else:
        raise OpenRouterError(
            "Terminal non-JSON failure requires an invalid JSON document"
        )
    return {**body, "marker_sha256": _stable_sha256(body)}


def _structured_terminal_rejected_part_failure_marker(
    ledger: StructuredContinuationLedger,
    response_schema: dict[str, Any],
    rejected_part: dict[str, Any],
) -> dict[str, Any]:
    """Seal a complete paid part rejected by the JSON boundary ledger."""

    raw_text = rejected_part.get("raw_text")
    transport = rejected_part.get("transport")
    sequence = rejected_part.get("sequence")
    expected_sequence = 0 if not ledger.text else ledger.continuation_count + 1
    if not (
        isinstance(raw_text, str)
        and rejected_part.get("text_sha256") == text_sha256(raw_text)
        and rejected_part.get("text_chars") == len(raw_text)
        and rejected_part.get("text_utf8_bytes")
        == len(raw_text.encode("utf-8"))
        and sequence == expected_sequence
        and isinstance(transport, dict)
        and transport.get("status") == "succeeded"
        and transport.get("http_status") == 200
        and transport.get("output_complete") is True
        and transport.get("output_limited") is False
    ):
        raise OpenRouterError(
            "Terminal rejected part requires one exact complete paid response"
        )
    failure_kind: str
    if raw_text == "":
        failure_kind = "complete_empty_response"
    else:
        try:
            if ledger.text:
                trial = copy.deepcopy(ledger)
                trial.append(raw_text, sequence=sequence)
            else:
                StructuredContinuationLedger(
                    document_id=ledger.document_id,
                    text=raw_text,
                    overlap_chars=ledger.overlap_chars,
                )
        except ValueError:
            failure_kind = "complete_rejected_json_part"
        else:
            raise OpenRouterError(
                "Terminal rejected part requires a failed JSON boundary"
            )
    if failure_kind in {
        "complete_empty_response",
        "complete_rejected_json_part",
    }:
        body = {
            "version": STRUCTURED_TERMINAL_SEMANTIC_FAILURE_VERSION,
            "failure_kind": failure_kind,
            "terminal": True,
            "document_sha256": text_sha256(ledger.text),
            "document_chars": len(ledger.text),
            "document_utf8_bytes": len(ledger.text.encode("utf-8")),
            "response_schema_sha256": _stable_sha256(response_schema),
            "overlap_chars": ledger.overlap_chars,
            "rejected_sequence": sequence,
            "rejected_text_sha256": text_sha256(raw_text),
            "rejected_text_chars": len(raw_text),
            "rejected_text_utf8_bytes": len(raw_text.encode("utf-8")),
            "rejected_transport_sha256": _stable_sha256(transport),
        }
    return {**body, "marker_sha256": _stable_sha256(body)}


def _validated_structured_terminal_semantic_failure(
    manifest: Any,
    marker: Any,
    *,
    response_schema: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate one sealed terminal semantic marker against its full ledger."""

    if not isinstance(manifest, dict) or not isinstance(marker, dict):
        return None
    normalized = copy.deepcopy(marker)
    marker_sha256 = str(normalized.pop("marker_sha256", ""))
    accepted_text = manifest.get("accepted_document_text")
    if (
        not marker_sha256
        or _stable_sha256(normalized) != marker_sha256
        or normalized.get("version")
        != STRUCTURED_TERMINAL_SEMANTIC_FAILURE_VERSION
        or normalized.get("failure_kind")
        not in {
            "complete_json_schema_violation",
            "complete_non_json_document",
            "complete_rejected_json_part",
            "complete_empty_response",
        }
        or normalized.get("terminal") is not True
        or not isinstance(accepted_text, str)
        or normalized.get("document_sha256") != text_sha256(accepted_text)
        or manifest.get("document_sha256") != text_sha256(accepted_text)
        or normalized.get("document_chars") != len(accepted_text)
        or normalized.get("document_utf8_bytes")
        != len(accepted_text.encode("utf-8"))
        or normalized.get("response_schema_sha256")
        != _stable_sha256(response_schema)
    ):
        return None
    failure_kind = normalized["failure_kind"]
    if failure_kind == "complete_json_schema_violation":
        try:
            parsed = _parse_strict_json_document(accepted_text)
        except OpenRouterError:
            return None
        try:
            Draft202012Validator(response_schema).validate(parsed)
        except ValidationError as exc:
            if (
                normalized.get("validation_path")
                != list(exc.absolute_path)
                or normalized.get("schema_path")
                != list(exc.absolute_schema_path)
                or normalized.get("validator")
                != str(exc.validator or "")
            ):
                return None
            return copy.deepcopy(marker)
        return None

    if failure_kind in {
        "complete_rejected_json_part",
        "complete_empty_response",
    }:
        rejected_part = manifest.get("terminal_rejected_part")
        if not isinstance(rejected_part, dict):
            rejected_part = manifest.get("rejected_part")
        raw_text = (
            rejected_part.get("raw_text")
            if isinstance(rejected_part, dict)
            else None
        )
        transport = (
            rejected_part.get("transport")
            if isinstance(rejected_part, dict)
            else None
        )
        sequence = (
            rejected_part.get("sequence")
            if isinstance(rejected_part, dict)
            else None
        )
        if not (
            isinstance(raw_text, str)
            and rejected_part.get("text_sha256") == text_sha256(raw_text)
            and rejected_part.get("text_chars") == len(raw_text)
            and rejected_part.get("text_utf8_bytes")
            == len(raw_text.encode("utf-8"))
            and normalized.get("rejected_sequence") == sequence
            and normalized.get("rejected_text_sha256")
            == text_sha256(raw_text)
            and normalized.get("rejected_text_chars") == len(raw_text)
            and normalized.get("rejected_text_utf8_bytes")
            == len(raw_text.encode("utf-8"))
            and isinstance(transport, dict)
            and normalized.get("rejected_transport_sha256")
            == _stable_sha256(transport)
            and transport.get("status") == "succeeded"
            and transport.get("http_status") == 200
            and transport.get("output_complete") is True
            and transport.get("output_limited") is False
        ):
            return None
        calls = manifest.get("calls")
        marker_overlap_chars = normalized.get("overlap_chars")
        if (
            isinstance(marker_overlap_chars, bool)
            or not isinstance(marker_overlap_chars, int)
            or marker_overlap_chars < 1
        ):
            return None
        try:
            if isinstance(calls, list) and calls:
                accepted_ledger = StructuredContinuationLedger(
                    document_id=str(manifest.get("document_id") or ""),
                    text=str(calls[0].get("raw_text") or ""),
                    overlap_chars=marker_overlap_chars,
                )
                for call_sequence, call in enumerate(calls[1:], start=1):
                    accepted_ledger.append(
                        str(call.get("raw_text") or ""),
                        sequence=call_sequence,
                    )
            elif accepted_text == "":
                accepted_ledger = StructuredContinuationLedger(
                    document_id=str(manifest.get("document_id") or ""),
                    text="",
                    overlap_chars=marker_overlap_chars,
                )
            else:
                return None
            if accepted_ledger.text != accepted_text:
                return None
            rebuilt_marker = _structured_terminal_rejected_part_failure_marker(
                accepted_ledger,
                response_schema,
                rejected_part,
            )
        except (OpenRouterError, ValueError, AttributeError):
            return None
        if rebuilt_marker.get("failure_kind") != failure_kind:
            return None
        return copy.deepcopy(marker)

    calls = manifest.get("calls")
    last_call = calls[-1] if isinstance(calls, list) and calls else None
    last_transport = (
        last_call.get("transport") if isinstance(last_call, dict) else None
    )
    if not (
        isinstance(last_transport, dict)
        and normalized.get("last_transport_sha256")
        == _stable_sha256(last_transport)
        and last_transport.get("status") == "succeeded"
        and last_transport.get("http_status") == 200
        and last_transport.get("output_complete") is True
        and last_transport.get("output_limited") is False
    ):
        return None
    try:
        _parse_strict_json_document(accepted_text)
    except OpenRouterError:
        return copy.deepcopy(marker)
    return None


def validated_structured_terminal_semantic_failure(
    error: BaseException,
    *,
    response_schema: dict[str, Any],
) -> dict[str, Any] | None:
    """Return an audited terminal schema failure carried by an exception."""

    if not isinstance(error, OpenRouterStructuredContinuationError):
        return None
    manifest = error.manifest
    marker = (
        manifest.get("terminal_semantic_failure")
        if isinstance(manifest, dict)
        else None
    )
    return _validated_structured_terminal_semantic_failure(
        manifest,
        marker,
        response_schema=response_schema,
    )


def _structured_checkpoint_error(
    error: BaseException | str | None,
) -> dict[str, Any] | None:
    """Serialize an error together with its sealed terminal disposition."""

    if error is None:
        return None
    payload: dict[str, Any] = {
        "type": (
            type(error).__name__
            if isinstance(error, BaseException)
            else "OpenRouterStructuredContinuationError"
        ),
        "message": str(error),
    }
    if isinstance(error, OpenRouterStructuredContinuationError):
        marker = error.manifest.get("terminal_semantic_failure")
        if isinstance(marker, dict):
            payload["terminal_semantic_failure"] = copy.deepcopy(marker)
            if marker.get("failure_kind") in {
                "complete_rejected_json_part",
                "complete_empty_response",
            }:
                rejected_part = error.manifest.get("rejected_part")
                if isinstance(rejected_part, dict):
                    payload["terminal_rejected_part"] = copy.deepcopy(
                        rejected_part
                    )
    return payload


def _structured_audit_context(
    *,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    base_result: ChatResult,
    sequence: int,
    resume_contract: dict[str, Any],
    audit_checkpoint: AuditCheckpoint | None,
    cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _structured_delta_audit_enabled(audit_checkpoint):
        resolved_cursor = dict(cursor or ledger.cursor())
        if resolved_cursor.get("next_sequence") != sequence:
            raise OpenRouterError(
                "Structured delta predecessor sequence mismatch"
            )
        return {
            "document_id": ledger.document_id,
            "sequence": sequence,
            "predecessor": {
                "document_id": ledger.document_id,
                "expected_sequence": sequence,
                "latest_sequence": sequence - 1,
                "complete": False,
                "document_sha256": resolved_cursor.get("document_sha256"),
                "document_chars": resolved_cursor.get("document_chars"),
                # The durable store derives exact UTF-8 bytes from immutable
                # receipt fragments.  Re-encoding the whole growing document
                # here on every POST would reintroduce quadratic producer work.
                "document_utf8_bytes": None,
                "expected_overlap_sha256": resolved_cursor.get(
                    "expected_overlap_sha256"
                ),
                "expected_overlap_chars": resolved_cursor.get(
                    "expected_overlap_chars"
                ),
            },
            "resume_contract": dict(resume_contract),
        }
    return {
        "document_id": ledger.document_id,
        "sequence": sequence,
        "partial_text": ledger.text,
        "manifest": _structured_manifest_with_calls(
            ledger,
            call_records=call_records,
            complete=False,
        ),
        "aggregate_usage": _aggregate_continuation_usage(
            base_result.usage,
            call_records,
        ),
        "call_records": [dict(record) for record in call_records],
        "resume_contract": dict(resume_contract),
    }


def _structured_checkpoint_delta_event(
    *,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    sequence: int,
    status: str,
    complete: bool,
    resume_contract: dict[str, Any],
    error: BaseException | str | None = None,
) -> dict[str, Any]:
    """Build one O(1)-sized head/fragment delta for the durable store."""

    latest_sequence = len(call_records) - 1
    if call_records and sequence != latest_sequence:
        raise OpenRouterError(
            "Structured delta checkpoint sequence/call mismatch"
        )
    if not call_records and sequence not in {-1, 0}:
        raise OpenRouterError(
            "Structured empty delta checkpoint sequence is invalid"
        )

    accepted_fragment: dict[str, Any] | None = None
    if call_records:
        latest_call = call_records[-1]
        latest_part = ledger.parts[-1]
        raw_text = latest_call.get("raw_text")
        if not isinstance(raw_text, str):
            raise OpenRouterError(
                "Structured delta checkpoint has no latest raw fragment"
            )
        excluded = {
            "raw_text",
            "usage",
            "transport",
            "request_policy",
            "web_attestation",
            "prior_transport_attempts",
        }
        accepted_fragment = {
            "sequence": latest_sequence,
            "raw_text_sha256": text_sha256(raw_text),
            "raw_text_chars": len(raw_text),
            "raw_text_utf8_bytes": len(raw_text.encode("utf-8")),
            "part": dict(latest_part),
            "call_metadata": {
                key: value
                for key, value in latest_call.items()
                if key not in excluded
            },
        }
        document_sha256 = latest_part.get("document_sha256")
        document_chars = latest_part.get("document_chars")
        part_count = len(call_records)
        continuation_count = max(0, part_count - 1)
    else:
        document_sha256 = text_sha256("")
        document_chars = 0
        part_count = 0
        continuation_count = 0

    return {
        "version": STRUCTURED_CHECKPOINT_DELTA_VERSION,
        "event_id": uuid.uuid4().hex,
        "event_kind": "structured_continuation_checkpoint",
        "document_id": ledger.document_id,
        "sequence": latest_sequence,
        "status": status,
        "complete": bool(complete),
        "head": {
            "version": STRUCTURED_CONTINUATION_VERSION,
            "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
            "document_id": ledger.document_id,
            "complete": bool(complete),
            "latest_sequence": latest_sequence,
            "continuation_count": continuation_count,
            "part_count": part_count,
            "document_sha256": document_sha256,
            "document_chars": document_chars,
            # Reconstructed from receipts by the durable store.  Keeping these
            # nullable avoids a second full-document scan in the producer.
            "document_utf8_bytes": None,
            "json_prefix": None,
        },
        "accepted_fragment": accepted_fragment,
        "resume_contract": dict(resume_contract),
        "error": _structured_checkpoint_error(error),
    }


def _structured_checkpoint_event(
    *,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    base_result: ChatResult,
    sequence: int,
    status: str,
    complete: bool,
    resume_contract: dict[str, Any],
    error: BaseException | str | None = None,
) -> dict[str, Any]:
    manifest = _structured_manifest_with_calls(
        ledger,
        call_records=call_records,
        complete=complete,
    )
    aggregate_usage = _aggregate_continuation_usage(
        base_result.usage,
        call_records,
    )
    return {
        "version": STRUCTURED_CHECKPOINT_VERSION,
        "event_id": uuid.uuid4().hex,
        "event_kind": "structured_continuation_checkpoint",
        "document_id": ledger.document_id,
        "sequence": sequence,
        "status": status,
        "partial_text": ledger.text,
        "manifest": manifest,
        "aggregate_usage": aggregate_usage,
        "call_records": [dict(record) for record in call_records],
        "resume_contract": dict(resume_contract),
        "error": _structured_checkpoint_error(error),
    }


async def _emit_structured_checkpoint(
    checkpoint: AuditCheckpoint | None,
    *,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    base_result: ChatResult,
    sequence: int,
    status: str,
    complete: bool,
    resume_contract: dict[str, Any],
    error: BaseException | str | None = None,
) -> None:
    if _structured_delta_audit_enabled(checkpoint):
        event = _structured_checkpoint_delta_event(
            ledger=ledger,
            call_records=call_records,
            sequence=sequence,
            status=status,
            complete=complete,
            resume_contract=resume_contract,
            error=error,
        )
    else:
        event = _structured_checkpoint_event(
            ledger=ledger,
            call_records=call_records,
            base_result=base_result,
            sequence=sequence,
            status=status,
            complete=complete,
            resume_contract=resume_contract,
            error=error,
        )
    await _emit_audit_checkpoint(checkpoint, event, result=base_result)


def _synthetic_failure_result(
    *,
    model: str,
    error: BaseException,
    fallback: ChatResult | None = None,
) -> ChatResult:
    attached = getattr(error, "result", None)
    if isinstance(attached, ChatResult):
        return attached
    audit_event = getattr(error, "audit_event", None)
    if fallback is not None:
        usage = dict(fallback.usage)
        transport = {
            **dict(fallback.transport),
            "status": "failed",
            "output_complete": False,
            "output_incomplete_reason": f"{type(error).__name__}: {error}",
        }
        usage["_aiv_transport"] = transport
        if isinstance(audit_event, dict):
            usage["_aiv_failed_post"] = dict(audit_event)
        return replace(
            fallback,
            parsed=None,
            usage=usage,
            transport=transport,
        )
    raw_text = ""
    usage: dict[str, Any] = {}
    if isinstance(audit_event, dict):
        response = audit_event.get("response")
        if isinstance(response, dict):
            body = response.get("body_json")
            if body is not None:
                raw_text = json.dumps(body, ensure_ascii=False)
                if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                    usage.update(body["usage"])
            elif response.get("body_text") is not None:
                raw_text = str(response.get("body_text") or "")
        usage["_aiv_failed_post"] = dict(audit_event)
    transport = {
        "version": TRANSPORT_METADATA_VERSION,
        "status": "failed",
        "http_status": (
            (audit_event.get("response") or {}).get("http_status")
            if isinstance(audit_event, dict)
            and isinstance(audit_event.get("response"), dict)
            else None
        ),
        "attempt": (
            audit_event.get("attempt") if isinstance(audit_event, dict) else None
        ),
        "requested_model": model,
        "response_model": "",
        "provider": "",
        "response_id": "",
        "finish_reason": "",
        "native_finish_reason": "",
        "output_complete": False,
        "output_limited": False,
        "output_incomplete_reason": f"{type(error).__name__}: {error}",
    }
    usage["_aiv_transport"] = transport
    return ChatResult(
        text=raw_text,
        parsed=None,
        citations=[],
        usage=usage,
        annotations=[],
        request_policy={},
        web_attestation={},
        router_metadata={},
        transport=transport,
    )


def structured_resume_contract(
    *,
    model: str,
    messages: list[dict[str, Any]],
    schema_name: str,
    response_schema: dict[str, Any],
    document_id: str,
    reasoning_effort: str | None = None,
    temperature: float = 0.2,
    overlap_chars: int = DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
) -> dict[str, Any]:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
    ):
        raise OpenRouterError(
            "Structured resume temperature must be a finite number"
        )
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, int)
        or overlap_chars < 1
    ):
        raise OpenRouterError(
            "Structured resume overlap_chars must be a positive integer"
        )
    normalized_reasoning = (
        str(reasoning_effort).strip() if reasoning_effort is not None else None
    )
    if normalized_reasoning == "":
        normalized_reasoning = None
    payload = {
        "version": "aiv-structured-resume-contract-v2",
        "long_response_harness_version": LONG_RESPONSE_HARNESS_VERSION,
        "continuation_protocol_version": STRUCTURED_CONTINUATION_VERSION,
        "response_mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
        "document_id": document_id,
        "model": model,
        "messages_sha256": _stable_sha256(messages),
        "schema_name": schema_name,
        "response_schema_sha256": _stable_sha256(response_schema),
        "reasoning_effort": normalized_reasoning,
        "temperature": float(temperature),
        "overlap_chars": overlap_chars,
        "web_policy": WebSearchPolicy.FORBIDDEN.value,
        "output_token_policy": OutputTokenPolicy.MODEL_MAX.value,
    }
    return {**payload, "sha256": _stable_sha256(payload)}


def _resume_int_field(
    record: dict[str, Any],
    key: str,
    *,
    expected: int,
    label: str,
) -> None:
    value = record.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected
    ):
        raise OpenRouterError(
            f"Structured resume {label} {key} mismatch"
        )


def _normalize_resume_raw_record(
    raw_record: dict[str, Any],
    *,
    label: str,
    expected_sequence: int | None = None,
) -> dict[str, Any]:
    """Recompute every raw-fragment identity field from untrusted storage."""

    record = dict(raw_record)
    raw_text = record.get("raw_text")
    if not isinstance(raw_text, str):
        raise OpenRouterError(
            f"Structured resume {label} raw_text is missing"
        )
    if expected_sequence is not None:
        _resume_int_field(
            record,
            "sequence",
            expected=expected_sequence,
            label=label,
        )
    if str(record.get("text_sha256") or "") != text_sha256(raw_text):
        raise OpenRouterError(f"Structured resume {label} digest mismatch")
    _resume_int_field(
        record,
        "text_chars",
        expected=len(raw_text),
        label=label,
    )
    utf8_bytes = len(raw_text.encode("utf-8"))
    stored_utf8_bytes = record.get("text_utf8_bytes")
    if stored_utf8_bytes is not None:
        _resume_int_field(
            record,
            "text_utf8_bytes",
            expected=utf8_bytes,
            label=label,
        )
    else:
        # v1 chat-attempt records predate this redundant byte counter.  Their
        # text digest and character count are still fully revalidated; add the
        # recomputed field to the normalized checkpoint rather than trusting a
        # missing legacy value.
        record["text_utf8_bytes"] = utf8_bytes
    for key in ("usage", "transport"):
        value = record.get(key)
        if not isinstance(value, dict):
            raise OpenRouterError(
                f"Structured resume {label} {key} is invalid"
            )
        record[key] = dict(value)
    for key in ("request_policy", "web_attestation"):
        value = record.get(key, {})
        if not isinstance(value, dict):
            raise OpenRouterError(
                f"Structured resume {label} {key} is invalid"
            )
        record[key] = dict(value)
    return record


def _resume_structured_state(
    checkpoint: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    overlap_chars: int,
    expected_complete: bool = False,
) -> tuple[StructuredContinuationLedger, list[dict[str, Any]], ChatResult]:
    """Revalidate a durable checkpoint without trusting stored derived fields."""

    if not isinstance(checkpoint, dict):
        raise OpenRouterError("resume_checkpoint must be an object")
    if checkpoint.get("version") != STRUCTURED_CHECKPOINT_VERSION or (
        checkpoint.get("event_kind")
        != "structured_continuation_checkpoint"
    ):
        raise OpenRouterError(
            "Structured resume checkpoint has an unsupported format"
        )
    contract = checkpoint.get("resume_contract")
    if not isinstance(contract, dict) or contract != expected_contract:
        raise OpenRouterError(
            "Structured resume checkpoint does not match model/input/schema contract"
        )
    if str(checkpoint.get("document_id") or "") != str(
        expected_contract["document_id"]
    ):
        raise OpenRouterError("Structured resume document_id mismatch")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict) or (
        manifest.get("complete") is not expected_complete
    ):
        raise OpenRouterError(
            "Structured checkpoint completion state mismatch: expected "
            f"complete={str(expected_complete).lower()}"
        )
    calls = checkpoint.get("call_records")
    if not isinstance(calls, list) or not calls:
        calls = manifest.get("calls")
    if not isinstance(calls, list) or not calls:
        raise OpenRouterError("Structured resume checkpoint has no provider calls")
    manifest_calls = manifest.get("calls")
    if not isinstance(manifest_calls, list) or _stable_sha256(
        manifest_calls
    ) != _stable_sha256(calls):
        raise OpenRouterError("Structured resume call ledger mismatch")

    normalized_calls: list[dict[str, Any]] = []
    for expected_sequence, raw_call in enumerate(calls):
        if not isinstance(raw_call, dict):
            raise OpenRouterError("Structured resume call record is invalid")
        call = _normalize_resume_raw_record(
            raw_call,
            label=f"call {expected_sequence}",
            expected_sequence=expected_sequence,
        )
        prior_attempts = call.get("prior_transport_attempts", [])
        if not isinstance(prior_attempts, list):
            raise OpenRouterError(
                "Structured resume prior transport ledger is invalid"
            )
        normalized_attempts: list[dict[str, Any]] = []
        for attempt_index, raw_attempt in enumerate(prior_attempts):
            if not isinstance(raw_attempt, dict):
                raise OpenRouterError(
                    "Structured resume prior transport record is invalid"
                )
            attempt = _normalize_resume_raw_record(
                raw_attempt,
                label=(
                    f"call {expected_sequence} prior attempt "
                    f"{attempt_index}"
                ),
            )
            attempt_number = attempt.get("attempt")
            if (
                isinstance(attempt_number, bool)
                or not isinstance(attempt_number, int)
                or attempt_number < 1
            ):
                raise OpenRouterError(
                    "Structured resume prior transport attempt number is invalid"
                )
            normalized_attempts.append(attempt)
        call["prior_transport_attempts"] = normalized_attempts
        normalized_calls.append(call)

    ledger = StructuredContinuationLedger(
        document_id=str(expected_contract["document_id"]),
        text=str(normalized_calls[0]["raw_text"]),
        overlap_chars=overlap_chars,
    )
    for sequence, call in enumerate(normalized_calls[1:], start=1):
        ledger.append(str(call["raw_text"]), sequence=sequence)
    partial_text = checkpoint.get("partial_text")
    if not isinstance(partial_text, str) or partial_text != ledger.text:
        raise OpenRouterError("Structured resume partial_text mismatch")
    if str(manifest.get("accepted_document_text") or "") != ledger.text:
        raise OpenRouterError("Structured resume accepted text mismatch")
    rebuilt = ledger.manifest(complete=False)
    for key in (
        "document_id",
        "continuation_count",
        "part_count",
        "document_sha256",
        "document_chars",
        "document_utf8_bytes",
        "json_prefix",
    ):
        if manifest.get(key) != rebuilt.get(key):
            raise OpenRouterError(
                f"Structured resume manifest mismatch: {key}"
            )
    stored_parts = manifest.get("parts")
    if not isinstance(stored_parts, list) or len(stored_parts) != len(
        rebuilt["parts"]
    ):
        raise OpenRouterError("Structured resume part ledger mismatch")
    # ``kind`` is a presentation label; every cryptographic/offset field must
    # match the chain rebuilt solely from raw provider fragments.
    for stored, actual in zip(stored_parts, rebuilt["parts"], strict=True):
        if not isinstance(stored, dict):
            raise OpenRouterError("Structured resume part record is invalid")
        for key, value in actual.items():
            if key == "kind":
                continue
            if stored.get(key) != value:
                raise OpenRouterError(
                    f"Structured resume part hash-chain mismatch: {key}"
                )

    first = normalized_calls[0]
    first_usage = dict(first["usage"])
    first_transport = dict(first["transport"])
    initial = ChatResult(
        text=str(first["raw_text"]),
        parsed=None,
        citations=[],
        usage=first_usage,
        annotations=[],
        request_policy=dict(first.get("request_policy") or {}),
        web_attestation=dict(first.get("web_attestation") or {}),
        router_metadata={},
        transport=first_transport,
    )
    return ledger, normalized_calls, initial


def _resume_terminal_rejected_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    overlap_chars: int,
    response_schema: dict[str, Any],
) -> OpenRouterStructuredContinuationError | None:
    """Restore a terminal paid part that was never accepted into the ledger."""

    error = checkpoint.get("error") if isinstance(checkpoint, dict) else None
    marker = (
        error.get("terminal_semantic_failure")
        if isinstance(error, dict)
        else None
    )
    if not isinstance(marker, dict) or marker.get("failure_kind") not in {
        "complete_rejected_json_part",
        "complete_empty_response",
    }:
        return None
    rejected_part = error.get("terminal_rejected_part")
    if not isinstance(rejected_part, dict):
        raise OpenRouterError(
            "Durable terminal rejected-part evidence is missing"
        )
    calls = checkpoint.get("call_records")
    if not isinstance(calls, list):
        raise OpenRouterError(
            "Durable terminal rejected-part call ledger is invalid"
        )
    if calls:
        ledger, normalized_calls, _initial = _resume_structured_state(
            checkpoint,
            expected_contract=expected_contract,
            overlap_chars=overlap_chars,
        )
    else:
        if checkpoint.get("version") != STRUCTURED_CHECKPOINT_VERSION or (
            checkpoint.get("event_kind")
            != "structured_continuation_checkpoint"
        ):
            raise OpenRouterError(
                "Terminal rejected-part checkpoint has an unsupported format"
            )
        if checkpoint.get("resume_contract") != expected_contract or (
            str(checkpoint.get("document_id") or "")
            != str(expected_contract["document_id"])
        ):
            raise OpenRouterError(
                "Terminal rejected-part checkpoint contract mismatch"
            )
        manifest = checkpoint.get("manifest")
        if not isinstance(manifest, dict) or (
            manifest.get("complete") is not False
            or manifest.get("calls") != []
            or manifest.get("accepted_document_text") != ""
            or checkpoint.get("partial_text") != ""
            or manifest.get("document_sha256") != text_sha256("")
            or manifest.get("document_chars") != 0
            or manifest.get("document_utf8_bytes", 0) != 0
            or manifest.get("continuation_count", 0) != 0
            or manifest.get("part_count") not in {0, 1}
            or checkpoint.get("sequence") not in {-1, 0}
        ):
            raise OpenRouterError(
                "Terminal rejected-part empty ledger is inconsistent"
            )
        ledger = StructuredContinuationLedger(
            document_id=str(expected_contract["document_id"]),
            text="",
            overlap_chars=overlap_chars,
        )
        normalized_calls = []

    rejected_sequence = marker.get("rejected_sequence")
    rejected = _normalize_resume_raw_record(
        copy.deepcopy(rejected_part),
        label="terminal rejected part",
        expected_sequence=(
            int(rejected_sequence)
            if isinstance(rejected_sequence, int)
            and not isinstance(rejected_sequence, bool)
            else None
        ),
    )
    terminal_manifest = _structured_manifest_with_calls(
        ledger,
        call_records=normalized_calls,
        complete=False,
    )
    terminal_manifest["terminal_semantic_failure"] = copy.deepcopy(marker)
    terminal_manifest["terminal_rejected_part"] = copy.deepcopy(rejected)
    if (
        _validated_structured_terminal_semantic_failure(
            terminal_manifest,
            marker,
            response_schema=response_schema,
        )
        is None
    ):
        raise OpenRouterError(
            "Durable terminal structured semantic marker is invalid"
        )
    result = ChatResult(
        text=str(rejected["raw_text"]),
        parsed=None,
        citations=[],
        usage={
            **dict(rejected["usage"]),
            "_aiv_structured_continuation": terminal_manifest,
            "_aiv_transport": dict(rejected["transport"]),
        },
        annotations=[],
        request_policy=dict(rejected.get("request_policy") or {}),
        web_attestation=dict(rejected.get("web_attestation") or {}),
        router_metadata={},
        transport=dict(rejected["transport"]),
    )
    return OpenRouterStructuredContinuationError(
        "Durable complete provider response was rejected by the JSON ledger",
        result=result,
        manifest=terminal_manifest,
    )


def restore_completed_structured_checkpoint(
    checkpoint: dict[str, Any],
    model: str,
    messages: list[dict[str, Any]],
    schema_name: str,
    response_schema: dict[str, Any],
    document_id: str,
    overlap_chars: int = DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
    reasoning_effort: str | None = None,
    temperature: float = 0.2,
) -> ChatResult:
    """Restore one completed structured result from an untrusted checkpoint.

    This is intentionally synchronous and network-free.  It rebuilds the
    contract and every accepted raw fragment using the same validator as
    incomplete resume, then independently parses and validates the assembled
    root document before returning a normal aggregated ``ChatResult``.
    """

    if not isinstance(response_schema, dict) or not response_schema:
        raise OpenRouterError("response_schema must be a non-empty JSON object")
    if not isinstance(document_id, str) or not document_id:
        raise OpenRouterError("document_id must be a non-empty string")
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, int)
        or overlap_chars < 1
    ):
        raise OpenRouterError("overlap_chars must be a positive integer")
    contract = structured_resume_contract(
        model=model,
        messages=messages,
        schema_name=schema_name,
        response_schema=response_schema,
        document_id=document_id,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        overlap_chars=overlap_chars,
    )
    ledger, call_records, initial = _resume_structured_state(
        checkpoint,
        expected_contract=contract,
        overlap_chars=overlap_chars,
        expected_complete=True,
    )
    try:
        parsed = _parse_strict_json_document(ledger.text)
        Draft202012Validator(response_schema).validate(parsed)
    except (OpenRouterError, ValidationError) as exc:
        raise OpenRouterError(
            f"Completed structured checkpoint is unusable: {exc}"
        ) from exc
    last_call = call_records[-1]
    latest = replace(
        initial,
        text=str(last_call["raw_text"]),
        usage=dict(last_call["usage"]),
        transport=dict(last_call["transport"]),
        request_policy=dict(last_call.get("request_policy") or {}),
        web_attestation=dict(last_call.get("web_attestation") or {}),
    )
    return _continued_result(
        initial=initial,
        latest=latest,
        ledger=ledger,
        call_records=call_records,
        parsed=parsed,
    )


def _validated_provider_event_result(
    provider_event: dict[str, Any],
    *,
    model: str,
    expected_contract: dict[str, Any],
    expected_sequence: int,
    expected_messages: list[dict[str, Any]],
    schema_name: str,
    response_schema: dict[str, Any],
) -> ChatResult:
    """Rebuild one ChatResult solely from a self-consistent physical event."""

    if not isinstance(provider_event, dict):
        raise OpenRouterError("provider_event must be an object")
    if provider_event.get("version") != PHYSICAL_POST_AUDIT_VERSION or (
        provider_event.get("event_kind") != "provider_post"
    ):
        raise OpenRouterError("Provider event has an unsupported format")
    if str(provider_event.get("model") or "") != model:
        raise OpenRouterError("Provider event model mismatch")
    if str(provider_event.get("document_id") or "") != str(
        expected_contract["document_id"]
    ):
        raise OpenRouterError("Provider event document_id mismatch")
    if provider_event.get("resume_contract") != expected_contract:
        raise OpenRouterError("Provider event resume contract mismatch")
    _resume_int_field(
        provider_event,
        "sequence",
        expected=expected_sequence,
        label="provider event",
    )
    attempt = provider_event.get("attempt")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise OpenRouterError("Provider event attempt is invalid")

    request_payload = provider_event.get("request_payload")
    if not isinstance(request_payload, dict) or (
        str(provider_event.get("request_sha256") or "")
        != _stable_sha256(request_payload)
    ):
        raise OpenRouterError("Provider event request digest mismatch")
    if request_payload.get("model") != model or (
        request_payload.get("messages") != expected_messages
    ):
        raise OpenRouterError("Provider event request input mismatch")
    if request_payload.get("temperature") != expected_contract.get(
        "temperature"
    ):
        raise OpenRouterError("Provider event request temperature mismatch")
    expected_reasoning_effort = expected_contract.get("reasoning_effort")
    expected_reasoning = (
        {
            "effort": expected_reasoning_effort,
            "exclude": True,
        }
        if expected_reasoning_effort is not None
        else None
    )
    if expected_reasoning is None:
        if "reasoning" in request_payload:
            raise OpenRouterError(
                "Provider event request has unexpected reasoning policy"
            )
    elif request_payload.get("reasoning") != expected_reasoning:
        raise OpenRouterError("Provider event request reasoning mismatch")
    policy_fields, expected_policy = web_request_policy(
        model=model,
        policy=WebSearchPolicy.FORBIDDEN,
    )
    for key, expected in policy_fields.items():
        if request_payload.get(key) != expected:
            raise OpenRouterError(
                f"Provider event request policy mismatch: {key}"
            )
    if expected_sequence == 0:
        expected_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        }
        if request_payload.get("response_format") != expected_format:
            raise OpenRouterError(
                "Provider event initial response schema mismatch"
            )
    elif "response_format" in request_payload:
        raise OpenRouterError(
            "Provider continuation unexpectedly contains a root response schema"
        )

    status = str(provider_event.get("status") or "")
    if status not in {"accepted", "rejected"}:
        raise OpenRouterError(
            f"Provider event status is not promotable: {status or 'missing'}"
        )
    response = provider_event.get("response")
    if not isinstance(response, dict):
        raise OpenRouterError("Provider event has no response snapshot")
    http_status = response.get("http_status")
    if (
        isinstance(http_status, bool)
        or not isinstance(http_status, int)
        or not 200 <= http_status < 300
    ):
        raise OpenRouterError("Provider event HTTP response is not successful")
    body = response.get("body_json")
    if not isinstance(body, dict):
        raise OpenRouterError("Provider event has no complete JSON body")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(
        choices[0], dict
    ):
        raise OpenRouterError("Provider event response has no valid choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError("Provider event response message is invalid")
    raw_text = provider_event.get("raw_text")
    if (
        not isinstance(raw_text, str)
        or _message_text(message) != raw_text
    ):
        raise OpenRouterError("Provider event raw response mismatch")

    usage = provider_event.get("usage")
    raw_usage = body.get("usage")
    if not isinstance(usage, dict) or not isinstance(raw_usage, dict):
        raise OpenRouterError("Provider event usage is invalid")
    for key, value in raw_usage.items():
        if key not in usage or usage[key] != value:
            raise OpenRouterError(
                f"Provider event raw usage mismatch: {key}"
            )
    annotations = _annotations(message)
    citations = _citations(message)
    router_metadata = body.get("openrouter_metadata") or {}
    if not isinstance(router_metadata, dict):
        raise OpenRouterError("Provider event router metadata is invalid")
    expected_attestation = attest_web_response(
        requested_model=model,
        response_model=str(body.get("model") or "").strip(),
        policy=WebSearchPolicy.FORBIDDEN,
        usage=raw_usage,
        annotations=annotations,
        citations=citations,
        router_metadata=router_metadata,
    )
    if usage.get("_aiv_request_policy") != expected_policy or (
        usage.get("_aiv_web_attestation") != expected_attestation
    ):
        raise OpenRouterError("Provider event policy evidence mismatch")
    if not expected_attestation.get("metric_eligible"):
        raise OpenRouterError("Provider event is not metric-eligible")

    transport = provider_event.get("transport")
    if not isinstance(transport, dict):
        raise OpenRouterError("Provider event transport is invalid")
    expected_transport = _transport_metadata(
        body=body,
        choice=choice,
        requested_model=model,
        response_model=str(body.get("model") or "").strip(),
        attempt=attempt,
    )
    for key, value in expected_transport.items():
        if transport.get(key) != value:
            raise OpenRouterError(
                f"Provider event transport mismatch: {key}"
            )
    if usage.get("_aiv_transport") != transport:
        raise OpenRouterError("Provider event usage/transport mismatch")
    if not transport.get("output_complete") and not transport.get(
        "output_limited"
    ):
        raise OpenRouterError("Provider event response is incomplete")
    error = provider_event.get("error")
    if status == "accepted":
        if error is not None:
            raise OpenRouterError("Accepted provider event contains an error")
    else:
        rejected_limit_part = bool(
            raw_text
            and
            transport.get("output_limited")
            and isinstance(error, dict)
            and error.get("type") == "OpenRouterOutputLimitError"
        )
        rejected_complete_contract = bool(
            transport.get("output_complete") is True
            and transport.get("output_limited") is False
            and isinstance(error, dict)
            and error.get("type") == "OpenRouterResponseContractError"
        )
        if not rejected_limit_part and not rejected_complete_contract:
            raise OpenRouterError(
                "Rejected provider event is not a usable structured part"
            )

    return ChatResult(
        text=raw_text,
        parsed=None,
        citations=citations,
        usage=dict(usage),
        annotations=annotations,
        request_policy=dict(expected_policy),
        web_attestation=dict(expected_attestation),
        router_metadata=dict(router_metadata),
        transport=dict(transport),
    )


def promote_provider_post_to_structured_checkpoint(
    provider_event: dict[str, Any],
    predecessor_checkpoint: dict[str, Any] | None,
    model: str,
    messages: list[dict[str, Any]],
    schema_name: str,
    response_schema: dict[str, Any],
    document_id: str,
    overlap_chars: int = DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
    reasoning_effort: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Promote a persisted physical POST across the structured crash gap.

    The helper is network-free and accepts only a successful response that can
    be chained exactly onto a fully revalidated predecessor.  HTTP failures,
    policy failures and malformed/tampered provider events remain audit
    evidence.  A complete schema-contract rejection may be promoted only into
    a sealed terminal failed checkpoint, never accepted document state.
    """

    if not isinstance(provider_event, dict):
        raise OpenRouterError("provider_event must be an object")
    if not isinstance(response_schema, dict) or not response_schema:
        raise OpenRouterError("response_schema must be a non-empty JSON object")
    if not isinstance(document_id, str) or not document_id:
        raise OpenRouterError("document_id must be a non-empty string")
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, int)
        or overlap_chars < 1
    ):
        raise OpenRouterError("overlap_chars must be a positive integer")
    contract = structured_resume_contract(
        model=model,
        messages=messages,
        schema_name=schema_name,
        response_schema=response_schema,
        document_id=document_id,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        overlap_chars=overlap_chars,
    )

    if predecessor_checkpoint is None:
        expected_sequence = 0
        event_manifest = provider_event.get("manifest")
        expected_pending_manifest = {
            "version": STRUCTURED_CONTINUATION_VERSION,
            "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
            "document_id": document_id,
            "complete": False,
            "part_count": 0,
            "calls": [],
            "accepted_document_text": "",
        }
        if (
            provider_event.get("partial_text") != ""
            or provider_event.get("call_records") != []
            or provider_event.get("aggregate_usage") != {}
            or event_manifest != expected_pending_manifest
        ):
            raise OpenRouterError(
                "Initial provider event predecessor state mismatch"
            )
        expected_messages = messages
        initial: ChatResult | None = None
        ledger: StructuredContinuationLedger | None = None
        call_records: list[dict[str, Any]] = []
    else:
        ledger, call_records, initial = _resume_structured_state(
            predecessor_checkpoint,
            expected_contract=contract,
            overlap_chars=overlap_chars,
            expected_complete=False,
        )
        expected_sequence = ledger.continuation_count + 1
        expected_prior_manifest = _structured_manifest_with_calls(
            ledger,
            call_records=call_records,
            complete=False,
        )
        if (
            provider_event.get("partial_text") != ledger.text
            or _stable_sha256(provider_event.get("manifest"))
            != _stable_sha256(expected_prior_manifest)
            or _stable_sha256(provider_event.get("call_records"))
            != _stable_sha256(call_records)
            or provider_event.get("aggregate_usage")
            != predecessor_checkpoint.get("aggregate_usage")
        ):
            raise OpenRouterError(
                "Provider event does not match its predecessor checkpoint"
            )
        cursor = ledger.cursor()
        expected_messages = _continuation_messages(
            messages=messages,
            document_text=ledger.text,
            cursor=cursor,
            response_schema=response_schema,
        )

    current = _validated_provider_event_result(
        provider_event,
        model=model,
        expected_contract=contract,
        expected_sequence=expected_sequence,
        expected_messages=expected_messages,
        schema_name=schema_name,
        response_schema=response_schema,
    )
    current_call = _continuation_call_record(
        current,
        sequence=expected_sequence,
    )

    def terminal_rejected_checkpoint(
        accepted_ledger: StructuredContinuationLedger,
        message: str,
    ) -> dict[str, Any]:
        marker = _structured_terminal_rejected_part_failure_marker(
            accepted_ledger,
            response_schema,
            current_call,
        )
        terminal_manifest = _structured_manifest_with_calls(
            accepted_ledger,
            call_records=call_records,
            complete=False,
        )
        terminal_manifest["rejected_part"] = copy.deepcopy(current_call)
        terminal_manifest["terminal_semantic_failure"] = marker
        terminal_failure = OpenRouterStructuredContinuationError(
            message,
            result=current,
            manifest=terminal_manifest,
        )
        failed_checkpoint = _structured_checkpoint_event(
            ledger=accepted_ledger,
            call_records=call_records,
            base_result=initial or current,
            # The rejected paid response is sealed separately in the terminal
            # marker.  Snapshot sequence tracks only accepted call records.
            sequence=accepted_ledger.continuation_count,
            status="failed",
            complete=False,
            resume_contract=contract,
            error=terminal_failure,
        )
        failed_checkpoint["promoted_from_provider_event_id"] = (
            provider_event.get("event_id")
        )
        return failed_checkpoint

    if (
        current.text == ""
        and current.transport.get("output_complete") is True
        and current.transport.get("output_limited") is False
    ):
        accepted_ledger = ledger or StructuredContinuationLedger(
            document_id=document_id,
            text="",
            overlap_chars=overlap_chars,
        )
        return terminal_rejected_checkpoint(
            accepted_ledger,
            "Promoted complete provider response is empty",
        )
    try:
        if ledger is None:
            ledger = StructuredContinuationLedger(
                document_id=document_id,
                text=current.text,
                overlap_chars=overlap_chars,
            )
            initial = current
        else:
            ledger.append(current.text, sequence=expected_sequence)
    except ValueError as exc:
        if not (
            current.transport.get("output_complete") is True
            and current.transport.get("output_limited") is False
        ):
            raise OpenRouterError(
                "Provider event did not form a continuable JSON boundary"
            ) from exc
        accepted_ledger = ledger
        if accepted_ledger is None:
            accepted_ledger = StructuredContinuationLedger(
                document_id=document_id,
                text="",
                overlap_chars=overlap_chars,
            )
        return terminal_rejected_checkpoint(
            accepted_ledger,
            "Promoted complete provider response violates the JSON boundary: "
            f"{exc}",
        )
    call_records.append(current_call)
    assert initial is not None

    terminal_failure: OpenRouterStructuredContinuationError | None = None
    try:
        parsed = _parse_strict_json_document(ledger.text)
    except OpenRouterError as exc:
        if current.transport.get("output_limited"):
            complete = False
        else:
            marker = _structured_terminal_non_json_failure_marker(
                ledger,
                response_schema,
                current.transport,
            )
            terminal_manifest = _structured_manifest_with_calls(
                ledger,
                call_records=call_records,
                complete=False,
            )
            terminal_manifest["terminal_semantic_failure"] = marker
            terminal_failure = OpenRouterStructuredContinuationError(
                "Promoted structured document reached a complete provider "
                f"turn but is not JSON: {exc}",
                result=current,
                manifest=terminal_manifest,
            )
            complete = False
    else:
        try:
            Draft202012Validator(response_schema).validate(parsed)
        except ValidationError as exc:
            marker = _structured_terminal_schema_failure_marker(
                ledger,
                response_schema,
            )
            terminal_manifest = _structured_manifest_with_calls(
                ledger,
                call_records=call_records,
                complete=False,
            )
            terminal_manifest["terminal_semantic_failure"] = marker
            terminal_failure = OpenRouterStructuredContinuationError(
                "Promoted structured document is complete JSON but violates "
                f"its schema: {exc}",
                result=current,
                manifest=terminal_manifest,
            )
            complete = False
        else:
            complete = True

    if terminal_failure is not None:
        failure_kind = terminal_failure.manifest[
            "terminal_semantic_failure"
        ]["failure_kind"]
        failure_label = (
            "non_json_failure"
            if failure_kind == "complete_non_json_document"
            else "contract_failure"
        )
        limit_suffix = (
            "_with_limit_signal"
            if current.transport.get("output_limited")
            else ""
        )
        if expected_sequence == 0:
            ledger.parts[0]["kind"] = (
                f"initial_complete_{failure_label}{limit_suffix}"
            )
        else:
            ledger.parts[-1]["kind"] = (
                "literal_continuation_"
                f"{failure_label}{limit_suffix}"
            )
    elif expected_sequence == 0:
        ledger.parts[0]["kind"] = (
            "initial_complete_with_limit_signal"
            if complete and current.transport.get("output_limited")
            else "initial_complete"
            if complete
            else "initial_partial"
        )
    elif complete and current.transport.get("output_limited"):
        ledger.parts[-1]["kind"] = (
            "literal_continuation_complete_with_limit_signal"
        )
    checkpoint = _structured_checkpoint_event(
        ledger=ledger,
        call_records=call_records,
        base_result=initial,
        sequence=expected_sequence,
        status=(
            "failed"
            if terminal_failure is not None
            else "completed"
            if complete
            else "partial"
        ),
        complete=complete,
        resume_contract=contract,
        error=terminal_failure,
    )
    checkpoint["promoted_from_provider_event_id"] = provider_event.get(
        "event_id"
    )
    return checkpoint


def _continued_result(
    *,
    initial: ChatResult,
    latest: ChatResult,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    parsed: dict[str, Any] | list[Any],
) -> ChatResult:
    manifest = ledger.manifest(complete=True)
    manifest["calls"] = [dict(record) for record in call_records]
    transport = {
        **dict(latest.transport),
        "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
        "output_complete": True,
        "output_limited": False,
        "output_incomplete_reason": None,
        "initial_output_limited": bool(initial.transport.get("output_limited")),
        "continuation_count": ledger.continuation_count,
        "document_sha256": manifest["document_sha256"],
    }
    usage = _aggregate_continuation_usage(initial.usage, call_records)
    usage["_aiv_structured_continuation"] = manifest
    usage["_aiv_transport"] = transport
    return replace(
        initial,
        text=ledger.text,
        parsed=parsed,
        usage=usage,
        transport=transport,
    )


async def _raise_structured_continuation_error(
    message: str,
    *,
    result: ChatResult,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    rejection: dict[str, Any] | None = None,
    audit_checkpoint: AuditCheckpoint | None = None,
    resume_contract: dict[str, Any] | None = None,
    terminal_semantic_failure_schema: dict[str, Any] | None = None,
    terminal_non_json_failure_schema: dict[str, Any] | None = None,
    terminal_rejected_part_failure_schema: dict[str, Any] | None = None,
) -> None:
    manifest = _structured_manifest_with_calls(
        ledger,
        call_records=call_records,
        complete=False,
    )
    if rejection is not None:
        manifest["rejected_part"] = dict(rejection)
    terminal_failure_kinds = sum(
        candidate is not None
        for candidate in (
            terminal_semantic_failure_schema,
            terminal_non_json_failure_schema,
            terminal_rejected_part_failure_schema,
        )
    )
    if terminal_failure_kinds > 1:
        raise OpenRouterError("Terminal semantic failure kinds are ambiguous")
    if terminal_semantic_failure_schema is not None:
        manifest["terminal_semantic_failure"] = (
            _structured_terminal_schema_failure_marker(
                ledger,
                terminal_semantic_failure_schema,
            )
        )
    elif terminal_non_json_failure_schema is not None:
        manifest["terminal_semantic_failure"] = (
            _structured_terminal_non_json_failure_marker(
                ledger,
                terminal_non_json_failure_schema,
                result.transport,
            )
        )
    elif terminal_rejected_part_failure_schema is not None:
        if not isinstance(rejection, dict):
            raise OpenRouterError(
                "Terminal rejected-part failure requires rejection evidence"
            )
        manifest["terminal_semantic_failure"] = (
            _structured_terminal_rejected_part_failure_marker(
                ledger,
                terminal_rejected_part_failure_schema,
                rejection,
            )
        )
    usage = _aggregate_continuation_usage(
        result.usage,
        [*call_records, *([rejection] if rejection is not None else [])],
    )
    usage["_aiv_structured_continuation"] = manifest
    failed_result = replace(
        result,
        # Keep the actual rejected provider response at the top level.  The
        # accepted prefix remains available verbatim in the manifest.
        text=result.text,
        parsed=None,
        usage=usage,
    )
    failure = OpenRouterStructuredContinuationError(
        message,
        result=failed_result,
        manifest=manifest,
    )
    await _emit_structured_checkpoint(
        audit_checkpoint,
        ledger=ledger,
        call_records=call_records,
        base_result=failed_result,
        sequence=ledger.continuation_count,
        status="failed",
        complete=False,
        resume_contract=dict(resume_contract or {}),
        error=failure,
    )
    raise failure


async def _raise_structured_continuation_cancelled(
    message: str,
    *,
    result: ChatResult,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    audit_checkpoint: AuditCheckpoint | None,
    resume_contract: dict[str, Any],
) -> None:
    manifest = _structured_manifest_with_calls(
        ledger,
        call_records=call_records,
        complete=False,
    )
    cancellation = OpenRouterStructuredContinuationCancelled(
        message,
        result=result,
        manifest=manifest,
    )
    try:
        await _emit_structured_checkpoint(
            audit_checkpoint,
            ledger=ledger,
            call_records=call_records,
            base_result=result,
            sequence=ledger.continuation_count,
            status="cancelled",
            complete=False,
            resume_contract=resume_contract,
            error=cancellation,
        )
    except asyncio.CancelledError:
        pass
    raise cancellation


async def _structured_composable_failover(
    *,
    callback: Callable[[dict[str, Any]], Awaitable[ChatResult]],
    model: str,
    messages: list[dict[str, Any]],
    schema_name: str,
    response_schema: dict[str, Any],
    document_id: str,
    ledger: StructuredContinuationLedger,
    call_records: list[dict[str, Any]],
    latest: ChatResult,
    audit_checkpoint: AuditCheckpoint | None,
    resume_contract: dict[str, Any],
    cause: OpenRouterContextHeadroomError,
) -> ChatResult:
    """Switch a context-exhausted prefix to a caller-owned shard plan.

    The accepted literal prefix is retained only as immutable audit evidence;
    it is never treated as a complete document or concatenated with an
    unrelated generation.  The callback must re-run the original source as
    independently schema-valid, code-owned shards and attest exact coverage.
    """

    manifest = _structured_manifest_with_calls(
        ledger,
        call_records=call_records,
        complete=False,
    )
    await _emit_structured_checkpoint(
        audit_checkpoint,
        ledger=ledger,
        call_records=call_records,
        base_result=latest,
        sequence=ledger.continuation_count,
        status="replanning_to_composable_shards",
        complete=False,
        resume_contract=resume_contract,
        error=cause,
    )
    request = {
        "version": "aiv-structured-composable-failover-v1",
        "reason": "prefix_context_exhausted",
        "model": model,
        "schema_name": schema_name,
        "document_id": document_id,
        "response_schema": copy.deepcopy(response_schema),
        "original_messages": copy.deepcopy(messages),
        "abandoned_prefix": {
            "document_sha256": manifest["document_sha256"],
            "document_chars": manifest["document_chars"],
            "document_utf8_bytes": manifest["document_utf8_bytes"],
            "part_count": manifest["part_count"],
            "continuation_count": manifest["continuation_count"],
        },
    }
    try:
        failover = await callback(request)
    except Exception as exc:
        result = _synthetic_failure_result(
            model=model,
            error=exc,
            fallback=latest,
        )
        await _raise_structured_continuation_error(
            "Composable structured failover failed after prefix context "
            f"exhaustion: {exc}",
            result=result,
            ledger=ledger,
            call_records=call_records,
            audit_checkpoint=audit_checkpoint,
            resume_contract=resume_contract,
        )
    if not isinstance(failover, ChatResult):
        result = _synthetic_failure_result(
            model=model,
            error=OpenRouterError(
                "Composable structured failover returned a non-ChatResult"
            ),
            fallback=latest,
        )
        await _raise_structured_continuation_error(
            "Composable structured failover returned a non-ChatResult",
            result=result,
            ledger=ledger,
            call_records=call_records,
            audit_checkpoint=audit_checkpoint,
            resume_contract=resume_contract,
        )
    sharded_manifest = (
        failover.usage.get("_aiv_sharded_document")
        if isinstance(failover.usage, dict)
        else None
    )
    try:
        Draft202012Validator(response_schema).validate(failover.parsed)
    except ValidationError as exc:
        await _raise_structured_continuation_error(
            "Composable structured failover violated the root schema: "
            f"{exc}",
            result=replace(failover, parsed=None),
            ledger=ledger,
            call_records=call_records,
            audit_checkpoint=audit_checkpoint,
            resume_contract=resume_contract,
        )
    if (
        not isinstance(sharded_manifest, dict)
        or sharded_manifest.get("complete") is not True
        or sharded_manifest.get("coverage_complete") is not True
        or sharded_manifest.get("response_mode")
        != ResponseMode.PARTITIONED.value
    ):
        await _raise_structured_continuation_error(
            "Composable structured failover has no complete partition "
            "coverage attestation",
            result=replace(failover, parsed=None),
            ledger=ledger,
            call_records=call_records,
            audit_checkpoint=audit_checkpoint,
            resume_contract=resume_contract,
        )
    usage = dict(failover.usage)
    usage["_aiv_abandoned_structured_prefix"] = {
        **request["abandoned_prefix"],
        "reason": request["reason"],
    }
    return replace(
        failover,
        usage=usage,
        transport={
            **dict(failover.transport),
            "output_complete": True,
            "output_limited": False,
            "continuation_replanned": True,
        },
    )


async def chat_continuable_structured(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    schema_name: str = "aiv_continuable_document",
    reasoning_effort: str | None = None,
    temperature: float = 0.2,
    document_id: str | None = None,
    overlap_chars: int = DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
    retry_transport_errors: bool = False,
    audit_checkpoint: AuditCheckpoint | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
    composable_failover: Callable[
        [dict[str, Any]], Awaitable[ChatResult]
    ]
    | None = None,
) -> ChatResult:
    """Generate one losslessly continued, schema-validated JSON document.

    This is an explicit derived-document primitive.  It always forbids web
    access and always asks for the model's advertised maximum output.  Do not
    use it for panel observations, measurements, critic verdicts, or any other
    atomic answer: those calls must remain one provider response.

    There is no local continuation-count, aggregate-time, document-length, or
    minimum-part-size ceiling.  Every overlap-proven fragment advances the
    document.  Exact-overlap, duplicate-response and zero-progress checks stop
    broken/paid loops; the transport's configured read timeout stops an
    inactive provider POST independently of document length.  Full-prefix
    replay is nevertheless bounded by the provider's physical input context:
    this primitive can cross output-turn ceilings, but it cannot assemble a
    document larger than the context needed for the original messages, the
    accepted prefix and the next output.  Larger composable documents require
    independently schema-valid shards and deterministic code-owned assembly.
    A caller that owns such a schema-specific plan may pass
    ``composable_failover``; after prefix-context exhaustion the exact accepted
    prefix is checkpointed and the callback must return a complete,
    coverage-attested partitioned result.  The callback is never used for a
    malformed boundary or ordinary transport error.
    atomic observations and verdicts must remain one turn.  Every guard rejects
    incomplete JSON and exposes the exact partial ledger.
    ``resume_checkpoint`` is accepted only after rebuilding its entire
    raw-fragment hash chain and matching model/input/schema digests.
    """

    if not isinstance(response_schema, dict) or not response_schema:
        raise OpenRouterError("response_schema must be a non-empty JSON object")
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, int)
        or overlap_chars < 1
    ):
        raise OpenRouterError("overlap_chars must be a positive integer")

    resolved_document_id = document_id or (
        "structured:"
        + _stable_sha256(
            {
                "model": model,
                "messages": messages,
                "schema_name": schema_name,
                "response_schema": response_schema,
            }
        )[:32]
    )
    resume_contract = structured_resume_contract(
        model=model,
        messages=messages,
        schema_name=schema_name,
        response_schema=response_schema,
        document_id=resolved_document_id,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        overlap_chars=overlap_chars,
    )
    # The contract is the canonical generation identity.  Use its normalized
    # value for every physical POST too; otherwise an input such as ``" high "``
    # would be persisted as contract ``"high"`` but sent to the provider with
    # whitespace, making the paid receipt impossible to promote after a crash.
    resolved_reasoning_effort = resume_contract["reasoning_effort"]
    delta_audit = _structured_delta_audit_enabled(audit_checkpoint)
    call_records: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        resume_manifest = resume_checkpoint.get("manifest")
        if (
            isinstance(resume_manifest, dict)
            and resume_manifest.get("complete") is True
        ):
            return restore_completed_structured_checkpoint(
                resume_checkpoint,
                model,
                messages,
                schema_name,
                response_schema,
                resolved_document_id,
                overlap_chars=overlap_chars,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
        rejected_terminal = _resume_terminal_rejected_checkpoint(
            resume_checkpoint,
            expected_contract=resume_contract,
            overlap_chars=overlap_chars,
            response_schema=response_schema,
        )
        if rejected_terminal is not None:
            raise rejected_terminal
        ledger, call_records, initial = _resume_structured_state(
            resume_checkpoint,
            expected_contract=resume_contract,
            overlap_chars=overlap_chars,
        )
        last_call = call_records[-1]
        latest = replace(
            initial,
            text=str(last_call.get("raw_text") or ""),
            usage=dict(last_call.get("usage") or {}),
            transport=dict(last_call.get("transport") or {}),
            request_policy=dict(last_call.get("request_policy") or {}),
            web_attestation=dict(last_call.get("web_attestation") or {}),
        )
        resume_error = resume_checkpoint.get("error")
        durable_terminal_marker = (
            resume_error.get("terminal_semantic_failure")
            if isinstance(resume_error, dict)
            else None
        )
        if durable_terminal_marker is not None:
            terminal_manifest = _structured_manifest_with_calls(
                ledger,
                call_records=call_records,
                complete=False,
            )
            terminal_manifest["terminal_semantic_failure"] = copy.deepcopy(
                durable_terminal_marker
            )
            if (
                _validated_structured_terminal_semantic_failure(
                    terminal_manifest,
                    durable_terminal_marker,
                    response_schema=response_schema,
                )
                is None
            ):
                raise OpenRouterError(
                    "Durable terminal structured semantic marker is invalid"
                )
            terminal_usage = _aggregate_continuation_usage(
                initial.usage,
                call_records,
            )
            terminal_usage["_aiv_structured_continuation"] = terminal_manifest
            terminal_usage["_aiv_transport"] = dict(latest.transport)
            raise OpenRouterStructuredContinuationError(
                "Durable structured document is complete JSON but violates "
                "its schema",
                result=replace(
                    latest,
                    parsed=None,
                    usage=terminal_usage,
                ),
                manifest=terminal_manifest,
            )
        try:
            parsed = _parse_strict_json_document(ledger.text)
            Draft202012Validator(response_schema).validate(parsed)
        except (OpenRouterError, ValidationError):
            pass
        else:
            await _emit_structured_checkpoint(
                audit_checkpoint,
                ledger=ledger,
                call_records=call_records,
                base_result=initial,
                sequence=ledger.continuation_count,
                status="completed_from_resume",
                complete=True,
                resume_contract=resume_contract,
            )
            return _continued_result(
                initial=initial,
                latest=latest,
                ledger=ledger,
                call_records=call_records,
                parsed=parsed,
            )
        sequence = ledger.continuation_count + 1
    else:
        if delta_audit:
            pending_audit_context = {
                "document_id": resolved_document_id,
                "sequence": 0,
                "predecessor": {
                    "document_id": resolved_document_id,
                    "expected_sequence": 0,
                    "latest_sequence": -1,
                    "complete": False,
                    "document_sha256": text_sha256(""),
                    "document_chars": 0,
                    "document_utf8_bytes": 0,
                    "expected_overlap_sha256": text_sha256(""),
                    "expected_overlap_chars": 0,
                },
                "resume_contract": resume_contract,
            }
        else:
            pending_audit_context = {
                "document_id": resolved_document_id,
                "sequence": 0,
                "partial_text": "",
                "manifest": {
                    "version": STRUCTURED_CONTINUATION_VERSION,
                    "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
                    "document_id": resolved_document_id,
                    "complete": False,
                    "part_count": 0,
                    "calls": [],
                    "accepted_document_text": "",
                },
                "aggregate_usage": {},
                "call_records": [],
                "resume_contract": resume_contract,
            }
        try:
            initial = await chat(
                model=model,
                messages=messages,
                response_schema=response_schema,
                schema_name=schema_name,
                web_policy=WebSearchPolicy.FORBIDDEN,
                reasoning_effort=resolved_reasoning_effort,
                output_token_policy=OutputTokenPolicy.MODEL_MAX,
                temperature=temperature,
                retry_response_contract_errors=False,
                retry_transport_errors=retry_transport_errors,
                audit_checkpoint=audit_checkpoint,
                audit_context=pending_audit_context,
            )
        except OpenRouterOutputLimitError as exc:
            initial = exc.result
            if not initial.text or not initial.web_attestation.get(
                "metric_eligible"
            ):
                empty_ledger = StructuredContinuationLedger(
                    document_id=resolved_document_id,
                    text="",
                    overlap_chars=overlap_chars,
                )
                rejection = _continuation_call_record(initial, sequence=0)
                await _raise_structured_continuation_error(
                    "Initial structured response hit its physical limit "
                    "without a usable JSON prefix",
                    result=initial,
                    ledger=empty_ledger,
                    call_records=[],
                    rejection=rejection,
                    audit_checkpoint=audit_checkpoint,
                    resume_contract=resume_contract,
                )
            try:
                ledger = StructuredContinuationLedger(
                    document_id=resolved_document_id,
                    text=initial.text,
                    overlap_chars=overlap_chars,
                )
            except ValueError as ledger_error:
                empty_ledger = StructuredContinuationLedger(
                    document_id=resolved_document_id,
                    text="",
                    overlap_chars=overlap_chars,
                )
                rejection = _continuation_call_record(initial, sequence=0)
                await _raise_structured_continuation_error(
                    f"Initial structured prefix is not continuable: {ledger_error}",
                    result=initial,
                    ledger=empty_ledger,
                    call_records=[],
                    rejection=rejection,
                    audit_checkpoint=audit_checkpoint,
                    resume_contract=resume_contract,
                )
            call_records.append(_continuation_call_record(initial, sequence=0))
            try:
                parsed = _parse_strict_json_document(ledger.text)
            except OpenRouterError:
                await _emit_structured_checkpoint(
                    audit_checkpoint,
                    ledger=ledger,
                    call_records=call_records,
                    base_result=initial,
                    sequence=0,
                    status="partial",
                    complete=False,
                    resume_contract=resume_contract,
                )
            else:
                try:
                    Draft202012Validator(response_schema).validate(parsed)
                except ValidationError as exc:
                    await _raise_structured_continuation_error(
                        "Initial structured response is complete JSON but "
                        f"violates its schema and cannot be repaired by "
                        f"appending: {exc}",
                        result=initial,
                        ledger=ledger,
                        call_records=call_records,
                        audit_checkpoint=audit_checkpoint,
                        resume_contract=resume_contract,
                        terminal_semantic_failure_schema=response_schema,
                    )
                ledger.parts[0]["kind"] = "initial_complete_with_limit_signal"
                await _emit_structured_checkpoint(
                    audit_checkpoint,
                    ledger=ledger,
                    call_records=call_records,
                    base_result=initial,
                    sequence=0,
                    status="completed",
                    complete=True,
                    resume_contract=resume_contract,
                )
                return _continued_result(
                    initial=initial,
                    latest=initial,
                    ledger=ledger,
                    call_records=call_records,
                    parsed=parsed,
                )
        except asyncio.CancelledError as exc:
            result = _synthetic_failure_result(model=model, error=exc)
            cancelled_ledger = StructuredContinuationLedger(
                document_id=resolved_document_id,
                text="",
                overlap_chars=overlap_chars,
            )
            cancelled_calls: list[dict[str, Any]] = []
            if result.text and result.web_attestation.get("metric_eligible"):
                try:
                    accepted_ledger = StructuredContinuationLedger(
                        document_id=resolved_document_id,
                        text=result.text,
                        overlap_chars=overlap_chars,
                    )
                except ValueError:
                    pass
                else:
                    accepted_ledger.parts[0]["kind"] = (
                        "initial_cancelled_after_checkpoint"
                    )
                    cancelled_ledger = accepted_ledger
                    cancelled_calls.append(
                        _continuation_call_record(result, sequence=0)
                    )
            await _raise_structured_continuation_cancelled(
                "Initial structured provider POST was cancelled",
                result=result,
                ledger=cancelled_ledger,
                call_records=cancelled_calls,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
            )
        except Exception as exc:
            result = _synthetic_failure_result(model=model, error=exc)
            if isinstance(exc, OpenRouterResponseContractError):
                if (
                    result.text == ""
                    and result.transport.get("output_complete") is True
                    and result.transport.get("output_limited") is False
                ):
                    empty_ledger = StructuredContinuationLedger(
                        document_id=resolved_document_id,
                        text="",
                        overlap_chars=overlap_chars,
                    )
                    rejection = _continuation_call_record(
                        result,
                        sequence=0,
                    )
                    await _raise_structured_continuation_error(
                        "Initial complete structured response is empty",
                        result=result,
                        ledger=empty_ledger,
                        call_records=[],
                        rejection=rejection,
                        audit_checkpoint=audit_checkpoint,
                        resume_contract=resume_contract,
                        terminal_rejected_part_failure_schema=response_schema,
                    )
                try:
                    terminal_ledger = StructuredContinuationLedger(
                        document_id=resolved_document_id,
                        text=result.text,
                        overlap_chars=overlap_chars,
                    )
                except ValueError as boundary_error:
                    if (
                        result.transport.get("output_complete") is True
                        and result.transport.get("output_limited") is False
                    ):
                        empty_ledger = StructuredContinuationLedger(
                            document_id=resolved_document_id,
                            text="",
                            overlap_chars=overlap_chars,
                        )
                        rejection = _continuation_call_record(
                            result,
                            sequence=0,
                        )
                        await _raise_structured_continuation_error(
                            "Initial complete structured response violates the "
                            f"JSON boundary: {boundary_error}",
                            result=result,
                            ledger=empty_ledger,
                            call_records=[],
                            rejection=rejection,
                            audit_checkpoint=audit_checkpoint,
                            resume_contract=resume_contract,
                            terminal_rejected_part_failure_schema=(
                                response_schema
                            ),
                        )
                else:
                    terminal_calls = [
                        _continuation_call_record(result, sequence=0)
                    ]
                    try:
                        terminal_parsed = _parse_strict_json_document(
                            terminal_ledger.text
                        )
                    except OpenRouterError as parse_error:
                        if (
                            result.transport.get("output_complete") is True
                            and result.transport.get("output_limited") is False
                        ):
                            terminal_ledger.parts[0]["kind"] = (
                                "initial_complete_non_json_failure"
                            )
                            await _raise_structured_continuation_error(
                                "Initial structured response reached a complete "
                                "turn but is not a JSON document: "
                                f"{parse_error}",
                                result=result,
                                ledger=terminal_ledger,
                                call_records=terminal_calls,
                                audit_checkpoint=audit_checkpoint,
                                resume_contract=resume_contract,
                                terminal_non_json_failure_schema=response_schema,
                            )
                    else:
                        try:
                            Draft202012Validator(response_schema).validate(
                                terminal_parsed
                            )
                        except ValidationError as schema_error:
                            terminal_ledger.parts[0]["kind"] = (
                                "initial_complete_contract_failure"
                            )
                            await _raise_structured_continuation_error(
                                "Initial structured response is complete JSON but "
                                "violates its schema: "
                                f"{schema_error}",
                                result=result,
                                ledger=terminal_ledger,
                                call_records=terminal_calls,
                                audit_checkpoint=audit_checkpoint,
                                resume_contract=resume_contract,
                                terminal_semantic_failure_schema=response_schema,
                            )
            empty_ledger = StructuredContinuationLedger(
                document_id=resolved_document_id,
                text="",
                overlap_chars=overlap_chars,
            )
            await _raise_structured_continuation_error(
                f"Initial structured provider POST failed: {exc}",
                result=result,
                ledger=empty_ledger,
                call_records=[],
                audit_checkpoint=(
                    None
                    if isinstance(exc, OpenRouterAuditCheckpointError)
                    else audit_checkpoint
                ),
                resume_contract=resume_contract,
            )
        else:
            ledger = StructuredContinuationLedger(
                document_id=resolved_document_id,
                text=initial.text,
                overlap_chars=overlap_chars,
            )
            ledger.parts[0]["kind"] = "initial_complete"
            call_records.append(_continuation_call_record(initial, sequence=0))
            try:
                parsed = _parse_strict_json_document(ledger.text)
            except OpenRouterError as exc:
                await _raise_structured_continuation_error(
                    f"Complete structured document is unusable: {exc}",
                    result=initial,
                    ledger=ledger,
                    call_records=call_records,
                    audit_checkpoint=audit_checkpoint,
                    resume_contract=resume_contract,
                    terminal_non_json_failure_schema=response_schema,
                )
            try:
                Draft202012Validator(response_schema).validate(parsed)
            except ValidationError as exc:
                await _raise_structured_continuation_error(
                    f"Complete structured document violates its schema: {exc}",
                    result=initial,
                    ledger=ledger,
                    call_records=call_records,
                    audit_checkpoint=audit_checkpoint,
                    resume_contract=resume_contract,
                    terminal_semantic_failure_schema=response_schema,
                )
            await _emit_structured_checkpoint(
                audit_checkpoint,
                ledger=ledger,
                call_records=call_records,
                base_result=initial,
                sequence=0,
                status="completed",
                complete=True,
                resume_contract=resume_contract,
            )
            return _continued_result(
                initial=initial,
                latest=initial,
                ledger=ledger,
                call_records=call_records,
                parsed=parsed,
            )

        latest = initial
        sequence = 1

    while True:
        try:
            cursor = ledger.cursor()
        except ValueError as exc:
            await _raise_structured_continuation_error(
                f"Structured continuation cursor is invalid: {exc}",
                result=latest,
                ledger=ledger,
                call_records=call_records,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
            )
        continuation_messages = _continuation_messages(
            messages=messages,
            document_text=ledger.text,
            cursor=cursor,
            response_schema=response_schema,
        )
        continuation_audit_context = _structured_audit_context(
            ledger=ledger,
            call_records=call_records,
            base_result=initial,
            sequence=sequence,
            resume_contract=resume_contract,
            audit_checkpoint=audit_checkpoint,
            cursor=cursor,
        )
        try:
            latest = await chat(
                model=model,
                messages=continuation_messages,
                web_policy=WebSearchPolicy.FORBIDDEN,
                reasoning_effort=resolved_reasoning_effort,
                output_token_policy=OutputTokenPolicy.MODEL_MAX,
                temperature=temperature,
                accept_output_limited=True,
                retry_response_contract_errors=False,
                retry_transport_errors=retry_transport_errors,
                audit_checkpoint=audit_checkpoint,
                audit_context=continuation_audit_context,
            )
        except OpenRouterResponseContractError as exc:
            rejected = _continuation_call_record(exc.result, sequence=sequence)
            await _raise_structured_continuation_error(
                f"Structured continuation call failed: {exc}",
                result=exc.result,
                ledger=ledger,
                call_records=call_records,
                rejection=rejected,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
                terminal_rejected_part_failure_schema=(
                    response_schema
                    if exc.result.text == ""
                    and exc.result.transport.get("output_complete") is True
                    and exc.result.transport.get("output_limited") is False
                    else None
                ),
            )
        except asyncio.CancelledError as exc:
            result = _synthetic_failure_result(
                model=model,
                error=exc,
                fallback=latest,
            )
            attached = getattr(exc, "result", None)
            if (
                isinstance(attached, ChatResult)
                and attached.text
                and attached.web_attestation.get("metric_eligible")
            ):
                accepted_call = _continuation_call_record(
                    attached,
                    sequence=sequence,
                )
                try:
                    ledger.append(attached.text, sequence=sequence)
                except ValueError:
                    pass
                else:
                    call_records.append(accepted_call)
                    result = attached
            await _raise_structured_continuation_cancelled(
                f"Structured continuation sequence {sequence} was cancelled",
                result=result,
                ledger=ledger,
                call_records=call_records,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
            )
        except Exception as exc:
            if (
                isinstance(exc, OpenRouterContextHeadroomError)
                and composable_failover is not None
            ):
                return await _structured_composable_failover(
                    callback=composable_failover,
                    model=model,
                    messages=messages,
                    schema_name=schema_name,
                    response_schema=response_schema,
                    document_id=resolved_document_id,
                    ledger=ledger,
                    call_records=call_records,
                    latest=latest,
                    audit_checkpoint=audit_checkpoint,
                    resume_contract=resume_contract,
                    cause=exc,
                )
            result = _synthetic_failure_result(
                model=model,
                error=exc,
                fallback=latest,
            )
            attached = getattr(exc, "result", None)
            rejection = (
                _continuation_call_record(attached, sequence=sequence)
                if isinstance(attached, ChatResult)
                else None
            )
            await _raise_structured_continuation_error(
                f"Structured continuation sequence {sequence} failed: {exc}",
                result=result,
                ledger=ledger,
                call_records=call_records,
                rejection=rejection,
                audit_checkpoint=(
                    None
                    if isinstance(exc, OpenRouterAuditCheckpointError)
                    else audit_checkpoint
                ),
                resume_contract=resume_contract,
            )
        call_record = _continuation_call_record(latest, sequence=sequence)
        try:
            ledger.append(latest.text, sequence=sequence)
        except ValueError as exc:
            await _raise_structured_continuation_error(
                f"Structured continuation boundary failed: {exc}",
                result=latest,
                ledger=ledger,
                call_records=call_records,
                rejection=call_record,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
                terminal_rejected_part_failure_schema=(
                    response_schema
                    if latest.transport.get("output_complete") is True
                    and latest.transport.get("output_limited") is False
                    else None
                ),
            )
        call_records.append(call_record)
        if latest.transport.get("output_limited"):
            try:
                parsed = _parse_strict_json_document(ledger.text)
            except OpenRouterError:
                parsed = None
            else:
                try:
                    Draft202012Validator(response_schema).validate(parsed)
                except ValidationError as exc:
                    await _raise_structured_continuation_error(
                        "Assembled structured response is complete JSON but "
                        "violates its schema and cannot be repaired by "
                        f"appending: {exc}",
                        result=latest,
                        ledger=ledger,
                        call_records=call_records,
                        audit_checkpoint=audit_checkpoint,
                        resume_contract=resume_contract,
                        terminal_semantic_failure_schema=response_schema,
                    )
                ledger.parts[-1]["kind"] = (
                    "literal_continuation_complete_with_limit_signal"
                )
                await _emit_structured_checkpoint(
                    audit_checkpoint,
                    ledger=ledger,
                    call_records=call_records,
                    base_result=initial,
                    sequence=sequence,
                    status="completed",
                    complete=True,
                    resume_contract=resume_contract,
                )
                return _continued_result(
                    initial=initial,
                    latest=latest,
                    ledger=ledger,
                    call_records=call_records,
                    parsed=parsed,
                )
            try:
                await _emit_structured_checkpoint(
                    audit_checkpoint,
                    ledger=ledger,
                    call_records=call_records,
                    base_result=initial,
                    sequence=sequence,
                    status="partial",
                    complete=False,
                    resume_contract=resume_contract,
                )
            except OpenRouterAuditCheckpointError as exc:
                await _raise_structured_continuation_error(
                    f"Structured continuation checkpoint failed: {exc}",
                    result=latest,
                    ledger=ledger,
                    call_records=call_records,
                    audit_checkpoint=None,
                    resume_contract=resume_contract,
                )
            sequence += 1
            continue
        try:
            parsed = _parse_strict_json_document(ledger.text)
        except OpenRouterError as exc:
            await _raise_structured_continuation_error(
                f"Assembled structured document is unusable: {exc}",
                result=latest,
                ledger=ledger,
                call_records=call_records,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
                terminal_non_json_failure_schema=response_schema,
            )
        try:
            Draft202012Validator(response_schema).validate(parsed)
        except ValidationError as exc:
            await _raise_structured_continuation_error(
                f"Assembled structured document violates its schema: {exc}",
                result=latest,
                ledger=ledger,
                call_records=call_records,
                audit_checkpoint=audit_checkpoint,
                resume_contract=resume_contract,
                terminal_semantic_failure_schema=response_schema,
            )
        await _emit_structured_checkpoint(
            audit_checkpoint,
            ledger=ledger,
            call_records=call_records,
            base_result=initial,
            sequence=sequence,
            status="completed",
            complete=True,
            resume_contract=resume_contract,
        )
        return _continued_result(
            initial=initial,
            latest=latest,
            ledger=ledger,
            call_records=call_records,
            parsed=parsed,
        )


async def generate_image(
    *,
    prompt: str,
    aspect_ratio: str = "16:9",
    resolution: str = "2K",
) -> ImageResult:
    payload = {
        "model": settings.OPENROUTER_IMAGE_MODEL,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "n": 1,
    }
    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.0, 3.0, 9.0), start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
                response = await client.post(
                    IMAGE_URL,
                    headers=_headers(),
                    json=payload,
                )
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                raise OpenRouterError(_response_error(response))
            if response.status_code >= 400:
                raise OpenRouterError(_response_error(response))
            body = response.json()
            images = body.get("data") or []
            if not images or not images[0].get("b64_json"):
                raise OpenRouterError("Image model returned no image")
            media_type = str(images[0].get("media_type") or "image/png")
            encoded = str(images[0]["b64_json"])
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise OpenRouterError("Image model returned invalid base64") from exc
            if not content or len(content) > 30 * 1024 * 1024:
                raise OpenRouterError("Image payload has an unexpected size")
            extension = {
                "image/jpeg": "jpg",
                "image/webp": "webp",
                "image/svg+xml": "svg",
            }.get(media_type, "png")
            return ImageResult(
                content=content,
                extension=extension,
                media_type=media_type,
                usage=dict(body.get("usage") or {}),
            )
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, OpenRouterError) as exc:
            last_error = exc
            if attempt >= 3:
                break
    raise OpenRouterError(str(last_error or "Image generation failed"))
