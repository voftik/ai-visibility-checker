from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import httpx

from app.config import settings

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGE_URL = "https://openrouter.ai/api/v1/images"


class OpenRouterError(RuntimeError):
    pass


class WebSearchPolicy(StrEnum):
    """Auditable web-access contract for one OpenRouter request."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    NATIVE_REQUIRED = "native_required"


WEB_ATTESTATION_VERSION = "aiv-openrouter-web-attestation-v1"
TRANSPORT_METADATA_VERSION = "aiv-openrouter-transport-v1"


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


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts).strip()
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
                "content": str(source.get("content") or "").strip()[:2000],
            }
        )
    return output


def _annotations(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep compact, JSON-safe response annotations for later provenance."""

    output: list[dict[str, Any]] = []
    raw_annotations = message.get("annotations")
    if not isinstance(raw_annotations, list):
        return output
    for annotation in raw_annotations[:50]:
        if not isinstance(annotation, dict):
            continue
        item: dict[str, Any] = {
            "type": str(annotation.get("type") or "").strip(),
        }
        source = annotation.get("url_citation")
        if isinstance(source, dict):
            citation: dict[str, Any] = {
                "url": str(source.get("url") or "").strip(),
                "title": str(source.get("title") or "").strip()[:500],
                "content": str(source.get("content") or "").strip()[:2000],
            }
            for key in ("start_index", "end_index"):
                value = source.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    citation[key] = value
            item["url_citation"] = citation
        output.append(item)
    return output


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
                            "max_results": 5,
                            "max_total_results": 12,
                            "max_uses": 3,
                            "search_context_size": "low",
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
                "max_tool_calls": 4,
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


def _parse_json(text: str) -> dict[str, Any] | list[Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1]
        if value.endswith("```"):
            value = value[:-3]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        start_object = value.find("{")
        start_array = value.find("[")
        starts = [position for position in (start_object, start_array) if position >= 0]
        if not starts:
            raise OpenRouterError("Model returned invalid JSON") from exc
        start = min(starts)
        end = max(value.rfind("}"), value.rfind("]"))
        if end <= start:
            raise OpenRouterError("Model returned incomplete JSON") from exc
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as nested:
            raise OpenRouterError("Model returned invalid JSON") from nested
    if not isinstance(parsed, (dict, list)):
        raise OpenRouterError("Model returned a JSON scalar")
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


async def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "aiv_result",
    web_policy: WebSearchPolicy | str = WebSearchPolicy.FORBIDDEN,
    reasoning_effort: str | None = None,
    max_tokens: int = 12_000,
    temperature: float = 0.2,
    retry_response_contract_errors: bool = True,
    retry_transport_errors: bool = True,
) -> ChatResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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

    timeout = httpx.Timeout(
        connect=15.0,
        read=float(settings.OPENROUTER_TIMEOUT_SECONDS),
        write=30.0,
        pool=15.0,
    )
    last_error: Exception | None = None
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
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
                headers = _headers()
                headers["X-OpenRouter-Metadata"] = "enabled"
                response = await client.post(
                    CHAT_URL,
                    headers=headers,
                    json=payload,
                )
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                raise OpenRouterError(_response_error(response))
            if response.status_code >= 400:
                raise OpenRouterError(_response_error(response))
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise OpenRouterError("OpenRouter returned no choices")
            choice = choices[0]
            message = choice.get("message") or {}
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
            if transport["output_limited"]:
                raise OpenRouterOutputLimitError(
                    "OpenRouter response hit the output limit "
                    f"(finish_reason={transport['finish_reason'] or '?'}, "
                    "native_finish_reason="
                    f"{transport['native_finish_reason'] or '?'}, "
                    f"max_tokens={max_tokens})",
                    result=result,
                )
            if not transport["output_complete"]:
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
                    result = replace(result, parsed=_parse_json(text))
                except OpenRouterError as exc:
                    raise OpenRouterResponseContractError(
                        f"Structured response is unusable: {exc}",
                        result=result,
                    ) from exc
            return result
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, OpenRouterError) as exc:
            if isinstance(exc, OpenRouterOutputLimitError):
                raise
            if (
                isinstance(exc, OpenRouterResponseContractError)
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
    raise OpenRouterError(str(last_error or "OpenRouter request failed"))


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
