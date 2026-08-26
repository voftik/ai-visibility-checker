"""Linear durable storage for continuable structured OpenRouter calls.

The transport callback emits rich, self-contained events. Keeping those events
verbatim makes persistence quadratic because every continuation carries the
complete accepted prefix and all earlier call records. This module normalizes
that stream on top of the existing ``RunArtifact`` table:

* every physical provider POST is an immutable receipt;
* every accepted sequence is an immutable fragment that references its receipt;
* one compact mutable head records the latest validated checkpoint.

Raw provider text and request/response content are stored once in the receipt.
Fragments and the head contain only hashes, offsets and references. Recovery
rebuilds the checkpoint through ``StructuredContinuationLedger`` and can
promote an accepted receipt written after the last head before buying another
provider turn.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db import SessionLocal
from app.models import RunArtifact
from app.services import openrouter as openrouter_service
from app.services.long_response import (
    DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
    STRUCTURED_CONTINUATION_VERSION,
    ResponseMode,
    StructuredContinuationLedger,
    text_sha256,
)
from app.services.openrouter import AuditCheckpoint, OpenRouterError
from app.services.run_lease import assert_run_lease


STRUCTURED_AUDIT_STORE_VERSION = "aiv-structured-audit-store-v2"
_PHYSICAL_EVENT_VERSION = "aiv-openrouter-physical-post-audit-v1"
_CHECKPOINT_EVENT_VERSION = "aiv-structured-checkpoint-v1"
_PHYSICAL_DELTA_EVENT_VERSION = "aiv-openrouter-physical-post-audit-v2"
_CHECKPOINT_DELTA_EVENT_VERSION = "aiv-structured-checkpoint-v2"
_HEAD_VERSION = "aiv-structured-audit-head-v2"
_FRAGMENT_VERSION = "aiv-structured-audit-fragment-v2"
_RECEIPT_VERSION = "aiv-structured-audit-receipt-v2"
_EVENT_ID_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RAW_TEXT_REF = {"$aiv_ref": "receipt.raw_text"}
_PREDECESSOR_DOCUMENT_REF = "predecessor.document_text"


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validated_resume_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpenRouterError("Structured audit resume contract is missing")
    contract = copy.deepcopy(value)
    digest = str(contract.get("sha256") or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise OpenRouterError("Structured audit resume contract digest is invalid")
    payload = {key: item for key, item in contract.items() if key != "sha256"}
    if _stable_json_sha256(payload) != digest:
        raise OpenRouterError("Structured audit resume contract digest mismatch")
    document_id = str(contract.get("document_id") or "")
    if not document_id:
        raise OpenRouterError("Structured audit resume contract has no document id")
    return contract


def _stream_identity(
    *,
    owner_artifact_key: str,
    source_input: dict[str, Any] | list[Any],
    model: str,
    owner_prompt_version: str,
    document_id: str,
    resume_contract: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    owner = str(owner_artifact_key or "").strip()
    if not owner:
        raise OpenRouterError("Structured audit owner artifact key is empty")
    resolved_model = str(model or "").strip()
    if not resolved_model:
        raise OpenRouterError("Structured audit model is empty")
    prompt_version = str(owner_prompt_version or "").strip()
    if not prompt_version:
        raise OpenRouterError("Structured audit owner prompt version is empty")
    contract = _validated_resume_contract(resume_contract)
    if str(contract.get("document_id") or "") != document_id:
        raise OpenRouterError("Structured audit document id/contract mismatch")
    identity = {
        "version": STRUCTURED_AUDIT_STORE_VERSION,
        "owner_artifact_key": owner,
        "source_input_sha256": _stable_json_sha256(source_input),
        "model": resolved_model,
        "owner_prompt_version": prompt_version,
        "document_id": document_id,
        "resume_contract_sha256": contract["sha256"],
    }
    return identity, _stable_json_sha256(identity)


def _event_stream(
    event: dict[str, Any],
    *,
    owner_artifact_key: str,
    source_input: dict[str, Any] | list[Any],
    model: str,
    owner_prompt_version: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document_id = str(event.get("document_id") or "")
    if not document_id:
        raise OpenRouterError("Structured audit event has no document id")
    contract = _validated_resume_contract(event.get("resume_contract"))
    identity, stream_sha256 = _stream_identity(
        owner_artifact_key=owner_artifact_key,
        source_input=source_input,
        model=model,
        owner_prompt_version=owner_prompt_version,
        document_id=document_id,
        resume_contract=contract,
    )
    event_model = str(event.get("model") or "").strip()
    if event_model and event_model != str(model).strip():
        raise OpenRouterError("Structured audit event model mismatch")
    return identity, stream_sha256, contract


def _validated_event(event: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(event, dict):
        raise OpenRouterError("Structured audit event must be an object")
    version = str(event.get("version") or "")
    kind = str(event.get("event_kind") or "")
    allowed = {
        _PHYSICAL_EVENT_VERSION: "provider_post",
        _CHECKPOINT_EVENT_VERSION: "structured_continuation_checkpoint",
        _PHYSICAL_DELTA_EVENT_VERSION: "provider_post",
        _CHECKPOINT_DELTA_EVENT_VERSION: "structured_continuation_checkpoint",
    }
    if allowed.get(version) != kind:
        raise OpenRouterError("Structured audit event version/kind is invalid")
    event_id = str(event.get("event_id") or "")
    if _EVENT_ID_RE.fullmatch(event_id) is None:
        raise OpenRouterError("Structured audit event id is invalid")
    return event_id, kind


def _sequence(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenRouterError(f"Structured audit {label} sequence is invalid")
    return value


def _artifact_prefix(kind: str, stream_sha256: str) -> str:
    return f"lsa2_{kind}_{stream_sha256}_"


def _head_key(stream_sha256: str) -> str:
    return f"lsa2_head_{stream_sha256}"


def _fragment_key(stream_sha256: str, sequence: int) -> str:
    return f"{_artifact_prefix('fragment', stream_sha256)}{sequence:012d}"


def _receipt_key(
    stream_sha256: str,
    *,
    sequence: int,
    attempt: int,
    event_id: str,
) -> str:
    return (
        f"{_artifact_prefix('receipt', stream_sha256)}"
        f"{sequence:012d}_{attempt:03d}_{event_id}"
    )


def _prefix_bounds(prefix: str) -> tuple[str, str]:
    # Range predicates keep '_' literal; SQL LIKE would treat it as a wildcard.
    return prefix, prefix + "\uffff"


def _replace_raw_text(value: Any, raw_text: str | None) -> Any:
    if raw_text is not None and isinstance(value, str) and value == raw_text:
        return dict(_RAW_TEXT_REF)
    if isinstance(value, list):
        return [_replace_raw_text(item, raw_text) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _replace_raw_text(item, raw_text)
            for key, item in value.items()
        }
    return copy.deepcopy(value)


def _restore_raw_text(value: Any, raw_text: str | None) -> Any:
    if isinstance(value, dict) and value == _RAW_TEXT_REF:
        if raw_text is None:
            raise OpenRouterError("Structured receipt raw-text reference is broken")
        return raw_text
    if isinstance(value, list):
        return [_restore_raw_text(item, raw_text) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _restore_raw_text(item, raw_text)
            for key, item in value.items()
        }
    return copy.deepcopy(value)


def _normalized_usage(value: Any, raw_text: str | None) -> dict[str, Any]:
    usage = copy.deepcopy(value) if isinstance(value, dict) else {}
    # Each result-bearing attempt already has its own immutable receipt.
    # Retaining the cumulative nested attempt ledger here would duplicate all
    # earlier responses on every retry.  The current attempt is normalized
    # separately by ``_call_attempt_metadata``.
    usage.pop("_aiv_call_attempts", None)
    usage.pop("_aiv_structured_continuation", None)
    usage.pop("_aiv_failed_post", None)
    for key in (
        "_aiv_transport",
        "_aiv_request_policy",
        "_aiv_web_attestation",
        "_aiv_router_metadata",
    ):
        usage.pop(key, None)
    return _replace_raw_text(usage, raw_text)


def _call_attempt_metadata(
    event: dict[str, Any],
    *,
    raw_text: str | None,
) -> dict[str, Any] | None:
    """Return only this receipt's attempt metadata, never its raw text."""

    if raw_text is None:
        return None
    usage = event.get("usage")
    attempts = usage.get("_aiv_call_attempts") if isinstance(usage, dict) else None
    event_attempt = event.get("attempt")
    matches: list[dict[str, Any]] = []
    if isinstance(attempts, list):
        for candidate in attempts:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("attempt") != event_attempt:
                continue
            if candidate.get("raw_text") != raw_text:
                continue
            matches.append(candidate)
    if len(matches) > 1:
        raise OpenRouterError("Structured provider attempt ledger is ambiguous")
    if matches:
        metadata = {
            str(key): copy.deepcopy(value)
            for key, value in matches[0].items()
            if key
            not in {
                "raw_text",
                "transport",
                "request_policy",
                "web_attestation",
            }
        }
        nested_usage = metadata.get("usage")
        if isinstance(nested_usage, dict):
            metadata["usage"] = _normalized_usage(nested_usage, raw_text)
        return _replace_raw_text(metadata, raw_text)

    # A first attempt has no earlier usage to separate and can be rebuilt from
    # its top-level fields. Later attempts must carry the code-owned attempt
    # ledger or their per-attempt billing metadata is irrecoverable.
    if event_attempt != 1:
        raise OpenRouterError("Structured provider attempt metadata is missing")
    error = event.get("error")
    return {
        "attempt": 1,
        "status": event.get("status"),
        "text_sha256": text_sha256(raw_text),
        "text_chars": len(raw_text),
        "text_utf8_bytes": len(raw_text.encode("utf-8")),
        "usage": _normalized_usage(event.get("usage"), raw_text),
        "error_type": error.get("type") if isinstance(error, dict) else None,
        "error_message": (
            error.get("message") if isinstance(error, dict) else None
        ),
    }


def _provider_predecessor(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("version") != _PHYSICAL_DELTA_EVENT_VERSION:
        return None
    predecessor = event.get("predecessor")
    if not isinstance(predecessor, dict):
        raise OpenRouterError("Structured delta provider predecessor is missing")
    sequence = _sequence(event.get("sequence"), label="delta provider")
    expected_sequence = predecessor.get("expected_sequence")
    latest_sequence = predecessor.get("latest_sequence")
    if (
        expected_sequence != sequence
        or isinstance(latest_sequence, bool)
        or not isinstance(latest_sequence, int)
        or latest_sequence != sequence - 1
        or predecessor.get("complete") is not False
        or predecessor.get("document_id") != event.get("document_id")
    ):
        raise OpenRouterError("Structured delta provider predecessor is inconsistent")
    for key in (
        "document_chars",
        "expected_overlap_chars",
    ):
        value = predecessor.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OpenRouterError(
                f"Structured delta provider predecessor {key} is invalid"
            )
    utf8_bytes = predecessor.get("document_utf8_bytes")
    if utf8_bytes is not None and (
        isinstance(utf8_bytes, bool)
        or not isinstance(utf8_bytes, int)
        or utf8_bytes < 0
    ):
        raise OpenRouterError(
            "Structured delta provider predecessor document_utf8_bytes is invalid"
        )
    if sequence == 0 and utf8_bytes != 0:
        raise OpenRouterError(
            "Initial structured delta predecessor byte count is invalid"
        )
    for key in ("document_sha256", "expected_overlap_sha256"):
        if _SHA256_RE.fullmatch(str(predecessor.get(key) or "")) is None:
            raise OpenRouterError(
                f"Structured delta provider predecessor {key} is invalid"
            )
    return copy.deepcopy(predecessor)


def _normalized_request_payload(
    event: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Replace a cumulative continuation prefix with a verified reference.

    The provider receives the literal full document for semantic continuity.
    Persisting that growing assistant turn in every receipt would make durable
    storage quadratic, so delta-v2 stores it once through accepted fragments.
    """

    normalized = copy.deepcopy(request_payload)
    if event.get("version") != _PHYSICAL_DELTA_EVENT_VERSION:
        return normalized
    sequence = _sequence(event.get("sequence"), label="delta provider")
    if sequence == 0:
        return normalized
    predecessor = _provider_predecessor(event)
    assert isinstance(predecessor, dict)
    messages = normalized.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise OpenRouterError(
            "Structured continuation request messages are missing"
        )
    assistant = messages[-2]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise OpenRouterError(
            "Structured continuation request predecessor turn is missing"
        )
    document_text = assistant.get("content")
    if not isinstance(document_text, str):
        raise OpenRouterError(
            "Structured continuation request predecessor text is invalid"
        )
    document_bytes = document_text.encode("utf-8")
    if (
        text_sha256(document_text) != predecessor.get("document_sha256")
        or len(document_text) != predecessor.get("document_chars")
        or (
            predecessor.get("document_utf8_bytes") is not None
            and len(document_bytes) != predecessor.get("document_utf8_bytes")
        )
    ):
        raise OpenRouterError(
            "Structured continuation request predecessor does not match "
            "its compact audit state"
        )
    assistant["content"] = {
        "$aiv_ref": _PREDECESSOR_DOCUMENT_REF,
        "document_sha256": text_sha256(document_text),
        "document_chars": len(document_text),
        "document_utf8_bytes": len(document_bytes),
    }
    return normalized


def _materialized_request_payload(
    request_payload: dict[str, Any],
    *,
    source_event_version: str,
    sequence: int,
    original_sha256: str,
    predecessor: dict[str, Any] | None,
    predecessor_document_text: str | None,
) -> dict[str, Any]:
    """Restore a normalized request and verify its original provider digest."""

    materialized = copy.deepcopy(request_payload)
    if source_event_version == _PHYSICAL_DELTA_EVENT_VERSION and sequence > 0:
        messages = materialized.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise OpenRouterError(
                "Stored structured continuation messages are corrupt"
            )
        assistant = messages[-2]
        if (
            not isinstance(assistant, dict)
            or assistant.get("role") != "assistant"
        ):
            raise OpenRouterError(
                "Stored structured continuation predecessor turn is corrupt"
            )
        content = assistant.get("content")
        if isinstance(content, dict) and (
            content.get("$aiv_ref") == _PREDECESSOR_DOCUMENT_REF
        ):
            if not isinstance(predecessor, dict):
                raise OpenRouterError(
                    "Stored continuation request has no predecessor metadata"
                )
            marker_sha256 = str(content.get("document_sha256") or "")
            marker_chars = content.get("document_chars")
            marker_bytes = content.get("document_utf8_bytes")
            if (
                _SHA256_RE.fullmatch(marker_sha256) is None
                or isinstance(marker_chars, bool)
                or not isinstance(marker_chars, int)
                or marker_chars < 0
                or isinstance(marker_bytes, bool)
                or not isinstance(marker_bytes, int)
                or marker_bytes < 0
                or marker_sha256 != predecessor.get("document_sha256")
                or marker_chars != predecessor.get("document_chars")
                or (
                    predecessor.get("document_utf8_bytes") is not None
                    and marker_bytes != predecessor.get("document_utf8_bytes")
                )
            ):
                raise OpenRouterError(
                    "Stored continuation request reference is corrupt"
                )
            if predecessor_document_text is None:
                return materialized
            document_bytes = predecessor_document_text.encode("utf-8")
            if (
                text_sha256(predecessor_document_text) != marker_sha256
                or len(predecessor_document_text) != marker_chars
                or len(document_bytes) != marker_bytes
            ):
                raise OpenRouterError(
                    "Reconstructed continuation predecessor does not match "
                    "its request reference"
                )
            assistant["content"] = predecessor_document_text
        elif isinstance(content, str):
            content_bytes = content.encode("utf-8")
            if (
                not isinstance(predecessor, dict)
                or text_sha256(content) != predecessor.get("document_sha256")
                or len(content) != predecessor.get("document_chars")
                or (
                    predecessor.get("document_utf8_bytes") is not None
                    and len(content_bytes)
                    != predecessor.get("document_utf8_bytes")
                )
                or (
                    predecessor_document_text is not None
                    and content != predecessor_document_text
                )
            ):
                raise OpenRouterError(
                    "Stored literal continuation predecessor is corrupt"
                )
        else:
            raise OpenRouterError(
                "Stored continuation request content is corrupt"
            )
    if (
        predecessor_document_text is not None
        or source_event_version != _PHYSICAL_DELTA_EVENT_VERSION
        or sequence == 0
    ) and _stable_json_sha256(materialized) != original_sha256:
        raise OpenRouterError(
            "Structured provider original request digest mismatch"
        )
    return materialized


def _usage_with_receipt_metadata(
    usage: dict[str, Any],
    output: dict[str, Any],
    *,
    raw_text: str | None,
) -> dict[str, Any]:
    restored = copy.deepcopy(usage)
    metadata_fields = {
        "_aiv_transport": "transport",
        "_aiv_request_policy": "request_policy",
        "_aiv_web_attestation": "web_attestation",
        "_aiv_router_metadata": "router_metadata",
    }
    for usage_key, output_key in metadata_fields.items():
        value = _restore_raw_text(output.get(output_key) or {}, raw_text)
        if not isinstance(value, dict):
            raise OpenRouterError("Structured provider usage metadata is corrupt")
        restored[usage_key] = value
    return restored


def _receipt_output(
    event: dict[str, Any],
    *,
    stream_sha256: str,
    raw_text: str | None,
) -> dict[str, Any]:
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    return {
        "version": _RECEIPT_VERSION,
        "row_kind": "provider_receipt",
        "source_event_version": event.get("version"),
        "stream_sha256": stream_sha256,
        "event_id": event["event_id"],
        "logical_call_id": event.get("logical_call_id"),
        "document_id": event.get("document_id"),
        "sequence": event.get("sequence"),
        "attempt": event.get("attempt"),
        "status": event.get("status"),
        "model": event.get("model"),
        "request_sha256": event.get("request_sha256"),
        "response": _replace_raw_text(event.get("response") or {}, raw_text),
        "raw_text_sha256": text_sha256(raw_text) if raw_text is not None else None,
        "raw_text_chars": len(raw_text) if raw_text is not None else None,
        "raw_text_utf8_bytes": (
            len(raw_text.encode("utf-8")) if raw_text is not None else None
        ),
        "transport": _replace_raw_text(event.get("transport") or {}, raw_text),
        "request_policy": _replace_raw_text(
            usage.get("_aiv_request_policy") or {}, raw_text
        ),
        "web_attestation": _replace_raw_text(
            usage.get("_aiv_web_attestation") or {}, raw_text
        ),
        "router_metadata": _replace_raw_text(
            usage.get("_aiv_router_metadata") or {}, raw_text
        ),
        "call_attempt_metadata": _call_attempt_metadata(
            event,
            raw_text=raw_text,
        ),
        "predecessor": _provider_predecessor(event),
        "error": _replace_raw_text(event.get("error"), raw_text),
    }


async def _persist_provider_receipt(
    run_id: str,
    *,
    stage_key: str,
    model: str,
    identity: dict[str, Any],
    stream_sha256: str,
    contract: dict[str, Any],
    event: dict[str, Any],
) -> None:
    event_id = str(event["event_id"])
    sequence = _sequence(event.get("sequence"), label="provider receipt")
    attempt = event.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise OpenRouterError("Structured audit provider attempt is invalid")
    raw_text = event.get("raw_text")
    if raw_text is not None and not isinstance(raw_text, str):
        raise OpenRouterError("Structured audit provider raw text is invalid")
    request_payload = event.get("request_payload")
    if not isinstance(request_payload, dict):
        raise OpenRouterError("Structured audit provider request is missing")
    request_sha256 = str(event.get("request_sha256") or "")
    if request_sha256 != _stable_json_sha256(request_payload):
        raise OpenRouterError("Structured audit provider request digest mismatch")
    stored_request_payload = _normalized_request_payload(event, request_payload)
    stored_request_sha256 = _stable_json_sha256(stored_request_payload)
    artifact_key = _receipt_key(
        stream_sha256,
        sequence=sequence,
        attempt=attempt,
        event_id=event_id,
    )
    output = _receipt_output(
        event,
        stream_sha256=stream_sha256,
        raw_text=raw_text,
    )
    usage = _normalized_usage(event.get("usage"), raw_text)
    row_digest = _stable_json_sha256(
        {
            "identity": identity,
            "request_payload": stored_request_payload,
            "output": output,
            "raw_text_sha256": output["raw_text_sha256"],
            "usage": usage,
            "resume_contract_sha256": contract["sha256"],
        }
    )
    event_input = {
        "version": _RECEIPT_VERSION,
        "row_kind": "provider_receipt",
        "stream_identity": identity,
        "stream_sha256": stream_sha256,
        "resume_contract_sha256": contract["sha256"],
        "request_payload": stored_request_payload,
        "request_sha256": request_sha256,
        "stored_request_sha256": stored_request_sha256,
        "row_sha256": row_digest,
    }
    error = event.get("error")
    error_message = (
        str(error.get("message"))
        if isinstance(error, dict) and error.get("message") is not None
        else None
    )
    await assert_run_lease(run_id)
    async with SessionLocal() as session:
        inserted = await session.execute(
            sqlite_insert(RunArtifact)
            .values(
                run_id=run_id,
                stage_key=stage_key,
                artifact_key=artifact_key,
                status="completed",
                model=model,
                prompt_version=STRUCTURED_AUDIT_STORE_VERSION,
                input_json=event_input,
                output_json=output,
                raw_text=raw_text,
                usage_json=usage or None,
                error_message=error_message,
            )
            .on_conflict_do_nothing(index_elements=("run_id", "artifact_key"))
        )
        if inserted.rowcount != 1:
            existing = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == artifact_key,
                    )
                )
            ).scalar_one_or_none()
            stored = existing.input_json if existing is not None else None
            if (
                not isinstance(stored, dict)
                or str(stored.get("row_sha256") or "") != row_digest
            ):
                raise OpenRouterError(
                    "Structured provider receipt collision or mutation detected"
                )
        await assert_run_lease(run_id)
        await session.commit()


def _normalized_call_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key
        not in {
            "raw_text",
            "usage",
            "transport",
            "request_policy",
            "web_attestation",
            "prior_transport_attempts",
        }
    }


def _provider_event_is_promotable(event: dict[str, Any]) -> bool:
    raw_text = event.get("raw_text")
    transport = event.get("transport")
    if not isinstance(raw_text, str) or not isinstance(transport, dict):
        return False
    status = event.get("status")
    if status == "accepted":
        return (
            bool(raw_text)
            and (
            transport.get("output_complete") is True
            or transport.get("output_limited") is True
            )
        )
    error = event.get("error")
    response = event.get("response")
    response_body = (
        response.get("body_json") if isinstance(response, dict) else None
    )
    http_status = (
        response.get("http_status") if isinstance(response, dict) else None
    )
    complete_http_receipt = bool(
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 200 <= http_status < 300
        and isinstance(response_body, dict)
        and isinstance(response_body.get("choices"), list)
        and bool(response_body.get("choices"))
        and isinstance(response_body.get("usage"), dict)
        and isinstance(event.get("usage"), dict)
        and isinstance(event.get("request_payload"), dict)
        and isinstance(event.get("request_sha256"), str)
    )
    return bool(
        status == "rejected"
        and isinstance(error, dict)
        and (
            (
                bool(raw_text)
                and
                transport.get("output_limited") is True
                and error.get("type") == "OpenRouterOutputLimitError"
            )
            or (
                complete_http_receipt
                and transport.get("output_complete") is True
                and transport.get("output_limited") is False
                and error.get("type") == "OpenRouterResponseContractError"
            )
        )
    )


def _provider_event_has_response_evidence(event: dict[str, Any]) -> bool:
    """Return whether retrying this sequence could hide a paid provider POST.

    A syntactically unusable 2xx body has no ``ChatResult``/``raw_text`` but
    may still be billable.  Conversely, a transport failure with no provider
    response is ordinary retry evidence and must not block recovery.  Explicit
    usage counters also count as response evidence even on a non-2xx response.
    """

    if isinstance(event.get("raw_text"), str):
        return True
    response = event.get("response")
    if not isinstance(response, dict):
        return False
    http_status = response.get("http_status")
    if (
        isinstance(http_status, int)
        and not isinstance(http_status, bool)
        and 200 <= http_status < 300
    ):
        return True
    body = response.get("body_json")
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return False
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "total_cost",
        "credits",
    ):
        value = usage.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            return True
    return False


def _validate_call_record(record: Any, *, sequence: int) -> tuple[str, str]:
    if not isinstance(record, dict):
        raise OpenRouterError("Structured checkpoint call record is invalid")
    if _sequence(record.get("sequence"), label="call record") != sequence:
        raise OpenRouterError("Structured checkpoint call sequence is not contiguous")
    raw_text = record.get("raw_text")
    if not isinstance(raw_text, str):
        raise OpenRouterError("Structured checkpoint call raw text is missing")
    digest = text_sha256(raw_text)
    if str(record.get("text_sha256") or "") != digest:
        raise OpenRouterError("Structured checkpoint call digest mismatch")
    if record.get("text_chars") != len(raw_text):
        raise OpenRouterError("Structured checkpoint call character count mismatch")
    stored_bytes = record.get("text_utf8_bytes")
    if stored_bytes is not None and stored_bytes != len(raw_text.encode("utf-8")):
        raise OpenRouterError("Structured checkpoint call byte count mismatch")
    return raw_text, digest


def _compact_manifest(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "version",
            "mode",
            "document_id",
            "complete",
            "continuation_count",
            "part_count",
            "document_sha256",
            "document_chars",
            "document_utf8_bytes",
            "json_prefix",
        )
    }


def _checkpoint_projection(
    event: dict[str, Any],
) -> tuple[
    bool,
    int,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Normalize snapshot-v1 or delta-v2 without retaining cumulative data."""

    version = event.get("version")
    if version == _CHECKPOINT_EVENT_VERSION:
        manifest = event.get("manifest")
        calls = event.get("call_records")
        if not isinstance(manifest, dict) or not isinstance(calls, list):
            raise OpenRouterError("Structured checkpoint manifest/calls are invalid")
        complete = manifest.get("complete")
        if not isinstance(complete, bool):
            raise OpenRouterError(
                "Structured checkpoint completion flag is invalid"
            )
        latest_sequence = len(calls) - 1
        event_sequence = event.get("sequence")
        if not calls:
            if (
                complete
                or event_sequence != 0
                or event.get("partial_text") != ""
                or manifest.get("accepted_document_text") != ""
                or manifest.get("document_sha256") != text_sha256("")
                or manifest.get("document_chars") != 0
            ):
                raise OpenRouterError(
                    "Empty structured checkpoint state is inconsistent"
                )
            compact_empty = _compact_manifest(manifest)
            compact_empty.update(
                {
                    "complete": False,
                    "continuation_count": 0,
                    "part_count": 0,
                    "document_sha256": text_sha256(""),
                    "document_chars": 0,
                    "document_utf8_bytes": 0,
                }
            )
            return (
                False,
                -1,
                compact_empty,
                [],
                (
                    copy.deepcopy(event.get("aggregate_usage"))
                    if isinstance(event.get("aggregate_usage"), dict)
                    else None
                ),
            )
        if (
            isinstance(event_sequence, bool)
            or not isinstance(event_sequence, int)
            or event_sequence != latest_sequence
        ):
            raise OpenRouterError(
                "Structured checkpoint sequence/call ledger mismatch"
            )
        parts = manifest.get("parts")
        if not isinstance(parts, list) or len(parts) != len(calls):
            raise OpenRouterError("Structured checkpoint part ledger is invalid")
        fragments: list[dict[str, Any]] = []
        for sequence, record in enumerate(calls):
            if not isinstance(parts[sequence], dict):
                raise OpenRouterError(
                    "Structured checkpoint part metadata is invalid"
                )
            raw_text, raw_digest = _validate_call_record(
                record,
                sequence=sequence,
            )
            fragments.append(
                {
                    "sequence": sequence,
                    "raw_text_sha256": raw_digest,
                    "raw_text_chars": len(raw_text),
                    "raw_text_utf8_bytes": len(raw_text.encode("utf-8")),
                    "part": copy.deepcopy(parts[sequence]),
                    "call_metadata": _normalized_call_metadata(record),
                }
            )
        aggregate_usage = (
            copy.deepcopy(event.get("aggregate_usage"))
            if isinstance(event.get("aggregate_usage"), dict)
            else None
        )
        return (
            complete,
            latest_sequence,
            _compact_manifest(manifest),
            fragments,
            aggregate_usage,
        )

    if version != _CHECKPOINT_DELTA_EVENT_VERSION:
        raise OpenRouterError("Structured checkpoint version is unsupported")
    complete = event.get("complete")
    head = event.get("head")
    accepted = event.get("accepted_fragment")
    if not isinstance(complete, bool) or not isinstance(head, dict):
        raise OpenRouterError("Structured delta checkpoint head is invalid")
    latest_sequence = head.get("latest_sequence")
    event_sequence = event.get("sequence")
    if (
        isinstance(latest_sequence, bool)
        or not isinstance(latest_sequence, int)
        or latest_sequence < -1
        or event_sequence != latest_sequence
        or head.get("complete") is not complete
        or head.get("document_id") != event.get("document_id")
    ):
        raise OpenRouterError("Structured delta checkpoint head is inconsistent")
    part_count = head.get("part_count")
    continuation_count = head.get("continuation_count")
    if (
        isinstance(part_count, bool)
        or not isinstance(part_count, int)
        or isinstance(continuation_count, bool)
        or not isinstance(continuation_count, int)
        or part_count != latest_sequence + 1
        or continuation_count != max(0, latest_sequence)
    ):
        raise OpenRouterError("Structured delta checkpoint counters are invalid")
    if latest_sequence == -1:
        if accepted is not None or complete:
            raise OpenRouterError(
                "Empty structured delta checkpoint has an accepted fragment"
            )
        fragments = []
    else:
        if not isinstance(accepted, dict):
            raise OpenRouterError(
                "Structured delta checkpoint accepted fragment is missing"
            )
        fragment_sequence = accepted.get("sequence")
        digest = str(accepted.get("raw_text_sha256") or "")
        raw_chars = accepted.get("raw_text_chars")
        raw_bytes = accepted.get("raw_text_utf8_bytes")
        part = accepted.get("part")
        metadata = accepted.get("call_metadata")
        if (
            fragment_sequence != latest_sequence
            or _SHA256_RE.fullmatch(digest) is None
            or isinstance(raw_chars, bool)
            or not isinstance(raw_chars, int)
            or raw_chars < 0
            or isinstance(raw_bytes, bool)
            or not isinstance(raw_bytes, int)
            or raw_bytes < 0
            or not isinstance(part, dict)
            or not isinstance(metadata, dict)
        ):
            raise OpenRouterError(
                "Structured delta checkpoint accepted fragment is invalid"
            )
        expected_fragment_fields = {
            "sequence": latest_sequence,
            "text_sha256": digest,
            "text_chars": raw_chars,
            "text_utf8_bytes": raw_bytes,
        }
        if any(
            metadata.get(key) != value
            for key, value in expected_fragment_fields.items()
        ) or any(
            part.get(key) != value
            for key, value in {
                "sequence": latest_sequence,
                "response_sha256": digest,
                "response_chars": raw_chars,
                "response_utf8_bytes": raw_bytes,
            }.items()
        ):
            raise OpenRouterError(
                "Structured delta checkpoint fragment metadata mismatch"
            )
        fragments = [
            {
                "sequence": latest_sequence,
                "raw_text_sha256": digest,
                "raw_text_chars": raw_chars,
                "raw_text_utf8_bytes": raw_bytes,
                "part": copy.deepcopy(part),
                "call_metadata": copy.deepcopy(metadata),
            }
        ]
    compact_head = _compact_manifest(head)
    for optional_key in ("document_utf8_bytes", "json_prefix"):
        if compact_head.get(optional_key) is None:
            compact_head.pop(optional_key, None)
    return complete, latest_sequence, compact_head, fragments, None


async def _receipt_for_call(
    session: Any,
    *,
    run_id: str,
    stream_sha256: str,
    identity: dict[str, Any],
    contract: dict[str, Any],
    sequence: int,
    raw_text_sha256: str,
) -> RunArtifact:
    prefix = f"{_artifact_prefix('receipt', stream_sha256)}{sequence:012d}_"
    lower, upper = _prefix_bounds(prefix)
    rows = list(
        (
            await session.execute(
                select(RunArtifact)
                .where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key >= lower,
                    RunArtifact.artifact_key < upper,
                )
                .order_by(RunArtifact.id.asc())
            )
        )
        .scalars()
        .all()
    )
    matching: list[RunArtifact] = []
    for row in rows:
        provider_event = _receipt_event(
            row,
            identity=identity,
            stream_sha256=stream_sha256,
            contract=contract,
            predecessor=None,
        )
        if (
            provider_event.get("sequence") == sequence
            and text_sha256(str(provider_event.get("raw_text") or ""))
            == raw_text_sha256
            and _provider_event_is_promotable(provider_event)
        ):
            matching.append(row)
    if len(matching) != 1:
        raise OpenRouterError(
            "Structured checkpoint must reference exactly one provider receipt"
        )
    return matching[0]


def _terminal_disposition(
    error: Any,
    *,
    latest_sequence: int,
) -> tuple[int, str] | None:
    if not isinstance(error, dict):
        return None
    marker = error.get("terminal_semantic_failure")
    if not isinstance(marker, dict) or marker.get("terminal") is not True:
        return None
    marker_sha256 = marker.get("marker_sha256")
    if not isinstance(marker_sha256, str) or len(marker_sha256) != 64:
        raise OpenRouterError("Structured terminal marker digest is invalid")
    failure_kind = marker.get("failure_kind")
    if failure_kind in {
        "complete_rejected_json_part",
        "complete_empty_response",
    }:
        sequence = marker.get("rejected_sequence")
        expected_sequence = latest_sequence + 1
    else:
        sequence = latest_sequence
        expected_sequence = latest_sequence
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or sequence != expected_sequence
    ):
        raise OpenRouterError(
            "Structured terminal disposition sequence is invalid"
        )
    return sequence, _stable_json_sha256(error)


async def _persist_checkpoint(
    run_id: str,
    *,
    stage_key: str,
    model: str,
    identity: dict[str, Any],
    stream_sha256: str,
    contract: dict[str, Any],
    event: dict[str, Any],
) -> None:
    (
        complete,
        latest_sequence,
        compact_manifest,
        fragment_candidates,
        aggregate_usage,
    ) = _checkpoint_projection(event)
    incoming_terminal = _terminal_disposition(
        event.get("error"),
        latest_sequence=latest_sequence,
    )

    await assert_run_lease(run_id)
    async with SessionLocal() as session:
        # Receipt and fragment callbacks can overlap during cancellation.  A
        # reserved SQLite writer lock makes the read/compare/upsert of the one
        # mutable head a transaction instead of a last-writer-wins downgrade.
        await session.execute(sql_text("BEGIN IMMEDIATE"))
        existing_head = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == _head_key(stream_sha256),
                )
            )
        ).scalar_one_or_none()
        existing_latest = -1
        existing_complete = False
        existing_terminal: tuple[int, str] | None = None
        if existing_head is not None:
            head_input, head_output = _validate_row_identity(
                existing_head,
                identity=identity,
                stream_sha256=stream_sha256,
                expected_kind="structured_head",
            )
            if head_input.get("row_sha256") != _stable_json_sha256(head_output):
                raise OpenRouterError("Structured audit head digest mismatch")
            if head_output.get("resume_contract_sha256") != contract["sha256"]:
                raise OpenRouterError("Structured audit head contract mismatch")
            stored_latest = head_output.get("latest_sequence", -1)
            if (
                isinstance(stored_latest, bool)
                or not isinstance(stored_latest, int)
                or stored_latest < -1
            ):
                raise OpenRouterError("Structured audit head sequence is corrupt")
            existing_latest = stored_latest
            existing_complete = head_output.get("complete") is True
            stored_terminal_sequence = head_output.get(
                "terminal_disposition_sequence"
            )
            stored_terminal_error_sha256 = head_output.get(
                "terminal_error_sha256"
            )
            if stored_terminal_sequence is not None or (
                stored_terminal_error_sha256 is not None
            ):
                if (
                    isinstance(stored_terminal_sequence, bool)
                    or not isinstance(stored_terminal_sequence, int)
                    or stored_terminal_sequence < 0
                    or not isinstance(stored_terminal_error_sha256, str)
                    or len(stored_terminal_error_sha256) != 64
                    or _stable_json_sha256(head_output.get("error"))
                    != stored_terminal_error_sha256
                ):
                    raise OpenRouterError(
                        "Structured terminal head disposition is corrupt"
                    )
                existing_terminal = (
                    stored_terminal_sequence,
                    stored_terminal_error_sha256,
                )

        # Decide whether this event is admissible before creating immutable
        # rows. Otherwise a late partial callback can leave fragments beyond a
        # completed head and make the next reconstruction fail closed.
        if existing_terminal is not None:
            if (
                incoming_terminal == existing_terminal
                and latest_sequence == existing_latest
                and compact_manifest == head_output.get("manifest")
            ):
                await session.commit()
                return
            raise OpenRouterError(
                "Terminal structured head cannot be downgraded or extended"
            )
        if existing_complete and incoming_terminal is not None:
            raise OpenRouterError(
                "Completed structured head cannot accept a terminal failure"
            )
        if existing_complete and not complete:
            await session.commit()
            return
        if latest_sequence < existing_latest:
            await session.commit()
            return
        if existing_complete and latest_sequence > existing_latest:
            raise OpenRouterError(
                "Completed structured stream cannot accept later fragments"
            )

        start = max(0, existing_latest + 1)
        candidates_by_sequence = {
            int(candidate["sequence"]): candidate
            for candidate in fragment_candidates
        }
        missing = [
            sequence
            for sequence in range(start, latest_sequence + 1)
            if sequence not in candidates_by_sequence
        ]
        if missing:
            raise OpenRouterError(
                "Structured checkpoint fragment sequence has a gap"
            )
        process_sequences = set(range(start, latest_sequence + 1))
        if (
            event.get("version") == _CHECKPOINT_DELTA_EVENT_VERSION
            and latest_sequence == existing_latest
            and latest_sequence >= 0
        ):
            process_sequences.add(latest_sequence)
        for sequence in sorted(process_sequences):
            candidate = candidates_by_sequence[sequence]
            raw_digest = str(candidate["raw_text_sha256"])
            receipt = await _receipt_for_call(
                session,
                run_id=run_id,
                stream_sha256=stream_sha256,
                identity=identity,
                contract=contract,
                sequence=sequence,
                raw_text_sha256=raw_digest,
            )
            raw_text = receipt.raw_text
            if not isinstance(raw_text, str) or (
                len(raw_text) != candidate["raw_text_chars"]
                or len(raw_text.encode("utf-8"))
                != candidate["raw_text_utf8_bytes"]
            ):
                raise OpenRouterError(
                    "Structured checkpoint fragment size metadata mismatch"
                )
            metadata = candidate["call_metadata"]
            if not isinstance(metadata, dict) or set(metadata).intersection(
                {
                    "raw_text",
                    "usage",
                    "transport",
                    "request_policy",
                    "web_attestation",
                    "prior_transport_attempts",
                }
            ):
                raise OpenRouterError(
                    "Structured checkpoint call metadata is not compact"
                )
            fragment_output = {
                "version": _FRAGMENT_VERSION,
                "row_kind": "accepted_fragment",
                "stream_sha256": stream_sha256,
                "sequence": sequence,
                "receipt_artifact_key": receipt.artifact_key,
                "raw_text_sha256": raw_digest,
                "part": copy.deepcopy(candidate["part"]),
                "call_metadata": copy.deepcopy(metadata),
            }
            fragment_digest = _stable_json_sha256(fragment_output)
            fragment_input = {
                "version": _FRAGMENT_VERSION,
                "row_kind": "accepted_fragment",
                "stream_identity": identity,
                "stream_sha256": stream_sha256,
                "resume_contract_sha256": contract["sha256"],
                "row_sha256": fragment_digest,
            }
            fragment_key = _fragment_key(stream_sha256, sequence)
            inserted = await session.execute(
                sqlite_insert(RunArtifact)
                .values(
                    run_id=run_id,
                    stage_key=stage_key,
                    artifact_key=fragment_key,
                    status="completed",
                    model=model,
                    prompt_version=STRUCTURED_AUDIT_STORE_VERSION,
                    input_json=fragment_input,
                    output_json=fragment_output,
                    raw_text=None,
                    usage_json=None,
                    error_message=None,
                )
                .on_conflict_do_nothing(
                    index_elements=("run_id", "artifact_key")
                )
            )
            if inserted.rowcount != 1:
                existing_fragment = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.artifact_key == fragment_key,
                        )
                    )
                ).scalar_one_or_none()
                stored = (
                    existing_fragment.input_json
                    if existing_fragment is not None
                    else None
                )
                if (
                    not isinstance(stored, dict)
                    or stored.get("row_sha256") != fragment_digest
                ):
                    raise OpenRouterError(
                        "Structured fragment collision or mutation detected"
                    )

        head_output = {
            "version": _HEAD_VERSION,
            "row_kind": "structured_head",
            "stream_sha256": stream_sha256,
            "document_id": event.get("document_id"),
            "latest_sequence": latest_sequence,
            "complete": complete,
            "checkpoint_status": event.get("status"),
            "checkpoint_event_id": event.get("event_id"),
            "source_event_version": event.get("version"),
            "manifest": compact_manifest,
            "resume_contract_sha256": contract["sha256"],
            "error": copy.deepcopy(event.get("error")),
            "terminal_disposition_sequence": (
                incoming_terminal[0]
                if incoming_terminal is not None
                else None
            ),
            "terminal_error_sha256": (
                incoming_terminal[1]
                if incoming_terminal is not None
                else None
            ),
        }
        head_digest = _stable_json_sha256(head_output)
        head_input = {
            "version": _HEAD_VERSION,
            "row_kind": "structured_head",
            "stream_identity": identity,
            "stream_sha256": stream_sha256,
            "resume_contract_sha256": contract["sha256"],
            "row_sha256": head_digest,
        }
        error = event.get("error")
        error_message = (
            str(error.get("message"))
            if isinstance(error, dict) and error.get("message") is not None
            else None
        )
        head_status = "completed" if complete else (
            "failed" if event.get("status") in {"failed", "cancelled"} else "running"
        )
        statement = sqlite_insert(RunArtifact).values(
            run_id=run_id,
            stage_key=stage_key,
            artifact_key=_head_key(stream_sha256),
            status=head_status,
            model=model,
            prompt_version=STRUCTURED_AUDIT_STORE_VERSION,
            input_json=head_input,
            output_json=head_output,
            raw_text=None,
            usage_json=(
                _normalized_usage(aggregate_usage, None) or None
                if isinstance(aggregate_usage, dict)
                else None
            ),
            error_message=error_message,
        )
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=("run_id", "artifact_key"),
                set_={
                    "stage_key": statement.excluded.stage_key,
                    "status": statement.excluded.status,
                    "model": statement.excluded.model,
                    "prompt_version": statement.excluded.prompt_version,
                    "input_json": statement.excluded.input_json,
                    "output_json": statement.excluded.output_json,
                    "raw_text": None,
                    "usage_json": statement.excluded.usage_json,
                    "error_message": statement.excluded.error_message,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        )
        await assert_run_lease(run_id)
        await session.commit()


async def persist_structured_audit_event(
    run_id: str,
    *,
    stage_key: str,
    owner_artifact_key: str,
    source_input: dict[str, Any] | list[Any],
    model: str,
    owner_prompt_version: str,
    event: dict[str, Any],
) -> None:
    """Persist one transport event in normalized, idempotent form."""

    _event_id, kind = _validated_event(event)
    identity, stream_sha256, contract = _event_stream(
        event,
        owner_artifact_key=owner_artifact_key,
        source_input=source_input,
        model=model,
        owner_prompt_version=owner_prompt_version,
    )
    if kind == "provider_post":
        await _persist_provider_receipt(
            run_id,
            stage_key=stage_key,
            model=model,
            identity=identity,
            stream_sha256=stream_sha256,
            contract=contract,
            event=event,
        )
        return
    await _persist_checkpoint(
        run_id,
        stage_key=stage_key,
        model=model,
        identity=identity,
        stream_sha256=stream_sha256,
        contract=contract,
        event=event,
    )


def structured_audit_checkpoint(
    run_id: str,
    *,
    stage_key: str,
    owner_artifact_key: str,
    source_input: dict[str, Any] | list[Any],
    model: str,
    owner_prompt_version: str,
) -> AuditCheckpoint:
    """Return the durable transport callback for one logical artifact."""

    async def checkpoint(event: dict[str, Any]) -> None:
        await persist_structured_audit_event(
            run_id,
            stage_key=stage_key,
            owner_artifact_key=owner_artifact_key,
            source_input=source_input,
            model=model,
            owner_prompt_version=owner_prompt_version,
            event=event,
        )

    capability_attr = getattr(
        openrouter_service,
        "STRUCTURED_AUDIT_EVENT_VERSION_ATTR",
        "aiv_structured_audit_event_version",
    )
    capability_value = getattr(
        openrouter_service,
        "STRUCTURED_AUDIT_DELTA_CAPABILITY",
        "aiv-structured-audit-delta-v2",
    )
    setattr(checkpoint, capability_attr, capability_value)
    return checkpoint


def _validate_row_identity(
    row: RunArtifact,
    *,
    identity: dict[str, Any],
    stream_sha256: str,
    expected_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stored_input = row.input_json
    output = row.output_json
    if not isinstance(stored_input, dict) or not isinstance(output, dict):
        raise OpenRouterError("Structured audit row is corrupt")
    if (
        stored_input.get("stream_identity") != identity
        or stored_input.get("stream_sha256") != stream_sha256
        or output.get("stream_sha256") != stream_sha256
        or output.get("row_kind") != expected_kind
    ):
        raise OpenRouterError("Structured audit row identity mismatch")
    return stored_input, output


def _receipt_event(
    row: RunArtifact,
    *,
    identity: dict[str, Any],
    stream_sha256: str,
    contract: dict[str, Any],
    predecessor: dict[str, Any] | None,
    inflate_delta_for_promotion: bool = False,
    predecessor_document_text: str | None = None,
) -> dict[str, Any]:
    stored_input, output = _validate_row_identity(
        row,
        identity=identity,
        stream_sha256=stream_sha256,
        expected_kind="provider_receipt",
    )
    if stored_input.get("resume_contract_sha256") != contract["sha256"]:
        raise OpenRouterError("Structured provider receipt contract mismatch")
    raw_text = row.raw_text
    if raw_text is not None and not isinstance(raw_text, str):
        raise OpenRouterError("Structured provider receipt raw text is corrupt")
    raw_digest = text_sha256(raw_text) if raw_text is not None else None
    if output.get("raw_text_sha256") != raw_digest:
        raise OpenRouterError("Structured provider receipt raw digest mismatch")
    stored_request_payload = stored_input.get("request_payload")
    if not isinstance(stored_request_payload, dict):
        raise OpenRouterError("Structured provider receipt request is corrupt")
    stored_request_digest = _stable_json_sha256(stored_request_payload)
    expected_stored_digest = stored_input.get("stored_request_sha256")
    original_request_digest = str(stored_input.get("request_sha256") or "")
    if expected_stored_digest is None:
        # Compatibility for snapshot-v1 rows written before request
        # normalization existed: their stored and provider payloads coincide.
        expected_stored_digest = original_request_digest
    if (
        _SHA256_RE.fullmatch(original_request_digest) is None
        or expected_stored_digest != stored_request_digest
        or output.get("request_sha256") != original_request_digest
    ):
        raise OpenRouterError("Structured provider stored request is corrupt")
    source_event_version = str(output.get("source_event_version") or "")
    if source_event_version not in {
        _PHYSICAL_EVENT_VERSION,
        _PHYSICAL_DELTA_EVENT_VERSION,
    }:
        raise OpenRouterError("Structured provider receipt source version is corrupt")
    receipt_sequence = output.get("sequence")
    if (
        isinstance(receipt_sequence, bool)
        or not isinstance(receipt_sequence, int)
        or receipt_sequence < 0
    ):
        raise OpenRouterError("Structured provider receipt sequence is corrupt")
    request_predecessor_text = predecessor_document_text
    if (
        request_predecessor_text is None
        and inflate_delta_for_promotion
        and receipt_sequence > 0
    ):
        if not isinstance(predecessor, dict) or not isinstance(
            predecessor.get("partial_text"), str
        ):
            raise OpenRouterError(
                "Structured recovery request predecessor text is missing"
            )
        request_predecessor_text = str(predecessor["partial_text"])
    request_payload = _materialized_request_payload(
        stored_request_payload,
        source_event_version=source_event_version,
        sequence=receipt_sequence,
        original_sha256=original_request_digest,
        predecessor=(
            output.get("predecessor")
            if isinstance(output.get("predecessor"), dict)
            else None
        ),
        predecessor_document_text=request_predecessor_text,
    )
    stored_usage = (
        _restore_raw_text(row.usage_json, raw_text)
        if isinstance(row.usage_json, dict)
        else {}
    )
    usage = _usage_with_receipt_metadata(
        stored_usage,
        output,
        raw_text=raw_text,
    )
    pending_manifest = {
        "version": STRUCTURED_CONTINUATION_VERSION,
        "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
        "document_id": contract["document_id"],
        "complete": False,
        "part_count": 0,
        "calls": [],
        "accepted_document_text": "",
    }
    reconstructed = {
        "version": source_event_version,
        "event_id": output.get("event_id"),
        "event_kind": "provider_post",
        "logical_call_id": output.get("logical_call_id"),
        "document_id": output.get("document_id"),
        "sequence": output.get("sequence"),
        "attempt": output.get("attempt"),
        "status": output.get("status"),
        "model": output.get("model"),
        "request_payload": copy.deepcopy(request_payload),
        "request_sha256": stored_input.get("request_sha256"),
        "response": _restore_raw_text(output.get("response") or {}, raw_text),
        "raw_text": raw_text,
        "usage": usage,
        "transport": _restore_raw_text(output.get("transport") or {}, raw_text),
        "resume_contract": contract,
        "error": _restore_raw_text(output.get("error"), raw_text),
    }
    if source_event_version == _PHYSICAL_DELTA_EVENT_VERSION:
        reconstructed["predecessor"] = copy.deepcopy(output.get("predecessor"))
        delta_predecessor = _provider_predecessor(reconstructed)
        if inflate_delta_for_promotion:
            assert isinstance(delta_predecessor, dict)
            if predecessor is None:
                expected_latest_sequence = -1
                expected_document_sha256 = text_sha256("")
                expected_document_chars = 0
                expected_document_utf8_bytes = 0
            else:
                predecessor_manifest = predecessor.get("manifest")
                if not isinstance(predecessor_manifest, dict):
                    raise OpenRouterError(
                        "Structured recovery predecessor manifest is missing"
                    )
                expected_latest_sequence = predecessor.get("sequence")
                expected_document_sha256 = predecessor_manifest.get(
                    "document_sha256"
                )
                expected_document_chars = predecessor_manifest.get(
                    "document_chars"
                )
                expected_document_utf8_bytes = predecessor_manifest.get(
                    "document_utf8_bytes"
                )
            if (
                delta_predecessor.get("latest_sequence")
                != expected_latest_sequence
                or delta_predecessor.get("expected_sequence")
                != reconstructed.get("sequence")
                or delta_predecessor.get("document_sha256")
                != expected_document_sha256
                or delta_predecessor.get("document_chars")
                != expected_document_chars
                or (
                    delta_predecessor.get("document_utf8_bytes") is not None
                    and delta_predecessor.get("document_utf8_bytes")
                    != expected_document_utf8_bytes
                )
            ):
                raise OpenRouterError(
                    "Structured delta provider predecessor does not match head"
                )
            reconstructed.pop("predecessor", None)
            reconstructed["version"] = _PHYSICAL_EVENT_VERSION
            reconstructed.update(
                {
                    "partial_text": (
                        predecessor.get("partial_text", "")
                        if isinstance(predecessor, dict)
                        else ""
                    ),
                    "manifest": (
                        copy.deepcopy(predecessor.get("manifest"))
                        if isinstance(predecessor, dict)
                        else pending_manifest
                    ),
                    "aggregate_usage": (
                        copy.deepcopy(predecessor.get("aggregate_usage") or {})
                        if isinstance(predecessor, dict)
                        else {}
                    ),
                    "call_records": (
                        copy.deepcopy(predecessor.get("call_records") or [])
                        if isinstance(predecessor, dict)
                        else []
                    ),
                }
            )
    else:
        reconstructed.update(
            {
                "partial_text": (
                    predecessor.get("partial_text", "")
                    if isinstance(predecessor, dict)
                    else ""
                ),
                "manifest": (
                    copy.deepcopy(predecessor.get("manifest"))
                    if isinstance(predecessor, dict)
                    else pending_manifest
                ),
                "aggregate_usage": (
                    copy.deepcopy(predecessor.get("aggregate_usage") or {})
                    if isinstance(predecessor, dict)
                    else {}
                ),
                "call_records": (
                    copy.deepcopy(predecessor.get("call_records") or [])
                    if isinstance(predecessor, dict)
                    else []
                ),
            }
        )
    digest_payload = {
        "identity": identity,
        "request_payload": stored_request_payload,
        "output": output,
        "raw_text_sha256": raw_digest,
        "usage": row.usage_json if isinstance(row.usage_json, dict) else {},
        "resume_contract_sha256": contract["sha256"],
    }
    if stored_input.get("row_sha256") != _stable_json_sha256(digest_payload):
        raise OpenRouterError("Structured provider receipt row digest mismatch")
    return reconstructed


def _prior_attempt_record(
    row: RunArtifact,
    provider_event: dict[str, Any],
) -> dict[str, Any]:
    output = row.output_json
    metadata = (
        output.get("call_attempt_metadata")
        if isinstance(output, dict)
        else None
    )
    raw_text = provider_event.get("raw_text")
    if not isinstance(metadata, dict) or not isinstance(raw_text, str):
        raise OpenRouterError("Structured prior attempt metadata is missing")
    record = _restore_raw_text(metadata, raw_text)
    if not isinstance(record, dict):
        raise OpenRouterError("Structured prior attempt metadata is corrupt")
    record_usage = record.get("usage")
    provider_usage = provider_event.get("usage")
    provider_transport = provider_event.get("transport")
    if (
        record.get("attempt") != provider_event.get("attempt")
        or record.get("status") != provider_event.get("status")
        or record.get("text_sha256") != text_sha256(raw_text)
        or record.get("text_chars") != len(raw_text)
        or record.get("text_utf8_bytes") != len(raw_text.encode("utf-8"))
        or not isinstance(record_usage, dict)
        or not isinstance(provider_usage, dict)
        or not isinstance(provider_transport, dict)
    ):
        raise OpenRouterError("Structured prior attempt metadata mismatch")
    for key in (
        "_aiv_transport",
        "_aiv_request_policy",
        "_aiv_web_attestation",
        "_aiv_router_metadata",
    ):
        if key in provider_usage:
            record_usage[key] = copy.deepcopy(provider_usage[key])
    return {
        **copy.deepcopy(record),
        "raw_text": raw_text,
        "usage": record_usage,
        "transport": copy.deepcopy(provider_transport),
    }


def _aggregate_call_usage(call_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not call_records:
        return {}
    first_usage = call_records[0].get("usage")
    aggregate = copy.deepcopy(first_usage) if isinstance(first_usage, dict) else {}
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
    for key in numeric_keys:
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


async def _stream_rows(
    session: Any,
    run_id: str,
    *,
    stream_sha256: str,
    kind: str,
) -> list[RunArtifact]:
    prefix = _artifact_prefix(kind, stream_sha256)
    lower, upper = _prefix_bounds(prefix)
    return list(
        (
            await session.execute(
                select(RunArtifact)
                .where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key >= lower,
                    RunArtifact.artifact_key < upper,
                )
                .order_by(RunArtifact.artifact_key.asc())
            )
        )
        .scalars()
        .all()
    )


async def _reconstruct_checkpoint(
    run_id: str,
    *,
    identity: dict[str, Any],
    stream_sha256: str,
    contract: dict[str, Any],
    overlap_chars: int,
) -> tuple[dict[str, Any] | None, list[RunArtifact]]:
    async with SessionLocal() as session:
        head = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == _head_key(stream_sha256),
                )
            )
        ).scalar_one_or_none()
        receipts = await _stream_rows(
            session,
            run_id,
            stream_sha256=stream_sha256,
            kind="receipt",
        )
        if head is None:
            return None, receipts
        fragments = await _stream_rows(
            session,
            run_id,
            stream_sha256=stream_sha256,
            kind="fragment",
        )

    head_input, head_output = _validate_row_identity(
        head,
        identity=identity,
        stream_sha256=stream_sha256,
        expected_kind="structured_head",
    )
    if head_input.get("row_sha256") != _stable_json_sha256(head_output):
        raise OpenRouterError("Structured audit head digest mismatch")
    if (
        head_input.get("resume_contract_sha256") != contract["sha256"]
        or head_output.get("resume_contract_sha256") != contract["sha256"]
    ):
        raise OpenRouterError("Structured audit head contract mismatch")
    source_event_version = str(head_output.get("source_event_version") or "")
    if source_event_version not in {
        _CHECKPOINT_EVENT_VERSION,
        _CHECKPOINT_DELTA_EVENT_VERSION,
    }:
        raise OpenRouterError("Structured audit head source version is corrupt")
    latest_sequence = head_output.get("latest_sequence")
    if (
        isinstance(latest_sequence, bool)
        or not isinstance(latest_sequence, int)
        or latest_sequence < -1
    ):
        raise OpenRouterError("Structured audit head sequence is corrupt")
    if latest_sequence == -1:
        if fragments:
            raise OpenRouterError(
                "Empty structured audit head has accepted fragments"
            )
        if head_output.get("complete") is not False:
            raise OpenRouterError(
                "Empty structured audit head cannot be complete"
            )
        empty_manifest = head_output.get("manifest")
        expected_empty = {
            "version": STRUCTURED_CONTINUATION_VERSION,
            "mode": ResponseMode.CONTINUABLE_DOCUMENT.value,
            "document_id": identity["document_id"],
            "complete": False,
            "continuation_count": 0,
            "part_count": 0,
            "document_sha256": text_sha256(""),
            "document_chars": 0,
        }
        if not isinstance(empty_manifest, dict) or any(
            empty_manifest.get(key) != value
            for key, value in expected_empty.items()
        ):
            raise OpenRouterError(
                "Empty structured audit head manifest is corrupt"
            )
        if empty_manifest.get("document_utf8_bytes", 0) != 0:
            raise OpenRouterError(
                "Empty structured audit head byte count is corrupt"
            )
        terminal = _terminal_disposition(
            head_output.get("error"),
            latest_sequence=-1,
        )
        if terminal is not None:
            if (
                head_output.get("terminal_disposition_sequence")
                != terminal[0]
                or head_output.get("terminal_error_sha256") != terminal[1]
                or head_output.get("checkpoint_status") != "failed"
            ):
                raise OpenRouterError(
                    "Empty terminal structured head disposition is corrupt"
                )
            rebuilt_empty_manifest = copy.deepcopy(empty_manifest)
            rebuilt_empty_manifest.update(
                {
                    "complete": False,
                    "continuation_count": 0,
                    "part_count": 0,
                    "document_sha256": text_sha256(""),
                    "document_chars": 0,
                    "document_utf8_bytes": 0,
                    "parts": [],
                    "calls": [],
                    "accepted_document_text": "",
                }
            )
            return (
                {
                    "version": _CHECKPOINT_EVENT_VERSION,
                    "event_id": head_output.get("checkpoint_event_id"),
                    "event_kind": "structured_continuation_checkpoint",
                    "document_id": identity["document_id"],
                    "sequence": -1,
                    "status": "failed",
                    "partial_text": "",
                    "manifest": rebuilt_empty_manifest,
                    "aggregate_usage": {},
                    "call_records": [],
                    "resume_contract": contract,
                    "error": copy.deepcopy(head_output.get("error")),
                },
                receipts,
            )
        return None, receipts

    if len(fragments) != latest_sequence + 1:
        raise OpenRouterError("Structured audit fragment sequence is incomplete")
    receipt_by_key = {row.artifact_key: row for row in receipts}
    receipt_event_by_key: dict[str, dict[str, Any]] = {}
    receipts_by_logical_call: dict[
        tuple[int, str], list[tuple[RunArtifact, dict[str, Any]]]
    ] = defaultdict(list)
    for receipt in receipts:
        provider_event = _receipt_event(
            receipt,
            identity=identity,
            stream_sha256=stream_sha256,
            contract=contract,
            predecessor=None,
        )
        receipt_event_by_key[receipt.artifact_key] = provider_event
        receipt_sequence = provider_event.get("sequence")
        logical_call_id = str(provider_event.get("logical_call_id") or "")
        if (
            isinstance(receipt_sequence, bool)
            or not isinstance(receipt_sequence, int)
            or receipt_sequence < 0
            or not logical_call_id
        ):
            raise OpenRouterError("Structured provider receipt grouping is corrupt")
        receipts_by_logical_call[(receipt_sequence, logical_call_id)].append(
            (receipt, provider_event)
        )
    call_records: list[dict[str, Any]] = []
    stored_parts: list[dict[str, Any]] = []
    request_ledger: StructuredContinuationLedger | None = None
    for sequence, fragment in enumerate(fragments):
        fragment_input, fragment_output = _validate_row_identity(
            fragment,
            identity=identity,
            stream_sha256=stream_sha256,
            expected_kind="accepted_fragment",
        )
        if fragment_input.get("resume_contract_sha256") != contract["sha256"]:
            raise OpenRouterError("Structured audit fragment contract mismatch")
        if fragment_input.get("row_sha256") != _stable_json_sha256(
            fragment_output
        ):
            raise OpenRouterError("Structured audit fragment digest mismatch")
        if fragment_output.get("sequence") != sequence:
            raise OpenRouterError("Structured audit fragment order mismatch")
        receipt_key = str(fragment_output.get("receipt_artifact_key") or "")
        receipt = receipt_by_key.get(receipt_key)
        if receipt is None:
            raise OpenRouterError("Structured audit fragment receipt is missing")
        provider_event = receipt_event_by_key[receipt.artifact_key]
        if sequence > 0:
            if request_ledger is None:
                raise OpenRouterError(
                    "Structured continuation request predecessor is missing"
                )
            provider_event = _receipt_event(
                receipt,
                identity=identity,
                stream_sha256=stream_sha256,
                contract=contract,
                predecessor=None,
                predecessor_document_text=request_ledger.text,
            )
            receipt_event_by_key[receipt.artifact_key] = provider_event
        raw_text = provider_event.get("raw_text")
        if not isinstance(raw_text, str) or (
            text_sha256(raw_text) != fragment_output.get("raw_text_sha256")
        ):
            raise OpenRouterError("Structured audit fragment raw text mismatch")
        metadata = fragment_output.get("call_metadata")
        stored_part = fragment_output.get("part")
        if not isinstance(metadata, dict) or not isinstance(stored_part, dict):
            raise OpenRouterError("Structured audit fragment metadata is corrupt")
        logical_call_id = str(provider_event.get("logical_call_id") or "")
        current_attempt = provider_event.get("attempt")
        if (
            isinstance(current_attempt, bool)
            or not isinstance(current_attempt, int)
            or current_attempt < 1
        ):
            raise OpenRouterError("Structured accepted receipt attempt is corrupt")
        prior_attempts_by_number: dict[int, dict[str, Any]] = {}
        for prior_row, prior_event in receipts_by_logical_call[
            (sequence, logical_call_id)
        ]:
            prior_attempt = prior_event.get("attempt")
            if (
                isinstance(prior_attempt, bool)
                or not isinstance(prior_attempt, int)
                or prior_attempt < 1
            ):
                raise OpenRouterError("Structured prior receipt attempt is corrupt")
            if prior_attempt >= current_attempt or not isinstance(
                prior_event.get("raw_text"), str
            ):
                continue
            if prior_event.get("status") != "rejected":
                raise OpenRouterError(
                    "Structured logical call has an impossible earlier result"
                )
            if prior_attempt in prior_attempts_by_number:
                raise OpenRouterError(
                    "Structured logical call has duplicate prior attempts"
                )
            prior_attempts_by_number[prior_attempt] = _prior_attempt_record(
                prior_row,
                prior_event,
            )
        prior_transport_attempts = [
            prior_attempts_by_number[attempt]
            for attempt in sorted(prior_attempts_by_number)
        ]
        call_records.append(
            {
                **copy.deepcopy(metadata),
                "sequence": sequence,
                "raw_text": raw_text,
                "text_sha256": text_sha256(raw_text),
                "text_chars": len(raw_text),
                "text_utf8_bytes": len(raw_text.encode("utf-8")),
                "usage": copy.deepcopy(provider_event.get("usage") or {}),
                "transport": copy.deepcopy(provider_event.get("transport") or {}),
                "request_policy": copy.deepcopy(
                    (provider_event.get("usage") or {}).get(
                        "_aiv_request_policy"
                    )
                    or {}
                ),
                "web_attestation": copy.deepcopy(
                    (provider_event.get("usage") or {}).get(
                        "_aiv_web_attestation"
                    )
                    or {}
                ),
                "prior_transport_attempts": prior_transport_attempts,
            }
        )
        stored_parts.append(copy.deepcopy(stored_part))
        if request_ledger is None:
            request_ledger = StructuredContinuationLedger(
                document_id=str(identity["document_id"]),
                text=raw_text,
                overlap_chars=overlap_chars,
            )
        else:
            request_ledger.append(raw_text, sequence=sequence)

    if request_ledger is None:
        raise OpenRouterError("Structured audit accepted ledger is empty")
    ledger = request_ledger
    complete = head_output.get("complete")
    if not isinstance(complete, bool):
        raise OpenRouterError("Structured audit head completion flag is invalid")
    rebuilt_manifest = ledger.manifest(complete=complete)
    for stored_part, rebuilt_part in zip(
        stored_parts,
        rebuilt_manifest["parts"],
        strict=True,
    ):
        for key, value in rebuilt_part.items():
            if key != "kind" and stored_part.get(key) != value:
                raise OpenRouterError(
                    f"Structured audit part metadata mismatch: {key}"
                )
    stored_manifest = head_output.get("manifest")
    if not isinstance(stored_manifest, dict):
        raise OpenRouterError("Structured audit head manifest is missing")
    manifest_keys = (
        "version",
        "mode",
        "document_id",
        "complete",
        "continuation_count",
        "part_count",
        "document_sha256",
        "document_chars",
        "document_utf8_bytes",
        "json_prefix",
    )
    for key in manifest_keys:
        if source_event_version == _CHECKPOINT_DELTA_EVENT_VERSION and (
            key not in stored_manifest
        ):
            continue
        if stored_manifest.get(key) != rebuilt_manifest.get(key):
            raise OpenRouterError(
                f"Structured audit reconstructed manifest mismatch: {key}"
            )
    rebuilt_manifest["calls"] = copy.deepcopy(call_records)
    rebuilt_manifest["accepted_document_text"] = ledger.text
    checkpoint = {
        "version": _CHECKPOINT_EVENT_VERSION,
        "event_id": head_output.get("checkpoint_event_id"),
        "event_kind": "structured_continuation_checkpoint",
        "document_id": identity["document_id"],
        "sequence": latest_sequence,
        "status": head_output.get("checkpoint_status"),
        "partial_text": ledger.text,
        "manifest": rebuilt_manifest,
        "aggregate_usage": _aggregate_call_usage(call_records),
        "call_records": call_records,
        "resume_contract": contract,
        "error": copy.deepcopy(head_output.get("error")),
    }
    terminal = _terminal_disposition(
        head_output.get("error"),
        latest_sequence=latest_sequence,
    )
    if terminal is not None and (
        head_output.get("terminal_disposition_sequence") != terminal[0]
        or head_output.get("terminal_error_sha256") != terminal[1]
        or head_output.get("checkpoint_status") != "failed"
    ):
        raise OpenRouterError(
            "Terminal structured head disposition is corrupt"
        )
    return checkpoint, receipts


def _recovery_receipts(
    rows: list[RunArtifact],
    *,
    identity: dict[str, Any],
    stream_sha256: str,
    contract: dict[str, Any],
    predecessor: dict[str, Any] | None,
) -> tuple[
    dict[int, list[tuple[RunArtifact, dict[str, Any]]]],
    dict[int, list[tuple[RunArtifact, dict[str, Any]]]],
]:
    current_sequence = (
        int(predecessor.get("sequence", -1))
        if isinstance(predecessor, dict)
        else -1
    )
    expected = current_sequence + 1
    response_bearing: dict[
        int, list[tuple[RunArtifact, dict[str, Any]]]
    ] = defaultdict(list)
    promotable: dict[
        int, list[tuple[RunArtifact, dict[str, Any]]]
    ] = defaultdict(list)
    for row in rows:
        event = _receipt_event(
            row,
            identity=identity,
            stream_sha256=stream_sha256,
            contract=contract,
            predecessor=predecessor,
        )
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise OpenRouterError("Structured provider receipt sequence is corrupt")
        if sequence < expected:
            continue
        if not _provider_event_has_response_evidence(event):
            continue
        response_bearing[sequence].append((row, event))
        if _provider_event_is_promotable(event):
            promotable[sequence].append((row, event))
    return response_bearing, promotable


def _recovery_candidate(
    *,
    sequence: int,
    response_bearing: list[tuple[RunArtifact, dict[str, Any]]],
    promotable: list[tuple[RunArtifact, dict[str, Any]]],
) -> tuple[RunArtifact, dict[str, Any]]:
    """Select one safe result while retaining earlier same-call attempts."""

    if not promotable:
        raise OpenRouterError(
            "Structured recovery found a response-bearing provider receipt "
            f"that cannot be promoted at sequence {sequence}"
        )
    if len(promotable) != 1:
        raise OpenRouterError(
            "Structured recovery has ambiguous provider receipts for "
            f"sequence {sequence}"
        )
    candidate_row, candidate = promotable[0]
    logical_call_id = str(candidate.get("logical_call_id") or "")
    candidate_attempt = candidate.get("attempt")
    if (
        not logical_call_id
        or isinstance(candidate_attempt, bool)
        or not isinstance(candidate_attempt, int)
        or candidate_attempt < 1
    ):
        raise OpenRouterError(
            "Structured recovery candidate identity is corrupt"
        )
    for row, event in response_bearing:
        event_logical_call_id = str(event.get("logical_call_id") or "")
        event_attempt = event.get("attempt")
        if event_logical_call_id != logical_call_id:
            raise OpenRouterError(
                "Structured recovery has response-bearing receipts from "
                f"multiple logical calls at sequence {sequence}"
            )
        if (
            isinstance(event_attempt, bool)
            or not isinstance(event_attempt, int)
            or event_attempt < 1
        ):
            raise OpenRouterError(
                "Structured recovery response attempt is corrupt"
            )
        if row.artifact_key != candidate_row.artifact_key and (
            event_attempt >= candidate_attempt
        ):
            raise OpenRouterError(
                "Structured recovery has a response after its accepted "
                f"candidate at sequence {sequence}"
            )
    return candidate_row, candidate


async def load_structured_checkpoint(
    run_id: str,
    *,
    owner_artifact_key: str,
    source_input: dict[str, Any] | list[Any],
    model: str,
    owner_prompt_version: str,
    messages: list[dict[str, Any]],
    schema_name: str,
    response_schema: dict[str, Any],
    document_id: str,
    complete: bool | None = None,
    overlap_chars: int = DEFAULT_STRUCTURED_CONTINUATION_OVERLAP_CHARS,
    reasoning_effort: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any] | None:
    """Load one exact-contract checkpoint and replay any newer paid receipt."""

    if complete is not None and not isinstance(complete, bool):
        raise OpenRouterError(
            "Structured checkpoint complete must be boolean or null"
        )
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, int)
        or overlap_chars < 1
    ):
        raise OpenRouterError("Structured checkpoint overlap must be positive")
    contract_builder = getattr(
        openrouter_service,
        "structured_resume_contract",
        None,
    )
    promoter = getattr(
        openrouter_service,
        "promote_provider_post_to_structured_checkpoint",
        None,
    )
    if not callable(contract_builder) or not callable(promoter):
        raise OpenRouterError(
            "OpenRouter structured recovery helpers are unavailable"
        )
    contract = contract_builder(
        model=model,
        messages=messages,
        schema_name=schema_name,
        response_schema=response_schema,
        document_id=document_id,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        overlap_chars=overlap_chars,
    )
    contract = _validated_resume_contract(contract)
    identity, stream_sha256 = _stream_identity(
        owner_artifact_key=owner_artifact_key,
        source_input=source_input,
        model=model,
        owner_prompt_version=owner_prompt_version,
        document_id=document_id,
        resume_contract=contract,
    )
    await assert_run_lease(run_id)
    checkpoint, receipts = await _reconstruct_checkpoint(
        run_id,
        identity=identity,
        stream_sha256=stream_sha256,
        contract=contract,
        overlap_chars=overlap_chars,
    )
    checkpoint_manifest = (
        checkpoint.get("manifest") if isinstance(checkpoint, dict) else None
    )
    response_bearing, promotable = _recovery_receipts(
        receipts,
        identity=identity,
        stream_sha256=stream_sha256,
        contract=contract,
        predecessor=checkpoint,
    )
    checkpoint_error = (
        checkpoint.get("error") if isinstance(checkpoint, dict) else None
    )
    terminal = (
        _terminal_disposition(
            checkpoint_error,
            latest_sequence=int(checkpoint.get("sequence", -1)),
        )
        if isinstance(checkpoint, dict)
        else None
    )
    if terminal is not None:
        terminal_marker = checkpoint_error.get("terminal_semantic_failure")
        terminal_kind = (
            terminal_marker.get("failure_kind")
            if isinstance(terminal_marker, dict)
            else None
        )
        if terminal_kind in {
            "complete_rejected_json_part",
            "complete_empty_response",
        }:
            terminal_sequence = terminal[0]
            if set(response_bearing) != {terminal_sequence}:
                raise OpenRouterError(
                    "Terminal rejected-part head has missing or later receipts"
                )
            row, _candidate = _recovery_candidate(
                sequence=terminal_sequence,
                response_bearing=response_bearing[terminal_sequence],
                promotable=promotable.get(terminal_sequence, []),
            )
            predecessor_for_promotion = (
                checkpoint
                if checkpoint.get("call_records")
                else None
            )
            provider_event = _receipt_event(
                row,
                identity=identity,
                stream_sha256=stream_sha256,
                contract=contract,
                predecessor=predecessor_for_promotion,
                inflate_delta_for_promotion=True,
            )
            reproduced = promoter(
                provider_event,
                predecessor_for_promotion,
                model=model,
                messages=messages,
                schema_name=schema_name,
                response_schema=response_schema,
                document_id=document_id,
                overlap_chars=overlap_chars,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
            if inspect.isawaitable(reproduced):
                reproduced = await reproduced
            reproduced_error = (
                reproduced.get("error")
                if isinstance(reproduced, dict)
                else None
            )
            if not (
                isinstance(reproduced_error, dict)
                and reproduced_error.get("terminal_semantic_failure")
                == checkpoint_error.get("terminal_semantic_failure")
                and reproduced_error.get("terminal_rejected_part")
                == checkpoint_error.get("terminal_rejected_part")
                and reproduced.get("partial_text")
                == checkpoint.get("partial_text")
                and reproduced.get("call_records")
                == checkpoint.get("call_records")
            ):
                raise OpenRouterError(
                    "Terminal rejected-part head does not match its receipt"
                )
        elif response_bearing:
            raise OpenRouterError(
                "Terminal structured head has a later provider receipt"
            )
        manifest = checkpoint.get("manifest")
        if complete is not None and (
            not isinstance(manifest, dict)
            or manifest.get("complete") is not complete
        ):
            return None
        return checkpoint
    next_sequence = (
        int(checkpoint.get("sequence", -1)) + 1
        if isinstance(checkpoint, dict)
        else 0
    )
    if (
        isinstance(checkpoint_manifest, dict)
        and checkpoint_manifest.get("complete") is True
        and response_bearing
    ):
        raise OpenRouterError(
            "Structured recovery found a response-bearing provider receipt "
            "after completion"
        )
    if response_bearing and next_sequence not in response_bearing:
        raise OpenRouterError(
            "Structured recovery provider receipt sequence has a gap"
        )
    promoted_any = False
    while next_sequence in response_bearing:
        row, _provider_event = _recovery_candidate(
            sequence=next_sequence,
            response_bearing=response_bearing[next_sequence],
            promotable=promotable.get(next_sequence, []),
        )
        provider_event = _receipt_event(
            row,
            identity=identity,
            stream_sha256=stream_sha256,
            contract=contract,
            predecessor=checkpoint,
            inflate_delta_for_promotion=True,
        )
        promoted = promoter(
            provider_event,
            checkpoint,
            model=model,
            messages=messages,
            schema_name=schema_name,
            response_schema=response_schema,
            document_id=document_id,
            overlap_chars=overlap_chars,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        if inspect.isawaitable(promoted):
            promoted = await promoted
        if not isinstance(promoted, dict):
            raise OpenRouterError(
                "OpenRouter provider-receipt promotion returned no checkpoint"
            )
        await persist_structured_audit_event(
            run_id,
            stage_key=str(row.stage_key),
            owner_artifact_key=owner_artifact_key,
            source_input=source_input,
            model=model,
            owner_prompt_version=owner_prompt_version,
            event=promoted,
        )
        promoted_any = True
        checkpoint = promoted
        promoted_error = promoted.get("error")
        promoted_terminal_marker = (
            promoted_error.get("terminal_semantic_failure")
            if isinstance(promoted_error, dict)
            else None
        )
        if (
            (promoted.get("manifest") or {}).get("complete") is True
            or isinstance(promoted_terminal_marker, dict)
        ):
            break
        next_sequence += 1

        if response_bearing:
            recovered_sequence = (
                int(checkpoint.get("sequence", -1))
                if isinstance(checkpoint, dict)
                else -1
            )
            checkpoint_error = (
                checkpoint.get("error")
                if isinstance(checkpoint, dict)
                else None
            )
            recovered_terminal = _terminal_disposition(
                checkpoint_error,
                latest_sequence=recovered_sequence,
            )
            if recovered_terminal is not None:
                # A rejected-part terminal head deliberately keeps the paid
                # response out of the accepted call ledger.  Its sealed
                # disposition sequence still counts that receipt as recovered.
                recovered_sequence = recovered_terminal[0]
            unrecovered = sorted(
                sequence
                for sequence in response_bearing
            if sequence > recovered_sequence
        )
        recovered_manifest = (
            checkpoint.get("manifest") if isinstance(checkpoint, dict) else None
        )
        if (
            unrecovered
            and isinstance(recovered_manifest, dict)
            and recovered_manifest.get("complete") is True
        ):
            raise OpenRouterError(
                "Structured recovery found a response-bearing provider "
                "receipt after completion"
            )
        if unrecovered and unrecovered[0] > recovered_sequence + 1:
            raise OpenRouterError(
                "Structured recovery provider receipt sequence has a gap"
            )
        if unrecovered:
            raise OpenRouterError(
                "Structured recovery left an orphan provider receipt"
            )

    if promoted_any:
        checkpoint, _receipts = await _reconstruct_checkpoint(
            run_id,
            identity=identity,
            stream_sha256=stream_sha256,
            contract=contract,
            overlap_chars=overlap_chars,
        )
    if not isinstance(checkpoint, dict):
        return None
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict):
        return None
    if complete is not None and manifest.get("complete") is not complete:
        return None
    return checkpoint
