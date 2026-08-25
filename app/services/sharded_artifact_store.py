"""Durable ``RunArtifact`` adapter for independently composable shards.

The generic sharded-document harness deliberately knows nothing about the
database or OpenRouter.  This adapter closes that boundary without weakening
either contract:

* every physical provider POST emitted by :func:`openrouter.chat` is stored as
  an immutable, content-addressed audit receipt;
* an accepted, schema-valid POST can be promoted after a crash, before another
  paid request is attempted;
* accepted shard receipts form one ordered predecessor chain whose mutable
  head advances through an atomic SQLite compare-and-set;
* every read revalidates the complete stored value and all request/response
  hashes.  Ambiguity, a gap, or mutation fails closed.

There are no response-length, shard-count, or aggregate-document limits here.
Provider context/output envelopes and operational liveness deadlines remain
the responsibility of the transport and the generic harness respectively.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db import SessionLocal
from app.models import Run, RunArtifact, RunStatus
from app.services.long_response import text_sha256
from app.services.run_lease import (
    RunLeaseLostError,
    assert_run_lease,
    current_run_lease,
)
from app.services.sharded_document import (
    PROVIDER_AUDIT_BINDING_VERSION,
    SHARD_REQUEST_VERSION,
    SHARD_RECEIPT_VERSION,
    GeneratedShard,
    ProviderAuditBinding,
    ShardComposability,
    ShardPlan,
    ShardReceiptError,
    ShardRequest,
    ShardSaveAck,
    shard_request,
    verify_shard_receipts,
)


SHARDED_ARTIFACT_STORE_VERSION = "aiv-sharded-artifact-store-v1"
SHARDED_PROVIDER_RECEIPT_VERSION = "aiv-sharded-provider-receipt-v1"
SHARDED_RECEIPT_ROW_VERSION = "aiv-sharded-receipt-row-v1"
SHARDED_RECEIPT_HEAD_VERSION = "aiv-sharded-receipt-head-v1"
SHARDED_PROVIDER_CONTRACT_VERSION = "aiv-sharded-provider-contract-v1"

_PHYSICAL_POST_EVENT_VERSION = "aiv-openrouter-physical-post-audit-v1"
_PROVIDER_KEY_PREFIX = "aiv_sdpa_"
_SHARD_RECEIPT_KEY_PREFIX = "aiv_sdr_"
_SHARD_HEAD_KEY_PREFIX = "aiv_sdh_"
_EVENT_ID_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ALLOWED_PROVIDER_STATUSES = {
    "accepted",
    "rejected",
    "http_error",
    "response_error",
    "transport_error",
    "cancelled",
}
_PHYSICAL_EVENT_KEYS = {
    "version",
    "event_id",
    "event_kind",
    "logical_call_id",
    "document_id",
    "sequence",
    "attempt",
    "status",
    "model",
    "request_payload",
    "request_sha256",
    "request_body_utf8_bytes",
    "request_body_encoding",
    "response",
    "raw_text",
    "citations",
    "annotations",
    "request_policy",
    "web_attestation",
    "router_metadata",
    "usage",
    "transport",
    "resume_contract",
    "error",
    "partial_text",
    "manifest",
    "aggregate_usage",
    "call_records",
}
_SHARD_RECEIPT_CORE_KEYS = {
    "version",
    "receipt_kind",
    "document_id",
    "plan_sha256",
    "plan_version",
    "merge_version",
    "composability",
    "generation_contract_sha256",
    "merge_contract_sha256",
    "shard_schema_sha256",
    "document_schema_sha256",
    "index",
    "shard_id",
    "shard_count",
    "spec_sha256",
    "payload_sha256",
    "request_sha256",
    "predecessor_receipt_sha256",
    "content",
    "content_sha256",
    "raw_text",
    "raw_text_sha256",
    "provider_audit",
    "metadata",
}
_SHARD_RECEIPT_KEYS = _SHARD_RECEIPT_CORE_KEYS | {
    "receipt_sha256",
    "receipt_id",
}


ProviderCall = Callable[
    [Callable[[dict[str, Any]], Awaitable[None]], dict[str, Any]],
    Awaitable[Any],
]


class ShardedArtifactStoreError(ShardReceiptError):
    """Raised when durable shard evidence is ambiguous or corrupt."""


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ShardedArtifactStoreError(
            f"Value is not canonical JSON: {exc}"
        ) from exc


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    """Return an exact JSON-domain copy and reject non-JSON Python values."""

    encoded = _stable_json(value)
    return json.loads(encoded)


def _require_sha256(value: Any, *, label: str) -> str:
    resolved = str(value or "")
    if _SHA256_RE.fullmatch(resolved) is None:
        raise ShardedArtifactStoreError(f"{label} is not a SHA-256 digest")
    return resolved


def _exact_json(raw_text: str, *, schema: Mapping[str, Any]) -> Any:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ShardedArtifactStoreError(
            "Accepted provider response is empty"
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw_text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ShardedArtifactStoreError(
            f"Accepted provider response is not one exact JSON value: {exc}"
        ) from exc
    try:
        validator = Draft202012Validator(dict(schema))
        validator.check_schema(dict(schema))
        validator.validate(parsed)
    except (SchemaError, ValidationError) as exc:
        raise ShardedArtifactStoreError(
            f"Accepted provider response violates the shard schema: {exc}"
        ) from exc
    return _json_copy(parsed)


def _stream_sha256(
    *, owner_artifact_key: str, document_id: str, plan_sha256: str
) -> str:
    return _stable_json_sha256(
        {
            "version": SHARDED_ARTIFACT_STORE_VERSION,
            "owner_artifact_key": owner_artifact_key,
            "document_id": document_id,
            "plan_sha256": plan_sha256,
        }
    )


def _provider_prefix(logical_request_sha256: str) -> str:
    _require_sha256(
        logical_request_sha256, label="Logical shard request digest"
    )
    return f"{_PROVIDER_KEY_PREFIX}{logical_request_sha256}_"


def _provider_key(
    logical_request_sha256: str, event_sha256: str
) -> str:
    _require_sha256(event_sha256, label="Provider event digest")
    return f"{_provider_prefix(logical_request_sha256)}{event_sha256}"


def _provider_ref(artifact_key: str) -> str:
    return f"run-artifact:{artifact_key}"


def _provider_key_from_ref(
    receipt_ref: str, *, logical_request_sha256: str
) -> str:
    prefix = "run-artifact:"
    if not isinstance(receipt_ref, str) or not receipt_ref.startswith(prefix):
        raise ShardedArtifactStoreError(
            "Provider receipt reference is invalid"
        )
    artifact_key = receipt_ref[len(prefix) :]
    expected_prefix = _provider_prefix(logical_request_sha256)
    if not artifact_key.startswith(expected_prefix):
        raise ShardedArtifactStoreError(
            "Provider receipt reference targets another artifact kind"
        )
    digest = artifact_key[len(expected_prefix) :]
    _require_sha256(digest, label="Provider receipt reference digest")
    return artifact_key


def _head_key(stream_sha256: str) -> str:
    return f"{_SHARD_HEAD_KEY_PREFIX}{stream_sha256}"


def _receipt_prefix(stream_sha256: str) -> str:
    _require_sha256(stream_sha256, label="Shard receipt stream digest")
    return f"{_SHARD_RECEIPT_KEY_PREFIX}{stream_sha256}_"


def _receipt_key(
    stream_sha256: str, *, receipt_sha256: str
) -> str:
    _require_sha256(receipt_sha256, label="Shard receipt digest")
    return f"{_receipt_prefix(stream_sha256)}{receipt_sha256}"


def _provider_contract(
    *,
    owner_artifact_key: str,
    owner_prompt_version: str,
    model: str,
    request: ShardRequest,
) -> dict[str, Any]:
    physical_base = _expected_physical_request_payload(
        request,
        max_completion_tokens=None,
    )
    core = {
        "version": SHARDED_PROVIDER_CONTRACT_VERSION,
        "owner_artifact_key": owner_artifact_key,
        "owner_prompt_version": owner_prompt_version,
        "model": model,
        "document_id": request.document_id,
        "plan_sha256": request.plan_sha256,
        "index": request.index,
        "shard_id": request.shard_id,
        "logical_request_sha256": request.request_sha256,
        "generation_contract_sha256": request.generation_contract_sha256,
        "shard_schema_sha256": request.shard_schema_sha256,
        "physical_request_base_sha256": _stable_json_sha256(physical_base),
    }
    return {**core, "sha256": _stable_json_sha256(core)}


def _generation_settings(request: ShardRequest) -> dict[str, Any]:
    """Validate the exact model-facing generation policy in the shard plan."""

    contract = request.generation_contract
    for key in ("model", "system_prompt", "prompt_template", "schema_name"):
        if not isinstance(contract.get(key), str) or not str(
            contract[key]
        ).strip():
            raise ShardedArtifactStoreError(
                f"Shard generation {key} must be non-empty text"
            )
    parameters = contract.get("parameters")
    web_policy = contract.get("web_policy")
    if not isinstance(parameters, dict) or not isinstance(web_policy, dict):
        raise ShardedArtifactStoreError(
            "Shard generation parameters/web policy are invalid"
        )
    allowed_parameters = {
        "temperature",
        "reasoning_effort",
        "output_token_policy",
    }
    unknown_parameters = sorted(set(parameters) - allowed_parameters)
    if unknown_parameters:
        raise ShardedArtifactStoreError(
            "Shard generation has unsupported physical parameters: "
            + ", ".join(unknown_parameters)
        )
    temperature = parameters.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= float(temperature) <= 2
    ):
        raise ShardedArtifactStoreError(
            "Shard generation temperature must be a number from 0 to 2"
        )
    reasoning_effort = parameters.get("reasoning_effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or not reasoning_effort.strip()
    ):
        raise ShardedArtifactStoreError(
            "Shard reasoning effort must be non-empty or null"
        )
    if parameters.get("output_token_policy") != "model_max_available":
        raise ShardedArtifactStoreError(
            "Shard generation must use model_max_available"
        )
    if web_policy != {"policy": "forbidden"}:
        raise ShardedArtifactStoreError(
            "Composable report shards must use the exact forbidden-web policy"
        )
    template = str(contract.get("prompt_template") or "")
    if "{{payload}}" not in template:
        raise ShardedArtifactStoreError(
            "Shard prompt template must include canonical {{payload}}"
        )
    allowed_markers = {
        "{{shard_id}}",
        "{{index}}",
        "{{shard_count}}",
        "{{payload}}",
    }
    template_markers = set(re.findall(r"\{\{[^{}]+\}\}", template))
    unsupported_markers = sorted(template_markers - allowed_markers)
    if unsupported_markers:
        raise ShardedArtifactStoreError(
            "Shard prompt template contains unsupported placeholders: "
            + ", ".join(unsupported_markers)
        )
    rendered = template
    replacements = {
        "{{shard_id}}": request.shard_id,
        "{{index}}": str(request.index),
        "{{shard_count}}": str(request.shard_count),
        "{{payload}}": _stable_json(request.payload),
    }
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    return {
        "messages": [
            {
                "role": "system",
                "content": str(contract["system_prompt"]),
            },
            {"role": "user", "content": rendered},
        ],
        "schema_name": str(contract["schema_name"]),
        "temperature": float(temperature),
        "reasoning_effort": reasoning_effort,
    }


def _expected_physical_request_payload(
    request: ShardRequest,
    *,
    max_completion_tokens: int | None,
) -> dict[str, Any]:
    settings = _generation_settings(request)
    payload: dict[str, Any] = {
        "model": str(request.generation_contract["model"]),
        "messages": settings["messages"],
        "temperature": settings["temperature"],
        "plugins": [{"id": "web", "enabled": False}],
        "tool_choice": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": settings["schema_name"],
                "strict": True,
                "schema": _json_copy(request.shard_schema),
            },
        },
    }
    reasoning_effort = settings["reasoning_effort"]
    if reasoning_effort is not None:
        payload["reasoning"] = {
            "effort": reasoning_effort,
            "exclude": True,
        }
    if max_completion_tokens is not None:
        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens <= 0
        ):
            raise ShardedArtifactStoreError(
                "Physical model-max completion value is invalid"
            )
        payload["max_completion_tokens"] = max_completion_tokens
    return payload


async def _assert_lease_in_session(session: Any, run_id: str) -> None:
    """Prove a bound lease inside the transaction that will be committed."""

    lease = current_run_lease()
    if lease is None:
        return
    if lease.run_id != run_id:
        raise RunLeaseLostError(
            f"Worker for run {lease.run_id} cannot write run {run_id}"
        )
    owned = (
        await session.execute(
            select(Run.id).where(
                Run.id == run_id,
                Run.execution_slot == 1,
                Run.lease_owner == lease.owner,
                Run.status.in_(
                    (
                        RunStatus.pending,
                        RunStatus.crawling,
                        RunStatus.analyzing,
                    )
                ),
            )
        )
    ).scalar_one_or_none()
    if owned is None:
        raise RunLeaseLostError(f"Run lease lost for {run_id}")


@dataclass(frozen=True, slots=True)
class ShardedArtifactStore:
    """Run-scoped callback bundle for :func:`compose_sharded_document`."""

    run_id: str
    stage_key: str
    owner_artifact_key: str
    model: str
    owner_prompt_version: str
    plan: ShardPlan

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "stage_key",
            "owner_artifact_key",
            "model",
            "owner_prompt_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ShardedArtifactStoreError(f"{name} must not be empty")
        if not isinstance(self.plan, ShardPlan):
            raise ShardedArtifactStoreError("plan must be a ShardPlan")

    def provider_audit_context(self, request: ShardRequest) -> dict[str, Any]:
        """Return the logical binding passed to ``openrouter.chat``."""

        self._validate_request(request)
        return {
            "document_id": request.document_id,
            "sequence": request.index,
            "resume_contract": _provider_contract(
                owner_artifact_key=self.owner_artifact_key,
                owner_prompt_version=self.owner_prompt_version,
                model=self.model,
                request=request,
            ),
        }

    def planned_requests(self) -> tuple[ShardRequest, ...]:
        """Return every exact logical request in bound-plan order."""

        return tuple(
            shard_request(self.plan, index)
            for index in range(len(self.plan.shards))
        )

    def provider_chat_arguments(self, request: ShardRequest) -> dict[str, Any]:
        """Return the exact keyword contract for ``openrouter.chat``.

        Callers may add operational retry flags and the two audit callbacks,
        but must not replace any returned model-facing field.  The durable
        callback reconstructs the resulting physical HTTP payload and rejects
        any prompt/schema/parameter drift.
        """

        self._validate_request(request)
        settings = _generation_settings(request)
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": copy.deepcopy(settings["messages"]),
            "response_schema": copy.deepcopy(request.shard_schema),
            "schema_name": settings["schema_name"],
            "web_policy": "forbidden",
            "output_token_policy": "model_max_available",
            "temperature": settings["temperature"],
        }
        if settings["reasoning_effort"] is not None:
            arguments["reasoning_effort"] = settings["reasoning_effort"]
        return arguments

    def provider_request_utf8_bytes(
        self,
        request: ShardRequest,
        *,
        max_completion_tokens: int | None,
    ) -> bytes:
        """Return the canonical physical POST body used by audit validation."""

        self._validate_request(request)
        payload = _expected_physical_request_payload(
            request,
            max_completion_tokens=max_completion_tokens,
        )
        return _stable_json(payload).encode("utf-8")

    def provider_audit_checkpoint(
        self, request: ShardRequest
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """Return an immutable physical-POST callback for ``openrouter.chat``."""

        self._validate_request(request)

        async def checkpoint(event: dict[str, Any]) -> None:
            await self.persist_provider_event(request, event)

        return checkpoint

    async def generate_or_resume(
        self,
        request: ShardRequest,
        provider_call: ProviderCall,
    ) -> GeneratedShard:
        """Reuse a paid accepted POST or execute exactly one logical call.

        ``provider_call`` receives ``(audit_checkpoint, audit_context)`` and
        normally calls :func:`app.services.openrouter.chat`.  The provider
        result is not trusted as persistence evidence: after it returns, this
        method reloads the durable POST receipt and binds the generated shard
        to that exact row.
        """

        self._validate_request(request)
        if not callable(provider_call):
            raise ShardedArtifactStoreError("provider_call must be callable")
        recovered = await self.promote_accepted_provider_response(request)
        if recovered is not None:
            return recovered
        outcome = provider_call(
            self.provider_audit_checkpoint(request),
            self.provider_audit_context(request),
        )
        if not inspect.isawaitable(outcome):
            raise ShardedArtifactStoreError(
                "provider_call must return an awaitable"
            )
        result = await outcome
        accepted = await self.promote_accepted_provider_response(request)
        if accepted is None:
            raise ShardedArtifactStoreError(
                "Provider returned without a durable accepted POST receipt"
            )
        returned_text = getattr(result, "text", None)
        if not isinstance(returned_text, str) or returned_text != (
            accepted.raw_text
        ):
            raise ShardedArtifactStoreError(
                "Provider return value does not match its durable POST receipt"
            )
        returned_parsed = getattr(result, "parsed", None)
        if returned_parsed is not None and _stable_json_sha256(
            returned_parsed
        ) != _stable_json_sha256(accepted.value):
            raise ShardedArtifactStoreError(
                "Provider parsed value does not match its exact raw response"
            )
        return accepted

    async def persist_provider_event(
        self, request: ShardRequest, event: dict[str, Any]
    ) -> None:
        """Persist one exact physical POST event idempotently."""

        self._validate_request(request)
        normalized = self._validated_provider_event(request, event)
        event_sha256 = _stable_json_sha256(normalized)
        artifact_key = _provider_key(request.request_sha256, event_sha256)
        raw_text = normalized.get("raw_text")
        raw_digest = (
            text_sha256(raw_text) if isinstance(raw_text, str) else None
        )
        event_input = {
            "version": SHARDED_PROVIDER_RECEIPT_VERSION,
            "row_kind": "physical_provider_post",
            "owner_artifact_key": self.owner_artifact_key,
            "owner_prompt_version": self.owner_prompt_version,
            "document_id": request.document_id,
            "plan_sha256": request.plan_sha256,
            "index": request.index,
            "logical_request_sha256": request.request_sha256,
            "event_sha256": event_sha256,
            "physical_request_sha256": normalized["request_sha256"],
            "raw_text_sha256": raw_digest,
        }
        await assert_run_lease(self.run_id)
        async with SessionLocal() as session:
            await _assert_lease_in_session(session, self.run_id)
            inserted = await session.execute(
                sqlite_insert(RunArtifact)
                .values(
                    run_id=self.run_id,
                    stage_key=self.stage_key,
                    artifact_key=artifact_key,
                    status="completed",
                    model=self.model,
                    prompt_version=SHARDED_ARTIFACT_STORE_VERSION,
                    input_json=event_input,
                    output_json=normalized,
                    raw_text=raw_text,
                    usage_json=(
                        copy.deepcopy(normalized.get("usage"))
                        if isinstance(normalized.get("usage"), dict)
                        else None
                    ),
                    error_message=(
                        str((normalized.get("error") or {}).get("message"))
                        if isinstance(normalized.get("error"), dict)
                        else None
                    ),
                )
                .on_conflict_do_nothing(
                    index_elements=("run_id", "artifact_key")
                )
            )
            if inserted.rowcount != 1:
                existing = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.artifact_key == artifact_key,
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    raise ShardedArtifactStoreError(
                        "Provider receipt collision cannot be resolved"
                    )
                self._provider_event_from_row(existing, request=request)
            await _assert_lease_in_session(session, self.run_id)
            await session.commit()

    async def promote_accepted_provider_response(
        self, request: ShardRequest
    ) -> GeneratedShard | None:
        """Resolve exactly one accepted, schema-valid paid POST, if present."""

        self._validate_request(request)
        await assert_run_lease(self.run_id)
        async with SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RunArtifact)
                        .where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.stage_key == self.stage_key,
                            RunArtifact.artifact_key.like(
                                f"{_provider_prefix(request.request_sha256)}%"
                            ),
                        )
                        .order_by(RunArtifact.id.asc())
                    )
                )
                .scalars()
                .all()
            )
        candidates: list[tuple[RunArtifact, dict[str, Any], Any]] = []
        for row in rows:
            stored_input = row.input_json
            if not isinstance(stored_input, dict):
                raise ShardedArtifactStoreError(
                    "Provider receipt identity is corrupt"
                )
            stored_event = row.output_json
            stored_contract = (
                stored_event.get("resume_contract")
                if isinstance(stored_event, dict)
                and isinstance(stored_event.get("resume_contract"), dict)
                else {}
            )
            event_claims_this_owner = stored_contract.get(
                "owner_artifact_key"
            ) == self.owner_artifact_key
            if event_claims_this_owner and (
                stored_input.get("owner_artifact_key")
                != self.owner_artifact_key
            ):
                raise ShardedArtifactStoreError(
                    "Provider receipt owner identity was mutated"
                )
            if stored_input.get("owner_artifact_key") != (
                self.owner_artifact_key
            ):
                continue
            if stored_input.get("owner_prompt_version") != (
                self.owner_prompt_version
            ):
                continue
            same_slot = (
                stored_input.get("document_id") == request.document_id
                and stored_input.get("plan_sha256") == request.plan_sha256
                and stored_input.get("index") == request.index
            )
            if not same_slot:
                continue
            if stored_input.get("logical_request_sha256") != (
                request.request_sha256
            ):
                raise ShardedArtifactStoreError(
                    "Provider receipt logical request identity was mutated"
                )
            event = self._provider_event_from_row(row, request=request)
            if event.get("status") != "accepted":
                continue
            raw_text = event.get("raw_text")
            if not isinstance(raw_text, str):
                raise ShardedArtifactStoreError(
                    "Accepted provider receipt has no exact response"
                )
            parsed = _exact_json(raw_text, schema=request.shard_schema)
            candidates.append((row, event, parsed))
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ShardedArtifactStoreError(
                "Ambiguous accepted provider receipts for one shard request"
            )
        row, event, parsed = candidates[0]
        raw_text = str(event["raw_text"])
        event_sha256 = _stable_json_sha256(event)
        binding = ProviderAuditBinding(
            event_id=str(event["event_id"]),
            receipt_ref=_provider_ref(row.artifact_key),
            receipt_sha256=event_sha256,
            physical_request_sha256=str(event["request_sha256"]),
            logical_request_sha256=request.request_sha256,
            raw_text_sha256=text_sha256(raw_text),
        )
        await self.verify_provider_audit(binding, request)
        return GeneratedShard(
            value=parsed,
            raw_text=raw_text,
            provider_audit=binding,
            metadata={
                "provider_event_id": event["event_id"],
                "provider_receipt_ref": binding.receipt_ref,
            },
        )

    async def verify_provider_audit(
        self, binding: ProviderAuditBinding, request: ShardRequest
    ) -> None:
        """Resolve and verify the durable physical POST bound to a shard."""

        self._validate_request(request)
        if not isinstance(binding, ProviderAuditBinding):
            raise ShardedArtifactStoreError(
                "Provider audit binding has an invalid type"
            )
        artifact_key = _provider_key_from_ref(
            binding.receipt_ref,
            logical_request_sha256=request.request_sha256,
        )
        await assert_run_lease(self.run_id)
        async with SessionLocal() as session:
            row = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == artifact_key,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            raise ShardedArtifactStoreError(
                "Provider audit receipt does not exist"
            )
        event = self._provider_event_from_row(row, request=request)
        raw_text = event.get("raw_text")
        if not isinstance(raw_text, str):
            raise ShardedArtifactStoreError(
                "Provider audit receipt has no exact raw response"
            )
        expected = {
            "event_id": str(event["event_id"]),
            "receipt_ref": _provider_ref(row.artifact_key),
            "receipt_sha256": _stable_json_sha256(event),
            "physical_request_sha256": str(event["request_sha256"]),
            "logical_request_sha256": request.request_sha256,
            "raw_text_sha256": text_sha256(raw_text),
        }
        for key, value in expected.items():
            if getattr(binding, key) != value:
                raise ShardedArtifactStoreError(
                    f"Provider audit binding mismatch: {key}"
                )
        if event.get("status") != "accepted":
            raise ShardedArtifactStoreError(
                "Provider audit binding does not reference an accepted POST"
            )
        _exact_json(raw_text, schema=request.shard_schema)

    async def load_receipts(
        self, document_id: str, plan_sha256: str
    ) -> list[dict[str, Any]]:
        """Load and verify the one ordered contiguous accepted-shard chain."""

        document_id, plan_sha256, stream_sha256 = self._stream_identity(
            document_id, plan_sha256
        )
        await assert_run_lease(self.run_id)
        async with SessionLocal() as session:
            head = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == _head_key(stream_sha256),
                    )
                )
            ).scalar_one_or_none()
            rows = list(
                (
                    await session.execute(
                        select(RunArtifact)
                        .where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.artifact_key.like(
                                f"{_receipt_prefix(stream_sha256)}%"
                            ),
                        )
                        .order_by(RunArtifact.id.asc())
                    )
                )
                .scalars()
                .all()
            )
        if head is None:
            if rows:
                raise ShardedArtifactStoreError(
                    "Shard receipts exist without their CAS head"
                )
            return []
        head_output = self._validated_head(
            head,
            document_id=document_id,
            plan_sha256=plan_sha256,
            stream_sha256=stream_sha256,
        )
        expected_count = head_output["next_index"]
        if len(rows) != expected_count:
            raise ShardedArtifactStoreError(
                "Shard receipt rows do not match the CAS head"
            )
        decoded = [
            self._receipt_from_row(
                row,
                document_id=document_id,
                plan_sha256=plan_sha256,
                stream_sha256=stream_sha256,
            )
            for row in rows
        ]
        decoded.sort(key=lambda item: item["index"])
        receipts: list[dict[str, Any]] = []
        predecessor: str | None = None
        for index, receipt in enumerate(decoded):
            if receipt.get("index") != index:
                raise ShardedArtifactStoreError(
                    "Shard receipt indexes are missing, duplicate, or reordered"
                )
            if receipt.get("predecessor_receipt_sha256") != predecessor:
                raise ShardedArtifactStoreError(
                    "Shard receipt predecessor chain is corrupt"
                )
            predecessor = _require_sha256(
                receipt.get("receipt_sha256"),
                label="Shard receipt digest",
            )
            receipts.append(receipt)
        if predecessor != head_output["head_receipt_sha256"]:
            raise ShardedArtifactStoreError(
                "Shard receipt chain does not terminate at the CAS head"
            )
        try:
            return list(
                verify_shard_receipts(
                    self.plan,
                    receipts,
                    require_complete=False,
                )
            )
        except ShardReceiptError as exc:
            raise ShardedArtifactStoreError(
                f"Stored shard receipt chain violates its plan: {exc}"
            ) from exc

    async def save_receipt(
        self,
        receipt: dict[str, Any],
        expected_predecessor: str | None,
    ) -> ShardSaveAck:
        """Atomically append one immutable shard receipt and advance its head."""

        normalized = self._validated_receipt_for_save(
            receipt, expected_predecessor=expected_predecessor
        )
        self._validate_receipt_plan_binding(normalized)
        document_id = str(normalized["document_id"])
        plan_sha256 = _require_sha256(
            normalized["plan_sha256"], label="Shard plan digest"
        )
        _, _, stream_sha256 = self._stream_identity(
            document_id, plan_sha256
        )
        index = int(normalized["index"])
        receipt_sha256 = str(normalized["receipt_sha256"])
        artifact_key = _receipt_key(
            stream_sha256,
            receipt_sha256=receipt_sha256,
        )
        row_input = {
            "version": SHARDED_RECEIPT_ROW_VERSION,
            "row_kind": "accepted_shard_receipt",
            "owner_artifact_key": self.owner_artifact_key,
            "owner_prompt_version": self.owner_prompt_version,
            "document_id": document_id,
            "plan_sha256": plan_sha256,
            "stream_sha256": stream_sha256,
            "index": index,
            "receipt_sha256": receipt_sha256,
            "row_sha256": _stable_json_sha256(normalized),
        }
        initial_head = self._head_output(
            document_id=document_id,
            plan_sha256=plan_sha256,
            stream_sha256=stream_sha256,
            next_index=0,
            head_receipt_sha256=None,
        )
        next_head = self._head_output(
            document_id=document_id,
            plan_sha256=plan_sha256,
            stream_sha256=stream_sha256,
            next_index=index + 1,
            head_receipt_sha256=receipt_sha256,
        )
        head_key = _head_key(stream_sha256)
        await assert_run_lease(self.run_id)
        async with SessionLocal() as session:
            await _assert_lease_in_session(session, self.run_id)
            await self._verify_receipt_provider_row(session, normalized)
            await session.execute(
                sqlite_insert(RunArtifact)
                .values(
                    run_id=self.run_id,
                    stage_key=self.stage_key,
                    artifact_key=head_key,
                    status="completed",
                    model=None,
                    prompt_version=SHARDED_ARTIFACT_STORE_VERSION,
                    input_json=self._head_input(
                        document_id=document_id,
                        plan_sha256=plan_sha256,
                        stream_sha256=stream_sha256,
                    ),
                    output_json=initial_head,
                )
                .on_conflict_do_nothing(
                    index_elements=("run_id", "artifact_key")
                )
            )
            head = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == head_key,
                    )
                )
            ).scalar_one_or_none()
            if head is None:
                raise ShardedArtifactStoreError(
                    "Shard CAS head could not be initialized"
                )
            current_head = self._validated_head(
                head,
                document_id=document_id,
                plan_sha256=plan_sha256,
                stream_sha256=stream_sha256,
            )
            if (
                current_head["next_index"] != index
                or current_head["head_receipt_sha256"]
                != expected_predecessor
            ):
                raise ShardedArtifactStoreError(
                    "Shard receipt compare-and-set predecessor mismatch"
                )
            inserted = await session.execute(
                sqlite_insert(RunArtifact)
                .values(
                    run_id=self.run_id,
                    stage_key=self.stage_key,
                    artifact_key=artifact_key,
                    status="completed",
                    model=self.model,
                    prompt_version=SHARDED_ARTIFACT_STORE_VERSION,
                    input_json=row_input,
                    output_json=normalized,
                    raw_text=str(normalized.get("raw_text") or ""),
                )
                .on_conflict_do_nothing(
                    index_elements=("run_id", "artifact_key")
                )
            )
            if inserted.rowcount != 1:
                raise ShardedArtifactStoreError(
                    "Immutable shard receipt already exists before CAS advance"
                )
            predicate = (
                update(RunArtifact)
                .where(
                    RunArtifact.id == head.id,
                    func.json_extract(
                        RunArtifact.output_json, "$.next_index"
                    )
                    == index,
                )
                .values(
                    output_json=next_head,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            if expected_predecessor is None:
                predicate = predicate.where(
                    func.json_extract(
                        RunArtifact.output_json,
                        "$.head_receipt_sha256",
                    ).is_(None)
                )
            else:
                predicate = predicate.where(
                    func.json_extract(
                        RunArtifact.output_json,
                        "$.head_receipt_sha256",
                    )
                    == expected_predecessor
                )
            advanced = await session.execute(predicate)
            if advanced.rowcount != 1:
                raise ShardedArtifactStoreError(
                    "Shard receipt CAS head advance lost a concurrent race"
                )
            await _assert_lease_in_session(session, self.run_id)
            await session.commit()
        return ShardSaveAck(
            document_id=document_id,
            plan_sha256=plan_sha256,
            index=index,
            receipt_id=str(normalized["receipt_id"]),
            receipt_sha256=receipt_sha256,
            predecessor_receipt_sha256=expected_predecessor,
        )

    def _validate_request(self, request: ShardRequest) -> None:
        if not isinstance(request, ShardRequest):
            raise ShardedArtifactStoreError(
                "request must be a ShardRequest"
            )
        if request.document_id != self.plan.document_id or (
            request.plan_sha256 != self.plan.plan_sha256
        ):
            raise ShardedArtifactStoreError(
                "Shard request belongs to another bound plan"
            )
        if (
            isinstance(request.index, bool)
            or not isinstance(request.index, int)
            or request.index < 0
            or request.index >= len(self.plan.shards)
        ):
            raise ShardedArtifactStoreError(
                "Shard request index is outside the bound plan"
            )
        canonical_request = shard_request(self.plan, request.index)
        if request.as_dict() != canonical_request.as_dict():
            raise ShardedArtifactStoreError(
                "Shard request differs from its bound plan"
            )
        if request.version != SHARD_REQUEST_VERSION:
            raise ShardedArtifactStoreError(
                "Shard request version is invalid"
            )
        for key in (
            "document_id",
            "plan_version",
            "merge_version",
            "shard_id",
        ):
            if not isinstance(getattr(request, key), str) or not getattr(
                request, key
            ).strip():
                raise ShardedArtifactStoreError(
                    f"Shard request {key} is invalid"
                )
        if request.composability != (
            ShardComposability.INDEPENDENT_DISJOINT.value
        ):
            raise ShardedArtifactStoreError(
                "Shard request composability is not allowlisted"
            )
        if (
            isinstance(request.index, bool)
            or not isinstance(request.index, int)
            or isinstance(request.shard_count, bool)
            or not isinstance(request.shard_count, int)
            or request.index < 0
            or request.shard_count <= 0
            or request.index >= request.shard_count
        ):
            raise ShardedArtifactStoreError(
                "Shard request coverage counters are invalid"
            )
        plan_sha256 = _require_sha256(
            request.plan_sha256, label="Shard plan digest"
        )
        for key in (
            "spec_sha256",
            "payload_sha256",
            "generation_contract_sha256",
            "merge_contract_sha256",
            "shard_schema_sha256",
            "request_sha256",
        ):
            _require_sha256(
                getattr(request, key), label=f"Shard request {key}"
            )
        payload = _json_copy(request.payload)
        payload_sha256 = _stable_json_sha256(payload)
        if payload_sha256 != request.payload_sha256:
            raise ShardedArtifactStoreError(
                "Shard request payload digest is inconsistent"
            )
        spec_core = {
            "document_id": request.document_id,
            "index": request.index,
            "shard_id": request.shard_id,
            "payload_sha256": payload_sha256,
            "payload": payload,
        }
        if _stable_json_sha256(spec_core) != request.spec_sha256:
            raise ShardedArtifactStoreError(
                "Shard request spec digest is inconsistent"
            )
        generation_contract = _json_copy(request.generation_contract)
        shard_schema = _json_copy(request.shard_schema)
        if (
            not isinstance(generation_contract, dict)
            or not isinstance(shard_schema, dict)
            or not shard_schema
        ):
            raise ShardedArtifactStoreError(
                "Shard request generation contract/schema is invalid"
            )
        try:
            Draft202012Validator.check_schema(shard_schema)
        except SchemaError as exc:
            raise ShardedArtifactStoreError(
                f"Shard request schema is invalid: {exc}"
            ) from exc
        if _stable_json_sha256(generation_contract) != (
            request.generation_contract_sha256
        ):
            raise ShardedArtifactStoreError(
                "Shard generation contract digest is inconsistent"
            )
        if _stable_json_sha256(shard_schema) != request.shard_schema_sha256:
            raise ShardedArtifactStoreError(
                "Shard schema digest is inconsistent"
            )
        request_core = {
            "version": SHARD_REQUEST_VERSION,
            "document_id": request.document_id,
            "plan_sha256": plan_sha256,
            "plan_version": request.plan_version,
            "merge_version": request.merge_version,
            "composability": request.composability,
            "shard_id": request.shard_id,
            "index": request.index,
            "shard_count": request.shard_count,
            "spec_sha256": request.spec_sha256,
            "payload_sha256": payload_sha256,
            "payload": payload,
            "generation_contract_sha256": (
                request.generation_contract_sha256
            ),
            "merge_contract_sha256": request.merge_contract_sha256,
            "shard_schema_sha256": request.shard_schema_sha256,
            "generation_contract": generation_contract,
            "shard_schema": shard_schema,
        }
        if _stable_json_sha256(request_core) != request.request_sha256:
            raise ShardedArtifactStoreError(
                "Logical shard request digest is inconsistent"
            )
        generation_model = request.generation_contract.get("model")
        if generation_model is not None and generation_model != self.model:
            raise ShardedArtifactStoreError(
                "Store model differs from the shard generation contract"
            )

    def _validated_provider_event(
        self, request: ShardRequest, event: Any
    ) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise ShardedArtifactStoreError(
                "Provider audit callback event must be an object"
            )
        normalized = _json_copy(event)
        if set(normalized) != _PHYSICAL_EVENT_KEYS:
            raise ShardedArtifactStoreError(
                "Provider audit event has an invalid exact shape"
            )
        if (
            normalized.get("version") != _PHYSICAL_POST_EVENT_VERSION
            or normalized.get("event_kind") != "provider_post"
        ):
            raise ShardedArtifactStoreError(
                "Provider audit event version/kind is unsupported"
            )
        event_id = str(normalized.get("event_id") or "")
        if _EVENT_ID_RE.fullmatch(event_id) is None:
            raise ShardedArtifactStoreError(
                "Provider audit event id is invalid"
            )
        logical_call_id = normalized.get("logical_call_id")
        attempt = normalized.get("attempt")
        if (
            not isinstance(logical_call_id, str)
            or not logical_call_id.strip()
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or normalized.get("partial_text") != ""
            or normalized.get("manifest") is not None
            or normalized.get("aggregate_usage") != {}
            or normalized.get("call_records") != []
        ):
            raise ShardedArtifactStoreError(
                "Provider audit physical-call identity is invalid"
            )
        if normalized.get("status") not in _ALLOWED_PROVIDER_STATUSES:
            raise ShardedArtifactStoreError(
                "Provider audit status is invalid"
            )
        if (
            normalized.get("model") != self.model
            or normalized.get("document_id") != request.document_id
            or normalized.get("sequence") != request.index
            or normalized.get("resume_contract")
            != self.provider_audit_context(request)["resume_contract"]
        ):
            raise ShardedArtifactStoreError(
                "Provider audit event belongs to another shard request"
            )
        request_payload = normalized.get("request_payload")
        if not isinstance(request_payload, dict):
            raise ShardedArtifactStoreError(
                "Provider audit event has no exact physical request"
            )
        request_digest = _require_sha256(
            normalized.get("request_sha256"),
            label="Physical provider request digest",
        )
        request_body = _stable_json(request_payload).encode("utf-8")
        if request_digest != hashlib.sha256(request_body).hexdigest():
            raise ShardedArtifactStoreError(
                "Physical provider request digest mismatch"
            )
        if (
            normalized.get("request_body_encoding")
            != "canonical-json-utf8-v1"
            or normalized.get("request_body_utf8_bytes") != len(request_body)
        ):
            raise ShardedArtifactStoreError(
                "Physical provider exact wire-body evidence is invalid"
            )
        self._validate_physical_request(request, normalized)
        raw_text = normalized.get("raw_text")
        if raw_text is not None and not isinstance(raw_text, str):
            raise ShardedArtifactStoreError(
                "Provider raw response must be text or null"
            )
        for field in (
            "response",
            "request_policy",
            "web_attestation",
            "router_metadata",
            "usage",
            "transport",
        ):
            if not isinstance(normalized.get(field), dict):
                raise ShardedArtifactStoreError(
                    f"Provider audit {field} must be an object"
                )
        for field in ("citations", "annotations"):
            if not isinstance(normalized.get(field), list):
                raise ShardedArtifactStoreError(
                    f"Provider audit {field} must be an array"
                )
        if normalized.get("status") == "accepted":
            transport = normalized["transport"]
            if (
                transport.get("output_complete") is not True
                or transport.get("output_limited") is not False
                or transport.get("http_status") != 200
                or normalized["response"].get("http_status") != 200
            ):
                raise ShardedArtifactStoreError(
                    "Accepted provider POST is not transport-complete"
                )
            if normalized.get("error") is not None:
                raise ShardedArtifactStoreError(
                    "Accepted provider POST unexpectedly carries an error"
                )
            _exact_json(str(raw_text or ""), schema=request.shard_schema)
        return normalized

    def _validate_physical_request(
        self, request: ShardRequest, event: Mapping[str, Any]
    ) -> None:
        """Prove the paid HTTP request is the plan's canonical generation."""

        payload = event.get("request_payload")
        usage = event.get("usage")
        if not isinstance(payload, dict) or not isinstance(usage, dict):
            raise ShardedArtifactStoreError(
                "Physical provider request/usage evidence is missing"
            )
        physical_max = payload.get("max_completion_tokens")
        if physical_max is not None and (
            isinstance(physical_max, bool)
            or not isinstance(physical_max, int)
            or physical_max <= 0
        ):
            raise ShardedArtifactStoreError(
                "Physical provider model-max value is invalid"
            )
        expected = _expected_physical_request_payload(
            request,
            max_completion_tokens=physical_max,
        )
        if payload != expected:
            raise ShardedArtifactStoreError(
                "Physical provider request does not match the shard "
                "generation contract"
            )
        envelope = usage.get("_aiv_output_envelope")
        if envelope is None and event.get("status") != "accepted":
            # Transport/HTTP failures can happen before OpenRouter has a
            # result usage envelope.  Their exact paid request is still
            # durably useful audit evidence, but they are never promotable.
            return
        if not isinstance(envelope, dict):
            raise ShardedArtifactStoreError(
                "Physical provider model-max envelope is missing"
            )
        if (
            envelope.get("policy") != "model_max_available"
            or envelope.get("requested_model") != self.model
            or envelope.get("effective_max_completion_tokens")
            != physical_max
        ):
            raise ShardedArtifactStoreError(
                "Physical provider model-max envelope is inconsistent"
            )
        estimate = envelope.get("request_estimate")
        if not isinstance(estimate, dict) or estimate.get(
            "request_sha256"
        ) != event.get("request_sha256"):
            raise ShardedArtifactStoreError(
                "Physical provider request-envelope digest is inconsistent"
            )

    def _provider_event_from_row(
        self, row: RunArtifact, *, request: ShardRequest
    ) -> dict[str, Any]:
        stored_input = row.input_json
        event = row.output_json
        if (
            row.status != "completed"
            or row.stage_key != self.stage_key
            or row.model != self.model
            or row.prompt_version != SHARDED_ARTIFACT_STORE_VERSION
            or not isinstance(stored_input, dict)
            or not isinstance(event, dict)
        ):
            raise ShardedArtifactStoreError(
                "Provider receipt row shape is corrupt"
            )
        normalized = self._validated_provider_event(request, event)
        event_sha256 = _stable_json_sha256(normalized)
        raw_text = normalized.get("raw_text")
        raw_digest = (
            text_sha256(raw_text) if isinstance(raw_text, str) else None
        )
        expected_input = {
            "version": SHARDED_PROVIDER_RECEIPT_VERSION,
            "row_kind": "physical_provider_post",
            "owner_artifact_key": self.owner_artifact_key,
            "owner_prompt_version": self.owner_prompt_version,
            "document_id": request.document_id,
            "plan_sha256": request.plan_sha256,
            "index": request.index,
            "logical_request_sha256": request.request_sha256,
            "event_sha256": event_sha256,
            "physical_request_sha256": normalized["request_sha256"],
            "raw_text_sha256": raw_digest,
        }
        if stored_input != expected_input:
            raise ShardedArtifactStoreError(
                "Provider receipt identity or hash is corrupt"
            )
        if row.artifact_key != _provider_key(
            request.request_sha256, event_sha256
        ):
            raise ShardedArtifactStoreError(
                "Provider receipt content address is corrupt"
            )
        if row.raw_text != raw_text:
            raise ShardedArtifactStoreError(
                "Provider receipt raw response was mutated"
            )
        if row.usage_json != normalized.get("usage"):
            raise ShardedArtifactStoreError(
                "Provider receipt usage evidence was mutated"
            )
        expected_error_message = (
            str((normalized.get("error") or {}).get("message"))
            if isinstance(normalized.get("error"), dict)
            else None
        )
        if row.error_message != expected_error_message:
            raise ShardedArtifactStoreError(
                "Provider receipt error evidence was mutated"
            )
        return normalized

    def _stream_identity(
        self, document_id: str, plan_sha256: str
    ) -> tuple[str, str, str]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ShardedArtifactStoreError("document_id must not be empty")
        plan_sha256 = _require_sha256(
            plan_sha256, label="Shard plan digest"
        )
        if document_id != self.plan.document_id or (
            plan_sha256 != self.plan.plan_sha256
        ):
            raise ShardedArtifactStoreError(
                "Receipt callbacks belong to another bound plan"
            )
        return (
            document_id,
            plan_sha256,
            _stream_sha256(
                owner_artifact_key=self.owner_artifact_key,
                document_id=document_id,
                plan_sha256=plan_sha256,
            ),
        )

    def _head_input(
        self, *, document_id: str, plan_sha256: str, stream_sha256: str
    ) -> dict[str, Any]:
        return {
            "version": SHARDED_RECEIPT_HEAD_VERSION,
            "row_kind": "accepted_shard_head",
            "owner_artifact_key": self.owner_artifact_key,
            "owner_prompt_version": self.owner_prompt_version,
            "document_id": document_id,
            "plan_sha256": plan_sha256,
            "stream_sha256": stream_sha256,
        }

    def _head_output(
        self,
        *,
        document_id: str,
        plan_sha256: str,
        stream_sha256: str,
        next_index: int,
        head_receipt_sha256: str | None,
    ) -> dict[str, Any]:
        core = {
            "version": SHARDED_RECEIPT_HEAD_VERSION,
            "row_kind": "accepted_shard_head",
            "document_id": document_id,
            "plan_sha256": plan_sha256,
            "stream_sha256": stream_sha256,
            "next_index": next_index,
            "head_receipt_sha256": head_receipt_sha256,
        }
        return {**core, "head_state_sha256": _stable_json_sha256(core)}

    def _validated_head(
        self,
        row: RunArtifact,
        *,
        document_id: str,
        plan_sha256: str,
        stream_sha256: str,
    ) -> dict[str, Any]:
        expected_input = self._head_input(
            document_id=document_id,
            plan_sha256=plan_sha256,
            stream_sha256=stream_sha256,
        )
        output = row.output_json
        if (
            row.status != "completed"
            or row.stage_key != self.stage_key
            or row.model is not None
            or row.prompt_version != SHARDED_ARTIFACT_STORE_VERSION
            or row.input_json != expected_input
            or not isinstance(output, dict)
            or row.raw_text is not None
            or row.usage_json is not None
            or row.error_message is not None
        ):
            raise ShardedArtifactStoreError("Shard CAS head is corrupt")
        next_index = output.get("next_index")
        head_digest = output.get("head_receipt_sha256")
        if (
            isinstance(next_index, bool)
            or not isinstance(next_index, int)
            or next_index < 0
            or (next_index == 0 and head_digest is not None)
            or (
                next_index > 0
                and _SHA256_RE.fullmatch(str(head_digest or "")) is None
            )
        ):
            raise ShardedArtifactStoreError(
                "Shard CAS head state is invalid"
            )
        core = {key: value for key, value in output.items() if key != "head_state_sha256"}
        if (
            set(output) != set(core) | {"head_state_sha256"}
            or output.get("head_state_sha256") != _stable_json_sha256(core)
            or core
            != {
                "version": SHARDED_RECEIPT_HEAD_VERSION,
                "row_kind": "accepted_shard_head",
                "document_id": document_id,
                "plan_sha256": plan_sha256,
                "stream_sha256": stream_sha256,
                "next_index": next_index,
                "head_receipt_sha256": head_digest,
            }
        ):
            raise ShardedArtifactStoreError(
                "Shard CAS head digest or identity is corrupt"
            )
        return copy.deepcopy(output)

    def _validated_receipt_for_save(
        self,
        receipt: Any,
        *,
        expected_predecessor: str | None,
    ) -> dict[str, Any]:
        if not isinstance(receipt, dict):
            raise ShardedArtifactStoreError("Shard receipt must be an object")
        normalized = _json_copy(receipt)
        if set(normalized) != _SHARD_RECEIPT_KEYS:
            raise ShardedArtifactStoreError(
                "Shard receipt has an invalid exact shape"
            )
        if (
            normalized.get("version") != SHARD_RECEIPT_VERSION
            or normalized.get("receipt_kind") != "accepted_shard"
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt version/kind is invalid"
            )
        index = normalized.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ShardedArtifactStoreError("Shard receipt index is invalid")
        shard_count = normalized.get("shard_count")
        if (
            isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count <= 0
            or index >= shard_count
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt coverage counters are invalid"
            )
        for key in (
            "plan_version",
            "merge_version",
            "composability",
            "shard_id",
        ):
            if not isinstance(normalized.get(key), str) or not str(
                normalized[key]
            ).strip():
                raise ShardedArtifactStoreError(
                    f"Shard receipt {key} is invalid"
                )
        for key in (
            "generation_contract_sha256",
            "merge_contract_sha256",
            "shard_schema_sha256",
            "document_schema_sha256",
            "spec_sha256",
            "payload_sha256",
            "request_sha256",
            "content_sha256",
            "raw_text_sha256",
        ):
            _require_sha256(
                normalized.get(key), label=f"Shard receipt {key}"
            )
        content = normalized.get("content")
        if normalized["content_sha256"] != _stable_json_sha256(content):
            raise ShardedArtifactStoreError(
                "Shard receipt content digest is invalid"
            )
        raw_text = normalized.get("raw_text")
        if (
            not isinstance(raw_text, str)
            or normalized["raw_text_sha256"] != text_sha256(raw_text)
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt raw response digest is invalid"
            )
        parsed_raw = _exact_json(raw_text, schema={})
        if _stable_json_sha256(parsed_raw) != normalized["content_sha256"]:
            raise ShardedArtifactStoreError(
                "Shard receipt raw response does not equal its content"
            )
        metadata = normalized.get("metadata")
        if not isinstance(metadata, dict):
            raise ShardedArtifactStoreError(
                "Shard receipt metadata must be an object"
            )
        provider_audit = normalized.get("provider_audit")
        expected_audit_keys = {
            "version",
            "event_id",
            "receipt_ref",
            "receipt_sha256",
            "physical_request_sha256",
            "logical_request_sha256",
            "raw_text_sha256",
        }
        if (
            not isinstance(provider_audit, dict)
            or set(provider_audit) != expected_audit_keys
            or provider_audit.get("version")
            != PROVIDER_AUDIT_BINDING_VERSION
            or not isinstance(provider_audit.get("event_id"), str)
            or not provider_audit["event_id"].strip()
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt provider audit binding is invalid"
            )
        for key in (
            "receipt_sha256",
            "physical_request_sha256",
            "logical_request_sha256",
            "raw_text_sha256",
        ):
            _require_sha256(
                provider_audit.get(key),
                label=f"Shard provider audit {key}",
            )
        if (
            provider_audit["logical_request_sha256"]
            != normalized["request_sha256"]
            or provider_audit["raw_text_sha256"]
            != normalized["raw_text_sha256"]
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt provider audit is bound to another response"
            )
        _provider_key_from_ref(
            str(provider_audit.get("receipt_ref") or ""),
            logical_request_sha256=normalized["request_sha256"],
        )
        receipt_sha256 = _require_sha256(
            normalized.get("receipt_sha256"),
            label="Shard receipt digest",
        )
        receipt_core = {
            key: normalized[key] for key in _SHARD_RECEIPT_CORE_KEYS
        }
        if receipt_sha256 != _stable_json_sha256(receipt_core):
            raise ShardedArtifactStoreError(
                "Shard receipt digest does not match its exact content"
            )
        if normalized.get("receipt_id") != f"sha256:{receipt_sha256}":
            raise ShardedArtifactStoreError(
                "Shard receipt content address is invalid"
            )
        if expected_predecessor is not None:
            _require_sha256(
                expected_predecessor,
                label="Expected predecessor receipt digest",
            )
        if normalized.get("predecessor_receipt_sha256") != (
            expected_predecessor
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt does not bind the expected predecessor"
            )
        if index == 0 and expected_predecessor is not None:
            raise ShardedArtifactStoreError(
                "First shard receipt cannot have a predecessor"
            )
        if index > 0 and expected_predecessor is None:
            raise ShardedArtifactStoreError(
                "Non-initial shard receipt requires a predecessor"
            )
        if not isinstance(normalized.get("document_id"), str) or not (
            normalized["document_id"].strip()
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt document id is invalid"
            )
        _require_sha256(
            normalized.get("plan_sha256"), label="Shard plan digest"
        )
        return normalized

    def _validate_receipt_plan_binding(
        self, receipt: Mapping[str, Any]
    ) -> None:
        index = receipt["index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self.plan.shards)
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt index is outside the bound plan"
            )
        request = shard_request(self.plan, index)
        expected = {
            "document_id": self.plan.document_id,
            "plan_sha256": self.plan.plan_sha256,
            "plan_version": self.plan.plan_version,
            "merge_version": self.plan.merge_version,
            "composability": self.plan.composability,
            "generation_contract_sha256": (
                self.plan.generation_contract_sha256
            ),
            "merge_contract_sha256": self.plan.merge_contract_sha256,
            "shard_schema_sha256": self.plan.shard_schema_sha256,
            "document_schema_sha256": self.plan.document_schema_sha256,
            "index": index,
            "shard_id": request.shard_id,
            "shard_count": len(self.plan.shards),
            "spec_sha256": request.spec_sha256,
            "payload_sha256": request.payload_sha256,
            "request_sha256": request.request_sha256,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ShardedArtifactStoreError(
                    f"Shard receipt differs from its bound plan: {key}"
                )

    async def _verify_receipt_provider_row(
        self, session: Any, receipt: Mapping[str, Any]
    ) -> None:
        """Require the shard receipt's accepted physical POST in this txn."""

        provider_audit = receipt["provider_audit"]
        logical_request_sha256 = str(receipt["request_sha256"])
        artifact_key = _provider_key_from_ref(
            str(provider_audit["receipt_ref"]),
            logical_request_sha256=logical_request_sha256,
        )
        row = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == self.run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise ShardedArtifactStoreError(
                "Shard receipt references a missing provider POST"
            )
        request = shard_request(self.plan, int(receipt["index"]))
        normalized_event = self._provider_event_from_row(
            row, request=request
        )
        event_sha256 = _stable_json_sha256(normalized_event)
        raw_text = normalized_event.get("raw_text")
        contract = normalized_event.get("resume_contract")
        if (
            normalized_event.get("version") != _PHYSICAL_POST_EVENT_VERSION
            or normalized_event.get("event_kind") != "provider_post"
            or normalized_event.get("status") != "accepted"
            or normalized_event.get("document_id") != receipt["document_id"]
            or normalized_event.get("sequence") != receipt["index"]
            or normalized_event.get("model") != self.model
            or not isinstance(raw_text, str)
            or raw_text != receipt["raw_text"]
            or not isinstance(contract, dict)
            or contract.get("plan_sha256") != receipt["plan_sha256"]
            or contract.get("logical_request_sha256")
            != logical_request_sha256
            or contract.get("generation_contract_sha256")
            != receipt["generation_contract_sha256"]
            or contract.get("shard_schema_sha256")
            != receipt["shard_schema_sha256"]
        ):
            raise ShardedArtifactStoreError(
                "Shard receipt provider POST binding is inconsistent"
            )
        expected_binding = {
            "version": PROVIDER_AUDIT_BINDING_VERSION,
            "event_id": normalized_event.get("event_id"),
            "receipt_ref": _provider_ref(row.artifact_key),
            "receipt_sha256": event_sha256,
            "physical_request_sha256": normalized_event.get(
                "request_sha256"
            ),
            "logical_request_sha256": logical_request_sha256,
            "raw_text_sha256": text_sha256(raw_text),
        }
        if provider_audit != expected_binding:
            raise ShardedArtifactStoreError(
                "Shard receipt provider audit binding was substituted"
            )

    def _receipt_from_row(
        self,
        row: RunArtifact,
        *,
        document_id: str,
        plan_sha256: str,
        stream_sha256: str,
    ) -> dict[str, Any]:
        stored_input = row.input_json
        receipt = row.output_json
        if (
            row.status != "completed"
            or row.stage_key != self.stage_key
            or row.model != self.model
            or row.prompt_version != SHARDED_ARTIFACT_STORE_VERSION
            or not isinstance(stored_input, dict)
            or not isinstance(receipt, dict)
        ):
            raise ShardedArtifactStoreError(
                "Accepted shard receipt row is corrupt"
            )
        normalized = self._validated_receipt_for_save(
            receipt,
            expected_predecessor=receipt.get(
                "predecessor_receipt_sha256"
            ),
        )
        index = normalized.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ShardedArtifactStoreError(
                "Accepted shard receipt index is corrupt"
            )
        receipt_sha256 = _require_sha256(
            normalized.get("receipt_sha256"),
            label="Accepted shard receipt digest",
        )
        receipt_core = {
            key: normalized[key] for key in _SHARD_RECEIPT_CORE_KEYS
        }
        if (
            normalized.get("receipt_id") != f"sha256:{receipt_sha256}"
            or receipt_sha256 != _stable_json_sha256(receipt_core)
        ):
            raise ShardedArtifactStoreError(
                "Accepted shard receipt content digest is corrupt"
            )
        expected_input = {
            "version": SHARDED_RECEIPT_ROW_VERSION,
            "row_kind": "accepted_shard_receipt",
            "owner_artifact_key": self.owner_artifact_key,
            "owner_prompt_version": self.owner_prompt_version,
            "document_id": document_id,
            "plan_sha256": plan_sha256,
            "stream_sha256": stream_sha256,
            "index": index,
            "receipt_sha256": receipt_sha256,
            "row_sha256": _stable_json_sha256(normalized),
        }
        if stored_input != expected_input:
            raise ShardedArtifactStoreError(
                "Accepted shard receipt row digest or identity is corrupt"
            )
        if row.artifact_key != _receipt_key(
            stream_sha256,
            receipt_sha256=receipt_sha256,
        ):
            raise ShardedArtifactStoreError(
                "Accepted shard receipt content address is corrupt"
            )
        if normalized.get("document_id") != document_id or (
            normalized.get("plan_sha256") != plan_sha256
        ):
            raise ShardedArtifactStoreError(
                "Accepted shard receipt belongs to another stream"
            )
        if row.raw_text != str(normalized.get("raw_text") or ""):
            raise ShardedArtifactStoreError(
                "Accepted shard receipt raw response was mutated"
            )
        if row.usage_json is not None or row.error_message is not None:
            raise ShardedArtifactStoreError(
                "Accepted shard receipt secondary evidence was mutated"
            )
        return normalized


def create_sharded_artifact_store(
    *,
    run_id: str,
    stage_key: str,
    owner_artifact_key: str,
    model: str,
    owner_prompt_version: str,
    plan: ShardPlan,
) -> ShardedArtifactStore:
    """Construct the callback bundle used by analyzer stages."""

    return ShardedArtifactStore(
        run_id=run_id,
        stage_key=stage_key,
        owner_artifact_key=owner_artifact_key,
        model=model,
        owner_prompt_version=owner_prompt_version,
        plan=plan,
    )


__all__ = [
    "SHARDED_ARTIFACT_STORE_VERSION",
    "ShardedArtifactStore",
    "ShardedArtifactStoreError",
    "create_sharded_artifact_store",
]
