"""Crash-safe harness for genuinely composable long structured documents.

Unlike literal continuation, this harness never sends an accumulated document
prefix to a shard generator.  Code owns a finite deterministic shard plan;
every shard is a complete JSON value validated against the same schema, and a
versioned code callback merges the ordered values only after exact coverage is
proven.

There is deliberately no shard-count or total-document-length ceiling.  An
optional wall-clock deadline is a liveness safeguard, not a content budget.
Completed shard receipts are immutable and content-addressed, so a restarted
caller can load the contiguous saved prefix and continue at the first missing
shard without regenerating accepted work.

This mechanism is *not* valid for atomic decisions, panel observations, votes,
rankings, or verdicts whose meaning depends on one indivisible model turn.
Those calls must remain ``ResponseMode.ATOMIC`` and fail visibly if they do not
fit the physical model envelope.

Persistence is callback-based on purpose.  This module imports no database,
OpenRouter, analyzer, or artifact service, avoiding circular dependencies.
Callers may back ``load_receipts``/``save_receipt`` with RunArtifact rows or
another immutable store.  A provider transport can additionally use its own
physical-POST audit callback to close the paid-response/save crash gap.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from app.services.long_response import ResponseMode, text_sha256


SHARDED_DOCUMENT_VERSION = "aiv-sharded-document-v1"
SHARD_PLAN_VERSION = "aiv-shard-plan-v1"
SHARD_RECEIPT_VERSION = "aiv-shard-receipt-v1"
SHARD_REQUEST_VERSION = "aiv-shard-request-v1"
PROVIDER_AUDIT_BINDING_VERSION = "aiv-provider-audit-binding-v1"
SHARD_SAVE_ACK_VERSION = "aiv-shard-save-ack-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ShardComposability(str, Enum):
    """Semantics the harness is allowed to compose.

    Keeping this an allowlist prevents callers from relabelling an atomic
    judgement as generic ``partitioned`` work merely to bypass an output
    envelope.
    """

    INDEPENDENT_DISJOINT = "independent_disjoint_shards"


class ShardedDocumentError(ValueError):
    """Base class for a rejected plan, receipt, shard, or merge."""


class AtomicShardingUnsupportedError(ShardedDocumentError):
    """Raised when an atomic decision is incorrectly routed to this harness."""


class ShardPlanError(ShardedDocumentError):
    """Raised when the code-owned coverage plan is invalid."""


class ShardReceiptError(ShardedDocumentError):
    """Raised when persisted shard evidence is incomplete or inconsistent."""


class ShardSchemaError(ShardedDocumentError):
    """Raised when one shard or the merged document violates its schema."""


class ShardMergeError(ShardedDocumentError):
    """Raised when a code-owned merge is invalid or non-deterministic."""


class ShardedDocumentLivenessError(TimeoutError):
    """Raised when the optional logical wall-clock deadline expires."""


@dataclass(frozen=True)
class ShardSpec:
    """One code-owned independent work item.

    ``payload`` contains only the context required for this shard.  The harness
    never adds earlier shard outputs to it.
    """

    shard_id: str
    payload: Any


@dataclass(frozen=True)
class _PreparedShard:
    shard_id: str
    index: int
    payload: Any
    payload_sha256: str
    spec_sha256: str


@dataclass(frozen=True)
class ShardPlan:
    document_id: str
    plan_version: str
    merge_version: str
    composability: str
    generation_contract: dict[str, Any]
    merge_contract: dict[str, Any]
    generation_contract_sha256: str
    merge_contract_sha256: str
    empty_plan_reason: str | None
    shard_schema: dict[str, Any]
    document_schema: dict[str, Any]
    shard_schema_sha256: str
    document_schema_sha256: str
    shards: tuple[_PreparedShard, ...]
    manifest: dict[str, Any]

    @property
    def plan_sha256(self) -> str:
        return str(self.manifest["plan_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.manifest)


@dataclass(frozen=True)
class ShardRequest:
    """The complete request contract for exactly one independent shard."""

    version: str
    document_id: str
    plan_sha256: str
    plan_version: str
    merge_version: str
    composability: str
    shard_id: str
    index: int
    shard_count: int
    spec_sha256: str
    payload_sha256: str
    payload: Any
    generation_contract_sha256: str
    merge_contract_sha256: str
    shard_schema_sha256: str
    generation_contract: dict[str, Any]
    shard_schema: dict[str, Any]
    request_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "document_id": self.document_id,
            "plan_sha256": self.plan_sha256,
            "plan_version": self.plan_version,
            "merge_version": self.merge_version,
            "composability": self.composability,
            "shard_id": self.shard_id,
            "index": self.index,
            "shard_count": self.shard_count,
            "spec_sha256": self.spec_sha256,
            "payload_sha256": self.payload_sha256,
            "payload": copy.deepcopy(self.payload),
            "generation_contract_sha256": self.generation_contract_sha256,
            "merge_contract_sha256": self.merge_contract_sha256,
            "shard_schema_sha256": self.shard_schema_sha256,
            "generation_contract": copy.deepcopy(self.generation_contract),
            "shard_schema": copy.deepcopy(self.shard_schema),
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True, kw_only=True)
class ProviderAuditBinding:
    """Content-addressed durable evidence for the paid physical POST.

    ``receipt_ref`` locates the immutable provider receipt.  Its content hash,
    physical request hash, exact raw response hash, and logical shard request
    hash are all bound into the accepted shard receipt.  The required verifier
    callback must resolve this reference against durable storage; this object
    alone is not treated as proof.
    """

    event_id: str
    receipt_ref: str
    receipt_sha256: str
    physical_request_sha256: str
    logical_request_sha256: str
    raw_text_sha256: str
    version: str = PROVIDER_AUDIT_BINDING_VERSION

    def as_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "receipt_ref": self.receipt_ref,
            "receipt_sha256": self.receipt_sha256,
            "physical_request_sha256": self.physical_request_sha256,
            "logical_request_sha256": self.logical_request_sha256,
            "raw_text_sha256": self.raw_text_sha256,
        }


@dataclass(frozen=True, kw_only=True)
class ShardSaveAck:
    """Proof returned after an atomic compare-and-set head advance."""

    document_id: str
    plan_sha256: str
    index: int
    receipt_id: str
    receipt_sha256: str
    predecessor_receipt_sha256: str | None
    version: str = SHARD_SAVE_ACK_VERSION


@dataclass(frozen=True)
class GeneratedShard:
    """One independently valid generator result.

    ``raw_text`` is the exact provider JSON and must parse as one value, with
    no trailing prose, equal to ``value``.  ``provider_audit`` must reference
    an already-durable physical-POST receipt.  The harness never synthesizes
    provider provenance.
    """

    value: Any
    raw_text: str
    provider_audit: ProviderAuditBinding
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ShardedDocumentResult:
    document: Any
    document_sha256: str
    manifest: dict[str, Any]
    receipts: tuple[dict[str, Any], ...]
    resumed_shards: int
    generated_shards: int


def _assert_json_value(value: Any, *, label: str, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ShardedDocumentError(
                f"{label} contains a non-finite number at {path}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, label=label, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ShardedDocumentError(
                    f"{label} contains a non-string key at {path}"
                )
            _assert_json_value(
                item,
                label=label,
                path=f"{path}.{key}",
            )
        return
    raise ShardedDocumentError(
        f"{label} contains a non-JSON value at {path}: "
        f"{type(value).__name__}"
    )


def _canonical_json(value: Any, *, label: str) -> Any:
    _assert_json_value(value, label=label)
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _stable_json_sha256(value: Any, *, label: str = "JSON value") -> str:
    canonical = _canonical_json(value, label=label)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validated_schema(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], Draft202012Validator]:
    if not isinstance(value, Mapping):
        raise ShardPlanError(f"{label} must be a JSON object")
    schema = _canonical_json(dict(value), label=label)
    if not isinstance(schema, dict) or not schema:
        raise ShardPlanError(f"{label} must be a non-empty JSON object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ShardPlanError(f"{label} is invalid: {exc.message}") from exc
    return schema, Draft202012Validator(schema)


def _schema_validate(
    validator: Draft202012Validator,
    value: Any,
    *,
    label: str,
) -> None:
    try:
        validator.validate(value)
    except ValidationError as exc:
        path = "".join(f"[{part!r}]" for part in exc.absolute_path)
        raise ShardSchemaError(
            f"{label} violates its schema{path}: {exc.message}"
        ) from exc


def _response_mode(value: ResponseMode | str) -> ResponseMode:
    try:
        mode = ResponseMode(value)
    except ValueError as exc:
        raise ShardPlanError(f"Unknown response mode: {value}") from exc
    if mode is ResponseMode.ATOMIC:
        raise AtomicShardingUnsupportedError(
            "Atomic decisions cannot be sharded; use one atomic model turn"
        )
    if mode is not ResponseMode.PARTITIONED:
        raise ShardPlanError(
            "The sharded document harness supports partitioned mode only"
        )
    return mode


def _composability(value: ShardComposability | str) -> ShardComposability:
    try:
        return ShardComposability(value)
    except ValueError as exc:
        raise ShardPlanError(
            "Only explicitly allowlisted independent-disjoint shard "
            "composition is supported"
        ) from exc


def _generation_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ShardPlanError("generation_contract must be a JSON object")
    contract = _canonical_json(dict(value), label="generation_contract")
    required = {
        "model",
        "system_prompt",
        "prompt_template",
        "parameters",
        "web_policy",
        "schema_name",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ShardPlanError(
            "generation_contract is missing exact generation fields: "
            + ", ".join(missing)
        )
    for key in ("model", "system_prompt", "prompt_template", "schema_name"):
        if not isinstance(contract[key], str) or not contract[key].strip():
            raise ShardPlanError(
                f"generation_contract.{key} must be a non-empty string"
            )
    for key in ("parameters", "web_policy"):
        if not isinstance(contract[key], dict):
            raise ShardPlanError(
                f"generation_contract.{key} must be a JSON object"
            )
    return contract


def _merge_contract(
    value: Mapping[str, Any],
    *,
    merge_version: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ShardPlanError("merge_contract must be a JSON object")
    contract = _canonical_json(dict(value), label="merge_contract")
    if not contract:
        raise ShardPlanError("merge_contract must not be empty")
    if contract.get("version") != merge_version:
        raise ShardPlanError(
            "merge_contract.version must exactly match merge_version"
        )
    if not isinstance(contract.get("algorithm"), str) or not str(
        contract["algorithm"]
    ).strip():
        raise ShardPlanError(
            "merge_contract.algorithm must be a non-empty string"
        )
    return contract


def build_shard_plan(
    *,
    document_id: str,
    shards: Iterable[ShardSpec],
    shard_schema: Mapping[str, Any],
    document_schema: Mapping[str, Any],
    plan_version: str,
    merge_version: str,
    generation_contract: Mapping[str, Any],
    merge_contract: Mapping[str, Any],
    response_mode: ResponseMode | str,
    composability: ShardComposability | str,
    empty_plan_reason: str | None = None,
) -> ShardPlan:
    """Build the exact finite coverage contract without imposing a count cap."""

    _response_mode(response_mode)
    resolved_composability = _composability(composability)
    resolved_document_id = str(document_id or "").strip()
    resolved_plan_version = str(plan_version or "").strip()
    resolved_merge_version = str(merge_version or "").strip()
    if not resolved_document_id:
        raise ShardPlanError("document_id must not be empty")
    if not resolved_plan_version:
        raise ShardPlanError("plan_version must not be empty")
    if not resolved_merge_version:
        raise ShardPlanError("merge_version must not be empty")
    generation_contract_value = _generation_contract(generation_contract)
    merge_contract_value = _merge_contract(
        merge_contract,
        merge_version=resolved_merge_version,
    )
    generation_contract_sha256 = _stable_json_sha256(
        generation_contract_value,
        label="generation_contract",
    )
    merge_contract_sha256 = _stable_json_sha256(
        merge_contract_value,
        label="merge_contract",
    )
    shard_schema_value, _shard_validator = _validated_schema(
        shard_schema,
        label="shard_schema",
    )
    document_schema_value, _document_validator = _validated_schema(
        document_schema,
        label="document_schema",
    )
    shard_schema_sha256 = _stable_json_sha256(
        shard_schema_value,
        label="shard_schema",
    )
    document_schema_sha256 = _stable_json_sha256(
        document_schema_value,
        label="document_schema",
    )
    prepared: list[_PreparedShard] = []
    seen_ids: set[str] = set()
    for index, spec in enumerate(shards):
        if not isinstance(spec, ShardSpec):
            raise ShardPlanError(
                f"Shard {index} must be a ShardSpec"
            )
        shard_id = str(spec.shard_id or "").strip()
        if not shard_id:
            raise ShardPlanError(f"Shard {index} has an empty shard_id")
        if shard_id in seen_ids:
            raise ShardPlanError(f"Duplicate shard_id: {shard_id}")
        seen_ids.add(shard_id)
        try:
            payload = _canonical_json(
                spec.payload,
                label=f"shard {shard_id} payload",
            )
        except ShardedDocumentError as exc:
            raise ShardPlanError(str(exc)) from exc
        payload_sha256 = _stable_json_sha256(
            payload,
            label=f"shard {shard_id} payload",
        )
        spec_payload = {
            "document_id": resolved_document_id,
            "index": index,
            "shard_id": shard_id,
            "payload_sha256": payload_sha256,
            "payload": payload,
        }
        prepared.append(
            _PreparedShard(
                shard_id=shard_id,
                index=index,
                payload=payload,
                payload_sha256=payload_sha256,
                spec_sha256=_stable_json_sha256(
                    spec_payload,
                    label=f"shard {shard_id} spec",
                ),
            )
        )
    resolved_empty_reason = (
        str(empty_plan_reason).strip()
        if empty_plan_reason is not None
        else None
    )
    if not prepared and not resolved_empty_reason:
        raise ShardPlanError(
            "An empty shard plan requires a non-empty empty_plan_reason"
        )
    if prepared and resolved_empty_reason is not None:
        raise ShardPlanError(
            "empty_plan_reason is valid only when the shard plan is empty"
        )
    plan_core = {
        "version": SHARD_PLAN_VERSION,
        "mode": ResponseMode.PARTITIONED.value,
        "composability": resolved_composability.value,
        "document_id": resolved_document_id,
        "plan_version": resolved_plan_version,
        "merge_version": resolved_merge_version,
        "generation_contract": generation_contract_value,
        "generation_contract_sha256": generation_contract_sha256,
        "merge_contract": merge_contract_value,
        "merge_contract_sha256": merge_contract_sha256,
        "empty_plan_reason": resolved_empty_reason,
        "shard_schema_sha256": shard_schema_sha256,
        "document_schema_sha256": document_schema_sha256,
        "shard_count": len(prepared),
        "shards": [
            {
                "index": shard.index,
                "shard_id": shard.shard_id,
                "payload_sha256": shard.payload_sha256,
                "spec_sha256": shard.spec_sha256,
            }
            for shard in prepared
        ],
    }
    plan_sha256 = _stable_json_sha256(plan_core, label="shard plan")
    manifest = {**plan_core, "plan_sha256": plan_sha256}
    return ShardPlan(
        document_id=resolved_document_id,
        plan_version=resolved_plan_version,
        merge_version=resolved_merge_version,
        composability=resolved_composability.value,
        generation_contract=generation_contract_value,
        merge_contract=merge_contract_value,
        generation_contract_sha256=generation_contract_sha256,
        merge_contract_sha256=merge_contract_sha256,
        empty_plan_reason=resolved_empty_reason,
        shard_schema=shard_schema_value,
        document_schema=document_schema_value,
        shard_schema_sha256=shard_schema_sha256,
        document_schema_sha256=document_schema_sha256,
        shards=tuple(prepared),
        manifest=manifest,
    )


def _request(plan: ShardPlan, shard: _PreparedShard) -> ShardRequest:
    request_core = {
        "version": SHARD_REQUEST_VERSION,
        "document_id": plan.document_id,
        "plan_sha256": plan.plan_sha256,
        "plan_version": plan.plan_version,
        "merge_version": plan.merge_version,
        "composability": plan.composability,
        "shard_id": shard.shard_id,
        "index": shard.index,
        "shard_count": len(plan.shards),
        "spec_sha256": shard.spec_sha256,
        "payload_sha256": shard.payload_sha256,
        "payload": shard.payload,
        "generation_contract_sha256": plan.generation_contract_sha256,
        "merge_contract_sha256": plan.merge_contract_sha256,
        "shard_schema_sha256": plan.shard_schema_sha256,
        "generation_contract": plan.generation_contract,
        "shard_schema": plan.shard_schema,
    }
    return ShardRequest(
        version=SHARD_REQUEST_VERSION,
        document_id=plan.document_id,
        plan_sha256=plan.plan_sha256,
        plan_version=plan.plan_version,
        merge_version=plan.merge_version,
        composability=plan.composability,
        shard_id=shard.shard_id,
        index=shard.index,
        shard_count=len(plan.shards),
        spec_sha256=shard.spec_sha256,
        payload_sha256=shard.payload_sha256,
        payload=copy.deepcopy(shard.payload),
        generation_contract_sha256=plan.generation_contract_sha256,
        merge_contract_sha256=plan.merge_contract_sha256,
        shard_schema_sha256=plan.shard_schema_sha256,
        generation_contract=copy.deepcopy(plan.generation_contract),
        shard_schema=copy.deepcopy(plan.shard_schema),
        request_sha256=_stable_json_sha256(
            request_core,
            label=f"shard {shard.shard_id} logical request",
        ),
    )


def shard_request(plan: ShardPlan, index: int) -> ShardRequest:
    """Return the exact content-addressed logical request for one shard."""

    if not isinstance(plan, ShardPlan):
        raise ShardPlanError("plan must be a ShardPlan")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ShardPlanError("shard request index must be an integer")
    if index < 0 or index >= len(plan.shards):
        raise ShardPlanError("shard request index is outside the plan")
    return _request(plan, plan.shards[index])


def _parse_exact_json(raw_text: str, *, label: str) -> Any:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ShardSchemaError(f"{label} raw_text must contain one JSON value")

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
        raise ShardSchemaError(
            f"{label} raw_text is not one exact JSON value: {exc}"
        ) from exc
    return _canonical_json(parsed, label=f"{label} raw JSON")


def _provider_audit_dict(
    binding: ProviderAuditBinding,
    *,
    request: ShardRequest,
    raw_text: str,
) -> dict[str, str]:
    if not isinstance(binding, ProviderAuditBinding):
        raise ShardReceiptError(
            "Generated shard must carry a ProviderAuditBinding"
        )
    value = binding.as_dict()
    if value["version"] != PROVIDER_AUDIT_BINDING_VERSION:
        raise ShardReceiptError("Provider audit binding version mismatch")
    for key in ("event_id", "receipt_ref"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ShardReceiptError(
                f"Provider audit binding {key} must not be empty"
            )
    for key in (
        "receipt_sha256",
        "physical_request_sha256",
        "logical_request_sha256",
        "raw_text_sha256",
    ):
        if _SHA256_RE.fullmatch(str(value[key])) is None:
            raise ShardReceiptError(
                f"Provider audit binding {key} is not a SHA-256 digest"
            )
    if value["logical_request_sha256"] != request.request_sha256:
        raise ShardReceiptError(
            "Provider audit binding belongs to another logical request"
        )
    if value["raw_text_sha256"] != text_sha256(raw_text):
        raise ShardReceiptError(
            "Provider audit binding raw response digest mismatch"
        )
    return value


def create_shard_receipt(
    *,
    plan: ShardPlan,
    request: ShardRequest,
    generated: GeneratedShard,
    predecessor_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Validate one output and return its immutable content-addressed receipt."""

    if not isinstance(request, ShardRequest):
        raise ShardReceiptError("Shard request must be a ShardRequest")
    if request.document_id != plan.document_id or request.plan_sha256 != (
        plan.plan_sha256
    ):
        raise ShardReceiptError("Shard request does not belong to the plan")
    if request.index < 0 or request.index >= len(plan.shards):
        raise ShardReceiptError("Shard request index is outside the plan")
    expected = plan.shards[request.index]
    if request.as_dict() != _request(plan, expected).as_dict():
        raise ShardReceiptError("Shard request identity is inconsistent")
    if request.index == 0:
        if predecessor_receipt_sha256 is not None:
            raise ShardReceiptError(
                "The first shard cannot have a predecessor receipt"
            )
    elif _SHA256_RE.fullmatch(
        str(predecessor_receipt_sha256 or "")
    ) is None:
        raise ShardReceiptError(
            "A non-initial shard requires its predecessor receipt digest"
        )
    if not isinstance(generated, GeneratedShard):
        raise ShardReceiptError("Shard generator must return GeneratedShard")
    value = _canonical_json(
        generated.value,
        label=f"shard {request.shard_id} value",
    )
    validator = Draft202012Validator(plan.shard_schema)
    _schema_validate(
        validator,
        value,
        label=f"shard {request.shard_id}",
    )
    raw_text = generated.raw_text
    parsed = _parse_exact_json(
        raw_text,
        label=f"shard {request.shard_id}",
    )
    if _stable_json_sha256(parsed) != _stable_json_sha256(value):
        raise ShardSchemaError(
            f"shard {request.shard_id} raw_text/value mismatch"
        )
    provider_audit = _provider_audit_dict(
        generated.provider_audit,
        request=request,
        raw_text=raw_text,
    )
    if generated.metadata is not None and not isinstance(
        generated.metadata,
        Mapping,
    ):
        raise ShardReceiptError("Generated shard metadata must be an object")
    metadata = _canonical_json(
        dict(generated.metadata or {}),
        label=f"shard {request.shard_id} metadata",
    )
    content_sha256 = _stable_json_sha256(
        value,
        label=f"shard {request.shard_id} value",
    )
    core = {
        "version": SHARD_RECEIPT_VERSION,
        "receipt_kind": "accepted_shard",
        "document_id": plan.document_id,
        "plan_sha256": plan.plan_sha256,
        "plan_version": plan.plan_version,
        "merge_version": plan.merge_version,
        "composability": plan.composability,
        "generation_contract_sha256": plan.generation_contract_sha256,
        "merge_contract_sha256": plan.merge_contract_sha256,
        "shard_schema_sha256": plan.shard_schema_sha256,
        "document_schema_sha256": plan.document_schema_sha256,
        "index": request.index,
        "shard_id": request.shard_id,
        "shard_count": len(plan.shards),
        "spec_sha256": request.spec_sha256,
        "payload_sha256": request.payload_sha256,
        "request_sha256": request.request_sha256,
        "predecessor_receipt_sha256": predecessor_receipt_sha256,
        "content": value,
        "content_sha256": content_sha256,
        "raw_text": raw_text,
        "raw_text_sha256": text_sha256(raw_text),
        "provider_audit": provider_audit,
        "metadata": metadata,
    }
    receipt_sha256 = _stable_json_sha256(core, label="shard receipt")
    return {
        **core,
        "receipt_sha256": receipt_sha256,
        "receipt_id": f"sha256:{receipt_sha256}",
    }


_RECEIPT_CORE_KEYS = {
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
_RECEIPT_KEYS = _RECEIPT_CORE_KEYS | {"receipt_sha256", "receipt_id"}


def verify_shard_receipts(
    plan: ShardPlan,
    receipts: Iterable[Mapping[str, Any]],
    *,
    require_complete: bool,
) -> tuple[dict[str, Any], ...]:
    """Verify an ordered contiguous prefix, or exact complete coverage."""

    rows = list(receipts)
    if len(rows) > len(plan.shards):
        raise ShardReceiptError("Shard receipt count exceeds the plan")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_receipts: set[str] = set()
    validator = Draft202012Validator(plan.shard_schema)
    for expected_index, raw_receipt in enumerate(rows):
        if not isinstance(raw_receipt, Mapping):
            raise ShardReceiptError(
                f"Shard receipt {expected_index} is not an object"
            )
        receipt = _canonical_json(
            dict(raw_receipt),
            label=f"shard receipt {expected_index}",
        )
        if set(receipt) != _RECEIPT_KEYS:
            raise ShardReceiptError(
                f"Shard receipt {expected_index} has an invalid shape"
            )
        if receipt.get("index") != expected_index:
            raise ShardReceiptError(
                "Shard receipts are missing, duplicate, or reordered: "
                f"expected index {expected_index}, got "
                f"{receipt.get('index')}"
            )
        expected = plan.shards[expected_index]
        expected_request = _request(plan, expected)
        expected_predecessor = (
            validated[-1]["receipt_sha256"] if validated else None
        )
        expected_fields = {
            "version": SHARD_RECEIPT_VERSION,
            "receipt_kind": "accepted_shard",
            "document_id": plan.document_id,
            "plan_sha256": plan.plan_sha256,
            "plan_version": plan.plan_version,
            "merge_version": plan.merge_version,
            "composability": plan.composability,
            "generation_contract_sha256": (
                plan.generation_contract_sha256
            ),
            "merge_contract_sha256": plan.merge_contract_sha256,
            "shard_schema_sha256": plan.shard_schema_sha256,
            "document_schema_sha256": plan.document_schema_sha256,
            "index": expected_index,
            "shard_id": expected.shard_id,
            "shard_count": len(plan.shards),
            "spec_sha256": expected.spec_sha256,
            "payload_sha256": expected.payload_sha256,
            "request_sha256": expected_request.request_sha256,
            "predecessor_receipt_sha256": expected_predecessor,
        }
        for key, value in expected_fields.items():
            if receipt.get(key) != value:
                raise ShardReceiptError(
                    f"Shard receipt {expected_index} identity mismatch: {key}"
                )
        shard_id = str(receipt["shard_id"])
        receipt_id = str(receipt.get("receipt_id") or "")
        if shard_id in seen_ids or receipt_id in seen_receipts:
            raise ShardReceiptError("Duplicate shard receipt detected")
        seen_ids.add(shard_id)
        seen_receipts.add(receipt_id)
        content = receipt.get("content")
        if receipt.get("content_sha256") != _stable_json_sha256(
            content,
            label=f"shard {shard_id} content",
        ):
            raise ShardReceiptError(
                f"Shard receipt {expected_index} content digest mismatch"
            )
        _schema_validate(
            validator,
            content,
            label=f"persisted shard {shard_id}",
        )
        raw_text = receipt.get("raw_text")
        raw_digest = receipt.get("raw_text_sha256")
        if not isinstance(raw_text, str) or raw_digest != text_sha256(raw_text):
            raise ShardReceiptError(
                f"Shard receipt {expected_index} raw text digest mismatch"
            )
        parsed = _parse_exact_json(
            raw_text,
            label=f"persisted shard {shard_id}",
        )
        if _stable_json_sha256(parsed) != receipt["content_sha256"]:
            raise ShardReceiptError(
                f"Shard receipt {expected_index} raw/content mismatch"
            )
        raw_audit = receipt.get("provider_audit")
        if not isinstance(raw_audit, dict) or set(raw_audit) != {
            "version",
            "event_id",
            "receipt_ref",
            "receipt_sha256",
            "physical_request_sha256",
            "logical_request_sha256",
            "raw_text_sha256",
        }:
            raise ShardReceiptError(
                f"Shard receipt {expected_index} provider audit is invalid"
            )
        try:
            binding = ProviderAuditBinding(**raw_audit)
        except TypeError as exc:
            raise ShardReceiptError(
                f"Shard receipt {expected_index} provider audit is invalid"
            ) from exc
        _provider_audit_dict(
            binding,
            request=expected_request,
            raw_text=raw_text,
        )
        core = {key: receipt[key] for key in _RECEIPT_CORE_KEYS}
        expected_receipt_sha256 = _stable_json_sha256(
            core,
            label=f"shard receipt {expected_index}",
        )
        if (
            receipt.get("receipt_sha256") != expected_receipt_sha256
            or receipt_id != f"sha256:{expected_receipt_sha256}"
        ):
            raise ShardReceiptError(
                f"Shard receipt {expected_index} digest mismatch"
            )
        validated.append(receipt)
    if require_complete and len(validated) != len(plan.shards):
        raise ShardReceiptError(
            "Shard coverage is incomplete: "
            f"expected {len(plan.shards)}, got {len(validated)}"
        )
    return tuple(validated)


def merge_shard_receipts(
    *,
    plan: ShardPlan,
    receipts: Iterable[Mapping[str, Any]],
    merge_shards: Callable[[tuple[Any, ...]], Any],
) -> tuple[Any, str]:
    """Merge exact complete coverage twice and verify deterministic output."""

    verified = verify_shard_receipts(
        plan,
        receipts,
        require_complete=True,
    )
    values = tuple(copy.deepcopy(row["content"]) for row in verified)
    try:
        first = merge_shards(copy.deepcopy(values))
        second = merge_shards(copy.deepcopy(values))
    except Exception as exc:
        raise ShardMergeError(f"Code-owned shard merge failed: {exc}") from exc
    first_value = _canonical_json(first, label="merged document")
    second_value = _canonical_json(second, label="repeated merged document")
    first_sha256 = _stable_json_sha256(
        first_value,
        label="merged document",
    )
    if first_sha256 != _stable_json_sha256(
        second_value,
        label="repeated merged document",
    ):
        raise ShardMergeError("Code-owned shard merge is non-deterministic")
    validator = Draft202012Validator(plan.document_schema)
    _schema_validate(validator, first_value, label="merged document")
    return first_value, first_sha256


async def _await_callback(
    value: Any,
    *,
    deadline_at: float | None,
    label: str,
) -> Any:
    if not inspect.isawaitable(value):
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise ShardedDocumentLivenessError(
                f"Sharded document deadline expired during {label}"
            )
        return value
    if deadline_at is None:
        return await value
    remaining = deadline_at - time.monotonic()
    if remaining <= 0:
        raise ShardedDocumentLivenessError(
            f"Sharded document deadline expired before {label}"
        )
    timeout_context = asyncio.timeout(remaining)
    try:
        async with timeout_context:
            return await value
    except TimeoutError as exc:
        if not timeout_context.expired():
            raise
        raise ShardedDocumentLivenessError(
            f"Sharded document deadline expired during {label}"
        ) from exc


def _require_async_callback(name: str, callback: Any) -> None:
    call_method = getattr(callback, "__call__", None)
    if not (
        inspect.iscoroutinefunction(callback)
        or inspect.iscoroutinefunction(call_method)
    ):
        raise ShardPlanError(
            f"{name} must be async so liveness deadlines are enforceable"
        )


def _binding_from_receipt(receipt: Mapping[str, Any]) -> ProviderAuditBinding:
    try:
        return ProviderAuditBinding(**dict(receipt["provider_audit"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ShardReceiptError(
            "Persisted shard provider audit binding is invalid"
        ) from exc


def _validate_save_ack(
    ack: Any,
    *,
    plan: ShardPlan,
    receipt: Mapping[str, Any],
    predecessor_receipt_sha256: str | None,
) -> None:
    if not isinstance(ack, ShardSaveAck):
        raise ShardReceiptError(
            "save_receipt must return a ShardSaveAck after durable CAS"
        )
    expected = ShardSaveAck(
        document_id=plan.document_id,
        plan_sha256=plan.plan_sha256,
        index=int(receipt["index"]),
        receipt_id=str(receipt["receipt_id"]),
        receipt_sha256=str(receipt["receipt_sha256"]),
        predecessor_receipt_sha256=predecessor_receipt_sha256,
    )
    if ack != expected:
        raise ShardReceiptError(
            "save_receipt returned a mismatched CAS acknowledgement"
        )


async def compose_sharded_document(
    *,
    document_id: str,
    shards: Iterable[ShardSpec],
    shard_schema: Mapping[str, Any],
    document_schema: Mapping[str, Any],
    plan_version: str,
    merge_version: str,
    generation_contract: Mapping[str, Any],
    merge_contract: Mapping[str, Any],
    response_mode: ResponseMode | str,
    composability: ShardComposability | str,
    generate_shard: Callable[[ShardRequest], Any],
    verify_provider_audit: Callable[[ProviderAuditBinding, ShardRequest], Any],
    merge_shards: Callable[[tuple[Any, ...]], Any],
    load_receipts: Callable[[str, str], Any],
    save_receipt: Callable[[dict[str, Any], str | None], Any],
    empty_plan_reason: str | None = None,
    deadline_seconds: float | None = None,
) -> ShardedDocumentResult:
    """Resume, generate, persist, verify, and deterministically merge shards.

    The loaded receipts must be an ordered contiguous prefix.  A missing tail
    is normal resumable state; a gap before a later receipt is corruption and
    fails closed.  Each newly generated receipt is durably saved before the
    next provider call begins.
    """

    if deadline_seconds is not None and (
        isinstance(deadline_seconds, bool)
        or not isinstance(deadline_seconds, (int, float))
        or not math.isfinite(float(deadline_seconds))
        or float(deadline_seconds) <= 0
    ):
        raise ShardPlanError("deadline_seconds must be positive or null")
    for callback_name, callback in (
        ("generate_shard", generate_shard),
        ("verify_provider_audit", verify_provider_audit),
        ("load_receipts", load_receipts),
        ("save_receipt", save_receipt),
    ):
        _require_async_callback(callback_name, callback)
    if not callable(merge_shards) or inspect.iscoroutinefunction(
        merge_shards
    ):
        raise ShardPlanError(
            "merge_shards must be a synchronous, code-owned pure function"
        )
    plan = build_shard_plan(
        document_id=document_id,
        shards=shards,
        shard_schema=shard_schema,
        document_schema=document_schema,
        plan_version=plan_version,
        merge_version=merge_version,
        generation_contract=generation_contract,
        merge_contract=merge_contract,
        response_mode=response_mode,
        composability=composability,
        empty_plan_reason=empty_plan_reason,
    )
    deadline_at = (
        time.monotonic() + float(deadline_seconds)
        if deadline_seconds is not None
        else None
    )
    loaded = await _await_callback(
        load_receipts(plan.document_id, plan.plan_sha256),
        deadline_at=deadline_at,
        label="receipt load",
    )
    if loaded is None:
        raise ShardReceiptError("load_receipts returned null")
    try:
        loaded_rows = list(loaded)
    except TypeError as exc:
        raise ShardReceiptError(
            "load_receipts must return an iterable"
        ) from exc
    accepted = list(
        verify_shard_receipts(
            plan,
            loaded_rows,
            require_complete=False,
        )
    )
    for receipt in accepted:
        shard = plan.shards[int(receipt["index"])]
        audit_result = await _await_callback(
            verify_provider_audit(
                _binding_from_receipt(receipt),
                _request(plan, shard),
            ),
            deadline_at=deadline_at,
            label=f"shard {shard.shard_id} provider audit verification",
        )
        if audit_result is not None:
            raise ShardReceiptError(
                "verify_provider_audit must return null or raise"
            )
    resumed_shards = len(accepted)
    shard_validator = Draft202012Validator(plan.shard_schema)
    for shard in plan.shards[resumed_shards:]:
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise ShardedDocumentLivenessError(
                "Sharded document deadline expired before shard generation"
            )
        request = _request(plan, shard)
        generated = await _await_callback(
            generate_shard(request),
            deadline_at=deadline_at,
            label=f"shard {shard.shard_id} generation",
        )
        if not isinstance(generated, GeneratedShard):
            raise ShardReceiptError(
                "generate_shard must return GeneratedShard"
            )
        # Validate before persistence; create_shard_receipt repeats this check
        # at the immutable receipt boundary for defense in depth.
        generated_value = _canonical_json(
            generated.value,
            label=f"shard {shard.shard_id} value",
        )
        _schema_validate(
            shard_validator,
            generated_value,
            label=f"shard {shard.shard_id}",
        )
        receipt = create_shard_receipt(
            plan=plan,
            request=request,
            generated=generated,
            predecessor_receipt_sha256=(
                accepted[-1]["receipt_sha256"] if accepted else None
            ),
        )
        audit_result = await _await_callback(
            verify_provider_audit(generated.provider_audit, request),
            deadline_at=deadline_at,
            label=f"shard {shard.shard_id} provider audit verification",
        )
        if audit_result is not None:
            raise ShardReceiptError(
                "verify_provider_audit must return null or raise"
            )
        predecessor_receipt_sha256 = receipt[
            "predecessor_receipt_sha256"
        ]
        save_ack = await _await_callback(
            save_receipt(
                copy.deepcopy(receipt),
                predecessor_receipt_sha256,
            ),
            deadline_at=deadline_at,
            label=f"shard {shard.shard_id} receipt save",
        )
        _validate_save_ack(
            save_ack,
            plan=plan,
            receipt=receipt,
            predecessor_receipt_sha256=predecessor_receipt_sha256,
        )
        accepted.append(receipt)

    verified = verify_shard_receipts(
        plan,
        accepted,
        require_complete=True,
    )
    document, document_sha256 = await _await_callback(
        asyncio.to_thread(
            merge_shard_receipts,
            plan=plan,
            receipts=verified,
            merge_shards=merge_shards,
        ),
        deadline_at=deadline_at,
        label="code-owned shard merge",
    )
    canonical_document = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    manifest_core = {
        "version": SHARDED_DOCUMENT_VERSION,
        "mode": ResponseMode.PARTITIONED.value,
        "composability": plan.composability,
        "document_id": plan.document_id,
        "plan_sha256": plan.plan_sha256,
        "plan_version": plan.plan_version,
        "merge_version": plan.merge_version,
        "generation_contract_sha256": plan.generation_contract_sha256,
        "merge_contract_sha256": plan.merge_contract_sha256,
        "empty_plan_reason": plan.empty_plan_reason,
        "shard_schema_sha256": plan.shard_schema_sha256,
        "document_schema_sha256": plan.document_schema_sha256,
        "complete": True,
        "shard_count": len(plan.shards),
        "covered_shard_count": len(verified),
        "coverage": [
            {
                "index": row["index"],
                "shard_id": row["shard_id"],
                "spec_sha256": row["spec_sha256"],
                "content_sha256": row["content_sha256"],
                "receipt_sha256": row["receipt_sha256"],
            }
            for row in verified
        ],
        "document_sha256": document_sha256,
        "document_json_chars": len(canonical_document),
        "document_json_utf8_bytes": len(canonical_document.encode("utf-8")),
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": _stable_json_sha256(
            manifest_core,
            label="sharded document manifest",
        ),
    }
    return ShardedDocumentResult(
        document=copy.deepcopy(document),
        document_sha256=document_sha256,
        manifest=manifest,
        receipts=tuple(copy.deepcopy(row) for row in verified),
        resumed_shards=resumed_shards,
        generated_shards=len(verified) - resumed_shards,
    )
