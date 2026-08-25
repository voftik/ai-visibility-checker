"""Bounded strong-model planner for exceptional pipeline recovery.

The orchestrator never executes arbitrary instructions.  A deterministic
caller supplies a small allow-list of actions and the exact rows/artifacts
that may be touched.  Claude Fable may choose and explain one action, but code
validates the decision before another layer sees it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.services.openrouter import OpenRouterError, WebSearchPolicy, chat

ORCHESTRATOR_VERSION = "aiv-recovery-orchestrator-v1"
ORCHESTRATOR_MODEL = settings.OPENROUTER_ORCHESTRATOR_MODEL
ORCHESTRATOR_MAX_OUTPUT_TOKENS = 4_000

ACTION_RETRY_WITH_GUIDANCE = "retry_stage_with_guidance"
ACTION_DETERMINISTIC_FALLBACK = "use_deterministic_fallback"
ACTION_TARGETED_ANNOTATION_REPAIR = "targeted_annotation_repair"
ACTION_RECOMPUTE_DERIVED = "recompute_derived_artifacts"
ACTION_PUBLISH_LIMITED = "publish_with_limitation"
ACTION_STOP = "stop_and_preserve_checkpoint"

KNOWN_ACTIONS = frozenset(
    {
        ACTION_RETRY_WITH_GUIDANCE,
        ACTION_DETERMINISTIC_FALLBACK,
        ACTION_TARGETED_ANNOTATION_REPAIR,
        ACTION_RECOMPUTE_DERIVED,
        ACTION_PUBLISH_LIMITED,
        ACTION_STOP,
    }
)

# The planner may only request checks that executable code understands.  Free
# prose here looks reassuring in an audit log but cannot prove that a recovery
# actually satisfied its contract.
CHECK_PROMPT_CONTRACT_VALID = "prompt_contract_valid"
CHECK_SEMANTIC_REVIEW_PASSED = "semantic_review_passed"
CHECK_RAW_CORPUS_UNCHANGED = "raw_corpus_unchanged"
CHECK_DERIVED_METRICS_RECOMPUTED = "derived_metrics_recomputed"
CHECK_CRITIC_GATE_PASSED = "critic_gate_passed"
CHECK_CHECKPOINT_PRESERVED = "checkpoint_preserved"

KNOWN_ACCEPTANCE_CHECKS = frozenset(
    {
        CHECK_PROMPT_CONTRACT_VALID,
        CHECK_SEMANTIC_REVIEW_PASSED,
        CHECK_RAW_CORPUS_UNCHANGED,
        CHECK_DERIVED_METRICS_RECOMPUTED,
        CHECK_CRITIC_GATE_PASSED,
        CHECK_CHECKPOINT_PRESERVED,
    }
)


class OrchestratorContractError(OpenRouterError):
    """The planner returned a decision outside the deterministic contract."""


@dataclass(frozen=True)
class OrchestratorResult:
    decision: dict[str, Any]
    raw_text: str
    usage: dict[str, Any]
    input_digest: str


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Keep planner context compact even when an exception carries raw data."""

    if depth >= 6:
        return "[depth limit]"
    if isinstance(value, str):
        if len(value) <= 6_000:
            return value
        return value[:6_000] + "\n[truncated]"
    if isinstance(value, dict):
        items = list(value.items())[:80]
        output = {
            str(key)[:160]: _bounded_value(item, depth=depth + 1)
            for key, item in items
        }
        if len(value) > len(items):
            output["_truncated_keys"] = len(value) - len(items)
        return output
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:40]
        output = [_bounded_value(item, depth=depth + 1) for item in items]
        if len(value) > len(items):
            output.append({"_truncated_items": len(value) - len(items)})
        return output
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1_000]


def _bounded_payload(value: dict[str, Any]) -> dict[str, Any]:
    bounded = _bounded_value(value)
    if not isinstance(bounded, dict):  # pragma: no cover - caller contract
        raise OrchestratorContractError("Orchestrator payload must be an object")
    serialized = json.dumps(bounded, ensure_ascii=False, sort_keys=True)
    limit = max(10_000, int(settings.PIPELINE_ORCHESTRATOR_MAX_INPUT_CHARS))
    if len(serialized) <= limit:
        return bounded
    # The incident summary and deterministic diagnostics remain available;
    # large optional context is replaced by a digest instead of being sliced
    # into invalid JSON or silently consuming an unbounded Fable request.
    compact = {
        key: bounded.get(key)
        for key in (
            "version",
            "incident",
            "allowed_actions",
            "permitted_answer_ids",
            "permitted_artifact_keys",
            "prior_decisions",
        )
        if key in bounded
    }
    compact["omitted_context"] = {
        "reason": "input_char_budget",
        "original_chars": len(serialized),
        "sha256": _stable_digest(bounded),
    }
    if len(json.dumps(compact, ensure_ascii=False, sort_keys=True)) > limit:
        raise OrchestratorContractError(
            "Recovery incident is too large even after deterministic compaction"
        )
    return compact


def _decision_schema(allowed_actions: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": allowed_actions},
            "rationale": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium"],
            },
            "guidance": {"type": "string"},
            "target_answer_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 40,
            },
            "invalidate_artifact_keys": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 20,
            },
            "acceptance_checks": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(KNOWN_ACCEPTANCE_CHECKS),
                },
                "minItems": 1,
                "maxItems": 8,
            },
        },
        "required": [
            "action",
            "rationale",
            "confidence",
            "guidance",
            "target_answer_ids",
            "invalidate_artifact_keys",
            "acceptance_checks",
        ],
    }


def validate_recovery_decision(
    decision: dict[str, Any],
    *,
    allowed_actions: set[str],
    permitted_answer_ids: set[int],
    permitted_artifact_keys: set[str],
    prior_decisions: list[dict[str, Any]],
    incident_fingerprint: str,
    incident_facts_digest: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    action = str(decision.get("action") or "")
    if action not in allowed_actions:
        errors.append("action is not allowed for this incident")
    rationale = str(decision.get("rationale") or "").strip()
    if len(rationale) < 20:
        errors.append("rationale is too short")
    confidence = str(decision.get("confidence") or "")
    if confidence not in {"high", "medium"}:
        errors.append("confidence must be high or medium for execution")
    guidance = str(decision.get("guidance") or "").strip()
    if len(guidance) > 4_000:
        errors.append("guidance exceeds the deterministic budget")

    raw_answer_ids = decision.get("target_answer_ids")
    if not isinstance(raw_answer_ids, list):
        errors.append("target_answer_ids must be an integer list")
        answer_ids: list[int] = []
    elif len(raw_answer_ids) > 40:
        errors.append("target_answer_ids exceeds the local limit of 40")
        answer_ids = []
    elif any(type(value) is not int or value <= 0 for value in raw_answer_ids):
        errors.append("target_answer_ids must contain positive integers")
        answer_ids = []
    else:
        answer_ids = list(dict.fromkeys(raw_answer_ids))
        if not set(answer_ids).issubset(permitted_answer_ids):
            errors.append("decision targets answer ids outside the incident")

    raw_artifacts = decision.get("invalidate_artifact_keys")
    if not isinstance(raw_artifacts, list):
        errors.append("invalidate_artifact_keys must be a string list")
        artifact_keys: list[str] = []
    elif len(raw_artifacts) > 20:
        errors.append("invalidate_artifact_keys exceeds the local limit of 20")
        artifact_keys = []
    elif any(not isinstance(value, str) for value in raw_artifacts):
        errors.append("invalidate_artifact_keys must be a string list")
        artifact_keys = []
    else:
        artifact_keys = list(dict.fromkeys(raw_artifacts))
        if not set(artifact_keys).issubset(permitted_artifact_keys):
            errors.append("decision invalidates artifacts outside the incident")

    checks = decision.get("acceptance_checks")
    if not isinstance(checks, list) or not checks:
        errors.append("acceptance_checks contains an unknown executable check")
        normalized_checks: list[str] = []
    elif len(checks) > 8:
        errors.append("acceptance_checks exceeds the local limit of 8")
        normalized_checks = []
    elif any(
        not isinstance(item, str) or item not in KNOWN_ACCEPTANCE_CHECKS
        for item in checks
    ):
        errors.append("acceptance_checks contains an unknown executable check")
        normalized_checks = []
    else:
        normalized_checks = list(dict.fromkeys(checks))

    blocking_statuses = {"failed", "no_progress", "blocked"}
    repeated = any(
        isinstance(item, dict)
        and item.get("incident_fingerprint") == incident_fingerprint
        and incident_facts_digest is not None
        and item.get("facts_digest") == incident_facts_digest
        and item.get("action") == action
        and item.get("status") in blocking_statuses
        for item in prior_decisions
    )
    if repeated:
        errors.append("the same action already failed for this fingerprint")
    if action == ACTION_TARGETED_ANNOTATION_REPAIR and not answer_ids:
        errors.append("targeted annotation repair requires answer ids")
    if action == ACTION_RETRY_WITH_GUIDANCE and not guidance:
        errors.append("stage retry requires concrete guidance")

    if errors:
        raise OrchestratorContractError("; ".join(errors))
    return {
        "action": action,
        "rationale": rationale[:4_000],
        "confidence": confidence,
        "guidance": guidance,
        "target_answer_ids": answer_ids,
        "invalidate_artifact_keys": artifact_keys,
        "acceptance_checks": normalized_checks,
        "incident_fingerprint": incident_fingerprint,
        "orchestrator_version": ORCHESTRATOR_VERSION,
    }


async def plan_recovery(
    *,
    incident: dict[str, Any],
    allowed_actions: set[str],
    permitted_answer_ids: set[int] | None = None,
    permitted_artifact_keys: set[str] | None = None,
    prior_decisions: list[dict[str, Any]] | None = None,
) -> OrchestratorResult:
    """Ask Fable for one decision, then enforce the code-owned boundary."""

    if not settings.PIPELINE_ORCHESTRATOR_ENABLED:
        raise OrchestratorContractError("Pipeline orchestrator is disabled")
    if not allowed_actions or not allowed_actions.issubset(KNOWN_ACTIONS):
        raise OrchestratorContractError("Unknown or empty recovery action set")
    answer_ids = set(permitted_answer_ids or set())
    artifact_keys = set(permitted_artifact_keys or set())
    history = list(prior_decisions or [])
    incident_fingerprint = str(
        incident.get("fingerprint") or _stable_digest(incident)
    )
    incident_facts_digest = (
        str(incident.get("facts_digest"))
        if incident.get("facts_digest") is not None
        else None
    )
    ordered_actions = sorted(allowed_actions)
    payload = _bounded_payload(
        {
            "version": ORCHESTRATOR_VERSION,
            "incident": incident,
            "allowed_actions": ordered_actions,
            "permitted_answer_ids": sorted(answer_ids),
            "permitted_artifact_keys": sorted(artifact_keys),
            "prior_decisions": history,
        }
    )
    system = """
Ты старший оркестратор восстановления аналитического пайплайна AIV. Тебя
вызывают только после того, как обычный детерминированный или дешёвый LLM-слой
не смог сойтись. Выбери ровно одно действие из allowed_actions.

Ты не исполняешь действие и не меняешь код, фильтры, исходные ответы или
метрики. Ты составляешь план для проверяемого кода. Нельзя придумывать факты,
answer_id, artifact_key или расширять область ремонта. target_answer_ids и
invalidate_artifact_keys могут содержать только значения из разрешённых
списков. Если доказательств недостаточно, выбери stop_and_preserve_checkpoint.

Предпочитай самый узкий обратимый ремонт. Не предлагай повторный опрос
модельной панели, если сохранён raw-корпус. Не повторяй действие, которое уже
закончилось failed, no_progress или blocked для тех же fingerprint и
facts_digest. Успешное прошлое действие или решение для других фактов не
считай петлёй. acceptance_checks выбирай только из кодов схемы: это исполняемые
проверки, а не свободный текст. Решение с confidence=low выполнять нельзя.
guidance — короткий дополнительный контекст для следующего слоя, а не новая
методология и не разрешение ослабить инварианты.
""".strip()
    result = await chat(
        model=ORCHESTRATOR_MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_schema=_decision_schema(ordered_actions),
        schema_name="aiv_recovery_orchestrator",
        web_policy=WebSearchPolicy.FORBIDDEN,
        reasoning_effort="high",
        max_tokens=ORCHESTRATOR_MAX_OUTPUT_TOKENS,
        temperature=0.1,
        # One durable epoch must correspond to at most one billable complete
        # planner response.  Transport 429/5xx retries remain available, but
        # an invalid/partial schema response is reconciled as a failed epoch
        # instead of silently buying two more Fable completions.
        retry_response_contract_errors=False,
        # The durable call budget counts provider requests, not retry loops.
        # An uncertain transport result may already have consumed the expensive
        # completion, so Fable must never be retried inside one epoch.
        retry_transport_errors=False,
    )
    if not isinstance(result.parsed, dict):
        raise OrchestratorContractError(
            "Recovery orchestrator returned a non-object response"
        )
    decision = validate_recovery_decision(
        result.parsed,
        allowed_actions=allowed_actions,
        permitted_answer_ids=answer_ids,
        permitted_artifact_keys=artifact_keys,
        prior_decisions=history,
        incident_fingerprint=incident_fingerprint,
        incident_facts_digest=incident_facts_digest,
    )
    return OrchestratorResult(
        decision=decision,
        raw_text=result.text,
        usage=result.usage,
        input_digest=_stable_digest(payload),
    )
