"""Durable, bounded execution state for strong-model recovery decisions."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import exists, func, select, update

from app.config import settings
from app.db import SessionLocal
from app.models import RecoveryEpoch, Run, RunStatus
from app.services.recovery_orchestrator import (
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_VERSION,
    OrchestratorContractError,
    RecoveryProviderCheckpointMissing,
    plan_recovery,
    resolve_recovery_model_envelopes,
    validate_recovery_decision,
)
from app.services.run_lease import (
    assert_run_lease,
    lease_owner_for,
)


class RecoveryBudgetExceeded(RuntimeError):
    pass


MAX_RECOVERY_EXECUTION_ATTEMPTS = 2
RECOVERY_PLANNER_REPLAY_CONTRACT_VERSION = "aiv-recovery-planner-replay-v1"
RECOVERY_PLANNER_REPLAY_AUDIT_KEY = "_aiv_recovery_planner_replay"
_ACTIVE_RUN_STATUSES = (
    RunStatus.pending,
    RunStatus.crawling,
    RunStatus.analyzing,
)
_PLANNER_CALL_STARTED_STATUSES = (
    "planning",
    "planned",
    "executing",
    "succeeded",
    "failed",
    "no_progress",
    "blocked",
)


@dataclass(frozen=True)
class DurableRecoveryPlan:
    run_id: str
    epoch_id: int
    epoch: int
    stage_key: str
    decision: dict[str, Any]
    reused: bool
    facts_digest: str
    failure_fingerprint: str
    plan_digest: str


@dataclass(frozen=True)
class DurableRecoveryExecutionState:
    status: str
    execution_attempts: int


def _owned_active_run_exists(run_id: str, owner: str | None):
    """Return a SQL lease guard, or ``None`` for legacy unbound callers."""

    if owner is None:
        return None
    return exists(
        select(Run.id).where(
            Run.id == run_id,
            Run.execution_slot == 1,
            Run.lease_owner == owner,
            Run.status.in_(_ACTIVE_RUN_STATUSES),
        )
    )


def _with_lease_guard(
    conditions: list[Any],
    *,
    run_id: str,
    owner: str | None,
) -> list[Any]:
    guard = _owned_active_run_exists(run_id, owner)
    if guard is not None:
        conditions.append(guard)
    return conditions


async def _raise_for_lost_lease_or_state(
    run_id: str,
    owner: str | None,
    message: str,
) -> None:
    if owner is not None:
        await assert_run_lease(run_id)
    raise RuntimeError(message)


def stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _planner_replay_audit(started_input: dict[str, Any]) -> dict[str, Any]:
    return {
        RECOVERY_PLANNER_REPLAY_AUDIT_KEY: {
            "version": RECOVERY_PLANNER_REPLAY_CONTRACT_VERSION,
            "input_sha256": stable_digest(started_input),
        }
    }


def _is_exact_legacy_planning_input(
    value: dict[str, Any],
    *,
    usage_json: object,
) -> bool:
    """Recognize only the pre-replay-contract planning row shape.

    This compatibility path is deliberately narrower than a generic
    ``model_envelopes is missing`` check. A current-version row with either
    replay marker present can never be downgraded to legacy after corruption.
    """

    if isinstance(usage_json, dict) and usage_json:
        return False
    if set(value) != {
        "incident",
        "allowed_actions",
        "permitted_answer_ids",
        "permitted_artifact_keys",
        "planner_attempt",
    }:
        return False
    incident = value.get("incident")
    attempt = value.get("planner_attempt")
    return (
        isinstance(incident, dict)
        and set(incident)
        == {
            "run_id",
            "stage",
            "failure_class",
            "code",
            "fingerprint",
            "attempt_count",
            "resume_count",
            "facts_digest",
            "diagnostics",
            "facts",
        }
        and isinstance(value.get("allowed_actions"), list)
        and isinstance(value.get("permitted_answer_ids"), list)
        and isinstance(value.get("permitted_artifact_keys"), list)
        and isinstance(attempt, dict)
        and set(attempt).issubset({"started", "completed", "lease_owner"})
        and attempt.get("started") is True
        and attempt.get("completed") is False
    )


def recovery_scope_digest(
    *,
    facts: dict[str, Any],
    allowed_actions: set[str],
    permitted_answer_ids: set[int] | None = None,
    permitted_artifact_keys: set[str] | None = None,
) -> str:
    """Bind durable reuse to both incident facts and executable scope."""

    return stable_digest(
        {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "facts": facts,
            "scope": {
                "allowed_actions": sorted(allowed_actions),
                "permitted_answer_ids": sorted(permitted_answer_ids or set()),
                "permitted_artifact_keys": sorted(permitted_artifact_keys or set()),
            },
        }
    )


def _assert_plan_integrity(
    plan: DurableRecoveryPlan,
    *,
    stored_decision: object,
    stored_plan_digest: object,
) -> None:
    if not isinstance(stored_decision, dict):
        raise OrchestratorContractError("Stored recovery plan is not an object")
    digest = str(stored_plan_digest or "")
    if (
        not digest
        or digest != plan.plan_digest
        or digest != stable_digest(stored_decision)
        or digest != stable_digest(plan.decision)
        or stored_decision != plan.decision
    ):
        raise OrchestratorContractError("Stored recovery plan digest mismatch")


def recovery_failure_fingerprint(
    *,
    stage_key: str,
    failure_class: str,
    failure_code: str,
    diagnostics: dict[str, Any],
) -> str:
    return stable_digest(
        {
            "stage_key": stage_key,
            "failure_class": failure_class,
            "failure_code": failure_code,
            "diagnostics": diagnostics,
        }
    )


async def _prior_decisions(run_id: str) -> list[dict[str, Any]]:
    async with SessionLocal() as session:
        rows = list(
            (
                await session.execute(
                    select(RecoveryEpoch)
                    .where(RecoveryEpoch.run_id == run_id)
                    .order_by(RecoveryEpoch.epoch)
                )
            )
            .scalars()
            .all()
        )
    return [
        {
            "epoch": row.epoch,
            "incident_fingerprint": row.failure_fingerprint,
            "facts_digest": row.facts_digest,
            "status": row.status,
            "action": (
                str((row.plan_json or {}).get("action") or "")
                if isinstance(row.plan_json, dict)
                else ""
            ),
            "outcome": row.outcome_json or {},
        }
        for row in rows
    ]


async def _complete_planning_epoch(
    *,
    run_id: str,
    owner: str | None,
    epoch_id: int,
    started_input: dict[str, Any],
    result: Any,
    resumed_after_restart: bool,
) -> str:
    await assert_run_lease(run_id)
    plan_digest = stable_digest(result.decision)
    planner_usage = dict(result.usage)
    planner_usage["_aiv_orchestrator"] = {
        "input_digest": result.input_digest,
        "raw_text": result.raw_text,
        "resumed_after_restart": resumed_after_restart,
    }
    async with SessionLocal() as session:
        changed = await session.execute(
            update(RecoveryEpoch)
            .where(
                *_with_lease_guard(
                    [
                        RecoveryEpoch.id == epoch_id,
                        RecoveryEpoch.run_id == run_id,
                        RecoveryEpoch.status == "planning",
                    ],
                    run_id=run_id,
                    owner=owner,
                )
            )
            .values(
                status="planned",
                plan_json=result.decision,
                plan_digest=plan_digest,
                usage_json=planner_usage,
                input_json={
                    **started_input,
                    "planner_attempt": {
                        **dict(started_input.get("planner_attempt") or {}),
                        "completed": True,
                        "succeeded": True,
                        "resumed_after_restart": resumed_after_restart,
                    },
                },
                error_message=None,
            )
        )
        await session.commit()
        if changed.rowcount != 1:
            if owner is not None:
                await assert_run_lease(run_id)
            raise OrchestratorContractError(
                "Recovery epoch changed while the planner was running"
            )
    return plan_digest


async def _fail_interrupted_planning_epoch(
    *,
    run_id: str,
    owner: str | None,
    epoch_id: int,
    started_input: dict[str, Any],
    error_message: str,
    checkpoint_missing: bool,
) -> None:
    await assert_run_lease(run_id)
    async with SessionLocal() as session:
        changed = await session.execute(
            update(RecoveryEpoch)
            .where(
                *_with_lease_guard(
                    [
                        RecoveryEpoch.id == epoch_id,
                        RecoveryEpoch.run_id == run_id,
                        RecoveryEpoch.status == "planning",
                        RecoveryEpoch.plan_json.is_(None),
                    ],
                    run_id=run_id,
                    owner=owner,
                )
            )
            .values(
                status="failed",
                input_json={
                    **started_input,
                    "planner_attempt": {
                        **dict(started_input.get("planner_attempt") or {}),
                        "completed": True,
                        "succeeded": False,
                        "resume_only_attempted": True,
                    },
                },
                error_message=error_message[:2000],
                usage_json={
                    "planner_attempt": {
                        "started": True,
                        "completed": False,
                        "reconciled_after_restart": True,
                        "exact_checkpoint_missing": checkpoint_missing,
                    }
                },
            )
        )
        await session.commit()
        if changed.rowcount != 1 and owner is not None:
            await assert_run_lease(run_id)


async def plan_durable_recovery(
    run_id: str,
    *,
    stage_key: str,
    failure_class: str,
    failure_code: str,
    diagnostics: dict[str, Any],
    facts: dict[str, Any],
    allowed_actions: set[str],
    permitted_answer_ids: set[int] | None = None,
    permitted_artifact_keys: set[str] | None = None,
    stage_planner_call_limit: int | None = None,
) -> DurableRecoveryPlan:
    """Reserve one epoch, call the planner once, and persist its safe plan."""

    if stage_planner_call_limit is not None and (
        not isinstance(stage_planner_call_limit, int)
        or isinstance(stage_planner_call_limit, bool)
        or stage_planner_call_limit < 1
    ):
        raise ValueError("stage_planner_call_limit must be a positive integer")

    # Disabled means no planner attempt exists at all.  Check before lease/DB
    # reservation so toggling the optional layer off cannot consume the
    # durable paid-call budget and later prevent a legitimate opt-in call.
    if not settings.PIPELINE_ORCHESTRATOR_ENABLED:
        raise OrchestratorContractError("Pipeline orchestrator is disabled")
    owner = lease_owner_for(run_id)
    await assert_run_lease(run_id)
    action_scope = set(allowed_actions)
    answer_scope = set(permitted_answer_ids or set())
    artifact_scope = set(permitted_artifact_keys or set())
    facts_digest = recovery_scope_digest(
        facts=facts,
        allowed_actions=action_scope,
        permitted_answer_ids=answer_scope,
        permitted_artifact_keys=artifact_scope,
    )
    fingerprint = recovery_failure_fingerprint(
        stage_key=stage_key,
        failure_class=failure_class,
        failure_code=failure_code,
        diagnostics=diagnostics,
    )
    # A planned action survives a worker restart. Reuse it only while both
    # the failure and all code-owned facts are byte-for-byte identical.
    async with SessionLocal() as session:
        reusable = (
            (
                await session.execute(
                    select(RecoveryEpoch)
                    .where(
                        *_with_lease_guard(
                            [
                                RecoveryEpoch.run_id == run_id,
                                RecoveryEpoch.failure_fingerprint == fingerprint,
                                RecoveryEpoch.facts_digest == facts_digest,
                                RecoveryEpoch.status.in_(("planned", "executing")),
                                RecoveryEpoch.plan_json.is_not(None),
                            ],
                            run_id=run_id,
                            owner=owner,
                        )
                    )
                    .order_by(RecoveryEpoch.epoch.desc())
                )
            )
            .scalars()
            .first()
        )
        if reusable is not None and isinstance(reusable.plan_json, dict):
            stored_decision = dict(reusable.plan_json)
            stored_plan_digest = str(reusable.plan_digest or "")
            if not stored_plan_digest or stored_plan_digest != stable_digest(
                stored_decision
            ):
                raise OrchestratorContractError("Stored recovery plan digest mismatch")
            validate_recovery_decision(
                stored_decision,
                allowed_actions=action_scope,
                permitted_answer_ids=answer_scope,
                permitted_artifact_keys=artifact_scope,
                prior_decisions=[],
                incident_fingerprint=fingerprint,
                incident_facts_digest=facts_digest,
            )
            await assert_run_lease(run_id)
            return DurableRecoveryPlan(
                run_id=run_id,
                epoch_id=reusable.id,
                epoch=reusable.epoch,
                stage_key=reusable.stage_key,
                decision=stored_decision,
                reused=True,
                facts_digest=facts_digest,
                failure_fingerprint=fingerprint,
                plan_digest=stored_plan_digest,
            )

        interrupted_epoch = (
            (
                await session.execute(
                    select(RecoveryEpoch)
                    .where(
                        *_with_lease_guard(
                            [
                                RecoveryEpoch.run_id == run_id,
                                RecoveryEpoch.failure_fingerprint == fingerprint,
                                RecoveryEpoch.facts_digest == facts_digest,
                                RecoveryEpoch.status == "planning",
                                RecoveryEpoch.plan_json.is_(None),
                            ],
                            run_id=run_id,
                            owner=owner,
                        )
                    )
                    .order_by(RecoveryEpoch.epoch.desc())
                )
            )
            .scalars()
            .first()
        )

    # A planning epoch can already have every paid physical response durably
    # checkpointed. Replay only those exact identities before charging the
    # interrupted logical call as failed. The replay receives frozen model
    # envelopes and ``resume_only=True``; neither metadata GETs nor model POSTs
    # are possible on this path.
    if interrupted_epoch is not None:
        stored_input = (
            copy.deepcopy(interrupted_epoch.input_json)
            if isinstance(interrupted_epoch.input_json, dict)
            else {}
        )
        stored_usage = (
            copy.deepcopy(interrupted_epoch.usage_json)
            if isinstance(interrupted_epoch.usage_json, dict)
            else {}
        )
        stored_incident = stored_input.get("incident")
        model_envelopes = stored_input.get("model_envelopes")
        replay_version = stored_input.get("replay_contract_version")
        replay_audit = stored_usage.get(RECOVERY_PLANNER_REPLAY_AUDIT_KEY)
        current_replay = replay_version == RECOVERY_PLANNER_REPLAY_CONTRACT_VERSION
        exact_legacy = replay_version is None and _is_exact_legacy_planning_input(
            stored_input,
            usage_json=stored_usage,
        )
        current_replay_valid = (
            current_replay
            and isinstance(replay_audit, dict)
            and replay_audit.get("version") == RECOVERY_PLANNER_REPLAY_CONTRACT_VERSION
            and replay_audit.get("input_sha256") == stable_digest(stored_input)
            and isinstance(stored_incident, dict)
            and isinstance(model_envelopes, dict)
        )
        if not exact_legacy and not current_replay_valid:
            error = OrchestratorContractError(
                "Interrupted recovery planner replay contract is missing, "
                "unknown, or corrupted"
            )
            await _fail_interrupted_planning_epoch(
                run_id=run_id,
                owner=owner,
                epoch_id=interrupted_epoch.id,
                started_input=stored_input,
                error_message=str(error),
                checkpoint_missing=False,
            )
            raise error
        if exact_legacy:
            await _fail_interrupted_planning_epoch(
                run_id=run_id,
                owner=owner,
                epoch_id=interrupted_epoch.id,
                started_input=stored_input,
                error_message=(
                    "Planner attempt predates durable cache-only replay metadata"
                ),
                checkpoint_missing=True,
            )
            model_envelopes = None
        stable_incident = {
            key: value
            for key, value in (
                stored_incident.items() if isinstance(stored_incident, dict) else ()
            )
            if key not in {"attempt_count", "resume_count"}
        }
        expected_incident = {
            "run_id": run_id,
            "stage": stage_key,
            "failure_class": failure_class,
            "code": failure_code,
            "fingerprint": fingerprint,
            "facts_digest": facts_digest,
            "diagnostics": diagnostics,
            "facts": facts,
        }
        exact_scope = (
            stable_incident == expected_incident
            and stored_input.get("allowed_actions") == sorted(action_scope)
            and stored_input.get("permitted_answer_ids") == sorted(answer_scope)
            and stored_input.get("permitted_artifact_keys") == sorted(artifact_scope)
        )
        if current_replay_valid and not exact_scope:
            error = OrchestratorContractError(
                "Interrupted recovery planner input does not match current scope"
            )
            await _fail_interrupted_planning_epoch(
                run_id=run_id,
                owner=owner,
                epoch_id=interrupted_epoch.id,
                started_input=stored_input,
                error_message=str(error),
                checkpoint_missing=False,
            )
            raise error
        if current_replay_valid:
            history = await _prior_decisions(run_id)
            try:
                resumed_result = await plan_recovery(
                    incident=dict(stored_incident),
                    allowed_actions=action_scope,
                    permitted_answer_ids=answer_scope,
                    permitted_artifact_keys=artifact_scope,
                    prior_decisions=history,
                    model_envelopes=copy.deepcopy(model_envelopes),
                    resume_only=True,
                )
            except RecoveryProviderCheckpointMissing as exc:
                await _fail_interrupted_planning_epoch(
                    run_id=run_id,
                    owner=owner,
                    epoch_id=interrupted_epoch.id,
                    started_input=stored_input,
                    error_message=str(exc),
                    checkpoint_missing=True,
                )
            except Exception as exc:
                await _fail_interrupted_planning_epoch(
                    run_id=run_id,
                    owner=owner,
                    epoch_id=interrupted_epoch.id,
                    started_input=stored_input,
                    error_message=str(exc),
                    checkpoint_missing=False,
                )
                raise
            else:
                plan_digest = await _complete_planning_epoch(
                    run_id=run_id,
                    owner=owner,
                    epoch_id=interrupted_epoch.id,
                    started_input=stored_input,
                    result=resumed_result,
                    resumed_after_restart=True,
                )
                return DurableRecoveryPlan(
                    run_id=run_id,
                    epoch_id=interrupted_epoch.id,
                    epoch=interrupted_epoch.epoch,
                    stage_key=interrupted_epoch.stage_key,
                    decision=resumed_result.decision,
                    reused=True,
                    facts_digest=facts_digest,
                    failure_fingerprint=fingerprint,
                    plan_digest=plan_digest,
                )

    async with SessionLocal() as session:
        planner_calls = int(
            (
                await session.execute(
                    select(func.count(RecoveryEpoch.id)).where(
                        RecoveryEpoch.run_id == run_id,
                        RecoveryEpoch.model.is_not(None),
                        RecoveryEpoch.status.in_(_PLANNER_CALL_STARTED_STATUSES),
                    )
                )
            ).scalar_one()
        )
        limit = max(0, int(settings.PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN))
        if planner_calls >= limit:
            raise RecoveryBudgetExceeded(
                f"Recovery planner call budget exhausted ({planner_calls}/{limit})"
            )
        if stage_planner_call_limit is not None:
            stage_planner_calls = int(
                (
                    await session.execute(
                        select(func.count(RecoveryEpoch.id)).where(
                            RecoveryEpoch.run_id == run_id,
                            RecoveryEpoch.stage_key == stage_key,
                            RecoveryEpoch.model.is_not(None),
                            RecoveryEpoch.status.in_(_PLANNER_CALL_STARTED_STATUSES),
                        )
                    )
                ).scalar_one()
            )
            if stage_planner_calls >= stage_planner_call_limit:
                raise RecoveryBudgetExceeded(
                    "Recovery stage planner call budget exhausted "
                    f"({stage_planner_calls}/{stage_planner_call_limit})"
                )
        run_conditions: list[Any] = [Run.id == run_id]
        if owner is not None:
            run_conditions.extend(
                (
                    Run.execution_slot == 1,
                    Run.lease_owner == owner,
                    Run.status.in_(_ACTIVE_RUN_STATUSES),
                )
            )
        run = (
            await session.execute(select(Run).where(*run_conditions))
        ).scalar_one_or_none()
        if run is None:
            if owner is not None:
                await assert_run_lease(run_id)
            raise LookupError(f"Run not found: {run_id}")
        attempt_count = int(run.attempt_count or 0)
        resume_count = int(run.resume_count or 0)

    # Never hold an SQLite transaction or pool slot while fetching provider
    # model metadata. The lease and both budgets are checked again before the
    # snapshot is attached to a durable diagnosing epoch.
    model_envelopes = await resolve_recovery_model_envelopes()
    await assert_run_lease(run_id)
    incident = {
        "run_id": run_id,
        "stage": stage_key,
        "failure_class": failure_class,
        "code": failure_code,
        "fingerprint": fingerprint,
        "attempt_count": attempt_count,
        "resume_count": resume_count,
        "facts_digest": facts_digest,
        "diagnostics": diagnostics,
        "facts": facts,
    }
    planner_input = {
        "replay_contract_version": RECOVERY_PLANNER_REPLAY_CONTRACT_VERSION,
        "incident": incident,
        "allowed_actions": sorted(action_scope),
        "permitted_answer_ids": sorted(answer_scope),
        "permitted_artifact_keys": sorted(artifact_scope),
        "model_envelopes": model_envelopes,
        "planner_attempt": {
            "started": False,
            "completed": False,
        },
    }
    async with SessionLocal() as session:
        planner_calls = int(
            (
                await session.execute(
                    select(func.count(RecoveryEpoch.id)).where(
                        RecoveryEpoch.run_id == run_id,
                        RecoveryEpoch.model.is_not(None),
                        RecoveryEpoch.status.in_(_PLANNER_CALL_STARTED_STATUSES),
                    )
                )
            ).scalar_one()
        )
        limit = max(0, int(settings.PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN))
        if planner_calls >= limit:
            raise RecoveryBudgetExceeded(
                f"Recovery planner call budget exhausted ({planner_calls}/{limit})"
            )
        if stage_planner_call_limit is not None:
            stage_planner_calls = int(
                (
                    await session.execute(
                        select(func.count(RecoveryEpoch.id)).where(
                            RecoveryEpoch.run_id == run_id,
                            RecoveryEpoch.stage_key == stage_key,
                            RecoveryEpoch.model.is_not(None),
                            RecoveryEpoch.status.in_(_PLANNER_CALL_STARTED_STATUSES),
                        )
                    )
                ).scalar_one()
            )
            if stage_planner_calls >= stage_planner_call_limit:
                raise RecoveryBudgetExceeded(
                    "Recovery stage planner call budget exhausted "
                    f"({stage_planner_calls}/{stage_planner_call_limit})"
                )
        next_epoch = (
            int(
                (
                    await session.execute(
                        select(func.coalesce(func.max(RecoveryEpoch.epoch), 0)).where(
                            RecoveryEpoch.run_id == run_id
                        )
                    )
                ).scalar_one()
            )
            + 1
        )
        prepared = (
            (
                await session.execute(
                    select(RecoveryEpoch)
                    .where(
                        *_with_lease_guard(
                            [
                                RecoveryEpoch.run_id == run_id,
                                RecoveryEpoch.failure_fingerprint == fingerprint,
                                RecoveryEpoch.facts_digest == facts_digest,
                                RecoveryEpoch.status == "diagnosing",
                                RecoveryEpoch.plan_json.is_(None),
                            ],
                            run_id=run_id,
                            owner=owner,
                        )
                    )
                    .order_by(RecoveryEpoch.epoch.desc())
                )
            )
            .scalars()
            .first()
        )
        if prepared is None:
            row = RecoveryEpoch(
                run_id=run_id,
                epoch=next_epoch,
                stage_key=stage_key,
                failure_class=failure_class,
                failure_code=failure_code,
                failure_fingerprint=fingerprint,
                facts_digest=facts_digest,
                status="diagnosing",
                model=None,
                input_json=planner_input,
                usage_json=_planner_replay_audit(planner_input),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            epoch_id = row.id
        else:
            row = prepared
            next_epoch = row.epoch
            epoch_id = row.id
            row.input_json = planner_input
            row.usage_json = _planner_replay_audit(planner_input)
            row.error_message = None
            await session.commit()

    # Reserve the paid call immediately before handing control to the network
    # client.  A crash before this update leaves a reusable, budget-free
    # diagnosing epoch; a crash after it leaves a chargeable planning epoch.
    await assert_run_lease(run_id)
    started_input = {
        **planner_input,
        "planner_attempt": {
            "started": True,
            "completed": False,
            "lease_owner": owner,
        },
    }
    async with SessionLocal() as session:
        started = await session.execute(
            update(RecoveryEpoch)
            .where(
                *_with_lease_guard(
                    [
                        RecoveryEpoch.id == epoch_id,
                        RecoveryEpoch.run_id == run_id,
                        RecoveryEpoch.failure_fingerprint == fingerprint,
                        RecoveryEpoch.facts_digest == facts_digest,
                        RecoveryEpoch.status == "diagnosing",
                        RecoveryEpoch.plan_json.is_(None),
                    ],
                    run_id=run_id,
                    owner=owner,
                )
            )
            .values(
                status="planning",
                model=ORCHESTRATOR_MODEL,
                input_json=started_input,
                usage_json=_planner_replay_audit(started_input),
                error_message=None,
            )
        )
        await session.commit()
        if started.rowcount != 1:
            await _raise_for_lost_lease_or_state(
                run_id,
                owner,
                "Recovery planner attempt could not be reserved",
            )

    history = await _prior_decisions(run_id)
    try:
        result = await plan_recovery(
            incident=incident,
            allowed_actions=action_scope,
            permitted_answer_ids=answer_scope,
            permitted_artifact_keys=artifact_scope,
            prior_decisions=history,
            model_envelopes=model_envelopes,
        )
    except Exception as exc:
        # A stale worker must not mutate even the diagnostic epoch. If the
        # lease is still valid, persist the planner failure under the same
        # owner guard so a concurrent lease hand-off cannot race the update.
        await assert_run_lease(run_id)
        async with SessionLocal() as session:
            changed = await session.execute(
                update(RecoveryEpoch)
                .where(
                    *_with_lease_guard(
                        [
                            RecoveryEpoch.id == epoch_id,
                            RecoveryEpoch.run_id == run_id,
                            RecoveryEpoch.status == "planning",
                        ],
                        run_id=run_id,
                        owner=owner,
                    )
                )
                .values(
                    status="failed",
                    input_json={
                        **started_input,
                        "planner_attempt": {
                            **started_input["planner_attempt"],
                            "completed": True,
                            "succeeded": False,
                        },
                    },
                    error_message=str(exc)[:2000],
                )
            )
            await session.commit()
            if changed.rowcount != 1 and owner is not None:
                await assert_run_lease(run_id)
        raise

    # Fable can run long enough for the coordinator to replace this worker.
    # The helper repeats the ownership predicate inside the final write.
    plan_digest = await _complete_planning_epoch(
        run_id=run_id,
        owner=owner,
        epoch_id=epoch_id,
        started_input=started_input,
        result=result,
        resumed_after_restart=False,
    )
    return DurableRecoveryPlan(
        run_id=run_id,
        epoch_id=epoch_id,
        epoch=next_epoch,
        stage_key=stage_key,
        decision=result.decision,
        reused=False,
        facts_digest=facts_digest,
        failure_fingerprint=fingerprint,
        plan_digest=plan_digest,
    )


async def plan_code_owned_recovery(
    run_id: str,
    *,
    stage_key: str,
    failure_class: str,
    failure_code: str,
    diagnostics: dict[str, Any],
    facts: dict[str, Any],
    allowed_actions: set[str],
    decision: dict[str, Any],
    fallback_reason: str,
    permitted_answer_ids: set[int] | None = None,
    permitted_artifact_keys: set[str] | None = None,
) -> DurableRecoveryPlan:
    """Persist a caller-owned safe plan when the optional planner cannot act.

    This path never guesses an action. The caller supplies a decision from its
    own narrow executor contract, and the same scope/action/loop validator used
    for a strong-model plan must accept it. It is intentionally model-free, so
    it neither hides nor increases the configured Fable call budget. A durable
    epoch makes restarts reuse the exact fallback instead of creating an
    unbounded local retry loop.
    """

    if not settings.PIPELINE_ORCHESTRATOR_ENABLED:
        raise OrchestratorContractError("Pipeline orchestrator is disabled")
    owner = lease_owner_for(run_id)
    await assert_run_lease(run_id)
    action_scope = set(allowed_actions)
    answer_scope = set(permitted_answer_ids or set())
    artifact_scope = set(permitted_artifact_keys or set())
    facts_digest = recovery_scope_digest(
        facts=facts,
        allowed_actions=action_scope,
        permitted_answer_ids=answer_scope,
        permitted_artifact_keys=artifact_scope,
    )
    fingerprint = recovery_failure_fingerprint(
        stage_key=stage_key,
        failure_class=failure_class,
        failure_code=failure_code,
        diagnostics=diagnostics,
    )
    history = await _prior_decisions(run_id)
    normalized = validate_recovery_decision(
        copy.deepcopy(decision),
        allowed_actions=action_scope,
        permitted_answer_ids=answer_scope,
        permitted_artifact_keys=artifact_scope,
        prior_decisions=history,
        incident_fingerprint=fingerprint,
        incident_facts_digest=facts_digest,
    )
    plan_digest = stable_digest(normalized)

    async with SessionLocal() as session:
        reusable = (
            (
                await session.execute(
                    select(RecoveryEpoch)
                    .where(
                        *_with_lease_guard(
                            [
                                RecoveryEpoch.run_id == run_id,
                                RecoveryEpoch.failure_fingerprint == fingerprint,
                                RecoveryEpoch.facts_digest == facts_digest,
                                RecoveryEpoch.model.is_(None),
                                RecoveryEpoch.status.in_(("planned", "executing")),
                                RecoveryEpoch.plan_json.is_not(None),
                            ],
                            run_id=run_id,
                            owner=owner,
                        )
                    )
                    .order_by(RecoveryEpoch.epoch.desc())
                )
            )
            .scalars()
            .first()
        )
        if reusable is not None:
            stored = (
                dict(reusable.plan_json)
                if isinstance(reusable.plan_json, dict)
                else None
            )
            if stored != normalized or str(reusable.plan_digest or "") != plan_digest:
                raise OrchestratorContractError(
                    "Stored code-owned recovery plan digest mismatch"
                )
            return DurableRecoveryPlan(
                run_id=run_id,
                epoch_id=reusable.id,
                epoch=reusable.epoch,
                stage_key=reusable.stage_key,
                decision=stored,
                reused=True,
                facts_digest=facts_digest,
                failure_fingerprint=fingerprint,
                plan_digest=plan_digest,
            )

        next_epoch = (
            int(
                (
                    await session.execute(
                        select(func.coalesce(func.max(RecoveryEpoch.epoch), 0)).where(
                            RecoveryEpoch.run_id == run_id
                        )
                    )
                ).scalar_one()
            )
            + 1
        )
        run_conditions: list[Any] = [Run.id == run_id]
        if owner is not None:
            run_conditions.extend(
                (
                    Run.execution_slot == 1,
                    Run.lease_owner == owner,
                    Run.status.in_(_ACTIVE_RUN_STATUSES),
                )
            )
        if (
            await session.execute(select(Run.id).where(*run_conditions))
        ).scalar_one_or_none() is None:
            if owner is not None:
                await assert_run_lease(run_id)
            raise LookupError(f"Run not found: {run_id}")
        incident = {
            "run_id": run_id,
            "stage": stage_key,
            "failure_class": failure_class,
            "code": failure_code,
            "fingerprint": fingerprint,
            "facts_digest": facts_digest,
            "diagnostics": diagnostics,
            "facts": facts,
        }
        row = RecoveryEpoch(
            run_id=run_id,
            epoch=next_epoch,
            stage_key=stage_key,
            failure_class=failure_class,
            failure_code=failure_code,
            failure_fingerprint=fingerprint,
            facts_digest=facts_digest,
            status="planned",
            model=None,
            input_json={
                "incident": incident,
                "allowed_actions": sorted(action_scope),
                "permitted_answer_ids": sorted(answer_scope),
                "permitted_artifact_keys": sorted(artifact_scope),
                "planner_attempt": {
                    "started": False,
                    "completed": True,
                    "succeeded": False,
                    "code_owned_fallback": True,
                    "reason": fallback_reason,
                },
            },
            plan_json=normalized,
            plan_digest=plan_digest,
            usage_json={
                "_aiv_code_owned_recovery": {
                    "version": "aiv-code-owned-recovery-plan-v1",
                    "reason": fallback_reason,
                    "plan_digest": plan_digest,
                    "strong_model_call_made": False,
                }
            },
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        epoch_id = row.id

    return DurableRecoveryPlan(
        run_id=run_id,
        epoch_id=epoch_id,
        epoch=next_epoch,
        stage_key=stage_key,
        decision=normalized,
        reused=False,
        facts_digest=facts_digest,
        failure_fingerprint=fingerprint,
        plan_digest=plan_digest,
    )


async def mark_recovery_executing(
    plan: DurableRecoveryPlan,
    *,
    stage_execution_limit: int | None = None,
) -> int:
    """Reserve a durable execution for this exact action.

    The generic per-epoch budget remains two attempts for restart-tolerant
    stages.  High-risk callers may additionally set a run+stage budget.  The
    analysis-critic recovery uses a budget of one: after annotations have been
    touched once, neither a restarted worker nor a second planner epoch may
    buy another repair/final-critic cycle.  The returned one-based attempt
    number lets a bounded caller address a distinct durable artifact for each
    execution without deriving mutable state outside this reservation.
    """

    if stage_execution_limit is not None and (
        not isinstance(stage_execution_limit, int)
        or isinstance(stage_execution_limit, bool)
        or stage_execution_limit < 1
    ):
        raise ValueError("stage_execution_limit must be a positive integer")

    owner = lease_owner_for(plan.run_id)
    await assert_run_lease(plan.run_id)
    base_conditions = _with_lease_guard(
        [
            RecoveryEpoch.id == plan.epoch_id,
            RecoveryEpoch.run_id == plan.run_id,
            RecoveryEpoch.facts_digest == plan.facts_digest,
            RecoveryEpoch.failure_fingerprint == plan.failure_fingerprint,
            RecoveryEpoch.status.in_(("planned", "executing")),
        ],
        run_id=plan.run_id,
        owner=owner,
    )
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(
                    RecoveryEpoch.status,
                    RecoveryEpoch.outcome_json,
                    RecoveryEpoch.plan_json,
                    RecoveryEpoch.plan_digest,
                ).where(*base_conditions)
            )
        ).one_or_none()
        if row is None:
            await _raise_for_lost_lease_or_state(
                plan.run_id,
                owner,
                "Recovery epoch lost its durable state",
            )

        current_status, stored_outcome, stored_decision, stored_plan_digest = row
        _assert_plan_integrity(
            plan,
            stored_decision=stored_decision,
            stored_plan_digest=stored_plan_digest,
        )
        current_outcome = (
            dict(stored_outcome) if isinstance(stored_outcome, dict) else {}
        )
        raw_attempts = current_outcome.get("execution_attempts", 0)
        attempts = (
            raw_attempts
            if isinstance(raw_attempts, int)
            and not isinstance(raw_attempts, bool)
            and raw_attempts >= 0
            else MAX_RECOVERY_EXECUTION_ATTEMPTS
        )
        optimistic_conditions = [
            RecoveryEpoch.id == plan.epoch_id,
            RecoveryEpoch.run_id == plan.run_id,
            RecoveryEpoch.facts_digest == plan.facts_digest,
            RecoveryEpoch.failure_fingerprint == plan.failure_fingerprint,
            RecoveryEpoch.status == current_status,
        ]
        optimistic_conditions.append(
            RecoveryEpoch.outcome_json.is_(None)
            if stored_outcome is None
            else RecoveryEpoch.outcome_json == stored_outcome
        )
        _with_lease_guard(
            optimistic_conditions,
            run_id=plan.run_id,
            owner=owner,
        )

        stage_attempts = 0
        if stage_execution_limit is not None:
            stage_outcomes = list(
                (
                    await session.execute(
                        select(RecoveryEpoch.outcome_json).where(
                            *_with_lease_guard(
                                [
                                    RecoveryEpoch.run_id == plan.run_id,
                                    RecoveryEpoch.stage_key == plan.stage_key,
                                ],
                                run_id=plan.run_id,
                                owner=owner,
                            )
                        )
                    )
                ).scalars()
            )
            for outcome in stage_outcomes:
                raw_stage_attempts = (
                    outcome.get("execution_attempts", 0)
                    if isinstance(outcome, dict)
                    else 0
                )
                if (
                    isinstance(raw_stage_attempts, int)
                    and not isinstance(raw_stage_attempts, bool)
                    and raw_stage_attempts > 0
                ):
                    stage_attempts += raw_stage_attempts

            if stage_attempts >= stage_execution_limit:
                blocked_outcome = {
                    **current_outcome,
                    "execution_attempts": attempts,
                    "max_execution_attempts": MAX_RECOVERY_EXECUTION_ATTEMPTS,
                    "stage_execution_attempts": stage_attempts,
                    "stage_execution_limit": stage_execution_limit,
                    "succeeded": False,
                    "reason": "stage_execution_attempt_budget_exhausted",
                }
                changed = await session.execute(
                    update(RecoveryEpoch)
                    .where(*optimistic_conditions)
                    .values(
                        status="blocked",
                        outcome_json=blocked_outcome,
                        error_message=(
                            "Recovery stage execution budget exhausted "
                            f"({stage_attempts}/{stage_execution_limit})"
                        ),
                    )
                )
                await session.commit()
                if changed.rowcount != 1:
                    await _raise_for_lost_lease_or_state(
                        plan.run_id,
                        owner,
                        "Recovery epoch changed while enforcing stage budget",
                    )
                raise RecoveryBudgetExceeded(
                    "Recovery stage execution budget exhausted "
                    f"({stage_attempts}/{stage_execution_limit})"
                )

        if attempts >= MAX_RECOVERY_EXECUTION_ATTEMPTS:
            blocked_outcome = {
                **current_outcome,
                "execution_attempts": attempts,
                "max_execution_attempts": MAX_RECOVERY_EXECUTION_ATTEMPTS,
                "succeeded": False,
                "reason": "execution_attempt_budget_exhausted",
            }
            changed = await session.execute(
                update(RecoveryEpoch)
                .where(*optimistic_conditions)
                .values(
                    status="blocked",
                    outcome_json=blocked_outcome,
                    error_message=(
                        "Recovery action execution budget exhausted "
                        f"({attempts}/{MAX_RECOVERY_EXECUTION_ATTEMPTS})"
                    ),
                )
            )
            await session.commit()
            if changed.rowcount != 1:
                await _raise_for_lost_lease_or_state(
                    plan.run_id,
                    owner,
                    "Recovery epoch changed while enforcing execution budget",
                )
            raise RecoveryBudgetExceeded(
                "Recovery action execution budget exhausted "
                f"({attempts}/{MAX_RECOVERY_EXECUTION_ATTEMPTS})"
            )

        next_outcome = {
            **current_outcome,
            "execution_attempts": attempts + 1,
            "max_execution_attempts": MAX_RECOVERY_EXECUTION_ATTEMPTS,
            **(
                {
                    "stage_execution_attempts": stage_attempts + 1,
                    "stage_execution_limit": stage_execution_limit,
                }
                if stage_execution_limit is not None
                else {}
            ),
        }
        changed = await session.execute(
            update(RecoveryEpoch)
            .where(*optimistic_conditions)
            .values(
                status="executing",
                outcome_json=next_outcome,
                error_message=None,
            )
        )
        await session.commit()
        if changed.rowcount != 1:
            await _raise_for_lost_lease_or_state(
                plan.run_id,
                owner,
                "Recovery epoch changed while reserving execution attempt",
            )
        return attempts + 1


async def recovery_execution_state(
    plan: DurableRecoveryPlan,
) -> DurableRecoveryExecutionState:
    """Read the exact durable execution reservation for a reusable plan.

    A caller uses this before resuming an owner artifact.  It may continue an
    already reserved attempt under the same artifact key, but it must not
    derive or reserve a later attempt from this read.
    """

    owner = lease_owner_for(plan.run_id)
    await assert_run_lease(plan.run_id)
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(
                    RecoveryEpoch.status,
                    RecoveryEpoch.outcome_json,
                    RecoveryEpoch.plan_json,
                    RecoveryEpoch.plan_digest,
                ).where(
                    *_with_lease_guard(
                        [
                            RecoveryEpoch.id == plan.epoch_id,
                            RecoveryEpoch.run_id == plan.run_id,
                            RecoveryEpoch.facts_digest == plan.facts_digest,
                            RecoveryEpoch.failure_fingerprint
                            == plan.failure_fingerprint,
                            RecoveryEpoch.status.in_(("planned", "executing")),
                        ],
                        run_id=plan.run_id,
                        owner=owner,
                    )
                )
            )
        ).one_or_none()
    if row is None:
        await _raise_for_lost_lease_or_state(
            plan.run_id,
            owner,
            "Recovery epoch has no resumable execution state",
        )
    status, stored_outcome, stored_decision, stored_plan_digest = row
    _assert_plan_integrity(
        plan,
        stored_decision=stored_decision,
        stored_plan_digest=stored_plan_digest,
    )
    outcome = dict(stored_outcome) if isinstance(stored_outcome, dict) else {}
    raw_attempts = outcome.get("execution_attempts", 0)
    if (
        not isinstance(raw_attempts, int)
        or isinstance(raw_attempts, bool)
        or raw_attempts < 0
        or raw_attempts > MAX_RECOVERY_EXECUTION_ATTEMPTS
        or (status == "planned" and raw_attempts != 0)
        or (status == "executing" and raw_attempts < 1)
    ):
        raise OrchestratorContractError(
            "Recovery epoch execution reservation is malformed"
        )
    return DurableRecoveryExecutionState(
        status=str(status),
        execution_attempts=raw_attempts,
    )


async def finish_recovery(
    plan: DurableRecoveryPlan,
    *,
    succeeded: bool,
    before_digest: str,
    after_digest: str,
    details: dict[str, Any] | None = None,
) -> None:
    await assert_run_lease(plan.run_id)
    no_progress = before_digest == after_digest
    status = (
        "succeeded"
        if succeeded and not no_progress
        else ("no_progress" if no_progress else "failed")
    )
    await _mark_recovery_status(
        plan,
        # Completion is valid only after mark_recovery_executing reserved and
        # persisted an execution attempt. This prevents callers from bypassing
        # the cross-restart attempt budget.
        from_statuses=("executing",),
        status=status,
        outcome_json={
            "succeeded": bool(succeeded and not no_progress),
            "before_digest": before_digest,
            "after_digest": after_digest,
            "details": details or {},
        },
    )


async def _mark_recovery_status(
    plan: DurableRecoveryPlan,
    *,
    from_statuses: tuple[str, ...],
    status: str,
    outcome_json: dict[str, Any] | None = None,
) -> None:
    owner = lease_owner_for(plan.run_id)
    await assert_run_lease(plan.run_id)
    async with SessionLocal() as session:
        base_conditions = _with_lease_guard(
            [
                RecoveryEpoch.id == plan.epoch_id,
                RecoveryEpoch.run_id == plan.run_id,
                RecoveryEpoch.facts_digest == plan.facts_digest,
                RecoveryEpoch.failure_fingerprint == plan.failure_fingerprint,
                RecoveryEpoch.status.in_(from_statuses),
            ],
            run_id=plan.run_id,
            owner=owner,
        )
        row = (
            await session.execute(
                select(
                    RecoveryEpoch.status,
                    RecoveryEpoch.outcome_json,
                    RecoveryEpoch.plan_json,
                    RecoveryEpoch.plan_digest,
                ).where(*base_conditions)
            )
        ).one_or_none()
        if row is None:
            await _raise_for_lost_lease_or_state(
                plan.run_id,
                owner,
                "Recovery epoch lost its durable state",
            )
        current_status, stored_outcome, stored_decision, stored_plan_digest = row
        _assert_plan_integrity(
            plan,
            stored_decision=stored_decision,
            stored_plan_digest=stored_plan_digest,
        )
        merged_outcome = (
            dict(stored_outcome) if isinstance(stored_outcome, dict) else {}
        )
        if outcome_json is not None:
            merged_outcome.update(outcome_json)
        optimistic_conditions = [
            RecoveryEpoch.id == plan.epoch_id,
            RecoveryEpoch.run_id == plan.run_id,
            RecoveryEpoch.facts_digest == plan.facts_digest,
            RecoveryEpoch.failure_fingerprint == plan.failure_fingerprint,
            RecoveryEpoch.status == current_status,
        ]
        optimistic_conditions.append(
            RecoveryEpoch.outcome_json.is_(None)
            if stored_outcome is None
            else RecoveryEpoch.outcome_json == stored_outcome
        )
        _with_lease_guard(
            optimistic_conditions,
            run_id=plan.run_id,
            owner=owner,
        )
        changed = await session.execute(
            update(RecoveryEpoch)
            .where(*optimistic_conditions)
            .values(
                status=status,
                outcome_json=(merged_outcome or None),
            )
        )
        await session.commit()
        if changed.rowcount != 1:
            await _raise_for_lost_lease_or_state(
                plan.run_id,
                owner,
                "Recovery epoch lost its durable state",
            )
