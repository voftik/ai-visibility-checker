"""Durable, bounded execution state for strong-model recovery decisions."""

from __future__ import annotations

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
    plan_recovery,
    validate_recovery_decision,
)
from app.services.run_lease import (
    assert_run_lease,
    lease_owner_for,
)


class RecoveryBudgetExceeded(RuntimeError):
    pass


MAX_RECOVERY_EXECUTION_ATTEMPTS = 2
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
                "permitted_artifact_keys": sorted(
                    permitted_artifact_keys or set()
                ),
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

    if (
        stage_planner_call_limit is not None
        and (
            not isinstance(stage_planner_call_limit, int)
            or isinstance(stage_planner_call_limit, bool)
            or stage_planner_call_limit < 1
        )
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
        ).scalars().first()
        if reusable is not None and isinstance(reusable.plan_json, dict):
            stored_decision = dict(reusable.plan_json)
            stored_plan_digest = str(reusable.plan_digest or "")
            if (
                not stored_plan_digest
                or stored_plan_digest != stable_digest(stored_decision)
            ):
                raise OrchestratorContractError(
                    "Stored recovery plan digest mismatch"
                )
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

        # ``planning`` means that the expensive request was reserved and may
        # have reached the provider.  If a worker disappeared before it could
        # persist a result, conservatively account for that call and close the
        # interrupted epoch before considering another one.  A merely
        # ``diagnosing`` epoch has not started a model call and costs no budget.
        interrupted = await session.execute(
            update(RecoveryEpoch)
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
            .values(
                status="failed",
                error_message=(
                    "Planner attempt was interrupted before a durable result"
                ),
                usage_json={
                    "planner_attempt": {
                        "started": True,
                        "completed": False,
                        "reconciled_after_restart": True,
                    }
                },
            )
        )
        if interrupted.rowcount:
            await session.commit()

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
                            RecoveryEpoch.status.in_(
                                _PLANNER_CALL_STARTED_STATUSES
                            ),
                        )
                    )
                ).scalar_one()
            )
            if stage_planner_calls >= stage_planner_call_limit:
                raise RecoveryBudgetExceeded(
                    "Recovery stage planner call budget exhausted "
                    f"({stage_planner_calls}/{stage_planner_call_limit})"
                )
        next_epoch = int(
            (
                await session.execute(
                    select(func.coalesce(func.max(RecoveryEpoch.epoch), 0)).where(
                        RecoveryEpoch.run_id == run_id
                    )
                )
            ).scalar_one()
        ) + 1
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
        incident = {
            "run_id": run_id,
            "stage": stage_key,
            "failure_class": failure_class,
            "code": failure_code,
            "fingerprint": fingerprint,
            "attempt_count": int(run.attempt_count or 0),
            "resume_count": int(run.resume_count or 0),
            "facts_digest": facts_digest,
            "diagnostics": diagnostics,
            "facts": facts,
        }
        planner_input = {
            "incident": incident,
            "allowed_actions": sorted(action_scope),
            "permitted_answer_ids": sorted(answer_scope),
            "permitted_artifact_keys": sorted(artifact_scope),
            "planner_attempt": {
                "started": False,
                "completed": False,
            },
        }
        prepared = (
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
        ).scalars().first()
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
    # Re-validate first, then repeat the ownership predicate inside the write.
    await assert_run_lease(run_id)
    plan_digest = stable_digest(result.decision)
    planner_usage = dict(result.usage)
    planner_usage["_aiv_orchestrator"] = {
        "input_digest": result.input_digest,
        "raw_text": result.raw_text,
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
                        **started_input["planner_attempt"],
                        "completed": True,
                        "succeeded": True,
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

    if (
        stage_execution_limit is not None
        and (
            not isinstance(stage_execution_limit, int)
            or isinstance(stage_execution_limit, bool)
            or stage_execution_limit < 1
        )
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
            if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool)
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
    status = "succeeded" if succeeded and not no_progress else (
        "no_progress" if no_progress else "failed"
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
