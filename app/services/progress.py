from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import update

from app.db import SessionLocal
from app.models import Run, RunStatus
from app.services.event_bus import bus
from app.services.run_lease import RunLeaseLostError, lease_owner_for

STAGES: dict[str, str] = {
    "site_discovery": "Изучаем сайт",
    "technical_access": "Проверяем доступ для ИИ",
    "scenario_design": "Формируем сценарии выбора",
    "web_visibility": "Сравниваем ответы ИИ-систем",
    "knowledge_gap": "Считаем видимость и разрывы знаний",
    "report": "Собираем отчёт и иллюстрации",
    "source_review_required": "Нужна проверка источников",
    "integrity_review_required": "Нужна проверка целостности",
    "panel_review_required": "Нужна проверка ответов",
}

NON_RETRYABLE_FAILURE_STAGES = frozenset(
    {
        "source_review_required",
        "integrity_review_required",
        "panel_review_required",
    }
)


async def update_progress(
    run_id: str,
    *,
    stage: str,
    percent: int,
    detail: str,
    eta_seconds: int | None,
    status: RunStatus | None = None,
) -> bool:
    label = STAGES.get(stage, stage)
    bounded_percent = max(0, min(100, int(percent)))
    changed_at = datetime.now(timezone.utc)
    owner = lease_owner_for(run_id)
    conditions = [Run.id == run_id]
    if owner is not None:
        conditions.extend(
            (
                Run.execution_slot == 1,
                Run.lease_owner == owner,
                Run.status.in_(
                    (
                        RunStatus.pending,
                        RunStatus.crawling,
                        RunStatus.analyzing,
                    )
                ),
            )
        )
    values = {
        "stage_key": stage,
        "stage_label": label,
        "stage_detail": detail[:500],
        "progress_percent": bounded_percent,
        "progress_current": bounded_percent,
        "progress_total": 100,
        "eta_seconds": eta_seconds,
        "checkpointed_at": changed_at,
        "state_changed_at": changed_at,
        "state_revision": Run.state_revision + 1,
    }
    if status is not None:
        values["status"] = status
    async with SessionLocal() as session:
        changed = await session.execute(
            update(Run)
            .where(*conditions)
            .values(**values)
            .returning(Run.state_revision)
        )
        revision = changed.scalar_one_or_none()
        await session.commit()
    if revision is None:
        if owner is not None:
            raise RunLeaseLostError(f"Run lease lost for {run_id}")
        return False

    payload = {
        "type": "progress",
        "stage": stage,
        "label": label,
        "detail": detail,
        "percent": bounded_percent,
        "eta_seconds": eta_seconds,
        "run_state": (
            "running"
            if status not in (RunStatus.completed, RunStatus.failed)
            else status.value
        ),
        "state_revision": revision,
    }
    bus.publish(run_id, payload)
    return True


async def fail_run(
    run_id: str,
    public_message: str,
    *,
    failure_stage: str | None = None,
) -> bool:
    if (
        failure_stage is not None
        and failure_stage not in NON_RETRYABLE_FAILURE_STAGES
    ):
        raise ValueError(f"Unsupported terminal failure stage: {failure_stage}")
    finished_at = datetime.now(timezone.utc)
    owner = lease_owner_for(run_id)
    conditions = [Run.id == run_id]
    if owner is not None:
        conditions.extend(
            (
                Run.execution_slot == 1,
                Run.lease_owner == owner,
                Run.status.in_(
                    (
                        RunStatus.pending,
                        RunStatus.crawling,
                        RunStatus.analyzing,
                    )
                ),
            )
        )
    failure_values = (
        {
            "stage_key": failure_stage,
            "stage_label": STAGES[failure_stage],
            "stage_detail": public_message[:500],
        }
        if failure_stage is not None
        else {}
    )
    async with SessionLocal() as session:
        changed = await session.execute(
            update(Run)
            .where(*conditions)
            .values(
                status=RunStatus.failed,
                error_message=public_message[:1000],
                eta_seconds=None,
                execution_slot=None,
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=finished_at,
                state_changed_at=finished_at,
                checkpointed_at=finished_at,
                state_revision=Run.state_revision + 1,
                **failure_values,
            )
            .returning(Run.state_revision)
        )
        revision = changed.scalar_one_or_none()
        await session.commit()
    if revision is None:
        return False
    bus.publish(
        run_id,
        {
            "type": "final",
            "status": "failed",
            "message": public_message,
            "stage": failure_stage,
            "label": STAGES.get(failure_stage) if failure_stage else None,
            "run_state": "failed",
            "state_revision": revision,
        },
    )
    return True


async def complete_run(run_id: str) -> bool:
    finished_at = datetime.now(timezone.utc)
    owner = lease_owner_for(run_id)
    conditions = [Run.id == run_id]
    if owner is not None:
        conditions.extend(
            (
                Run.execution_slot == 1,
                Run.lease_owner == owner,
                Run.status.in_(
                    (
                        RunStatus.pending,
                        RunStatus.crawling,
                        RunStatus.analyzing,
                    )
                ),
            )
        )
    async with SessionLocal() as session:
        changed = await session.execute(
            update(Run)
            .where(*conditions)
            .values(
                status=RunStatus.completed,
                stage_key="report",
                stage_label=STAGES["report"],
                stage_detail="Отчёт готов.",
                progress_current=100,
                progress_total=100,
                progress_percent=100,
                eta_seconds=0,
                error_message=None,
                execution_slot=None,
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
                finished_at=finished_at,
                state_changed_at=finished_at,
                checkpointed_at=finished_at,
                state_revision=Run.state_revision + 1,
            )
            .returning(Run.state_revision)
        )
        revision = changed.scalar_one_or_none()
        await session.commit()
    if revision is None:
        return False
    bus.publish(
        run_id,
        {
            "type": "final",
            "status": "completed",
            "run_state": "completed",
            "state_revision": revision,
        },
    )
    return True
