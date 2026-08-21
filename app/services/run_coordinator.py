"""Durable, single-slot execution queue for public site checks.

The queue itself lives in SQLite.  In-memory tasks only execute the row that
currently owns the unique database slot, so a process restart never loses the
order of pending work and two app processes cannot start two runs at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import SessionLocal, engine
from app.models import Run, RunStatus
from app.services.crawler import run_crawl
from app.services.event_bus import bus

logger = logging.getLogger(__name__)

EXECUTION_SLOT = 1
SAVED_ANSWERS_ONLY_MARKER_KEY = "_aiv_saved_answers_only_reprocess"
SAVED_ANSWERS_ONLY_MARKER_VERSION = "aiv-saved-answers-only-v1"
SAVED_ANSWERS_ONLY_MODE = "saved_answers_only"
SAVED_ONLY_TERMINAL_CLEANUP_GRACE_SECONDS = 120
ACTIVE_STATUSES = (
    RunStatus.pending,
    RunStatus.crawling,
    RunStatus.analyzing,
)


def _saved_answers_only_marker_present(config_json: object) -> bool:
    return bool(
        isinstance(config_json, dict)
        and SAVED_ANSWERS_ONLY_MARKER_KEY in config_json
    )


def _generic_queue_eligible_clause():
    """Exclude even malformed saved-only markers from the generic worker."""

    return func.json_type(
        Run.config_json,
        f'$."{SAVED_ANSWERS_ONLY_MARKER_KEY}"',
    ).is_(None)


def _saved_answers_only_marker_clause():
    return func.json_type(
        Run.config_json,
        f'$."{SAVED_ANSWERS_ONLY_MARKER_KEY}"',
    ).is_not(None)


def _marker_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid marker datetime")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _lease_is_expired(value: datetime | None, now: datetime) -> bool:
    if value is None:
        return True
    comparable = value
    if comparable.tzinfo is None and now.tzinfo is not None:
        comparable = comparable.replace(tzinfo=timezone.utc)
    return comparable <= now


def _terminal_cleanup_due(value: datetime | None, now: datetime) -> bool:
    if value is None:
        return True
    comparable = value
    if comparable.tzinfo is None and now.tzinfo is not None:
        comparable = comparable.replace(tzinfo=timezone.utc)
    return comparable + timedelta(
        seconds=SAVED_ONLY_TERMINAL_CLEANUP_GRACE_SECONDS
    ) <= now


def _saved_only_terminal_restore_values(
    run: Run,
    *,
    now: datetime,
) -> tuple[dict[str, Any], RunStatus] | None:
    """Validate a durable operator marker and restore its terminal state.

    The published report fields are deliberately absent from the returned
    values. Recovery only releases the abandoned operator lease and restores
    queue/progress metadata; it never rewrites ``report_json`` or raw answers.
    """

    config_json = run.config_json if isinstance(run.config_json, dict) else {}
    marker = config_json.get(SAVED_ANSWERS_ONLY_MARKER_KEY)
    if not isinstance(marker, dict):
        return None
    previous = marker.get("previous_terminal_state")
    previous_config = marker.get("previous_config_json")
    raw_sha256 = str(marker.get("raw_answers_sha256") or "")
    if (
        marker.get("version") != SAVED_ANSWERS_ONLY_MARKER_VERSION
        or marker.get("mode") != SAVED_ANSWERS_ONLY_MODE
        or marker.get("run_id") != run.id
        or len(raw_sha256) != 64
        or any(char not in "0123456789abcdef" for char in raw_sha256)
        or not isinstance(previous, dict)
        or not isinstance(previous_config, dict)
        or SAVED_ANSWERS_ONLY_MARKER_KEY in previous_config
    ):
        return None
    try:
        previous_status = RunStatus(str(previous.get("status") or ""))
        if previous_status not in (RunStatus.completed, RunStatus.failed):
            return None
        progress_current = int(previous["progress_current"])
        progress_total = int(previous["progress_total"])
        progress_percent = int(previous["progress_percent"])
        checkpointed_at = _marker_datetime(previous.get("checkpointed_at"))
        finished_at = _marker_datetime(previous.get("finished_at"))
    except (KeyError, TypeError, ValueError):
        return None
    return (
        {
            "status": previous_status,
            "config_json": previous_config,
            "progress_current": progress_current,
            "progress_total": progress_total,
            "progress_percent": progress_percent,
            "stage_key": previous.get("stage_key"),
            "stage_label": previous.get("stage_label"),
            "stage_detail": previous.get("stage_detail"),
            "eta_seconds": previous.get("eta_seconds"),
            "error_message": previous.get("error_message"),
            "checkpointed_at": checkpointed_at,
            "finished_at": finished_at,
            "execution_slot": None,
            "lease_owner": None,
            "lease_expires_at": None,
            "heartbeat_at": None,
            "state_revision": func.coalesce(Run.state_revision, 0) + 1,
            "state_changed_at": now,
        },
        previous_status,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RunClaim:
    run_id: str
    owner: str


@dataclass(frozen=True)
class ReleasedRun:
    run_id: str
    progress_percent: int
    state_revision: int
    resume_count: int


async def queue_positions(
    session: AsyncSession,
) -> dict[str, int]:
    """Return stable 1-based positions for rows waiting for the global slot."""

    ids = (
        await session.execute(
            select(Run.id)
            .where(
                Run.status == RunStatus.pending,
                Run.execution_slot.is_(None),
                _generic_queue_eligible_clause(),
            )
            .order_by(Run.created_at.asc(), Run.id.asc())
        )
    ).scalars()
    return {
        run_id: position
        for position, run_id in enumerate(ids, start=1)
    }


async def pending_run_count(session: AsyncSession) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(Run)
                .where(
                    Run.status == RunStatus.pending,
                    Run.execution_slot.is_(None),
                    _generic_queue_eligible_clause(),
                )
            )
        ).scalar_one()
    )


async def recover_expired_leases(
    *,
    now: datetime | None = None,
) -> int:
    """Recover abandoned generic leases and fail-close saved-only markers."""

    current_time = now or _utcnow()
    async with SessionLocal() as session:
        terminal_rows = list(
            (
                await session.execute(
                    select(Run).where(
                        Run.status.in_(
                            (RunStatus.completed, RunStatus.failed)
                        ),
                        Run.execution_slot.is_(None),
                        _saved_answers_only_marker_clause(),
                    )
                )
            )
            .scalars()
            .all()
        )
        active_rows = list(
            (
                await session.execute(
                    select(Run).where(Run.status.in_(ACTIVE_STATUSES))
                )
            )
            .scalars()
            .all()
        )
        saved_only_recovered: list[
            tuple[str, RunStatus, int, int]
        ] = []
        generic_expired_ids: list[str] = []
        for run in terminal_rows:
            if not _terminal_cleanup_due(
                run.state_changed_at,
                current_time,
            ):
                continue
            restored = _saved_only_terminal_restore_values(
                run,
                now=current_time,
            )
            # Unknown/malformed reserved data stays terminal and excluded from
            # the generic queue. An operator must inspect it explicitly.
            if restored is None:
                continue
            if run.status == RunStatus.completed:
                cleaned_config = dict(run.config_json or {})
                cleaned_config.pop(SAVED_ANSWERS_ONLY_MARKER_KEY, None)
                values: dict[str, Any] = {
                    "config_json": cleaned_config,
                    "execution_slot": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "heartbeat_at": None,
                    "state_revision": (
                        func.coalesce(Run.state_revision, 0) + 1
                    ),
                    "state_changed_at": current_time,
                }
                restored_status = RunStatus.completed
            else:
                values, restored_status = restored
            changed = await session.execute(
                update(Run)
                .where(
                    Run.id == run.id,
                    Run.status == run.status,
                    Run.execution_slot.is_(None),
                    Run.state_revision == run.state_revision,
                    _saved_answers_only_marker_clause(),
                )
                .values(**values)
                .returning(Run.state_revision, Run.resume_count)
            )
            changed_row = changed.one_or_none()
            if changed_row is not None:
                revision, resume_count = changed_row
                saved_only_recovered.append(
                    (
                        run.id,
                        restored_status,
                        int(revision or 0),
                        int(resume_count or 0),
                    )
                )
        for run in active_rows:
            marker_present = _saved_answers_only_marker_present(run.config_json)
            lease_abandoned = bool(
                run.status == RunStatus.pending
                or run.execution_slot is None
                or run.lease_owner is None
                or _lease_is_expired(run.lease_expires_at, current_time)
            )
            if marker_present:
                if not lease_abandoned:
                    continue
                restored = _saved_only_terminal_restore_values(
                    run,
                    now=current_time,
                )
                if restored is None:
                    restored_status = RunStatus.failed
                    values: dict[str, Any] = {
                        "status": restored_status,
                        "execution_slot": None,
                        "lease_owner": None,
                        "lease_expires_at": None,
                        "heartbeat_at": None,
                        "eta_seconds": None,
                        "error_message": (
                            "Не удалось безопасно восстановить операторский "
                            "переанализ. Исходные ответы и опубликованный "
                            "отчёт не изменялись этим восстановлением."
                        ),
                        "finished_at": current_time,
                        "checkpointed_at": current_time,
                        "state_revision": (
                            func.coalesce(Run.state_revision, 0) + 1
                        ),
                        "state_changed_at": current_time,
                    }
                else:
                    values, restored_status = restored
                changed = await session.execute(
                    update(Run)
                    .where(
                        Run.id == run.id,
                        Run.status == run.status,
                        Run.state_revision == run.state_revision,
                    )
                    .values(**values)
                    .returning(Run.state_revision, Run.resume_count)
                )
                changed_row = changed.one_or_none()
                if changed_row is not None:
                    revision, resume_count = changed_row
                    saved_only_recovered.append(
                        (
                            run.id,
                            restored_status,
                            int(revision or 0),
                            int(resume_count or 0),
                        )
                    )
                continue
            if (
                run.execution_slot is not None
                and (
                    run.lease_owner is None
                    or _lease_is_expired(
                        run.lease_expires_at,
                        current_time,
                    )
                )
            ):
                generic_expired_ids.append(run.id)

        recovered = await session.execute(
            update(Run)
            .where(
                Run.id.in_(generic_expired_ids),
                Run.execution_slot.is_not(None),
                Run.status.in_(ACTIVE_STATUSES),
                _generic_queue_eligible_clause(),
                or_(
                    Run.lease_owner.is_(None),
                    Run.lease_expires_at.is_(None),
                    Run.lease_expires_at <= current_time,
                ),
            )
            .values(
                status=RunStatus.pending,
                execution_slot=None,
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
                stage_key="recovering",
                stage_label="Восстанавливаем проверку",
                stage_detail=(
                    "Возобновляем проверку с уже сохранённых данных."
                ),
                eta_seconds=None,
                error_message=None,
                resume_count=func.coalesce(Run.resume_count, 0) + 1,
                resume_reason="lease_expired",
                last_resumed_at=current_time,
                state_revision=func.coalesce(Run.state_revision, 0) + 1,
                state_changed_at=current_time,
            )
            .returning(
                Run.id,
                Run.progress_percent,
                Run.state_revision,
                Run.resume_count,
            )
        )
        recovered_rows = list(recovered.all())
        await session.commit()
        count = len(recovered_rows) + len(saved_only_recovered)
    for run_id, status, revision, resume_count in saved_only_recovered:
        bus.reset(run_id)
        payload = {
            "type": "final",
            "status": status.value,
            "run_state": status.value,
            "state_revision": revision,
            "stream_epoch": resume_count,
        }
        bus.publish(run_id, payload)
    for run_id, progress_percent, revision, resume_count in recovered_rows:
        bus.reset(run_id)
        bus.publish(
            run_id,
            {
                "type": "progress",
                "stage": "recovering",
                "label": "Восстанавливаем проверку",
                "detail": "Возобновляем проверку с уже сохранённых данных.",
                "percent": int(progress_percent or 0),
                "eta_seconds": None,
                "run_state": "recovering",
                "state_revision": int(revision or 0),
                "stream_epoch": int(resume_count or 0),
            },
        )
    if count:
        logger.warning("Recovered %d abandoned run lease/marker(s)", count)
    return count


async def _claim_next_run(
    owner: str,
    *,
    lease_seconds: int,
    now: datetime | None = None,
) -> RunClaim | None:
    """Atomically reserve the oldest queued run.

    ``BEGIN IMMEDIATE`` serializes the short claim transaction in SQLite.  The
    partial unique index on ``execution_slot`` is the final cross-process guard.
    """

    current_time = now or _utcnow()
    lease_until = current_time + timedelta(seconds=max(15, lease_seconds))
    async with engine.connect() as connection:
        await connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            candidate = (
                await connection.execute(
                    select(Run.id, Run.started_at)
                    .where(
                        Run.status == RunStatus.pending,
                        Run.execution_slot.is_(None),
                        _generic_queue_eligible_clause(),
                    )
                    .order_by(Run.created_at.asc(), Run.id.asc())
                    .limit(1)
                )
            ).one_or_none()
            if candidate is None:
                await connection.commit()
                return None
            candidate_id, started_at = candidate

            slot_busy = (
                await connection.execute(
                    select(func.count())
                    .select_from(Run)
                    .where(Run.execution_slot == EXECUTION_SLOT)
                )
            ).scalar_one()
            if slot_busy:
                await connection.commit()
                return None

            claimed = await connection.execute(
                update(Run)
                .where(
                    Run.id == candidate_id,
                    Run.status == RunStatus.pending,
                    Run.execution_slot.is_(None),
                    _generic_queue_eligible_clause(),
                )
                .values(
                    execution_slot=EXECUTION_SLOT,
                    lease_owner=owner,
                    lease_expires_at=lease_until,
                    heartbeat_at=current_time,
                    attempt_count=func.coalesce(Run.attempt_count, 0) + 1,
                    started_at=started_at or current_time,
                    finished_at=None,
                    stage_key="starting",
                    stage_label="Запускаем проверку",
                    stage_detail=(
                        "Освободили вычислительную ячейку и начинаем с "
                        "сохранённых данных."
                    ),
                    eta_seconds=None,
                    error_message=None,
                    state_revision=Run.state_revision + 1,
                    state_changed_at=current_time,
                )
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
    if claimed.rowcount != 1:
        return None
    return RunClaim(run_id=candidate_id, owner=owner)


async def _renew_lease(
    claim: RunClaim,
    *,
    lease_seconds: int,
) -> bool:
    current_time = _utcnow()
    async with SessionLocal() as session:
        renewed = await session.execute(
            update(Run)
            .where(
                Run.id == claim.run_id,
                Run.execution_slot == EXECUTION_SLOT,
                Run.lease_owner == claim.owner,
                Run.status.in_(ACTIVE_STATUSES),
            )
            .values(
                heartbeat_at=current_time,
                lease_expires_at=(
                    current_time
                    + timedelta(seconds=max(15, lease_seconds))
                ),
            )
        )
        await session.commit()
        return renewed.rowcount == 1


async def _release_claim(
    claim: RunClaim,
    *,
    detail: str,
    reason: str,
) -> ReleasedRun | None:
    """Requeue a non-terminal claimed row owned by this process."""

    changed_at = _utcnow()
    async with SessionLocal() as session:
        released = await session.execute(
            update(Run)
            .where(
                Run.id == claim.run_id,
                Run.execution_slot == EXECUTION_SLOT,
                Run.lease_owner == claim.owner,
                Run.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status=RunStatus.pending,
                execution_slot=None,
                lease_owner=None,
                lease_expires_at=None,
                heartbeat_at=None,
                stage_key="recovering",
                stage_label="Восстанавливаем проверку",
                stage_detail=detail[:500],
                eta_seconds=None,
                resume_count=func.coalesce(Run.resume_count, 0) + 1,
                resume_reason=reason[:160],
                last_resumed_at=changed_at,
                state_revision=func.coalesce(Run.state_revision, 0) + 1,
                state_changed_at=changed_at,
            )
            .returning(
                Run.progress_percent,
                Run.state_revision,
                Run.resume_count,
            )
        )
        row = released.one_or_none()
        await session.commit()
    if row is None:
        return None
    progress_percent, revision, resume_count = row
    return ReleasedRun(
        run_id=claim.run_id,
        progress_percent=int(progress_percent or 0),
        state_revision=int(revision or 0),
        resume_count=int(resume_count or 0),
    )


class RunCoordinator:
    """Own the global execution slot and keep its lease alive."""

    def __init__(
        self,
        *,
        lease_seconds: int | None = None,
        poll_seconds: float | None = None,
    ) -> None:
        identity = (
            f"{socket.gethostname()}:{os.getpid()}:"
            f"{uuid.uuid4().hex[:10]}"
        )
        self.instance_id = identity[:72]
        self.lease_seconds = max(
            15,
            int(lease_seconds or settings.RUN_LEASE_SECONDS),
        )
        self.poll_seconds = max(
            0.2,
            float(
                poll_seconds
                if poll_seconds is not None
                else settings.RUN_COORDINATOR_POLL_SECONDS
            ),
        )
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._active_worker: asyncio.Task | None = None
        self._stopping = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def wake(self) -> None:
        self._wake_event.set()

    async def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        await recover_expired_leases()
        self._task = asyncio.create_task(
            self._run(),
            name="aiv-run-coordinator",
        )
        self.wake()

    async def stop(self) -> None:
        self._stopping = True
        self.wake()
        task = self._task
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task = None
        self._active_worker = None

    async def _wait_for_work(self) -> None:
        try:
            await asyncio.wait_for(
                self._wake_event.wait(),
                timeout=self.poll_seconds,
            )
        except asyncio.TimeoutError:
            pass
        self._wake_event.clear()

    async def _heartbeat(self, claim: RunClaim) -> None:
        interval = max(5.0, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not await _renew_lease(
                claim,
                lease_seconds=self.lease_seconds,
            ):
                raise RuntimeError(
                    f"Run lease lost for {claim.run_id}"
                )

    async def _execute(self, claim: RunClaim) -> None:
        # Defence in depth: even if a future queue query accidentally claims a
        # saved-answer-only row, never hand it to run_crawl/analyze_run. The
        # recovery path returns its durable marker to a terminal state.
        async with SessionLocal() as session:
            claimed_config = (
                await session.execute(
                    select(Run.config_json).where(Run.id == claim.run_id)
                )
            ).scalar_one_or_none()
        if _saved_answers_only_marker_present(claimed_config):
            logger.error(
                "Refusing generic execution for saved-answer-only run %s",
                claim.run_id,
            )
            await recover_expired_leases()
            return

        worker = asyncio.create_task(
            run_crawl(claim.run_id, lease_owner=claim.owner),
            name=f"aiv-run-{claim.run_id}",
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(claim),
            name=f"aiv-run-heartbeat-{claim.run_id}",
        )
        self._active_worker = worker
        interrupted = False
        try:
            done, _pending = await asyncio.wait(
                {worker, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker in done:
                await worker
            else:
                interrupted = True
                heartbeat_error = heartbeat.exception()
                logger.error(
                    "Lease heartbeat stopped before run %s completed: %s",
                    claim.run_id,
                    heartbeat_error,
                )
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        except asyncio.CancelledError:
            interrupted = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            raise
        except Exception:
            interrupted = True
            logger.exception(
                "Run coordinator wrapper failed for %s",
                claim.run_id,
            )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._active_worker = None
            detail = (
                "Сервис перезапускается. Проверка продолжится автоматически."
                if self._stopping
                else (
                    "Проверка продолжится автоматически с уже сохранённых "
                    "данных."
                )
            )
            released = await _release_claim(
                claim,
                detail=detail,
                reason=(
                    "service_restart"
                    if self._stopping
                    else (
                        "lease_interrupted"
                        if interrupted
                        else "worker_returned_active"
                    )
                ),
            )
            if released is not None:
                bus.reset(claim.run_id)
                bus.publish(
                    claim.run_id,
                    {
                        "type": "progress",
                        "stage": "recovering",
                        "label": "Восстанавливаем проверку",
                        "detail": detail,
                        "percent": released.progress_percent,
                        "eta_seconds": None,
                        "run_state": "recovering",
                        "state_revision": released.state_revision,
                        "stream_epoch": released.resume_count,
                    },
                )

    async def _run(self) -> None:
        while not self._stopping:
            await recover_expired_leases()
            claim_owner = (
                f"{self.instance_id}:{uuid.uuid4().hex[:12]}"
            )[:96]
            claim = await _claim_next_run(
                claim_owner,
                lease_seconds=self.lease_seconds,
            )
            if claim is None:
                await self._wait_for_work()
                continue
            await self._execute(claim)


coordinator = RunCoordinator()
