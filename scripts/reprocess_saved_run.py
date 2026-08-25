"""Rebuild one report from saved model answers without rerunning the panel."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, func, or_, select, update

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import SessionLocal, engine
from app.models import ModelAnswer, Run, RunStatus, VisibilityPrompt
from app.services import analyzer
from app.services.run_coordinator import (
    SAVED_ANSWERS_ONLY_MARKER_KEY,
    SAVED_ANSWERS_ONLY_MARKER_VERSION,
    SAVED_ANSWERS_ONLY_MODE,
)
from app.services.run_lease import bind_run_lease


EXECUTION_SLOT = 1
REPROCESS_LEASE_SECONDS = 90
REPROCESS_HEARTBEAT_SECONDS = 30.0
ACTIVE_QUEUE_STATUSES = (
    RunStatus.pending,
    RunStatus.crawling,
    RunStatus.analyzing,
)
EXPECTED_PROMPT_COUNT = 9
EXPECTED_DISCOVERY_PROMPT_COUNT = 6
EXPECTED_BRAND_PROMPT_COUNT = 3
EXPECTED_WEB_PROVIDERS = frozenset(
    {"openai", "gemini", "perplexity", "deepseek", "claude"}
)
EXPECTED_MEMORY_PROVIDERS = frozenset(
    {"openai", "gemini", "deepseek", "claude"}
)
EXPECTED_PANEL_CELL_COUNT = 81


class ReprocessGuardError(RuntimeError):
    """The requested run is unsafe or unsuitable for saved-answer reprocessing."""


class ReprocessExecutionError(RuntimeError):
    """Saved-answer reprocessing ran but did not complete the report."""


@dataclass(frozen=True)
class PreviousRunState:
    config_json: dict
    status: RunStatus
    progress_current: int
    progress_total: int
    progress_percent: int
    stage_key: str | None
    stage_label: str | None
    stage_detail: str | None
    eta_seconds: int | None
    error_message: str | None
    checkpointed_at: datetime | None
    finished_at: datetime | None
    analysis_markdown: str | None
    report_json: dict | None


@dataclass(frozen=True)
class ReprocessClaim:
    run_id: str
    owner: str
    attempt_count: int
    resume_count: int
    completed_answers: int
    total_answers: int
    raw_answers_sha256: str
    raw_answer_snapshot: tuple[dict[str, object], ...]
    previous: PreviousRunState


def canonical_run_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ReprocessGuardError("run_id должен быть корректным UUID.") from exc


def _operator_owner() -> str:
    return f"operator-reprocess:{os.getpid()}:{uuid.uuid4().hex}"[:96]


def _json_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _saved_answers_only_marker(
    *,
    run_id: str,
    owner: str,
    attempt_count: int,
    raw_answers_sha256: str,
    previous: PreviousRunState,
) -> dict[str, object]:
    return {
        "version": SAVED_ANSWERS_ONLY_MARKER_VERSION,
        "mode": SAVED_ANSWERS_ONLY_MODE,
        "run_id": run_id,
        "owner": owner,
        "attempt_count": attempt_count,
        "raw_answers_sha256": raw_answers_sha256,
        "previous_config_json": copy.deepcopy(previous.config_json),
        "previous_terminal_state": {
            "status": previous.status.value,
            "progress_current": previous.progress_current,
            "progress_total": previous.progress_total,
            "progress_percent": previous.progress_percent,
            "stage_key": previous.stage_key,
            "stage_label": previous.stage_label,
            "stage_detail": previous.stage_detail,
            "eta_seconds": previous.eta_seconds,
            "error_message": previous.error_message,
            "checkpointed_at": _json_datetime(previous.checkpointed_at),
            "finished_at": _json_datetime(previous.finished_at),
        },
    }


def _claim_marker_matches(
    config_json: object,
    claim: ReprocessClaim,
) -> bool:
    if not isinstance(config_json, dict):
        return False
    marker = config_json.get(SAVED_ANSWERS_ONLY_MARKER_KEY)
    return bool(
        isinstance(marker, dict)
        and marker.get("version") == SAVED_ANSWERS_ONLY_MARKER_VERSION
        and marker.get("mode") == SAVED_ANSWERS_ONLY_MODE
        and marker.get("run_id") == claim.run_id
        and marker.get("owner") == claim.owner
        and marker.get("attempt_count") == claim.attempt_count
        and marker.get("raw_answers_sha256") == claim.raw_answers_sha256
    )


def _model_answer_fingerprint_rows(
    rows: list[ModelAnswer] | list[Mapping[str, object]],
) -> str:
    """Hash every persisted panel-answer field in a canonical order."""

    def value(row: ModelAnswer | Mapping[str, object], key: str) -> object:
        return row.get(key) if isinstance(row, Mapping) else getattr(row, key)

    payload = [
        {
            "id": value(row, "id"),
            "run_id": value(row, "run_id"),
            "prompt_id": value(row, "prompt_id"),
            "provider_key": value(row, "provider_key"),
            "model": value(row, "model"),
            "mode": value(row, "mode"),
            "status": value(row, "status"),
            "response_text": value(row, "response_text"),
            "citations_json": value(row, "citations_json"),
            "usage_json": value(row, "usage_json"),
            "error_message": value(row, "error_message"),
            "created_at": (
                value(row, "created_at").isoformat()
                if value(row, "created_at") is not None
                else None
            ),
        }
        for row in sorted(rows, key=lambda item: int(value(item, "id")))
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


async def _model_answer_fingerprint(run_id: str) -> tuple[int, str]:
    async with SessionLocal() as session:
        rows = list(
            (
                await session.execute(
                    select(ModelAnswer)
                    .where(ModelAnswer.run_id == run_id)
                    .order_by(ModelAnswer.id)
                )
            )
            .scalars()
            .all()
        )
    return len(rows), _model_answer_fingerprint_rows(rows)


def _validate_complete_saved_panel(
    prompt_rows: list[Mapping[str, object]],
    answer_rows: list[Mapping[str, object]],
) -> None:
    """Reject partial panels before any downstream LLM token is spent."""

    if len(prompt_rows) != EXPECTED_PROMPT_COUNT:
        raise ReprocessGuardError(
            "Saved-answer reprocess requires exactly nine persisted prompts."
        )
    roles = [str(row.get("role") or "") for row in prompt_rows]
    if roles.count("unbranded_discovery") != EXPECTED_DISCOVERY_PROMPT_COUNT or roles.count(
        "brand_diagnostic"
    ) != EXPECTED_BRAND_PROMPT_COUNT:
        raise ReprocessGuardError(
            "Persisted prompt roles do not match the 6 discovery + 3 brand contract."
        )
    prompt_ids = {int(row["id"]) for row in prompt_rows}
    if len(answer_rows) != EXPECTED_PANEL_CELL_COUNT:
        raise ReprocessGuardError(
            "Saved-answer reprocess requires the complete 81-cell panel; "
            f"found {len(answer_rows)} cells."
        )
    cells_by_prompt: dict[int, dict[str, set[str]]] = {
        prompt_id: {"web": set(), "memory": set()}
        for prompt_id in prompt_ids
    }
    for row in answer_rows:
        prompt_id = int(row.get("prompt_id") or 0)
        provider = str(row.get("provider_key") or "")
        mode = str(row.get("mode") or "")
        response_text = str(row.get("response_text") or "").strip()
        if prompt_id not in cells_by_prompt or mode not in {"web", "memory"}:
            raise ReprocessGuardError(
                "Saved panel contains a cell outside the persisted 9-prompt grid."
            )
        if str(row.get("status") or "") != "completed" or not response_text:
            raise ReprocessGuardError(
                "Every saved panel cell must be completed and contain raw text."
            )
        if provider in cells_by_prompt[prompt_id][mode]:
            raise ReprocessGuardError(
                "Saved panel contains a duplicate provider/mode cell."
            )
        cells_by_prompt[prompt_id][mode].add(provider)
    for prompt_id, modes in cells_by_prompt.items():
        if modes["web"] != EXPECTED_WEB_PROVIDERS or modes[
            "memory"
        ] != EXPECTED_MEMORY_PROVIDERS:
            raise ReprocessGuardError(
                "Saved panel is incomplete for prompt "
                f"{prompt_id}: web={sorted(modes['web'])}, "
                f"memory={sorted(modes['memory'])}."
            )


async def _claim_eligible_run(run_id: str) -> ReprocessClaim:
    """Check the whole durable queue and reserve its only execution slot."""

    now = datetime.now(timezone.utc)
    lease_expires_at = now + timedelta(seconds=REPROCESS_LEASE_SECONDS)
    owner = _operator_owner()
    async with engine.connect() as connection:
        await connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            run = (
                await connection.execute(
                    select(
                        Run.config_json,
                        Run.status,
                        Run.progress_current,
                        Run.progress_total,
                        Run.progress_percent,
                        Run.stage_key,
                        Run.stage_label,
                        Run.stage_detail,
                        Run.eta_seconds,
                        Run.error_message,
                        Run.checkpointed_at,
                        Run.finished_at,
                        Run.analysis_markdown,
                        Run.report_json,
                        Run.execution_slot,
                        Run.attempt_count,
                        Run.resume_count,
                    ).where(Run.id == run_id)
                )
            ).one_or_none()
            if run is None:
                await connection.rollback()
                raise ReprocessGuardError(f"Проверка {run_id} не найдена.")
            original_config = (
                copy.deepcopy(run.config_json)
                if isinstance(run.config_json, dict)
                else {}
            )
            if SAVED_ANSWERS_ONLY_MARKER_KEY in original_config:
                await connection.rollback()
                raise ReprocessGuardError(
                    "У проверки уже есть незавершённый saved-answers-only "
                    "маркер. Дождитесь fail-closed восстановления lease."
                )
            if run.status in ACTIVE_QUEUE_STATUSES:
                await connection.rollback()
                raise ReprocessGuardError(
                    "Нельзя переанализировать активную проверку "
                    f"со статусом {run.status.value}."
                )
            if run.execution_slot is not None:
                await connection.rollback()
                raise ReprocessGuardError(
                    "Нельзя переанализировать проверку, пока её "
                    "execution_slot уже занят."
                )

            blocker = (
                await connection.execute(
                    select(Run.id, Run.status, Run.execution_slot)
                    .where(
                        Run.id != run_id,
                        or_(
                            Run.status.in_(ACTIVE_QUEUE_STATUSES),
                            Run.execution_slot.is_not(None),
                        ),
                    )
                    .order_by(Run.created_at.asc(), Run.id.asc())
                    .limit(1)
                )
            ).one_or_none()
            if blocker is not None:
                blocker_id, blocker_status, blocker_slot = blocker
                slot_detail = (
                    ", execution_slot занят"
                    if blocker_slot is not None
                    else ""
                )
                await connection.rollback()
                raise ReprocessGuardError(
                    "Общая durable queue занята: "
                    f"{blocker_id} ({blocker_status.value}{slot_detail}). "
                    "Дождитесь завершения всех pending/crawling/analyzing "
                    "проверок."
                )

            completed_answers = int(
                (
                    await connection.execute(
                        select(func.count(ModelAnswer.id)).where(
                            ModelAnswer.run_id == run_id,
                            ModelAnswer.status == "completed",
                            ModelAnswer.response_text.is_not(None),
                            func.length(
                                func.trim(ModelAnswer.response_text)
                            )
                            > 0,
                        )
                    )
                ).scalar_one()
            )
            if completed_answers < 1:
                await connection.rollback()
                raise ReprocessGuardError(
                    "У проверки нет завершённых сохранённых raw-ответов моделей."
                )

            answer_rows = list(
                (
                    await connection.execute(
                        select(
                            ModelAnswer.id,
                            ModelAnswer.run_id,
                            ModelAnswer.prompt_id,
                            ModelAnswer.provider_key,
                            ModelAnswer.model,
                            ModelAnswer.mode,
                            ModelAnswer.status,
                            ModelAnswer.response_text,
                            ModelAnswer.citations_json,
                            ModelAnswer.usage_json,
                            ModelAnswer.error_message,
                            ModelAnswer.created_at,
                        )
                        .where(ModelAnswer.run_id == run_id)
                        .order_by(ModelAnswer.id)
                    )
                )
                .mappings()
                .all()
            )
            prompt_rows = list(
                (
                    await connection.execute(
                        select(
                            VisibilityPrompt.id,
                            VisibilityPrompt.prompt_key,
                            VisibilityPrompt.role,
                            VisibilityPrompt.sequence,
                        )
                        .where(VisibilityPrompt.run_id == run_id)
                        .order_by(
                            VisibilityPrompt.sequence,
                            VisibilityPrompt.id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            _validate_complete_saved_panel(prompt_rows, answer_rows)
            raw_answers_sha256 = _model_answer_fingerprint_rows(answer_rows)

            previous = PreviousRunState(
                config_json=original_config,
                status=run.status,
                progress_current=run.progress_current,
                progress_total=run.progress_total,
                progress_percent=run.progress_percent,
                stage_key=run.stage_key,
                stage_label=run.stage_label,
                stage_detail=run.stage_detail,
                eta_seconds=run.eta_seconds,
                error_message=run.error_message,
                checkpointed_at=run.checkpointed_at,
                finished_at=run.finished_at,
                analysis_markdown=run.analysis_markdown,
                report_json=run.report_json,
            )
            claim_attempt_count = int(run.attempt_count or 0) + 1
            claim_resume_count = int(run.resume_count or 0)
            marked_config = copy.deepcopy(original_config)
            marked_config[SAVED_ANSWERS_ONLY_MARKER_KEY] = (
                _saved_answers_only_marker(
                    run_id=run_id,
                    owner=owner,
                    attempt_count=claim_attempt_count,
                    raw_answers_sha256=raw_answers_sha256,
                    previous=previous,
                )
            )
            claimed = await connection.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status == run.status,
                    Run.status.in_(
                        [RunStatus.failed, RunStatus.completed]
                    ),
                    Run.execution_slot.is_(None),
                    Run.config_json == original_config,
                )
                .values(
                    config_json=marked_config,
                    status=RunStatus.analyzing,
                    execution_slot=EXECUTION_SLOT,
                    lease_owner=owner,
                    lease_expires_at=lease_expires_at,
                    heartbeat_at=now,
                    error_message=None,
                    finished_at=None,
                    attempt_count=claim_attempt_count,
                    state_revision=func.coalesce(Run.state_revision, 0) + 1,
                    state_changed_at=now,
                    stage_key="knowledge_gap",
                    stage_label="Переанализируем сохранённые ответы",
                    stage_detail=(
                        "Исходные ответы сохранены; модельная панель "
                        "не вызывается."
                    ),
                )
            )
            if claimed.rowcount != 1:
                await connection.rollback()
                raise ReprocessGuardError(
                    "Проверку уже забрал другой процесс. "
                    "Повторный запуск отменён."
                )
            await connection.commit()
        except BaseException:
            if connection.in_transaction():
                await connection.rollback()
            raise
    return ReprocessClaim(
        run_id=run_id,
        owner=owner,
        attempt_count=claim_attempt_count,
        resume_count=claim_resume_count,
        completed_answers=completed_answers,
        total_answers=len(answer_rows),
        raw_answers_sha256=raw_answers_sha256,
        raw_answer_snapshot=tuple(dict(row) for row in answer_rows),
        previous=previous,
    )


async def _renew_reprocess_lease(claim: ReprocessClaim) -> bool:
    now = datetime.now(timezone.utc)
    async with SessionLocal() as session:
        renewed = await session.execute(
            update(Run)
            .where(
                Run.id == claim.run_id,
                Run.status == RunStatus.analyzing,
                Run.execution_slot == EXECUTION_SLOT,
                Run.lease_owner == claim.owner,
            )
            .values(
                heartbeat_at=now,
                lease_expires_at=(
                    now + timedelta(seconds=REPROCESS_LEASE_SECONDS)
                ),
            )
        )
        await session.commit()
        return renewed.rowcount == 1


async def _heartbeat_reprocess_lease(claim: ReprocessClaim) -> None:
    while True:
        await asyncio.sleep(REPROCESS_HEARTBEAT_SECONDS)
        if not await _renew_reprocess_lease(claim):
            raise ReprocessExecutionError(
                "Потерян общий execution_slot; переанализ остановлен."
            )


async def _execute_claimed_reprocess(claim: ReprocessClaim) -> None:
    with bind_run_lease(claim.run_id, claim.owner):
        worker = asyncio.create_task(
            analyzer.reprocess_saved_answers(claim.run_id),
            name=f"operator-reprocess-{claim.run_id}",
        )
    heartbeat = asyncio.create_task(
        _heartbeat_reprocess_lease(claim),
        name=f"operator-reprocess-heartbeat-{claim.run_id}",
    )
    try:
        done, _pending = await asyncio.wait(
            {worker, heartbeat},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if worker in done:
            await worker
            return
        heartbeat_error = heartbeat.exception()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        if isinstance(heartbeat_error, ReprocessExecutionError):
            raise heartbeat_error
        raise ReprocessExecutionError(
            "Не удалось продлить общий execution_slot; "
            "переанализ остановлен."
        ) from heartbeat_error
    except asyncio.CancelledError:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        raise
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)


async def _restore_model_answer_snapshot(claim: ReprocessClaim) -> None:
    """Restore the immutable raw corpus after an unexpected write."""

    snapshot_ids = {
        int(item["id"])
        for item in claim.raw_answer_snapshot
    }
    async with SessionLocal() as session:
        current_ids = set(
            (
                await session.execute(
                    select(ModelAnswer.id).where(
                        ModelAnswer.run_id == claim.run_id
                    )
                )
            ).scalars()
        )
        extra_ids = current_ids - snapshot_ids
        if extra_ids:
            await session.execute(
                delete(ModelAnswer).where(
                    ModelAnswer.run_id == claim.run_id,
                    ModelAnswer.id.in_(extra_ids),
                )
            )
        for item in claim.raw_answer_snapshot:
            values = {
                key: item[key]
                for key in (
                    "run_id",
                    "prompt_id",
                    "provider_key",
                    "model",
                    "mode",
                    "status",
                    "response_text",
                    "citations_json",
                    "usage_json",
                    "error_message",
                    "created_at",
                )
            }
            restored = await session.execute(
                update(ModelAnswer)
                .where(
                    ModelAnswer.id == int(item["id"]),
                    ModelAnswer.run_id == claim.run_id,
                )
                .values(**values)
            )
            if restored.rowcount == 0:
                session.add(ModelAnswer(id=int(item["id"]), **values))
        await session.commit()


async def _assert_raw_answers_unchanged(claim: ReprocessClaim) -> None:
    answer_count, raw_answers_sha256 = await _model_answer_fingerprint(
        claim.run_id
    )
    if (
        answer_count == claim.total_answers
        and raw_answers_sha256 == claim.raw_answers_sha256
    ):
        return
    await _restore_model_answer_snapshot(claim)
    restored_count, restored_sha256 = await _model_answer_fingerprint(
        claim.run_id
    )
    if (
        restored_count != claim.total_answers
        or restored_sha256 != claim.raw_answers_sha256
    ):
        raise ReprocessExecutionError(
            "Защитная сверка обнаружила изменение сохранённых ответов, "
            "и автоматическое восстановление raw-корпуса не удалось."
        )
    raise ReprocessExecutionError(
        "Защитная сверка обнаружила изменение сохранённых ответов. "
        "Raw-корпус восстановлен из контрольного снимка; результат "
        "переанализа отклонён."
    )


async def _release_reprocess_claim(
    claim: ReprocessClaim,
    *,
    successful: bool,
) -> bool:
    """Release our attempt or restore its terminal failure safely.

    ``fail_run`` deliberately clears the lease before returning.  In that
    state ownership can no longer be proved by ``lease_owner`` alone, so the
    monotonically increasing attempt/resume counters form the generation
    token.  A newer retry changes at least one of them and therefore cannot be
    overwritten by this cleanup.
    """

    now = datetime.now(timezone.utc)
    async with engine.connect() as connection:
        await connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            current = (
                await connection.execute(
                    select(
                        Run.config_json,
                        Run.status,
                        Run.execution_slot,
                        Run.lease_owner,
                        Run.attempt_count,
                        Run.resume_count,
                    ).where(Run.id == claim.run_id)
                )
            ).one_or_none()
            if current is None:
                await connection.commit()
                return False

            marker_matches = _claim_marker_matches(
                current.config_json,
                claim,
            )
            owned = bool(
                marker_matches
                and current.execution_slot == EXECUTION_SLOT
                and current.lease_owner == claim.owner
                and int(current.attempt_count or 0) == claim.attempt_count
                and int(current.resume_count or 0) == claim.resume_count
            )
            orphaned_own_failure = bool(
                marker_matches
                and not successful
                and current.status in (RunStatus.failed, RunStatus.completed)
                and current.execution_slot is None
                and current.lease_owner is None
                and int(current.attempt_count or 0) == claim.attempt_count
                and int(current.resume_count or 0) == claim.resume_count
            )
            terminal_own_success = bool(
                marker_matches
                and successful
                and current.status == RunStatus.completed
                and current.execution_slot is None
                and current.lease_owner is None
                and int(current.attempt_count or 0) == claim.attempt_count
                and int(current.resume_count or 0) == claim.resume_count
            )
            if not owned and not orphaned_own_failure and not terminal_own_success:
                await connection.commit()
                return False

            cleaned_config = copy.deepcopy(current.config_json)
            cleaned_config.pop(SAVED_ANSWERS_ONLY_MARKER_KEY, None)
            values: dict[str, object] = {
                "config_json": cleaned_config,
                "execution_slot": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
                "state_revision": func.coalesce(Run.state_revision, 0) + 1,
                "state_changed_at": now,
            }
            # A reprocess may persist ``failed`` before the CLI observes the
            # exception. A previously published report must remain published
            # when the replacement attempt fails: its report_json is still
            # intact and the operator can retry after fixing the cause.
            if not terminal_own_success and (
                current.status != RunStatus.completed or not successful
            ):
                previous = claim.previous
                values.update(
                    status=previous.status,
                    progress_current=previous.progress_current,
                    progress_total=previous.progress_total,
                    progress_percent=previous.progress_percent,
                    stage_key=previous.stage_key,
                    stage_label=previous.stage_label,
                    stage_detail=previous.stage_detail,
                    eta_seconds=previous.eta_seconds,
                    error_message=previous.error_message,
                    checkpointed_at=previous.checkpointed_at,
                    finished_at=previous.finished_at,
                    analysis_markdown=previous.analysis_markdown,
                    report_json=previous.report_json,
                )
            ownership_conditions = (
                (
                    Run.execution_slot == EXECUTION_SLOT,
                    Run.lease_owner == claim.owner,
                )
                if owned
                else (
                    Run.status == current.status,
                    Run.execution_slot.is_(None),
                    Run.lease_owner.is_(None),
                )
            )
            released = await connection.execute(
                update(Run)
                .where(
                    Run.id == claim.run_id,
                    Run.attempt_count == claim.attempt_count,
                    Run.resume_count == claim.resume_count,
                    *ownership_conditions,
                )
                .values(**values)
            )
            await connection.commit()
            return released.rowcount == 1
        except BaseException:
            await connection.rollback()
            raise


async def reprocess_saved_run(
    value: str,
    *,
    announce: Callable[[str], None] = print,
) -> str:
    run_id = canonical_run_id(value)
    claim = await _claim_eligible_run(run_id)
    announce(
        "Повторный опрос модельной панели запрещён. "
        f"Найдено сохранённых исходных ответов: {claim.completed_answers}. "
        "Аналитические и отчётные LLM-слои могут расходовать токены."
    )
    announce(f"Контрольный SHA-256 raw-корпуса: {claim.raw_answers_sha256}.")
    announce(
        f"Запускаем переанализ {run_id}; исходный статус — "
        f"{claim.previous.status.value}."
    )

    successful = False
    try:
        # This is deliberately the only pipeline entrypoint used by the
        # operator CLI. It rebuilds annotations, metrics and the report from
        # persisted answers while holding the shared durable execution slot.
        await _execute_claimed_reprocess(claim)

        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one_or_none()
            if run is None:
                raise ReprocessExecutionError(
                    "Проверка исчезла во время переанализа."
                )
            if run.status != RunStatus.completed:
                detail = (run.error_message or "").strip()
                message = (
                    f"Переанализ завершился со статусом {run.status.value}."
                    + (f" {detail}" if detail else "")
                )
                raise ReprocessExecutionError(message)
        await _assert_raw_answers_unchanged(claim)
        successful = True
        announce(
            "Raw-корпус не изменился: "
            f"{claim.total_answers} ответов, "
            f"SHA-256 {claim.raw_answers_sha256}."
        )
        return run_id
    finally:
        try:
            if not successful:
                await _assert_raw_answers_unchanged(claim)
        finally:
            await _release_reprocess_claim(
                claim,
                successful=successful,
            )


def _install_cancellation_handlers(task: "asyncio.Task[str]") -> None:
    """Turn SIGTERM/SIGINT into cancellation of the reprocess task.

    Without this the interpreter dies exactly where the signal lands, so the
    ``finally`` block in :func:`reprocess_saved_run` never runs: the claim is
    never released and the run stays ``analyzing`` until the coordinator
    recovers the expired lease.  Cancelling instead lets the normal rollback
    path restore ``PreviousRunState`` before we exit.
    """

    loop = asyncio.get_running_loop()
    already_requested = False

    def _request_stop(signame: str) -> None:
        nonlocal already_requested
        if already_requested:
            print(
                f"Повторный {signame}: остановка уже идёт, ждём откат состояния.",
                file=sys.stderr,
            )
            return
        already_requested = True
        print(
            f"Получен {signame}: отменяем переанализ и откатываем состояние прогона.",
            file=sys.stderr,
        )
        task.cancel()

    for signame in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, signame)
        try:
            loop.add_signal_handler(signum, _request_stop, signame)
        except NotImplementedError:  # pragma: no cover — не Unix
            signal.signal(signum, lambda *_, name=signame: _request_stop(name))


async def _reprocess_with_signal_handling(value: str) -> str:
    """Run the reprocess as a cancellable task so cleanup always happens."""

    task = asyncio.create_task(reprocess_saved_run(value))
    _install_cancellation_handlers(task)
    return await task


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Пересобрать аналитику и отчёт из сохранённых ответов "
            "без повторного опроса модельной панели."
        )
    )
    parser.add_argument("run_id", help="UUID сохранённой проверки")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        run_id = asyncio.run(_reprocess_with_signal_handling(args.run_id))
    except ReprocessGuardError as exc:
        print(f"Отказ: {exc}", file=sys.stderr)
        return 2
    except ReprocessExecutionError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except asyncio.CancelledError:
        # Остановка оператором или systemd. Claim уже освобождён в finally
        # внутри reprocess_saved_run, поэтому прогон не остаётся подвешенным.
        print(
            "Переанализ остановлен по сигналу; состояние прогона откатано.",
            file=sys.stderr,
        )
        return 143
    print(f"Переанализ {run_id} завершён.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
