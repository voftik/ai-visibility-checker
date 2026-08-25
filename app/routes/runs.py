from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.db import get_session
from app.models import Run, RunStatus
from app.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    RetryRunResponse,
    RunDetail,
    RunLookupRequest,
    RunSummary,
    ShareTokenResponse,
)
from app.services.crawler import AUDIT_USER_AGENTS, normalize_domain
from app.services.event_bus import bus
from app.services.run_coordinator import (
    ACTIVE_STATUSES,
    coordinator,
    pending_run_count,
    queue_positions,
)

router = APIRouter(prefix="/api/runs", tags=["runs"])

def _share_url(token: str) -> str:
    return f"/r/{token}"


def _new_share_token() -> str:
    return secrets.token_urlsafe(24)


def _summary_with_queue(
    run: Run,
    positions: dict[str, int],
) -> RunSummary:
    return RunSummary.model_validate(run).model_copy(
        update={
            "queue_position": positions.get(run.id),
            "queue_total": (
                len(positions)
                if run.id in positions
                else None
            ),
        }
    )


def _detail_with_queue(
    run: Run,
    positions: dict[str, int],
) -> RunDetail:
    return RunDetail.model_validate(run).model_copy(
        update={
            "queue_position": positions.get(run.id),
            "queue_total": (
                len(positions)
                if run.id in positions
                else None
            ),
        }
    )


def _public_event(event: dict) -> dict | None:
    """Keep operational internals out of the browser event stream."""

    event_type = event.get("type")
    if event_type == "stage":
        return {
            key: event.get(key)
            for key in (
                "type",
                "stage",
                "label",
                "detail",
                "percent",
                "eta_seconds",
                "queue_position",
                "queue_total",
                "run_state",
                "state_revision",
                "ts",
            )
            if event.get(key) is not None
        }
    if event_type == "progress":
        return {
            key: event.get(key)
            for key in (
                "type",
                "stage",
                "label",
                "percent",
                "detail",
                "eta_seconds",
                "queue_position",
                "queue_total",
                "run_state",
                "state_revision",
                "ts",
            )
            if event.get(key) is not None
        }
    if event_type == "final":
        return {
            key: event.get(key)
            for key in (
                "type",
                "status",
                "message",
                "run_state",
                "state_revision",
                "ts",
            )
            if event.get(key) is not None
        }
    return None


def _database_snapshot_event(
    run: Run,
    positions: dict[str, int],
) -> dict:
    """Build an authoritative first SSE event from durable state."""

    if run.status in (RunStatus.completed, RunStatus.failed):
        return {
            "type": "final",
            "status": run.status.value,
            "run_state": run.run_state,
            "message": run.error_message if run.status == RunStatus.failed else None,
            "state_revision": int(run.state_revision or 0),
            "ts": (
                run.state_changed_at.isoformat()
                if run.state_changed_at is not None
                else datetime.now(timezone.utc).isoformat()
            ),
        }
    return {
        "type": "stage",
        "stage": run.stage_key or "queued",
        "label": run.stage_label or "В очереди",
        "detail": run.stage_detail or "Проверка сохранена и запустится автоматически.",
        "percent": int(run.progress_percent or 0),
        "eta_seconds": run.eta_seconds,
        "queue_position": positions.get(run.id),
        "queue_total": len(positions) if run.id in positions else None,
        "run_state": run.run_state,
        "state_revision": int(run.state_revision or 0),
        "ts": (
            run.state_changed_at.isoformat()
            if run.state_changed_at is not None
            else datetime.now(timezone.utc).isoformat()
        ),
    }


def _last_sequence_for_epoch(
    header_value: str | None,
    *,
    stream_epoch: int,
) -> int | None:
    """Accept replay cursors only from the same durable retry epoch."""

    if not header_value or ":" not in header_value:
        # Numeric IDs came from the pre-queue release and cannot distinguish a
        # retry or a process restart. Replaying is safer than skipping events.
        return None
    epoch_text, sequence_text = header_value.split(":", 1)
    if not epoch_text.isdigit() or not sequence_text.isdigit():
        return None
    if int(epoch_text) != stream_epoch:
        return None
    return int(sequence_text)


def _sse_event_id(stream_epoch: int, sequence: int) -> str:
    return f"{stream_epoch}:{sequence}"


@router.post("", response_model=CreateRunResponse)
async def create_run(
    payload: CreateRunRequest,
    session: AsyncSession = Depends(get_session),
) -> CreateRunResponse:
    domain = normalize_domain(payload.domain)
    if not domain:
        raise HTTPException(
            status_code=400,
            detail="Не удалось распознать домен. Проверьте адрес и попробуйте ещё раз.",
        )

    await session.execute(text("BEGIN IMMEDIATE"))
    existing = (
        await session.execute(
            select(Run)
            .where(
                Run.domain == domain,
                Run.status.in_(ACTIVE_STATUSES),
            )
            .order_by(Run.created_at.asc(), Run.id.asc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        await session.commit()
        coordinator.wake()
        return CreateRunResponse(run_id=existing.id)

    queue_limit = max(1, int(settings.RUN_QUEUE_MAX_PENDING))
    if await pending_run_count(session) >= queue_limit:
        await session.rollback()
        raise HTTPException(
            status_code=429,
            detail=(
                "Сейчас в очереди слишком много проверок. "
                "Попробуйте снова немного позже."
            ),
            headers={"Retry-After": "60"},
        )

    changed_at = datetime.now(timezone.utc)
    config = {
        "domains": [domain],
        "user_agents": list(AUDIT_USER_AGENTS),
        "concurrency": settings.DEFAULT_CONCURRENCY,
        "timeout_seconds": settings.DEFAULT_TIMEOUT_SECONDS,
        "page_limit": settings.AUDIT_PAGE_LIMIT,
        "pipeline_version": "aiv-2026-07",
    }
    run = Run(
        status=RunStatus.pending,
        domain=domain,
        config_json=config,
        progress_current=0,
        progress_total=100,
        progress_percent=0,
        stage_key="queued",
        stage_label="В очереди",
        stage_detail=(
            "Проверка сохранена и запустится автоматически."
        ),
        eta_seconds=None,
        state_changed_at=changed_at,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    coordinator.wake()
    return CreateRunResponse(run_id=run.id)


@router.get("", response_model=list[RunSummary])
async def list_runs(
    session: AsyncSession = Depends(get_session),
) -> list[RunSummary]:
    """Return the public history of every check, newest first."""

    result = await session.execute(select(Run).order_by(Run.created_at.desc()))
    positions = await queue_positions(session)
    return [
        _summary_with_queue(run, positions)
        for run in result.scalars().all()
    ]


@router.post("/lookup", response_model=list[RunSummary])
async def lookup_runs(
    payload: RunLookupRequest,
    session: AsyncSession = Depends(get_session),
) -> list[RunSummary]:
    """Return only the unguessable run IDs supplied by this browser."""

    if not payload.ids:
        return []
    result = await session.execute(select(Run).where(Run.id.in_(payload.ids)))
    by_id = {run.id: run for run in result.scalars().all()}
    positions = await queue_positions(session)
    return [
        _summary_with_queue(by_id[run_id], positions)
        for run_id in payload.ids
        if run_id in by_id
    ]


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)) -> RunDetail:
    result = await session.execute(
        select(Run)
        .where(Run.id == run_id)
        .options(selectinload(Run.illustrations))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Проверка не найдена.")
    positions = await queue_positions(session)
    return _detail_with_queue(run, positions)


@router.post("/{run_id}/retry", response_model=RetryRunResponse)
async def retry_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
) -> RetryRunResponse:
    await session.execute(text("BEGIN IMMEDIATE"))
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        await session.rollback()
        raise HTTPException(status_code=404, detail="Проверка не найдена.")
    if run.status == RunStatus.completed:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail="Готовую проверку нельзя перезапустить.",
        )
    if run.status in ACTIVE_STATUSES:
        await session.commit()
        coordinator.wake()
        return RetryRunResponse(ok=True, run_id=run_id)
    queue_limit = max(1, int(settings.RUN_QUEUE_MAX_PENDING))
    if await pending_run_count(session) >= queue_limit:
        await session.rollback()
        raise HTTPException(
            status_code=429,
            detail=(
                "Сейчас в очереди слишком много проверок. "
                "Попробуйте снова немного позже."
            ),
            headers={"Retry-After": "60"},
        )
    resumed_at = datetime.now(timezone.utc)
    resume_percent = min(95, max(0, int(run.progress_percent or 0)))
    claimed = await session.execute(
        update(Run)
        .where(
            Run.id == run_id,
            Run.status == RunStatus.failed,
        )
        .values(
            status=RunStatus.pending,
            error_message=None,
            stage_key="recovering",
            stage_label="Восстанавливаем проверку",
            stage_detail=(
                "Используем уже сохранённые результаты и завершаем "
                "недостающие этапы."
            ),
            progress_current=resume_percent,
            progress_percent=resume_percent,
            eta_seconds=None,
            execution_slot=None,
            lease_owner=None,
            lease_expires_at=None,
            heartbeat_at=None,
            finished_at=None,
            resume_count=Run.resume_count + 1,
            resume_reason="manual_retry",
            last_resumed_at=resumed_at,
            state_revision=Run.state_revision + 1,
            state_changed_at=resumed_at,
        )
    )
    await session.commit()
    if claimed.rowcount != 1:
        return RetryRunResponse(ok=True, run_id=run_id)
    bus.reset(run_id)
    coordinator.wake()
    return RetryRunResponse(ok=True, run_id=run_id)


@router.post("/{run_id}/share", response_model=ShareTokenResponse)
async def share_run(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> ShareTokenResponse:
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Проверка не найдена.")
    if not run.share_token:
        run.share_token = _new_share_token()
        await session.commit()
        await session.refresh(run)
    return ShareTokenResponse(share_token=run.share_token, share_url=_share_url(run.share_token))


@router.get("/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Проверка не найдена.")

    positions = await queue_positions(session)
    initial_event = _database_snapshot_event(run, positions)
    already_finished = run.status in (RunStatus.completed, RunStatus.failed)
    stream_epoch = int(run.resume_count or 0)
    last_event_id_header = request.headers.get("last-event-id")
    last_sequence = _last_sequence_for_epoch(
        last_event_id_header,
        stream_epoch=stream_epoch,
    )
    history = bus.channel_history(run_id)
    if (
        last_sequence is not None
        and (
            not history
            or last_sequence > history[-1][0]
        )
    ):
        last_sequence = None
    # Streaming responses can stay open for many minutes. End the snapshot
    # read transaction now so it cannot pin SQLite's WAL while the generator
    # waits for events.
    await session.rollback()

    async def gen() -> AsyncIterator[bytes]:
        snapshot_payload = json.dumps(
            initial_event,
            ensure_ascii=False,
        )
        # The snapshot intentionally has no SSE id: it must not overwrite the
        # replay cursor supplied by EventSource before buffered events follow.
        yield f"data: {snapshot_payload}\n\n".encode()
        if already_finished:
            return

        keepalive = 15.0
        subscriber = bus.subscribe(
            run_id,
            last_event_id=last_sequence,
        ).__aiter__()
        while True:
            try:
                seq, raw_event = await asyncio.wait_for(
                    subscriber.__anext__(),
                    timeout=keepalive,
                )
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
                continue
            except StopAsyncIteration:
                return
            event = _public_event(raw_event)
            if event is None:
                continue
            yield (
                f"id: {_sse_event_id(stream_epoch, seq)}\n"
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            ).encode()
            if event.get("type") == "final":
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
