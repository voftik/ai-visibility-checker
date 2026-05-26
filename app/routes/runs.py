from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.db import get_session
from app.models import Run, RunStatus
from app.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    RunDetail,
    RunSummary,
    ShareRevokeResponse,
    ShareTokenResponse,
)
from app.services.crawler import USER_AGENT_STRINGS, normalize_domains, run_crawl
from app.services.event_bus import bus


def _share_url(token: str) -> str:
    return f"/r/{token}"


def _new_share_token() -> str:
    # token_urlsafe(24) yields ~32 url-safe chars (no padding) — well within
    # the 64-char column budget and impractical to brute-force.
    return secrets.token_urlsafe(24)

router = APIRouter(prefix="/api/runs", tags=["runs"])

# Hold strong refs to background tasks so the GC doesn't kill them mid-flight.
_background_tasks: set[asyncio.Task] = set()
# Map run_id -> task so DELETE can cancel an in-flight crawl/analysis cleanly
# and avoid foreign-key explosions when the cascade removes the parent row
# while workers are still writing children.
_run_tasks: dict[str, asyncio.Task] = {}


def _dedupe_user_agents(labels: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label == "robots-fetcher" or label not in USER_AGENT_STRINGS or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _bounded_int(value: int | None, default: int, *, low: int, high: int, name: str) -> int:
    n = default if value is None else int(value)
    if n < low or n > high:
        raise HTTPException(status_code=400, detail=f"{name} must be between {low} and {high}")
    return n


@router.post("", response_model=CreateRunResponse)
async def create_run(
    payload: CreateRunRequest,
    session: AsyncSession = Depends(get_session),
) -> CreateRunResponse:
    domains = normalize_domains(payload.domains)
    ua_labels = _dedupe_user_agents(payload.user_agents)
    if not domains:
        raise HTTPException(status_code=400, detail="at least one valid domain is required")
    if not ua_labels:
        raise HTTPException(status_code=400, detail="at least one known user-agent is required")
    concurrency = _bounded_int(
        payload.concurrency,
        settings.DEFAULT_CONCURRENCY,
        low=1,
        high=32,
        name="concurrency",
    )
    timeout_seconds = _bounded_int(
        payload.timeout_seconds,
        settings.DEFAULT_TIMEOUT_SECONDS,
        low=1,
        high=120,
        name="timeout_seconds",
    )

    config: dict = {
        "domains": domains,
        "user_agents": ua_labels,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
    }
    if payload.source_breakdown:
        config["source_breakdown"] = payload.source_breakdown
    run = Run(
        status=RunStatus.pending,
        config_json=config,
        progress_total=len(domains) * (len(ua_labels) + 1),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    task = asyncio.create_task(run_crawl(run.id))
    _background_tasks.add(task)
    _run_tasks[run.id] = task

    def _cleanup(t: asyncio.Task, rid: str = run.id) -> None:
        _background_tasks.discard(t)
        if _run_tasks.get(rid) is t:
            _run_tasks.pop(rid, None)

    task.add_done_callback(_cleanup)
    return CreateRunResponse(run_id=run.id)


@router.get("", response_model=list[RunSummary])
async def list_runs(session: AsyncSession = Depends(get_session)) -> list[RunSummary]:
    result = await session.execute(select(Run).order_by(Run.created_at.desc()).limit(50))
    runs = result.scalars().all()
    return [RunSummary.model_validate(r) for r in runs]


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)) -> RunDetail:
    result = await session.execute(
        select(Run)
        .where(Run.id == run_id)
        .options(selectinload(Run.probes), selectinload(Run.robots_rules))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunDetail.model_validate(run)


@router.delete("/{run_id}")
async def delete_run(run_id: str, session: AsyncSession = Depends(get_session)) -> dict[str, bool]:
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    # Cancel an in-flight background crawl/analyze task BEFORE the cascade
    # delete fires, so workers don't hit FK violations writing children for
    # an already-removed parent row.
    task = _run_tasks.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    # Drain the SSE channel: tell any subscribers the run ended.
    bus.publish(run_id, {"type": "phase_change", "phase": "failed"})
    bus.publish(run_id, {"type": "final", "status": "failed"})

    await session.execute(delete(Run).where(Run.id == run_id))
    await session.commit()
    return {"ok": True}


@router.post("/{run_id}/share", response_model=ShareTokenResponse)
async def share_run(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> ShareTokenResponse:
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if not run.share_token:
        run.share_token = _new_share_token()
        await session.commit()
        await session.refresh(run)
    return ShareTokenResponse(share_token=run.share_token, share_url=_share_url(run.share_token))


@router.post("/{run_id}/share/revoke", response_model=ShareRevokeResponse)
async def revoke_share(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> ShareRevokeResponse:
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    run.share_token = None
    await session.commit()
    return ShareRevokeResponse(ok=True)


@router.post("/{run_id}/share/regenerate", response_model=ShareTokenResponse)
async def regenerate_share(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> ShareTokenResponse:
    result = await session.execute(select(Run).where(Run.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
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
        raise HTTPException(status_code=404, detail="run not found")

    already_finished = run.status in (RunStatus.completed, RunStatus.failed)
    final_status = run.status.value if already_finished else None

    # Browser EventSource auto-sends Last-Event-ID on every reconnect. We use
    # that to replay events the client missed during a network blip.
    last_event_id_header = request.headers.get("last-event-id")
    last_id_int: int | None = None
    if last_event_id_header and last_event_id_header.isdigit():
        last_id_int = int(last_event_id_header)

    async def gen() -> AsyncIterator[bytes]:
        # If the run already finished, prefer replaying the in-memory channel
        # history (so the client gets the full timeline including the final
        # event). If that history was already evicted by the cleanup loop,
        # fall back to a synthetic final event so the UI still closes out.
        if already_finished:
            history = bus.channel_history(run_id)
            if history:
                for seq, ev in history:
                    if last_id_int is not None and seq <= last_id_int:
                        continue
                    yield f"id: {seq}\ndata: {json.dumps(ev)}\n\n".encode()
                # If the last history event isn't a `final`, append one so the
                # client always sees a terminal marker.
                if history[-1][1].get("type") != "final":
                    payload = json.dumps({"type": "final", "status": final_status})
                    yield f"data: {payload}\n\n".encode()
                return
            payload = json.dumps({"type": "final", "status": final_status})
            yield f"data: {payload}\n\n".encode()
            return

        keepalive = 15.0
        agen = bus.subscribe(run_id, last_event_id=last_id_int).__aiter__()
        while True:
            try:
                seq, event = await asyncio.wait_for(agen.__anext__(), timeout=keepalive)
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
                continue
            except StopAsyncIteration:
                return
            yield f"id: {seq}\ndata: {json.dumps(event)}\n\n".encode()
            if event.get("type") == "final":
                return

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
