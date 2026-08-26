from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Run, RunStatus
from app.schemas import RunDetail, build_public_run_detail
from app.services.publication_contract import (
    PublicationContractError,
    ensure_publication_contract,
    has_visible_publication_snapshot,
)
from app.services.run_coordinator import queue_positions

router = APIRouter(prefix="/api/shared", tags=["shared"])


@router.get("/{token}", response_model=RunDetail)
async def get_shared_run(
    token: str, session: AsyncSession = Depends(get_session)
) -> RunDetail:
    # Empty / falsy tokens are rejected up-front so we never match a row that
    # has share_token = NULL via a NULL == NULL trick.
    if not token:
        raise HTTPException(status_code=404, detail="run not found")
    result = await session.execute(select(Run).where(Run.share_token == token))
    run = result.scalar_one_or_none()
    if run is None or not run.share_token:
        raise HTTPException(status_code=404, detail="run not found")
    if not has_visible_publication_snapshot(run):
        raise HTTPException(
            status_code=409,
            detail="Отчёт по этой ссылке ещё не опубликован.",
        )
    try:
        await ensure_publication_contract(session, run)
    except PublicationContractError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Готовый отчёт не прошёл проверку целостности. "
                "Публичный снимок временно недоступен."
            ),
        ) from exc
    positions = await queue_positions(session)
    return build_public_run_detail(
        run,
        queue_position=positions.get(run.id),
        queue_total=len(positions) if run.id in positions else None,
    )
