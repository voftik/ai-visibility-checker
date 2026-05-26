from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_session
from app.models import Run
from app.schemas import RunDetail

router = APIRouter(prefix="/api/shared", tags=["shared"])


@router.get("/{token}", response_model=RunDetail)
async def get_shared_run(
    token: str, session: AsyncSession = Depends(get_session)
) -> RunDetail:
    # Empty / falsy tokens are rejected up-front so we never match a row that
    # has share_token = NULL via a NULL == NULL trick.
    if not token:
        raise HTTPException(status_code=404, detail="run not found")
    result = await session.execute(
        select(Run)
        .where(Run.share_token == token)
        .options(selectinload(Run.probes), selectinload(Run.robots_rules))
    )
    run = result.scalar_one_or_none()
    if run is None or not run.share_token:
        raise HTTPException(status_code=404, detail="run not found")
    return RunDetail.model_validate(run)
