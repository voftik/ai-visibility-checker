"""Execution-lease context shared by one durable run worker.

The coordinator owns the database lease.  A context variable propagates that
owner into crawler/analyzer child tasks, allowing every critical state write
to prove that it still belongs to the current execution attempt.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Run, RunStatus


class RunLeaseLostError(RuntimeError):
    """Raised when a stale worker tries to continue after losing its lease."""


@dataclass(frozen=True)
class BoundRunLease:
    run_id: str
    owner: str


_CURRENT_RUN_LEASE: ContextVar[BoundRunLease | None] = ContextVar(
    "aiv_current_run_lease",
    default=None,
)


def current_run_lease() -> BoundRunLease | None:
    return _CURRENT_RUN_LEASE.get()


def lease_owner_for(run_id: str) -> str | None:
    lease = current_run_lease()
    if lease is None:
        return None
    if lease.run_id != run_id:
        raise RunLeaseLostError(
            f"Worker for run {lease.run_id} cannot write run {run_id}"
        )
    return lease.owner


async def assert_run_lease(run_id: str) -> None:
    """Stop a bound stale worker before it persists another checkpoint."""

    owner = lease_owner_for(run_id)
    if owner is None:
        return
    async with SessionLocal() as session:
        owned = (
            await session.execute(
                select(Run.id).where(
                    Run.id == run_id,
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
        ).scalar_one_or_none()
    if owned is None:
        raise RunLeaseLostError(f"Run lease lost for {run_id}")


@contextmanager
def bind_run_lease(run_id: str, owner: str) -> Iterator[None]:
    token: Token[BoundRunLease | None] = _CURRENT_RUN_LEASE.set(
        BoundRunLease(run_id=run_id, owner=owner)
    )
    try:
        yield
    finally:
        _CURRENT_RUN_LEASE.reset(token)
