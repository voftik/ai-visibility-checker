from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, text

from app.db import DB_PATH, SessionLocal, init_db
from app.main import app
from app.models import Run, RunStatus
from app.routes.runs import (
    _database_snapshot_event,
    _last_sequence_for_epoch,
)
from app.services import crawler, run_coordinator
from app.services.event_bus import bus
from app.services.progress import complete_run, fail_run, update_progress
from app.services.run_coordinator import (
    SAVED_ANSWERS_ONLY_MARKER_KEY,
    SAVED_ANSWERS_ONLY_MARKER_VERSION,
    SAVED_ANSWERS_ONLY_MODE,
    RunClaim,
    RunCoordinator,
    _claim_next_run,
    _release_claim,
    pending_run_count,
    queue_positions,
    recover_expired_leases,
)
from app.services.run_lease import RunLeaseLostError, bind_run_lease


class DurableQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        for run_id in self.run_ids:
            bus.reset(run_id)
        if self.run_ids:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id.in_(self.run_ids)))
                await session.commit()

    async def _add_run(
        self,
        *,
        status: RunStatus = RunStatus.pending,
        created_at: datetime | None = None,
        **values,
    ) -> str:
        run_id = f"test-queue-{uuid.uuid4()}"
        self.run_ids.append(run_id)
        config_json = values.pop("config_json", {})
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain=f"{uuid.uuid4().hex}.example.com",
                    status=status,
                    config_json=config_json,
                    created_at=created_at or datetime.now(timezone.utc),
                    **values,
                )
            )
            await session.commit()
        return run_id

    async def _mark_saved_answers_only(
        self,
        run_id: str,
        *,
        previous_status: RunStatus = RunStatus.completed,
        previous_config: dict | None = None,
        owner: str = "dead-operator",
    ) -> None:
        original_config = previous_config or {"page_limit": 6}
        marker = {
            "version": SAVED_ANSWERS_ONLY_MARKER_VERSION,
            "mode": SAVED_ANSWERS_ONLY_MODE,
            "run_id": run_id,
            "owner": owner,
            "attempt_count": 1,
            "raw_answers_sha256": "a" * 64,
            "previous_config_json": original_config,
            "previous_terminal_state": {
                "status": previous_status.value,
                "progress_current": 100,
                "progress_total": 100,
                "progress_percent": 100,
                "stage_key": "report",
                "stage_label": "Собираем отчёт и иллюстрации",
                "stage_detail": "Отчёт готов.",
                "eta_seconds": 0,
                "error_message": None,
                "checkpointed_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            run.config_json = {
                **original_config,
                SAVED_ANSWERS_ONLY_MARKER_KEY: marker,
            }
            await session.commit()

    async def test_concurrent_claims_reserve_only_one_global_slot(self) -> None:
        first_id = await self._add_run(
            created_at=datetime.now(timezone.utc) - timedelta(seconds=2)
        )
        second_id = await self._add_run()

        claims = await asyncio.gather(
            _claim_next_run("owner-a", lease_seconds=90),
            _claim_next_run("owner-b", lease_seconds=90),
        )
        claimed = [claim for claim in claims if claim is not None]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].run_id, first_id)

        async with SessionLocal() as session:
            rows = (
                await session.execute(
                    select(Run.id, Run.execution_slot, Run.lease_owner)
                    .where(Run.id.in_([first_id, second_id]))
                    .order_by(Run.created_at)
                )
            ).all()
        self.assertEqual(rows[0][1], 1)
        self.assertIsNotNone(rows[0][2])
        self.assertIsNone(rows[1][1])

    async def test_queue_positions_are_fifo_and_exclude_active_slot(self) -> None:
        first_id = await self._add_run(
            created_at=datetime.now(timezone.utc) - timedelta(seconds=2)
        )
        second_id = await self._add_run()
        claim = await _claim_next_run("owner-fifo", lease_seconds=90)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.run_id, first_id)

        async with SessionLocal() as session:
            positions = await queue_positions(session)
        self.assertNotIn(first_id, positions)
        self.assertEqual(positions[second_id], 1)

    async def test_expired_lease_requeues_without_losing_checkpoint(self) -> None:
        run_id = await self._add_run(
            status=RunStatus.analyzing,
            execution_slot=1,
            lease_owner="dead-owner",
            lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            progress_current=64,
            progress_total=100,
            progress_percent=64,
            state_revision=7,
            resume_count=2,
        )
        bus.publish(run_id, {"type": "final", "status": "failed"})

        recovered = await recover_expired_leases()
        self.assertEqual(recovered, 1)

        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.pending)
        self.assertIsNone(run.execution_slot)
        self.assertEqual(run.stage_key, "recovering")
        self.assertEqual(run.progress_percent, 64)
        self.assertEqual(run.state_revision, 8)
        self.assertEqual(run.resume_count, 3)
        history = bus.channel_history(run_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][1]["percent"], 64)
        self.assertEqual(history[0][1]["state_revision"], 8)

    async def test_active_run_without_execution_slot_is_requeued(self) -> None:
        for status in (RunStatus.crawling, RunStatus.analyzing):
            with self.subTest(status=status.value):
                run_id = await self._add_run(
                    status=status,
                    execution_slot=None,
                    lease_owner=None,
                    lease_expires_at=None,
                    progress_current=37,
                    progress_total=100,
                    progress_percent=37,
                    state_revision=4,
                    resume_count=1,
                )
                recovered = await recover_expired_leases()
                self.assertEqual(recovered, 1)
                async with SessionLocal() as session:
                    run = await session.get(Run, run_id)
                self.assertEqual(run.status, RunStatus.pending)
                self.assertIsNone(run.execution_slot)
                self.assertEqual(run.stage_key, "recovering")
                self.assertEqual(run.progress_percent, 37)
                self.assertEqual(run.state_revision, 5)
                self.assertEqual(run.resume_count, 2)
                self.assertIsNotNone(
                    await _claim_next_run(
                        f"recovered-{status.value}",
                        lease_seconds=90,
                    )
                )
                async with SessionLocal() as session:
                    claimed = await session.get(Run, run_id)
                    claimed.status = RunStatus.completed
                    claimed.execution_slot = None
                    claimed.lease_owner = None
                    claimed.lease_expires_at = None
                    await session.commit()

    async def test_unexpired_lease_is_not_recovered(self) -> None:
        run_id = await self._add_run(
            status=RunStatus.crawling,
            execution_slot=1,
            lease_owner="live-owner",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
        recovered = await recover_expired_leases()
        self.assertEqual(recovered, 0)
        async with SessionLocal() as session:
            slot = (
                await session.execute(
                    select(Run.execution_slot).where(Run.id == run_id)
                )
            ).scalar_one()
        self.assertEqual(slot, 1)

    async def test_expired_saved_only_lease_restores_terminal_state(self) -> None:
        report_json = {"narrative": {"headline": "Сохранённый отчёт"}}
        run_id = await self._add_run(
            status=RunStatus.analyzing,
            execution_slot=1,
            lease_owner="dead-operator",
            lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=30),
            progress_current=70,
            progress_total=100,
            progress_percent=70,
            state_revision=9,
            attempt_count=1,
            report_json=report_json,
        )
        await self._mark_saved_answers_only(run_id)

        recovered = await recover_expired_leases()
        self.assertEqual(recovered, 1)
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertIsNone(run.execution_slot)
        self.assertEqual(run.progress_percent, 100)
        self.assertEqual(run.config_json, {"page_limit": 6})
        self.assertEqual(run.report_json, report_json)
        self.assertIsNone(await _claim_next_run("generic-owner", lease_seconds=90))

    async def test_saved_only_recovery_preserves_lease_renewed_after_read(
        self,
    ) -> None:
        owner = "live-saved-only-owner"
        observed_at = datetime.now(timezone.utc)
        renewed_until = observed_at + timedelta(minutes=5)
        run_id = await self._add_run(
            status=RunStatus.analyzing,
            execution_slot=1,
            lease_owner=owner,
            lease_expires_at=observed_at - timedelta(seconds=1),
            heartbeat_at=observed_at - timedelta(seconds=30),
            progress_current=70,
            progress_total=100,
            progress_percent=70,
            state_revision=9,
            attempt_count=1,
        )
        await self._mark_saved_answers_only(run_id, owner=owner)

        original_restore = run_coordinator._saved_only_terminal_restore_values
        renewed = False

        def renew_between_read_and_recovery(run, *, now):
            nonlocal renewed
            if not renewed:
                renewed = True
                with sqlite3.connect(DB_PATH, timeout=2.0) as connection:
                    connection.execute(
                        "UPDATE runs SET heartbeat_at = ?, lease_expires_at = ? "
                        "WHERE id = ? AND lease_owner = ?",
                        (
                            observed_at.astimezone(timezone.utc)
                            .replace(tzinfo=None)
                            .isoformat(sep=" ", timespec="microseconds"),
                            renewed_until.astimezone(timezone.utc)
                            .replace(tzinfo=None)
                            .isoformat(sep=" ", timespec="microseconds"),
                            run_id,
                            owner,
                        ),
                    )
            return original_restore(run, now=now)

        with patch.object(
            run_coordinator,
            "_saved_only_terminal_restore_values",
            side_effect=renew_between_read_and_recovery,
        ):
            recovered = await recover_expired_leases(now=observed_at)

        self.assertEqual(recovered, 0)
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.analyzing)
        self.assertEqual(run.execution_slot, 1)
        self.assertEqual(run.lease_owner, owner)
        self.assertIn(SAVED_ANSWERS_ONLY_MARKER_KEY, run.config_json)
        self.assertGreater(
            run.lease_expires_at.replace(tzinfo=timezone.utc),
            observed_at,
        )

    async def test_generic_queue_skips_any_saved_only_marker(self) -> None:
        marked_id = await self._add_run(
            config_json={SAVED_ANSWERS_ONLY_MARKER_KEY: None},
            created_at=datetime.now(timezone.utc) - timedelta(seconds=2),
        )
        normal_id = await self._add_run()

        async with SessionLocal() as session:
            positions = await queue_positions(session)
            count = await pending_run_count(session)
        self.assertNotIn(marked_id, positions)
        self.assertEqual(positions, {normal_id: 1})
        self.assertEqual(count, 1)
        claim = await _claim_next_run("generic-owner", lease_seconds=90)
        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(claim.run_id, normal_id)

    async def test_execute_fail_safe_never_calls_crawl_for_marker(self) -> None:
        run_id = await self._add_run(
            status=RunStatus.pending,
            execution_slot=1,
            lease_owner="generic-owner",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            attempt_count=1,
        )
        await self._mark_saved_answers_only(run_id)
        local_coordinator = RunCoordinator(
            lease_seconds=90,
            poll_seconds=0.01,
        )
        with patch(
            "app.services.run_coordinator.run_crawl",
            new_callable=AsyncMock,
        ) as run_crawl:
            await local_coordinator._execute(
                RunClaim(run_id=run_id, owner="generic-owner")
            )
        run_crawl.assert_not_awaited()
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertIsNone(run.execution_slot)

    async def test_execute_heartbeat_survives_cpu_block_and_terminal_clear(
        self,
    ) -> None:
        owner = "threaded-heartbeat-owner"
        run_id = await self._add_run(
            status=RunStatus.analyzing,
            execution_slot=1,
            lease_owner=owner,
            heartbeat_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            attempt_count=1,
            resume_count=0,
        )
        observed: list[datetime] = []

        async def blocking_worker(value: str, *, lease_owner: str) -> None:
            self.assertEqual(value, run_id)
            self.assertEqual(lease_owner, owner)
            time.sleep(0.55)  # noqa: ASYNC251 - deliberately starve event loop
            async with SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                self.assertIsNotNone(run.heartbeat_at)
                assert run.heartbeat_at is not None
                observed.append(
                    run.heartbeat_at.replace(
                        tzinfo=run.heartbeat_at.tzinfo or timezone.utc
                    )
                )
                run.status = RunStatus.completed
                run.execution_slot = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                await session.commit()
            await asyncio.sleep(0.2)

        local_coordinator = RunCoordinator(
            lease_seconds=15,
            poll_seconds=0.01,
        )
        local_coordinator.lease_seconds = 0.3
        with patch(
            "app.services.run_coordinator.run_crawl",
            new=blocking_worker,
        ):
            await local_coordinator._execute(
                RunClaim(
                    run_id=run_id,
                    owner=owner,
                    attempt_count=1,
                    resume_count=0,
                )
            )

        self.assertEqual(len(observed), 1)
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertIsNone(run.execution_slot)
        self.assertIsNone(run.lease_owner)

    async def test_terminal_lookup_failure_cancels_worker_before_release(
        self,
    ) -> None:
        owner = "terminal-lookup-owner"
        run_id = await self._add_run(
            status=RunStatus.analyzing,
            execution_slot=1,
            lease_owner=owner,
            heartbeat_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            attempt_count=1,
            resume_count=0,
        )
        cancelled = asyncio.Event()

        async def terminal_then_wait(value: str, *, lease_owner: str) -> None:
            self.assertEqual(value, run_id)
            self.assertEqual(lease_owner, owner)
            async with SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                run.status = RunStatus.completed
                run.execution_slot = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                await session.commit()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        local_coordinator = RunCoordinator(
            lease_seconds=15,
            poll_seconds=0.01,
        )
        local_coordinator.lease_seconds = 0.3
        with (
            patch(
                "app.services.run_coordinator.run_crawl",
                new=terminal_then_wait,
            ),
            patch(
                "app.services.run_coordinator._terminal_transition_belongs_to_claim",
                new=AsyncMock(side_effect=RuntimeError("terminal lookup failed")),
            ),
        ):
            await local_coordinator._execute(
                RunClaim(
                    run_id=run_id,
                    owner=owner,
                    attempt_count=1,
                    resume_count=0,
                )
            )

        self.assertTrue(cancelled.is_set())
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertIsNone(run.execution_slot)
        self.assertIsNone(run.lease_owner)

    async def test_terminal_completed_marker_cleanup_preserves_new_report(
        self,
    ) -> None:
        report_json = {"narrative": {"headline": "Новый отчёт"}}
        run_id = await self._add_run(
            status=RunStatus.completed,
            progress_current=100,
            progress_total=100,
            progress_percent=100,
            stage_key="report",
            stage_label="Собираем отчёт и иллюстрации",
            stage_detail="Новый отчёт опубликован атомарно.",
            state_revision=11,
            state_changed_at=(datetime.now(timezone.utc) - timedelta(minutes=3)),
            report_json=report_json,
        )
        await self._mark_saved_answers_only(
            run_id,
            previous_status=RunStatus.failed,
        )

        recovered = await recover_expired_leases()
        self.assertEqual(recovered, 1)
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.progress_percent, 100)
        self.assertEqual(
            run.stage_detail,
            "Новый отчёт опубликован атомарно.",
        )
        self.assertEqual(run.report_json, report_json)
        self.assertEqual(run.config_json, {"page_limit": 6})

    async def test_recent_terminal_marker_waits_for_cli_cleanup(self) -> None:
        report_json = {"narrative": {"headline": "Новый отчёт"}}
        run_id = await self._add_run(
            status=RunStatus.completed,
            progress_current=100,
            progress_total=100,
            progress_percent=100,
            state_revision=11,
            state_changed_at=datetime.now(timezone.utc),
            report_json=report_json,
        )
        await self._mark_saved_answers_only(run_id)

        recovered = await recover_expired_leases()
        self.assertEqual(recovered, 0)
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.report_json, report_json)
        self.assertIn(SAVED_ANSWERS_ONLY_MARKER_KEY, run.config_json)
        self.assertIsNone(await _claim_next_run("generic-owner", lease_seconds=90))

    async def test_terminal_failed_marker_restores_previous_metadata(
        self,
    ) -> None:
        last_written_report = {"narrative": {"headline": "Последняя атомарная запись"}}
        run_id = await self._add_run(
            status=RunStatus.failed,
            progress_current=82,
            progress_total=100,
            progress_percent=82,
            stage_key="report",
            state_revision=13,
            state_changed_at=(datetime.now(timezone.utc) - timedelta(minutes=3)),
            report_json=last_written_report,
            error_message="Неудачная попытка.",
        )
        await self._mark_saved_answers_only(run_id)

        recovered = await recover_expired_leases()
        self.assertEqual(recovered, 1)
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.progress_percent, 100)
        self.assertIsNone(run.error_message)
        self.assertEqual(run.report_json, last_written_report)
        self.assertEqual(run.config_json, {"page_limit": 6})

    async def test_stale_worker_cannot_write_progress_or_terminal_state(self) -> None:
        run_id = await self._add_run(
            status=RunStatus.analyzing,
            execution_slot=1,
            lease_owner="current-owner",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            progress_current=45,
            progress_total=100,
            progress_percent=45,
            state_revision=5,
        )

        with bind_run_lease(run_id, "stale-owner"):
            with self.assertRaises(RunLeaseLostError):
                await update_progress(
                    run_id,
                    stage="report",
                    percent=90,
                    detail="stale",
                    eta_seconds=10,
                )
            self.assertFalse(await fail_run(run_id, "stale failure"))
            self.assertFalse(await complete_run(run_id))

        async with SessionLocal() as session:
            status, owner, percent, revision = (
                await session.execute(
                    select(
                        Run.status,
                        Run.lease_owner,
                        Run.progress_percent,
                        Run.state_revision,
                    ).where(Run.id == run_id)
                )
            ).one()
        self.assertEqual(status, RunStatus.analyzing)
        self.assertEqual(owner, "current-owner")
        self.assertEqual(percent, 45)
        self.assertEqual(revision, 5)

        with bind_run_lease(run_id, "current-owner"):
            self.assertTrue(
                await update_progress(
                    run_id,
                    stage="report",
                    percent=90,
                    detail="current",
                    eta_seconds=10,
                )
            )
            self.assertTrue(await complete_run(run_id))

        async with SessionLocal() as session:
            status, slot, revision = (
                await session.execute(
                    select(
                        Run.status,
                        Run.execution_slot,
                        Run.state_revision,
                    ).where(Run.id == run_id)
                )
            ).one()
        self.assertEqual(status, RunStatus.completed)
        self.assertIsNone(slot)
        self.assertEqual(revision, 7)

    async def test_release_preserves_progress_and_returns_revision(self) -> None:
        run_id = await self._add_run(
            status=RunStatus.crawling,
            execution_slot=1,
            lease_owner="release-owner",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            progress_current=31,
            progress_total=100,
            progress_percent=31,
            state_revision=4,
            resume_count=0,
        )
        released = await _release_claim(
            RunClaim(run_id=run_id, owner="release-owner"),
            detail="resume",
            reason="test",
        )
        self.assertIsNotNone(released)
        assert released is not None
        self.assertEqual(released.progress_percent, 31)
        self.assertEqual(released.state_revision, 5)
        self.assertEqual(released.resume_count, 1)

    async def test_coordinator_shutdown_requeues_active_work(self) -> None:
        run_id = await self._add_run(
            progress_current=22,
            progress_total=100,
            progress_percent=22,
            state_revision=3,
        )
        worker_started = asyncio.Event()

        async def blocking_worker(*args, **kwargs):
            del args, kwargs
            worker_started.set()
            await asyncio.Event().wait()

        local_coordinator = RunCoordinator(
            lease_seconds=15,
            poll_seconds=0.01,
        )
        with patch(
            "app.services.run_coordinator.run_crawl",
            new=blocking_worker,
        ):
            await local_coordinator.start()
            await asyncio.wait_for(worker_started.wait(), timeout=1)
            await local_coordinator.stop()

        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.pending)
        self.assertEqual(run.stage_key, "recovering")
        self.assertEqual(run.progress_percent, 22)
        self.assertIsNone(run.execution_slot)
        self.assertEqual(run.resume_reason, "service_restart")
        history = bus.channel_history(run_id)
        self.assertEqual(history[-1][1]["percent"], 22)

    async def test_preview_child_is_cancelled_when_crawl_setup_fails(self) -> None:
        run_id = await self._add_run(
            status=RunStatus.pending,
            execution_slot=1,
            lease_owner="preview-owner",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
        preview_started = asyncio.Event()
        preview_cancelled = asyncio.Event()

        async def blocking_preview(*args, **kwargs):
            del args, kwargs
            preview_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                preview_cancelled.set()
                raise

        async def failing_progress(*args, **kwargs):
            del args, kwargs
            await preview_started.wait()
            raise RuntimeError("progress failed")

        with (
            patch.object(
                crawler,
                "discover_site_pages",
                new=AsyncMock(return_value=[("https://example.com/", "home")]),
            ),
            patch.object(
                crawler,
                "capture_site_preview",
                new=blocking_preview,
            ),
            patch.object(
                crawler,
                "update_progress",
                new=failing_progress,
            ),
            patch.object(
                crawler,
                "fail_run",
                new=AsyncMock(return_value=True),
            ),
        ):
            await crawler.run_crawl(
                run_id,
                lease_owner="preview-owner",
            )

        self.assertTrue(preview_started.is_set())
        self.assertTrue(preview_cancelled.is_set())

    async def test_sqlite_queue_pragmas_and_unique_guard_are_enabled(self) -> None:
        async with SessionLocal() as session:
            journal_mode = (
                await session.execute(text("PRAGMA journal_mode"))
            ).scalar_one()
            busy_timeout = (
                await session.execute(text("PRAGMA busy_timeout"))
            ).scalar_one()
            indexes = (await session.execute(text("PRAGMA index_list('runs')"))).all()
        self.assertEqual(str(journal_mode).lower(), "wal")
        self.assertGreaterEqual(int(busy_timeout), 10_000)
        self.assertIn(
            ("uq_runs_execution_slot", 1),
            {(str(row[1]), int(row[2])) for row in indexes},
        )

    async def test_sse_snapshot_helpers_are_retry_epoch_aware(self) -> None:
        run_id = await self._add_run(
            stage_key="recovering",
            stage_label="Восстанавливаем проверку",
            progress_percent=57,
            state_revision=9,
            resume_count=3,
        )
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        snapshot = _database_snapshot_event(run, {run_id: 2})
        self.assertEqual(snapshot["run_state"], "recovering")
        self.assertEqual(snapshot["percent"], 57)
        self.assertEqual(snapshot["queue_position"], 2)
        self.assertEqual(snapshot["state_revision"], 9)
        self.assertEqual(
            _last_sequence_for_epoch("3:18", stream_epoch=3),
            18,
        )
        self.assertIsNone(_last_sequence_for_epoch("2:18", stream_epoch=3))
        self.assertIsNone(_last_sequence_for_epoch("18", stream_epoch=3))


class DurableQueueApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )
        self.run_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        if self.run_ids:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id.in_(self.run_ids)))
                await session.commit()

    async def test_concurrent_duplicate_posts_return_one_durable_run(self) -> None:
        domain = f"duplicate-{uuid.uuid4().hex}.example.com"
        with patch("app.routes.runs.coordinator.wake"):
            first, second = await asyncio.gather(
                self.client.post("/api/runs", json={"domain": domain}),
                self.client.post("/api/runs", json={"domain": domain}),
            )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["run_id"], second.json()["run_id"])
        run_id = first.json()["run_id"]
        self.run_ids.append(run_id)
        async with SessionLocal() as session:
            rows = (
                (
                    await session.execute(
                        select(Run.id).where(
                            Run.domain == domain,
                            Run.status.in_(
                                (
                                    RunStatus.pending,
                                    RunStatus.crawling,
                                    RunStatus.analyzing,
                                )
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(rows, [run_id])

    async def test_queue_capacity_is_enforced_atomically(self) -> None:
        first_domain = f"capacity-a-{uuid.uuid4().hex}.example.com"
        second_domain = f"capacity-b-{uuid.uuid4().hex}.example.com"
        with (
            patch("app.routes.runs.coordinator.wake"),
            patch(
                "app.routes.runs.settings.RUN_QUEUE_MAX_PENDING",
                1,
            ),
        ):
            first = await self.client.post(
                "/api/runs",
                json={"domain": first_domain},
            )
            second = await self.client.post(
                "/api/runs",
                json={"domain": second_domain},
            )
        self.assertEqual(first.status_code, 200)
        self.run_ids.append(first.json()["run_id"])
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers.get("retry-after"), "60")

    async def test_finished_sse_always_starts_from_database_snapshot(self) -> None:
        run_id = f"test-sse-{uuid.uuid4()}"
        self.run_ids.append(run_id)
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="snapshot.example.com",
                    status=RunStatus.completed,
                    config_json={},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    stage_key="report",
                    stage_label="Собираем отчёт и иллюстрации",
                    state_revision=12,
                    resume_count=4,
                )
            )
            await session.commit()
        response = await self.client.get(
            f"/api/runs/{run_id}/events",
            headers={"Last-Event-ID": "3:999"},
        )
        self.assertEqual(response.status_code, 200)
        data_lines = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(len(data_lines), 1)
        payload = json.loads(data_lines[0])
        self.assertEqual(payload["type"], "final")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["state_revision"], 12)
