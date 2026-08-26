from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import (
    Base,
    ModelAnswer,
    ReportIllustration,
    Run,
    RunStatus,
    VisibilityPrompt,
)
from app.services import run_coordinator
from app.services.analyzer import (
    _reuse_saved_illustration_assets,
    _synchronize_reused_illustration_metadata,
)
from app.services.run_coordinator import SAVED_ANSWERS_ONLY_MARKER_KEY
from scripts import reprocess_saved_run as reprocess_cli


class SavedRunReprocessCliTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "reprocess.sqlite3"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(
                text(
                    "CREATE UNIQUE INDEX uq_test_runs_execution_slot "
                    "ON runs(execution_slot) "
                    "WHERE execution_slot IS NOT NULL"
                )
            )
        self._db_patches = (
            patch.object(reprocess_cli, "engine", self.engine),
            patch.object(reprocess_cli, "SessionLocal", self.SessionLocal),
        )
        for db_patch in self._db_patches:
            db_patch.start()
        self.run_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        for db_patch in reversed(self._db_patches):
            db_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    async def _create_run(
        self,
        *,
        status: RunStatus,
        response_text: str | None = None,
        **values: object,
    ) -> str:
        run_id = str(uuid.uuid4())
        self.run_ids.append(run_id)
        config_json = values.pop("config_json", {})
        async with self.SessionLocal() as session:
            run = Run(
                id=run_id,
                domain="example.com",
                status=status,
                config_json=config_json,
                **values,
            )
            session.add(run)
            if response_text is not None:
                prompts = [
                    VisibilityPrompt(
                        run_id=run_id,
                        prompt_key=(
                            f"u-{sequence}" if sequence <= 6 else f"b-{sequence - 6}"
                        ),
                        intent_class=(
                            ("I", "E", "T", "NB", "NAV", "TR")[sequence - 1]
                            if sequence <= 6
                            else "B"
                        ),
                        role=(
                            "unbranded_discovery"
                            if sequence <= 6
                            else "brand_diagnostic"
                        ),
                        text=f"Сценарий проверки {sequence}",
                        rationale="Проверка полного сохранённого корпуса.",
                        sequence=sequence,
                    )
                    for sequence in range(1, 10)
                ]
                session.add_all(prompts)
                await session.flush()
                providers_by_mode = {
                    "web": (
                        "openai",
                        "gemini",
                        "perplexity",
                        "deepseek",
                        "claude",
                    ),
                    "memory": (
                        "openai",
                        "gemini",
                        "deepseek",
                        "claude",
                    ),
                }
                session.add_all(
                    [
                        ModelAnswer(
                            run_id=run_id,
                            prompt_id=prompt.id,
                            provider_key=provider,
                            model=f"test/{provider}",
                            mode=mode,
                            status="completed",
                            response_text=response_text,
                        )
                        for prompt in prompts
                        for mode, providers in providers_by_mode.items()
                        for provider in providers
                    ]
                )
            await session.commit()
        return run_id

    async def _delete_run(self, run_id: str) -> None:
        async with self.SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id == run_id))
            await session.commit()

    async def test_rejects_invalid_or_unknown_uuid(self) -> None:
        with self.assertRaisesRegex(
            reprocess_cli.ReprocessGuardError,
            "корректным UUID",
        ):
            await reprocess_cli.reprocess_saved_run("not-a-uuid")

        missing = str(uuid.uuid4())
        with self.assertRaisesRegex(
            reprocess_cli.ReprocessGuardError,
            "не найдена",
        ):
            await reprocess_cli.reprocess_saved_run(missing)

    async def test_rejects_active_runs(self) -> None:
        for status in (
            RunStatus.pending,
            RunStatus.crawling,
            RunStatus.analyzing,
        ):
            with self.subTest(status=status.value):
                run_id = await self._create_run(
                    status=status,
                    response_text="Сохранённый ответ.",
                )
                with self.assertRaisesRegex(
                    reprocess_cli.ReprocessGuardError,
                    status.value,
                ):
                    await reprocess_cli.reprocess_saved_run(run_id)

    async def test_requires_completed_nonempty_raw_answer(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.failed,
            response_text="   ",
        )
        with self.assertRaisesRegex(
            reprocess_cli.ReprocessGuardError,
            "raw-ответов",
        ):
            await reprocess_cli.reprocess_saved_run(run_id)

    async def test_rejects_partial_panel_before_downstream_tokens(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Сохранённый ответ.",
        )
        async with self.SessionLocal() as session:
            missing_cell_id = (
                await session.execute(
                    select(ModelAnswer.id)
                    .join(
                        VisibilityPrompt,
                        VisibilityPrompt.id == ModelAnswer.prompt_id,
                    )
                    .where(
                        ModelAnswer.run_id == run_id,
                        VisibilityPrompt.sequence == 9,
                        ModelAnswer.provider_key == "claude",
                        ModelAnswer.mode == "memory",
                    )
                )
            ).scalar_one()
            await session.execute(
                delete(ModelAnswer).where(ModelAnswer.id == missing_cell_id)
            )
            await session.commit()

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new_callable=AsyncMock,
            ) as reprocess,
            patch.object(
                reprocess_cli.analyzer,
                "_run_panel",
                new_callable=AsyncMock,
            ) as panel,
            self.assertRaisesRegex(
                reprocess_cli.ReprocessGuardError,
                "complete 81-cell panel; found 80 cells",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        reprocess.assert_not_awaited()
        panel.assert_not_awaited()
        async with self.SessionLocal() as session:
            status, slot, owner, config_json = (
                await session.execute(
                    select(
                        Run.status,
                        Run.execution_slot,
                        Run.lease_owner,
                        Run.config_json,
                    ).where(Run.id == run_id)
                )
            ).one()
        self.assertEqual(status, RunStatus.failed)
        self.assertIsNone(slot)
        self.assertIsNone(owner)
        self.assertNotIn(SAVED_ANSWERS_ONLY_MARKER_KEY, config_json)

    async def test_rejects_when_any_other_run_is_in_durable_queue(self) -> None:
        target_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Сохранённый ответ.",
        )
        for blocker_status in (
            RunStatus.pending,
            RunStatus.crawling,
            RunStatus.analyzing,
        ):
            with self.subTest(status=blocker_status.value):
                blocker_id = await self._create_run(status=blocker_status)
                try:
                    with self.assertRaisesRegex(
                        reprocess_cli.ReprocessGuardError,
                        "durable queue занята",
                    ):
                        await reprocess_cli.reprocess_saved_run(target_id)
                finally:
                    await self._delete_run(blocker_id)

                async with self.SessionLocal() as session:
                    status, slot = (
                        await session.execute(
                            select(Run.status, Run.execution_slot).where(
                                Run.id == target_id
                            )
                        )
                    ).one()
                self.assertEqual(status, RunStatus.failed)
                self.assertIsNone(slot)

    async def test_rejects_an_occupied_slot_even_on_terminal_row(self) -> None:
        target_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Сохранённый ответ.",
        )
        await self._create_run(
            status=RunStatus.completed,
            execution_slot=1,
            lease_owner="another-operator",
            heartbeat_at=datetime.now(timezone.utc),
            lease_expires_at=(datetime.now(timezone.utc) + timedelta(minutes=2)),
        )

        with self.assertRaisesRegex(
            reprocess_cli.ReprocessGuardError,
            "durable queue занята",
        ):
            await reprocess_cli.reprocess_saved_run(target_id)

    async def test_saved_guard_keeps_sealed_historical_lane_topology(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Сохранённый ответ.",
        )
        async with self.SessionLocal() as session:
            prompt_rows = list(
                (
                    await session.execute(
                        select(
                            VisibilityPrompt.id,
                            VisibilityPrompt.role,
                        )
                        .where(VisibilityPrompt.run_id == run_id)
                        .order_by(VisibilityPrompt.sequence)
                    )
                )
                .mappings()
                .all()
            )
            answer_rows = [
                dict(row)
                for row in (
                    (
                        await session.execute(
                            select(
                                ModelAnswer.prompt_id,
                                ModelAnswer.provider_key,
                                ModelAnswer.model,
                                ModelAnswer.mode,
                                ModelAnswer.status,
                            )
                            .where(ModelAnswer.run_id == run_id)
                            .order_by(ModelAnswer.id)
                        )
                    )
                    .mappings()
                    .all()
                )
            ]
        sealed_cells = [
            {
                key: row[key]
                for key in ("prompt_id", "provider_key", "model", "mode")
            }
            for row in answer_rows
        ]

        reprocess_cli._validate_complete_saved_panel(
            prompt_rows,
            answer_rows,
            sealed_expected_cells=sealed_cells,
        )

        current_grid_rows = [dict(row) for row in answer_rows]
        for row in current_grid_rows:
            if row["mode"] == "memory" and row["provider_key"] == "claude":
                row["provider_key"] = "perplexity"
                row["model"] = "test/perplexity"
        with self.assertRaisesRegex(
            reprocess_cli.ReprocessGuardError,
            "outside its sealed historical grid",
        ):
            reprocess_cli._validate_complete_saved_panel(
                prompt_rows,
                current_grid_rows,
                sealed_expected_cells=sealed_cells,
            )

    async def test_accepts_complete_nine_prompt_eighty_one_cell_corpus(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Сохранённый ответ.",
            config_json={"page_limit": 6, "keep": "value"},
        )
        messages: list[str] = []

        async with self.SessionLocal() as session:
            prompt_ids = list(
                (
                    await session.execute(
                        select(VisibilityPrompt.id).where(
                            VisibilityPrompt.run_id == run_id
                        )
                    )
                ).scalars()
            )
            answer_ids = list(
                (
                    await session.execute(
                        select(ModelAnswer.id).where(ModelAnswer.run_id == run_id)
                    )
                ).scalars()
            )
        self.assertEqual(len(prompt_ids), 9)
        self.assertEqual(len(answer_ids), 81)

        async def complete_reprocess(value: str) -> None:
            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                self.assertEqual(run.status, RunStatus.analyzing)
                self.assertEqual(run.execution_slot, 1)
                self.assertTrue(
                    (run.lease_owner or "").startswith("operator-reprocess:")
                )
                self.assertIsNotNone(run.lease_expires_at)
                marker = run.config_json.get(SAVED_ANSWERS_ONLY_MARKER_KEY)
                self.assertIsInstance(marker, dict)
                self.assertEqual(marker["mode"], "saved_answers_only")
                self.assertEqual(marker["run_id"], value)
                self.assertEqual(
                    marker["raw_answers_sha256"],
                    reprocess_cli._model_answer_fingerprint_rows(
                        list(
                            (
                                await session.execute(
                                    select(ModelAnswer).where(
                                        ModelAnswer.run_id == value
                                    )
                                )
                            )
                            .scalars()
                            .all()
                        )
                    ),
                )
                run.status = RunStatus.completed
                run.execution_slot = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                await session.commit()

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=complete_reprocess),
            ) as reprocess,
            patch.object(
                reprocess_cli.analyzer,
                "_run_panel",
                new_callable=AsyncMock,
            ) as panel,
        ):
            result = await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=messages.append,
            )

        self.assertEqual(result, run_id)
        reprocess.assert_awaited_once_with(run_id)
        panel.assert_not_awaited()
        self.assertIn("панели запрещён", messages[0])
        self.assertIn("исходных ответов: 81", messages[0])
        async with self.SessionLocal() as session:
            status, slot, owner, config_json = (
                await session.execute(
                    select(
                        Run.status,
                        Run.execution_slot,
                        Run.lease_owner,
                        Run.config_json,
                    ).where(Run.id == run_id)
                )
            ).one()
        self.assertEqual(status, RunStatus.completed)
        self.assertIsNone(slot)
        self.assertIsNone(owner)
        self.assertEqual(config_json, {"page_limit": 6, "keep": "value"})
        self.assertNotIn(SAVED_ANSWERS_ONLY_MARKER_KEY, config_json)

    async def test_threaded_heartbeat_survives_cpu_block_and_terminal_race(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Сохранённый ответ.",
            progress_percent=100,
            stage_key="report",
            report_json={"version": "old"},
        )
        observed_heartbeat: list[datetime] = []

        async def blocking_complete(value: str) -> None:
            time.sleep(0.55)  # noqa: ASYNC251 - deliberately starve event loop
            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                heartbeat_at = run.heartbeat_at
                lease_expires_at = run.lease_expires_at
                self.assertIsNotNone(heartbeat_at)
                self.assertIsNotNone(lease_expires_at)
                assert heartbeat_at is not None
                assert lease_expires_at is not None
                comparable_heartbeat = heartbeat_at.replace(
                    tzinfo=heartbeat_at.tzinfo or timezone.utc
                )
                comparable_expiry = lease_expires_at.replace(
                    tzinfo=lease_expires_at.tzinfo or timezone.utc
                )
                observed_heartbeat.append(comparable_heartbeat)
                self.assertGreater(comparable_expiry, datetime.now(timezone.utc))
                run.status = RunStatus.completed
                run.progress_percent = 100
                run.stage_key = "report"
                run.report_json = {"version": "new"}
                run.execution_slot = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                await session.commit()
            # Let the heartbeat observe rowcount=0 before this worker returns.
            await asyncio.sleep(0.15)

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=blocking_complete),
            ),
            patch.object(reprocess_cli, "REPROCESS_LEASE_SECONDS", 0.3),
            patch.object(reprocess_cli, "REPROCESS_HEARTBEAT_SECONDS", 0.05),
            patch.object(reprocess_cli, "REPROCESS_TERMINAL_GRACE_SECONDS", 1.0),
        ):
            result = await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        self.assertEqual(result, run_id)
        self.assertEqual(len(observed_heartbeat), 1)
        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.report_json, {"version": "new"})
        self.assertIsNone(run.execution_slot)
        self.assertNotIn(SAVED_ANSWERS_ONLY_MARKER_KEY, run.config_json)

    async def test_claim_lease_starts_after_lossless_snapshot_work(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Сохранённый ответ.",
            config_json={"page_limit": 6},
            progress_percent=100,
            stage_key="report",
            report_json={"version": "published"},
        )
        fingerprint = reprocess_cli._model_answer_fingerprint_rows

        def slow_fingerprint(rows):
            time.sleep(0.35)  # noqa: ASYNC251 - emulate an unbounded raw corpus
            return fingerprint(rows)

        with (
            patch.object(
                reprocess_cli,
                "_model_answer_fingerprint_rows",
                side_effect=slow_fingerprint,
            ),
            patch.object(reprocess_cli, "REPROCESS_LEASE_SECONDS", 0.3),
        ):
            claim = await reprocess_cli._claim_eligible_run(run_id)

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertIsNotNone(run.lease_expires_at)
        assert run.lease_expires_at is not None
        comparable_expiry = run.lease_expires_at.replace(
            tzinfo=run.lease_expires_at.tzinfo or timezone.utc
        )
        self.assertGreater(
            comparable_expiry,
            datetime.now(timezone.utc) + timedelta(seconds=0.15),
        )
        released = await reprocess_cli._release_reprocess_claim(
            claim,
            successful=False,
        )
        self.assertTrue(released)

    async def test_terminal_lookup_failure_cancels_reprocess_worker(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Сохранённый ответ.",
            config_json={"page_limit": 6},
            progress_percent=100,
            stage_key="report",
            report_json={"version": "published"},
        )
        cancelled = asyncio.Event()

        async def terminal_then_wait(value: str) -> None:
            async with self.SessionLocal() as session:
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

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=terminal_then_wait),
            ),
            patch.object(
                reprocess_cli,
                "_terminal_transition_belongs_to_claim",
                new=AsyncMock(side_effect=RuntimeError("terminal lookup failed")),
            ),
            patch.object(reprocess_cli, "REPROCESS_LEASE_SECONDS", 0.3),
            patch.object(reprocess_cli, "REPROCESS_HEARTBEAT_SECONDS", 0.05),
            self.assertRaisesRegex(RuntimeError, "terminal lookup failed"),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        self.assertTrue(cancelled.is_set())
        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.report_json, {"version": "published"})
        self.assertEqual(run.config_json, {"page_limit": 6})
        self.assertIsNone(run.execution_slot)

    async def test_external_terminal_restore_is_not_accepted_as_own_success(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Сохранённый ответ.",
            config_json={"page_limit": 6},
            progress_percent=100,
            stage_key="report",
            report_json={"version": "published"},
        )
        cancelled = asyncio.Event()

        async def externally_recovered(value: str) -> None:
            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                run.status = RunStatus.completed
                run.config_json = {"page_limit": 6}
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

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=externally_recovered),
            ),
            patch.object(reprocess_cli, "REPROCESS_LEASE_SECONDS", 0.3),
            patch.object(reprocess_cli, "REPROCESS_HEARTBEAT_SECONDS", 0.05),
            self.assertRaisesRegex(
                reprocess_cli.ReprocessExecutionError,
                "Потерян общий execution_slot",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        self.assertTrue(cancelled.is_set())
        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.report_json, {"version": "published"})
        self.assertIsNone(run.execution_slot)
        self.assertEqual(run.config_json, {"page_limit": 6})

    async def test_expired_claim_restores_terminal_without_touching_raw(
        self,
    ) -> None:
        raw_text = "Сохранённый raw-ответ для аварийного восстановления."
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text=raw_text,
            config_json={"page_limit": 6},
            progress_current=100,
            progress_total=100,
            progress_percent=100,
            stage_key="report",
            stage_label="Отчёт готов",
            stage_detail="Готово.",
            report_json={"narrative": {"headline": "До аварии"}},
        )
        claim = await reprocess_cli._claim_eligible_run(run_id)
        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            run.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await session.commit()

        with patch.object(
            run_coordinator,
            "SessionLocal",
            self.SessionLocal,
        ):
            recovered = await run_coordinator.recover_expired_leases()
        self.assertEqual(recovered, 1)

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            answer = (
                await session.execute(
                    select(ModelAnswer)
                    .where(ModelAnswer.run_id == run_id)
                    .order_by(ModelAnswer.id)
                    .limit(1)
                )
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertIsNone(run.execution_slot)
        self.assertEqual(run.config_json, {"page_limit": 6})
        self.assertEqual(answer.response_text, raw_text)
        count, raw_sha256 = await reprocess_cli._model_answer_fingerprint(run_id)
        self.assertEqual(count, claim.total_answers)
        self.assertEqual(raw_sha256, claim.raw_answers_sha256)

    async def test_reused_illustration_copy_stays_in_sync_with_database(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            report_json={
                "illustrations": [
                    {
                        "sequence": 1,
                        "title": "Старый вывод",
                        "caption": "Старые расчёты.",
                        "alt_text": "Фактическая сцена на сохранённой картинке.",
                        "file_url": "/static/generated/test/01.png",
                    }
                ]
            },
        )
        original_usage = {
            "generation_version": "image-v1",
            "image_sha256": "saved-image",
        }
        async with self.SessionLocal() as session:
            session.add(
                ReportIllustration(
                    run_id=run_id,
                    sequence=1,
                    title="Старый вывод",
                    caption="Старые расчёты.",
                    alt_text="Фактическая сцена на сохранённой картинке.",
                    file_url="/static/generated/test/01.png",
                    generation_prompt="original-generation-prompt",
                    model="test/image-model",
                    usage_json=original_usage,
                )
            )
            await session.commit()

        refreshed = _reuse_saved_illustration_assets(
            [
                {
                    "sequence": 1,
                    "title": "Старый вывод",
                    "caption": "Старые расчёты.",
                    "alt_text": "Фактическая сцена на сохранённой картинке.",
                    "file_url": "/static/generated/test/01.png",
                }
            ],
            [
                {
                    "title": "Новый вывод",
                    "caption": "Подпись по пересчитанным метрикам.",
                    "alt_text": "Описание другой, ещё не созданной сцены.",
                }
            ],
        )
        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            run.report_json = {"illustrations": refreshed}
            await _synchronize_reused_illustration_metadata(
                session,
                run_id=run_id,
                illustrations=refreshed,
            )
            await session.commit()

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            row = (
                await session.execute(
                    select(ReportIllustration).where(
                        ReportIllustration.run_id == run_id,
                        ReportIllustration.sequence == 1,
                    )
                )
            ).scalar_one()

        public = run.report_json["illustrations"][0]
        self.assertEqual(row.title, public["title"])
        self.assertEqual(row.caption, public["caption"])
        self.assertEqual(row.alt_text, public["alt_text"])
        self.assertEqual(
            row.alt_text,
            "Описание другой, ещё не созданной сцены.",
        )
        self.assertEqual(row.file_url, "/static/generated/test/01.png")
        self.assertEqual(row.generation_prompt, "original-generation-prompt")
        self.assertEqual(row.model, "test/image-model")
        self.assertEqual(row.usage_json, original_usage)

    async def test_rejects_result_if_saved_raw_answer_changed(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Неизменяемый сохранённый ответ.",
        )

        async def corrupt_reprocess(value: str) -> None:
            async with self.SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer)
                        .where(ModelAnswer.run_id == value)
                        .order_by(ModelAnswer.id)
                        .limit(1)
                    )
                ).scalar_one()
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                answer.response_text = "Случайно изменённый ответ."
                run.status = RunStatus.completed
                await session.commit()

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=corrupt_reprocess),
            ),
            self.assertRaisesRegex(
                reprocess_cli.ReprocessExecutionError,
                "изменение сохранённых ответов",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        async with self.SessionLocal() as session:
            answer = (
                await session.execute(
                    select(ModelAnswer)
                    .where(ModelAnswer.run_id == run_id)
                    .order_by(ModelAnswer.id)
                    .limit(1)
                )
            ).scalar_one()
        self.assertEqual(answer.response_text, "Неизменяемый сохранённый ответ.")

    async def test_failure_path_restores_raw_and_previous_public_report(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Исходный raw-ответ.",
            progress_current=9,
            progress_total=9,
            progress_percent=100,
            stage_key="report",
            analysis_markdown="# Прежний отчёт",
            report_json={"version": "published"},
        )

        async def corrupt_and_fail(value: str) -> None:
            async with self.SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer)
                        .where(ModelAnswer.run_id == value)
                        .order_by(ModelAnswer.id)
                        .limit(1)
                    )
                ).scalar_one()
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                answer.response_text = "Повреждённый raw-ответ."
                run.status = RunStatus.failed
                run.analysis_markdown = "# Неудачная замена"
                run.report_json = {"version": "broken"}
                await session.commit()

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=corrupt_and_fail),
            ),
            self.assertRaisesRegex(
                reprocess_cli.ReprocessExecutionError,
                "Raw-корпус восстановлен",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            answer = (
                await session.execute(
                    select(ModelAnswer)
                    .where(ModelAnswer.run_id == run_id)
                    .order_by(ModelAnswer.id)
                    .limit(1)
                )
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.analysis_markdown, "# Прежний отчёт")
        self.assertEqual(run.report_json, {"version": "published"})
        self.assertEqual(answer.response_text, "Исходный raw-ответ.")
        self.assertNotIn(
            SAVED_ANSWERS_ONLY_MARKER_KEY,
            run.config_json,
        )

    async def test_terminal_failed_cell_is_preserved_for_coverage_admission(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Завершённый ответ.",
        )
        async with self.SessionLocal() as session:
            answer = (
                await session.execute(
                    select(ModelAnswer)
                    .where(ModelAnswer.run_id == run_id)
                    .order_by(ModelAnswer.id)
                    .limit(1)
                )
            ).scalar_one()
            answer.status = "failed"
            answer.error_message = "Сохранённая ошибка."
            await session.commit()

        claim = await reprocess_cli._claim_eligible_run(run_id)
        try:
            self.assertEqual(claim.total_answers, 81)
            self.assertEqual(claim.completed_answers, 80)
            failed = [
                row
                for row in claim.raw_answer_snapshot
                if row.get("status") == "failed"
            ]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0].get("error_message"), "Сохранённая ошибка.")
        finally:
            await reprocess_cli._release_reprocess_claim(
                claim,
                successful=False,
            )

    async def test_concurrent_operator_run_cannot_bypass_slot(self) -> None:
        first_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Первый сохранённый ответ.",
        )
        second_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Второй сохранённый ответ.",
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        called: list[str] = []

        async def held_reprocess(value: str) -> None:
            called.append(value)
            entered.set()
            await release.wait()
            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                run.status = RunStatus.completed
                await session.commit()

        with patch.object(
            reprocess_cli.analyzer,
            "reprocess_saved_answers",
            new=AsyncMock(side_effect=held_reprocess),
        ):
            first_task = asyncio.create_task(
                reprocess_cli.reprocess_saved_run(
                    first_id,
                    announce=lambda _message: None,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            try:
                with self.assertRaisesRegex(
                    reprocess_cli.ReprocessGuardError,
                    "durable queue занята",
                ):
                    await reprocess_cli.reprocess_saved_run(
                        second_id,
                        announce=lambda _message: None,
                    )
            finally:
                release.set()
            self.assertEqual(await first_task, first_id)

        self.assertEqual(called, [first_id])
        async with self.SessionLocal() as session:
            second_status, second_slot = (
                await session.execute(
                    select(Run.status, Run.execution_slot).where(Run.id == second_id)
                )
            ).one()
        self.assertEqual(second_status, RunStatus.completed)
        self.assertIsNone(second_slot)

    async def test_reports_reprocess_failure_from_persisted_status(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.failed,
            response_text="Сохранённый ответ.",
        )
        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new_callable=AsyncMock,
            ),
            self.assertRaisesRegex(
                reprocess_cli.ReprocessExecutionError,
                "статусом analyzing",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.failed)
        self.assertIsNone(run.execution_slot)
        self.assertIsNone(run.lease_owner)
        self.assertNotIn(
            SAVED_ANSWERS_ONLY_MARKER_KEY,
            run.config_json,
        )

    async def test_failed_reprocess_keeps_previous_completed_report_public(
        self,
    ) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Сохранённый ответ.",
            progress_current=9,
            progress_total=9,
            progress_percent=100,
            stage_key="report",
            stage_label="Отчёт готов",
            stage_detail="Проверка завершена.",
            analysis_markdown="# Готовый отчёт",
            report_json={"version": "published"},
        )

        async def fail_reprocess(value: str) -> None:
            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                run.status = RunStatus.failed
                run.error_message = "Новая попытка не завершилась."
                # Production ``fail_run`` releases the lease before the
                # operator wrapper gets a chance to restore the published
                # report. This is the regression state, not merely a failed
                # row that remains owned by the wrapper.
                run.execution_slot = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                run.state_revision += 1
                await session.commit()

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=fail_reprocess),
            ),
            self.assertRaisesRegex(
                reprocess_cli.ReprocessExecutionError,
                "статусом failed",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.completed)
        self.assertEqual(run.progress_percent, 100)
        self.assertEqual(run.stage_key, "report")
        self.assertEqual(run.analysis_markdown, "# Готовый отчёт")
        self.assertEqual(run.report_json, {"version": "published"})
        self.assertIsNone(run.error_message)
        self.assertIsNone(run.execution_slot)
        self.assertIsNone(run.lease_owner)
        self.assertNotIn(
            SAVED_ANSWERS_ONLY_MARKER_KEY,
            run.config_json,
        )

    async def test_cleanup_never_overwrites_a_new_foreign_attempt(self) -> None:
        run_id = await self._create_run(
            status=RunStatus.completed,
            response_text="Сохранённый ответ.",
            progress_current=9,
            progress_total=9,
            progress_percent=100,
            stage_key="report",
            analysis_markdown="# Старый отчёт",
            report_json={"version": "old"},
        )

        async def fail_then_start_foreign_attempt(value: str) -> None:
            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                # Our analyzer failed and cleared its lease.
                run.status = RunStatus.failed
                run.execution_slot = None
                run.lease_owner = None
                run.lease_expires_at = None
                run.heartbeat_at = None
                await session.commit()

            async with self.SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == value))
                ).scalar_one()
                # Before the old wrapper reaches finally, a newer generation
                # legitimately claims the same run.
                run.status = RunStatus.analyzing
                run.execution_slot = 1
                run.lease_owner = "foreign-attempt"
                run.attempt_count += 1
                run.analysis_markdown = "# Новый незавершённый отчёт"
                run.report_json = {"version": "new-attempt"}
                await session.commit()

        with (
            patch.object(
                reprocess_cli.analyzer,
                "reprocess_saved_answers",
                new=AsyncMock(side_effect=fail_then_start_foreign_attempt),
            ),
            self.assertRaisesRegex(
                reprocess_cli.ReprocessExecutionError,
                "статусом analyzing",
            ),
        ):
            await reprocess_cli.reprocess_saved_run(
                run_id,
                announce=lambda _message: None,
            )

        async with self.SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.analyzing)
        self.assertEqual(run.execution_slot, 1)
        self.assertEqual(run.lease_owner, "foreign-attempt")
        self.assertEqual(run.analysis_markdown, "# Новый незавершённый отчёт")
        self.assertEqual(run.report_json, {"version": "new-attempt"})


class ReprocessServiceContractTests(unittest.TestCase):
    def test_systemd_does_not_impose_a_total_reprocess_duration_cap(self) -> None:
        service_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "aiv-reprocess@.service"
        )
        service = service_path.read_text(encoding="utf-8")

        self.assertIn("RuntimeMaxSec=infinity", service)
        self.assertNotRegex(service, r"RuntimeMaxSec=(?:\d|\d+[smhd])")
        self.assertIn(
            "ExecStart=/root/projects/aiv-venvs/current/bin/python ",
            service,
        )
        self.assertNotIn("/root/.local/bin/uv run", service)
