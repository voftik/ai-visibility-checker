import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import delete, update

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import ModelAnswer, Run, RunStatus, VisibilityPrompt
from app.services import analyzer
from app.services.run_lease import RunLeaseLostError, bind_run_lease


class PanelLeaseWaiterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        if not self.run_ids:
            return
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id.in_(self.run_ids)))
            await session.commit()

    async def _fixture(
        self,
        *,
        attempt_count: int,
        claim_generation: int,
        heartbeat_age_seconds: int = 0,
    ) -> tuple[str, int, str]:
        run_id = f"test-panel-waiter-{uuid.uuid4()}"
        owner = f"owner-{uuid.uuid4()}"
        self.run_ids.append(run_id)
        now = datetime.now(timezone.utc)
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                    execution_slot=1,
                    lease_owner=owner,
                    lease_expires_at=now + timedelta(minutes=5),
                    heartbeat_at=now
                    - timedelta(seconds=heartbeat_age_seconds),
                    attempt_count=attempt_count,
                )
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="waiter-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения выбрать?",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status=analyzer._new_panel_claim_status(claim_generation),
            )
            session.add(answer)
            await session.commit()
            return run_id, answer.id, owner

    async def test_waiter_uses_fresh_lease_instead_of_total_wall_clock(self) -> None:
        run_id, answer_id, owner = await self._fixture(
            attempt_count=7,
            claim_generation=7,
        )
        polls = 0

        async def complete_after_poll(_delay: float) -> None:
            nonlocal polls
            polls += 1
            async with SessionLocal() as session:
                await session.execute(
                    update(ModelAnswer)
                    .where(ModelAnswer.id == answer_id)
                    .values(status="completed", response_text="Полный ответ")
                )
                await session.commit()

        # A tiny/zero provider timeout used to make the historical aggregate
        # deadline expire. The new waiter is governed only by durable lease
        # freshness and therefore completes when the claimed row does.
        with (
            bind_run_lease(run_id, owner),
            patch.object(settings, "OPENROUTER_TIMEOUT_SECONDS", 0),
            patch.object(analyzer.asyncio, "sleep", new=complete_after_poll),
        ):
            statuses = await analyzer._wait_for_panel_claims(run_id, mode="web")

        self.assertEqual(statuses, ["completed"])
        self.assertEqual(polls, 1)

    async def test_waiter_rejects_same_generation_with_stale_heartbeat(self) -> None:
        run_id, _answer_id, owner = await self._fixture(
            attempt_count=3,
            claim_generation=3,
            heartbeat_age_seconds=max(30, int(settings.RUN_LEASE_SECONDS) + 5),
        )
        with bind_run_lease(run_id, owner):
            with self.assertRaises(RunLeaseLostError):
                await analyzer._wait_for_panel_claims(run_id, mode="web")

    async def test_waiter_rejects_claim_from_previous_generation(self) -> None:
        run_id, _answer_id, owner = await self._fixture(
            attempt_count=8,
            claim_generation=7,
        )
        with bind_run_lease(run_id, owner):
            with self.assertRaises(analyzer.PanelCheckpointMismatchError):
                await analyzer._wait_for_panel_claims(run_id, mode="web")


if __name__ == "__main__":
    unittest.main()
