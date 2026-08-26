from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base, ModelAnswer, Run, RunStatus, VisibilityPrompt
from app.services.analyzer import (
    PanelCheckpointMismatchError,
    _assert_saved_answer_marker_raw_corpus,
)
from app.services.raw_answer_integrity import model_answer_fingerprint_rows
from app.services.run_coordinator import (
    SAVED_ANSWERS_ONLY_MARKER_KEY,
    SAVED_ANSWERS_ONLY_MARKER_VERSION,
    SAVED_ANSWERS_ONLY_MODE,
)


class PublicationRawAnswerIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "raw-integrity.sqlite3"
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()
        self.temp_dir.cleanup()

    async def _create_marked_run(self) -> tuple[str, int]:
        run_id = str(uuid.uuid4())
        owner = "operator-reprocess:test"
        async with self.SessionLocal() as session:
            run = Run(
                id=run_id,
                domain="example.com",
                status=RunStatus.analyzing,
                config_json={},
                execution_slot=1,
                lease_owner=owner,
                attempt_count=3,
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения подходят для задачи?",
                rationale="Проверка raw boundary.",
                sequence=1,
            )
            session.add_all((run, prompt))
            await session.flush()
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="openai/example",
                mode="web",
                status="completed",
                response_text="Сохранённый полный ответ",
                citations_json=[{"url": "https://example.com/source"}],
                usage_json={"completion_tokens": 42},
            )
            session.add(answer)
            await session.commit()
            answer_id = answer.id

        async with self.SessionLocal() as session:
            run = await session.get(Run, run_id)
            answer = await session.get(ModelAnswer, answer_id)
            assert run is not None
            assert answer is not None
            run.config_json = {
                SAVED_ANSWERS_ONLY_MARKER_KEY: {
                    "version": SAVED_ANSWERS_ONLY_MARKER_VERSION,
                    "mode": SAVED_ANSWERS_ONLY_MODE,
                    "run_id": run_id,
                    "owner": owner,
                    "attempt_count": 3,
                    "raw_answers_sha256": model_answer_fingerprint_rows([answer]),
                }
            }
            await session.commit()
        return run_id, answer_id

    async def test_exact_marker_digest_passes_inside_publication_transaction(
        self,
    ) -> None:
        run_id, _answer_id = await self._create_marked_run()

        async with self.SessionLocal() as session:
            await _assert_saved_answer_marker_raw_corpus(session, run_id=run_id)

    async def test_changed_raw_answer_blocks_publication(self) -> None:
        run_id, answer_id = await self._create_marked_run()
        async with self.SessionLocal() as session:
            answer = await session.get(ModelAnswer, answer_id)
            assert answer is not None
            answer.response_text = "Незаметно изменённый ответ"
            await session.commit()

        async with self.SessionLocal() as session:
            with self.assertRaisesRegex(
                PanelCheckpointMismatchError,
                "persisted_corpus_changed",
            ):
                await _assert_saved_answer_marker_raw_corpus(session, run_id=run_id)
