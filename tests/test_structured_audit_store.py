from __future__ import annotations

import json
import tempfile
import json as json_lib
import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import Base, Run, RunArtifact, RunStatus
from app.services import structured_audit_store as audit_store
from app.services.openrouter import (
    OpenRouterError,
    OpenRouterStructuredContinuationError,
    chat_continuable_structured,
)


MODEL = "google/gemini-3.6-flash"
MESSAGES = [{"role": "user", "content": "Верни документ."}]
SCHEMA_NAME = "aiv_continuable_document"
SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
SOURCE_INPUT = {"domain": "example.com", "analysis": "critic"}


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def json(self) -> dict[str, Any]:
        return deepcopy(self._body)


class _SequenceClient:
    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self._bodies = bodies
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> _SequenceClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        content: bytes,
    ) -> _FakeResponse:
        payload = json_lib.loads(content.decode("utf-8"))
        self.requests.append(
            {"headers": dict(headers), "json": payload}
        )
        index = len(self.requests) - 1
        if index >= len(self._bodies):
            raise AssertionError("Unexpected provider POST")
        return _FakeResponse(self._bodies[index])


def _body(text: str, *, limited: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "provider": "Test Provider",
        "choices": [
            {
                "finish_reason": "length" if limited else "stop",
                "native_finish_reason": "MAX_TOKENS" if limited else "stop",
                "message": {
                    "role": "assistant",
                    "content": text,
                    "annotations": [],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": 10 + max(1, len(text) // 4),
            "server_tool_use": {"web_search_requests": 0},
        },
        "openrouter_metadata": {"pipeline": []},
    }


def _parts(count: int, *, overlap_chars: int = 16) -> tuple[list[dict[str, Any]], str]:
    if count < 1:
        raise ValueError("count must be positive")
    values = [
        {"name": f"{index:03d}-" + (chr(65 + index % 26) * 512)}
        for index in range(count)
    ]
    current = '{"items":[' + json.dumps(
        values[0], ensure_ascii=False, separators=(",", ":")
    )
    bodies = [_body(current, limited=count > 1)]
    for index, value in enumerate(values[1:], start=1):
        appended = "," + json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        )
        if index == count - 1:
            appended += "]}"
        response = current[-overlap_chars:] + appended
        current += appended
        bodies.append(_body(response, limited=index < count - 1))
    if count == 1:
        current += "]}"
        bodies[0] = _body(current, limited=False)
    return bodies, current


def _delta_capable(callback: Any) -> Any:
    setattr(
        callback,
        "aiv_structured_audit_event_version",
        "aiv-structured-audit-delta-v2",
    )
    return callback


class StructuredAuditStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "structured-audit.sqlite3"
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
        self._session_patch = patch.object(
            audit_store,
            "SessionLocal",
            self.SessionLocal,
        )
        self._lease_patch = patch.object(
            audit_store,
            "assert_run_lease",
            new=AsyncMock(),
        )
        self._session_patch.start()
        self._lease_patch.start()
        self.run_id = str(uuid.uuid4())
        async with self.SessionLocal() as session:
            session.add(
                Run(
                    id=self.run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            await session.commit()
        self.delta_sink = audit_store.structured_audit_checkpoint(
            self.run_id,
            stage_key="analysis_critic",
            owner_artifact_key="critic_report",
            source_input=SOURCE_INPUT,
            model=MODEL,
            owner_prompt_version="critic-v3",
        )

    async def asyncTearDown(self) -> None:
        self._lease_patch.stop()
        self._session_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    async def _persist(self, event: dict[str, Any]) -> None:
        await audit_store.persist_structured_audit_event(
            self.run_id,
            stage_key="analysis_critic",
            owner_artifact_key="critic_report",
            source_input=SOURCE_INPUT,
            model=MODEL,
            owner_prompt_version="critic-v3",
            event=event,
        )

    async def _generate(
        self,
        bodies: list[dict[str, Any]],
        *,
        document_id: str,
        checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        overlap_chars: int = 16,
    ) -> Any:
        client = _SequenceClient(bodies)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=envelope,
            ),
        ):
            result = await chat_continuable_structured(
                model=MODEL,
                messages=MESSAGES,
                response_schema=SCHEMA,
                schema_name=SCHEMA_NAME,
                document_id=document_id,
                overlap_chars=overlap_chars,
                retry_transport_errors=False,
                audit_checkpoint=checkpoint or self.delta_sink,
            )
        self.assertEqual(len(client.requests), len(bodies))
        return result

    async def _load(
        self,
        *,
        document_id: str,
        complete: bool | None = None,
        messages: list[dict[str, Any]] | None = None,
        model: str = MODEL,
        owner_prompt_version: str = "critic-v3",
        owner_artifact_key: str = "critic_report",
        source_input: dict[str, Any] | list[Any] = SOURCE_INPUT,
        overlap_chars: int = 16,
        reasoning_effort: str | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any] | None:
        return await audit_store.load_structured_checkpoint(
            self.run_id,
            owner_artifact_key=owner_artifact_key,
            source_input=source_input,
            model=model,
            owner_prompt_version=owner_prompt_version,
            messages=messages or MESSAGES,
            schema_name=SCHEMA_NAME,
            response_schema=SCHEMA,
            document_id=document_id,
            complete=complete,
            overlap_chars=overlap_chars,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )

    async def test_completed_receipt_crash_gap_is_promoted_without_network(
        self,
    ) -> None:
        document_id = "critic:completed-gap"
        bodies, expected = _parts(1)

        async def receipt_only(event: dict[str, Any]) -> None:
            if event.get("event_kind") == "provider_post":
                await self._persist(event)

        _delta_capable(receipt_only)

        await self._generate(
            bodies,
            document_id=document_id,
            checkpoint=receipt_only,
        )
        with patch(
            "app.services.openrouter.httpx.AsyncClient",
            side_effect=AssertionError("replay must be network-free"),
        ) as client_factory:
            checkpoint = await self._load(
                document_id=document_id,
                complete=None,
            )

        client_factory.assert_not_called()
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertTrue(checkpoint["manifest"]["complete"])
        self.assertEqual(checkpoint["partial_text"], expected)
        self.assertIsNotNone(
            await self._load(document_id=document_id, complete=True)
        )
        self.assertIsNone(
            await self._load(document_id=document_id, complete=False)
        )

    async def test_terminal_schema_receipt_crash_gaps_are_promoted_without_network(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"},'
        invalid_final = initial[-16:] + "{}]}"
        scenarios = [
            (
                "initial-normal",
                [_body('{"items":[{}]}', limited=False)],
                "complete_json_schema_violation",
            ),
            (
                "initial-output-limited",
                [_body('{"items":[{}]}', limited=True)],
                "complete_json_schema_violation",
            ),
            (
                "continuation-normal",
                [
                    _body(initial, limited=True),
                    _body(invalid_final, limited=False),
                ],
                "complete_json_schema_violation",
            ),
            (
                "continuation-output-limited",
                [
                    _body(initial, limited=True),
                    _body(invalid_final, limited=True),
                ],
                "complete_json_schema_violation",
            ),
            (
                "initial-impossible-boundary",
                [_body('{"items":]', limited=False)],
                "complete_rejected_json_part",
            ),
            (
                "initial-empty",
                [_body("", limited=False)],
                "complete_empty_response",
            ),
            (
                "continuation-rejected-boundary",
                [
                    _body(initial, limited=True),
                    _body("no-overlap]}", limited=False),
                ],
                "complete_rejected_json_part",
            ),
            (
                "continuation-empty",
                [
                    _body(initial, limited=True),
                    _body("", limited=False),
                ],
                "complete_empty_response",
            ),
        ]

        for label, bodies, expected_kind in scenarios:
            with self.subTest(label=label):
                document_id = f"critic:terminal-gap:{label}"

                async def omit_failed_head(event: dict[str, Any]) -> None:
                    if (
                        event.get("event_kind")
                        == "structured_continuation_checkpoint"
                        and event.get("status") == "failed"
                    ):
                        return
                    await self._persist(event)

                _delta_capable(omit_failed_head)
                with self.assertRaises(OpenRouterStructuredContinuationError):
                    await self._generate(
                        bodies,
                        document_id=document_id,
                        checkpoint=omit_failed_head,
                    )

                with patch(
                    "app.services.openrouter.httpx.AsyncClient",
                    side_effect=AssertionError(
                        "receipt promotion must be network-free"
                    ),
                ) as load_client_factory:
                    checkpoint = await self._load(document_id=document_id)
                    replayed = await self._load(document_id=document_id)
                load_client_factory.assert_not_called()
                self.assertIsNotNone(checkpoint)
                assert checkpoint is not None
                self.assertEqual(replayed, checkpoint)
                self.assertEqual(checkpoint["status"], "failed")
                self.assertFalse(checkpoint["manifest"]["complete"])
                marker = checkpoint["error"]["terminal_semantic_failure"]
                self.assertEqual(
                    marker["failure_kind"],
                    expected_kind,
                )

                with (
                    patch(
                        "app.services.openrouter.httpx.AsyncClient",
                        side_effect=AssertionError(
                            "terminal resume must not repeat a provider POST"
                        ),
                    ) as resume_client_factory,
                    self.assertRaises(OpenRouterStructuredContinuationError),
                ):
                    await chat_continuable_structured(
                        model=MODEL,
                        messages=MESSAGES,
                        response_schema=SCHEMA,
                        schema_name=SCHEMA_NAME,
                        document_id=document_id,
                        overlap_chars=16,
                        retry_transport_errors=False,
                        resume_checkpoint=checkpoint,
                    )
                resume_client_factory.assert_not_called()

    async def test_terminal_head_is_idempotent_and_absorbing(self) -> None:
        document_id = "critic:terminal-absorbing"
        events: list[dict[str, Any]] = []

        async def capture_and_persist(event: dict[str, Any]) -> None:
            events.append(deepcopy(event))
            await self._persist(event)

        _delta_capable(capture_and_persist)
        with self.assertRaises(OpenRouterStructuredContinuationError):
            await self._generate(
                [_body('{"items":[{}]}', limited=False)],
                document_id=document_id,
                checkpoint=capture_and_persist,
            )

        terminal = next(
            event
            for event in events
            if event.get("event_kind")
            == "structured_continuation_checkpoint"
            and event.get("status") == "failed"
        )
        await self._persist(deepcopy(terminal))

        downgrade = deepcopy(terminal)
        downgrade["event_id"] = uuid.uuid4().hex
        downgrade["status"] = "partial"
        downgrade["error"] = None
        with self.assertRaisesRegex(
            OpenRouterError,
            "cannot be downgraded or extended",
        ):
            await self._persist(downgrade)

    async def test_newer_paid_continuation_receipt_replays_after_partial_head(
        self,
    ) -> None:
        document_id = "critic:continuation-gap"
        bodies, expected = _parts(2)

        async def omit_completed_head(event: dict[str, Any]) -> None:
            manifest = event.get("manifest")
            event_complete = event.get("complete")
            if (
                event.get("event_kind") == "structured_continuation_checkpoint"
                and (
                    event_complete is True
                    or (
                        isinstance(manifest, dict)
                        and manifest.get("complete") is True
                    )
                )
            ):
                return
            await self._persist(event)

        _delta_capable(omit_completed_head)

        await self._generate(
            bodies,
            document_id=document_id,
            checkpoint=omit_completed_head,
        )
        checkpoint = await self._load(
            document_id=document_id,
            complete=None,
        )

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertTrue(checkpoint["manifest"]["complete"])
        self.assertEqual(checkpoint["partial_text"], expected)
        self.assertEqual(len(checkpoint["call_records"]), 2)

    async def test_latest_prefers_completed_and_rejects_stale_downgrade(
        self,
    ) -> None:
        document_id = "critic:latest-completed"
        bodies, _expected = _parts(1)
        events: list[dict[str, Any]] = []

        async def capture(event: dict[str, Any]) -> None:
            events.append(deepcopy(event))
            await self._persist(event)

        _delta_capable(capture)

        await self._generate(
            bodies,
            document_id=document_id,
            checkpoint=capture,
        )
        completed = next(
            event
            for event in events
            if event.get("event_kind")
            == "structured_continuation_checkpoint"
        )
        stale = deepcopy(completed)
        stale["event_id"] = uuid.uuid4().hex
        stale["status"] = "partial"
        stale["complete"] = False
        stale["head"]["complete"] = False
        await self._persist(stale)

        latest = await self._load(document_id=document_id, complete=None)
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertTrue(latest["manifest"]["complete"])

    async def test_snapshot_v1_events_remain_loadable(self) -> None:
        document_id = "critic:snapshot-v1"
        bodies, expected = _parts(2)
        await self._generate(
            bodies,
            document_id=document_id,
            checkpoint=self._persist,
        )

        latest = await self._load(document_id=document_id, complete=None)
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertTrue(latest["manifest"]["complete"])
        self.assertEqual(latest["partial_text"], expected)

    async def test_snapshot_v1_no_call_failure_remains_retryable(self) -> None:
        document_id = "critic:snapshot-v1-no-call"
        client = _SequenceClient([])
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                side_effect=OpenRouterError("transport unavailable"),
            ),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=envelope,
            ),
            self.assertRaises(OpenRouterStructuredContinuationError),
        ):
            await chat_continuable_structured(
                model=MODEL,
                messages=MESSAGES,
                response_schema=SCHEMA,
                schema_name=SCHEMA_NAME,
                document_id=document_id,
                overlap_chars=16,
                retry_transport_errors=False,
                audit_checkpoint=self._persist,
            )

        self.assertEqual(client.requests, [])
        self.assertIsNone(await self._load(document_id=document_id))
        async with self.SessionLocal() as session:
            head = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("lsa2_head_%"),
                    )
                )
            ).scalar_one()
        assert isinstance(head.output_json, dict)
        self.assertEqual(head.output_json.get("latest_sequence"), -1)

    async def test_orphan_provider_receipt_and_fragment_gap_fail_closed(
        self,
    ) -> None:
        orphan_document_id = "critic:orphan-receipt"
        bodies, _expected = _parts(2)
        events: list[dict[str, Any]] = []

        async def capture_only(event: dict[str, Any]) -> None:
            events.append(deepcopy(event))

        _delta_capable(capture_only)
        await self._generate(
            bodies,
            document_id=orphan_document_id,
            checkpoint=capture_only,
        )
        orphan = next(
            event
            for event in events
            if event.get("event_kind") == "provider_post"
            and event.get("sequence") == 1
        )
        await self._persist(orphan)
        with self.assertRaisesRegex(OpenRouterError, "gap"):
            await self._load(document_id=orphan_document_id)

        gap_document_id = "critic:fragment-gap"
        gap_bodies, _gap_expected = _parts(2)
        await self._generate(gap_bodies, document_id=gap_document_id)
        async with self.SessionLocal() as session:
            fragment = (
                await session.execute(
                    select(RunArtifact)
                    .where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("lsa2_fragment_%"),
                    )
                    .order_by(RunArtifact.artifact_key.asc())
                    .limit(1)
                )
            ).scalar_one()
            await session.delete(fragment)
            await session.commit()
        with self.assertRaisesRegex(OpenRouterError, "incomplete"):
            await self._load(document_id=gap_document_id)

    async def test_unpromotable_paid_response_is_not_silently_retried(
        self,
    ) -> None:
        document_id = "critic:paid-unpromotable"
        bodies, _expected = _parts(1)
        events: list[dict[str, Any]] = []

        async def capture_only(event: dict[str, Any]) -> None:
            events.append(deepcopy(event))

        _delta_capable(capture_only)
        await self._generate(
            bodies,
            document_id=document_id,
            checkpoint=capture_only,
        )
        receipt = next(
            event
            for event in events
            if event.get("event_kind") == "provider_post"
        )
        receipt["status"] = "response_error"
        receipt["raw_text"] = None
        receipt["usage"] = {}
        receipt["transport"] = {}
        receipt["response"] = {
            "http_status": 200,
            "body_json": {
                "id": "paid-but-unusable",
                "choices": [],
                "usage": {"total_tokens": 17, "cost": 0.01},
            },
        }
        receipt["error"] = {
            "type": "OpenRouterError",
            "message": "provider returned no choices",
        }
        await self._persist(receipt)

        with self.assertRaisesRegex(OpenRouterError, "cannot be promoted"):
            await self._load(document_id=document_id)

    async def test_completed_head_rejects_late_unpromotable_response(
        self,
    ) -> None:
        document_id = "critic:late-paid-response"
        bodies, expected = _parts(1)
        events: list[dict[str, Any]] = []

        async def capture(event: dict[str, Any]) -> None:
            events.append(deepcopy(event))
            await self._persist(event)

        _delta_capable(capture)
        await self._generate(
            bodies,
            document_id=document_id,
            checkpoint=capture,
        )
        provider = next(
            event
            for event in events
            if event.get("event_kind") == "provider_post"
        )
        completed = next(
            event
            for event in events
            if event.get("event_kind") == "structured_continuation_checkpoint"
            and event.get("complete") is True
        )
        late = deepcopy(provider)
        late["event_id"] = uuid.uuid4().hex
        late["logical_call_id"] = uuid.uuid4().hex
        late["sequence"] = 1
        late["status"] = "response_error"
        late["error"] = {
            "type": "OpenRouterResponseContractError",
            "message": "late unusable response",
        }
        request_payload = deepcopy(late["request_payload"])
        request_payload.pop("response_format", None)
        request_payload["messages"] = [
            *MESSAGES,
            {"role": "assistant", "content": expected},
            {"role": "user", "content": "Continue the same document."},
        ]
        late["request_payload"] = request_payload
        late["request_sha256"] = audit_store._stable_json_sha256(
            request_payload
        )
        late["predecessor"] = {
            "document_id": document_id,
            "expected_sequence": 1,
            "latest_sequence": 0,
            "complete": False,
            "document_sha256": completed["head"]["document_sha256"],
            "document_chars": len(expected),
            "document_utf8_bytes": None,
            "expected_overlap_sha256": audit_store.text_sha256(expected[-16:]),
            "expected_overlap_chars": min(16, len(expected)),
        }
        await self._persist(late)

        with self.assertRaisesRegex(OpenRouterError, "after completion"):
            await self._load(document_id=document_id)

    async def test_exact_contract_isolation_and_tamper_fail_closed(self) -> None:
        document_id = "critic:isolated"
        bodies, _expected = _parts(1)
        await self._generate(bodies, document_id=document_id)

        self.assertIsNone(
            await self._load(
                document_id=document_id,
                messages=[{"role": "user", "content": "Другой запрос."}],
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                owner_prompt_version="critic-v4",
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                owner_artifact_key="other_report",
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                source_input={"domain": "other.example"},
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                model="anthropic/claude-fable-5",
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                reasoning_effort="high",
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                temperature=0.1,
            )
        )
        self.assertIsNone(
            await self._load(
                document_id=document_id,
                overlap_chars=17,
            )
        )
        self.assertIsNone(
            await self._load(
                document_id="critic:other-document",
            )
        )

        async with self.SessionLocal() as session:
            receipt = (
                await session.execute(
                    select(RunArtifact)
                    .where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.raw_text.is_not(None),
                    )
                    .limit(1)
                )
            ).scalar_one()
            receipt.raw_text = str(receipt.raw_text) + "tampered"
            await session.commit()
        with self.assertRaisesRegex(OpenRouterError, "digest"):
            await self._load(document_id=document_id)

    async def test_coherently_tampered_contract_reference_fails_closed(
        self,
    ) -> None:
        document_id = "critic:contract-tamper"
        bodies, _expected = _parts(1)
        await self._generate(bodies, document_id=document_id)

        async with self.SessionLocal() as session:
            head = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("lsa2_head_%"),
                    )
                )
            ).scalar_one()
            output = deepcopy(head.output_json)
            stored_input = deepcopy(head.input_json)
            assert isinstance(output, dict)
            assert isinstance(stored_input, dict)
            output["resume_contract_sha256"] = "0" * 64
            # Simulate an attacker/corrupt writer that also updates the local
            # row digest. Exact public-contract matching must still reject it.
            stored_input["row_sha256"] = audit_store._stable_json_sha256(output)
            head.output_json = output
            head.input_json = stored_input
            await session.commit()

        with self.assertRaisesRegex(OpenRouterError, "contract"):
            await self._load(document_id=document_id)

    async def test_coherently_tampered_request_reference_fails_closed(
        self,
    ) -> None:
        document_id = "critic:request-ref-tamper"
        bodies, _expected = _parts(2)
        await self._generate(bodies, document_id=document_id)

        async with self.SessionLocal() as session:
            receipt = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like(
                            "lsa2_receipt_%_000000000001_%"
                        ),
                    )
                )
            ).scalar_one()
            stored_input = deepcopy(receipt.input_json)
            output = deepcopy(receipt.output_json)
            assert isinstance(stored_input, dict)
            assert isinstance(output, dict)
            request_payload = stored_input["request_payload"]
            request_payload["messages"][-2]["content"][
                "document_sha256"
            ] = "0" * 64
            stored_input["stored_request_sha256"] = (
                audit_store._stable_json_sha256(request_payload)
            )
            usage = receipt.usage_json if isinstance(receipt.usage_json, dict) else {}
            stored_input["row_sha256"] = audit_store._stable_json_sha256(
                {
                    "identity": stored_input["stream_identity"],
                    "request_payload": request_payload,
                    "output": output,
                    "raw_text_sha256": output["raw_text_sha256"],
                    "usage": usage,
                    "resume_contract_sha256": stored_input[
                        "resume_contract_sha256"
                    ],
                }
            )
            receipt.input_json = stored_input
            await session.commit()

        with self.assertRaisesRegex(OpenRouterError, "reference"):
            await self._load(document_id=document_id)

    async def test_one_hundred_parts_use_linear_normalized_storage(self) -> None:
        document_id = "critic:linear-100"
        bodies, expected = _parts(100)
        await self._generate(bodies, document_id=document_id)

        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RunArtifact)
                        .where(RunArtifact.run_id == self.run_id)
                        .order_by(RunArtifact.artifact_key.asc())
                    )
                )
                .scalars()
                .all()
            )
        receipts = [row for row in rows if "lsa2_receipt_" in row.artifact_key]
        fragments = [row for row in rows if "lsa2_fragment_" in row.artifact_key]
        heads = [row for row in rows if "lsa2_head_" in row.artifact_key]
        self.assertEqual((len(receipts), len(fragments), len(heads)), (100, 100, 1))
        self.assertTrue(all(row.raw_text is not None for row in receipts))
        self.assertTrue(all(row.raw_text is None for row in [*fragments, *heads]))
        for receipt in receipts[1:]:
            stored_input = receipt.input_json
            assert isinstance(stored_input, dict)
            request_payload = stored_input.get("request_payload")
            assert isinstance(request_payload, dict)
            messages = request_payload.get("messages")
            assert isinstance(messages, list)
            predecessor_ref = messages[-2]["content"]
            self.assertEqual(
                predecessor_ref.get("$aiv_ref"),
                "predecessor.document_text",
            )
            self.assertNotEqual(
                stored_input.get("request_sha256"),
                stored_input.get("stored_request_sha256"),
            )
        self.assertEqual(
            sum(len(str(row.raw_text)) for row in receipts),
            sum(len(body["choices"][0]["message"]["content"]) for body in bodies),
        )

        def row_size(row: RunArtifact) -> int:
            json_fields = (
                row.input_json,
                row.output_json,
                row.usage_json,
            )
            return sum(
                len(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                for value in json_fields
                if value is not None
            ) + len((row.raw_text or "").encode("utf-8"))

        stored_bytes = sum(row_size(row) for row in rows)
        # The bound includes per-POST request/response audit metadata.  A
        # cumulative-checkpoint implementation exceeds it by orders of
        # magnitude because it stores the growing document 100 times.
        self.assertLess(stored_bytes, len(expected.encode("utf-8")) * 12 + 500_000)
        latest = await self._load(document_id=document_id, complete=None)
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest["partial_text"], expected)


if __name__ == "__main__":
    unittest.main()
