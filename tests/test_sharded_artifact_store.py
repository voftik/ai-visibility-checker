from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import Base, Run, RunArtifact, RunStatus
from app.services.long_response import ResponseMode
from app.services import openrouter
from app.services.run_lease import RunLeaseLostError
from app.services.sharded_artifact_store import (
    SHARDED_ARTIFACT_STORE_VERSION,
    ShardedArtifactStoreError,
    create_sharded_artifact_store,
)
from app.services.sharded_document import (
    ShardComposability,
    ShardSpec,
    build_shard_plan,
    create_shard_receipt,
    shard_request,
)


MODEL = "test/provider-model"
SHARD_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": SHARD_SCHEMA["properties"]["items"]["items"]}
    },
    "required": ["items"],
    "additionalProperties": False,
}
GENERATION_CONTRACT = {
    "model": MODEL,
    "system_prompt": "Return one exact JSON shard.",
    "prompt_template": "{{payload}}",
    "parameters": {
        "temperature": 0,
        "reasoning_effort": "high",
        "output_token_policy": "model_max_available",
    },
    "web_policy": {"policy": "forbidden"},
    "schema_name": "test_shard",
}
MERGE_CONTRACT = {
    "algorithm": "flatten_in_order",
    "version": "flatten-v1",
}


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class ShardedArtifactStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "sharded-store.sqlite3"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", echo=False
        )
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
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
        self._session_patch = patch(
            "app.services.sharded_artifact_store.SessionLocal",
            self.SessionLocal,
        )
        self._lease_patch = patch(
            "app.services.sharded_artifact_store.assert_run_lease",
            new=AsyncMock(),
        )
        self._session_patch.start()
        self.assert_lease = self._lease_patch.start()
        self.plan = build_shard_plan(
            document_id="report:example",
            shards=[
                ShardSpec(
                    shard_id=f"section-{index}",
                    payload={"key": f"k{index}"},
                )
                for index in range(3)
            ],
            shard_schema=SHARD_SCHEMA,
            document_schema=DOCUMENT_SCHEMA,
            plan_version="sections-v1",
            merge_version="flatten-v1",
            generation_contract=GENERATION_CONTRACT,
            merge_contract=MERGE_CONTRACT,
            response_mode=ResponseMode.PARTITIONED,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
        )
        self.store = create_sharded_artifact_store(
            run_id=self.run_id,
            stage_key="analysis_report",
            owner_artifact_key="final_report_shards",
            model=MODEL,
            owner_prompt_version="final-report-v24",
            plan=self.plan,
        )

    async def asyncTearDown(self) -> None:
        self._lease_patch.stop()
        self._session_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    def _request(self, index: int = 0):
        return shard_request(self.plan, index)

    def _value(self, request, *, suffix: str = "") -> dict:
        return {
            "items": [
                {
                    "key": str(request.payload["key"]),
                    "value": f"value-{request.index}{suffix}",
                }
            ]
        }

    def _event(
        self,
        request,
        *,
        ordinal: int = 1,
        value: dict | None = None,
        raw_text: str | None = None,
        status: str = "accepted",
    ) -> dict:
        resolved_value = value if value is not None else self._value(request)
        if raw_text is None and status == "accepted":
            raw_text = json.dumps(resolved_value, ensure_ascii=False)
        chat_arguments = self.store.provider_chat_arguments(request)
        payload = {
            "model": chat_arguments["model"],
            "messages": chat_arguments["messages"],
            "temperature": chat_arguments["temperature"],
            "plugins": [{"id": "web", "enabled": False}],
            "tool_choice": "none",
            "reasoning": {
                "effort": chat_arguments["reasoning_effort"],
                "exclude": True,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": chat_arguments["schema_name"],
                    "strict": True,
                    "schema": request.shard_schema,
                },
            },
            "max_completion_tokens": 65_536,
        }
        request_body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "version": "aiv-openrouter-physical-post-audit-v1",
            "event_id": f"{ordinal:032x}",
            "event_kind": "provider_post",
            "logical_call_id": f"logical-{ordinal}",
            "document_id": request.document_id,
            "sequence": request.index,
            "attempt": 1,
            "status": status,
            "model": MODEL,
            "request_payload": payload,
            "request_sha256": _stable_sha256(payload),
            "request_body_utf8_bytes": len(request_body),
            "request_body_encoding": "canonical-json-utf8-v1",
            "response": {"http_status": 200, "body_json": resolved_value},
            "raw_text": raw_text,
            "citations": [],
            "annotations": [],
            "request_policy": {},
            "web_attestation": {},
            "router_metadata": {},
            "usage": {
                "completion_tokens": 100,
                "_aiv_output_envelope": {
                    "policy": "model_max_available",
                    "requested_model": MODEL,
                    "effective_max_completion_tokens": 65_536,
                    "request_estimate": {
                        "request_sha256": _stable_sha256(payload),
                    },
                },
            },
            "transport": {
                "http_status": 200,
                "output_complete": status == "accepted",
                "output_limited": False,
            },
            "resume_contract": self.store.provider_audit_context(request)[
                "resume_contract"
            ],
            "error": None if status == "accepted" else {"message": "bad"},
            "partial_text": "",
            "manifest": None,
            "aggregate_usage": {},
            "call_records": [],
        }

    def _rehash_physical_request(self, event: dict) -> None:
        request_body = json.dumps(
            event["request_payload"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(request_body).hexdigest()
        event["request_sha256"] = digest
        event["request_body_utf8_bytes"] = len(request_body)
        event["usage"]["_aiv_output_envelope"]["request_estimate"][
            "request_sha256"
        ] = digest

    async def _accepted(self, request, *, ordinal: int = 1, suffix: str = ""):
        value = self._value(request, suffix=suffix)
        event = self._event(request, ordinal=ordinal, value=value)
        await self.store.persist_provider_event(request, event)
        generated = await self.store.promote_accepted_provider_response(request)
        self.assertIsNotNone(generated)
        return generated

    async def test_generate_resumes_paid_post_without_duplicate_call(self) -> None:
        request = self._request()
        calls = 0

        async def provider(checkpoint, context):
            nonlocal calls
            calls += 1
            self.assertEqual(context, self.store.provider_audit_context(request))
            event = self._event(request)
            await checkpoint(event)
            return SimpleNamespace(
                text=event["raw_text"], parsed=self._value(request)
            )

        first = await self.store.generate_or_resume(request, provider)
        second = await self.store.generate_or_resume(request, provider)

        self.assertEqual(calls, 1)
        self.assertEqual(first.value, second.value)
        self.assertEqual(first.metadata, second.metadata)
        self.assertTrue(
            second.provider_audit.receipt_ref.startswith(
                "run-artifact:aiv_sdpa_"
            )
        )

    async def test_callbacks_accept_the_exact_openrouter_chat_event(self) -> None:
        request = self._request()
        value = self._value(request)
        raw_text = json.dumps(value, ensure_ascii=False)

        class Response:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "id": "response-test",
                    "model": MODEL,
                    "provider": "test-provider",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "native_finish_reason": "stop",
                            "message": {"content": raw_text},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                    },
                }

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                return Response()

        async def provider(checkpoint, context):
            return await openrouter.chat(
                **self.store.provider_chat_arguments(request),
                audit_checkpoint=checkpoint,
                audit_context=context,
                retry_transport_errors=False,
            )

        envelope = {
            "version": "aiv-openrouter-output-envelope-v1",
            "policy": "model_max_available",
            "requested_model": MODEL,
            "resolution": "provider_metadata",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=Client(),
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
            patch(
                "app.services.openrouter.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
        ):
            generated = await self.store.generate_or_resume(request, provider)

        self.assertEqual(generated.value, value)
        await self.store.verify_provider_audit(
            generated.provider_audit, request
        )

    async def test_public_physical_body_bytes_equal_audited_request(self) -> None:
        request = self._request()
        event = self._event(request)
        body = self.store.provider_request_utf8_bytes(
            request,
            max_completion_tokens=65_536,
        )
        self.assertIsInstance(body, bytes)
        self.assertEqual(json.loads(body), event["request_payload"])
        self.assertEqual(hashlib.sha256(body).hexdigest(), event["request_sha256"])

    async def test_planned_requests_are_exact_later_accepted_identities(self) -> None:
        planned = self.store.planned_requests()
        self.assertEqual(
            tuple(request.as_dict() for request in planned),
            tuple(self._request(index).as_dict() for index in range(3)),
        )
        for index, request in enumerate(planned, start=1):
            event = self._event(request, ordinal=index)
            await self.store.persist_provider_event(request, event)
            accepted = await self.store.promote_accepted_provider_response(
                request
            )
            self.assertEqual(
                accepted.provider_audit.logical_request_sha256,
                request.request_sha256,
            )

    async def test_paid_post_survives_crash_before_provider_return(self) -> None:
        request = self._request()

        async def crash_after_checkpoint(checkpoint, _context):
            await checkpoint(self._event(request))
            raise RuntimeError("worker crashed after durable callback")

        with self.assertRaisesRegex(RuntimeError, "worker crashed"):
            await self.store.generate_or_resume(
                request, crash_after_checkpoint
            )

        async def must_not_call(_checkpoint, _context):
            raise AssertionError("paid provider POST must be promoted")

        recovered = await self.store.generate_or_resume(request, must_not_call)
        self.assertEqual(recovered.value, self._value(request))
        self.assertEqual(
            recovered.metadata["provider_event_id"],
            self._event(request)["event_id"],
        )

    async def test_accepted_event_requires_exact_schema_valid_json(self) -> None:
        request = self._request()
        trailing = self._event(
            request,
            raw_text=json.dumps(self._value(request)) + " trailing caveat",
        )
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "one exact JSON value"
        ):
            await self.store.persist_provider_event(request, trailing)

        wrong_shape = self._event(request, value={"unexpected": True})
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "violates the shard schema"
        ):
            await self.store.persist_provider_event(request, wrong_shape)

    async def test_physical_request_digest_mismatch_is_rejected(self) -> None:
        request = self._request()
        event = self._event(request)
        event["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "request digest mismatch"
        ):
            await self.store.persist_provider_event(request, event)

    async def test_physical_request_wire_byte_count_mismatch_is_rejected(
        self,
    ) -> None:
        request = self._request()
        event = self._event(request)
        event["request_body_utf8_bytes"] += 1
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "exact wire-body evidence is invalid"
        ):
            await self.store.persist_provider_event(request, event)

    async def test_tampered_request_object_cannot_reuse_logical_digest(self) -> None:
        request = self._request()
        tampered = replace(request, payload={"key": "substituted"})
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "differs from its bound plan"
        ):
            self.store.provider_chat_arguments(tampered)

    async def test_prompt_schema_temperature_and_web_drift_are_rejected(
        self,
    ) -> None:
        request = self._request()
        mutations = {
            "prompt": lambda payload: payload["messages"].__setitem__(
                1, {"role": "user", "content": "different prompt"}
            ),
            "schema": lambda payload: payload["response_format"][
                "json_schema"
            ].__setitem__("schema", {"type": "object"}),
            "temperature": lambda payload: payload.__setitem__(
                "temperature", 0.9
            ),
            "reasoning": lambda payload: payload["reasoning"].__setitem__(
                "effort", "low"
            ),
            "web": lambda payload: payload.__setitem__(
                "tool_choice", "auto"
            ),
        }
        for ordinal, (label, mutate) in enumerate(
            mutations.items(), start=20
        ):
            with self.subTest(label=label):
                event = self._event(request, ordinal=ordinal)
                mutate(event["request_payload"])
                self._rehash_physical_request(event)
                with self.assertRaisesRegex(
                    ShardedArtifactStoreError,
                    "does not match the shard generation contract",
                ):
                    await self.store.persist_provider_event(request, event)

    async def test_payload_braces_are_data_not_template_placeholders(self) -> None:
        brace_plan = build_shard_plan(
            document_id="report:braces",
            shards=[
                ShardSpec(
                    shard_id="brace-data",
                    payload={"key": "{{customer_name}}"},
                )
            ],
            shard_schema=SHARD_SCHEMA,
            document_schema=DOCUMENT_SCHEMA,
            plan_version="braces-v1",
            merge_version="flatten-v1",
            generation_contract=GENERATION_CONTRACT,
            merge_contract=MERGE_CONTRACT,
            response_mode=ResponseMode.PARTITIONED,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
        )
        request = shard_request(brace_plan, 0)
        brace_store = create_sharded_artifact_store(
            run_id=self.run_id,
            stage_key="analysis_report",
            owner_artifact_key="brace_shards",
            model=MODEL,
            owner_prompt_version="brace-v1",
            plan=brace_plan,
        )
        arguments = brace_store.provider_chat_arguments(request)
        self.assertIn("{{customer_name}}", arguments["messages"][1]["content"])

    async def test_model_max_envelope_must_bind_the_physical_request(self) -> None:
        request = self._request()
        event = self._event(request)
        event["usage"]["_aiv_output_envelope"][
            "effective_max_completion_tokens"
        ] = 8_192
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "model-max envelope is inconsistent"
        ):
            await self.store.persist_provider_event(request, event)

    async def test_transport_failure_without_usage_envelope_is_still_audited(
        self,
    ) -> None:
        request = self._request()
        event = self._event(
            request,
            status="transport_error",
            raw_text=None,
        )
        event["usage"] = {}
        event["response"] = {}
        await self.store.persist_provider_event(request, event)
        self.assertIsNone(
            await self.store.promote_accepted_provider_response(request)
        )
        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.artifact_key.like("aiv_sdpa_%"),
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].output_json["status"], "transport_error")

    async def test_ambiguous_accepted_posts_fail_closed(self) -> None:
        request = self._request()
        await self.store.persist_provider_event(request, self._event(request, ordinal=1))
        await self.store.persist_provider_event(request, self._event(request, ordinal=2))
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "Ambiguous accepted"
        ):
            await self.store.promote_accepted_provider_response(request)

    async def test_provider_verifier_detects_raw_mutation(self) -> None:
        request = self._request()
        generated = await self._accepted(request)
        artifact_key = generated.provider_audit.receipt_ref.removeprefix(
            "run-artifact:"
        )
        async with self.SessionLocal() as session:
            await session.execute(
                update(RunArtifact)
                .where(
                    RunArtifact.run_id == self.run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
                .values(raw_text='{"items":[]}')
            )
            await session.commit()
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "raw response was mutated"
        ):
            await self.store.verify_provider_audit(
                generated.provider_audit, request
            )

    async def test_provider_verifier_detects_error_column_mutation(
        self,
    ) -> None:
        request = self._request()
        generated = await self._accepted(request)
        artifact_key = generated.provider_audit.receipt_ref.removeprefix(
            "run-artifact:"
        )
        async with self.SessionLocal() as session:
            await session.execute(
                update(RunArtifact)
                .where(
                    RunArtifact.run_id == self.run_id,
                    RunArtifact.artifact_key == artifact_key,
                )
                .values(error_message="substituted failure")
            )
            await session.commit()
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "error evidence was mutated"
        ):
            await self.store.verify_provider_audit(
                generated.provider_audit, request
            )

    async def test_large_response_round_trips_without_store_cap(self) -> None:
        request = self._request()
        large_value = {
            "items": [
                {"key": "large", "value": "Ж" * 300_000}
            ]
        }
        event = self._event(request, value=large_value)
        await self.store.persist_provider_event(request, event)
        recovered = await self.store.promote_accepted_provider_response(request)
        self.assertEqual(recovered.value, large_value)
        self.assertEqual(recovered.raw_text, event["raw_text"])

    async def test_ordered_receipts_round_trip_through_atomic_head(self) -> None:
        receipts = []
        predecessor = None
        for index in range(3):
            request = self._request(index)
            generated = await self._accepted(request, ordinal=index + 1)
            receipt = create_shard_receipt(
                plan=self.plan,
                request=request,
                generated=generated,
                predecessor_receipt_sha256=predecessor,
            )
            ack = await self.store.save_receipt(receipt, predecessor)
            self.assertEqual(ack.receipt_sha256, receipt["receipt_sha256"])
            receipts.append(receipt)
            predecessor = receipt["receipt_sha256"]

        loaded = await self.store.load_receipts(
            self.plan.document_id, self.plan.plan_sha256
        )
        self.assertEqual(loaded, receipts)
        self.assertGreaterEqual(self.assert_lease.await_count, 6)

    async def test_stale_cas_fails_without_orphan_receipt(self) -> None:
        request = self._request(0)
        generated = await self._accepted(request)
        receipt = create_shard_receipt(
            plan=self.plan,
            request=request,
            generated=generated,
            predecessor_receipt_sha256=None,
        )
        await self.store.save_receipt(receipt, None)
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "predecessor mismatch"
        ):
            await self.store.save_receipt(receipt, None)

        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.prompt_version
                            == SHARDED_ARTIFACT_STORE_VERSION,
                        )
                    )
                )
                .scalars()
                .all()
            )
        shard_rows = [
            row for row in rows if row.artifact_key.startswith("aiv_sdr_")
        ]
        self.assertEqual(len(shard_rows), 1)

    async def test_concurrent_cas_has_exactly_one_winner(self) -> None:
        request = self._request(0)
        generated = await self._accepted(request)
        receipt = create_shard_receipt(
            plan=self.plan,
            request=request,
            generated=generated,
            predecessor_receipt_sha256=None,
        )
        outcomes = await asyncio.gather(
            self.store.save_receipt(deepcopy(receipt), None),
            self.store.save_receipt(deepcopy(receipt), None),
            return_exceptions=True,
        )
        successes = [item for item in outcomes if not isinstance(item, Exception)]
        failures = [item for item in outcomes if isinstance(item, Exception)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        loaded = await self.store.load_receipts(
            self.plan.document_id, self.plan.plan_sha256
        )
        self.assertEqual(loaded, [receipt])

    async def test_minimal_self_hashed_receipt_cannot_poison_store(self) -> None:
        minimal_core = {
            "document_id": self.plan.document_id,
            "plan_sha256": self.plan.plan_sha256,
            "index": 0,
            "predecessor_receipt_sha256": None,
        }
        digest = _stable_sha256(minimal_core)
        minimal = {
            **minimal_core,
            "receipt_sha256": digest,
            "receipt_id": f"sha256:{digest}",
        }
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "invalid exact shape"
        ):
            await self.store.save_receipt(minimal, None)

    async def test_rehashed_receipt_cannot_change_bound_plan_fields(self) -> None:
        request = self._request(0)
        generated = await self._accepted(request)
        canonical = create_shard_receipt(
            plan=self.plan,
            request=request,
            generated=generated,
            predecessor_receipt_sha256=None,
        )
        mutations = {
            "shard_id": "substituted-shard",
            "shard_count": 99,
            "spec_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "merge_version": "foreign-merge-v9",
            "document_schema_sha256": "3" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                poisoned = deepcopy(canonical)
                poisoned[field] = value
                core = {
                    key: item
                    for key, item in poisoned.items()
                    if key not in {"receipt_sha256", "receipt_id"}
                }
                digest = _stable_sha256(core)
                poisoned["receipt_sha256"] = digest
                poisoned["receipt_id"] = f"sha256:{digest}"
                with self.assertRaisesRegex(
                    ShardedArtifactStoreError,
                    "differs from its bound plan",
                ):
                    await self.store.save_receipt(poisoned, None)

    async def test_missing_receipt_row_is_detected_against_head(self) -> None:
        request = self._request(0)
        generated = await self._accepted(request)
        receipt = create_shard_receipt(
            plan=self.plan,
            request=request,
            generated=generated,
            predecessor_receipt_sha256=None,
        )
        await self.store.save_receipt(receipt, None)
        async with self.SessionLocal() as session:
            await session.execute(
                delete(RunArtifact).where(
                    RunArtifact.run_id == self.run_id,
                    RunArtifact.artifact_key.like("aiv_sdr_%"),
                )
            )
            await session.commit()
        with self.assertRaisesRegex(
            ShardedArtifactStoreError, "do not match the CAS head"
        ):
            await self.store.load_receipts(
                self.plan.document_id, self.plan.plan_sha256
            )

    async def test_head_and_receipt_secondary_column_mutation_fails_closed(
        self,
    ) -> None:
        request = self._request(0)
        generated = await self._accepted(request)
        receipt = create_shard_receipt(
            plan=self.plan,
            request=request,
            generated=generated,
            predecessor_receipt_sha256=None,
        )
        await self.store.save_receipt(receipt, None)
        mutations = (
            ("aiv_sdh_%", "raw_text", "tampered", "CAS head is corrupt"),
            (
                "aiv_sdh_%",
                "usage_json",
                {"tampered": True},
                "CAS head is corrupt",
            ),
            (
                "aiv_sdh_%",
                "error_message",
                "tampered",
                "CAS head is corrupt",
            ),
            (
                "aiv_sdr_%",
                "usage_json",
                {"tampered": True},
                "secondary evidence was mutated",
            ),
            (
                "aiv_sdr_%",
                "error_message",
                "tampered",
                "secondary evidence was mutated",
            ),
        )
        for prefix, column, value, message in mutations:
            with self.subTest(prefix=prefix, column=column):
                async with self.SessionLocal() as session:
                    await session.execute(
                        update(RunArtifact)
                        .where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.artifact_key.like(prefix),
                        )
                        .values(**{column: value})
                    )
                    await session.commit()
                with self.assertRaisesRegex(
                    ShardedArtifactStoreError, message
                ):
                    await self.store.load_receipts(
                        self.plan.document_id, self.plan.plan_sha256
                    )
                async with self.SessionLocal() as session:
                    await session.execute(
                        update(RunArtifact)
                        .where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.artifact_key.like(prefix),
                        )
                        .values(**{column: None})
                    )
                    await session.commit()

    async def test_lost_run_lease_prevents_provider_persistence(self) -> None:
        request = self._request()
        self.assert_lease.side_effect = RunLeaseLostError("stale worker")
        with self.assertRaisesRegex(RunLeaseLostError, "stale worker"):
            await self.store.persist_provider_event(request, self._event(request))


if __name__ == "__main__":
    unittest.main()
