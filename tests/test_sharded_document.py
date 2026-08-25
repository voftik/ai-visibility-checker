from __future__ import annotations

import asyncio
import hashlib
import json
import time
import unittest
from copy import deepcopy
from dataclasses import replace
from typing import Any

from app.services.long_response import ResponseMode, text_sha256
from app.services.sharded_document import (
    AtomicShardingUnsupportedError,
    GeneratedShard,
    ProviderAuditBinding,
    ShardComposability,
    ShardMergeError,
    ShardPlanError,
    ShardReceiptError,
    ShardSchemaError,
    ShardedDocumentLivenessError,
    ShardRequest,
    ShardSaveAck,
    ShardSpec,
    build_shard_plan,
    compose_sharded_document,
    create_shard_receipt,
    merge_shard_receipts,
    shard_request,
    verify_shard_receipts,
)


SHARD_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "integer"},
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
        "items": {
            "type": "array",
            "items": SHARD_SCHEMA["properties"]["items"]["items"],
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

GENERATION_CONTRACT = {
    "model": "test/provider-model",
    "system_prompt": "Return exactly one independently valid JSON shard.",
    "prompt_template": "Process shard {{shard_id}} with {{payload}}.",
    "parameters": {"temperature": 0, "reasoning_effort": "high"},
    "web_policy": {"enabled": False},
    "schema_name": "test_shard",
}

MERGE_CONTRACT = {
    "algorithm": "flatten_items_in_exact_plan_order",
    "version": "flatten-items-v1",
}


def _specs(count: int) -> list[ShardSpec]:
    return [
        ShardSpec(
            shard_id=f"section-{index:04d}",
            payload={"key": f"key-{index:04d}", "value": index},
        )
        for index in range(count)
    ]


def _merge(values: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "items": [
            deepcopy(item)
            for shard in values
            for item in shard["items"]
        ]
    }


def _provider_audit(
    request: ShardRequest,
    raw_text: str,
) -> ProviderAuditBinding:
    physical_request_sha256 = hashlib.sha256(
        f"physical:{request.request_sha256}".encode()
    ).hexdigest()
    receipt_sha256 = hashlib.sha256(
        (
            f"receipt:{request.request_sha256}:"
            f"{text_sha256(raw_text)}"
        ).encode()
    ).hexdigest()
    return ProviderAuditBinding(
        event_id=f"event:{request.shard_id}",
        receipt_ref=f"artifact:{request.plan_sha256}:{request.index}",
        receipt_sha256=receipt_sha256,
        physical_request_sha256=physical_request_sha256,
        logical_request_sha256=request.request_sha256,
        raw_text_sha256=text_sha256(raw_text),
    )


def _generated(request: ShardRequest) -> GeneratedShard:
    value = {
        "items": [
            {
                "key": request.payload["key"],
                "value": request.payload["value"],
            }
        ]
    }
    raw_text = json.dumps(value, ensure_ascii=False)
    return GeneratedShard(
        value=value,
        raw_text=raw_text,
        provider_audit=_provider_audit(request, raw_text),
        metadata={"provider_receipt_ref": f"post:{request.shard_id}"},
    )


async def _generate_async(request: ShardRequest) -> GeneratedShard:
    return _generated(request)


async def _verify_provider_audit(
    binding: ProviderAuditBinding,
    request: ShardRequest,
) -> None:
    expected_physical = hashlib.sha256(
        f"physical:{request.request_sha256}".encode()
    ).hexdigest()
    expected_receipt = hashlib.sha256(
        (
            f"receipt:{request.request_sha256}:"
            f"{binding.raw_text_sha256}"
        ).encode()
    ).hexdigest()
    if (
        binding.logical_request_sha256 != request.request_sha256
        or binding.physical_request_sha256 != expected_physical
        or binding.receipt_sha256 != expected_receipt
        or binding.receipt_ref
        != f"artifact:{request.plan_sha256}:{request.index}"
        or binding.event_id != f"event:{request.shard_id}"
    ):
        raise AssertionError("durable provider receipt mismatch")


class _MemoryReceiptStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.loads: list[tuple[str, str]] = []
        self.saves: list[str] = []

    async def load(self, document_id: str, plan_sha256: str) -> list[dict[str, Any]]:
        self.loads.append((document_id, plan_sha256))
        return deepcopy(self.rows)

    async def save(
        self,
        receipt: dict[str, Any],
        expected_predecessor: str | None,
    ) -> ShardSaveAck:
        current_head = self.rows[-1]["receipt_sha256"] if self.rows else None
        if current_head != expected_predecessor:
            raise AssertionError("compare-and-set head mismatch")
        receipt_id = str(receipt["receipt_id"])
        existing = next(
            (row for row in self.rows if row["receipt_id"] == receipt_id),
            None,
        )
        if existing is not None:
            if existing != receipt:
                raise AssertionError("content-address collision")
            return ShardSaveAck(
                document_id=receipt["document_id"],
                plan_sha256=receipt["plan_sha256"],
                index=receipt["index"],
                receipt_id=receipt_id,
                receipt_sha256=receipt["receipt_sha256"],
                predecessor_receipt_sha256=expected_predecessor,
            )
        if any(row["index"] == receipt["index"] for row in self.rows):
            raise AssertionError("immutable sequence collision")
        self.rows.append(deepcopy(receipt))
        self.saves.append(receipt_id)
        return ShardSaveAck(
            document_id=receipt["document_id"],
            plan_sha256=receipt["plan_sha256"],
            index=receipt["index"],
            receipt_id=receipt_id,
            receipt_sha256=receipt["receipt_sha256"],
            predecessor_receipt_sha256=expected_predecessor,
        )


class ShardedDocumentHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def _compose(
        self,
        *,
        count: int,
        store: _MemoryReceiptStore,
        generate: Any = _generate_async,
        merge: Any = _merge,
        deadline_seconds: float | None = None,
        response_mode: ResponseMode | str = ResponseMode.PARTITIONED,
        plan_version: str = "sections-v1",
        merge_version: str = "flatten-items-v1",
        shard_schema: dict[str, Any] = SHARD_SCHEMA,
        document_schema: dict[str, Any] = DOCUMENT_SCHEMA,
    ) -> Any:
        return await compose_sharded_document(
            document_id="report:sharded",
            shards=_specs(count),
            shard_schema=shard_schema,
            document_schema=document_schema,
            plan_version=plan_version,
            merge_version=merge_version,
            generation_contract=GENERATION_CONTRACT,
            merge_contract={**MERGE_CONTRACT, "version": merge_version},
            response_mode=response_mode,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
            generate_shard=generate,
            verify_provider_audit=_verify_provider_audit,
            merge_shards=merge,
            load_receipts=store.load,
            save_receipt=store.save,
            deadline_seconds=deadline_seconds,
            empty_plan_reason=("no source records" if count == 0 else None),
        )

    async def test_unbounded_plan_generates_many_independent_shards(self) -> None:
        store = _MemoryReceiptStore()
        requests: list[dict[str, Any]] = []

        async def generate(request: ShardRequest) -> GeneratedShard:
            snapshot = request.as_dict()
            requests.append(snapshot)
            self.assertNotIn("previous", snapshot)
            self.assertNotIn("receipts", snapshot)
            self.assertNotIn("document", snapshot)
            self.assertEqual(snapshot["payload"]["value"], request.index)
            return _generated(request)

        result = await self._compose(
            count=1_000,
            store=store,
            generate=generate,
        )

        self.assertEqual(len(requests), 1_000)
        self.assertEqual(len(store.rows), 1_000)
        self.assertEqual(result.generated_shards, 1_000)
        self.assertEqual(result.resumed_shards, 0)
        self.assertEqual(result.manifest["covered_shard_count"], 1_000)
        self.assertEqual(
            [item["value"] for item in result.document["items"]],
            list(range(1_000)),
        )

    async def test_total_document_length_has_no_harness_cap(self) -> None:
        store = _MemoryReceiptStore()
        chunk_schema = {
            "type": "object",
            "properties": {"chunk": {"type": "string"}},
            "required": ["chunk"],
            "additionalProperties": False,
        }
        document_schema = {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["chunks"],
            "additionalProperties": False,
        }

        async def generate(request: ShardRequest) -> GeneratedShard:
            value = {"chunk": chr(65 + request.index % 26) * 32_768}
            raw_text = json.dumps(value)
            return GeneratedShard(
                value=value,
                raw_text=raw_text,
                provider_audit=_provider_audit(request, raw_text),
            )

        result = await compose_sharded_document(
            document_id="report:large-sharded",
            shards=_specs(64),
            shard_schema=chunk_schema,
            document_schema=document_schema,
            plan_version="large-sections-v1",
            merge_version="join-chunks-v1",
            generation_contract=GENERATION_CONTRACT,
            merge_contract={
                "algorithm": "collect_chunks_in_exact_plan_order",
                "version": "join-chunks-v1",
            },
            response_mode=ResponseMode.PARTITIONED,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
            generate_shard=generate,
            verify_provider_audit=_verify_provider_audit,
            merge_shards=lambda values: {
                "chunks": [value["chunk"] for value in values]
            },
            load_receipts=store.load,
            save_receipt=store.save,
        )

        self.assertEqual(len(result.document["chunks"]), 64)
        self.assertEqual(
            sum(len(chunk) for chunk in result.document["chunks"]),
            64 * 32_768,
        )
        self.assertGreater(result.manifest["document_json_utf8_bytes"], 2_000_000)

    async def test_restart_resumes_contiguous_prefix_without_regeneration(
        self,
    ) -> None:
        store = _MemoryReceiptStore()
        first_calls: list[int] = []

        async def crash_on_third(request: ShardRequest) -> GeneratedShard:
            first_calls.append(request.index)
            if request.index == 2:
                raise RuntimeError("simulated worker crash")
            return _generated(request)

        with self.assertRaisesRegex(RuntimeError, "worker crash"):
            await self._compose(
                count=5,
                store=store,
                generate=crash_on_third,
            )
        self.assertEqual(first_calls, [0, 1, 2])
        self.assertEqual([row["index"] for row in store.rows], [0, 1])

        resumed_calls: list[int] = []

        async def resume(request: ShardRequest) -> GeneratedShard:
            resumed_calls.append(request.index)
            return _generated(request)

        result = await self._compose(
            count=5,
            store=store,
            generate=resume,
        )
        self.assertEqual(resumed_calls, [2, 3, 4])
        self.assertEqual(result.resumed_shards, 2)
        self.assertEqual(result.generated_shards, 3)
        self.assertEqual(len(result.receipts), 5)

        async def must_not_generate(_request: ShardRequest) -> GeneratedShard:
            raise AssertionError("completed shards must not be regenerated")

        replay = await self._compose(
            count=5,
            store=store,
            generate=must_not_generate,
        )
        self.assertEqual(replay.resumed_shards, 5)
        self.assertEqual(replay.generated_shards, 0)
        self.assertEqual(replay.document, result.document)

    async def test_save_failure_stops_before_the_next_provider_call(self) -> None:
        store = _MemoryReceiptStore()
        calls: list[int] = []

        async def generate(request: ShardRequest) -> GeneratedShard:
            calls.append(request.index)
            return _generated(request)

        async def fail_second_save(
            receipt: dict[str, Any],
            predecessor: str | None,
        ) -> ShardSaveAck:
            if receipt["index"] == 1:
                raise RuntimeError("database unavailable")
            return await store.save(receipt, predecessor)

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            await compose_sharded_document(
                document_id="report:sharded",
                shards=_specs(4),
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
                generate_shard=generate,
                verify_provider_audit=_verify_provider_audit,
                merge_shards=_merge,
                load_receipts=store.load,
                save_receipt=fail_second_save,
            )

        self.assertEqual(calls, [0, 1])
        self.assertEqual([row["index"] for row in store.rows], [0])

    async def test_mismatched_cas_ack_fails_before_next_provider_call(
        self,
    ) -> None:
        store = _MemoryReceiptStore()
        calls: list[int] = []

        async def generate(request: ShardRequest) -> GeneratedShard:
            calls.append(request.index)
            return _generated(request)

        async def lying_save(
            receipt: dict[str, Any],
            predecessor: str | None,
        ) -> ShardSaveAck:
            return replace(
                await store.save(receipt, predecessor),
                receipt_sha256="0" * 64,
            )

        with self.assertRaisesRegex(ShardReceiptError, "mismatched CAS"):
            await compose_sharded_document(
                document_id="report:sharded",
                shards=_specs(3),
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
                generate_shard=generate,
                verify_provider_audit=_verify_provider_audit,
                merge_shards=_merge,
                load_receipts=store.load,
                save_receipt=lying_save,
            )
        self.assertEqual(calls, [0])

    async def test_missing_duplicate_and_reordered_receipts_fail_closed(
        self,
    ) -> None:
        store = _MemoryReceiptStore()
        result = await self._compose(count=3, store=store)
        original = list(result.receipts)
        cases = {
            "missing middle": [original[0], original[2]],
            "duplicate": [original[0], original[0]],
            "reordered": [original[1], original[0], original[2]],
        }
        for label, rows in cases.items():
            with self.subTest(label=label):
                corrupted = _MemoryReceiptStore()
                corrupted.rows = deepcopy(rows)
                with self.assertRaisesRegex(
                    ShardReceiptError,
                    "missing|duplicate|reordered",
                ):
                    await self._compose(count=3, store=corrupted)

    async def test_tampered_content_and_raw_text_fail_closed(self) -> None:
        store = _MemoryReceiptStore()
        result = await self._compose(count=2, store=store)

        content_tamper = _MemoryReceiptStore()
        content_tamper.rows = deepcopy(list(result.receipts))
        content_tamper.rows[0]["content"]["items"][0]["value"] = 9000
        with self.assertRaisesRegex(ShardReceiptError, "content digest"):
            await self._compose(count=2, store=content_tamper)

        raw_tamper = _MemoryReceiptStore()
        raw_tamper.rows = deepcopy(list(result.receipts))
        raw_tamper.rows[0]["raw_text"] += " "
        with self.assertRaisesRegex(ShardReceiptError, "raw text digest"):
            await self._compose(count=2, store=raw_tamper)

    async def test_wrong_plan_or_schema_cannot_reuse_receipts(self) -> None:
        store = _MemoryReceiptStore()
        await self._compose(count=2, store=store)

        with self.assertRaisesRegex(ShardReceiptError, "identity mismatch"):
            await self._compose(
                count=2,
                store=store,
                merge_version="different-merge-v2",
            )

        changed_schema = deepcopy(SHARD_SCHEMA)
        changed_schema["properties"]["items"]["minItems"] = 1
        with self.assertRaisesRegex(ShardReceiptError, "identity mismatch"):
            await self._compose(
                count=2,
                store=store,
                shard_schema=changed_schema,
            )

    async def test_invalid_generated_shard_is_never_saved(self) -> None:
        store = _MemoryReceiptStore()

        async def invalid(request: ShardRequest) -> GeneratedShard:
            value = {"items": [{"key": "x"}]}
            raw_text = json.dumps(value)
            return GeneratedShard(
                value=value,
                raw_text=raw_text,
                provider_audit=_provider_audit(request, raw_text),
            )

        with self.assertRaisesRegex(ShardSchemaError, "violates"):
            await self._compose(count=1, store=store, generate=invalid)
        self.assertEqual(store.rows, [])

    async def test_raw_text_must_be_one_exact_matching_json_value(self) -> None:
        for raw_text, expected in (
            ('{"items":[]} trailing caveat', "one exact JSON"),
            ('{"items":[],"items":[]}', "one exact JSON"),
            ('{"items":[]}', "raw_text/value mismatch"),
        ):
            with self.subTest(raw_text=raw_text):
                store = _MemoryReceiptStore()

                async def generate(request: ShardRequest) -> GeneratedShard:
                    value = _generated(request).value
                    if raw_text == '{"items":[]}':
                        value = {
                            "items": [
                                {"key": request.shard_id, "value": request.index}
                            ]
                        }
                    return GeneratedShard(
                        value=value,
                        raw_text=raw_text,
                        provider_audit=_provider_audit(request, raw_text),
                    )

                with self.assertRaisesRegex(ShardSchemaError, expected):
                    await self._compose(
                        count=1,
                        store=store,
                        generate=generate,
                    )
                self.assertEqual(store.rows, [])

    async def test_non_deterministic_merge_fails_closed(self) -> None:
        store = _MemoryReceiptStore()
        counter = 0

        def non_deterministic(values: tuple[Any, ...]) -> dict[str, Any]:
            nonlocal counter
            counter += 1
            document = _merge(values)
            document["items"][0]["value"] = counter
            return document

        with self.assertRaisesRegex(ShardMergeError, "non-deterministic"):
            await self._compose(
                count=1,
                store=store,
                merge=non_deterministic,
            )
        self.assertEqual(len(store.rows), 1)

    async def test_merged_document_schema_is_authoritative(self) -> None:
        store = _MemoryReceiptStore()

        def wrong_merge(_values: tuple[Any, ...]) -> dict[str, Any]:
            return {"wrong": []}

        with self.assertRaisesRegex(ShardSchemaError, "merged document"):
            await self._compose(count=2, store=store, merge=wrong_merge)

    async def test_atomic_and_continuable_modes_are_explicitly_rejected(
        self,
    ) -> None:
        store = _MemoryReceiptStore()
        with self.assertRaisesRegex(
            AtomicShardingUnsupportedError,
            "Atomic decisions cannot be sharded",
        ):
            await self._compose(
                count=1,
                store=store,
                response_mode=ResponseMode.ATOMIC,
            )
        with self.assertRaisesRegex(ShardPlanError, "partitioned mode only"):
            await self._compose(
                count=1,
                store=store,
                response_mode=ResponseMode.CONTINUABLE_DOCUMENT,
            )

    async def test_deadline_is_liveness_guard_not_a_shard_cap(self) -> None:
        store = _MemoryReceiptStore()

        async def stuck(_request: ShardRequest) -> GeneratedShard:
            await asyncio.sleep(1)
            raise AssertionError("deadline did not cancel the shard")

        with self.assertRaisesRegex(
            ShardedDocumentLivenessError,
            "deadline expired",
        ):
            await self._compose(
                count=1,
                store=store,
                generate=stuck,
                deadline_seconds=0.01,
            )
        self.assertEqual(store.rows, [])

    async def test_deadline_also_bounds_code_owned_merge(self) -> None:
        store = _MemoryReceiptStore()

        def slow_merge(values: tuple[Any, ...]) -> dict[str, Any]:
            time.sleep(0.1)
            return _merge(values)

        with self.assertRaisesRegex(
            ShardedDocumentLivenessError,
            "deadline expired",
        ):
            await self._compose(
                count=1,
                store=store,
                merge=slow_merge,
                deadline_seconds=0.02,
            )

    async def test_resumed_provider_receipt_is_verified_again(self) -> None:
        store = _MemoryReceiptStore()
        await self._compose(count=1, store=store)

        async def revoked_audit(
            _binding: ProviderAuditBinding,
            _request: ShardRequest,
        ) -> None:
            raise RuntimeError("provider receipt revoked")

        with self.assertRaisesRegex(RuntimeError, "receipt revoked"):
            await compose_sharded_document(
                document_id="report:sharded",
                shards=_specs(1),
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
                generate_shard=_generate_async,
                verify_provider_audit=revoked_audit,
                merge_shards=_merge,
                load_receipts=store.load,
                save_receipt=store.save,
            )

    async def test_sync_provider_callback_is_rejected(self) -> None:
        store = _MemoryReceiptStore()
        with self.assertRaisesRegex(ShardPlanError, "must be async"):
            await compose_sharded_document(
                document_id="report:sharded",
                shards=_specs(1),
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
                generate_shard=_generated,
                verify_provider_audit=_verify_provider_audit,
                merge_shards=_merge,
                load_receipts=store.load,
                save_receipt=store.save,
            )

    async def test_empty_code_owned_plan_merges_without_generation(self) -> None:
        store = _MemoryReceiptStore()

        async def must_not_generate(_request: ShardRequest) -> GeneratedShard:
            raise AssertionError("empty plan must not generate")

        result = await self._compose(
            count=0,
            store=store,
            generate=must_not_generate,
        )

        self.assertEqual(result.document, {"items": []})
        self.assertEqual(result.receipts, ())
        self.assertEqual(result.generated_shards, 0)
        self.assertTrue(result.manifest["complete"])


class ShardedDocumentPrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = build_shard_plan(
            document_id="report:primitive",
            shards=_specs(2),
            shard_schema=SHARD_SCHEMA,
            document_schema=DOCUMENT_SCHEMA,
            plan_version="sections-v1",
            merge_version="flatten-items-v1",
            generation_contract=GENERATION_CONTRACT,
            merge_contract=MERGE_CONTRACT,
            response_mode=ResponseMode.PARTITIONED,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
        )

    def _receipt(self, index: int) -> dict[str, Any]:
        request = shard_request(self.plan, index)
        predecessor = (
            self._receipt(index - 1)["receipt_sha256"] if index else None
        )
        return create_shard_receipt(
            plan=self.plan,
            request=request,
            generated=_generated(request),
            predecessor_receipt_sha256=predecessor,
        )

    def test_receipt_is_stable_and_content_addressed(self) -> None:
        first = self._receipt(0)
        second = self._receipt(0)

        self.assertEqual(first, second)
        self.assertEqual(
            first["receipt_id"],
            f"sha256:{first['receipt_sha256']}",
        )

    def test_plan_identity_binds_generation_and_merge_contracts(self) -> None:
        changed_generation = deepcopy(GENERATION_CONTRACT)
        changed_generation["parameters"]["temperature"] = 0.2
        changed_generation_plan = build_shard_plan(
            document_id="report:primitive",
            shards=_specs(2),
            shard_schema=SHARD_SCHEMA,
            document_schema=DOCUMENT_SCHEMA,
            plan_version="sections-v1",
            merge_version="flatten-items-v1",
            generation_contract=changed_generation,
            merge_contract=MERGE_CONTRACT,
            response_mode=ResponseMode.PARTITIONED,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
        )
        changed_merge = {**MERGE_CONTRACT, "algorithm": "other_algorithm"}
        changed_merge_plan = build_shard_plan(
            document_id="report:primitive",
            shards=_specs(2),
            shard_schema=SHARD_SCHEMA,
            document_schema=DOCUMENT_SCHEMA,
            plan_version="sections-v1",
            merge_version="flatten-items-v1",
            generation_contract=GENERATION_CONTRACT,
            merge_contract=changed_merge,
            response_mode=ResponseMode.PARTITIONED,
            composability=ShardComposability.INDEPENDENT_DISJOINT,
        )
        self.assertNotEqual(self.plan.plan_sha256, changed_generation_plan.plan_sha256)
        self.assertNotEqual(self.plan.plan_sha256, changed_merge_plan.plan_sha256)

    def test_request_and_provider_audit_must_match_exactly(self) -> None:
        request = shard_request(self.plan, 0)
        with self.assertRaisesRegex(ShardReceiptError, "request identity"):
            create_shard_receipt(
                plan=self.plan,
                request=replace(request, shard_count=999),
                generated=_generated(request),
                predecessor_receipt_sha256=None,
            )
        generated = _generated(request)
        bad_audit = replace(
            generated.provider_audit,
            logical_request_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ShardReceiptError, "another logical"):
            create_shard_receipt(
                plan=self.plan,
                request=request,
                generated=replace(generated, provider_audit=bad_audit),
                predecessor_receipt_sha256=None,
            )

    def test_empty_plan_requires_an_explicit_reason(self) -> None:
        with self.assertRaisesRegex(ShardPlanError, "empty_plan_reason"):
            build_shard_plan(
                document_id="report:empty",
                shards=[],
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
            )

    def test_partial_prefix_is_resumable_but_incomplete_final_is_not(self) -> None:
        first = self._receipt(0)
        self.assertEqual(
            len(
                verify_shard_receipts(
                    self.plan,
                    [first],
                    require_complete=False,
                )
            ),
            1,
        )
        with self.assertRaisesRegex(ShardReceiptError, "coverage is incomplete"):
            verify_shard_receipts(
                self.plan,
                [first],
                require_complete=True,
            )

    def test_merge_uses_exact_plan_order(self) -> None:
        receipts = [self._receipt(0), self._receipt(1)]
        document, digest = merge_shard_receipts(
            plan=self.plan,
            receipts=receipts,
            merge_shards=_merge,
        )

        self.assertEqual(
            [item["value"] for item in document["items"]],
            [0, 1],
        )
        self.assertEqual(len(digest), 64)

    def test_plan_rejects_duplicate_ids_and_non_json_payloads(self) -> None:
        with self.assertRaisesRegex(ShardPlanError, "Duplicate shard_id"):
            build_shard_plan(
                document_id="report:duplicate",
                shards=[ShardSpec("same", {}), ShardSpec("same", {})],
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
            )
        with self.assertRaisesRegex(ShardPlanError, "non-JSON"):
            build_shard_plan(
                document_id="report:non-json",
                shards=[ShardSpec("bad", {"bad": {1, 2, 3}})],
                shard_schema=SHARD_SCHEMA,
                document_schema=DOCUMENT_SCHEMA,
                plan_version="sections-v1",
                merge_version="flatten-items-v1",
                generation_contract=GENERATION_CONTRACT,
                merge_contract=MERGE_CONTRACT,
                response_mode=ResponseMode.PARTITIONED,
                composability=ShardComposability.INDEPENDENT_DISJOINT,
            )


if __name__ == "__main__":
    unittest.main()
