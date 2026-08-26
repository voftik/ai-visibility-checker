from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm.attributes import flag_modified

from app.models import Base, Run, RunArtifact, RunStatus
from app.services import openrouter as openrouter_service
from app.services import structured_audit_store as audit_store
from app.services.analyzer import (
    ENTITY_CATALOG_CONTRACT_RECOVERY_MAX_ATTEMPTS,
    ENTITY_CATALOG_CONTRACT_RECOVERY_STAGE,
    ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
    ENTITY_CATALOG_LEAF_SCHEMA,
    ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY,
    ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY_SHA256,
    ENTITY_CATALOG_RECOVERY_PARENT_MANIFEST_VERSION,
    PROCESSING_MODEL,
    _EntityCatalogRecoveryShardSemanticError,
    _EntityCatalogRecoverySingletonContextError,
    _core_unit_claim,
    _deterministic_entity_catalog_union,
    _entity_catalog,
    _entity_catalog_quote_recovery_incident,
    _entity_catalog_recovery_artifact_key,
    _entity_catalog_recovery_checkpoint_identity,
    _entity_catalog_recovery_checkpoint_key,
    _entity_catalog_recovery_checkpoint_legacy_key,
    _entity_catalog_recovery_legacy_artifact_key,
    _entity_catalog_recovery_stage_key,
    _execute_entity_catalog_recovery_shards,
    _preflight_entity_catalog_recovery_shards,
    _recover_entity_catalog_chunk_contract,
    stable_digest,
)
from app.services.long_response import partition_text_records, text_sha256
from app.services.long_response import StructuredContinuationLedger
from app.services.openrouter import (
    ChatResult,
    OpenRouterError,
    OpenRouterResponseContractError,
    OpenRouterStructuredContinuationError,
)
from app.services.recovery_orchestrator import (
    ACTION_RETRY_WITH_GUIDANCE,
    CHECK_ENTITY_CATALOG_GROUNDING_FILTER_VALID,
    CHECK_ENTITY_CATALOG_SOURCE_BINDING_VALID,
    CHECK_PROMPT_CONTRACT_VALID,
    CHECK_RAW_CORPUS_UNCHANGED,
    OrchestratorContractError,
    RecoveryPlannerUnavailable,
)
from app.services.recovery_state import RecoveryBudgetExceeded


def _catalog_candidate(
    claim: dict[str, Any],
    *,
    quote: str,
) -> dict[str, Any]:
    return {
        "catalog": {
            "target_aliases": [quote],
            "entities": [],
            "uncertainties": [],
        },
        "core_dispositions": [
            {
                "claim_id": claim["claim_id"],
                "unit_id": claim["unit_id"],
                "core_sha256": claim["core_sha256"],
                "disposition": "grounded_fact",
                "evidence_quote": quote,
                "reason": "В core должна быть дословная сущность каталога.",
            }
        ],
    }


def _catalog_candidate_with_entity(
    claim: dict[str, Any],
    *,
    quote: str,
    canonical_name: str,
    aliases: list[str],
    evidence: str,
    target_aliases: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "catalog": {
            "target_aliases": target_aliases or ["ALPHA"],
            "entities": [
                {
                    "canonical_name": canonical_name,
                    "aliases": aliases,
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": evidence,
                }
            ],
            "uncertainties": [],
        },
        "core_dispositions": [
            {
                "claim_id": claim["claim_id"],
                "unit_id": claim["unit_id"],
                "core_sha256": claim["core_sha256"],
                "disposition": "grounded_fact",
                "evidence_quote": quote,
                "reason": "Имена должны быть связаны с дословной цитатой.",
            }
        ],
    }


def _retry_plan(*, reused: bool = False) -> SimpleNamespace:
    decision = {
        "action": ACTION_RETRY_WITH_GUIDANCE,
        "guidance": ("Повторно извлеки строку, сохранив точные координаты core."),
        "acceptance_checks": [
            CHECK_PROMPT_CONTRACT_VALID,
            CHECK_RAW_CORPUS_UNCHANGED,
            CHECK_ENTITY_CATALOG_SOURCE_BINDING_VALID,
            CHECK_ENTITY_CATALOG_GROUNDING_FILTER_VALID,
        ],
        "invalidate_artifact_keys": [],
    }
    return SimpleNamespace(
        run_id="run-id",
        epoch_id=40,
        epoch=4,
        stage_key=f"{ENTITY_CATALOG_CONTRACT_RECOVERY_STAGE}:test",
        decision=decision,
        reused=reused,
        facts_digest="f" * 64,
        failure_fingerprint="e" * 64,
        plan_digest="p" * 64,
    )


def _response_contract_error(*, complete: bool) -> Exception:
    raw_text = '{"catalog":'
    transport = {
        "status": "succeeded",
        "http_status": 200,
        "output_complete": complete,
        "output_limited": False,
        "output_incomplete_reason": (None if complete else "missing_finish_reason"),
    }
    usage = {
        "_aiv_transport": copy.deepcopy(transport),
        "_aiv_call_attempts": [
            {
                "attempt": 1,
                "status": "rejected",
                "raw_text": raw_text,
                "text_sha256": text_sha256(raw_text),
                "text_chars": len(raw_text),
                "text_utf8_bytes": len(raw_text.encode("utf-8")),
                "usage": {},
                "transport": copy.deepcopy(transport),
                "error_type": "OpenRouterResponseContractError",
                "error_message": "Structured response is unusable",
            }
        ],
    }
    return OpenRouterResponseContractError(
        "Structured response is unusable",
        result=ChatResult(
            text=raw_text,
            parsed=None,
            citations=[],
            usage=usage,
            annotations=[],
            request_policy={},
            web_attestation={},
            router_metadata={},
            transport=transport,
        ),
    )


def _terminal_schema_contract_error() -> Exception:
    raw_text = "{}"
    ledger = StructuredContinuationLedger(
        document_id="entity-catalog-terminal-schema",
        text=raw_text,
    )
    manifest = openrouter_service._structured_manifest_with_calls(
        ledger,
        call_records=[],
        complete=False,
    )
    manifest["terminal_semantic_failure"] = (
        openrouter_service._structured_terminal_schema_failure_marker(
            ledger,
            ENTITY_CATALOG_LEAF_SCHEMA,
        )
    )
    transport = {
        "status": "succeeded",
        "http_status": 200,
        "output_complete": False,
        "output_limited": True,
        "output_incomplete_reason": "length",
    }
    return OpenRouterStructuredContinuationError(
        "Complete JSON violates the entity-catalog schema",
        result=ChatResult(
            text=raw_text,
            parsed=None,
            citations=[],
            usage={
                "_aiv_transport": copy.deepcopy(transport),
                "_aiv_structured_continuation": copy.deepcopy(manifest),
            },
            annotations=[],
            request_policy={},
            web_attestation={},
            router_metadata={},
            transport=transport,
        ),
        manifest=manifest,
    )


def _terminal_rejected_part_contract_error() -> Exception:
    accepted_text = '{"catalog":'
    accepted_transport = {
        "status": "succeeded",
        "http_status": 200,
        "output_complete": False,
        "output_limited": True,
        "output_incomplete_reason": "length",
    }
    accepted_result = ChatResult(
        text=accepted_text,
        parsed=None,
        citations=[],
        usage={"_aiv_transport": copy.deepcopy(accepted_transport)},
        annotations=[],
        request_policy={},
        web_attestation={},
        router_metadata={},
        transport=accepted_transport,
    )
    rejected_transport = {
        "status": "succeeded",
        "http_status": 200,
        "output_complete": True,
        "output_limited": False,
        "output_incomplete_reason": None,
    }
    rejected_result = ChatResult(
        text="",
        parsed=None,
        citations=[],
        usage={"_aiv_transport": copy.deepcopy(rejected_transport)},
        annotations=[],
        request_policy={},
        web_attestation={},
        router_metadata={},
        transport=rejected_transport,
    )
    ledger = StructuredContinuationLedger(
        document_id="entity-catalog-terminal-rejected-part",
        text=accepted_text,
    )
    accepted_call = openrouter_service._continuation_call_record(
        accepted_result,
        sequence=0,
    )
    rejected_part = openrouter_service._continuation_call_record(
        rejected_result,
        sequence=1,
    )
    manifest = openrouter_service._structured_manifest_with_calls(
        ledger,
        call_records=[accepted_call],
        complete=False,
    )
    manifest["rejected_part"] = copy.deepcopy(rejected_part)
    manifest["terminal_semantic_failure"] = (
        openrouter_service._structured_terminal_rejected_part_failure_marker(
            ledger,
            ENTITY_CATALOG_LEAF_SCHEMA,
            rejected_part,
        )
    )
    return OpenRouterStructuredContinuationError(
        "Complete continuation response is empty",
        result=rejected_result,
        manifest=manifest,
    )


def _sealed_checkpoint_output(identity: dict[str, Any]) -> dict[str, Any]:
    body = {
        "version": ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
        "identity": copy.deepcopy(identity),
        "identity_sha256": stable_digest(identity),
    }
    return {**body, "checkpoint_sha256": stable_digest(body)}


def _sharded_recovery_job(count: int = 3) -> dict[str, Any]:
    units, _manifests = partition_text_records(
        [
            {
                "answer_id": index,
                "answer": f"ALPHA — подтверждённое имя, ответ {index}.",
            }
            for index in range(1, count + 1)
        ],
        text_key="answer",
        id_key="answer_id",
        target_chars=1_000,
    )
    return {
        "artifact_key": "entity_catalog_chunk_test",
        "schema_name": "aiv_entity_catalog_chunk_test",
        "answers": units,
        "model_answers": [
            {
                "answer_id": item["answer_id"],
                "answer": item["answer"],
                "core_claim": _core_unit_claim(item),
            }
            for item in units
        ],
    }


def _sharded_recovery_payload(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": {"brand_name": "ALPHA", "aliases": ["ALPHA"]},
        "answers": copy.deepcopy(job["model_answers"]),
        "recovery": {
            "version": ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
            "epoch": 7,
            "execution_attempt": 1,
            "failure_code": "core_quote_not_exact_substring",
            "failed_unit_id": "",
            "failed_unit_ids": [],
            "binding_failures": [],
            "source_units_sha256": stable_digest(job["answers"]),
            "quote_repair_version": "test",
            "grounding_filter_version": "test",
            "acceptance_policy": copy.deepcopy(
                ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY
            ),
            "acceptance_policy_sha256": (
                ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY_SHA256
            ),
            "invalid_candidate_sha256": "c" * 64,
            "orchestrator_guidance": "Сохрани точные цитаты.",
            "immutable_contract": "No source mutation.",
        },
    }


class _RecoveryProviderResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = copy.deepcopy(body)

    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self._body)


def _recovery_provider_body(
    text: str,
    *,
    limited: bool,
) -> dict[str, Any]:
    return {
        "model": PROCESSING_MODEL,
        "provider": "Recovery Harness Test",
        "choices": [
            {
                "finish_reason": "length" if limited else "stop",
                "native_finish_reason": (
                    "MAX_TOKENS" if limited else "stop"
                ),
                "message": {
                    "role": "assistant",
                    "content": text,
                    "annotations": [],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": 100 + max(1, len(text) // 4),
            "server_tool_use": {"web_search_requests": 0},
        },
        "openrouter_metadata": {"pipeline": []},
    }


class _RecoveryPrefixClient:
    def __init__(self, full_text: str, prefix_text: str) -> None:
        self.full_text = full_text
        self.prefix_text = prefix_text
        self.requests: list[dict[str, Any]] = []
        self.request_kinds: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        content: bytes,
    ) -> _RecoveryProviderResponse:
        payload = json.loads(content.decode("utf-8"))
        self.requests.append({"headers": dict(headers), "json": payload})
        messages = payload["messages"]
        is_continuation = "CONTINUATION_CONTRACT_JSON:" in str(
            messages[-1].get("content") or ""
        )
        if not is_continuation:
            self.request_kinds.append("initial")
            if self.request_kinds.count("initial") > 1:
                raise AssertionError("Paid initial recovery POST was repeated")
            return _RecoveryProviderResponse(
                _recovery_provider_body(self.prefix_text, limited=True)
            )
        self.request_kinds.append("continuation")
        accepted_prefix = str(messages[-2].get("content") or "")
        if accepted_prefix != self.prefix_text:
            raise AssertionError("Continuation did not resume the durable prefix")
        response_text = accepted_prefix[-512:] + self.full_text[
            len(accepted_prefix) :
        ]
        return _RecoveryProviderResponse(
            _recovery_provider_body(response_text, limited=False)
        )


class _RecoveryHeadroomFailoverClient:
    def __init__(self, full_text: str, prefix_text: str) -> None:
        self.full_text = full_text
        self.prefix_text = prefix_text
        self.requests: list[dict[str, Any]] = []
        self.request_kinds: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        content: bytes,
    ) -> _RecoveryProviderResponse:
        payload = json.loads(content.decode("utf-8"))
        self.requests.append({"headers": dict(headers), "json": payload})
        if len(self.requests) == 1:
            self.request_kinds.append("direct_initial")
            return _RecoveryProviderResponse(
                _recovery_provider_body(self.prefix_text, limited=True)
            )
        if len(self.requests) == 2:
            if "CONTINUATION_CONTRACT_JSON:" in str(
                payload["messages"][-1].get("content") or ""
            ):
                raise AssertionError(
                    "Context-exhausted direct continuation reached provider"
                )
            self.request_kinds.append("shard_initial")
            return _RecoveryProviderResponse(
                _recovery_provider_body(self.full_text, limited=False)
            )
        raise AssertionError("Unexpected recovery provider POST")


class EntityCatalogRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _run_catalog(
        self,
        processing: Any,
        *,
        enabled: bool = True,
        plan: Any | None = None,
        attempts: list[int] | None = None,
        planner_mock: AsyncMock | None = None,
        fallback_plan_mock: AsyncMock | None = None,
        reserve_mock: AsyncMock | None = None,
        finish_mock: AsyncMock | None = None,
        artifact_output_mock: AsyncMock | None = None,
        execution_state_mock: AsyncMock | None = None,
        save_mock: AsyncMock | None = None,
        epoch_binding_mock: AsyncMock | None = None,
        recovery_artifact_mock: AsyncMock | None = None,
        answers: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
        mark_contract_failed_mock: AsyncMock | None = None,
    ) -> tuple[dict[str, Any] | None, AsyncMock, AsyncMock, AsyncMock]:
        planner = planner_mock or AsyncMock(return_value=plan or _retry_plan())
        fallback_plan = fallback_plan_mock or AsyncMock(
            return_value=plan or _retry_plan()
        )
        reserve = reserve_mock or AsyncMock(side_effect=attempts or [1])
        finish = finish_mock or AsyncMock()
        artifact_output = artifact_output_mock or AsyncMock(return_value=None)
        execution_state = execution_state_mock or AsyncMock(
            return_value=SimpleNamespace(
                status="planned",
                execution_attempts=0,
            )
        )
        save = save_mock or AsyncMock()
        epoch_binding = epoch_binding_mock or AsyncMock(
            return_value=("succeeded", None)
        )
        recovery_artifact = recovery_artifact_mock or AsyncMock(return_value=None)
        mark_contract_failed = mark_contract_failed_mock or AsyncMock()
        result: dict[str, Any] | None = None
        with (
            patch(
                "app.services.analyzer.settings.PIPELINE_ORCHESTRATOR_ENABLED",
                enabled,
            ),
            patch(
                "app.services.analyzer._processing_artifact",
                new=processing,
            ),
            patch(
                "app.services.analyzer._analyzer_model_input_window",
                new=AsyncMock(
                    return_value={
                        "input_utf8_window": 10_000_000,
                        "model_envelope": {},
                    }
                ),
            ),
            patch(
                "app.services.analyzer.plan_durable_recovery",
                planner,
            ),
            patch(
                "app.services.analyzer.plan_code_owned_recovery",
                fallback_plan,
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                reserve,
            ),
            patch("app.services.analyzer.finish_recovery", finish),
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch(
                "app.services.analyzer.recovery_execution_state",
                execution_state,
            ),
            patch("app.services.analyzer._save_artifact", save),
            patch(
                "app.services.analyzer._validate_entity_catalog_recovery_epoch_binding",
                epoch_binding,
            ),
            patch(
                "app.services.analyzer._entity_catalog_recovery_artifact",
                recovery_artifact,
            ),
            patch(
                "app.services.analyzer._mark_completed_artifact_contract_failed",
                new=mark_contract_failed,
            ),
        ):
            result = await _entity_catalog(
                "run-id",
                profile
                or {
                    "brand_name": "ALPHA",
                    "brand_aliases": [],
                    "products": [],
                    "topics": [],
                    "entity_scope": [],
                },
                answers or [{"answer_id": 1, "answer": "ALPHA"}],
            )
        return result, planner, reserve, finish

    async def test_valid_leaf_never_invokes_fable_planner(self) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(processing)

        self.assertIsNotNone(result)
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()

    async def test_exact_quote_failure_uses_one_plan_and_strict_retry(
        self,
    ) -> None:
        calls: list[str] = []
        completion_order: list[str] = []
        answers = [{"answer_id": 1, "answer": "ALPHA"}]
        original_answers = copy.deepcopy(answers)
        save = AsyncMock(
            side_effect=lambda *args, **kwargs: completion_order.append("checkpoint")
        )
        finish = AsyncMock(
            side_effect=lambda *args, **kwargs: completion_order.append("finish")
        )

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, _finish = await self._run_catalog(
            processing,
            save_mock=save,
            finish_mock=finish,
        )

        self.assertEqual(answers, original_answers)
        self.assertIsNotNone(result)
        self.assertIn("entity_catalog_chunk_1_", calls[0])
        self.assertTrue(calls[1].endswith("_recovery_e4_a1"))
        self.assertEqual(calls[-1], "entity_catalog")
        planner.assert_awaited_once()
        planner_kwargs = planner.await_args.kwargs
        self.assertTrue(
            planner_kwargs["stage_key"].startswith(
                ENTITY_CATALOG_CONTRACT_RECOVERY_STAGE + ":"
            )
        )
        self.assertEqual(planner_kwargs["stage_planner_call_limit"], 1)
        self.assertEqual(
            planner_kwargs["allowed_actions"],
            {ACTION_RETRY_WITH_GUIDANCE, "stop_and_preserve_checkpoint"},
        )
        facts = planner_kwargs["facts"]
        self.assertEqual(
            facts["acceptance_policy_sha256"],
            ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY_SHA256,
        )
        self.assertEqual(
            facts["acceptance_policy"],
            ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY,
        )
        self.assertEqual(
            facts["checkpoint_identity_sha256"],
            stable_digest(facts["checkpoint_identity"]),
        )
        self.assertEqual(
            facts["target_sha256"],
            facts["checkpoint_identity"]["target_sha256"],
        )
        self.assertEqual(
            set(
                facts["executor_contract"][ACTION_RETRY_WITH_GUIDANCE][
                    "acceptance_checks"
                ]
            ),
            {
                CHECK_PROMPT_CONTRACT_VALID,
                CHECK_RAW_CORPUS_UNCHANGED,
                CHECK_ENTITY_CATALOG_SOURCE_BINDING_VALID,
                CHECK_ENTITY_CATALOG_GROUNDING_FILTER_VALID,
            },
        )
        reserve.assert_awaited_once()
        self.assertEqual(
            reserve.await_args.kwargs["stage_execution_limit"],
            ENTITY_CATALOG_CONTRACT_RECOVERY_MAX_ATTEMPTS,
        )
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertTrue(finish.await_args.kwargs["details"]["raw_source_unchanged"])
        self.assertEqual(
            set(finish.await_args.kwargs["details"]["executed_acceptance_checks"]),
            {
                CHECK_PROMPT_CONTRACT_VALID,
                CHECK_RAW_CORPUS_UNCHANGED,
                CHECK_ENTITY_CATALOG_SOURCE_BINDING_VALID,
                CHECK_ENTITY_CATALOG_GROUNDING_FILTER_VALID,
            },
        )
        save.assert_awaited_once()
        saved = save.await_args.kwargs
        self.assertTrue(saved["artifact_key"].endswith("_accepted"))
        self.assertEqual(saved["status"], "completed")
        self.assertIsNone(saved["model"])
        self.assertEqual(
            saved["output_json"]["accepted_output_sha256"],
            saved["usage_json"]["_aiv_entity_catalog_recovery"][
                "accepted_output_sha256"
            ],
        )
        self.assertEqual(completion_order, ["checkpoint", "finish"])

    async def test_compound_representative_quote_is_preserved_without_retry(
        self,
    ) -> None:
        calls: list[str] = []

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if "answers" in payload:
                invalid = _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA source",
                )
                invalid["catalog"]["target_aliases"] = ["ALPHA"]
                return invalid
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[{"answer_id": 1, "answer": "ALPHA source"}],
        )

        self.assertIsNotNone(result)
        self.assertFalse(any("_recovery_" in key for key in calls))
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()
        self.assertTrue(
            any("ALPHA source" in value for value in result.get("uncertainties", []))
        )

    async def test_evidence_binding_failure_uses_strict_recovery(
        self,
    ) -> None:
        calls: list[str] = []

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                return _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                    canonical_name="ALPHA",
                    aliases=[],
                    evidence="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                    canonical_name="ALPHA",
                    aliases=["GAMMA"],
                    evidence="ALPHA GAMMA",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[{"answer_id": 1, "answer": "ALPHA source"}],
        )

        self.assertIsNotNone(result)
        self.assertTrue(any(key.endswith("_recovery_e4_a1") for key in calls))
        planner.assert_awaited_once()
        self.assertEqual(
            planner.await_args.kwargs["failure_code"],
            "entity_catalog_evidence_binding_failed",
        )
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_entity_recovery_uses_bounded_code_owned_plan_after_cap(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            artifact_key = str(kwargs["artifact_key"])
            if "_recovery_" in artifact_key:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        planner = AsyncMock(side_effect=RecoveryBudgetExceeded("planner cap reached"))
        fallback = AsyncMock(return_value=_retry_plan())
        result, _planner, reserve, finish = await self._run_catalog(
            processing,
            planner_mock=planner,
            fallback_plan_mock=fallback,
        )

        self.assertIsNotNone(result)
        planner.assert_awaited_once()
        fallback.assert_awaited_once()
        fallback_kwargs = fallback.await_args.kwargs
        self.assertEqual(
            fallback_kwargs["decision"]["action"],
            ACTION_RETRY_WITH_GUIDANCE,
        )
        self.assertEqual(
            set(fallback_kwargs["decision"]["acceptance_checks"]),
            {
                CHECK_PROMPT_CONTRACT_VALID,
                CHECK_RAW_CORPUS_UNCHANGED,
                CHECK_ENTITY_CATALOG_SOURCE_BINDING_VALID,
                CHECK_ENTITY_CATALOG_GROUNDING_FILTER_VALID,
            },
        )
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_entity_recovery_uses_code_owned_plan_on_provider_failure(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            artifact_key = str(kwargs["artifact_key"])
            quote = "ALPHA" if "_recovery_" in artifact_key else "Alpha"
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote=quote,
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        planner = AsyncMock(
            side_effect=RecoveryPlannerUnavailable("provider transport failed")
        )
        fallback = AsyncMock(return_value=_retry_plan())
        result, _planner, reserve, finish = await self._run_catalog(
            processing,
            planner_mock=planner,
            fallback_plan_mock=fallback,
        )

        self.assertIsNotNone(result)
        planner.assert_awaited_once()
        fallback.assert_awaited_once()
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_entity_recovery_never_masks_durable_plan_corruption(self) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        planner = AsyncMock(
            side_effect=OrchestratorContractError(
                "Stored recovery plan digest mismatch"
            )
        )
        fallback = AsyncMock(return_value=_retry_plan())
        with self.assertRaisesRegex(
            OpenRouterError,
            "recovery planner failed",
        ):
            await self._run_catalog(
                processing,
                planner_mock=planner,
                fallback_plan_mock=fallback,
            )
        planner.assert_awaited_once()
        fallback.assert_not_awaited()

    def test_obsolete_absent_output_incident_is_not_admitted(self) -> None:
        core_text = "ALPHA source"
        claim = {
            "claim_id": "claim-1",
            "unit_id": "1:000000",
            "core_sha256": text_sha256(core_text),
            "core_text": core_text,
        }
        error = OpenRouterError(
            "Grounded core-unit quote is absent from the analytic output: 1:000000"
        )
        valid_incident = _catalog_candidate(claim, quote=core_text)
        valid_incident["catalog"]["target_aliases"] = ["ALPHA"]
        self.assertIsNone(
            _entity_catalog_quote_recovery_incident(
                error,
                candidate=valid_incident,
                expected_claims=[claim],
            )
        )

        quote_not_in_core = _catalog_candidate(claim, quote="BETA")
        quote_not_in_core["catalog"]["target_aliases"] = ["ALPHA"]
        self.assertIsNone(
            _entity_catalog_quote_recovery_incident(
                error,
                candidate=quote_not_in_core,
                expected_claims=[claim],
            )
        )

        quote_already_visible = _catalog_candidate(claim, quote=core_text)
        self.assertIsNone(
            _entity_catalog_quote_recovery_incident(
                error,
                candidate=quote_already_visible,
                expected_claims=[claim],
            )
        )

        duplicated_row = copy.deepcopy(valid_incident)
        duplicated_row["core_dispositions"].append(
            copy.deepcopy(duplicated_row["core_dispositions"][0])
        )
        self.assertIsNone(
            _entity_catalog_quote_recovery_incident(
                error,
                candidate=duplicated_row,
                expected_claims=[claim],
            )
        )

    def test_binding_incident_rejects_unknown_or_mismatched_paths(self) -> None:
        core_text = "ALPHA source"
        claim = {
            "claim_id": "claim-1",
            "unit_id": "1:000000",
            "core_sha256": text_sha256(core_text),
            "core_text": core_text,
        }
        candidate = _catalog_candidate_with_entity(
            claim,
            quote="ALPHA",
            canonical_name="ALPHA",
            aliases=["GAMMA"],
            evidence="ALPHA",
        )
        exact_error = OpenRouterError(
            "Entity-catalog evidence binding failed: "
            "entities[0].aliases[0] is not grounded"
        )
        admitted = _entity_catalog_quote_recovery_incident(
            exact_error,
            candidate=candidate,
            expected_claims=[claim],
            profile={"brand_name": "ALPHA", "brand_aliases": []},
        )
        self.assertIsNotNone(admitted)

        for spoofed in (
            "Entity-catalog evidence binding failed: entities[0].aliases is not a list",
            "Entity-catalog evidence binding failed: "
            "entities[0].canonical_name is not grounded",
        ):
            with self.subTest(spoofed=spoofed):
                self.assertIsNone(
                    _entity_catalog_quote_recovery_incident(
                        OpenRouterError(spoofed),
                        candidate=candidate,
                        expected_claims=[claim],
                        profile={
                            "brand_name": "ALPHA",
                            "brand_aliases": [],
                        },
                    )
                )

    def test_policy_digest_versions_checkpoint_identity_key_and_stage(self) -> None:
        units, _manifests = partition_text_records(
            [{"answer_id": 1, "answer": "ALPHA"}],
            text_key="answer",
            id_key="answer_id",
            target_chars=1_000,
        )
        job = {
            "artifact_key": "entity_catalog_chunk_1_test",
            "answers": units,
            "model_answers": [{"answer_id": 1, "answer": "ALPHA"}],
        }
        target = {"brand_name": "ALPHA", "aliases": [], "products": []}
        identity = _entity_catalog_recovery_checkpoint_identity(
            job=job,
            target=target,
        )
        checkpoint_key = _entity_catalog_recovery_checkpoint_key(
            job=job,
            target=target,
        )
        stage_key = _entity_catalog_recovery_stage_key(
            job=job,
            target=target,
        )

        with patch(
            "app.services.analyzer.ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY_SHA256",
            "f" * 64,
        ):
            changed_identity = _entity_catalog_recovery_checkpoint_identity(
                job=job,
                target=target,
            )
            changed_checkpoint_key = _entity_catalog_recovery_checkpoint_key(
                job=job,
                target=target,
            )
            changed_stage_key = _entity_catalog_recovery_stage_key(
                job=job,
                target=target,
            )

        self.assertNotEqual(identity, changed_identity)
        self.assertNotEqual(checkpoint_key, changed_checkpoint_key)
        self.assertNotEqual(stage_key, changed_stage_key)

        changed_target = {**target, "products": ["BETA"]}
        self.assertNotEqual(
            checkpoint_key,
            _entity_catalog_recovery_checkpoint_key(
                job=job,
                target=changed_target,
            ),
        )
        self.assertNotEqual(
            stage_key,
            _entity_catalog_recovery_stage_key(
                job=job,
                target=changed_target,
            ),
        )
        self.assertNotEqual(
            _entity_catalog_recovery_artifact_key(
                job=job,
                target=target,
                epoch=1,
                attempt=1,
            ),
            _entity_catalog_recovery_artifact_key(
                job=job,
                target=changed_target,
                epoch=1,
                attempt=1,
            ),
        )
        self.assertEqual(
            _entity_catalog_recovery_checkpoint_legacy_key(job),
            _entity_catalog_recovery_checkpoint_legacy_key(
                {**job, "model_answers": [{"changed": True}]}
            ),
        )

    async def _assert_restart_resumes_reserved_attempt(
        self,
        attempt_number: int,
    ) -> None:
        calls: list[str] = []
        plan = _retry_plan(reused=True)
        reserve = AsyncMock()
        finish = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith(f"_recovery_e4_a{attempt_number}"):
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, _reserve, _finish = await self._run_catalog(
            processing,
            plan=plan,
            reserve_mock=reserve,
            finish_mock=finish,
            execution_state_mock=AsyncMock(
                return_value=SimpleNamespace(
                    status="executing",
                    execution_attempts=attempt_number,
                )
            ),
        )

        self.assertIsNotNone(result)
        planner.assert_awaited_once()
        reserve.assert_not_awaited()
        finish.assert_awaited_once()
        self.assertTrue(
            any(key.endswith(f"_recovery_e4_a{attempt_number}") for key in calls)
        )
        if attempt_number == 2:
            self.assertFalse(any(key.endswith("_recovery_e4_a1") for key in calls))

    async def test_restart_resumes_reserved_attempt_one_without_new_mark(
        self,
    ) -> None:
        await self._assert_restart_resumes_reserved_attempt(1)

    async def test_restart_resumes_reserved_attempt_two_without_new_mark(
        self,
    ) -> None:
        await self._assert_restart_resumes_reserved_attempt(2)

    async def test_retry_is_bounded_to_two_and_preserves_source(self) -> None:
        calls: list[str] = []
        answers = [{"answer_id": 1, "answer": "ALPHA"}]
        original_answers = copy.deepcopy(answers)
        planner = AsyncMock(return_value=_retry_plan())
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            return _catalog_candidate(
                payload["answers"][0]["core_claim"],
                quote="Alpha",
            )

        with self.assertRaisesRegex(
            OpenRouterError,
            "contract recovery exhausted",
        ):
            await self._run_catalog(
                processing,
                planner_mock=planner,
                reserve_mock=reserve,
                finish_mock=finish,
                answers=answers,
            )

        recovery_calls = [key for key in calls if "_recovery_" in key]
        self.assertEqual(len(recovery_calls), 2)
        self.assertTrue(recovery_calls[0].endswith("_a1"))
        self.assertTrue(recovery_calls[1].endswith("_a2"))
        self.assertEqual(answers, original_answers)
        planner.assert_awaited_once()
        self.assertEqual(reserve.await_count, 2)
        finish.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["succeeded"])
        self.assertTrue(finish.await_args.kwargs["details"]["raw_source_unchanged"])

    async def test_transport_failure_preserves_reserved_attempt_for_resume(
        self,
    ) -> None:
        calls: list[str] = []
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                raise OpenRouterError("temporary provider transport failure")
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            raise AssertionError("Transport failure must stop before reducer")

        with self.assertRaisesRegex(
            OpenRouterError,
            "temporary provider transport failure",
        ):
            await self._run_catalog(
                processing,
                reserve_mock=reserve,
                finish_mock=finish,
            )

        self.assertEqual(reserve.await_count, 1)
        finish.assert_not_awaited()
        self.assertTrue(any(key.endswith("_recovery_e4_a1") for key in calls))
        self.assertFalse(any(key.endswith("_recovery_e4_a2") for key in calls))

    async def test_unmarked_complete_contract_failure_keeps_a1_reserved(
        self,
    ) -> None:
        calls: list[str] = []
        reserve = AsyncMock(return_value=1)
        finish = AsyncMock()
        mark_contract_failed = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                raise _response_contract_error(complete=True)
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        with self.assertRaisesRegex(
            OpenRouterResponseContractError,
            "Structured response is unusable",
        ):
            await self._run_catalog(
                processing,
                reserve_mock=reserve,
                finish_mock=finish,
                mark_contract_failed_mock=mark_contract_failed,
            )

        recovery_calls = [key for key in calls if "_recovery_" in key]
        self.assertEqual(len(recovery_calls), 1)
        self.assertTrue(recovery_calls[0].endswith("_a1"))
        self.assertEqual(reserve.await_count, 1)
        self.assertFalse(
            any(
                call.kwargs["artifact_key"].endswith("_recovery_e4_a1")
                for call in mark_contract_failed.await_args_list
            )
        )
        finish.assert_not_awaited()

    async def test_terminal_schema_failure_spends_a1_and_uses_a2(self) -> None:
        calls: list[str] = []
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()
        mark_contract_failed = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                raise _terminal_schema_contract_error()
            if artifact_key.endswith("_recovery_e4_a2"):
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            reserve_mock=reserve,
            finish_mock=finish,
            mark_contract_failed_mock=mark_contract_failed,
        )

        self.assertIsNotNone(result)
        recovery_calls = [key for key in calls if "_recovery_" in key]
        self.assertEqual(len(recovery_calls), 2)
        self.assertTrue(recovery_calls[0].endswith("_a1"))
        self.assertTrue(recovery_calls[1].endswith("_a2"))
        self.assertEqual(reserve.await_count, 2)
        self.assertTrue(
            any(
                call.kwargs["artifact_key"].endswith("_recovery_e4_a1")
                for call in mark_contract_failed.await_args_list
            )
        )
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            finish.await_args.kwargs["details"]["execution_attempt"],
            2,
        )

    async def test_terminal_rejected_part_spends_a1_and_uses_a2(self) -> None:
        calls: list[str] = []
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()
        mark_contract_failed = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                raise _terminal_rejected_part_contract_error()
            if artifact_key.endswith("_recovery_e4_a2"):
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            reserve_mock=reserve,
            finish_mock=finish,
            mark_contract_failed_mock=mark_contract_failed,
        )

        self.assertIsNotNone(result)
        recovery_calls = [key for key in calls if "_recovery_" in key]
        self.assertEqual(len(recovery_calls), 2)
        self.assertTrue(recovery_calls[0].endswith("_a1"))
        self.assertTrue(recovery_calls[1].endswith("_a2"))
        self.assertEqual(reserve.await_count, 2)
        self.assertTrue(
            any(
                call.kwargs["artifact_key"].endswith("_recovery_e4_a1")
                for call in mark_contract_failed.await_args_list
            )
        )
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            finish.await_args.kwargs["details"]["execution_attempt"],
            2,
        )

    async def test_incomplete_response_contract_keeps_same_a1_for_resume(
        self,
    ) -> None:
        calls: list[str] = []
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                raise _response_contract_error(complete=False)
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            raise AssertionError("Incomplete response must stop before reducer")

        with self.assertRaises(OpenRouterResponseContractError):
            await self._run_catalog(
                processing,
                reserve_mock=reserve,
                finish_mock=finish,
            )

        self.assertEqual(reserve.await_count, 1)
        finish.assert_not_awaited()
        self.assertTrue(any(key.endswith("_recovery_e4_a1") for key in calls))
        self.assertFalse(any(key.endswith("_recovery_e4_a2") for key in calls))

    async def test_existing_invalid_checkpoint_never_falls_back_to_model(
        self,
    ) -> None:
        variants = {
            "status": SimpleNamespace(
                status="running",
                model=None,
                prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
                input_json={},
                output_json={},
            ),
            "input_json": SimpleNamespace(
                status="completed",
                model=None,
                prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
                input_json={"tampered": True},
                output_json={},
            ),
            "prompt_version": SimpleNamespace(
                status="completed",
                model=None,
                prompt_version="tampered-version",
                input_json={},
                output_json={},
            ),
        }
        for expected_mismatch, artifact in variants.items():
            with self.subTest(expected_mismatch=expected_mismatch):
                processing = AsyncMock()
                planner = AsyncMock()
                with self.assertRaisesRegex(
                    OpenRouterError,
                    expected_mismatch,
                ):
                    await self._run_catalog(
                        processing,
                        planner_mock=planner,
                        recovery_artifact_mock=AsyncMock(return_value=artifact),
                    )
                processing.assert_not_awaited()
                planner.assert_not_awaited()

    async def test_stale_legacy_checkpoint_does_not_block_new_target_identity(
        self,
    ) -> None:
        artifact_lookups: list[str] = []
        checkpoint_identities: list[dict[str, Any]] = []

        async def artifact_output(
            _run_id: str,
            _artifact_key: str,
            **kwargs: Any,
        ) -> None:
            checkpoint_identities.append(copy.deepcopy(kwargs["input_json"]))
            return None

        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            payload = kwargs["user_payload"]
            if "_recovery_" in artifact_key:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        async def recovery_artifact(
            _run_id: str,
            artifact_key: str,
        ) -> Any:
            artifact_lookups.append(artifact_key)
            if len(artifact_lookups) == 1:
                return None
            stale_identity = {
                **checkpoint_identities[-1],
                "target_sha256": "0" * 64,
            }
            return SimpleNamespace(
                status="completed",
                model=None,
                prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
                input_json=copy.deepcopy(stale_identity),
                output_json=_sealed_checkpoint_output(stale_identity),
            )

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            artifact_output_mock=AsyncMock(side_effect=artifact_output),
            recovery_artifact_mock=AsyncMock(side_effect=recovery_artifact),
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(artifact_lookups), 2)
        self.assertNotEqual(artifact_lookups[0], artifact_lookups[1])
        self.assertTrue(artifact_lookups[0].endswith("_accepted"))
        self.assertTrue(artifact_lookups[1].endswith("_accepted"))
        planner.assert_awaited_once()
        reserve.assert_awaited_once()
        finish.assert_awaited_once()

    async def test_exact_identity_corrupt_legacy_checkpoint_stays_fail_closed(
        self,
    ) -> None:
        identities: list[dict[str, Any]] = []
        artifact_lookups = 0

        async def artifact_output(
            _run_id: str,
            _artifact_key: str,
            **kwargs: Any,
        ) -> None:
            identities.append(copy.deepcopy(kwargs["input_json"]))
            return None

        async def recovery_artifact(
            _run_id: str,
            _artifact_key: str,
        ) -> Any:
            nonlocal artifact_lookups
            artifact_lookups += 1
            if artifact_lookups == 1:
                return None
            current_identity = copy.deepcopy(identities[-1])
            return SimpleNamespace(
                status="completed",
                model=None,
                prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
                input_json={**current_identity, "target_sha256": "f" * 64},
                output_json=_sealed_checkpoint_output(current_identity),
            )

        processing = AsyncMock()
        planner = AsyncMock()
        with self.assertRaisesRegex(
            OpenRouterError,
            "legacy recovery checkpoint.*envelope_identity_binding",
        ):
            await self._run_catalog(
                processing,
                planner_mock=planner,
                artifact_output_mock=AsyncMock(side_effect=artifact_output),
                recovery_artifact_mock=AsyncMock(side_effect=recovery_artifact),
            )

        self.assertEqual(artifact_lookups, 2)
        processing.assert_not_awaited()
        planner.assert_not_awaited()

    async def test_accepted_checkpoint_reuses_exact_recovery_without_leaf_post(
        self,
    ) -> None:
        stored: dict[str, Any] = {}
        recovered_leaf: dict[str, Any] = {}
        recovery_input: dict[str, Any] = {}

        async def first_processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            payload = kwargs["user_payload"]
            if artifact_key.endswith("_recovery_e4_a1"):
                value = _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
                recovered_leaf.update(copy.deepcopy(value))
                recovery_input.update(copy.deepcopy(payload))
                return value
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        async def capture_checkpoint(*args: Any, **kwargs: Any) -> None:
            stored.update(
                {
                    "artifact_key": kwargs["artifact_key"],
                    "input_json": copy.deepcopy(kwargs["input_json"]),
                    "output_json": copy.deepcopy(kwargs["output_json"]),
                }
            )

        await self._run_catalog(
            first_processing,
            save_mock=AsyncMock(side_effect=capture_checkpoint),
        )
        self.assertTrue(stored["artifact_key"].endswith("_accepted"))

        async def saved_artifacts(
            _run_id: str,
            artifact_key: str,
            **kwargs: Any,
        ) -> dict[str, Any] | None:
            if artifact_key == stored["artifact_key"]:
                self.assertEqual(kwargs["input_json"], stored["input_json"])
                return copy.deepcopy(stored["output_json"])
            return None

        recovery_artifact = AsyncMock(
            return_value=SimpleNamespace(
                status="completed",
                model=PROCESSING_MODEL,
                prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
                input_json=copy.deepcopy(recovery_input),
                output_json=copy.deepcopy(recovered_leaf),
            )
        )

        second_calls: list[str] = []

        async def second_processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            second_calls.append(artifact_key)
            if artifact_key != "entity_catalog":
                self.fail("Accepted leaf checkpoint triggered a new leaf POST")
            return _deterministic_entity_catalog_union(
                list(kwargs["user_payload"]["chunk_catalogs"])
            )

        result, planner, reserve, finish = await self._run_catalog(
            second_processing,
            artifact_output_mock=AsyncMock(side_effect=saved_artifacts),
            recovery_artifact_mock=recovery_artifact,
        )

        self.assertIsNotNone(result)
        self.assertEqual(second_calls, ["entity_catalog"])
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()
        recovery_artifact.assert_awaited_once_with(
            "run-id",
            stored["output_json"]["recovery_artifact_key"],
        )

        legacy_checkpoint_key = stored["artifact_key"].rsplit("_", 2)[0] + "_accepted"
        legacy_checkpoint_output = copy.deepcopy(stored["output_json"])
        legacy_checkpoint_output["recovery_artifact_key"] = (
            _entity_catalog_recovery_legacy_artifact_key(
                job={
                    "artifact_key": legacy_checkpoint_output["identity"][
                        "base_artifact_key"
                    ]
                },
                epoch=legacy_checkpoint_output["epoch"],
                attempt=legacy_checkpoint_output["execution_attempt"],
            )
        )
        legacy_checkpoint_body = copy.deepcopy(legacy_checkpoint_output)
        legacy_checkpoint_body.pop("checkpoint_sha256")
        legacy_checkpoint_output["checkpoint_sha256"] = stable_digest(
            legacy_checkpoint_body
        )

        async def saved_legacy_artifact(
            _run_id: str,
            artifact_key: str,
            **kwargs: Any,
        ) -> dict[str, Any] | None:
            if artifact_key == legacy_checkpoint_key:
                self.assertEqual(kwargs["input_json"], stored["input_json"])
                return copy.deepcopy(legacy_checkpoint_output)
            return None

        recovery_leaf = SimpleNamespace(
            status="completed",
            model=PROCESSING_MODEL,
            prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
            input_json=copy.deepcopy(recovery_input),
            output_json=copy.deepcopy(recovered_leaf),
        )

        async def legacy_artifact_lookup(
            _run_id: str,
            artifact_key: str,
        ) -> Any:
            if artifact_key == stored["artifact_key"]:
                return None
            if artifact_key == legacy_checkpoint_output["recovery_artifact_key"]:
                return recovery_leaf
            self.fail(f"Unexpected legacy artifact lookup: {artifact_key}")

        legacy_calls: list[str] = []

        async def legacy_processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            legacy_calls.append(artifact_key)
            if artifact_key != "entity_catalog":
                self.fail("Exact legacy checkpoint triggered a new leaf POST")
            return _deterministic_entity_catalog_union(
                list(kwargs["user_payload"]["chunk_catalogs"])
            )

        result, planner, reserve, finish = await self._run_catalog(
            legacy_processing,
            artifact_output_mock=AsyncMock(side_effect=saved_legacy_artifact),
            recovery_artifact_mock=AsyncMock(side_effect=legacy_artifact_lookup),
        )
        self.assertIsNotNone(result)
        self.assertEqual(legacy_calls, ["entity_catalog"])
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()

        tampered_recovery_input = copy.deepcopy(recovery_input)
        tampered_recovery_input["recovery"]["execution_attempt"] = 2
        with self.assertRaisesRegex(
            OpenRouterError,
            "checkpoint output binding mismatch",
        ):
            await self._run_catalog(
                AsyncMock(
                    side_effect=AssertionError(
                        "Mismatched recovery input must fail before processing"
                    )
                ),
                artifact_output_mock=AsyncMock(side_effect=saved_artifacts),
                recovery_artifact_mock=AsyncMock(
                    return_value=SimpleNamespace(
                        status="completed",
                        model=PROCESSING_MODEL,
                        prompt_version=(ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION),
                        input_json=tampered_recovery_input,
                        output_json=copy.deepcopy(recovered_leaf),
                    )
                ),
            )

        resumed_plan = _retry_plan(reused=True)
        reconcile_finish = AsyncMock()
        await self._run_catalog(
            second_processing,
            artifact_output_mock=AsyncMock(side_effect=saved_artifacts),
            recovery_artifact_mock=AsyncMock(
                return_value=SimpleNamespace(
                    status="completed",
                    model=PROCESSING_MODEL,
                    prompt_version=ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
                    input_json=copy.deepcopy(recovery_input),
                    output_json=copy.deepcopy(recovered_leaf),
                )
            ),
            epoch_binding_mock=AsyncMock(return_value=("executing", resumed_plan)),
            finish_mock=reconcile_finish,
        )
        reconcile_finish.assert_awaited_once()
        self.assertTrue(reconcile_finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            reconcile_finish.await_args.kwargs["details"]["accepted_artifact_key"],
            stored["output_json"]["recovery_artifact_key"],
        )

        tampered = copy.deepcopy(stored["output_json"])
        tampered["accepted"]["target_aliases"].append("GAMMA")

        async def tampered_checkpoint(
            _run_id: str,
            artifact_key: str,
            **_kwargs: Any,
        ) -> dict[str, Any] | None:
            if artifact_key == stored["artifact_key"]:
                return tampered
            return None

        with self.assertRaisesRegex(
            OpenRouterError,
            "checkpoint digest mismatch",
        ):
            await self._run_catalog(
                AsyncMock(
                    side_effect=AssertionError(
                        "Tampered checkpoint must fail before processing"
                    )
                ),
                artifact_output_mock=AsyncMock(side_effect=tampered_checkpoint),
                recovery_artifact_mock=AsyncMock(
                    side_effect=AssertionError(
                        "Tampered checkpoint must fail before artifact load"
                    )
                ),
            )

    async def test_ungrounded_alias_is_removed_without_blocking_recovery(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                    canonical_name="ALPHA",
                    aliases=["GAMMA"],
                    evidence="ALPHA",
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        planner = AsyncMock(return_value=_retry_plan())
        finish = AsyncMock()
        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            planner_mock=planner,
            attempts=[1, 2],
            finish_mock=finish,
        )
        alpha = next(
            item for item in result["entities"] if item["canonical_name"] == "ALPHA"
        )
        self.assertEqual(alpha["aliases"], [])
        planner.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_recovery_rebinds_entity_to_its_exact_grounded_span(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            payload = kwargs["user_payload"]
            if "_recovery_" in artifact_key:
                value = _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA GAMMA",
                    canonical_name="ALPHA",
                    aliases=[],
                    evidence="ALPHA GAMMA",
                )
                value["catalog"]["entities"].append(
                    {
                        "canonical_name": "GAMMA",
                        "aliases": [],
                        "category": "competitor",
                        "target_relationship": "competitor",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                        "evidence": "gamma",
                    }
                )
                return value
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        finish = AsyncMock()
        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            attempts=[1, 2],
            finish_mock=finish,
            answers=[{"answer_id": 1, "answer": "ALPHA GAMMA"}],
        )
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        gamma = next(
            item for item in result["entities"] if item["canonical_name"] == "GAMMA"
        )
        self.assertEqual(gamma["evidence"], "GAMMA")

    async def test_literal_entity_and_alias_in_same_quote_are_accepted(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA Alpha",
                    canonical_name="ALPHA",
                    aliases=["Alpha"],
                    evidence="ALPHA Alpha",
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[{"answer_id": 1, "answer": "ALPHA Alpha"}],
        )
        self.assertIsNotNone(result)
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()

    async def test_one_core_accepts_separate_exact_quote_per_entity(self) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                claim = payload["answers"][0]["core_claim"]
                return {
                    "catalog": {
                        "target_aliases": ["ALPHA"],
                        "entities": [
                            {
                                "canonical_name": "ALPHA",
                                "aliases": [],
                                "category": "target",
                                "target_relationship": "exact_target",
                                "commercially_relevant": True,
                                "mention_policy": "standalone",
                                "evidence": "Бренд назван: «ALPHA».",
                            },
                            {
                                "canonical_name": "BETA product",
                                "aliases": [],
                                "category": "competitor",
                                "target_relationship": "competitor",
                                "commercially_relevant": True,
                                "mention_policy": "standalone",
                                "evidence": "Альтернатива названа: «BETA product».",
                            },
                        ],
                        "uncertainties": [],
                    },
                    "core_dispositions": [
                        {
                            "claim_id": claim["claim_id"],
                            "unit_id": claim["unit_id"],
                            "core_sha256": claim["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": "ALPHA",
                            "reason": "Core содержит несколько именованных сущностей.",
                        }
                    ],
                }
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[
                {
                    "answer_id": 1,
                    "answer": "ALPHA работает рядом с BETA product.",
                }
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            {item["canonical_name"] for item in result["entities"]},
            {"ALPHA", "BETA product"},
        )
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()

    async def test_entity_name_in_prose_cannot_borrow_another_quoted_span(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                claim = payload["answers"][0]["core_claim"]
                return _catalog_candidate_with_entity(
                    claim,
                    quote="ALPHA",
                    canonical_name="GAMMA",
                    aliases=[],
                    evidence="GAMMA якобы названа здесь: «ALPHA».",
                    target_aliases=["ALPHA"],
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        finish = AsyncMock()
        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            attempts=[1, 2],
            finish_mock=finish,
            answers=[{"answer_id": 1, "answer": "ALPHA source"}],
        )
        self.assertNotIn(
            "GAMMA",
            [item["canonical_name"] for item in result["entities"]],
        )
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_outer_quote_from_another_core_cannot_ground_name(self) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                claims = [item["core_claim"] for item in payload["answers"]]
                return {
                    "catalog": {
                        "target_aliases": ["ALPHA"],
                        "entities": [
                            {
                                "canonical_name": "ALPHA",
                                "aliases": [],
                                "category": "target",
                                "target_relationship": "exact_target",
                                "commercially_relevant": True,
                                "mention_policy": "standalone",
                                "evidence": (
                                    "ALPHA приписана чужой цитате: "
                                    '«BETA unrelated» ("ALPHA").'
                                ),
                            }
                        ],
                        "uncertainties": [],
                    },
                    "core_dispositions": [
                        {
                            "claim_id": claim["claim_id"],
                            "unit_id": claim["unit_id"],
                            "core_sha256": claim["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": quote,
                            "reason": "Core содержит проверяемую строку.",
                        }
                        for claim, quote in zip(
                            claims,
                            ("ALPHA", "BETA unrelated"),
                            strict=True,
                        )
                    ],
                }
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        finish = AsyncMock()
        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            attempts=[1, 2],
            finish_mock=finish,
            answers=[
                {"answer_id": 1, "answer": "ALPHA source"},
                {"answer_id": 2, "answer": "BETA unrelated"},
            ],
        )
        self.assertNotIn(
            "ALPHA",
            [item["canonical_name"] for item in result["entities"]],
        )
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_domain_and_markdown_only_entity_quotes_are_accepted(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                claim = payload["answers"][0]["core_claim"]
                return {
                    "catalog": {
                        "target_aliases": ["ALPHA"],
                        "entities": [
                            {
                                "canonical_name": "ALPHA",
                                "aliases": [],
                                "category": "target",
                                "target_relationship": "exact_target",
                                "commercially_relevant": True,
                                "mention_policy": "standalone",
                                "evidence": "Бренд: «ALPHA».",
                            },
                            {
                                "canonical_name": "makarskatattoo.com",
                                "aliases": [],
                                "category": "other",
                                "target_relationship": "unrelated",
                                "commercially_relevant": False,
                                "mention_policy": "standalone",
                                "evidence": (
                                    "Источник: «https://makarskatattoo.com/path»."
                                ),
                            },
                        ],
                        "uncertainties": [],
                    },
                    "core_dispositions": [
                        {
                            "claim_id": claim["claim_id"],
                            "unit_id": claim["unit_id"],
                            "core_sha256": claim["core_sha256"],
                            "disposition": "grounded_fact",
                            "evidence_quote": "ALPHA",
                            "reason": "Core содержит бренд и точный домен.",
                        }
                    ],
                }
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[
                {
                    "answer_id": 1,
                    "answer": "*ALPHA* — https://makarskatattoo.com/path",
                }
            ],
        )
        self.assertIsNotNone(result)
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()

    async def test_code_owned_realweb_canonical_uses_literal_russian_alias(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="Риалвеб",
                    canonical_name="Realweb",
                    aliases=["Риалвеб"],
                    evidence="Риалвеб",
                    target_aliases=["Риалвеб"],
                )
            return _deterministic_entity_catalog_union(list(payload["chunk_catalogs"]))

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            profile={
                "brand_name": "Realweb",
                "brand_aliases": ["Риалвеб"],
                "products": [],
                "topics": [],
                "entity_scope": [],
            },
            answers=[{"answer_id": 1, "answer": "Риалвеб"}],
        )

        self.assertIsNotNone(result)
        planner.assert_not_awaited()
        reserve.assert_not_awaited()
        finish.assert_not_awaited()

    async def test_unowned_beta_canonical_is_quarantined_not_rebound_to_alias(
        self,
    ) -> None:
        async def processing(
            _run_id: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return _catalog_candidate_with_entity(
                    payload["answers"][0]["core_claim"],
                    quote="Риалвеб Gamma",
                    canonical_name="Beta",
                    aliases=["Gamma"],
                    evidence="Риалвеб Gamma",
                    target_aliases=["Риалвеб"],
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        planner = AsyncMock(return_value=_retry_plan())
        finish = AsyncMock()
        result, _planner, _reserve, _finish = await self._run_catalog(
            processing,
            planner_mock=planner,
            attempts=[1, 2],
            finish_mock=finish,
            profile={
                "brand_name": "Realweb",
                "brand_aliases": ["Риалвеб"],
                "products": [],
                "topics": [],
                "entity_scope": [],
            },
            answers=[{"answer_id": 1, "answer": "Риалвеб Gamma"}],
        )
        self.assertNotIn(
            "Beta",
            [item["canonical_name"] for item in result["entities"]],
        )
        planner.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    def test_recovery_stage_is_stable_and_distinct_per_chunk(self) -> None:
        target = {"brand_name": "ALPHA", "aliases": [], "products": []}
        first_job = {
            "artifact_key": "entity_catalog_chunk_1_alpha",
            "answers": [],
            "model_answers": [],
        }
        first = _entity_catalog_recovery_stage_key(
            job=first_job,
            target=target,
        )
        repeated = _entity_catalog_recovery_stage_key(
            job=first_job,
            target=target,
        )
        second = _entity_catalog_recovery_stage_key(
            job={
                **first_job,
                "artifact_key": "entity_catalog_chunk_2_beta",
            },
            target=target,
        )
        changed_target = _entity_catalog_recovery_stage_key(
            job=first_job,
            target={**target, "products": ["BETA"]},
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, changed_target)
        self.assertTrue(first.startswith(ENTITY_CATALOG_CONTRACT_RECOVERY_STAGE + ":"))
        self.assertLessEqual(len(first), 64)

    def test_recovery_overflow_preflight_shards_all_cores_exactly_once(
        self,
    ) -> None:
        job = _sharded_recovery_job()
        payload = _sharded_recovery_payload(job)

        class FakeStore:
            def __init__(self, plan: Any) -> None:
                self.plan = plan

            def planned_requests(self) -> tuple[Any, ...]:
                return tuple(
                    SimpleNamespace(index=index, shard_id=shard.shard_id)
                    for index, shard in enumerate(self.plan.shards)
                )

            def provider_request_utf8_bytes(
                self,
                request: Any,
                *,
                max_completion_tokens: int | None,
            ) -> bytes:
                self.assert_model_max(max_completion_tokens)
                answer_count = len(
                    self.plan.shards[request.index].payload["answers"]
                )
                return b"x" * (100 + 100 * answer_count)

            @staticmethod
            def assert_model_max(value: int | None) -> None:
                if value != 100:
                    raise AssertionError(f"unexpected model max: {value}")

        def request_bytes(*_args: Any, **kwargs: Any) -> int:
            self.assertTrue(kwargs["user_payload"]["answers"])
            return 200

        with (
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=request_bytes,
            ),
            patch(
                "app.services.analyzer.create_sharded_artifact_store",
                side_effect=lambda **kwargs: FakeStore(kwargs["plan"]),
            ),
        ):
            groups = _preflight_entity_catalog_recovery_shards(
                run_id="run-id",
                job=job,
                target=payload["target"],
                recovery_system="strict recovery",
                extraction_window={
                    "input_utf8_window": 350,
                    "model_envelope": {"max_completion_tokens": 100},
                },
                recovery_payload=payload,
                recovery_artifact_key="recovery-parent",
                source_digest=stable_digest(job["answers"]),
            )

        self.assertEqual([len(group) for group in groups], [1, 2])
        observed = [
            str(row["claim"]["unit_id"])
            for group in groups
            for row in group
        ]
        expected = [
            str(_core_unit_claim(item)["unit_id"])
            for item in job["answers"]
        ]
        self.assertEqual(observed, expected)
        self.assertEqual(len(observed), len(set(observed)))

    async def test_recovery_overflow_merges_losslessly_and_seals_manifest(
        self,
    ) -> None:
        job = _sharded_recovery_job()
        payload = _sharded_recovery_payload(job)
        save = AsyncMock()

        class FakeStore:
            def __init__(self, plan: Any) -> None:
                self.plan = plan

            def planned_requests(self) -> tuple[Any, ...]:
                return tuple(
                    SimpleNamespace(index=index, shard_id=shard.shard_id)
                    for index, shard in enumerate(self.plan.shards)
                )

            def provider_request_utf8_bytes(
                self,
                _request: Any,
                *,
                max_completion_tokens: int | None,
            ) -> bytes:
                if max_completion_tokens != 100:
                    raise AssertionError("model max drift")
                return b"x" * 250

            async def verify_provider_audit(self, *_args: Any) -> None:
                return None

            async def load_receipts(self, *_args: Any) -> list[Any]:
                return []

            async def save_receipt(self, *_args: Any) -> None:
                return None

        def request_bytes(*_args: Any, **kwargs: Any) -> int:
            return 100 + 100 * len(kwargs["user_payload"]["answers"])

        def result_for_messages(messages: list[dict[str, Any]]) -> ChatResult:
            leaf_payload = json.loads(messages[-1]["content"])
            claims = [
                copy.deepcopy(item["core_claim"])
                for item in leaf_payload["answers"]
            ]
            value = {
                "catalog": {
                    "target_aliases": ["ALPHA"],
                    "entities": [],
                    "uncertainties": [],
                },
                "core_dispositions": [
                    {
                        "claim_id": claim["claim_id"],
                        "unit_id": claim["unit_id"],
                        "core_sha256": claim["core_sha256"],
                        "disposition": "grounded_fact",
                        "evidence_quote": "ALPHA",
                        "reason": "В core дословно назван бренд ALPHA.",
                    }
                    for claim in claims
                ],
            }
            return ChatResult(
                text=json.dumps(value, ensure_ascii=False),
                parsed=value,
                citations=[],
                usage={},
                annotations=[],
                request_policy={"policy": "forbidden"},
                web_attestation={"metric_eligible": True},
                router_metadata={},
                transport={"output_complete": True},
            )

        async def load_checkpoint(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            result = result_for_messages(kwargs["messages"])
            return {
                "status": "completed",
                "manifest": {
                    "complete": True,
                    "document_sha256": text_sha256(result.text),
                },
                "resume_contract": {"sha256": "a" * 64},
                "call_records": [{"sequence": 0}],
                "partial_text": result.text,
                "parsed_value": result.parsed,
            }

        def restore_checkpoint(
            checkpoint: dict[str, Any],
            *_args: Any,
            **_kwargs: Any,
        ) -> ChatResult:
            return ChatResult(
                text=checkpoint["partial_text"],
                parsed=copy.deepcopy(checkpoint["parsed_value"]),
                citations=[],
                usage={},
                annotations=[],
                request_policy={"policy": "forbidden"},
                web_attestation={"metric_eligible": True},
                router_metadata={},
                transport={"output_complete": True},
            )

        with (
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=request_bytes,
            ),
            patch(
                "app.services.analyzer.create_sharded_artifact_store",
                side_effect=lambda **kwargs: FakeStore(kwargs["plan"]),
            ),
            patch(
                "app.services.analyzer._entity_catalog_recovery_artifact",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer._durable_structured_transport",
                new=AsyncMock(return_value=(AsyncMock(), None)),
            ),
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new=AsyncMock(
                    side_effect=lambda **kwargs: result_for_messages(
                        kwargs["messages"]
                    )
                ),
            ),
            patch(
                "app.services.analyzer.load_structured_checkpoint",
                new=AsyncMock(side_effect=load_checkpoint),
            ),
            patch(
                "app.services.analyzer.restore_completed_structured_checkpoint",
                side_effect=restore_checkpoint,
            ),
            patch("app.services.analyzer._save_artifact", save),
        ):
            recovered, manifest = await _execute_entity_catalog_recovery_shards(
                "run-id",
                job=job,
                target=payload["target"],
                profile={
                    "brand_name": "ALPHA",
                    "brand_aliases": ["ALPHA"],
                    "products": [],
                    "entity_scope": [],
                },
                recovery_system="strict recovery",
                extraction_window={
                    "input_utf8_window": 350,
                    "model_envelope": {"max_completion_tokens": 100},
                },
                recovery_payload=payload,
                recovery_artifact_key="recovery-parent",
                source_digest=stable_digest(job["answers"]),
            )

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["covered_shard_count"], 2)
        decisions = recovered["core_dispositions"]
        self.assertEqual(len(decisions), len(job["answers"]))
        self.assertEqual(
            [row["unit_id"] for row in decisions],
            [
                _core_unit_claim(item)["unit_id"]
                for item in job["answers"]
            ],
        )
        completed = [
            call.kwargs
            for call in save.await_args_list
            if call.kwargs.get("status") == "completed"
            and call.kwargs.get("artifact_key") == "recovery-parent"
        ]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["output_json"], recovered)
        self.assertEqual(
            completed[0]["usage_json"]["_aiv_sharded_document"],
            manifest,
        )

    async def test_cached_overflow_requires_direct_or_sharded_receipt(
        self,
    ) -> None:
        job = _sharded_recovery_job()
        payload = _sharded_recovery_payload(job)
        cached = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [],
        }
        artifact = SimpleNamespace(
            usage_json={},
            raw_text=json.dumps(cached, ensure_ascii=False),
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=copy.deepcopy(cached)),
            ),
            patch(
                "app.services.analyzer._entity_catalog_recovery_artifact",
                new=AsyncMock(return_value=artifact),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(),
            ) as save,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "no complete direct or sharded execution receipt",
            ):
                await _execute_entity_catalog_recovery_shards(
                    "run-id",
                    job=job,
                    target=payload["target"],
                    profile={"brand_name": "ALPHA"},
                    recovery_system="strict recovery",
                    extraction_window={
                        "input_utf8_window": 350,
                        "model_envelope": {"max_completion_tokens": 100},
                    },
                    recovery_payload=payload,
                    recovery_artifact_key="recovery-parent",
                    source_digest=stable_digest(job["answers"]),
                )
        save.assert_not_awaited()

    async def test_completed_parent_is_reused_before_drifted_preflight(
        self,
    ) -> None:
        def candidate_for(payload: dict[str, Any], *, quote: str) -> dict[str, Any]:
            claims = [
                copy.deepcopy(answer["core_claim"])
                for answer in payload["answers"]
            ]
            return {
                "catalog": {
                    "target_aliases": [quote],
                    "entities": [],
                    "uncertainties": [],
                },
                "core_dispositions": [
                    {
                        "claim_id": claim["claim_id"],
                        "unit_id": claim["unit_id"],
                        "core_sha256": claim["core_sha256"],
                        "disposition": "grounded_fact",
                        "evidence_quote": quote,
                        "reason": "В core дословно назван бренд ALPHA.",
                    }
                    for claim in claims
                ],
            }

        processing_calls: list[str] = []

        async def processing(_run_id: str, **kwargs: Any) -> dict[str, Any]:
            artifact_key = str(kwargs["artifact_key"])
            processing_calls.append(artifact_key)
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return candidate_for(payload, quote="Alpha")
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        recovered_holder: dict[str, dict[str, Any]] = {}
        manifest_holder: dict[str, dict[str, Any]] = {}

        async def artifact_output(
            _run_id: str,
            artifact_key: str,
            **kwargs: Any,
        ) -> dict[str, Any] | None:
            if artifact_key.endswith("_accepted"):
                return None
            if artifact_key.endswith("_recovery_e4_a1"):
                payload = kwargs["input_json"]
                recovered = candidate_for(payload, quote="ALPHA")
                recovered_holder[artifact_key] = copy.deepcopy(recovered)
                return recovered
            return None

        async def recovery_artifact(
            _run_id: str,
            artifact_key: str,
        ) -> Any | None:
            recovered = recovered_holder.get(artifact_key)
            if recovered is None:
                return None
            manifest = {
                "complete": True,
                "shard_count": 2,
                "covered_shard_count": 2,
                "document_sha256": stable_digest(recovered),
            }
            manifest_holder[artifact_key] = copy.deepcopy(manifest)
            return SimpleNamespace(
                usage_json={"_aiv_sharded_document": manifest},
                raw_text=None,
            )

        def request_bytes(*_args: Any, **kwargs: Any) -> int:
            payload = kwargs["user_payload"]
            return 20_000_000 if "recovery" in payload else 100

        reserve = AsyncMock()
        save = AsyncMock()
        preflight = patch(
            "app.services.analyzer._preflight_entity_catalog_recovery_shards",
            side_effect=AssertionError(
                "A completed exact parent must bypass fresh shard preflight"
            ),
        )
        with (
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=request_bytes,
            ),
            preflight as preflight_mock,
        ):
            result, _planner, _reserve, finish = await self._run_catalog(
                processing,
                plan=_retry_plan(reused=True),
                reserve_mock=reserve,
                save_mock=save,
                execution_state_mock=AsyncMock(
                    return_value=SimpleNamespace(
                        status="executing",
                        execution_attempts=1,
                    )
                ),
                artifact_output_mock=AsyncMock(side_effect=artifact_output),
                recovery_artifact_mock=AsyncMock(side_effect=recovery_artifact),
                answers=[
                    {"answer_id": 1, "answer": "ALPHA — первый ответ."},
                    {"answer_id": 2, "answer": "ALPHA — второй ответ."},
                ],
            )

        self.assertIsNotNone(result)
        preflight_mock.assert_not_called()
        reserve.assert_not_awaited()
        self.assertFalse(any("_recovery_" in key for key in processing_calls))
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            finish.await_args.kwargs["details"]["execution_attempt"],
            1,
        )
        checkpoint = save.await_args.kwargs["output_json"]
        parent_key = checkpoint["recovery_artifact_key"]
        self.assertEqual(
            checkpoint["recovery_execution_manifest_sha256"],
            stable_digest(manifest_holder[parent_key]),
        )

    async def test_current_parent_routes_through_deep_cache_executor(
        self,
    ) -> None:
        def candidate_for(payload: dict[str, Any], *, quote: str) -> dict[str, Any]:
            claims = [
                copy.deepcopy(answer["core_claim"])
                for answer in payload["answers"]
            ]
            return {
                "catalog": {
                    "target_aliases": [quote],
                    "entities": [],
                    "uncertainties": [],
                },
                "core_dispositions": [
                    {
                        "claim_id": claim["claim_id"],
                        "unit_id": claim["unit_id"],
                        "core_sha256": claim["core_sha256"],
                        "disposition": "grounded_fact",
                        "evidence_quote": quote,
                        "reason": "В core дословно назван бренд ALPHA.",
                    }
                    for claim in claims
                ],
            }

        async def processing(_run_id: str, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            if "answers" in payload:
                return candidate_for(payload, quote="Alpha")
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        cached_result: dict[str, Any] = {}
        cached_manifest = {
            "version": ENTITY_CATALOG_RECOVERY_PARENT_MANIFEST_VERSION,
            "complete": True,
        }

        async def cached_parent(
            _run_id: str,
            **kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            recovered = candidate_for(
                kwargs["recovery_payload"],
                quote="ALPHA",
            )
            cached_result.clear()
            cached_result.update(copy.deepcopy(recovered))
            return recovered, copy.deepcopy(cached_manifest)

        executor = AsyncMock(
            side_effect=lambda *_args, **_kwargs: (
                copy.deepcopy(cached_result),
                copy.deepcopy(cached_manifest),
            )
        )

        def request_bytes(*_args: Any, **kwargs: Any) -> int:
            if "recovery" in kwargs["user_payload"]:
                raise AssertionError(
                    "current parent validation must not remeasure or POST"
                )
            return 100

        with (
            patch(
                "app.services.analyzer._cached_entity_catalog_recovery_parent",
                new=AsyncMock(side_effect=cached_parent),
            ) as cache_lookup,
            patch(
                "app.services.analyzer._execute_entity_catalog_recovery_shards",
                new=executor,
            ),
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=request_bytes,
            ) as request_measurement,
        ):
            result, _planner, reserve, finish = await self._run_catalog(
                processing,
            )

        self.assertIsNotNone(result)
        cache_lookup.assert_awaited_once()
        executor.assert_awaited_once()
        self.assertTrue(request_measurement.called)
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

    async def test_direct_failover_singleton_is_terminal_without_attempt_two(
        self,
    ) -> None:
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()
        executor = AsyncMock(
            side_effect=_EntityCatalogRecoverySingletonContextError(
                "minimum core exhausted context"
            )
        )
        recovery_processing_calls = 0

        async def processing(_run_id: str, **kwargs: Any) -> dict[str, Any]:
            nonlocal recovery_processing_calls
            payload = kwargs["user_payload"]
            failover = kwargs.get("composable_failover")
            if failover is not None:
                recovery_processing_calls += 1
                request = {
                    "reason": "prefix_context_exhausted",
                    "model": PROCESSING_MODEL,
                    "schema_name": kwargs["schema_name"],
                    "document_id": (
                        f"run-id:{kwargs['artifact_key']}:"
                        f"{stable_digest(payload)[:20]}"
                    ),
                    "response_schema": copy.deepcopy(kwargs["schema"]),
                }
                try:
                    await failover(request)
                except Exception as exc:
                    raise OpenRouterStructuredContinuationError(
                        "wrapped direct failover",
                        result=ChatResult(
                            text="{",
                            parsed=None,
                            citations=[],
                            usage={},
                            annotations=[],
                            request_policy={"policy": "forbidden"},
                            web_attestation={"metric_eligible": True},
                            router_metadata={},
                            transport={"output_complete": False},
                        ),
                        manifest={"complete": False},
                    ) from exc
                raise AssertionError("singleton failover unexpectedly succeeded")
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        with (
            patch(
                "app.services.analyzer._execute_entity_catalog_recovery_shards",
                executor,
            ),
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("direct semantic test must not POST"),
            ) as client_factory,
            self.assertRaises(OpenRouterError),
        ):
            await self._run_catalog(
                processing,
                reserve_mock=reserve,
                finish_mock=finish,
            )

        client_factory.assert_not_called()
        self.assertEqual(recovery_processing_calls, 1)
        self.assertEqual(executor.await_count, 1)
        self.assertEqual(reserve.await_count, 1)
        finish.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["succeeded"])
        failures = finish.await_args.kwargs["details"]["attempt_failures"]
        self.assertEqual(
            failures[0]["failure_kind"],
            "singleton_prefix_context_exhausted",
        )

    async def test_direct_semantic_failover_advances_bounded_attempt_two(
        self,
    ) -> None:
        reserve = AsyncMock(side_effect=[1, 2])
        finish = AsyncMock()
        executor_calls = 0

        async def executor(
            _run_id: str,
            **kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal executor_calls
            executor_calls += 1
            if executor_calls == 1:
                raise _EntityCatalogRecoveryShardSemanticError(
                    "complete child failed semantic validation"
                )
            recovered = {
                "catalog": {
                    "target_aliases": ["ALPHA"],
                    "entities": [],
                    "uncertainties": [],
                },
                "core_dispositions": [
                    {
                        "claim_id": claim["claim_id"],
                        "unit_id": claim["unit_id"],
                        "core_sha256": claim["core_sha256"],
                        "disposition": "grounded_fact",
                        "evidence_quote": "ALPHA",
                        "reason": "В core дословно назван бренд ALPHA.",
                    }
                    for claim in [
                        _core_unit_claim(item)
                        for item in kwargs["job"]["answers"]
                    ]
                ],
            }
            manifest = {
                "complete": True,
                "coverage_complete": True,
                "response_mode": "partitioned",
                "shard_count": 1,
                "covered_shard_count": 1,
                "document_sha256": stable_digest(recovered),
            }
            return recovered, manifest

        async def processing(_run_id: str, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["user_payload"]
            failover = kwargs.get("composable_failover")
            if failover is not None:
                request = {
                    "reason": "prefix_context_exhausted",
                    "model": PROCESSING_MODEL,
                    "schema_name": kwargs["schema_name"],
                    "document_id": (
                        f"run-id:{kwargs['artifact_key']}:"
                        f"{stable_digest(payload)[:20]}"
                    ),
                    "response_schema": copy.deepcopy(kwargs["schema"]),
                }
                try:
                    result = await failover(request)
                except Exception as exc:
                    raise OpenRouterStructuredContinuationError(
                        "wrapped direct failover",
                        result=ChatResult(
                            text="{",
                            parsed=None,
                            citations=[],
                            usage={},
                            annotations=[],
                            request_policy={"policy": "forbidden"},
                            web_attestation={"metric_eligible": True},
                            router_metadata={},
                            transport={"output_complete": False},
                        ),
                        manifest={"complete": False},
                    ) from exc
                assert isinstance(result.parsed, dict)
                return copy.deepcopy(result.parsed)
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        with (
            patch(
                "app.services.analyzer._execute_entity_catalog_recovery_shards",
                new=AsyncMock(side_effect=executor),
            ),
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError(
                    "completed child checkpoint reuse must not POST"
                ),
            ) as client_factory,
        ):
            result, _planner, _reserve, _finish = await self._run_catalog(
                processing,
                reserve_mock=reserve,
                finish_mock=finish,
                attempts=[1, 2],
            )

        client_factory.assert_not_called()
        self.assertIsNotNone(result)
        self.assertEqual(executor_calls, 2)
        self.assertEqual(reserve.await_count, 2)
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            finish.await_args.kwargs["details"]["execution_attempt"],
            2,
        )

    async def test_ambiguous_markdown_failure_stays_fail_closed(self) -> None:
        planner = AsyncMock()
        with (
            patch(
                "app.services.analyzer.settings.PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer.plan_durable_recovery",
                planner,
            ),
        ):
            from app.services.analyzer import (
                _entity_catalog_quote_recovery_incident,
            )

            admitted = _entity_catalog_quote_recovery_incident(
                OpenRouterError("Core-unit Markdown-normalized quote is ambiguous"),
                candidate={"core_dispositions": []},
                expected_claims=[],
            )

        self.assertIsNone(admitted)
        planner.assert_not_awaited()


class EntityCatalogRecoveryContinuationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self._temp_dir.name) / "recovery.sqlite3"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{database_path}",
            echo=False,
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
        self._patches = [
            patch(
                "app.services.analyzer.SessionLocal",
                self.SessionLocal,
            ),
            patch.object(audit_store, "SessionLocal", self.SessionLocal),
            patch(
                "app.services.analyzer.assert_run_lease",
                new=AsyncMock(),
            ),
            patch.object(
                audit_store,
                "assert_run_lease",
                new=AsyncMock(),
            ),
        ]
        for active_patch in self._patches:
            active_patch.start()

    async def asyncTearDown(self) -> None:
        for active_patch in reversed(self._patches):
            active_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    async def _seal_singleton_recovery_leaf(
        self,
        *,
        recovery_system: str = "strict recovery",
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        _RecoveryPrefixClient,
    ]:
        job = _sharded_recovery_job(count=1)
        payload = _sharded_recovery_payload(job)
        profile = {
            "brand_name": "ALPHA",
            "brand_aliases": ["ALPHA"],
            "products": [],
            "entity_scope": [],
        }
        claim = payload["answers"][0]["core_claim"]
        candidate = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [
                {
                    "claim_id": claim["claim_id"],
                    "unit_id": claim["unit_id"],
                    "core_sha256": claim["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "ALPHA",
                    "reason": "В core назван бренд ALPHA. " + "факт " * 180,
                }
            ],
        }
        full_text = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prefix_text = full_text[:-32]
        self.assertGreater(len(prefix_text), 512)
        client = _RecoveryPrefixClient(full_text, prefix_text)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": PROCESSING_MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        extraction_window = {
            "input_utf8_window": 2_000_000,
            "model_envelope": envelope,
        }
        real_headroom = openrouter_service._apply_model_output_headroom

        def exhaust_continuation_headroom(
            **kwargs: Any,
        ) -> tuple[int | None, dict[str, Any]]:
            messages = kwargs["payload"].get("messages") or []
            if messages and "CONTINUATION_CONTRACT_JSON:" in str(
                messages[-1].get("content") or ""
            ):
                raise openrouter_service.OpenRouterContextHeadroomError(
                    "simulated minimum-core prefix context exhaustion"
                )
            return real_headroom(**kwargs)

        with (
            patch(
                "app.services.openrouter._apply_model_output_headroom",
                side_effect=exhaust_continuation_headroom,
            ),
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
            self.assertRaises(_EntityCatalogRecoverySingletonContextError),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system=recovery_system,
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key="singleton-parent",
                source_digest=stable_digest(job["answers"]),
            )
        self.assertEqual(client.request_kinds, ["initial"])
        return job, payload, profile, extraction_window, client

    async def test_singleton_marker_is_physical_absorbing_and_contract_addressed(
        self,
    ) -> None:
        job, payload, profile, extraction_window, _client = (
            await self._seal_singleton_recovery_leaf()
        )
        async with self.SessionLocal() as session:
            first_leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            marker = first_leaf.usage_json[
                "_aiv_entity_catalog_recovery_singleton_block"
            ]
            first_leaf_key = first_leaf.artifact_key
        self.assertEqual(first_leaf.status, "failed")
        self.assertEqual(marker["physical_call_count"], 1)
        self.assertEqual(marker["model"], PROCESSING_MODEL)
        self.assertEqual(
            marker["source_identity_sha256"],
            stable_digest(first_leaf.input_json),
        )

        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError(
                    "identical singleton negative cache must not POST"
                ),
            ) as client_factory,
            self.assertRaises(_EntityCatalogRecoverySingletonContextError),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key="singleton-parent",
                source_digest=stable_digest(job["answers"]),
            )
        client_factory.assert_not_called()

        await self._seal_singleton_recovery_leaf(
            recovery_system="materially changed recovery contract"
        )
        async with self.SessionLocal() as session:
            leaves = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalars().all()
        self.assertEqual(len(leaves), 2)
        self.assertEqual(len({leaf.artifact_key for leaf in leaves}), 2)
        self.assertIn(first_leaf_key, {leaf.artifact_key for leaf in leaves})

    async def test_singleton_marker_tamper_and_missing_checkpoint_are_network_free(
        self,
    ) -> None:
        job, payload, profile, extraction_window, _client = (
            await self._seal_singleton_recovery_leaf()
        )
        async with self.SessionLocal() as session:
            leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            leaf.usage_json[
                "_aiv_entity_catalog_recovery_singleton_block"
            ]["acceptance_policy_sha256"] = "0" * 64
            flag_modified(leaf, "usage_json")
            await session.commit()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("tampered singleton must not POST"),
            ) as tamper_factory,
            self.assertRaisesRegex(OpenRouterError, "tampered or stale"),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key="singleton-parent",
                source_digest=stable_digest(job["answers"]),
            )
        tamper_factory.assert_not_called()

        async with self.SessionLocal() as session:
            leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            leaf.usage_json[
                "_aiv_entity_catalog_recovery_singleton_block"
            ]["acceptance_policy_sha256"] = (
                ENTITY_CATALOG_RECOVERY_ACCEPTANCE_POLICY_SHA256
            )
            flag_modified(leaf, "usage_json")
            structured_rows = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("lsa2_%"),
                    )
                )
            ).scalars().all()
            self.assertTrue(structured_rows)
            for row in structured_rows:
                await session.delete(row)
            await session.commit()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError(
                    "missing singleton checkpoint must not POST"
                ),
            ) as missing_factory,
            self.assertRaisesRegex(OpenRouterError, "checkpoint is missing"),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key="singleton-parent",
                source_digest=stable_digest(job["answers"]),
            )
        missing_factory.assert_not_called()

    async def _complete_current_parent(
        self,
        *,
        artifact_key: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        job = _sharded_recovery_job(count=2)
        payload = _sharded_recovery_payload(job)
        profile = {
            "brand_name": "ALPHA",
            "brand_aliases": ["ALPHA"],
            "products": [],
            "entity_scope": [],
        }
        candidate = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [
                {
                    "claim_id": item["core_claim"]["claim_id"],
                    "unit_id": item["core_claim"]["unit_id"],
                    "core_sha256": item["core_claim"]["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "ALPHA",
                    "reason": "В core дословно назван бренд ALPHA.",
                }
                for item in payload["answers"]
            ],
        }
        full_text = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        client = _RecoveryPrefixClient(full_text, full_text)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": PROCESSING_MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        extraction_window = {
            "input_utf8_window": 2_000_000,
            "model_envelope": envelope,
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
            recovered, manifest = (
                await _execute_entity_catalog_recovery_shards(
                    self.run_id,
                    job=job,
                    target=payload["target"],
                    profile=profile,
                    recovery_system="strict recovery",
                    extraction_window=extraction_window,
                    recovery_payload=payload,
                    recovery_artifact_key=artifact_key,
                    source_digest=stable_digest(job["answers"]),
                )
            )
        self.assertEqual(recovered, candidate)
        self.assertIsNotNone(manifest)
        return job, payload, profile, extraction_window

    async def test_terminal_semantic_leaf_is_absorbing_and_tamper_fails_without_http(
        self,
    ) -> None:
        job = _sharded_recovery_job(count=2)
        payload = _sharded_recovery_payload(job)
        profile = {
            "brand_name": "ALPHA",
            "brand_aliases": ["ALPHA"],
            "products": [],
            "entity_scope": [],
        }
        claims = [item["core_claim"] for item in payload["answers"]]
        invalid = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [
                {
                    "claim_id": claim["claim_id"],
                    "unit_id": (
                        "foreign-unit" if index == 0 else claim["unit_id"]
                    ),
                    "core_sha256": claim["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "ALPHA",
                    "reason": "В core дословно назван бренд ALPHA.",
                }
                for index, claim in enumerate(claims)
            ],
        }
        full_text = json.dumps(
            invalid,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        client = _RecoveryPrefixClient(full_text, full_text)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": PROCESSING_MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        extraction_window = {
            "input_utf8_window": 2_000_000,
            "model_envelope": envelope,
        }
        call = lambda: _execute_entity_catalog_recovery_shards(
            self.run_id,
            job=job,
            target=payload["target"],
            profile=profile,
            recovery_system="strict recovery",
            extraction_window=extraction_window,
            recovery_payload=payload,
            recovery_artifact_key="semantic-parent",
            source_digest=stable_digest(job["answers"]),
        )
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
            self.assertRaisesRegex(OpenRouterError, "identity mismatch"),
        ):
            await call()
        self.assertEqual(client.request_kinds.count("initial"), 1)

        async with self.SessionLocal() as session:
            leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            self.assertEqual(leaf.status, "failed")
            marker = leaf.usage_json[
                "_aiv_entity_catalog_recovery_semantic_block"
            ]
            self.assertEqual(marker["parsed_output_sha256"], stable_digest(invalid))

        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("sealed semantic leaf must not use HTTP"),
            ) as client_factory,
            self.assertRaisesRegex(OpenRouterError, "identity mismatch"),
        ):
            await call()
        client_factory.assert_not_called()

        async with self.SessionLocal() as session:
            leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            leaf.usage_json[
                "_aiv_entity_catalog_recovery_semantic_block"
            ]["acceptance_policy_sha256"] = "0" * 64
            flag_modified(leaf, "usage_json")
            await session.commit()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("tampered marker must not use HTTP"),
            ) as tamper_factory,
            self.assertRaisesRegex(OpenRouterError, "tampered or stale"),
        ):
            await call()
        tamper_factory.assert_not_called()

    async def test_current_parent_missing_leaf_fails_without_http(
        self,
    ) -> None:
        artifact_key = "recovery-parent-missing-leaf"
        job, payload, profile, extraction_window = (
            await self._complete_current_parent(artifact_key=artifact_key)
        )
        async with self.SessionLocal() as session:
            leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            await session.delete(leaf)
            await session.commit()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError(
                    "parent cache validation must not use HTTP"
                ),
            ) as client_factory,
            self.assertRaisesRegex(
                OpenRouterError,
                "missing child leaf",
            ),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key=artifact_key,
                source_digest=stable_digest(job["answers"]),
            )
        client_factory.assert_not_called()

    async def test_current_parent_missing_checkpoint_fails_without_http(
        self,
    ) -> None:
        artifact_key = "recovery-parent-missing-checkpoint"
        job, payload, profile, extraction_window = (
            await self._complete_current_parent(artifact_key=artifact_key)
        )
        async with self.SessionLocal() as session:
            structured_rows = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("lsa2_%"),
                    )
                )
            ).scalars().all()
            self.assertTrue(structured_rows)
            for row in structured_rows:
                await session.delete(row)
            await session.commit()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError(
                    "checkpoint validation must not use HTTP"
                ),
            ) as client_factory,
            self.assertRaisesRegex(
                OpenRouterError,
                "physical receipts are missing or mismatched",
            ),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key=artifact_key,
                source_digest=stable_digest(job["answers"]),
            )
        client_factory.assert_not_called()

    async def test_output_limited_prefix_resumes_without_repeating_post(
        self,
    ) -> None:
        job = _sharded_recovery_job(count=2)
        payload = _sharded_recovery_payload(job)
        dispositions = []
        for item in payload["answers"]:
            claim = item["core_claim"]
            dispositions.append(
                {
                    "claim_id": claim["claim_id"],
                    "unit_id": claim["unit_id"],
                    "core_sha256": claim["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "ALPHA",
                    "reason": (
                        "В core дословно назван бренд ALPHA. " + "факт " * 180
                    ),
                }
            )
        candidate = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": dispositions,
        }
        catalog_text = json.dumps(
            candidate["catalog"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        disposition_texts = [
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for value in dispositions
        ]
        prefix_text = (
            '{"catalog":'
            + catalog_text
            + ',"core_dispositions":['
            + disposition_texts[0]
            + ","
        )
        full_text = prefix_text + disposition_texts[1] + "]}"
        self.assertGreater(len(prefix_text), 512)
        client = _RecoveryPrefixClient(full_text, prefix_text)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": PROCESSING_MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        extraction_window = {
            "input_utf8_window": 2_000_000,
            "model_envelope": envelope,
        }
        profile = {
            "brand_name": "ALPHA",
            "brand_aliases": ["ALPHA"],
            "products": [],
            "entity_scope": [],
        }
        source_digest = stable_digest(job["answers"])
        real_persist = audit_store.persist_structured_audit_event
        crash_state = {"raised": False}

        async def persist_receipt_then_crash(
            *args: Any,
            **kwargs: Any,
        ) -> None:
            await real_persist(*args, **kwargs)
            event = kwargs["event"]
            if (
                not crash_state["raised"]
                and event.get("event_kind") == "provider_post"
                and event.get("sequence") == 0
            ):
                crash_state["raised"] = True
                raise RuntimeError("simulated crash after durable provider receipt")

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
            patch.object(
                audit_store,
                "persist_structured_audit_event",
                new=persist_receipt_then_crash,
            ),
            self.assertRaises(RuntimeError),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key="recovery-parent",
                source_digest=source_digest,
            )

        self.assertTrue(crash_state["raised"])
        self.assertEqual(client.request_kinds, ["initial"])

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
            recovered, manifest = (
                await _execute_entity_catalog_recovery_shards(
                    self.run_id,
                    job=job,
                    target=payload["target"],
                    profile=profile,
                    recovery_system="strict recovery",
                    extraction_window=extraction_window,
                    recovery_payload=payload,
                    recovery_artifact_key="recovery-parent",
                    source_digest=source_digest,
                )
            )

        self.assertEqual(client.request_kinds, ["initial", "continuation"])
        self.assertEqual(recovered, candidate)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertTrue(manifest["complete"])
        self.assertTrue(manifest["coverage_complete"])
        self.assertEqual(manifest["document_sha256"], stable_digest(candidate))

        with patch(
            "app.services.openrouter.httpx.AsyncClient",
            side_effect=AssertionError("completed parent must be network-free"),
        ) as client_factory:
            reused, reused_manifest = (
                await _execute_entity_catalog_recovery_shards(
                    self.run_id,
                    job=job,
                    target=payload["target"],
                    profile=profile,
                    recovery_system="strict recovery",
                    extraction_window=extraction_window,
                    recovery_payload=payload,
                    recovery_artifact_key="recovery-parent",
                    source_digest=source_digest,
                )
            )
        client_factory.assert_not_called()
        self.assertEqual(reused, recovered)
        self.assertEqual(reused_manifest, manifest)

        async with self.SessionLocal() as session:
            parent = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "recovery-parent",
                    )
                )
            ).scalar_one()
            receipts = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("lsa2_receipt_%"),
                    )
                )
            ).scalars().all()
        self.assertEqual(parent.status, "completed")
        self.assertEqual(parent.output_json, candidate)
        self.assertEqual(len(receipts), 2)

        async with self.SessionLocal() as session:
            parent = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "recovery-parent",
                    )
                )
            ).scalar_one()
            parent.status = "failed"
            await session.commit()
        with patch(
            "app.services.openrouter.httpx.AsyncClient",
            side_effect=AssertionError(
                "completed leaf reuse must be network-free"
            ),
        ) as leaf_reuse_factory:
            leaf_reused, _leaf_reused_manifest = (
                await _execute_entity_catalog_recovery_shards(
                    self.run_id,
                    job=job,
                    target=payload["target"],
                    profile=profile,
                    recovery_system="strict recovery",
                    extraction_window=extraction_window,
                    recovery_payload=payload,
                    recovery_artifact_key="recovery-parent",
                    source_digest=source_digest,
                )
            )
        leaf_reuse_factory.assert_not_called()
        self.assertEqual(leaf_reused, candidate)

        async with self.SessionLocal() as session:
            parent = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "recovery-parent",
                    )
                )
            ).scalar_one()
            leaf = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key.like("ecr_leaf_%"),
                    )
                )
            ).scalar_one()
            parent.status = "failed"
            assert isinstance(leaf.raw_text, str)
            leaf.raw_text += " "
            await session.commit()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("tamper path must be network-free"),
            ) as tamper_factory,
            self.assertRaisesRegex(
                OpenRouterError,
                "tampered or mismatched",
            ),
        ):
            await _execute_entity_catalog_recovery_shards(
                self.run_id,
                job=job,
                target=payload["target"],
                profile=profile,
                recovery_system="strict recovery",
                extraction_window=extraction_window,
                recovery_payload=payload,
                recovery_artifact_key="recovery-parent",
                source_digest=source_digest,
            )
        tamper_factory.assert_not_called()

    async def test_overflow_route_reaches_harness_and_seals_checkpoint(
        self,
    ) -> None:
        job = _sharded_recovery_job(count=2)
        target = {"brand_name": "ALPHA", "aliases": ["ALPHA"]}
        profile = {
            "brand_name": "ALPHA",
            "brand_aliases": ["ALPHA"],
            "products": [],
            "entity_scope": [],
        }
        claims = [
            copy.deepcopy(item["core_claim"])
            for item in job["model_answers"]
        ]
        invalid_candidate = {
            "catalog": {
                "target_aliases": ["Alpha"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [
                {
                    "claim_id": claim["claim_id"],
                    "unit_id": claim["unit_id"],
                    "core_sha256": claim["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "Alpha",
                    "reason": "Регистр цитаты намеренно нарушен.",
                }
                for claim in claims
            ],
        }
        recovered_candidate = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [
                {
                    "claim_id": claim["claim_id"],
                    "unit_id": claim["unit_id"],
                    "core_sha256": claim["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "ALPHA",
                    "reason": "В core дословно назван бренд ALPHA.",
                }
                for claim in claims
            ],
        }
        full_text = json.dumps(
            recovered_candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        client = _RecoveryPrefixClient(full_text, full_text)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": PROCESSING_MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        input_window = 2_000_000
        plan = _retry_plan()
        plan.run_id = self.run_id
        finish = AsyncMock()

        def measured_request_bytes(*_args: Any, **kwargs: Any) -> int:
            user_payload = kwargs["user_payload"]
            recovery_contract = user_payload.get("recovery")
            if (
                isinstance(recovery_contract, dict)
                and "shard_source_binding" not in recovery_contract
            ):
                return input_window + 1
            return 1_000

        with (
            patch(
                "app.services.analyzer.settings.PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.recovery_execution_state",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        status="planned",
                        execution_attempts=0,
                    )
                ),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new=AsyncMock(return_value=1),
            ) as reserve,
            patch(
                "app.services.analyzer.finish_recovery",
                finish,
            ),
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                side_effect=measured_request_bytes,
            ),
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
            accepted = await _recover_entity_catalog_chunk_contract(
                self.run_id,
                job=job,
                target=target,
                profile=profile,
                extraction_system="strict extraction",
                extraction_window={
                    "input_utf8_window": input_window,
                    "model_envelope": envelope,
                },
                candidate=invalid_candidate,
                contract_error=OpenRouterError(
                    "Core-unit quote is not an exact core substring: "
                    + str(claims[0]["unit_id"])
                ),
            )

        self.assertEqual(
            accepted["target_aliases"],
            recovered_candidate["catalog"]["target_aliases"],
        )
        self.assertEqual(client.request_kinds, ["initial"])
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

        async with self.SessionLocal() as session:
            artifacts = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                    )
                )
            ).scalars().all()
        checkpoints = [
            artifact
            for artifact in artifacts
            if artifact.model is None
            and artifact.status == "completed"
            and isinstance(artifact.output_json, dict)
            and "checkpoint_sha256" in artifact.output_json
        ]
        parents = [
            artifact
            for artifact in artifacts
            if artifact.status == "completed"
            and isinstance(artifact.usage_json, dict)
            and "_aiv_sharded_document" in artifact.usage_json
        ]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(len(parents), 1)
        checkpoint = checkpoints[0].output_json
        assert isinstance(checkpoint, dict)
        self.assertEqual(
            checkpoint["recovery_execution_manifest_sha256"],
            stable_digest(
                parents[0].usage_json["_aiv_sharded_document"]
            ),
        )
        self.assertEqual(
            checkpoint["identity"]["source_units_sha256"],
            stable_digest(job["answers"]),
        )

    async def test_fitting_direct_recovery_repartitions_on_headroom(
        self,
    ) -> None:
        job = _sharded_recovery_job(count=2)
        target = {"brand_name": "ALPHA", "aliases": ["ALPHA"]}
        profile = {
            "brand_name": "ALPHA",
            "brand_aliases": ["ALPHA"],
            "products": [],
            "entity_scope": [],
        }
        claims = [
            copy.deepcopy(item["core_claim"])
            for item in job["model_answers"]
        ]
        invalid_candidate = {
            "catalog": {
                "target_aliases": ["Alpha"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": [
                {
                    "claim_id": claim["claim_id"],
                    "unit_id": claim["unit_id"],
                    "core_sha256": claim["core_sha256"],
                    "disposition": "grounded_fact",
                    "evidence_quote": "Alpha",
                    "reason": "Регистр цитаты намеренно нарушен.",
                }
                for claim in claims
            ],
        }
        dispositions = [
            {
                "claim_id": claim["claim_id"],
                "unit_id": claim["unit_id"],
                "core_sha256": claim["core_sha256"],
                "disposition": "grounded_fact",
                "evidence_quote": "ALPHA",
                "reason": (
                    "В core дословно назван бренд ALPHA. " + "факт " * 180
                ),
            }
            for claim in claims
        ]
        recovered_candidate = {
            "catalog": {
                "target_aliases": ["ALPHA"],
                "entities": [],
                "uncertainties": [],
            },
            "core_dispositions": dispositions,
        }
        catalog_text = json.dumps(
            recovered_candidate["catalog"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        disposition_texts = [
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for value in dispositions
        ]
        prefix_text = (
            '{"catalog":'
            + catalog_text
            + ',"core_dispositions":['
            + disposition_texts[0]
            + ","
        )
        full_text = prefix_text + disposition_texts[1] + "]}"
        self.assertGreater(len(prefix_text), 512)
        client = _RecoveryHeadroomFailoverClient(full_text, prefix_text)
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": PROCESSING_MODEL,
            "resolution": "test",
            "context_length": 1_000_000,
            "max_completion_tokens": 65_536,
        }
        input_window = 2_000_000
        plan = _retry_plan()
        plan.run_id = self.run_id
        finish = AsyncMock()
        real_headroom = openrouter_service._apply_model_output_headroom
        continuation_headroom_attempts = 0

        def fail_direct_continuation_headroom(
            **kwargs: Any,
        ) -> tuple[int | None, dict[str, Any]]:
            nonlocal continuation_headroom_attempts
            messages = kwargs["payload"].get("messages") or []
            if messages and "CONTINUATION_CONTRACT_JSON:" in str(
                messages[-1].get("content") or ""
            ):
                continuation_headroom_attempts += 1
                raise openrouter_service.OpenRouterContextHeadroomError(
                    "simulated prefix context exhaustion"
                )
            return real_headroom(**kwargs)

        with (
            patch(
                "app.services.analyzer.settings.PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.recovery_execution_state",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        status="planned",
                        execution_attempts=0,
                    )
                ),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new=AsyncMock(return_value=1),
            ) as reserve,
            patch(
                "app.services.analyzer.finish_recovery",
                finish,
            ),
            patch(
                "app.services.analyzer._structured_provider_request_utf8_bytes",
                return_value=1_000,
            ) as request_measurement,
            patch(
                "app.services.openrouter._apply_model_output_headroom",
                side_effect=fail_direct_continuation_headroom,
            ),
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
            accepted = await _recover_entity_catalog_chunk_contract(
                self.run_id,
                job=job,
                target=target,
                profile=profile,
                extraction_system="strict extraction",
                extraction_window={
                    "input_utf8_window": input_window,
                    "model_envelope": envelope,
                },
                candidate=invalid_candidate,
                contract_error=OpenRouterError(
                    "Core-unit quote is not an exact core substring: "
                    + str(claims[0]["unit_id"])
                ),
            )

        self.assertEqual(accepted["target_aliases"], ["ALPHA"])
        self.assertEqual(
            client.request_kinds,
            ["direct_initial", "shard_initial"],
        )
        self.assertEqual(continuation_headroom_attempts, 1)
        self.assertGreaterEqual(request_measurement.call_count, 3)
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

        async with self.SessionLocal() as session:
            artifacts = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                    )
                )
            ).scalars().all()
        parents = [
            artifact
            for artifact in artifacts
            if artifact.status == "completed"
            and isinstance(artifact.usage_json, dict)
            and "_aiv_sharded_document" in artifact.usage_json
            and "_aiv_abandoned_structured_prefix" in artifact.usage_json
        ]
        checkpoints = [
            artifact
            for artifact in artifacts
            if artifact.model is None
            and artifact.status == "completed"
            and isinstance(artifact.output_json, dict)
            and "recovery_execution_manifest_sha256"
            in artifact.output_json
        ]
        self.assertEqual(len(parents), 1)
        self.assertEqual(len(checkpoints), 1)
        parent_manifest = parents[0].usage_json["_aiv_sharded_document"]
        checkpoint = checkpoints[0].output_json
        assert isinstance(checkpoint, dict)
        self.assertEqual(
            checkpoint["recovery_execution_manifest_sha256"],
            stable_digest(parent_manifest),
        )
        self.assertTrue(parent_manifest["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
