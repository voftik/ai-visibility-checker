from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from app.services.analyzer import (
    ENTITY_CATALOG_CONTRACT_RECOVERY_MAX_ATTEMPTS,
    ENTITY_CATALOG_CONTRACT_RECOVERY_STAGE,
    ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION,
    PROCESSING_MODEL,
    _deterministic_entity_catalog_union,
    _entity_catalog,
    _entity_catalog_quote_recovery_incident,
    _entity_catalog_recovery_stage_key,
)
from app.services.long_response import text_sha256
from app.services.openrouter import (
    ChatResult,
    OpenRouterError,
    OpenRouterResponseContractError,
)
from app.services.recovery_orchestrator import (
    ACTION_RETRY_WITH_GUIDANCE,
    CHECK_PROMPT_CONTRACT_VALID,
    CHECK_RAW_CORPUS_UNCHANGED,
)


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
        "guidance": (
            "Повторно извлеки строку, сохранив точные координаты core."
        ),
        "acceptance_checks": [
            CHECK_PROMPT_CONTRACT_VALID,
            CHECK_RAW_CORPUS_UNCHANGED,
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
        "output_incomplete_reason": (
            None if complete else "missing_finish_reason"
        ),
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


class EntityCatalogRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _run_catalog(
        self,
        processing: Any,
        *,
        enabled: bool = True,
        plan: Any | None = None,
        attempts: list[int] | None = None,
        planner_mock: AsyncMock | None = None,
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
        recovery_artifact = recovery_artifact_mock or AsyncMock(
            return_value=None
        )
        mark_contract_failed = mark_contract_failed_mock or AsyncMock()
        result: dict[str, Any] | None = None
        with (
            patch(
                "app.services.analyzer.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
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
                "app.services.analyzer."
                "_validate_entity_catalog_recovery_epoch_binding",
                epoch_binding,
            ),
            patch(
                "app.services.analyzer."
                "_entity_catalog_recovery_artifact",
                recovery_artifact,
            ),
            patch(
                "app.services.analyzer."
                "_mark_completed_artifact_contract_failed",
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
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

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
            side_effect=lambda *args, **kwargs: completion_order.append(
                "checkpoint"
            )
        )
        finish = AsyncMock(
            side_effect=lambda *args, **kwargs: completion_order.append(
                "finish"
            )
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
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

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
        reserve.assert_awaited_once()
        self.assertEqual(
            reserve.await_args.kwargs["stage_execution_limit"],
            ENTITY_CATALOG_CONTRACT_RECOVERY_MAX_ATTEMPTS,
        )
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertTrue(
            finish.await_args.kwargs["details"]["raw_source_unchanged"]
        )
        save.assert_awaited_once()
        saved = save.await_args.kwargs
        self.assertTrue(saved["artifact_key"].endswith("_recovery_accepted"))
        self.assertEqual(saved["status"], "completed")
        self.assertIsNone(saved["model"])
        self.assertEqual(
            saved["output_json"]["accepted_output_sha256"],
            saved["usage_json"]["_aiv_entity_catalog_recovery"][
                "accepted_output_sha256"
            ],
        )
        self.assertEqual(completion_order, ["checkpoint", "finish"])

    async def test_absent_output_quote_uses_one_plan_and_strict_retry(
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
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA",
                )
            if "answers" in payload:
                invalid = _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="ALPHA source",
                )
                invalid["catalog"]["target_aliases"] = ["ALPHA"]
                return invalid
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[{"answer_id": 1, "answer": "ALPHA source"}],
        )

        self.assertIsNotNone(result)
        self.assertTrue(any(key.endswith("_recovery_e4_a1") for key in calls))
        planner.assert_awaited_once()
        self.assertEqual(
            planner.await_args.kwargs["failure_code"],
            "core_quote_absent_from_analytic_output",
        )
        reserve.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])

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
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

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

    def test_absent_output_incident_rejects_spoofed_candidates(self) -> None:
        core_text = "ALPHA source"
        claim = {
            "claim_id": "claim-1",
            "unit_id": "1:000000",
            "core_sha256": text_sha256(core_text),
            "core_text": core_text,
        }
        error = OpenRouterError(
            "Grounded core-unit quote is absent from the analytic output: "
            "1:000000"
        )
        valid_incident = _catalog_candidate(claim, quote=core_text)
        valid_incident["catalog"]["target_aliases"] = ["ALPHA"]
        admitted = _entity_catalog_quote_recovery_incident(
            error,
            candidate=valid_incident,
            expected_claims=[claim],
        )
        self.assertIsNotNone(admitted)

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
            evidence="ALPHA GAMMA",
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
            "Entity-catalog evidence binding failed: "
            "entities[0].aliases is not a list",
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
            if artifact_key.endswith(
                f"_recovery_e4_a{attempt_number}"
            ):
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
            any(
                key.endswith(f"_recovery_e4_a{attempt_number}")
                for key in calls
            )
        )
        if attempt_number == 2:
            self.assertFalse(
                any(key.endswith("_recovery_e4_a1") for key in calls)
            )

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
        self.assertTrue(
            finish.await_args.kwargs["details"]["raw_source_unchanged"]
        )

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

    async def test_complete_response_contract_failure_spends_a1_and_uses_a2(
        self,
    ) -> None:
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
                raise _response_contract_error(complete=True)
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
                        recovery_artifact_mock=AsyncMock(
                            return_value=artifact
                        ),
                    )
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
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

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
        self.assertTrue(stored["artifact_key"].endswith("_recovery_accepted"))

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
                        prompt_version=(
                            ENTITY_CATALOG_CONTRACT_RECOVERY_VERSION
                        ),
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
            epoch_binding_mock=AsyncMock(
                return_value=("executing", resumed_plan)
            ),
            finish_mock=reconcile_finish,
        )
        reconcile_finish.assert_awaited_once()
        self.assertTrue(reconcile_finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            reconcile_finish.await_args.kwargs["details"][
                "accepted_artifact_key"
            ],
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
                artifact_output_mock=AsyncMock(
                    side_effect=tampered_checkpoint
                ),
                recovery_artifact_mock=AsyncMock(
                    side_effect=AssertionError(
                        "Tampered checkpoint must fail before artifact load"
                    )
                ),
            )

    async def test_ungrounded_alias_recovery_remains_bounded_and_fail_closed(
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
            raise AssertionError("Invalid leaf must not reach the reducer")

        planner = AsyncMock(return_value=_retry_plan())
        finish = AsyncMock()
        with self.assertRaisesRegex(
            OpenRouterError,
            "contract recovery exhausted",
        ):
            await self._run_catalog(
                processing,
                planner_mock=planner,
                attempts=[1, 2],
                finish_mock=finish,
            )
        planner.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["succeeded"])

    async def test_recovery_rejects_entity_supported_only_by_sibling_material(
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
                        "evidence": "ALPHA",
                    }
                )
                return value
            if "answers" in payload:
                return _catalog_candidate(
                    payload["answers"][0]["core_claim"],
                    quote="Alpha",
                )
            raise AssertionError("Invalid recovery must not reach the reducer")

        finish = AsyncMock()
        with self.assertRaisesRegex(
            OpenRouterError,
            "contract recovery exhausted",
        ):
            await self._run_catalog(
                processing,
                attempts=[1, 2],
                finish_mock=finish,
                answers=[{"answer_id": 1, "answer": "ALPHA GAMMA"}],
            )
        finish.assert_awaited_once()
        failures = finish.await_args.kwargs["details"]["attempt_failures"]
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            all(
                "entities[1].canonical_name is not grounded" in row["error"]
                for row in failures
            )
        )

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
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

        result, planner, reserve, finish = await self._run_catalog(
            processing,
            answers=[{"answer_id": 1, "answer": "ALPHA Alpha"}],
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
            return _deterministic_entity_catalog_union(
                list(payload["chunk_catalogs"])
            )

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

    async def test_unowned_beta_canonical_cannot_borrow_literal_gamma_alias(
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
            raise AssertionError("Invalid leaf must not reach reducer")

        planner = AsyncMock(return_value=_retry_plan())
        finish = AsyncMock()
        with self.assertRaisesRegex(
            OpenRouterError,
            "contract recovery exhausted",
        ):
            await self._run_catalog(
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
                answers=[
                    {"answer_id": 1, "answer": "Риалвеб Gamma"}
                ],
            )
        planner.assert_awaited_once()
        finish.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["succeeded"])

    def test_recovery_stage_is_stable_and_distinct_per_chunk(self) -> None:
        first = _entity_catalog_recovery_stage_key(
            {"artifact_key": "entity_catalog_chunk_1_alpha"}
        )
        repeated = _entity_catalog_recovery_stage_key(
            {"artifact_key": "entity_catalog_chunk_1_alpha"}
        )
        second = _entity_catalog_recovery_stage_key(
            {"artifact_key": "entity_catalog_chunk_2_beta"}
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertTrue(
            first.startswith(ENTITY_CATALOG_CONTRACT_RECOVERY_STAGE + ":")
        )

    async def test_ambiguous_markdown_failure_stays_fail_closed(self) -> None:
        planner = AsyncMock()
        with (
            patch(
                "app.services.analyzer.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
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
                OpenRouterError(
                    "Core-unit Markdown-normalized quote is ambiguous"
                ),
                candidate={"core_dispositions": []},
                expected_claims=[],
            )

        self.assertIsNone(admitted)
        planner.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
