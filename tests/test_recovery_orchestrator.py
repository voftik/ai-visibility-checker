from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import Base, RecoveryEpoch, Run, RunArtifact, RunStatus
from app.services.long_response import text_sha256
from app.services.openrouter import (
    ChatResult,
    OpenRouterOutputLimitError,
    OutputTokenPolicy,
)
from app.services.recovery_orchestrator import (
    ACTION_DETERMINISTIC_FALLBACK,
    ACTION_RETRY_WITH_GUIDANCE,
    ACTION_STOP,
    ACTION_TARGETED_ANNOTATION_REPAIR,
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_VERSION,
    PROCESSING_MODEL,
    OrchestratorContractError,
    OrchestratorResult,
    _decide_from_exact_claim_ledger,
    _decision_shard_payload,
    _input_window,
    _map_recovery_source,
    _pack_decision_claims,
    _recovery_atomic_chat,
    _reduce_recovery_nodes,
    _source_units,
    _stable_digest,
    _structured_request_utf8_bytes,
    _validate_decision_shard,
    _validate_reduce_node_summaries,
    plan_recovery,
    validate_recovery_decision,
)
from app.services.recovery_state import (
    RecoveryBudgetExceeded,
    finish_recovery,
    mark_recovery_executing,
    plan_durable_recovery,
    recovery_failure_fingerprint,
    recovery_scope_digest,
    stable_digest,
)


def _map_unit_summaries(payload: dict) -> list[dict]:
    summaries: list[dict] = []
    for unit in payload["source_units"]:
        context = unit["context_text"]
        core = context[
            unit["core_start_in_context"] : unit["core_end_in_context"]
        ]
        excerpt = core
        summaries.append(
            {
                "unit_id": unit["unit_id"],
                "core_sha256": unit["core_sha256"],
                "summary": "Содержимое lossless-фрагмента учтено при планировании.",
                "relevance": "context",
                "source_excerpt": excerpt,
                "source_excerpt_sha256": hashlib.sha256(
                    excerpt.encode("utf-8")
                ).hexdigest(),
            }
        )
    return summaries


def _reduce_node_summaries(payload: dict) -> list[dict]:
    return [
        {
            "source_node_id": node["node_id"],
            "source_semantic_sha256": _stable_digest(node["semantic"]),
            "summary": "Смысл дочернего узла сохранён в reducer-квитанции.",
            "relevance": "context",
        }
        for node in payload["nodes"]
    ]
from app.services.run_lease import RunLeaseLostError, bind_run_lease


def _chat_result(parsed: dict) -> ChatResult:
    return ChatResult(
        text="{}",
        parsed=parsed,
        citations=[],
        usage={"prompt_tokens": 100},
        annotations=[],
        request_policy={},
        web_attestation={"metric_eligible": True},
        router_metadata={},
    )


class RecoveryDecisionContractTests(unittest.TestCase):
    def test_failure_fingerprint_is_stable_and_stage_specific(self) -> None:
        first = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics={"errors": ["x"]},
        )
        second = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics={"errors": ["x"]},
        )
        other = recovery_failure_fingerprint(
            stage_key="knowledge_gap",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics={"errors": ["x"]},
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(stable_digest({"a": 1})), 64)

    def test_durable_scope_digest_binds_every_executable_boundary(self) -> None:
        base = recovery_scope_digest(
            facts={"profile": "same"},
            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            permitted_answer_ids={41},
            permitted_artifact_keys={"prompt_set"},
        )
        variants = (
            recovery_scope_digest(
                facts={"profile": "same"},
                allowed_actions={ACTION_RETRY_WITH_GUIDANCE},
                permitted_answer_ids={41},
                permitted_artifact_keys={"prompt_set"},
            ),
            recovery_scope_digest(
                facts={"profile": "same"},
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                permitted_answer_ids={42},
                permitted_artifact_keys={"prompt_set"},
            ),
            recovery_scope_digest(
                facts={"profile": "same"},
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                permitted_answer_ids={41},
                permitted_artifact_keys={"semantic_review"},
            ),
        )
        self.assertTrue(all(value != base for value in variants))

    def test_decision_cannot_touch_unlisted_answers_or_artifacts(self) -> None:
        with self.assertRaises(OrchestratorContractError):
            validate_recovery_decision(
                {
                    "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                    "rationale": "Нужен узкий ремонт конкретной строки.",
                    "confidence": "high",
                    "guidance": "Проверь буквальную связь владельца и услуги.",
                    "target_answer_ids": [41, 999],
                    "invalidate_artifact_keys": ["annotations_999"],
                    "acceptance_checks": ["raw_corpus_unchanged"],
                },
                allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                permitted_answer_ids={41},
                permitted_artifact_keys={"annotations_41"},
                prior_decisions=[],
                incident_fingerprint="abc",
            )

    def test_same_action_cannot_loop_on_same_fingerprint(self) -> None:
        with self.assertRaises(OrchestratorContractError):
            validate_recovery_decision(
                {
                    "action": ACTION_DETERMINISTIC_FALLBACK,
                    "rationale": "Локальные попытки уже исчерпаны безопасно.",
                    "confidence": "high",
                    "guidance": "",
                    "target_answer_ids": [],
                    "invalidate_artifact_keys": [],
                    "acceptance_checks": [
                        "prompt_contract_valid",
                        "semantic_review_passed",
                    ],
                },
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                permitted_answer_ids=set(),
                permitted_artifact_keys=set(),
                prior_decisions=[
                    {
                        "incident_fingerprint": "abc",
                        "facts_digest": "facts-a",
                        "status": "failed",
                        "action": ACTION_DETERMINISTIC_FALLBACK,
                    }
                ],
                incident_fingerprint="abc",
                incident_facts_digest="facts-a",
            )

    def test_success_or_changed_facts_do_not_trigger_loop_guard(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
        }
        for prior in (
            {
                "incident_fingerprint": "abc",
                "facts_digest": "facts-a",
                "status": "succeeded",
                "action": ACTION_DETERMINISTIC_FALLBACK,
            },
            {
                "incident_fingerprint": "abc",
                "facts_digest": "facts-old",
                "status": "failed",
                "action": ACTION_DETERMINISTIC_FALLBACK,
            },
        ):
            with self.subTest(prior=prior):
                accepted = validate_recovery_decision(
                    decision,
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                    permitted_answer_ids=set(),
                    permitted_artifact_keys=set(),
                    prior_decisions=[prior],
                    incident_fingerprint="abc",
                    incident_facts_digest="facts-a",
                )
                self.assertEqual(
                    accepted["action"],
                    ACTION_DETERMINISTIC_FALLBACK,
                )

    def test_local_validator_rejects_bool_nonpositive_and_low_confidence(self) -> None:
        base = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "rationale": "Нужен узкий ремонт конкретной строки ответа модели.",
            "confidence": "high",
            "guidance": "Проверь буквальную связь владельца и услуги.",
            "target_answer_ids": [1],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["raw_corpus_unchanged"],
        }
        invalid_cases = (
            {"target_answer_ids": [True]},
            {"target_answer_ids": [0]},
            {"confidence": "low"},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(OrchestratorContractError):
                    validate_recovery_decision(
                        {**base, **changes},
                        allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                        permitted_answer_ids=set(range(0, 50)),
                        permitted_artifact_keys={
                            f"artifact-{i}" for i in range(30)
                        },
                        prior_decisions=[],
                        incident_fingerprint="abc",
                        incident_facts_digest="facts-a",
                    )

    def test_local_validator_preserves_large_permitted_scope(self) -> None:
        answer_ids = list(range(1, 82))
        artifact_keys = [f"artifact-{i}" for i in range(48)]
        accepted = validate_recovery_decision(
            {
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "rationale": (
                    "Критик связал каждую строку с одной и той же исправимой "
                    "ошибкой разметки, поэтому нужен полный пакетный ремонт."
                ),
                "confidence": "high",
                "guidance": (
                    "Проверь буквальную связь сущностей отдельно в каждом "
                    "разрешённом ответе."
                ),
                "target_answer_ids": answer_ids,
                "invalidate_artifact_keys": artifact_keys,
                "acceptance_checks": [
                    "raw_corpus_unchanged",
                    "raw_corpus_unchanged",
                ],
            },
            allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
            permitted_answer_ids=set(answer_ids),
            permitted_artifact_keys=set(artifact_keys),
            prior_decisions=[],
            incident_fingerprint="abc",
            incident_facts_digest="facts-a",
        )
        self.assertEqual(accepted["target_answer_ids"], answer_ids)
        self.assertEqual(accepted["invalidate_artifact_keys"], artifact_keys)
        self.assertEqual(
            accepted["acceptance_checks"],
            ["raw_corpus_unchanged"],
        )


class RecoveryPlannerTests(unittest.IsolatedAsyncioTestCase):
    def test_progressing_harness_has_no_aggregate_wall_clock_deadline(
        self,
    ) -> None:
        for function in (
            _map_recovery_source,
            _reduce_recovery_nodes,
            _decide_from_exact_claim_ledger,
        ):
            with self.subTest(function=function.__name__):
                self.assertNotIn("deadline", inspect.signature(function).parameters)

    def test_bare_metric_number_does_not_create_answer_lineage(self) -> None:
        units, _manifest = _source_units(
            {
                "unrelated": "Конверсия составила 42%, HTTP 200.",
                "record": {
                    "answer_id": 7,
                    "message": "Ошибка разметки относится к этой записи.",
                },
                "explicit": "Нужно проверить ответ 9 по исходному raw.",
            },
            target_chars=1_024,
            permitted_answer_ids={7, 9, 42, 200},
            permitted_artifact_keys=set(),
        )
        by_pointer = {unit["json_pointer"]: unit for unit in units}
        self.assertEqual(by_pointer["/unrelated"]["linked_answer_ids"], [])
        self.assertEqual(by_pointer["/record/message"]["linked_answer_ids"], [7])
        self.assertEqual(by_pointer["/explicit"]["linked_answer_ids"], [9])

    async def test_decision_shard_rejects_unrelated_permitted_answer(self) -> None:
        excerpt = "TAIL_MARKER без ссылки на конкретный ответ."
        entry = {
            "claim_id": "claim-tail",
            "source_excerpt_sha256": text_sha256(excerpt),
            "source_excerpt": excerpt,
            "json_pointer": "/incident/facts/message",
            "value_kind": "str",
            "linked_answer_ids": [],
            "linked_artifact_keys": [],
        }
        parsed = {
            "covered_claims": [
                {
                    "claim_id": entry["claim_id"],
                    "source_excerpt_sha256": entry[
                        "source_excerpt_sha256"
                    ],
                }
            ],
            "dispositions": [
                {
                    "claim_id": entry["claim_id"],
                    "source_excerpt_sha256": entry[
                        "source_excerpt_sha256"
                    ],
                    "semantic_observation": "TAIL_MARKER",
                    "relevance": "actionable",
                    "candidate_answer_ids": [42],
                    "candidate_artifact_keys": [],
                }
            ],
            "candidate_decision": {
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "rationale": "Достаточно длинное объяснение для валидатора.",
                "confidence": "high",
                "guidance": "Переразметить ответ 42.",
                "target_answer_ids": [42],
                "invalidate_artifact_keys": [],
                "acceptance_checks": ["critic_gate_passed"],
            },
        }
        with self.assertRaisesRegex(
            OrchestratorContractError,
            "without literal source-record lineage",
        ):
            _validate_decision_shard(
                parsed,
                entries=[entry],
                allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                permitted_answer_ids={42},
                permitted_artifact_keys=set(),
                prior_decisions=[],
                incident_fingerprint="fingerprint",
                incident_facts_digest="facts",
            )

    async def test_control_plane_leaves_cannot_authorize_mutation(self) -> None:
        cases = [
            {
                "pointer": "/incident/facts/answer_id",
                "value_kind": "int",
                "excerpt": "9",
                "answer_ids": [9],
                "artifact_keys": [],
            },
            {
                "pointer": "/incident/facts/artifact_key",
                "value_kind": "str",
                "excerpt": "annotations_9",
                "answer_ids": [],
                "artifact_keys": ["annotations_9"],
            },
            {
                "pointer": "/incident/facts/error_count",
                "value_kind": "int",
                "excerpt": "3",
                "answer_ids": [9],
                "artifact_keys": [],
            },
            {
                "pointer": "/incident/facts/raw_sha256",
                "value_kind": "str",
                "excerpt": "Ошибка",
                "answer_ids": [9],
                "artifact_keys": [],
            },
        ]
        for index, case in enumerate(cases):
            with self.subTest(pointer=case["pointer"]):
                entry = {
                    "claim_id": f"claim-control-{index}",
                    "source_excerpt_sha256": text_sha256(case["excerpt"]),
                    "source_excerpt": case["excerpt"],
                    "json_pointer": case["pointer"],
                    "value_kind": case["value_kind"],
                    "linked_answer_ids": case["answer_ids"],
                    "linked_artifact_keys": case["artifact_keys"],
                }
                parsed = {
                    "covered_claims": [
                        {
                            "claim_id": entry["claim_id"],
                            "source_excerpt_sha256": entry[
                                "source_excerpt_sha256"
                            ],
                        }
                    ],
                    "dispositions": [
                        {
                            "claim_id": entry["claim_id"],
                            "source_excerpt_sha256": entry[
                                "source_excerpt_sha256"
                            ],
                            "semantic_observation": case["excerpt"],
                            "relevance": "actionable",
                            "candidate_answer_ids": case["answer_ids"],
                            "candidate_artifact_keys": case["artifact_keys"],
                        }
                    ],
                    "candidate_decision": {
                        "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                        "rationale": (
                            "Контрольное поле ошибочно принято за доказательство."
                        ),
                        "confidence": "high",
                        "guidance": "Исправить выбранный объект.",
                        "target_answer_ids": case["answer_ids"],
                        "invalidate_artifact_keys": case["artifact_keys"],
                        "acceptance_checks": ["critic_gate_passed"],
                    },
                }
                with self.assertRaisesRegex(
                    OrchestratorContractError,
                    "control-plane identifier without substantive failure",
                ):
                    _validate_decision_shard(
                        parsed,
                        entries=[entry],
                        allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                        permitted_answer_ids={9},
                        permitted_artifact_keys={"annotations_9"},
                        prior_decisions=[],
                        incident_fingerprint="fingerprint",
                        incident_facts_digest="facts",
                    )

    async def test_decision_payload_keeps_full_scope_ledger_code_side(
        self,
    ) -> None:
        artifact_keys = [f"artifact-{index}-" + ("z" * 40) for index in range(500)]
        entry = {
            "claim_id": "claim-1",
            "unit_id": "unit-1",
            "core_sha256": "a" * 64,
            "source_excerpt_sha256": "b" * 64,
            "json_pointer": "/incident/facts/message",
            "value_kind": "str",
            "source_excerpt": "Короткий проверяемый факт.",
            "mapper_summary": "Короткий факт.",
            "mapper_relevance": "context",
            "linked_answer_ids": [],
            "linked_artifact_keys": artifact_keys,
        }
        ledger = {
            "version": "ledger",
            "source_manifest_sha256": "c" * 64,
            "claim_count": 1,
            "claim_ids_sha256": "d" * 64,
            "claim_receipts_sha256": "e" * 64,
            "coverage_complete": True,
            "ledger_sha256": "f" * 64,
        }
        manifest = {
            "version": "manifest",
            "source_sha256": "1" * 64,
            "source_utf8_bytes": 1,
            "leaf_count": 1,
            "unit_count": 1,
            "unit_ids_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
        }
        scope_manifest = {
            "permitted_artifact_key_count": len(artifact_keys),
            "permitted_artifact_keys_sha256": _stable_digest(artifact_keys),
            "coverage_complete": True,
        }
        control_plane = {
            "allowed_actions": [ACTION_STOP],
            "authorization_scope_manifest": scope_manifest,
            "scope_sha256": _stable_digest(scope_manifest),
        }
        payload = _decision_shard_payload(
            [entry],
            shard_index=0,
            ledger=ledger,
            source_manifest=manifest,
            control_plane=control_plane,
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(artifact_keys[-1], serialized)
        batches = _pack_decision_claims(
            [entry],
            ledger=ledger,
            source_manifest=manifest,
            control_plane=control_plane,
            allowed_actions=[ACTION_STOP],
            model_envelope={"max_completion_tokens": 1_000},
            window_bytes=12_000,
        )
        self.assertEqual(batches, [[entry]])

    def setUp(self) -> None:
        self._enabled = patch(
            "app.services.recovery_orchestrator.settings."
            "PIPELINE_ORCHESTRATOR_ENABLED",
            True,
        )
        self._enabled.start()
        self._envelope = patch(
            "app.services.recovery_orchestrator.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "resolution": "test",
                    "context_length": 256_000,
                    "max_completion_tokens": 32_000,
                }
            ),
        )
        self._envelope.start()

    def tearDown(self) -> None:
        self._envelope.stop()
        self._enabled.stop()

    async def test_disabled_orchestrator_fails_before_provider_call(self) -> None:
        with (
            patch(
                "app.services.recovery_orchestrator.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                False,
            ),
            patch(
                "app.services.recovery_orchestrator.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
        ):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "orchestrator is disabled",
            ):
                await plan_recovery(
                    incident={"stage": "scenario_design"},
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )
        chat_mock.assert_not_awaited()

    async def test_fable_only_selects_from_stage_allowlist(self) -> None:
        parsed = {
            "action": ACTION_RETRY_WITH_GUIDANCE,
            "rationale": (
                "Последний кандидат близок к контракту; нужен один узкий ремонт."
            ),
            "confidence": "medium",
            "guidance": "Считай открытый вопрос измерением, а не утверждением.",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
        }
        with patch(
            "app.services.recovery_orchestrator.chat",
            new=AsyncMock(return_value=_chat_result(parsed)),
        ) as chat_mock:
            result = await plan_recovery(
                incident={
                    "stage": "scenario_design",
                    "code": "prompt_set_non_convergent",
                    "fingerprint": "incident-1",
                },
                allowed_actions={
                    ACTION_RETRY_WITH_GUIDANCE,
                    ACTION_DETERMINISTIC_FALLBACK,
                },
            )
        self.assertEqual(result.decision["action"], ACTION_RETRY_WITH_GUIDANCE)
        kwargs = chat_mock.await_args.kwargs
        self.assertEqual(kwargs["model"], "anthropic/claude-fable-5")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(
            kwargs["output_token_policy"],
            OutputTokenPolicy.MODEL_MAX,
        )
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertFalse(kwargs["retry_response_contract_errors"])
        self.assertFalse(kwargs["retry_transport_errors"])
        self.assertEqual(
            result.usage["_aiv_recovery_input_harness"]["mode"],
            "direct_atomic",
        )

    async def test_invalid_model_scope_is_rejected_after_the_call(self) -> None:
        parsed = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "rationale": "Следует проверить только проблемный raw-ответ.",
            "confidence": "high",
            "guidance": "Скопируй буквальный непрерывный фрагмент.",
            "target_answer_ids": [78],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["raw_corpus_unchanged"],
        }
        with patch(
            "app.services.recovery_orchestrator.chat",
            new=AsyncMock(return_value=_chat_result(parsed)),
        ):
            with self.assertRaises(OrchestratorContractError):
                await plan_recovery(
                    incident={"stage": "knowledge_gap", "fingerprint": "x"},
                    allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                    permitted_answer_ids={77},
                )

    async def test_history_payload_keeps_facts_digest_and_status(self) -> None:
        parsed = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Нужен безопасный детерминированный резервный сценарий.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["prompt_contract_valid"],
        }
        prior = {
            "incident_fingerprint": "old-fingerprint",
            "facts_digest": "old-facts",
            "status": "failed",
            "action": ACTION_RETRY_WITH_GUIDANCE,
        }
        with patch(
            "app.services.recovery_orchestrator.chat",
            new=AsyncMock(return_value=_chat_result(parsed)),
        ) as chat_mock:
            await plan_recovery(
                incident={
                    "stage": "scenario_design",
                    "fingerprint": "current-fingerprint",
                    "facts_digest": "current-facts",
                },
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                prior_decisions=[prior],
            )

        payload = json.loads(
            chat_mock.await_args.kwargs["messages"][1]["content"]
        )
        self.assertEqual(
            payload["prior_decisions"][0]["facts_digest"],
            "old-facts",
        )
        self.assertEqual(payload["prior_decisions"][0]["status"], "failed")

    async def test_huge_unicode_corpus_is_losslessly_mapped_and_reduced(self) -> None:
        decision = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "rationale": (
                "Полный корпус указывает на исправимую ошибку одной разметки."
            ),
            "confidence": "high",
            "guidance": "Повтори только разметку ответа 9 по исходному raw.",
            "target_answer_ids": [9],
            "invalidate_artifact_keys": ["annotations_9"],
            "acceptance_checks": ["raw_corpus_unchanged"],
        }
        envelopes = {
            ORCHESTRATOR_MODEL: {
                "resolution": "test",
                "context_length": 12_000,
                "max_completion_tokens": 2_500,
            },
            PROCESSING_MODEL: {
                "resolution": "test",
                "context_length": 16_000,
                "max_completion_tokens": 4_000,
            },
        }

        async def provider(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            if kwargs["schema_name"] == "aiv_recovery_input_map":
                covered = [
                    {
                        "unit_id": unit["unit_id"],
                        "core_sha256": unit["core_sha256"],
                    }
                    for unit in payload["source_units"]
                ]
                parsed = {
                    "covered_units": covered,
                    "unit_summaries": _map_unit_summaries(payload),
                    "findings": [
                        {
                            "unit_ids": [covered[0]["unit_id"]],
                            "statement": "Найдена исправимая ошибка разметки.",
                            "relevance": "actionable",
                            "candidate_answer_ids": [9],
                            "candidate_artifact_keys": ["annotations_9"],
                        }
                    ],
                    "uncertainties": [],
                }
            elif kwargs["schema_name"] == "aiv_recovery_input_reduce":
                parsed = {
                    "covered_node_ids": [
                        node["node_id"] for node in payload["nodes"]
                    ],
                    "synthesis": "Нужен узкий ремонт сохранённой разметки.",
                    "node_summaries": _reduce_node_summaries(payload),
                    "findings": [
                        {
                            "source_node_ids": [
                                node["node_id"] for node in payload["nodes"]
                            ],
                            "statement": "Ошибка локализована в ответе 9.",
                            "relevance": "actionable",
                            "candidate_answer_ids": [9],
                            "candidate_artifact_keys": ["annotations_9"],
                        }
                    ],
                    "uncertainties": [],
                }
            elif kwargs["schema_name"] == "aiv_recovery_decision_shard":
                entries = payload["shard"]["exact_claim_ledger"]
                relevant = [
                    entry
                    for entry in entries
                    if "Ошибка разметки" in str(entry["source_excerpt"])
                ]
                shard_decision = (
                    decision
                    if relevant
                    else {
                        "action": ACTION_STOP,
                        "rationale": (
                            "Этот shard не содержит связанного answer_id."
                        ),
                        "confidence": "high",
                        "guidance": "",
                        "target_answer_ids": [],
                        "invalidate_artifact_keys": [],
                        "acceptance_checks": ["checkpoint_preserved"],
                    }
                )
                parsed = {
                    "covered_claims": [
                        {
                            "claim_id": entry["claim_id"],
                            "source_excerpt_sha256": entry[
                                "source_excerpt_sha256"
                            ],
                        }
                        for entry in entries
                    ],
                    "dispositions": [
                        {
                            "claim_id": entry["claim_id"],
                            "source_excerpt_sha256": entry[
                                "source_excerpt_sha256"
                            ],
                            "semantic_observation": (
                                f"{entry['value_kind']} {entry['json_pointer']} "
                                f"{entry['source_excerpt']}"
                            ),
                            "relevance": "actionable",
                            "candidate_answer_ids": (
                                [9] if entry in relevant else []
                            ),
                            "candidate_artifact_keys": (
                                ["annotations_9"] if entry in relevant else []
                            ),
                        }
                        for entry in entries
                    ],
                    "candidate_decision": shard_decision,
                }
            elif kwargs["schema_name"] == "aiv_recovery_decision_arbiter":
                has_target_scope = any(
                    9 in node["decision"]["target_answer_ids"]
                    for node in payload["candidate_nodes"]
                )
                parsed = {
                    "covered_candidate_ids": [
                        node["candidate_id"]
                        for node in payload["candidate_nodes"]
                    ],
                    "decision": (
                        decision
                        if has_target_scope
                        else {
                            "action": ACTION_STOP,
                            "rationale": (
                                "В дочерних решениях нет связанного answer_id."
                            ),
                            "confidence": "high",
                            "guidance": "",
                            "target_answer_ids": [],
                            "invalidate_artifact_keys": [],
                            "acceptance_checks": ["checkpoint_preserved"],
                        }
                    ),
                }
            else:
                self.assertEqual(kwargs["model"], ORCHESTRATOR_MODEL)
                parsed = decision
            return _chat_result(parsed)

        long_text = (
            "TAIL_ALPHA: Ошибка разметки ответа 9.\n"
            + ("🔥 Длинный факт о видимости бренда — Москва.\n" * 2_500)
            + "TAIL_BETA 終"
        )
        with (
            patch(
                "app.services.recovery_orchestrator.model_output_envelope",
                new=AsyncMock(side_effect=lambda model: envelopes[model]),
            ),
            patch(
                "app.services.recovery_orchestrator.chat",
                new=AsyncMock(side_effect=provider),
            ) as chat_mock,
        ):
            result = await plan_recovery(
                incident={
                    "stage": "knowledge_gap",
                    "fingerprint": "huge-unicode",
                    "facts_digest": "f" * 64,
                    "facts": {
                        "answer_id": 9,
                        "artifact_key": "annotations_9",
                        "raw": long_text,
                    },
                },
                allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                permitted_answer_ids={3, 9},
                permitted_artifact_keys={"annotations_3", "annotations_9"},
                prior_decisions=[
                    {
                        "epoch": 1,
                        "incident_fingerprint": "old",
                        "facts_digest": "e" * 64,
                        "status": "failed",
                        "action": ACTION_RETRY_WITH_GUIDANCE,
                        "outcome": {"unicode": "ёжик 🦔" * 1_000},
                    }
                ],
            )

        audit = result.usage["_aiv_recovery_input_harness"]
        self.assertEqual(
            audit["mode"],
            "lossless_exact_claim_decision_shards",
        )
        self.assertGreater(audit["source_manifest"]["unit_count"], 1)
        self.assertGreater(audit["map_receipt_count"], 1)
        self.assertEqual(audit["reduce_receipt_count"], 0)
        self.assertGreater(audit["decision_shard_count"], 1)
        self.assertGreaterEqual(audit["decision_arbiter_rounds"], 1)
        expected_source = {
            "incident": {
                "stage": "knowledge_gap",
                "fingerprint": "huge-unicode",
                "facts_digest": "f" * 64,
                "facts": {
                    "answer_id": 9,
                    "artifact_key": "annotations_9",
                    "raw": long_text,
                },
            },
            "prior_decisions": [
                {
                    "epoch": 1,
                    "incident_fingerprint": "old",
                    "facts_digest": "e" * 64,
                    "status": "failed",
                    "action": ACTION_RETRY_WITH_GUIDANCE,
                    "outcome": {"unicode": "ёжик 🦔" * 1_000},
                }
            ],
        }
        self.assertEqual(
            audit["source_manifest"]["source_sha256"],
            _stable_digest(expected_source),
        )

        terra_envelope = envelopes[PROCESSING_MODEL]
        terra_window = int(
            _input_window(PROCESSING_MODEL, terra_envelope)["input_utf8_window"]
        )
        mapped_ids: list[str] = []
        fable_calls = 0
        decision_visible_text = ""
        total_claims = audit["decision_ledger"]["claim_count"]
        for call in chat_mock.await_args_list:
            kwargs = call.kwargs
            payload = json.loads(kwargs["messages"][1]["content"])
            if kwargs["model"] == ORCHESTRATOR_MODEL:
                fable_calls += 1
                fable_window = int(
                    _input_window(
                        ORCHESTRATOR_MODEL,
                        envelopes[ORCHESTRATOR_MODEL],
                    )["input_utf8_window"]
                )
                self.assertLessEqual(
                    _structured_request_utf8_bytes(
                        model=kwargs["model"],
                        model_envelope=envelopes[ORCHESTRATOR_MODEL],
                        system=kwargs["messages"][0]["content"],
                        user_payload=payload,
                        schema=kwargs["response_schema"],
                        schema_name=kwargs["schema_name"],
                        reasoning_effort=kwargs["reasoning_effort"],
                        temperature=kwargs["temperature"],
                    ),
                    fable_window,
                )
                if kwargs["schema_name"] == "aiv_recovery_decision_shard":
                    manifest_pointer = payload["claim_ledger_manifest"]
                    self.assertNotIn("entries", manifest_pointer)
                    shard_entries = payload["shard"]["exact_claim_ledger"]
                    self.assertLess(len(shard_entries), total_claims)
                    decision_visible_text += "".join(
                        entry["source_excerpt"] for entry in shard_entries
                    )
                else:
                    self.assertNotIn("exact_claim_ledger", payload)
                continue
            if kwargs["schema_name"] == "aiv_recovery_input_map":
                mapped_ids.extend(
                    unit["unit_id"] for unit in payload["source_units"]
                )
            self.assertLessEqual(
                _structured_request_utf8_bytes(
                    model=kwargs["model"],
                    model_envelope=terra_envelope,
                    system=kwargs["messages"][0]["content"],
                    user_payload=payload,
                    schema=kwargs["response_schema"],
                    schema_name=kwargs["schema_name"],
                    reasoning_effort=kwargs["reasoning_effort"],
                    temperature=kwargs["temperature"],
                ),
                terra_window,
            )
        expected_ids = [
            unit["unit_id"]
            for leaf in audit["source_manifest"]["leaf_manifests"]
            for unit in leaf["partition"]["units"]
        ]
        self.assertEqual(mapped_ids, expected_ids)
        self.assertEqual(len(mapped_ids), len(set(mapped_ids)))
        self.assertGreater(fable_calls, 1)
        self.assertIn("TAIL_ALPHA", decision_visible_text)
        self.assertIn("TAIL_BETA", decision_visible_text)
        self.assertEqual(result.decision["target_answer_ids"], [9])

    def test_generic_reducer_summary_with_correct_ids_and_digest_fails(self) -> None:
        node = {
            "node_id": "node-tail",
            "semantic": {
                "unit_summaries": [
                    {
                        "summary": "TAIL_ALPHA означает ошибку атрибуции",
                    }
                ]
            },
        }
        with self.assertRaisesRegex(
            OrchestratorContractError,
            "generic child summary",
        ):
            _validate_reduce_node_summaries(
                {
                    "node_summaries": [
                        {
                            "source_node_id": "node-tail",
                            "source_semantic_sha256": _stable_digest(
                                node["semantic"]
                            ),
                            "summary": "данные",
                            "relevance": "context",
                        }
                    ]
                },
                nodes=[node],
            )

    async def test_missing_or_tampered_map_coverage_fails_before_fable(self) -> None:
        envelope = {
            "resolution": "test",
            "context_length": 9_000,
            "max_completion_tokens": 2_500,
        }
        for mode in ("missing", "tampered"):
            calls: list[str] = []

            async def provider(**kwargs):
                calls.append(kwargs["model"])
                payload = json.loads(kwargs["messages"][1]["content"])
                covered = [
                    {
                        "unit_id": unit["unit_id"],
                        "core_sha256": unit["core_sha256"],
                    }
                    for unit in payload["source_units"]
                ]
                if mode == "missing":
                    covered = covered[:-1]
                else:
                    covered[0]["core_sha256"] = "0" * 64
                return _chat_result(
                    {
                        "covered_units": covered,
                        "findings": [],
                        "uncertainties": [],
                    }
                )

            with self.subTest(mode=mode):
                with (
                    patch(
                        "app.services.recovery_orchestrator."
                        "model_output_envelope",
                        new=AsyncMock(return_value=envelope),
                    ),
                    patch(
                        "app.services.recovery_orchestrator.chat",
                        new=AsyncMock(side_effect=provider),
                    ),
                ):
                    with self.assertRaisesRegex(
                        OrchestratorContractError,
                        "coverage",
                    ):
                        await plan_recovery(
                            incident={"facts": "юникод 🧪" * 10_000},
                            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                        )
                self.assertTrue(calls)
                self.assertNotIn(ORCHESTRATOR_MODEL, calls)

    async def test_mapper_cannot_cover_tail_without_semantic_receipts(self) -> None:
        envelope = {
            "resolution": "test",
            "context_length": 9_000,
            "max_completion_tokens": 2_500,
        }
        calls: list[str] = []

        async def provider(**kwargs):
            calls.append(kwargs["model"])
            payload = json.loads(kwargs["messages"][1]["content"])
            covered = [
                {
                    "unit_id": unit["unit_id"],
                    "core_sha256": unit["core_sha256"],
                }
                for unit in payload["source_units"]
            ]
            return _chat_result(
                {
                    "covered_units": covered,
                    "unit_summaries": [],
                    "findings": [],
                    "uncertainties": [],
                }
            )

        with (
            patch(
                "app.services.recovery_orchestrator.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
            patch(
                "app.services.recovery_orchestrator.chat",
                new=AsyncMock(side_effect=provider),
            ),
        ):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "unit semantic receipts",
            ):
                await plan_recovery(
                    incident={
                        "facts": ("long context " * 10_000) + "TAIL_MARKER"
                    },
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )
        self.assertTrue(calls)
        self.assertNotIn(ORCHESTRATOR_MODEL, calls)

    async def test_mapper_cannot_replace_full_core_with_one_byte_quote(self) -> None:
        envelope = {
            "resolution": "test",
            "context_length": 9_000,
            "max_completion_tokens": 2_500,
        }

        async def provider(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            covered = [
                {
                    "unit_id": unit["unit_id"],
                    "core_sha256": unit["core_sha256"],
                }
                for unit in payload["source_units"]
            ]
            summaries = _map_unit_summaries(payload)
            summaries[0]["source_excerpt"] = summaries[0]["source_excerpt"][:1]
            summaries[0]["source_excerpt_sha256"] = hashlib.sha256(
                summaries[0]["source_excerpt"].encode("utf-8")
            ).hexdigest()
            return _chat_result(
                {
                    "covered_units": covered,
                    "unit_summaries": summaries,
                    "findings": [],
                    "uncertainties": [],
                }
            )

        with (
            patch(
                "app.services.recovery_orchestrator.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
            patch(
                "app.services.recovery_orchestrator.chat",
                new=AsyncMock(side_effect=provider),
            ),
        ):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "complete exact core",
            ):
                await plan_recovery(
                    incident={
                        "facts": ("alpha important qualitative " * 4_000)
                        + "TAIL_MARKER"
                    },
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )

    async def test_mapper_output_limit_fails_closed_without_fable_retry(self) -> None:
        envelope = {
            "resolution": "test",
            "context_length": 9_000,
            "max_completion_tokens": 2_500,
        }
        limited = OpenRouterOutputLimitError(
            "mapper reached provider output envelope",
            result=_chat_result({}),
        )
        with (
            patch(
                "app.services.recovery_orchestrator.model_output_envelope",
                new=AsyncMock(return_value=envelope),
            ),
            patch(
                "app.services.recovery_orchestrator.chat",
                new=AsyncMock(side_effect=limited),
            ) as chat_mock,
        ):
            with self.assertRaises(OpenRouterOutputLimitError):
                await plan_recovery(
                    incident={"facts": "x" * 50_000},
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )
        self.assertEqual(chat_mock.await_count, 1)
        self.assertEqual(chat_mock.await_args.kwargs["model"], PROCESSING_MODEL)
        self.assertFalse(chat_mock.await_args.kwargs["retry_transport_errors"])


class DurableRecoveryStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "recovery.sqlite3"
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
        self._session_patch = patch(
            "app.services.recovery_state.SessionLocal",
            self.SessionLocal,
        )
        self._session_patch.start()
        self._lease_session_patch = patch(
            "app.services.run_lease.SessionLocal",
            self.SessionLocal,
        )
        self._lease_session_patch.start()
        self._enabled_patch = patch(
            "app.services.recovery_state.settings."
            "PIPELINE_ORCHESTRATOR_ENABLED",
            True,
        )
        self._enabled_patch.start()
        self.run_id = str(uuid.uuid4())
        async with self.SessionLocal() as session:
            session.add(
                Run(
                    id=self.run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                    attempt_count=1,
                    resume_count=0,
                )
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        self._enabled_patch.stop()
        self._lease_session_patch.stop()
        self._session_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    async def test_atomic_provider_checkpoint_resumes_without_second_post(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        }
        messages = [
            {"role": "system", "content": "Read the exact fact."},
            {"role": "user", "content": "TAIL_ALPHA"},
        ]
        raw_text = json.dumps({"fact": "TAIL_ALPHA"}, ensure_ascii=False)
        post_count = 0

        class Response:
            status_code = 200
            text = raw_text

            def json(self):
                return {
                    "id": "recovery-checkpoint-response",
                    "model": ORCHESTRATOR_MODEL,
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
                        "completion_tokens": 5,
                    },
                }

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *_args, **_kwargs):
                nonlocal post_count
                post_count += 1
                return Response()

        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": ORCHESTRATOR_MODEL,
            "resolution": "test",
            "context_length": 128_000,
            "max_completion_tokens": 16_000,
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
            first, first_events, first_resumed = await _recovery_atomic_chat(
                run_id=self.run_id,
                sequence_key="checkpoint-test",
                model=ORCHESTRATOR_MODEL,
                messages=messages,
                response_schema=schema,
                schema_name="aiv_recovery_checkpoint_test",
                reasoning_effort="high",
                temperature=0.1,
            )
            second, second_events, second_resumed = await _recovery_atomic_chat(
                run_id=self.run_id,
                sequence_key="checkpoint-test",
                model=ORCHESTRATOR_MODEL,
                messages=messages,
                response_schema=schema,
                schema_name="aiv_recovery_checkpoint_test",
                reasoning_effort="high",
                temperature=0.1,
            )

        self.assertEqual(post_count, 1)
        self.assertFalse(first_resumed)
        self.assertTrue(second_resumed)
        self.assertEqual(first.parsed, {"fact": "TAIL_ALPHA"})
        self.assertEqual(second.parsed, first.parsed)
        self.assertEqual(len(first_events), 1)
        self.assertEqual(second_events, first_events)
        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == self.run_id,
                            RunArtifact.stage_key
                            == "recovery_provider_checkpoint",
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].output_json["status"], "accepted")

    async def test_disabled_state_does_not_reserve_or_spend_epoch(self) -> None:
        with (
            patch(
                "app.services.recovery_state.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                False,
            ),
            patch(
                "app.services.recovery_state.plan_recovery",
                new_callable=AsyncMock,
            ) as planner,
        ):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "orchestrator is disabled",
            ):
                await plan_durable_recovery(
                    self.run_id,
                    stage_key="scenario_design",
                    failure_class="repairable_semantic",
                    failure_code="prompt_set_non_convergent",
                    diagnostics={"validation_errors": ["x"]},
                    facts={"profile_sha256": "a" * 64},
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )
        planner.assert_not_awaited()
        async with self.SessionLocal() as session:
            count = len(
                list(
                    (
                        await session.execute(
                            select(RecoveryEpoch).where(
                                RecoveryEpoch.run_id == self.run_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            )
        self.assertEqual(count, 0)

    async def test_unstarted_diagnosis_is_reused_without_spending_budget(self) -> None:
        diagnostics = {"validation_errors": ["x"]}
        facts = {"profile_sha256": "a" * 64}
        fingerprint = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics=diagnostics,
        )
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
        )
        async with self.SessionLocal() as session:
            prepared = RecoveryEpoch(
                run_id=self.run_id,
                epoch=1,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                failure_fingerprint=fingerprint,
                facts_digest=facts_digest,
                status="diagnosing",
                # Legacy/pre-reservation rows may already carry the configured
                # model name; status, not model presence, proves call start.
                model=ORCHESTRATOR_MODEL,
                input_json={"planner_attempt": {"started": False}},
            )
            session.add(prepared)
            await session.commit()
            await session.refresh(prepared)
            prepared_id = prepared.id

        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["prompt_contract_valid"],
            "incident_fingerprint": fingerprint,
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        with (
            patch(
                "app.services.recovery_state.settings."
                "PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN",
                1,
            ),
            patch(
                "app.services.recovery_state.plan_recovery",
                new=AsyncMock(return_value=planned),
            ) as planner,
        ):
            result = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics=diagnostics,
                facts=facts,
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )

        self.assertEqual(result.epoch_id, prepared_id)
        self.assertFalse(result.reused)
        planner.assert_awaited_once()
        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RecoveryEpoch).where(
                            RecoveryEpoch.run_id == self.run_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "planned")
        self.assertEqual(rows[0].model, ORCHESTRATOR_MODEL)
        self.assertTrue(rows[0].input_json["planner_attempt"]["started"])
        self.assertTrue(rows[0].input_json["planner_attempt"]["completed"])
        self.assertEqual(
            rows[0].usage_json["_aiv_orchestrator"]["input_digest"],
            "input-digest",
        )
        self.assertEqual(
            rows[0].usage_json["_aiv_orchestrator"]["raw_text"],
            "{}",
        )

    async def test_interrupted_planner_attempt_is_accounted_and_reconciled(self) -> None:
        diagnostics = {"validation_errors": ["x"]}
        facts = {"profile_sha256": "a" * 64}
        fingerprint = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics=diagnostics,
        )
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
        )
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="scenario_design",
                    failure_class="repairable_semantic",
                    failure_code="prompt_set_non_convergent",
                    failure_fingerprint=fingerprint,
                    facts_digest=facts_digest,
                    status="planning",
                    model=ORCHESTRATOR_MODEL,
                    input_json={
                        "planner_attempt": {
                            "started": True,
                            "completed": False,
                        }
                    },
                )
            )
            await session.commit()

        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["prompt_contract_valid"],
            "incident_fingerprint": fingerprint,
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        with (
            patch(
                "app.services.recovery_state.settings."
                "PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN",
                2,
            ),
            patch(
                "app.services.recovery_state.plan_recovery",
                new=AsyncMock(return_value=planned),
            ) as planner,
        ):
            result = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics=diagnostics,
                facts=facts,
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )
            reused = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics=diagnostics,
                facts=facts,
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )

        planner.assert_awaited_once()
        self.assertTrue(reused.reused)
        self.assertEqual(reused.epoch_id, result.epoch_id)
        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RecoveryEpoch)
                        .where(RecoveryEpoch.run_id == self.run_id)
                        .order_by(RecoveryEpoch.epoch)
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual([row.status for row in rows], ["failed", "planned"])
        self.assertTrue(
            rows[0].usage_json["planner_attempt"]["reconciled_after_restart"]
        )

    async def test_stage_planner_budget_blocks_second_post_after_restart(
        self,
    ) -> None:
        diagnostics = {"critic_iteration": 2, "issue_answer_ids": [569]}
        facts = {"critic_review_sha256": "a" * 64}
        fingerprint = recovery_failure_fingerprint(
            stage_key="analysis_critic",
            failure_class="repairable_semantic",
            failure_code="analysis_critic_non_convergent",
            diagnostics=diagnostics,
        )
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
            permitted_answer_ids={569},
        )
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    failure_fingerprint=fingerprint,
                    facts_digest=facts_digest,
                    status="planning",
                    model=ORCHESTRATOR_MODEL,
                    input_json={
                        "planner_attempt": {
                            "started": True,
                            "completed": False,
                        }
                    },
                )
            )
            await session.commit()

        with patch(
            "app.services.recovery_state.plan_recovery",
            new_callable=AsyncMock,
        ) as planner:
            with self.assertRaisesRegex(
                RecoveryBudgetExceeded,
                "stage planner call budget exhausted",
            ):
                await plan_durable_recovery(
                    self.run_id,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    diagnostics=diagnostics,
                    facts=facts,
                    allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                    permitted_answer_ids={569},
                    stage_planner_call_limit=1,
                )
        planner.assert_not_awaited()
        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "failed")
        self.assertTrue(
            epoch.usage_json["planner_attempt"][
                "reconciled_after_restart"
            ]
        )

    async def test_plan_survives_restart_and_finishes_once(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        kwargs = {
            "stage_key": "scenario_design",
            "failure_class": "repairable_semantic",
            "failure_code": "prompt_set_non_convergent",
            "diagnostics": {"validation_errors": ["x"]},
            "facts": {"profile_sha256": "a" * 64},
            "allowed_actions": {ACTION_DETERMINISTIC_FALLBACK},
        }
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ) as planner:
            first = await plan_durable_recovery(self.run_id, **kwargs)
            second = await plan_durable_recovery(self.run_id, **kwargs)

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(first.run_id, self.run_id)
        self.assertEqual(second.run_id, self.run_id)
        self.assertEqual(first.epoch_id, second.epoch_id)
        planner.assert_awaited_once()

        await mark_recovery_executing(second)
        await finish_recovery(
            second,
            succeeded=True,
            before_digest="before",
            after_digest="after",
            details={"accepted": True},
        )
        async with self.SessionLocal() as session:
            epochs = list(
                (
                    await session.execute(
                        select(RecoveryEpoch).where(
                            RecoveryEpoch.run_id == self.run_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(len(epochs), 1)
        self.assertEqual(epochs[0].status, "succeeded")
        self.assertEqual(epochs[0].plan_json, decision)
        self.assertTrue(epochs[0].outcome_json["succeeded"])
        self.assertEqual(epochs[0].outcome_json["execution_attempts"], 1)

    async def test_reused_plan_is_scope_validated_and_digest_checked(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text=json.dumps(decision, ensure_ascii=False),
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        kwargs = {
            "stage_key": "scenario_design",
            "failure_class": "repairable_semantic",
            "failure_code": "prompt_set_non_convergent",
            "diagnostics": {"validation_errors": ["x"]},
            "facts": {"profile_sha256": "a" * 64},
            "allowed_actions": {ACTION_DETERMINISTIC_FALLBACK},
        }
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ) as planner:
            first = await plan_durable_recovery(self.run_id, **kwargs)

            out_of_scope = {
                **decision,
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "target_answer_ids": [999],
                "acceptance_checks": ["raw_corpus_unchanged"],
            }
            async with self.SessionLocal() as session:
                await session.execute(
                    update(RecoveryEpoch)
                    .where(RecoveryEpoch.id == first.epoch_id)
                    .values(
                        plan_json=out_of_scope,
                        plan_digest=stable_digest(out_of_scope),
                    )
                )
                await session.commit()
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "not allowed|outside the incident",
            ):
                await plan_durable_recovery(self.run_id, **kwargs)

            async with self.SessionLocal() as session:
                await session.execute(
                    update(RecoveryEpoch)
                    .where(RecoveryEpoch.id == first.epoch_id)
                    .values(
                        plan_json=decision,
                        plan_digest=stable_digest(decision),
                    )
                )
                await session.commit()
            reused = await plan_durable_recovery(self.run_id, **kwargs)
            self.assertTrue(reused.reused)

            async with self.SessionLocal() as session:
                await session.execute(
                    update(RecoveryEpoch)
                    .where(RecoveryEpoch.id == first.epoch_id)
                    .values(plan_digest="0" * 64)
                )
                await session.commit()
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "digest mismatch",
            ):
                await mark_recovery_executing(reused)

        planner.assert_awaited_once()

    async def test_lost_lease_after_planner_cannot_publish_plan(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(execution_slot=1, lease_owner="current-owner")
            )
            await session.commit()

        async def lose_lease_during_planning(**_kwargs):
            async with self.SessionLocal() as session:
                await session.execute(
                    update(Run)
                    .where(Run.id == self.run_id)
                    .values(lease_owner="replacement-owner")
                )
                await session.commit()
            return planned

        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(side_effect=lose_lease_during_planning),
        ):
            with bind_run_lease(self.run_id, "current-owner"):
                with self.assertRaises(RunLeaseLostError):
                    await plan_durable_recovery(
                        self.run_id,
                        stage_key="scenario_design",
                        failure_class="repairable_semantic",
                        failure_code="prompt_set_non_convergent",
                        diagnostics={"validation_errors": ["x"]},
                        facts={"profile_sha256": "a" * 64},
                        allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                    )

        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "planning")
        self.assertEqual(epoch.model, ORCHESTRATOR_MODEL)
        self.assertTrue(epoch.input_json["planner_attempt"]["started"])
        self.assertFalse(epoch.input_json["planner_attempt"]["completed"])
        self.assertIsNone(epoch.plan_json)

    async def test_stale_lease_cannot_mark_or_finish_recovery(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ):
            plan = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics={"validation_errors": ["x"]},
                facts={"profile_sha256": "a" * 64},
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )

        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(execution_slot=1, lease_owner="current-owner")
            )
            await session.commit()

        with bind_run_lease(self.run_id, "stale-owner"):
            with self.assertRaises(RunLeaseLostError):
                await mark_recovery_executing(plan)
        async with self.SessionLocal() as session:
            status = (
                await session.execute(
                    select(RecoveryEpoch.status).where(
                        RecoveryEpoch.id == plan.epoch_id
                    )
                )
            ).scalar_one()
        self.assertEqual(status, "planned")

        with bind_run_lease(self.run_id, "current-owner"):
            await mark_recovery_executing(plan)
        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(lease_owner="replacement-owner")
            )
            await session.commit()

        with bind_run_lease(self.run_id, "current-owner"):
            with self.assertRaises(RunLeaseLostError):
                await finish_recovery(
                    plan,
                    succeeded=True,
                    before_digest="before",
                    after_digest="after",
                )
        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.id == plan.epoch_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "executing")
        self.assertEqual(epoch.outcome_json["execution_attempts"], 1)

    async def test_execution_attempt_budget_survives_worker_restarts(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        kwargs = {
            "stage_key": "scenario_design",
            "failure_class": "repairable_semantic",
            "failure_code": "prompt_set_non_convergent",
            "diagnostics": {"validation_errors": ["x"]},
            "facts": {"profile_sha256": "a" * 64},
            "allowed_actions": {ACTION_DETERMINISTIC_FALLBACK},
        }
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ) as planner:
            first_worker = await plan_durable_recovery(self.run_id, **kwargs)
            await mark_recovery_executing(first_worker)
            second_worker = await plan_durable_recovery(self.run_id, **kwargs)
            await mark_recovery_executing(second_worker)
            third_worker = await plan_durable_recovery(self.run_id, **kwargs)
            with self.assertRaises(RecoveryBudgetExceeded):
                await mark_recovery_executing(third_worker)

        planner.assert_awaited_once()
        self.assertTrue(second_worker.reused)
        self.assertTrue(third_worker.reused)
        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.id == first_worker.epoch_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "blocked")
        self.assertEqual(epoch.outcome_json["execution_attempts"], 2)
        self.assertEqual(
            epoch.outcome_json["reason"],
            "execution_attempt_budget_exhausted",
        )


if __name__ == "__main__":
    unittest.main()
