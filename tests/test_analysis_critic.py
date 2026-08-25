import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import (
    AnswerAnnotation,
    Base,
    ModelAnswer,
    RecoveryEpoch,
    Run,
    RunArtifact,
    RunStatus,
    VisibilityPrompt,
)

from app.services.analysis_critic import (
    CRITIC_MAX_TOKENS,
    CRITIC_MODEL,
    CRITIC_PRIMARY_RAW_CHAR_BUDGET,
    CRITIC_REASONING_EFFORT,
    CRITIC_REPAIR_MAX_TOKENS,
    CRITIC_REPAIR_REASONING_EFFORT,
    CRITIC_VERSION,
    MAX_CRITIC_RECOVERY_FINAL_REVIEWS,
    MAX_CRITIC_ITERATIONS,
    MAX_CRITIC_REPAIR_ATTEMPTS,
    _compact_repair_context,
    _transport_repair_may_pass,
    repair_analysis_review,
    review_analysis,
)
from app.services.analyzer import (
    ANALYSIS_CRITIC_VERSION,
    _AnalysisCriticRecoveryBlocked,
    ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
    ANNOTATION_VERSION,
    _analysis_critic_artifact,
    _annotation_context_sha256,
    _annotation_matches_answer,
    _artifact_cache_matches,
    _apply_critic_policy,
    _critic_payload,
    _critic_provenance_digests,
    _critic_review_errors,
    _critic_analysis_state_digest,
    _current_annotation_input_digests,
    _literal_target_attribution_evidence,
    _reconcile_annotation,
    _recover_analysis_critic_exhaustion,
    _raw_corpus_digest,
    _run_analysis_critic_loop,
    _save_critic_gate,
    _save_targeted_recovery_annotations,
    _scope_entity_catalog_to_profile,
    _terminal_analysis_critic_recovery_reason,
)
from app.services.openrouter import OpenRouterError, OpenRouterOutputLimitError
from app.services.recovery_orchestrator import (
    ACTION_STOP,
    ACTION_TARGETED_ANNOTATION_REPAIR,
    CHECK_CHECKPOINT_PRESERVED,
    CHECK_CRITIC_GATE_PASSED,
    CHECK_DERIVED_METRICS_RECOMPUTED,
    CHECK_RAW_CORPUS_UNCHANGED,
    OrchestratorContractError,
)
from app.services.recovery_state import stable_digest
from app.services.run_lease import (
    RunLeaseLostError,
    bind_run_lease,
)


def _critic_review(
    verdict: str,
    *,
    adjustments: list[dict] | None = None,
    guidance: str = "",
    anomalies: list[dict] | None = None,
) -> dict:
    return {
        "verdict": verdict,
        "summary": f"Critic verdict: {verdict}",
        "anomalies": anomalies or [],
        "policy_adjustments": adjustments or [],
        "annotation_guidance": guidance,
        "acceptance_checks": (
            ["Проверены числители, знаменатели и evidence."]
            if verdict == "pass"
            else []
        ),
    }


PROFILE = {
    "brand_name": "Example",
    "brand_aliases": [],
    "entity_scope": [
        {
            "canonical_name": "Campaign 360",
            "aliases": [],
            "relationship": "owned_by",
            "commercially_relevant": True,
            "confidence": "high",
        }
    ],
}
CATALOG = {
    "target_aliases": ["Example"],
    "entities": [
        {
            "canonical_name": "Campaign 360",
            "aliases": ["campaign"],
            "category": "target",
            "target_relationship": "portfolio_entity",
            "commercially_relevant": True,
            "mention_policy": "standalone",
        }
    ],
}
ROWS = [
    {
        "answer_id": 11,
        "mode": "web",
        "provider_key": "openai",
        "model": "openai/gpt-chat-latest",
        "prompt_id": 7,
        "prompt_key": "intent-t",
        "scenario": "Как выбрать систему управления кампанией?",
        "role": "unbranded_discovery",
        "intent_class": "T",
        "status": "completed",
        "annotation_state": "current",
        "citations_count": 2,
        "answer_text": "Для этой задачи подходит Campaign 360.",
        "annotation": {
            "_annotation_version": ANNOTATION_VERSION,
            "_answer_sha256": hashlib.sha256(
                "Для этой задачи подходит Campaign 360.".encode("utf-8")
            ).hexdigest(),
            "_answer_model": "openai/gpt-chat-latest",
            "_annotation_input_sha256": "annotation-context",
            "valid": True,
            "target_mentioned": False,
            "entity_mentions": [
                {
                    "canonical_name": "Campaign 360",
                    "position": 1,
                    "role": "recommended",
                    "attributed_to_target": False,
                    "evidence": "Campaign 360",
                }
            ],
        },
    }
]
METRICS = {
    "portfolio_visibility": {
        "web": {
            "mention_count": 1,
            "valid_answers": 1,
        }
    },
    "providers": [],
}


class AnalysisCriticTests(unittest.IsolatedAsyncioTestCase):
    async def test_critic_uses_independent_gemini_model_with_bounded_iteration(
        self,
    ) -> None:
        verdict = {
            "verdict": "pass",
            "summary": "Расчёт подтверждён.",
            "anomalies": [],
            "policy_adjustments": [],
            "annotation_guidance": "",
            "acceptance_checks": ["Числители сверены с raw-ответами."],
        }
        response = SimpleNamespace(
            parsed=verdict,
            text=json.dumps(verdict, ensure_ascii=False),
            usage={"total_tokens": 42},
        )
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ) as chat_mock:
            parsed, _raw, usage = await review_analysis(
                {"site_profile": {"brand_name": "Example"}},
                iteration=1,
            )

        self.assertEqual(parsed, verdict)
        self.assertEqual(usage["total_tokens"], 42)
        request = chat_mock.await_args.kwargs
        self.assertEqual(CRITIC_MODEL, "google/gemini-3.6-flash")
        self.assertEqual(request["model"], CRITIC_MODEL)
        self.assertEqual(request["reasoning_effort"], CRITIC_REASONING_EFFORT)
        self.assertEqual(request["max_tokens"], CRITIC_MAX_TOKENS)
        self.assertFalse(request["retry_response_contract_errors"])
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["iteration"], 1)
        self.assertEqual(payload["max_iterations"], MAX_CRITIC_ITERATIONS)
        self.assertEqual(CRITIC_VERSION, "aiv-analysis-critic-v19")
        self.assertEqual(
            usage["_aiv_critic_contract"]["semantic_verdict_status"],
            "pending_deterministic_validation",
        )
        system_prompt = request["messages"][0]["content"]
        self.assertNotIn("Realweb", system_prompt)
        self.assertIn("attribution_owner_aliases", system_prompt)
        self.assertIn("entity_attribution_aliases", system_prompt)
        self.assertIn("не переносит услуги", system_prompt)
        self.assertIn("не искажает", system_prompt)
        self.assertIn(
            "не исправляют потерю подтверждённого standalone-продукта",
            system_prompt,
        )

    def test_critic_cache_requires_validated_semantic_status(self) -> None:
        artifact = SimpleNamespace(
            status="completed",
            prompt_version="critic-v",
            output_json=_critic_review("pass"),
            input_json={"facts": "same"},
            model=CRITIC_MODEL,
            usage_json={
                "_aiv_critic_contract": {
                    "semantic_verdict_status": (
                        "pending_deterministic_validation"
                    )
                }
            },
        )

        self.assertFalse(
            _artifact_cache_matches(
                artifact,
                input_json={"facts": "same"},
                model=CRITIC_MODEL,
                prompt_version="critic-v",
                require_validated_critic_usage=True,
            )
        )
        artifact.usage_json["_aiv_critic_contract"][
            "semantic_verdict_status"
        ] = "validated"
        self.assertTrue(
            _artifact_cache_matches(
                artifact,
                input_json={"facts": "same"},
                model=CRITIC_MODEL,
                prompt_version="critic-v",
                require_validated_critic_usage=True,
            )
        )

    async def test_validated_primary_verdict_is_saved_with_semantic_proof(
        self,
    ) -> None:
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        pending_usage = {
            "total_tokens": 12,
            "_aiv_critic_contract": {
                "semantic_verdict_status": (
                    "pending_deterministic_validation"
                ),
                "semantic_validation_owner": "critic_gate",
            },
        }
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ) as artifact_output,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(_critic_review("pass"), "{}", pending_usage),
            ),
        ):
            result = await _analysis_critic_artifact(
                "run-primary-validated",
                iteration=1,
                payload=payload,
            )

        self.assertEqual(result["verdict"], "pass")
        self.assertTrue(
            artifact_output.await_args.kwargs[
                "require_validated_critic_usage"
            ]
        )
        completed = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"] == "analysis_critic_r1"
            and call.kwargs["status"] == "completed"
        ]
        self.assertEqual(len(completed), 1)
        self.assertEqual(
            completed[0]["usage_json"]["_aiv_critic_contract"][
                "semantic_verdict_status"
            ],
            "validated",
        )

    async def test_recovery_final_never_promotes_invalid_primary_via_repair(
        self,
    ) -> None:
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        invalid_primary = _critic_review("revise")
        invalid_primary["policy_adjustments"] = []
        invalid_primary["annotation_guidance"] = ""
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(invalid_primary, "{}", {}),
            ),
            patch(
                "app.services.analyzer._analysis_critic_repair_artifact",
                new_callable=AsyncMock,
                return_value=_critic_review("pass"),
            ) as repair,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "Final recovery critic primary decision is inconsistent",
            ):
                await _analysis_critic_artifact(
                    "run-final-primary-only",
                    iteration=3,
                    payload=payload,
                    recovery_final=True,
                )
        repair.assert_not_awaited()

    async def test_incomplete_review_repair_gets_compact_audit_context(
        self,
    ) -> None:
        incomplete = _critic_review("revise")
        incomplete["anomalies"] = [
            {
                "code": "generic_term_leakage",
                "severity": "important",
                "finding": "Общий термин ошибочно привязан к бренду.",
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]
        repaired = _critic_review("pass")
        response = SimpleNamespace(
            parsed=repaired,
            text=json.dumps(repaired, ensure_ascii=False),
            usage={"total_tokens": 21},
        )
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ) as chat_mock:
            parsed, _raw, usage = await repair_analysis_review(
                payload,
                incomplete,
                iteration=1,
                validation_errors=[
                    "revise contains no policy adjustments",
                ],
            )

        self.assertEqual(parsed, repaired)
        self.assertEqual(usage["total_tokens"], 21)
        request = chat_mock.await_args.kwargs
        self.assertEqual(request["model"], CRITIC_MODEL)
        self.assertEqual(
            request["reasoning_effort"],
            CRITIC_REPAIR_REASONING_EFFORT,
        )
        self.assertEqual(request["max_tokens"], CRITIC_REPAIR_MAX_TOKENS)
        self.assertFalse(request["retry_response_contract_errors"])
        self.assertEqual(request["temperature"], 0.0)
        repair_payload = json.loads(request["messages"][1]["content"])
        self.assertNotIn("audit_payload", repair_payload)
        self.assertEqual(
            repair_payload["audit_payload_sha256"],
            repair_payload["repair_context"]["audit_payload_sha256"],
        )
        context = repair_payload["repair_context"]
        self.assertEqual(context["candidate_metrics"], payload["candidate_metrics"])
        self.assertEqual(context["answer_index"][0]["answer_id"], 11)
        self.assertEqual(
            context["affected_answer_evidence"][0]["raw_answer"],
            ROWS[0]["answer_text"],
        )
        self.assertEqual(repair_payload["incomplete_review"], incomplete)
        self.assertEqual(repair_payload["repair_attempt"], 1)
        self.assertEqual(
            repair_payload["max_repair_attempts"],
            MAX_CRITIC_REPAIR_ATTEMPTS,
        )
        self.assertEqual(MAX_CRITIC_REPAIR_ATTEMPTS, 1)

    def test_repair_context_does_not_resend_the_full_answer_corpus(self) -> None:
        large_payload = {
            "site_profile": PROFILE,
            "entity_catalog": CATALOG,
            "metric_contract": {"portfolio_visibility": "test"},
            "candidate_metrics": METRICS,
            "deterministic_warnings": [
                {
                    "code": "annotation_evidence_mismatch",
                    "severity": "important",
                    "answer_ids": [11],
                    "entities": ["Campaign 360"],
                }
            ],
            "answers": [
                {
                    "answer_id": answer_id,
                    "provider": "openai",
                    "mode": "web",
                    "scenario_role": "unbranded_discovery",
                    "status": "completed",
                    "annotation_state": "current",
                    "raw_answer_sha256": f"hash-{answer_id}",
                    "raw_answer_truncated": False,
                    "raw_answer": "x" * 20_000,
                    "annotation": {"valid": True},
                }
                for answer_id in range(11, 92)
            ],
        }
        incomplete = _critic_review("revise")
        incomplete["anomalies"] = [
            {
                "code": "annotation_evidence_mismatch",
                "severity": "important",
                "finding": "Требуется проверить один ответ.",
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]

        compact = _compact_repair_context(large_payload, incomplete)

        full_size = len(json.dumps(large_payload, ensure_ascii=False))
        compact_size = len(json.dumps(compact, ensure_ascii=False))
        self.assertLess(compact_size, full_size // 10)
        self.assertEqual(len(compact["answer_index"]), 81)
        self.assertEqual(len(compact["affected_answer_evidence"]), 1)
        self.assertEqual(
            len(compact["affected_answer_evidence"][0]["raw_answer"]),
            6_000,
        )
        self.assertTrue(
            compact["affected_answer_evidence"][0]["repair_raw_truncated"]
        )

    def test_transport_repair_cannot_invent_a_passing_verdict(self) -> None:
        self.assertFalse(
            _transport_repair_may_pass({"_parsed_partial_review": None})
        )
        self.assertFalse(
            _transport_repair_may_pass(
                {
                    "_parsed_partial_review": {
                        "verdict": "revise",
                        "anomalies": [],
                    }
                }
            )
        )
        self.assertFalse(
            _transport_repair_may_pass(
                {
                    "_parsed_partial_review": {
                        "verdict": "pass",
                        "anomalies": [{"severity": "important"}],
                    }
                }
            )
        )
        self.assertTrue(
            _transport_repair_may_pass(
                {
                    "_parsed_partial_review": {
                        "verdict": "pass",
                        "anomalies": [],
                    }
                }
            )
        )

    async def test_repair_blocks_when_referenced_evidence_exceeds_cap(
        self,
    ) -> None:
        answer_ids = list(range(100, 113))
        payload = {
            "site_profile": PROFILE,
            "entity_catalog": CATALOG,
            "candidate_metrics": METRICS,
            "deterministic_warnings": [
                {
                    "code": "annotation_evidence_mismatch",
                    "severity": "important",
                    "answer_ids": answer_ids,
                    "entities": ["Campaign 360"],
                }
            ],
            "answers": [
                {
                    "answer_id": answer_id,
                    "raw_answer": "Evidence",
                    "raw_answer_truncated": False,
                }
                for answer_id in answer_ids
            ],
        }
        incomplete = _critic_review("revise")
        response = SimpleNamespace(
            parsed=_critic_review("pass"),
            text="{}",
            usage={},
        )
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ),
            self.assertRaises(OpenRouterError) as raised,
        ):
            await repair_analysis_review(
                payload,
                incomplete,
                iteration=1,
                validation_errors=["incomplete"],
            )

        self.assertIn("only block is safe", str(raised.exception))

    async def test_repair_blocks_when_affected_raw_answer_is_truncated(
        self,
    ) -> None:
        payload = {
            "site_profile": PROFILE,
            "entity_catalog": CATALOG,
            "candidate_metrics": METRICS,
            "deterministic_warnings": [
                {
                    "code": "annotation_evidence_mismatch",
                    "severity": "important",
                    "answer_ids": [11],
                    "entities": ["Campaign 360"],
                }
            ],
            "answers": [
                {
                    "answer_id": 11,
                    "raw_answer": "Evidence " * 1_000,
                    "raw_answer_truncated": False,
                }
            ],
        }
        response = SimpleNamespace(
            parsed=_critic_review("pass"),
            text="{}",
            usage={},
        )
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ),
            self.assertRaises(OpenRouterError) as raised,
        ):
            await repair_analysis_review(
                payload,
                _critic_review("revise"),
                iteration=1,
                validation_errors=["incomplete"],
            )

        self.assertIn("only block is safe", str(raised.exception))

    async def test_repair_blocks_when_referenced_raw_was_manifest_only(
        self,
    ) -> None:
        payload = {
            "site_profile": PROFILE,
            "entity_catalog": CATALOG,
            "candidate_metrics": METRICS,
            "deterministic_warnings": [
                {
                    "code": "annotation_evidence_mismatch",
                    "severity": "important",
                    "answer_ids": [11],
                    "entities": ["Campaign 360"],
                }
            ],
            "answers": [
                {
                    "answer_id": 11,
                    "raw_answer": "",
                    "raw_answer_included": False,
                    "raw_answer_char_count": 420,
                    "raw_answer_omission_reason": (
                        "deterministic_stratified_manifest"
                    ),
                }
            ],
        }
        response = SimpleNamespace(
            parsed=_critic_review("pass"),
            text="{}",
            usage={},
        )
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ),
            self.assertRaises(OpenRouterError) as raised,
        ):
            await repair_analysis_review(
                payload,
                _critic_review("revise"),
                iteration=1,
                validation_errors=["incomplete"],
            )

        self.assertIn("only block is safe", str(raised.exception))

    async def test_output_limit_uses_one_compact_repair_attempt(self) -> None:
        transport = {
            "status": "succeeded",
            "output_complete": False,
            "output_limited": True,
            "finish_reason": "length",
            "native_finish_reason": "MAX_TOKENS",
        }
        primary_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20_000,
            "total_tokens": 20_100,
            "_aiv_transport": transport,
        }
        limited = OpenRouterOutputLimitError(
            "OpenRouter response hit the output limit",
            result=SimpleNamespace(
                text='{"verdict":"revise","summary":"Оборвано"',
                usage=primary_usage,
                transport=transport,
            ),
        )
        repaired = _critic_review("block")
        repair_usage = {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
            "_aiv_transport": {
                "status": "succeeded",
                "output_complete": True,
            },
        }
        repair_response = SimpleNamespace(
            parsed=repaired,
            text=json.dumps(repaired, ensure_ascii=False),
            usage=repair_usage,
        )
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            side_effect=[limited, repair_response],
        ) as chat_mock:
            parsed, _raw, usage = await review_analysis(payload, iteration=1)

        self.assertEqual(parsed, repaired)
        self.assertEqual(chat_mock.await_count, 2)
        primary_request, repair_request = [
            call.kwargs for call in chat_mock.await_args_list
        ]
        self.assertEqual(primary_request["max_tokens"], CRITIC_MAX_TOKENS)
        self.assertEqual(
            repair_request["max_tokens"],
            CRITIC_REPAIR_MAX_TOKENS,
        )
        repair_command = json.loads(repair_request["messages"][1]["content"])
        self.assertNotIn("audit_payload", repair_command)
        self.assertEqual(
            repair_command["incomplete_review"]["_transport_failure"],
            "output_limit",
        )
        self.assertEqual(usage["total_tokens"], 20_800)
        self.assertEqual(len(usage["_aiv_critic_attempts"]), 2)
        self.assertEqual(
            usage["_aiv_critic_contract"]["recovered_from"],
            "output_limit",
        )
        self.assertEqual(
            usage["_aiv_critic_contract"]["semantic_verdict_status"],
            "pending_deterministic_validation",
        )

    async def test_recovery_final_transport_failure_never_calls_repair(
        self,
    ) -> None:
        transport = {
            "status": "succeeded",
            "output_complete": False,
            "output_limited": True,
            "finish_reason": "length",
        }
        limited = OpenRouterOutputLimitError(
            "OpenRouter response hit the output limit",
            result=SimpleNamespace(
                text=(
                    '{"verdict":"pass","summary":"Проверено",'
                    '"anomalies":[]'
                ),
                usage={"_aiv_transport": transport},
                transport=transport,
            ),
        )
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=limited,
            ) as chat_mock,
            patch(
                "app.services.analysis_critic.repair_analysis_review",
                new_callable=AsyncMock,
            ) as repair,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "compact repair is forbidden",
            ):
                await review_analysis(
                    payload,
                    iteration=3,
                    recovery_final=True,
                )
        chat_mock.assert_awaited_once()
        repair.assert_not_awaited()

    def test_payload_contains_every_row_field_needed_to_audit_metrics(
        self,
    ) -> None:
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )

        answer = payload["answers"][0]
        self.assertIn(
            "brand_diagnostic",
            payload["metric_contract"]["portfolio_visibility"],
        )
        self.assertEqual(answer["prompt_id"], 7)
        self.assertEqual(answer["prompt_key"], "intent-t")
        self.assertEqual(answer["model"], "openai/gpt-chat-latest")
        self.assertEqual(answer["annotation_state"], "current")
        self.assertEqual(answer["citations_count"], 2)
        self.assertEqual(answer["raw_answer"], ROWS[0]["answer_text"])
        self.assertFalse(answer["raw_answer_truncated"])
        self.assertIn(
            "attributed_to_target=false",
            payload["metric_contract"]["portfolio_visibility"],
        )

    def test_primary_critic_payload_uses_bounded_stratified_raw(self) -> None:
        provider_keys = ["openai", "gemini", "perplexity", "deepseek", "claude"]
        rows: list[dict] = []
        for index in range(81):
            rows.append(
                {
                    **ROWS[0],
                    "answer_id": index + 1,
                    "provider_key": provider_keys[index % len(provider_keys)],
                    "mode": "web" if index % 2 else "memory",
                    "role": "unbranded_discovery",
                    "status": "failed" if index == 1 else "completed",
                    "answer_text": f"Ответ {index}. " + ("x" * 12_000),
                    "annotation": {
                        **ROWS[0]["annotation"],
                        "target_mentioned": index % 4 == 0,
                        "entity_mentions": [],
                    },
                }
            )

        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=rows,
            metrics=METRICS,
            policy_history=[],
        )

        included = [
            answer
            for answer in payload["answers"]
            if answer["raw_answer_included"]
        ]
        omitted = [
            answer
            for answer in payload["answers"]
            if not answer["raw_answer_included"]
        ]
        self.assertTrue(included)
        self.assertTrue(omitted)
        self.assertLess(len(included), 50)
        self.assertEqual(
            payload["raw_evidence_selection"]["included_raw_count"],
            len(included),
        )
        self.assertTrue(all(not answer["raw_answer"] for answer in omitted))
        self.assertTrue(
            all(answer["raw_answer_sha256"] for answer in omitted)
        )
        full_raw_chars = sum(len(row["answer_text"]) for row in rows)
        sent_raw_chars = sum(
            len(answer["raw_answer"]) for answer in payload["answers"]
        )
        self.assertLess(sent_raw_chars, full_raw_chars // 2)
        self.assertLessEqual(
            sent_raw_chars,
            CRITIC_PRIMARY_RAW_CHAR_BUDGET,
        )
        self.assertEqual(
            sent_raw_chars,
            payload["raw_evidence_selection"]["included_raw_chars"],
        )
        self.assertGreater(
            payload["raw_evidence_selection"]["included_class_counts"][
                "positive"
            ],
            0,
        )
        self.assertGreater(
            payload["raw_evidence_selection"]["included_class_counts"][
                "negative"
            ],
            0,
        )
        self.assertGreater(
            payload["raw_evidence_selection"]["included_class_counts"][
                "failure"
            ],
            0,
        )

    def test_warning_raw_over_budget_is_manifested_and_fail_closed(self) -> None:
        rows = [
            {
                **ROWS[0],
                "answer_id": index,
                "answer_text": "x" * 24_000,
            }
            for index in range(1, 9)
        ]
        warning = {
            "code": "annotation_evidence_mismatch",
            "severity": "important",
            "finding": "Материальное предупреждение требует raw-проверки.",
            "answer_ids": list(range(1, 9)),
            "entities": ["Campaign 360"],
        }
        with (
            patch(
                "app.services.analyzer._deterministic_metric_warnings",
                return_value=[warning],
            ),
            patch(
                "app.services.analyzer._deterministic_annotation_warnings",
                return_value=[],
            ),
        ):
            payload = _critic_payload(
                profile=PROFILE,
                catalog=CATALOG,
                rows=rows,
                metrics=METRICS,
                policy_history=[],
            )

        selection = payload["raw_evidence_selection"]
        self.assertLessEqual(
            selection["included_raw_chars"],
            CRITIC_PRIMARY_RAW_CHAR_BUDGET,
        )
        self.assertTrue(selection["omitted_warning_answer_ids"])
        omitted = set(selection["omitted_warning_answer_ids"])
        self.assertTrue(
            all(
                answer["raw_answer_omission_reason"]
                == "primary_raw_budget_warning_evidence"
                for answer in payload["answers"]
                if answer["answer_id"] in omitted
            )
        )
        errors = _critic_review_errors(
            _critic_review("pass"),
            payload=payload,
        )
        self.assertTrue(
            any(
                "warning-linked raw answers missing or omitted" in error
                for error in errors
            )
        )

    def test_truncated_raw_cannot_open_or_revise_gate(self) -> None:
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=[{**ROWS[0], "answer_text": "x" * 24_001}],
            metrics=METRICS,
            policy_history=[],
        )

        for verdict in ("pass", "revise"):
            with self.subTest(verdict=verdict):
                errors = _critic_review_errors(
                    _critic_review(verdict),
                    payload=payload,
                )
                self.assertTrue(
                    any(
                        "truncated raw answers require block" in error
                        for error in errors
                    )
                )
        self.assertFalse(
            _critic_review_errors(
                _critic_review("block"),
                payload=payload,
            )
        )

    async def test_critic_refuses_a_third_iteration(self) -> None:
        with self.assertRaises(ValueError):
            await review_analysis({}, iteration=MAX_CRITIC_ITERATIONS + 1)

    async def test_critic_preserves_null_required_fields_for_gate_repair(
        self,
    ) -> None:
        response = SimpleNamespace(
            parsed={
                "verdict": "pass",
                "summary": "Расчёт подтверждён.",
                "anomalies": None,
                "policy_adjustments": None,
                "annotation_guidance": None,
                "acceptance_checks": None,
            },
            text="{}",
            usage={},
        )
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ):
            parsed, _raw, _usage = await review_analysis({}, iteration=1)

        self.assertIsNone(parsed["anomalies"])
        self.assertIsNone(parsed["policy_adjustments"])
        self.assertIsNone(parsed["annotation_guidance"])
        self.assertIsNone(parsed["acceptance_checks"])
        self.assertTrue(_critic_review_errors(parsed))

    async def test_incomplete_revise_is_repaired_once_before_artifact_completes(
        self,
    ) -> None:
        incomplete = _critic_review("revise")
        incomplete["anomalies"] = [
            {
                "code": "generic_term_leakage",
                "severity": "important",
                "finding": "Общий alias требует явной атрибуции.",
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]
        adjustment = {
            "action": "require_alias_attribution",
            "entity_name": "Campaign 360",
            "alias": "campaign",
            "reason": "Общий alias не идентифицирует продукт.",
            "answer_ids": [11],
        }
        repaired = _critic_review(
            "revise",
            adjustments=[adjustment],
            guidance="Требовать буквальную связь с целевым брендом.",
        )
        repaired["anomalies"] = incomplete["anomalies"]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )

        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(incomplete, "{}", {"total_tokens": 10}),
            ),
            patch(
                "app.services.analyzer.repair_analysis_review",
                new_callable=AsyncMock,
                return_value=(repaired, "{}", {"total_tokens": 5}),
            ) as repair_mock,
        ):
            result = await _analysis_critic_artifact(
                "run-repair",
                iteration=1,
                payload=payload,
            )

        self.assertEqual(result, repaired)
        repair_mock.assert_awaited_once()
        repair_request = repair_mock.await_args
        self.assertEqual(repair_request.args[0], payload)
        self.assertEqual(repair_request.args[1], incomplete)
        self.assertIn(
            "no policy adjustments",
            " ".join(repair_request.kwargs["validation_errors"]),
        )
        writes = [call.kwargs for call in save_artifact.await_args_list]
        self.assertEqual(
            [
                item["status"]
                for item in writes
                if item["artifact_key"] == "analysis_critic_r1_repair"
            ],
            ["running", "completed"],
        )
        repair_completed = [
            item
            for item in writes
            if item["artifact_key"] == "analysis_critic_r1_repair"
            and item["status"] == "completed"
        ]
        self.assertEqual(
            repair_completed[0]["usage_json"]["_aiv_critic_contract"][
                "semantic_verdict_status"
            ],
            "validated",
        )
        main_completed = [
            item
            for item in writes
            if item["artifact_key"] == "analysis_critic_r1"
            and item["status"] == "completed"
        ]
        self.assertEqual(main_completed[0]["output_json"], repaired)

    async def test_transport_repair_budget_cannot_be_spent_twice(self) -> None:
        still_invalid = _critic_review("revise")
        still_invalid["anomalies"] = [
            {
                "code": "generic_term_leakage",
                "severity": "important",
                "finding": "Нужна сужающая политика.",
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        already_repaired_usage = {
            "_aiv_critic_attempts": [
                {"kind": "primary", "usage": {}},
                {"kind": "compact_repair", "usage": {}},
            ]
        }
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(
                    still_invalid,
                    "{}",
                    already_repaired_usage,
                ),
            ),
            patch(
                "app.services.analyzer.repair_analysis_review",
                new_callable=AsyncMock,
            ) as second_repair,
        ):
            result = await _analysis_critic_artifact(
                "run-transport-repair-budget",
                iteration=1,
                payload=payload,
            )

        second_repair.assert_not_awaited()
        self.assertEqual(result["verdict"], "block")
        self.assertEqual(result["fallback"]["kind"], "deterministic_block")
        fallback_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"]
            == "analysis_critic_r1_fallback"
        ]
        self.assertEqual(len(fallback_writes), 1)
        self.assertEqual(fallback_writes[0]["status"], "failed")

    async def test_repair_may_pass_only_after_material_findings_are_observations(
        self,
    ) -> None:
        incomplete = _critic_review("revise")
        incomplete["anomalies"] = [
            {
                "code": "unsupported_membership",
                "severity": "important",
                "finding": (
                    "Вспомогательный флаг стоит у уже исключённой сущности."
                ),
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]
        repaired = _critic_review("pass")
        repaired["anomalies"] = [
            {
                **incomplete["anomalies"][0],
                "severity": "observation",
            }
        ]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(incomplete, "{}", {}),
            ),
            patch(
                "app.services.analyzer.repair_analysis_review",
                new_callable=AsyncMock,
                return_value=(repaired, "{}", {}),
            ) as repair_mock,
        ):
            result = await _analysis_critic_artifact(
                "run-observation-repair",
                iteration=1,
                payload=payload,
            )

        self.assertEqual(result["verdict"], "pass")
        repair_mock.assert_awaited_once()

    async def test_invalid_repair_returns_deterministic_block_without_retry(
        self,
    ) -> None:
        incomplete = _critic_review("revise")
        incomplete["anomalies"] = [
            {
                "code": "scope_leakage",
                "severity": "critical",
                "finding": "Обнаружена утечка scope.",
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(incomplete, "{}", {}),
            ),
            patch(
                "app.services.analyzer.repair_analysis_review",
                new_callable=AsyncMock,
                return_value=(incomplete, "{}", {}),
            ) as repair_mock,
        ):
            result = await _analysis_critic_artifact(
                "run-invalid-repair",
                iteration=1,
                payload=payload,
            )

        self.assertEqual(repair_mock.await_count, 1)
        self.assertEqual(result["verdict"], "block")
        self.assertEqual(
            result["fallback"]["kind"],
            "deterministic_block",
        )
        failed_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["status"] == "failed"
        ]
        self.assertEqual(
            {item["artifact_key"] for item in failed_writes},
            {
                "analysis_critic_r1_repair",
                "analysis_critic_r1_fallback",
                "analysis_critic_r1",
            },
        )
        fallback_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"]
            == "analysis_critic_r1_fallback"
        ]
        self.assertEqual(len(fallback_writes), 1)
        self.assertEqual(
            fallback_writes[0]["output_json"]["verdict"],
            "block",
        )
        self.assertEqual(fallback_writes[0]["status"], "failed")

    async def test_deterministic_safe_pass_never_poison_main_model_cache(
        self,
    ) -> None:
        incomplete = _critic_review("pass")
        incomplete["acceptance_checks"] = []
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.review_analysis",
                new_callable=AsyncMock,
                return_value=(incomplete, "{}", {"total_tokens": 10}),
            ),
            patch(
                "app.services.analyzer.repair_analysis_review",
                new_callable=AsyncMock,
                side_effect=OpenRouterError("repair failed"),
            ),
        ):
            result = await _analysis_critic_artifact(
                "run-safe-fallback",
                iteration=1,
                payload=payload,
            )

        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(
            result["fallback"]["kind"],
            "deterministic_safe_pass",
        )
        main_terminal = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"] == "analysis_critic_r1"
            and call.kwargs["status"] != "running"
        ]
        self.assertEqual(len(main_terminal), 1)
        self.assertEqual(main_terminal[0]["status"], "failed")
        self.assertEqual(
            main_terminal[0]["model"],
            "deterministic/critic-fallback-v1",
        )
        self.assertEqual(
            main_terminal[0]["usage_json"]["_aiv_critic_contract"][
                "semantic_verdict_status"
            ],
            "validated",
        )
        fallback_terminal = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"]
            == "analysis_critic_r1_fallback"
        ]
        self.assertEqual(fallback_terminal[0]["status"], "completed")
        self.assertEqual(
            fallback_terminal[0]["usage_json"]["_aiv_critic_contract"][
                "semantic_verdict_status"
            ],
            "validated",
        )

    async def test_gate_binds_profile_catalog_rows_and_metrics(self) -> None:
        with patch(
            "app.services.analyzer._save_artifact",
            new_callable=AsyncMock,
        ) as save_artifact:
            gate = await _save_critic_gate(
                "run-provenance",
                passed=True,
                iteration=1,
                profile=PROFILE,
                catalog=CATALOG,
                rows=ROWS,
                metrics=METRICS,
                policy_history=[],
                reason="Проверка пройдена.",
            )

        expected = _critic_provenance_digests(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        self.assertEqual(gate["provenance"], expected)
        self.assertEqual(gate["metrics_sha256"], expected["metrics_sha256"])
        self.assertTrue(gate["corpus_manifest"]["complete"])
        self.assertEqual(gate["corpus_manifest"]["answer_ids"], [11])
        self.assertEqual(
            gate["corpus_manifest"]["observed_cells"][0][
                "raw_answer_sha256"
            ],
            expected_raw_sha256 := hashlib.sha256(
                ROWS[0]["answer_text"].encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(len(expected_raw_sha256), 64)
        self.assertEqual(
            gate["corpus_manifest"]["critic_rows_sha256"],
            expected["rows_sha256"],
        )
        self.assertEqual(
            save_artifact.await_args.kwargs["input_json"]["provenance"],
            expected,
        )

        changed_raw = [{**ROWS[0], "answer_text": "Другой исходный ответ."}]
        changed_scenario = [{**ROWS[0], "scenario": "Другой сценарий"}]
        changed_annotation = [
            {
                **ROWS[0],
                "annotation": {
                    **ROWS[0]["annotation"],
                    "target_mentioned": True,
                },
            }
        ]
        for changed_rows in (
            changed_raw,
            changed_scenario,
            changed_annotation,
        ):
            with self.subTest(changed_rows=changed_rows):
                changed = _critic_provenance_digests(
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=changed_rows,
                    metrics=METRICS,
                    policy_history=[],
                )
                self.assertNotEqual(
                    changed["rows_sha256"],
                    expected["rows_sha256"],
                )

    async def test_pass_publishes_without_reannotation(self) -> None:
        gate = {
            "passed": True,
            "iteration": 1,
            "reason": "Расчёт подтверждён.",
        }
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                return_value=_critic_review("pass"),
            ) as critic_mock,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value=gate,
            ) as gate_mock,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate_mock,
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
            ) as rows_mock,
            patch("app.services.analyzer._compute_metrics") as metrics_mock,
        ):
            catalog, rows, metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-pass",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
            )

        self.assertEqual(
            catalog["entities"][0]["canonical_name"],
            CATALOG["entities"][0]["canonical_name"],
        )
        self.assertTrue(
            catalog["entities"][0]["_profile_membership_confirmed"]
        )
        self.assertNotIn(
            "_profile_membership_confirmed",
            CATALOG["entities"][0],
        )
        self.assertEqual(rows, ROWS)
        self.assertEqual(metrics, METRICS)
        self.assertEqual(returned_gate, gate)
        self.assertEqual(critic_mock.await_count, 1)
        self.assertEqual(
            critic_mock.await_args.kwargs["iteration"],
            1,
        )
        self.assertTrue(gate_mock.await_args.kwargs["passed"])
        annotate_mock.assert_not_awaited()
        rows_mock.assert_not_awaited()
        metrics_mock.assert_not_called()

    async def test_revise_then_pass_reannotates_and_recomputes_once(
        self,
    ) -> None:
        adjustment = {
            "action": "require_alias_attribution",
            "entity_name": "Campaign 360",
            "alias": "campaign",
            "reason": "Общий alias не идентифицирует продукт сам по себе.",
            "answer_ids": [11],
        }
        guidance = (
            "Считай alias campaign только при явной связи с брендом Example."
        )
        revised_rows = [
            {
                **ROWS[0],
                "annotation": {
                    **ROWS[0]["annotation"],
                    "entity_mentions": [],
                },
            }
        ]
        revised_metrics = {
            "portfolio_visibility": {
                "web": {
                    "mention_count": 0,
                    "valid_answers": 1,
                }
            },
            "providers": [],
        }
        gate = {
            "passed": True,
            "iteration": 2,
            "reason": "Повторная проверка пройдена.",
        }
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review(
                        "revise",
                        adjustments=[adjustment],
                        guidance=guidance,
                    ),
                    _critic_review("pass"),
                ],
            ) as critic_mock,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value=gate,
            ) as gate_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as artifact_mock,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate_mock,
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
                return_value=revised_rows,
            ) as rows_mock,
            patch(
                "app.services.analyzer._compute_metrics",
                return_value=revised_metrics,
            ) as metrics_mock,
        ):
            catalog, rows, metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-revise-pass",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
            )

        self.assertEqual(critic_mock.await_count, 2)
        self.assertEqual(
            [
                call.kwargs["iteration"]
                for call in critic_mock.await_args_list
            ],
            [1, 2],
        )
        annotate_mock.assert_awaited_once()
        rows_mock.assert_awaited_once()
        metrics_mock.assert_called_once()
        self.assertEqual(rows, revised_rows)
        self.assertEqual(metrics, revised_metrics)
        self.assertEqual(returned_gate, gate)
        self.assertTrue(gate_mock.await_args.kwargs["passed"])
        self.assertEqual(
            gate_mock.await_args.kwargs["iteration"],
            2,
        )

        effective_entity = catalog["entities"][0]
        self.assertEqual(effective_entity["mention_policy"], "standalone")
        self.assertEqual(
            effective_entity["aliases"],
            [
                {
                    "value": "campaign",
                    "match_policy": "requires_target_attribution",
                }
            ],
        )
        applied_guidance = annotate_mock.await_args.kwargs[
            "research_guidance"
        ]
        self.assertIn("Campaign 360", applied_guidance)
        self.assertIn("campaign", applied_guidance)
        self.assertNotEqual(applied_guidance, guidance)
        self.assertNotEqual(
            _annotation_context_sha256(PROFILE, CATALOG),
            _annotation_context_sha256(
                PROFILE,
                catalog,
                applied_guidance,
            ),
        )

        policy_write = artifact_mock.await_args.kwargs
        self.assertEqual(
            policy_write["artifact_key"],
            "analysis_critic_policy",
        )
        self.assertEqual(
            policy_write["output_json"]["effective_catalog"],
            catalog,
        )
        second_payload = critic_mock.await_args_list[1].kwargs["payload"]
        self.assertEqual(
            second_payload["candidate_metrics"],
            revised_metrics,
        )
        self.assertEqual(
            len(second_payload["previous_policy_changes"]),
            1,
        )

    async def test_second_revise_blocks_without_a_second_repair(self) -> None:
        adjustment = {
            "action": "require_alias_attribution",
            "entity_name": "Campaign 360",
            "alias": "campaign",
            "reason": "Нужна явная атрибуция.",
            "answer_ids": [11],
        }
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review(
                        "revise",
                        adjustments=[adjustment],
                    ),
                    _critic_review("revise"),
                ],
            ) as critic_mock,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
            ) as gate_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as artifact_mock,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate_mock,
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
                return_value=ROWS,
            ),
            patch(
                "app.services.analyzer._compute_metrics",
                return_value=METRICS,
            ) as metrics_mock,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "blocked report publication",
            ):
                await _run_analysis_critic_loop(
                    "run-second-revise",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )

        self.assertEqual(critic_mock.await_count, MAX_CRITIC_ITERATIONS)
        annotate_mock.assert_awaited_once()
        metrics_mock.assert_called_once()
        artifact_mock.assert_awaited_once()
        gate_mock.assert_awaited_once()
        self.assertFalse(gate_mock.await_args.kwargs["passed"])
        self.assertEqual(
            gate_mock.await_args.kwargs["iteration"],
            MAX_CRITIC_ITERATIONS,
        )

    async def test_r2_exhaustion_uses_fable_targeted_repair_then_final_gate(
        self,
    ) -> None:
        completed_rows = []
        for answer_id in range(1, 76):
            answer_text = f"Готовый ответ {answer_id} про Campaign 360."
            row = copy.deepcopy(ROWS[0])
            row["answer_id"] = answer_id
            row["answer_text"] = answer_text
            row["annotation"]["_answer_sha256"] = hashlib.sha256(
                answer_text.encode("utf-8")
            ).hexdigest()
            completed_rows.append(row)
        failed_rows = [
            {
                **copy.deepcopy(ROWS[0]),
                "answer_id": answer_id,
                "status": "failed",
                "answer_text": "",
                "annotation_state": "missing",
                "annotation": {},
            }
            for answer_id in range(76, 82)
        ]
        panel_rows = [*completed_rows, *failed_rows]
        adjustment = {
            "action": "require_literal_attribution_evidence",
            "entity_name": "Campaign 360",
            "alias": None,
            "reason": "Нужен полный буквальный блок владельца и услуги.",
            "answer_ids": [11],
        }
        anomaly = {
            "code": "annotation_evidence_mismatch",
            "severity": "important",
            "finding": "Короткий evidence потерял Markdown-владельца.",
            "answer_ids": [11],
            "entities": ["Campaign 360"],
        }
        recovered_rows = copy.deepcopy(panel_rows)
        recovered_rows[10]["annotation"]["entity_mentions"] = [
            {
                **recovered_rows[10]["annotation"]["entity_mentions"][0],
                "attributed_to_target": True,
                "evidence": (
                    "Example предлагает Campaign 360 для управления "
                    "кампаниями."
                ),
            }
        ]
        recovered_metrics = {
            "portfolio_visibility": {
                "web": {"mention_count": 1, "valid_answers": 1}
            },
            "providers": [{"provider_key": "openai"}],
        }
        required_checks = {
            CHECK_RAW_CORPUS_UNCHANGED,
            CHECK_DERIVED_METRICS_RECOMPUTED,
            CHECK_CRITIC_GATE_PASSED,
        }
        plan = SimpleNamespace(
            epoch=2,
            plan_digest="f" * 64,
            decision={
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "rationale": (
                    "Нужно повторно разметить только строку с потерянным "
                    "Markdown-контекстом."
                ),
                "guidance": (
                    "Скопируй один непрерывный raw-блок от заголовка "
                    "владельца до дочерней услуги."
                ),
                "target_answer_ids": [11],
                "invalidate_artifact_keys": [],
                "acceptance_checks": sorted(required_checks),
            },
        )
        gate = {"passed": True, "iteration": 3}
        with (
            patch(
                "app.services.analyzer.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review(
                        "revise",
                        adjustments=[adjustment],
                        guidance="Проверь буквальную атрибуцию.",
                    ),
                    _critic_review(
                        "revise",
                        adjustments=[adjustment],
                        anomalies=[anomaly],
                    ),
                    _critic_review("pass"),
                ],
            ) as critic_mock,
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ) as planner,
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ) as mark,
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value=gate,
            ) as gate_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate,
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
                side_effect=[panel_rows, recovered_rows],
            ) as metric_rows,
            patch(
                "app.services.analyzer._compute_metrics",
                side_effect=[METRICS, recovered_metrics],
            ),
        ):
            catalog, rows, metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-fable-recovery",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=panel_rows,
                    metrics=METRICS,
                )
            )

        self.assertEqual(catalog["entities"][0]["canonical_name"], "Campaign 360")
        self.assertEqual(rows, recovered_rows)
        self.assertEqual(metrics, recovered_metrics)
        self.assertEqual(returned_gate, gate)
        self.assertEqual(critic_mock.await_count, 3)
        final_call = critic_mock.await_args_list[2]
        self.assertEqual(final_call.kwargs["iteration"], 3)
        self.assertTrue(final_call.kwargs["recovery_final"])
        final_payload = final_call.kwargs["payload"]
        targeted_answer = next(
            item
            for item in final_payload["answers"]
            if item["answer_id"] == 11
        )
        self.assertTrue(targeted_answer["raw_answer_included"])
        self.assertFalse(targeted_answer["raw_answer_truncated"])
        self.assertEqual(
            targeted_answer["raw_answer"],
            recovered_rows[10]["answer_text"],
        )
        self.assertEqual(
            final_payload["raw_evidence_selection"][
                "mandatory_answer_ids"
            ],
            [11],
        )
        planner_kwargs = planner.await_args.kwargs
        self.assertEqual(
            planner_kwargs["allowed_actions"],
            {ACTION_TARGETED_ANNOTATION_REPAIR, ACTION_STOP},
        )
        self.assertEqual(planner_kwargs["permitted_answer_ids"], {11})
        executor_contract = planner_kwargs["facts"]["executor_contract"]
        self.assertEqual(
            set(
                executor_contract[ACTION_TARGETED_ANNOTATION_REPAIR][
                    "acceptance_checks"
                ]
            ),
            required_checks,
        )
        mark.assert_awaited_once_with(plan, stage_execution_limit=1)
        targeted_calls = [
            call
            for call in annotate.await_args_list
            if call.kwargs.get("target_answer_ids") is not None
        ]
        self.assertEqual(len(targeted_calls), 1)
        self.assertEqual(targeted_calls[0].kwargs["target_answer_ids"], {11})
        self.assertEqual(
            targeted_calls[0].kwargs["repair_mode"],
            ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
        )
        provenance = targeted_calls[0].kwargs[
            "annotation_repair_provenance"
        ]
        self.assertEqual(provenance["orchestrator_epoch"], 2)
        self.assertEqual(provenance["target_answer_ids"], [11])
        self.assertEqual(metric_rows.await_count, 2)
        self.assertIn(
            11,
            metric_rows.await_args_list[1].kwargs[
                "annotation_input_sha256_by_answer_id"
            ],
        )
        annotation_digests = metric_rows.await_args_list[1].kwargs[
            "annotation_input_sha256_by_answer_id"
        ]
        self.assertEqual(len(annotation_digests), 75)
        self.assertTrue(
            set(range(76, 82)).isdisjoint(annotation_digests)
        )
        gate_mock.assert_awaited_once()
        self.assertTrue(gate_mock.await_args.kwargs["passed"])
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            set(
                finish.await_args.kwargs["details"][
                    "executed_acceptance_checks"
                ]
            ),
            required_checks,
        )

    async def test_r2_targeted_repair_fails_closed_when_final_critic_blocks(
        self,
    ) -> None:
        adjustment = {
            "action": "require_literal_attribution_evidence",
            "entity_name": "Campaign 360",
            "alias": None,
            "reason": "Нужен полный буквальный фрагмент.",
            "answer_ids": [11],
        }
        required_checks = {
            CHECK_RAW_CORPUS_UNCHANGED,
            CHECK_DERIVED_METRICS_RECOMPUTED,
            CHECK_CRITIC_GATE_PASSED,
        }
        plan = SimpleNamespace(
            epoch=1,
            plan_digest="e" * 64,
            decision={
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "guidance": "Проверь непрерывный Markdown-блок.",
                "target_answer_ids": [11],
                "acceptance_checks": sorted(required_checks),
            },
        )
        recovered_rows = [
            {
                **ROWS[0],
                "annotation": {
                    **ROWS[0]["annotation"],
                    "target_mentioned": True,
                },
            }
        ]
        with (
            patch(
                "app.services.analyzer.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review("revise", adjustments=[adjustment]),
                    _critic_review("revise", adjustments=[adjustment]),
                    _critic_review("block"),
                ],
            ) as critic,
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
            ) as gate,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
                side_effect=[ROWS, recovered_rows],
            ),
            patch(
                "app.services.analyzer._compute_metrics",
                side_effect=[METRICS, {**METRICS, "recovered": True}],
            ),
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "blocked report publication",
            ):
                await _run_analysis_critic_loop(
                    "run-fable-final-block",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )

        self.assertEqual(critic.await_count, 3)
        finish.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["succeeded"])
        gate.assert_awaited_once()
        self.assertFalse(gate.await_args.kwargs["passed"])
        self.assertEqual(gate.await_args.kwargs["rows"], recovered_rows)
        self.assertTrue(gate.await_args.kwargs["metrics"]["recovered"])

    async def test_r2_fable_stop_preserves_checkpoint_without_repair(self) -> None:
        adjustment = {
            "action": "require_literal_attribution_evidence",
            "entity_name": "Campaign 360",
            "alias": None,
            "reason": "Нужен полный буквальный фрагмент.",
            "answer_ids": [11],
        }
        plan = SimpleNamespace(
            epoch=1,
            plan_digest="d" * 64,
            decision={
                "action": ACTION_STOP,
                "rationale": (
                    "Показанного evidence недостаточно для безопасной правки."
                ),
                "guidance": "",
                "target_answer_ids": [],
                "acceptance_checks": [CHECK_CHECKPOINT_PRESERVED],
            },
        )
        with (
            patch(
                "app.services.analyzer.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review("revise", adjustments=[adjustment]),
                    _critic_review("revise", adjustments=[adjustment]),
                ],
            ) as critic,
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new=AsyncMock(return_value=plan),
            ),
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.finish_recovery",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate,
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
                return_value=ROWS,
            ),
            patch(
                "app.services.analyzer._compute_metrics",
                return_value=METRICS,
            ),
        ):
            with self.assertRaises(OpenRouterError):
                await _run_analysis_critic_loop(
                    "run-fable-stop",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )

        self.assertEqual(critic.await_count, 2)
        self.assertEqual(
            sum(
                call.kwargs.get("target_answer_ids") is not None
                for call in annotate.await_args_list
            ),
            0,
        )
        finish.assert_awaited_once()
        self.assertFalse(finish.await_args.kwargs["succeeded"])
        self.assertEqual(
            finish.await_args.kwargs["before_digest"],
            finish.await_args.kwargs["after_digest"],
        )

    def test_targeted_repair_provenance_survives_standard_resume(self) -> None:
        resume_digest = _annotation_context_sha256(PROFILE, CATALOG)
        repair_digest = _annotation_context_sha256(
            PROFILE,
            CATALOG,
            "Скопируй непрерывный Markdown-блок.",
            repair_mode=ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
        )
        annotation = {
            **ROWS[0]["annotation"],
            "_annotation_input_sha256": repair_digest,
            "_annotation_repair_provenance": {
                "version": "analysis-critic-targeted-repair-v1",
                "repair_annotation_input_sha256": repair_digest,
                "resume_annotation_input_sha256": resume_digest,
                "orchestrator_epoch": 2,
            },
        }

        self.assertTrue(
            _annotation_matches_answer(
                annotation,
                answer_text=ROWS[0]["answer_text"],
                answer_model=ROWS[0]["model"],
                annotation_input_sha256=resume_digest,
            )
        )
        self.assertFalse(
            _annotation_matches_answer(
                annotation,
                answer_text=ROWS[0]["answer_text"],
                answer_model=ROWS[0]["model"],
                annotation_input_sha256=_annotation_context_sha256(
                    {**PROFILE, "brand_name": "Changed"},
                    CATALOG,
                ),
            )
        )

    def test_annotation_provenance_ignores_six_failed_panel_cells(
        self,
    ) -> None:
        completed = [
            {
                "answer_id": answer_id,
                "status": "completed",
                "answer_text": f"Готовый ответ {answer_id}",
                "annotation_state": "current",
                "annotation": {
                    "_annotation_input_sha256": f"digest-{answer_id}",
                },
            }
            for answer_id in range(1, 76)
        ]
        failed = [
            {
                "answer_id": answer_id,
                "status": "failed",
                "answer_text": "",
                "annotation_state": "missing",
                "annotation": {},
            }
            for answer_id in range(76, 82)
        ]

        digests = _current_annotation_input_digests([*completed, *failed])

        self.assertEqual(len(digests), 75)
        self.assertEqual(digests[1], "digest-1")
        self.assertEqual(digests[75], "digest-75")
        self.assertTrue(set(range(76, 82)).isdisjoint(digests))

    def test_final_critic_includes_late_same_bucket_target_raw(self) -> None:
        rows = []
        for answer_id in range(1, 76):
            row = copy.deepcopy(ROWS[0])
            row["answer_id"] = answer_id
            row["answer_text"] = (
                f"Полный точный ответ {answer_id} для одной и той же страты."
            )
            rows.append(row)

        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=rows,
            metrics=METRICS,
            policy_history=[],
            mandatory_raw_answer_ids={74, 75},
        )

        by_id = {
            int(item["answer_id"]): item
            for item in payload["answers"]
        }
        for answer_id in (74, 75):
            with self.subTest(answer_id=answer_id):
                self.assertTrue(by_id[answer_id]["raw_answer_included"])
                self.assertFalse(by_id[answer_id]["raw_answer_truncated"])
                self.assertEqual(
                    by_id[answer_id]["raw_answer"],
                    rows[answer_id - 1]["answer_text"],
                )
        self.assertEqual(
            payload["raw_evidence_selection"]["mandatory_answer_ids"],
            [74, 75],
        )

    def test_production_makarska_markdown_cards_stay_contiguous(
        self,
    ) -> None:
        target_aliases = ["Makarska Tattoo & Piercing Studio"]
        cases = {
            569: (
                "1. **Makarska Tattoo & Piercing Studio**\n"
                "   * **Особенности:** специализируется на кавер-апах и "
                "пирсинге.",
                ["пирсинг"],
            ),
            571: (
                "### ✅ Makarska Tattoo & Piercing Studio\n"
                "- **Профиль:** татуировки и пирсинг, индивидуальные "
                "эскизы.",
                ["пирсинг"],
            ),
            584: (
                "1. **Makarska Tattoo & Piercing Studio**\n"
                "   * **Локация:** центр города.\n"
                "   * **Особенности:** специализация на *cover-up*, "
                "*rework* и пирсинге.",
                ["cover-up", "rework"],
            ),
        }

        for answer_id, (raw, aliases) in cases.items():
            with self.subTest(answer_id=answer_id):
                self.assertEqual(
                    _literal_target_attribution_evidence(
                        raw,
                        aliases,
                        target_aliases,
                    ),
                    raw,
                )

    def test_markdown_card_does_not_cross_competitor_peer_boundary(
        self,
    ) -> None:
        raw = (
            "1. **Makarska Tattoo & Piercing Studio**\n"
            "   * **Локация:** центр города.\n"
            "2. **Competitor Tattoo**\n"
            "   * **Особенности:** пирсинг и cover-up."
        )

        self.assertEqual(
            _literal_target_attribution_evidence(
                raw,
                ["пирсинг", "cover-up"],
                ["Makarska Tattoo & Piercing Studio"],
            ),
            "",
        )

    async def test_targeted_recovery_refuses_raw_beyond_critic_limit(
        self,
    ) -> None:
        adjustment = {
            "action": "require_literal_attribution_evidence",
            "entity_name": "Campaign 360",
            "alias": None,
            "reason": "Нужен полный буквальный фрагмент.",
            "answer_ids": [11],
        }
        oversized_rows = [
            {
                **ROWS[0],
                "answer_text": "x" * 24_001,
            }
        ]
        with patch(
            "app.services.analyzer.plan_durable_recovery",
            new_callable=AsyncMock,
        ) as planner:
            with self.assertRaisesRegex(
                OpenRouterError,
                "would truncate issue raw",
            ):
                await _recover_analysis_critic_exhaustion(
                    "run-oversized-issue",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=oversized_rows,
                    metrics=METRICS,
                    review=_critic_review(
                        "revise",
                        adjustments=[adjustment],
                    ),
                    policy_history=[],
                    accumulated_guidance=[],
                    valid_answer_ids={11},
                    resume_annotation_input_sha256=(
                        _annotation_context_sha256(PROFILE, CATALOG)
                    ),
                    expected_corpus_cells=None,
                )
        planner.assert_not_awaited()

    async def test_r2_block_never_invokes_fable_recovery(self) -> None:
        adjustment = {
            "action": "require_literal_attribution_evidence",
            "entity_name": "Campaign 360",
            "alias": None,
            "reason": "Нужен полный буквальный фрагмент.",
            "answer_ids": [11],
        }
        with (
            patch(
                "app.services.analyzer.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review("revise", adjustments=[adjustment]),
                    _critic_review("block", adjustments=[adjustment]),
                ],
            ),
            patch(
                "app.services.analyzer.plan_durable_recovery",
                new_callable=AsyncMock,
            ) as planner,
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._metric_rows",
                new_callable=AsyncMock,
                return_value=ROWS,
            ),
            patch(
                "app.services.analyzer._compute_metrics",
                return_value=METRICS,
            ),
        ):
            with self.assertRaises(OpenRouterError):
                await _run_analysis_critic_loop(
                    "run-r2-block",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
        planner.assert_not_awaited()

    def test_policy_application_can_only_narrow_known_entities(self) -> None:
        catalog = {
            "entities": [
                {
                    "canonical_name": "Garpun",
                    "aliases": ["Гарпун"],
                    "category": "target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
                {
                    "canonical_name": "Campaign 360",
                    "aliases": ["campaign", "Campaign360"],
                    "category": "target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
                {
                    "canonical_name": "Unsupported Unit",
                    "aliases": [],
                    "category": "target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
            ]
        }
        review = _critic_review(
            "revise",
            adjustments=[
                {
                    "action": "require_target_attribution",
                    "entity_name": "Garpun",
                    "alias": None,
                    "reason": "Нужно подтверждение связи.",
                    "answer_ids": [11],
                },
                {
                    "action": "require_alias_attribution",
                    "entity_name": "Campaign 360",
                    "alias": "campaign",
                    "reason": "Общий alias.",
                    "answer_ids": [11],
                },
                {
                    "action": "exclude_portfolio_entity",
                    "entity_name": "Unsupported Unit",
                    "alias": None,
                    "reason": "Сайт не подтверждает принадлежность.",
                    "answer_ids": [11],
                },
                {
                    "action": "include_portfolio_entity",
                    "entity_name": "New Product",
                    "alias": None,
                    "reason": "Попытка расширить scope.",
                    "answer_ids": [11],
                },
                {
                    "action": "exclude_portfolio_entity",
                    "entity_name": "Campaign 360",
                    "alias": None,
                    "reason": "Неизвестный answer_id.",
                    "answer_ids": [999],
                },
            ],
            guidance="Учитывай только буквальные доказательства.",
        )

        tightened, applied, guidance = _apply_critic_policy(
            catalog,
            review,
            valid_answer_ids={11},
        )

        self.assertEqual(catalog["entities"][0]["mention_policy"], "standalone")
        self.assertTrue(catalog["entities"][2]["commercially_relevant"])
        self.assertEqual(len(applied), 3)
        self.assertEqual(
            {item["action"] for item in applied},
            {
                "require_target_attribution",
                "require_alias_attribution",
                "exclude_portfolio_entity",
            },
        )
        garpun, campaign, unsupported = tightened["entities"]
        self.assertEqual(
            garpun["mention_policy"],
            "requires_target_attribution",
        )
        self.assertEqual(
            garpun["aliases"],
            [
                {
                    "value": "Гарпун",
                    "match_policy": "requires_target_attribution",
                }
            ],
        )
        self.assertEqual(campaign["mention_policy"], "standalone")
        self.assertEqual(
            campaign["aliases"],
            [
                {
                    "value": "campaign",
                    "match_policy": "requires_target_attribution",
                },
                "Campaign360",
            ],
        )
        self.assertFalse(unsupported["commercially_relevant"])
        self.assertTrue(unsupported["_critic_excluded"])
        self.assertNotIn("New Product", json.dumps(tightened))
        self.assertIn("Garpun", guidance)
        self.assertIn("Campaign 360", guidance)
        self.assertIn("Unsupported Unit", guidance)
        self.assertNotIn("Учитывай только буквальные доказательства", guidance)

    def test_policy_targets_profile_owned_record_on_canonical_collision(
        self,
    ) -> None:
        catalog = {
            "entities": [
                {
                    "canonical_name": "Orbit Cloud",
                    "aliases": ["orbitcloud.io"],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "_profile_membership_confirmed": True,
                },
                {
                    "canonical_name": "Orbit Cloud",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
            ]
        }
        review = _critic_review(
            "revise",
            adjustments=[
                {
                    "action": "require_target_attribution",
                    "entity_name": "Orbit Cloud",
                    "alias": None,
                    "reason": "Проверить явную связь.",
                    "answer_ids": [11],
                }
            ],
            guidance="Проверить связь.",
        )

        tightened, applied, _guidance = _apply_critic_policy(
            catalog,
            review,
            valid_answer_ids={11},
        )

        self.assertEqual(len(tightened["entities"]), 1)
        entity = tightened["entities"][0]
        self.assertEqual(entity["category"], "target")
        self.assertTrue(entity["_profile_membership_confirmed"])
        self.assertEqual(
            entity["mention_policy"],
            "requires_target_attribution",
        )
        self.assertEqual(len(applied), 1)

    def test_pass_with_unresolved_work_is_rejected(self) -> None:
        inconsistent = _critic_review(
            "pass",
            adjustments=[
                {
                    "action": "require_target_attribution",
                    "entity_name": "Campaign 360",
                    "alias": None,
                    "reason": "Нужна правка.",
                    "answer_ids": [11],
                }
            ],
        )
        inconsistent["anomalies"] = [
            {
                "code": "scope_leakage",
                "severity": "critical",
                "finding": "Scope смешан.",
                "answer_ids": [11],
                "entities": ["Campaign 360"],
            }
        ]

        errors = _critic_review_errors(inconsistent)

        self.assertTrue(
            any("unresolved critical" in error for error in errors)
        )
        self.assertTrue(
            any("pending policy adjustments" in error for error in errors)
        )


class AnalysisCriticRecoveryPersistenceTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "critic-recovery.sqlite3"
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
        self._analyzer_session_patch = patch(
            "app.services.analyzer.SessionLocal",
            self.SessionLocal,
        )
        self._analyzer_session_patch.start()
        self._lease_session_patch = patch(
            "app.services.run_lease.SessionLocal",
            self.SessionLocal,
        )
        self._lease_session_patch.start()
        self._recovery_session_patch = patch(
            "app.services.recovery_state.SessionLocal",
            self.SessionLocal,
        )
        self._recovery_session_patch.start()

        self.run_id = str(uuid.uuid4())
        self.raw = "### Realweb\n- Предлагает programmatic DOOH."
        self.original_annotation = {
            "answer_id": 1,
            "valid": True,
            "_annotation_version": ANNOTATION_VERSION,
            "_answer_sha256": hashlib.sha256(
                self.raw.encode("utf-8")
            ).hexdigest(),
            "_answer_model": "provider/model",
            "_annotation_input_sha256": "base-context",
        }
        async with self.SessionLocal() as session:
            run = Run(
                id=self.run_id,
                domain="example.com",
                status=RunStatus.analyzing,
                config_json={},
                execution_slot=1,
                lease_owner="current-owner",
            )
            prompt = VisibilityPrompt(
                run_id=self.run_id,
                prompt_key="intent-i",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие агентства предлагают DOOH?",
                sequence=1,
            )
            session.add_all((run, prompt))
            await session.flush()
            answer = ModelAnswer(
                run_id=self.run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="provider/model",
                mode="web",
                status="completed",
                response_text=self.raw,
            )
            session.add(answer)
            await session.flush()
            self.answer_id = answer.id
            self.original_annotation["answer_id"] = self.answer_id
            session.add(
                AnswerAnnotation(
                    answer_id=self.answer_id,
                    annotation_json=self.original_annotation,
                )
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        self._recovery_session_patch.stop()
        self._lease_session_patch.stop()
        self._analyzer_session_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    def _pending_snapshot(self) -> dict:
        return {
            "answer_id": self.answer_id,
            "answer_model": "provider/model",
            "answer_sha256": hashlib.sha256(
                self.raw.encode("utf-8")
            ).hexdigest(),
            "_prior_annotation_sha256": stable_digest(
                self.original_annotation
            ),
        }

    def _recovered_fixture(
        self,
        *,
        epoch: int = 1,
    ) -> tuple[
        dict,
        list[dict],
        dict,
        dict,
        str,
        str,
    ]:
        scoped_catalog = _scope_entity_catalog_to_profile(CATALOG, PROFILE)
        plan = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "rationale": "Исправить только подтверждённую строку.",
            "target_answer_ids": [11],
            "guidance": "Вернуть точный непрерывный Markdown-блок.",
            "acceptance_checks": sorted(
                {
                    CHECK_RAW_CORPUS_UNCHANGED,
                    CHECK_DERIVED_METRICS_RECOMPUTED,
                    CHECK_CRITIC_GATE_PASSED,
                }
            ),
        }
        plan_digest = stable_digest(plan)
        recovery_step = {
            "iteration": MAX_CRITIC_ITERATIONS + 1,
            "kind": "orchestrated_targeted_annotation_repair",
            "orchestrator_epoch": epoch,
            "target_answer_ids": [11],
            "critic_adjustments": [],
            "annotation_guidance": plan["guidance"],
            "raw_corpus_sha256": "",
        }
        repaired_rows = copy.deepcopy(ROWS)
        recovery_step["raw_corpus_sha256"] = _raw_corpus_digest(
            repaired_rows
        )
        repaired_rows[0]["annotation"][
            "_annotation_repair_provenance"
        ] = {
            "version": "analysis-critic-targeted-repair-v1",
            "orchestrator_epoch": epoch,
            "orchestrator_plan_digest": plan_digest,
            "target_answer_ids": [11],
            "recovery_policy_step": copy.deepcopy(recovery_step),
        }
        repaired_rows[0]["annotation"]["uncertainties"] = [
            "targeted repair completed"
        ]
        state_digest = _critic_analysis_state_digest(
            repaired_rows,
            METRICS,
        )
        before_digest = stable_digest({"pre_repair": state_digest})
        return (
            scoped_catalog,
            repaired_rows,
            plan,
            recovery_step,
            before_digest,
            state_digest,
        )

    async def _insert_recovery_epoch(
        self,
        *,
        status: str,
        plan: dict,
        before_digest: str,
        state_digest: str,
        rows: list[dict],
        gate: dict | None = None,
    ) -> None:
        outcome = {
            "execution_attempts": 1,
            "max_execution_attempts": 2,
            "stage_execution_attempts": 1,
            "stage_execution_limit": 1,
        }
        if status == "succeeded":
            assert gate is not None
            outcome = {
                "succeeded": True,
                "before_digest": before_digest,
                "after_digest": state_digest,
                "details": {
                    "successful_analysis_state_digest": state_digest,
                    "successful_gate_sha256": stable_digest(gate),
                },
            }
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    failure_fingerprint="f" * 64,
                    facts_digest="a" * 64,
                    status=status,
                    input_json={
                        "incident": {
                            "facts": {
                                "analysis_state_sha256": before_digest,
                                "raw_corpus_sha256": _raw_corpus_digest(rows),
                                "prior_policy_history": [],
                            }
                        }
                    },
                    plan_json=copy.deepcopy(plan),
                    plan_digest=stable_digest(plan),
                    outcome_json=outcome,
                )
            )
            await session.commit()

    async def _insert_completed_r3(
        self,
        *,
        catalog: dict,
        rows: list[dict],
        recovery_step: dict,
        epoch: int = 1,
    ) -> dict:
        payload = _critic_payload(
            profile=PROFILE,
            catalog=catalog,
            rows=rows,
            metrics=METRICS,
            policy_history=[recovery_step],
            mandatory_raw_answer_ids={11},
        )
        payload["orchestrated_recovery"] = {
            "epoch": epoch,
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "target_answer_ids": [11],
            "raw_corpus_sha256": _raw_corpus_digest(rows),
            "required_acceptance_checks": sorted(
                {
                    CHECK_RAW_CORPUS_UNCHANGED,
                    CHECK_DERIVED_METRICS_RECOMPUTED,
                    CHECK_CRITIC_GATE_PASSED,
                }
            ),
        }
        review = _critic_review("pass")
        async with self.SessionLocal() as session:
            session.add(
                RunArtifact(
                    run_id=self.run_id,
                    stage_key="knowledge_gap",
                    artifact_key=(
                        "analysis_critic_r"
                        + str(
                            MAX_CRITIC_ITERATIONS
                            + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                        )
                    ),
                    status="completed",
                    model=CRITIC_MODEL,
                    prompt_version=ANALYSIS_CRITIC_VERSION,
                    input_json=payload,
                    output_json=review,
                    usage_json={
                        "_aiv_critic_contract": {
                            "semantic_verdict_status": "validated",
                        }
                    },
                )
            )
            await session.commit()
        return review

    async def test_targeted_annotation_cas_writes_nothing_when_prior_changed(
        self,
    ) -> None:
        concurrent_annotation = {
            **self.original_annotation,
            "uncertainties": ["concurrent-worker"],
        }
        async with self.SessionLocal() as session:
            stored = (
                await session.execute(
                    select(AnswerAnnotation).where(
                        AnswerAnnotation.answer_id == self.answer_id
                    )
                )
            ).scalar_one()
            stored.annotation_json = concurrent_annotation
            await session.commit()

        with bind_run_lease(self.run_id, "current-owner"):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "CAS input changed",
            ):
                await _save_targeted_recovery_annotations(
                    self.run_id,
                    [
                        {
                            **self.original_annotation,
                            "uncertainties": ["targeted-repair"],
                        }
                    ],
                    {self.answer_id: self._pending_snapshot()},
                )

        async with self.SessionLocal() as session:
            persisted = (
                await session.execute(
                    select(AnswerAnnotation.annotation_json).where(
                        AnswerAnnotation.answer_id == self.answer_id
                    )
                )
            ).scalar_one()
        self.assertEqual(persisted, concurrent_annotation)

    async def test_stale_lease_cannot_publish_targeted_annotation(self) -> None:
        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(lease_owner="replacement-owner")
            )
            await session.commit()

        with bind_run_lease(self.run_id, "current-owner"):
            with self.assertRaises(RunLeaseLostError):
                await _save_targeted_recovery_annotations(
                    self.run_id,
                    [
                        {
                            **self.original_annotation,
                            "uncertainties": ["targeted-repair"],
                        }
                    ],
                    {self.answer_id: self._pending_snapshot()},
                )

        async with self.SessionLocal() as session:
            persisted = (
                await session.execute(
                    select(AnswerAnnotation.annotation_json).where(
                        AnswerAnnotation.answer_id == self.answer_id
                    )
                )
            ).scalar_one()
        self.assertEqual(persisted, self.original_annotation)

    async def test_succeeded_recovery_gate_is_reused_before_r1(self) -> None:
        (
            scoped_catalog,
            repaired_rows,
            plan,
            recovery_step,
            before_digest,
            state_digest,
        ) = self._recovered_fixture()
        await self._insert_completed_r3(
            catalog=scoped_catalog,
            rows=repaired_rows,
            recovery_step=recovery_step,
        )
        with bind_run_lease(self.run_id, "current-owner"):
            gate = await _save_critic_gate(
                self.run_id,
                passed=True,
                iteration=(
                    MAX_CRITIC_ITERATIONS
                    + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                ),
                profile=PROFILE,
                catalog=scoped_catalog,
                rows=repaired_rows,
                metrics=METRICS,
                policy_history=[recovery_step],
                reason="Primary r3 passed.",
            )
        await self._insert_recovery_epoch(
            status="succeeded",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
            gate=gate,
        )

        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
            ) as critic,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
            bind_run_lease(self.run_id, "current-owner"),
        ):
            _catalog, _rows, _metrics, reused = (
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )
            )

        self.assertTrue(reused["passed"])
        self.assertEqual(reused["provenance"], gate["provenance"])
        self.assertEqual(
            reused["corpus_manifest"],
            gate["corpus_manifest"],
        )
        critic.assert_not_awaited()
        progress.assert_not_awaited()

    async def test_crash_after_gate_commit_finalizes_without_provider_call(
        self,
    ) -> None:
        (
            scoped_catalog,
            repaired_rows,
            plan,
            recovery_step,
            before_digest,
            state_digest,
        ) = self._recovered_fixture()
        await self._insert_completed_r3(
            catalog=scoped_catalog,
            rows=repaired_rows,
            recovery_step=recovery_step,
        )
        with bind_run_lease(self.run_id, "current-owner"):
            gate = await _save_critic_gate(
                self.run_id,
                passed=True,
                iteration=(
                    MAX_CRITIC_ITERATIONS
                    + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                ),
                profile=PROFILE,
                catalog=scoped_catalog,
                rows=repaired_rows,
                metrics=METRICS,
                policy_history=[recovery_step],
                reason="Primary r3 passed before process crash.",
            )
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
        )

        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
            ) as critic,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
            bind_run_lease(self.run_id, "current-owner"),
        ):
            _catalog, _rows, _metrics, reused = (
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )
            )

        self.assertTrue(reused["passed"])
        self.assertEqual(reused["provenance"], gate["provenance"])
        self.assertEqual(
            reused["corpus_manifest"],
            gate["corpus_manifest"],
        )
        critic.assert_not_awaited()
        progress.assert_not_awaited()
        async with self.SessionLocal() as session:
            status = (
                await session.execute(
                    select(RecoveryEpoch.status).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
        self.assertEqual(status, "succeeded")

    async def test_crash_before_r3_reservation_uses_one_final_call_only(
        self,
    ) -> None:
        (
            _scoped_catalog,
            repaired_rows,
            plan,
            _recovery_step,
            before_digest,
            state_digest,
        ) = self._recovered_fixture()
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
        )
        async with self.SessionLocal() as session:
            session.add_all(
                [
                    RunArtifact(
                    run_id=self.run_id,
                    stage_key="knowledge_gap",
                    artifact_key="analysis_critic_gate",
                    status="failed",
                    model=CRITIC_MODEL,
                    prompt_version="stale-analysis-critic-version",
                    input_json={"iteration": MAX_CRITIC_ITERATIONS},
                    output_json={
                        "passed": False,
                        "iteration": MAX_CRITIC_ITERATIONS,
                    },
                    error_message="Previous r2 gate failed.",
                    ),
                    RunArtifact(
                        run_id=self.run_id,
                        stage_key="knowledge_gap",
                        artifact_key=(
                            "analysis_critic_r"
                            + str(
                                MAX_CRITIC_ITERATIONS
                                + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                            )
                        ),
                        status="completed",
                        model="old/provider-model",
                        prompt_version="stale-analysis-critic-version",
                        input_json={
                            "orchestrated_recovery": {"epoch": 999}
                        },
                        output_json=_critic_review("pass"),
                    ),
                ]
            )
            await session.commit()
        primary_pass = _critic_review("pass")

        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                return_value=primary_pass,
            ) as critic,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
            bind_run_lease(self.run_id, "current-owner"),
        ):
            _catalog, _rows, _metrics, gate = (
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )
            )

        self.assertTrue(gate["passed"])
        critic.assert_awaited_once()
        self.assertTrue(critic.await_args.kwargs["recovery_final"])
        self.assertEqual(
            critic.await_args.kwargs["iteration"],
            MAX_CRITIC_ITERATIONS + MAX_CRITIC_RECOVERY_FINAL_REVIEWS,
        )
        progress.assert_not_awaited()

    async def test_crash_after_completed_r3_reuses_primary_without_call(
        self,
    ) -> None:
        (
            scoped_catalog,
            repaired_rows,
            plan,
            recovery_step,
            before_digest,
            state_digest,
        ) = self._recovered_fixture()
        await self._insert_completed_r3(
            catalog=scoped_catalog,
            rows=repaired_rows,
            recovery_step=recovery_step,
        )
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
        )

        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
            ) as critic,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
            bind_run_lease(self.run_id, "current-owner"),
        ):
            _catalog, _rows, _metrics, gate = (
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )
            )

        self.assertTrue(gate["passed"])
        critic.assert_not_awaited()
        progress.assert_not_awaited()
        async with self.SessionLocal() as session:
            status = (
                await session.execute(
                    select(RecoveryEpoch.status).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
        self.assertEqual(status, "succeeded")

    async def test_crash_during_r3_reservation_fails_closed_without_call(
        self,
    ) -> None:
        (
            _scoped_catalog,
            repaired_rows,
            plan,
            _recovery_step,
            before_digest,
            state_digest,
        ) = self._recovered_fixture()
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
        )
        async with self.SessionLocal() as session:
            session.add(
                RunArtifact(
                    run_id=self.run_id,
                    stage_key="knowledge_gap",
                    artifact_key=(
                        "analysis_critic_r"
                        + str(
                            MAX_CRITIC_ITERATIONS
                            + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                        )
                    ),
                    status="running",
                    model=CRITIC_MODEL,
                    prompt_version=ANALYSIS_CRITIC_VERSION,
                    input_json={"reserved": True},
                )
            )
            await session.commit()

        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
            ) as critic,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
            bind_run_lease(self.run_id, "current-owner"),
        ):
            with self.assertRaises(_AnalysisCriticRecoveryBlocked):
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )

        critic.assert_not_awaited()
        progress.assert_not_awaited()
        async with self.SessionLocal() as session:
            epoch_status = (
                await session.execute(
                    select(RecoveryEpoch.status).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
            gate_status = (
                await session.execute(
                    select(RunArtifact.status).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "analysis_critic_gate",
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch_status, "failed")
        self.assertEqual(gate_status, "failed")

    async def test_terminal_post_repair_state_blocks_resume_before_r1(
        self,
    ) -> None:
        state_digest = _critic_analysis_state_digest(ROWS, METRICS)
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    failure_fingerprint="f" * 64,
                    facts_digest="a" * 64,
                    status="failed",
                    outcome_json={
                        "succeeded": False,
                        "details": {
                            "terminal_analysis_critic_block": True,
                            "terminal_analysis_state_digest": state_digest,
                            "error": "Gemini r3 blocked the repair",
                        },
                    },
                )
            )
            await session.commit()

        self.assertIsNone(
            await _terminal_analysis_critic_recovery_reason(
                self.run_id,
                state_digest="different-state",
            )
        )
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
            ) as critic,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "terminal for the current analysis state",
            ):
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
        critic.assert_not_awaited()
        progress.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
