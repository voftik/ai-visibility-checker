import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.analysis_critic import (
    CRITIC_MAX_TOKENS,
    CRITIC_MODEL,
    CRITIC_PRIMARY_RAW_CHAR_BUDGET,
    CRITIC_REASONING_EFFORT,
    CRITIC_REPAIR_MAX_TOKENS,
    CRITIC_REPAIR_REASONING_EFFORT,
    CRITIC_VERSION,
    MAX_CRITIC_ITERATIONS,
    MAX_CRITIC_REPAIR_ATTEMPTS,
    _compact_repair_context,
    _transport_repair_may_pass,
    repair_analysis_review,
    review_analysis,
)
from app.services.analyzer import (
    ANNOTATION_VERSION,
    _analysis_critic_artifact,
    _annotation_context_sha256,
    _artifact_cache_matches,
    _apply_critic_policy,
    _critic_payload,
    _critic_provenance_digests,
    _critic_review_errors,
    _run_analysis_critic_loop,
    _save_critic_gate,
)
from app.services.openrouter import OpenRouterError, OpenRouterOutputLimitError


def _critic_review(
    verdict: str,
    *,
    adjustments: list[dict] | None = None,
    guidance: str = "",
) -> dict:
    return {
        "verdict": verdict,
        "summary": f"Critic verdict: {verdict}",
        "anomalies": [],
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
        self.assertEqual(CRITIC_VERSION, "aiv-analysis-critic-v17")
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


if __name__ == "__main__":
    unittest.main()
