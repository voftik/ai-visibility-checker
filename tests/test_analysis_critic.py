import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app import db as app_db
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
    CRITIC_CALL_AUDIT_VERSION,
    CRITIC_MAP_REDUCE_VERSION,
    CRITIC_MODEL,
    CRITIC_REASONING_EFFORT,
    CRITIC_REPAIR_REASONING_EFFORT,
    CRITIC_VERSION,
    MAX_CRITIC_RECOVERY_FINAL_REVIEWS,
    MAX_CRITIC_ITERATIONS,
    MAX_CRITIC_REPAIR_ATTEMPTS,
    _build_critic_leaf_payloads,
    _build_critic_map_plan,
    _build_context_fact_units,
    _build_context_join_tasks,
    _compact_repair_context,
    _critic_input_budget_bytes,
    _critic_physical_request_preflight,
    _merge_reviews_preserving_material_findings,
    _reduce_corpus_reviews,
    _reduce_fragmented_answer,
    _transport_repair_may_pass,
    _shared_context_semantic_inventory,
    _validate_context_join_coverage,
    _verify_fragment_core_accounting,
    repair_analysis_review,
    review_analysis,
)
from app.services.long_response import split_lossless_text
from app.services.analyzer import (
    ANALYSIS_CRITIC_VERSION,
    _AnalysisCriticRecoveryBlocked,
    _ConfirmedCriticIntegrityBlock,
    ANALYSIS_CRITIC_RECOVERY_CHECKPOINT_VERSION,
    ANALYSIS_CRITIC_RECOVERY_STAGE,
    ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
    ANNOTATION_COMPLETION_ATTEMPTS,
    ANNOTATION_VERSION,
    _analysis_critic_artifact,
    _annotation_context_manifest,
    _annotation_context_sha256,
    _annotation_matches_answer,
    _artifact_cache_matches,
    _apply_critic_policy,
    _critic_payload,
    _critic_provenance_digests,
    _critic_review_errors,
    _critic_review_validation_errors,
    _critic_analysis_state_digest,
    _compute_metrics,
    _code_owned_target_mention_receipts,
    _current_annotation_input_digests,
    _deterministic_annotation_warnings,
    _deterministic_critic_fallback_review,
    _finish_saved_answer_analysis,
    _literal_target_attribution_evidence,
    _load_executing_analysis_critic_checkpoint,
    _persist_prepared_analysis_critic_recovery,
    _reconcile_annotation,
    _recover_analysis_critic_exhaustion,
    _row_target_mention_is_grounded,
    _raw_corpus_digest,
    _refresh_target_mention_receipt_manifest,
    _run_analysis_critic_loop,
    _save_critic_gate,
    _save_targeted_recovery_annotations,
    _scope_entity_catalog_to_profile,
    _stable_json_sha256,
    _target_mention_receipt,
    _terminal_analysis_critic_recovery_reason,
    _validated_target_mention_receipt_manifest,
    _visibility_slice,
)
from app.services.openrouter import (
    OpenRouterError,
    OpenRouterOutputLimitError,
    OpenRouterResponseContractError,
    OutputTokenPolicy,
)
from app.services.recovery_orchestrator import (
    ACTION_STOP,
    ACTION_TARGETED_ANNOTATION_REPAIR,
    CHECK_CHECKPOINT_PRESERVED,
    CHECK_CRITIC_GATE_PASSED,
    CHECK_DERIVED_METRICS_RECOMPUTED,
    CHECK_RAW_CORPUS_UNCHANGED,
    OrchestratorContractError,
)
from app.services.recovery_state import recovery_scope_digest, stable_digest
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
        "metric_eligible": True,
        "context_eligible": True,
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


MAKAR_PROFILE = {
    "brand_name": "Makarska Tattoo & Piercing Studio",
    "brand_aliases": ["Makarska Tattoo & Piercing Salon"],
    "entity_scope": [
        {
            "canonical_name": "Makarska Tattoo & Piercing Studio",
            "aliases": ["Makarska Tattoo & Piercing Salon"],
            "relationship": "self",
            "entity_type": "primary_brand",
            "commercially_relevant": True,
            "confidence": "high",
        }
    ],
    "offer_catalog": {
        "client_domain": "makarskatattoo.com",
        "accepted_offers": [],
    },
}
MAKAR_CATALOG = {
    "entities": [
        {
            "canonical_name": "Makarska Tattoo & Piercing Studio",
            "aliases": [
                "Makarska Tattoo",
                "Tattoo & Piercing Makarska",
            ],
            "category": "target",
            "target_relationship": "exact_target",
            "commercially_relevant": True,
            "mention_policy": "standalone",
        }
    ]
}
MAKAR_RAW = (
    "В центре Макарски можно записаться в **Tattoo & Piercing Makarska "
    "(Makarska Tattoo)** на улице Lištun.[2][3]\n\n"
    "Другая студия приведена ниже.[1]"
)
MAKAR_ROW = {
    "answer_id": 580,
    "mode": "web",
    "provider_key": "perplexity",
    "model": "perplexity/sonar-pro-search",
    "prompt_id": 17,
    "prompt_key": "intent-nav",
    "scenario": (
        "Я в Макарске и хочу записаться на татуировку. "
        "Какие студии в центре принимают заявки?"
    ),
    "role": "unbranded_discovery",
    "intent_class": "NAV",
    "status": "completed",
    "metric_eligible": True,
    "annotation_state": "current",
    "citations_count": 3,
    "citations": [
        {"url": "https://example.org/directory", "title": "Directory"},
        {
            "url": "https://makarskatattoo.com/",
            "title": "Makarska Tattoo & Piercing Studio",
        },
        {
            "url": "https://makarskatattoo.com/contacts",
            "title": "Contacts",
        },
    ],
    "response_annotations": [
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.org/directory",
                "title": "Directory",
                "start_index": 0,
                "end_index": 0,
            },
        },
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://makarskatattoo.com/",
                "title": "Makarska Tattoo & Piercing Studio",
                "start_index": 0,
                "end_index": 0,
            },
        },
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://makarskatattoo.com/contacts",
                "title": "Contacts",
                "start_index": 0,
                "end_index": 0,
            },
        },
    ],
    "answer_text": MAKAR_RAW,
    "annotation": {
        "valid": True,
        "target_mentioned": False,
        "target_position": None,
        "target_role": "absent",
        "sentiment": "unknown",
        "entity_mentions": [],
        "evidence": [],
        "brand_answer": {
            "directness": "not_applicable",
            "specificity": "not_applicable",
            "supported_facets": [],
            "contradictions": [],
        },
    },
}


def _model_envelope(
    *,
    context_length: int = 81_920,
    max_completion_tokens: int = 8_192,
) -> dict:
    return {
        "context_length": context_length,
        "max_completion_tokens": max_completion_tokens,
        "resolution": "test",
    }


def _large_critic_payload(
    *,
    answer_count: int = 5,
    raw_chars: int = 25_000,
) -> dict:
    rows: list[dict] = []
    for answer_id in range(1, answer_count + 1):
        answer_text = f"Ответ {answer_id}. " + ("x" * raw_chars)
        annotation = {
            **ROWS[0]["annotation"],
            "_answer_sha256": hashlib.sha256(
                answer_text.encode("utf-8")
            ).hexdigest(),
            "entity_mentions": [],
        }
        rows.append(
            {
                **ROWS[0],
                "answer_id": answer_id,
                "answer_text": answer_text,
                "annotation": annotation,
            }
        )
    return _critic_payload(
        profile=PROFILE,
        catalog=CATALOG,
        rows=rows,
        metrics=METRICS,
        policy_history=[],
    )


def _context_join_base(answer_id: int, prompt_key: str) -> dict:
    return {
        "base_kind": "leaf",
        "base_index": answer_id,
        "payload": {
            "critic_map_partition": {
                "assigned_answer_ids": [answer_id],
            },
            "answers": [
                {
                    "answer_id": answer_id,
                    "prompt_id": answer_id * 10,
                    "prompt_key": prompt_key,
                    "scenario": f"Сценарий {prompt_key}",
                    "scenario_role": "unbranded_discovery",
                    "intent_class": "I",
                    "raw_answer": f"Ответ {answer_id}",
                }
            ],
        },
    }


def _context_receipt() -> dict:
    return {
        "status": "context_reduce_receipt_pending_answer_binding",
        "semantic_context": {"delivery": "pending_lossless_answer_context_join"},
        "context_review": {"verdict": "pass", "model_digest_sha256": "0" * 64},
    }


class AnalysisCriticTests(unittest.IsolatedAsyncioTestCase):
    async def test_finish_resumes_checkpoint_without_base_reannotation(
        self,
    ) -> None:
        class StopAfterCriticEntry(RuntimeError):
            pass

        receipts = _code_owned_target_mention_receipts(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            row=copy.deepcopy(MAKAR_ROW),
        )
        self.assertEqual(len(receipts), 1)
        annotation_context = _annotation_context_manifest(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            rows=[MAKAR_ROW],
            research_guidance="Persisted pre-repair guidance.",
            target_mention_receipts=receipts,
        )
        recovered_metrics = _compute_metrics(
            [copy.deepcopy(MAKAR_ROW)],
            MAKAR_PROFILE,
            MAKAR_CATALOG,
        )
        with (
            patch(
                "app.services.analyzer._admit_panel_metric_coverage",
                new=AsyncMock(return_value=([MAKAR_ROW], [], {})),
            ),
            patch(
                "app.services.analyzer._attach_offer_identity_policy",
                new=AsyncMock(return_value=MAKAR_PROFILE),
            ),
            patch(
                "app.services.analyzer."
                "_load_executing_analysis_critic_checkpoint",
                new=AsyncMock(
                    return_value=(
                        MAKAR_CATALOG,
                        [MAKAR_ROW],
                        recovered_metrics,
                        annotation_context,
                    )
                ),
            ),
            patch(
                "app.services.analyzer._answers_for_catalog",
                new_callable=AsyncMock,
            ) as answers_for_catalog,
            patch(
                "app.services.analyzer._entity_catalog",
                new_callable=AsyncMock,
            ) as entity_catalog,
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate,
            patch(
                "app.services.analyzer._run_analysis_critic_loop",
                new=AsyncMock(side_effect=StopAfterCriticEntry),
            ) as critic_loop,
        ):
            with self.assertRaises(StopAfterCriticEntry):
                await _finish_saved_answer_analysis(
                    "run-resume-checkpoint",
                    profile=MAKAR_PROFILE,
                    technical={},
                    technical_review={},
                    regenerate_illustrations=False,
                )

        answers_for_catalog.assert_not_awaited()
        entity_catalog.assert_not_awaited()
        annotate.assert_not_awaited()
        critic_loop.assert_awaited_once()
        self.assertEqual(
            critic_loop.await_args.kwargs["rows"],
            [MAKAR_ROW],
        )
        self.assertEqual(
            critic_loop.await_args.kwargs[
                "initial_target_mention_receipts"
            ],
            receipts,
        )
        self.assertEqual(
            critic_loop.await_args.kwargs["initial_annotation_context"],
            annotation_context,
        )

    async def test_recovery_checkpoint_commit_precedes_execution_reservation(
        self,
    ) -> None:
        class SimulatedCheckpointCrash(RuntimeError):
            pass

        plan = SimpleNamespace()
        with (
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(side_effect=SimulatedCheckpointCrash),
            ) as save_artifact,
            patch(
                "app.services.analyzer.mark_recovery_executing",
                new_callable=AsyncMock,
            ) as mark,
        ):
            with self.assertRaises(SimulatedCheckpointCrash):
                await _persist_prepared_analysis_critic_recovery(
                    "run-checkpoint-before-mark",
                    plan=plan,
                    profile=PROFILE,
                    catalog=CATALOG,
                    policy_history=[],
                    resume_annotation_context={"version": "fixture"},
                    checkpoint={"phase": "prepared"},
                )

        save_artifact.assert_awaited_once()
        mark.assert_not_awaited()

    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "critic-unit.sqlite3"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._db_patches = (
            patch.object(app_db, "engine", self.engine),
            patch(
                "app.services.analyzer.SessionLocal",
                self.SessionLocal,
            ),
            patch(
                "app.services.recovery_state.SessionLocal",
                self.SessionLocal,
            ),
        )
        for db_patch in self._db_patches:
            db_patch.start()
        await app_db.init_db()

    async def asyncTearDown(self) -> None:
        for db_patch in reversed(self._db_patches):
            db_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    async def test_clean_database_bootstrap_creates_recovery_epochs(
        self,
    ) -> None:
        async with self.engine.connect() as connection:
            table_names = {
                str(row[0])
                for row in (
                    await connection.exec_driver_sql(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table'"
                    )
                ).all()
            }
            recovery_columns = {
                str(row[1])
                for row in (
                    await connection.exec_driver_sql(
                        "PRAGMA table_info('recovery_epochs')"
                    )
                ).all()
            }

        self.assertIn("recovery_epochs", table_names)
        self.assertTrue(
            {
                "run_id",
                "epoch",
                "failure_fingerprint",
                "facts_digest",
                "status",
                "plan_json",
                "outcome_json",
            }.issubset(recovery_columns)
        )

    async def test_bootstrap_adds_recovery_epochs_without_losing_runs(
        self,
    ) -> None:
        legacy_run_id = str(uuid.uuid4())
        async with self.SessionLocal() as session:
            session.add(
                Run(
                    id=legacy_run_id,
                    domain="legacy.example",
                    status=RunStatus.completed,
                    config_json={"preserve": True},
                )
            )
            await session.commit()
        async with self.engine.begin() as connection:
            await connection.exec_driver_sql("DROP TABLE recovery_epochs")

        await app_db.init_db()

        async with self.engine.connect() as connection:
            recovery_table = (
                await connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'recovery_epochs'"
                )
            ).scalar_one()
        async with self.SessionLocal() as session:
            preserved = await session.get(Run, legacy_run_id)

        self.assertEqual(recovery_table, "recovery_epochs")
        self.assertIsNotNone(preserved)
        self.assertEqual(preserved.domain, "legacy.example")
        self.assertEqual(preserved.config_json, {"preserve": True})

    async def test_input_budget_reserves_only_model_output_window(
        self,
    ) -> None:
        with patch(
            "app.services.analysis_critic.model_output_envelope",
            new_callable=AsyncMock,
            return_value={
                "context_length": 81_920,
                "max_completion_tokens": 8_192,
                "resolution": "test",
            },
        ):
            budget, envelope = await _critic_input_budget_bytes()

        self.assertEqual(budget, 73_472)
        self.assertEqual(envelope["reserved_output_tokens"], 8_192)
        self.assertEqual(envelope["fixed_context_safety_tokens"], 256)
        self.assertEqual(envelope["input_budget_bytes"], 73_472)

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
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ) as envelope_mock,
        ):
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
        self.assertEqual(
            request["output_token_policy"],
            OutputTokenPolicy.MODEL_MAX,
        )
        self.assertIs(request["retry_response_contract_errors"], False)
        self.assertIs(request["retry_transport_errors"], False)
        self.assertNotIn("document_id", request)
        self.assertNotIn("audit_checkpoint", request)
        self.assertNotIn("resume_checkpoint", request)
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["iteration"], 1)
        self.assertEqual(payload["max_iterations"], MAX_CRITIC_ITERATIONS)
        self.assertEqual(CRITIC_VERSION, "aiv-analysis-critic-v27")
        self.assertEqual(
            usage["_aiv_critic_contract"]["semantic_verdict_status"],
            "pending_deterministic_validation",
        )
        envelope_mock.assert_awaited_once_with(CRITIC_MODEL)
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

    def test_large_corpus_partition_accounts_for_every_whole_answer_once(
        self,
    ) -> None:
        payload = _large_critic_payload()

        leaves, manifest = _build_critic_leaf_payloads(
            payload,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=65_536,
        )

        self.assertGreater(len(leaves), 1)
        self.assertEqual(
            manifest["version"],
            CRITIC_MAP_REDUCE_VERSION,
        )
        self.assertTrue(manifest["exact_accounting"])
        self.assertEqual(manifest["missing_answer_ids"], [])
        self.assertEqual(manifest["duplicate_answer_ids"], [])
        assigned = [
            answer_id
            for answer_ids in manifest["leaf_answer_ids"]
            for answer_id in answer_ids
        ]
        self.assertEqual(assigned, [1, 2, 3, 4, 5])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertTrue(
            all(size <= 65_536 for size in manifest["leaf_request_utf8_bytes"])
        )
        raw_by_id = {
            answer["answer_id"]: answer["raw_answer"]
            for answer in payload["answers"]
        }
        for leaf in leaves:
            self.assertEqual(leaf["site_profile"], payload["site_profile"])
            self.assertEqual(
                leaf["entity_catalog"], payload["entity_catalog"]
            )
            self.assertEqual(
                leaf["candidate_metrics"], payload["candidate_metrics"]
            )
            self.assertEqual(
                leaf["deterministic_warnings"],
                payload["deterministic_warnings"],
            )
            for answer in leaf["answers"]:
                self.assertEqual(
                    answer["raw_answer"],
                    raw_by_id[answer["answer_id"]],
                )

        reversed_payload = copy.deepcopy(payload)
        reversed_payload["answers"].reverse()
        _reversed_leaves, reversed_manifest = _build_critic_leaf_payloads(
            reversed_payload,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=65_536,
        )
        self.assertEqual(
            reversed_manifest["leaf_answer_ids"],
            manifest["leaf_answer_ids"],
        )

    async def test_map_reduce_preserves_revise_and_complete_child_provenance(
        self,
    ) -> None:
        payload = _large_critic_payload()

        async def critic_response(**kwargs):
            command = json.loads(kwargs["messages"][1]["content"])
            if "critic_map_partition" in command:
                answer_ids = command["critic_map_partition"][
                    "assigned_answer_ids"
                ]
                if 3 in answer_ids:
                    review = _critic_review(
                        "revise",
                        anomalies=[
                            {
                                "code": "annotation_evidence_mismatch",
                                "severity": "important",
                                "finding": (
                                    "В ответе № 3 буквальный фрагмент не "
                                    "подтверждает рассчитанную атрибуцию."
                                ),
                                "answer_ids": [3],
                                "entities": ["Campaign 360"],
                            }
                        ],
                        adjustments=[
                            {
                                "action": (
                                    "require_literal_attribution_evidence"
                                ),
                                "entity_name": "Campaign 360",
                                "alias": None,
                                "reason": (
                                    "Нужно повторно проверить точный "
                                    "непрерывный фрагмент ответа № 3."
                                ),
                                "answer_ids": [3],
                            }
                        ],
                        guidance=(
                            "Повторно разметить ответ № 3 по буквальному "
                            "непрерывному evidence."
                        ),
                    )
                else:
                    review = _critic_review("pass")
            else:
                # A model reducer is not trusted to preserve the floor: the
                # deterministic merger must recover the leaf revise below.
                self.assertEqual(
                    command["complete_answer_index"][0]["answer_id"],
                    1,
                )
                self.assertEqual(
                    command["complete_answer_manifests"][-1]["answer_id"],
                    5,
                )
                review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "_aiv_transport": {
                        "status": "succeeded",
                        "output_complete": True,
                    },
                },
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 81_920,
                    "max_completion_tokens": 8_192,
                    "resolution": "test",
                },
            ) as envelope_mock,
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=critic_response,
            ) as chat_mock,
        ):
            review, raw_text, usage = await review_analysis(
                payload,
                iteration=1,
            )

        self.assertEqual(review["verdict"], "revise")
        envelope_mock.assert_awaited_once_with(CRITIC_MODEL)
        self.assertTrue(
            any(
                anomaly["answer_ids"] == [3]
                and anomaly["severity"] == "important"
                for anomaly in review["anomalies"]
            )
        )
        self.assertEqual(review["policy_adjustments"][0]["answer_ids"], [3])
        provenance = usage["_aiv_critic_map_reduce"]
        self.assertTrue(provenance["exact_accounting"])
        self.assertEqual(provenance["complete_answer_ids"], [1, 2, 3, 4, 5])
        assigned = [
            answer_id
            for ids in provenance["leaf_answer_ids"]
            for answer_id in ids
        ]
        self.assertEqual(assigned, [1, 2, 3, 4, 5])
        self.assertEqual(len(assigned), len(set(assigned)))
        child_calls = provenance["child_calls"]
        self.assertEqual(
            [call["kind"] for call in child_calls].count("leaf"),
            provenance["leaf_count"],
        )
        self.assertEqual(
            [call["kind"] for call in child_calls].count("reducer"),
            1,
        )
        self.assertEqual(
            usage["total_tokens"],
            15 * len(child_calls),
        )
        raw_provenance = json.loads(raw_text)
        self.assertEqual(
            len(raw_provenance["leaf_responses"]),
            provenance["leaf_count"],
        )
        self.assertEqual(
            raw_provenance["final_review_sha256"],
            hashlib.sha256(
                json.dumps(
                    review,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        for call in chat_mock.await_args_list:
            self.assertEqual(
                call.kwargs["output_token_policy"],
                OutputTokenPolicy.MODEL_MAX,
            )
            self.assertNotIn("document_id", call.kwargs)
            self.assertNotIn("audit_checkpoint", call.kwargs)
            self.assertNotIn("resume_checkpoint", call.kwargs)
            self.assertNotIn("max_continuations", call.kwargs)

    async def test_map_leaf_audit_survives_sibling_failure_before_reducer(
        self,
    ) -> None:
        payload = _large_critic_payload()
        audited: list[dict[str, object]] = []

        async def audit_sink(event: dict[str, object]) -> None:
            audited.append(copy.deepcopy(event))

        async def critic_response(**kwargs):
            command = json.loads(kwargs["messages"][1]["content"])
            answer_ids = command.get("critic_map_partition", {}).get(
                "assigned_answer_ids",
                [],
            )
            if 3 in answer_ids:
                raise OpenRouterError("synthetic sibling failure")
            # The failing sibling returns first. A non-return_exceptions
            # gather would abandon this paid successful response before its
            # append-only audit record is written.
            await asyncio.sleep(0.02)
            review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "_aiv_transport": {
                        "status": "succeeded",
                        "output_complete": True,
                    },
                },
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 81_920,
                    "max_completion_tokens": 8_192,
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=critic_response,
            ),
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "after durable per-call audit",
            ):
                await review_analysis(
                    payload,
                    iteration=1,
                    audit_sink=audit_sink,
                )

        statuses = [str(event["status"]) for event in audited]
        self.assertIn("failed", statuses)
        self.assertIn("completed", statuses)
        self.assertTrue(
            all(
                event["version"] == CRITIC_CALL_AUDIT_VERSION
                for event in audited
            )
        )
        self.assertEqual(
            len({str(event["attempt_id"]) for event in audited}),
            len(audited),
        )
        completed = next(
            event for event in audited if event["status"] == "completed"
        )
        self.assertTrue(completed["provider_response_present"])
        self.assertTrue(str(completed["raw_text"]))
        self.assertEqual(
            completed["usage"]["total_tokens"],
            15,
        )
        failed = next(event for event in audited if event["status"] == "failed")
        self.assertEqual(failed["error_type"], "OpenRouterError")
        self.assertIn("synthetic sibling failure", str(failed["error_message"]))

    async def test_map_leaf_cancellation_is_explicitly_audited(self) -> None:
        payload = _large_critic_payload()
        audited: list[dict[str, object]] = []
        provider_started = asyncio.Event()
        never_release = asyncio.Event()

        async def audit_sink(event: dict[str, object]) -> None:
            audited.append(copy.deepcopy(event))

        async def blocked_provider(**_kwargs):
            provider_started.set()
            await never_release.wait()
            raise AssertionError("unreachable")

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 81_920,
                    "max_completion_tokens": 8_192,
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=blocked_provider,
            ),
        ):
            task = asyncio.create_task(
                review_analysis(
                    payload,
                    iteration=1,
                    audit_sink=audit_sink,
                )
            )
            await asyncio.wait_for(provider_started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(audited)
        self.assertTrue(
            all(event["status"] == "cancelled" for event in audited)
        )
        self.assertTrue(
            all(
                event["error_type"] == "CancelledError"
                for event in audited
            )
        )
        self.assertTrue(
            all(not event["provider_response_present"] for event in audited)
        )

    async def test_completed_paid_leaves_are_reused_after_reducer_crash(
        self,
    ) -> None:
        payload = _large_critic_payload()

        class DurableAudit:
            def __init__(self) -> None:
                self.events: list[dict] = []

            async def __call__(self, event: dict) -> None:
                self.events.append(copy.deepcopy(event))

            async def lookup_completed(self, descriptor: dict) -> dict | None:
                for event in reversed(self.events):
                    if (
                        event.get("logical_call_key")
                        == descriptor.get("logical_call_key")
                        and event.get("status") == "completed"
                    ):
                        return copy.deepcopy(event)
                return None

        audit = DurableAudit()
        first_leaf_posts = 0

        async def first_provider(**kwargs):
            nonlocal first_leaf_posts
            command = json.loads(kwargs["messages"][1]["content"])
            if command.get("stage") == "corpus_reduce":
                raise OpenRouterError("synthetic crash before final reducer")
            if "critic_map_partition" in command:
                first_leaf_posts += 1
            review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={"total_tokens": 1},
            )

        envelope = {
            "context_length": 81_920,
            "max_completion_tokens": 8_192,
        }
        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=envelope,
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=first_provider,
            ),
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "after durable sibling audit",
            ):
                await review_analysis(
                    payload,
                    iteration=1,
                    audit_sink=audit,
                )

        self.assertGreater(first_leaf_posts, 0)
        completed_leaves = [
            event
            for event in audit.events
            if event["kind"] == "leaf" and event["status"] == "completed"
        ]
        self.assertEqual(len(completed_leaves), first_leaf_posts)
        self.assertTrue(
            all(
                event["attempt_id"]
                == str(event["logical_call_key"])[:32]
                for event in completed_leaves
            )
        )

        second_commands: list[dict] = []

        async def second_provider(**kwargs):
            command = json.loads(kwargs["messages"][1]["content"])
            second_commands.append(command)
            if "critic_map_partition" in command:
                raise AssertionError("completed leaf was billed twice")
            review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={"total_tokens": 1},
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=envelope,
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=second_provider,
            ) as provider_mock,
        ):
            review, _raw, usage = await review_analysis(
                payload,
                iteration=1,
                audit_sink=audit,
            )

        self.assertEqual(review["verdict"], "pass")
        self.assertEqual(provider_mock.await_count, 1)
        self.assertEqual(second_commands[0]["stage"], "corpus_reduce")
        reused = [
            call
            for call in usage["_aiv_critic_map_reduce"]["child_calls"]
            if call["kind"] == "leaf"
            and call["usage"].get("_aiv_critic_resume", {}).get("reused")
        ]
        self.assertEqual(len(reused), first_leaf_posts)

    async def test_public_critic_entrypoints_keep_verdicts_atomic(self) -> None:
        checkpoints: list[dict] = []
        looked_up: list[str] = []

        async def audit_checkpoint(event: dict) -> None:
            checkpoints.append(copy.deepcopy(event))

        async def resume_lookup(document_id: str) -> dict | None:
            looked_up.append(document_id)
            return None

        response = SimpleNamespace(
            parsed=_critic_review("pass"),
            text=json.dumps(_critic_review("pass"), ensure_ascii=False),
            usage={},
        )
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as provider_mock,
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ),
        ):
            await review_analysis(
                {"site_profile": {"brand_name": "Example"}},
                iteration=1,
                transport_audit_checkpoint=audit_checkpoint,
                transport_resume_lookup=resume_lookup,
            )
            await repair_analysis_review(
                {"answers": []},
                _critic_review("revise"),
                iteration=1,
                validation_errors=["synthetic"],
                transport_audit_checkpoint=audit_checkpoint,
                transport_resume_lookup=resume_lookup,
            )

        self.assertEqual(provider_mock.await_count, 2)
        self.assertEqual(len(looked_up), 2)
        self.assertTrue(all(len(value) == 64 for value in looked_up))
        for call in provider_mock.await_args_list:
            kwargs = call.kwargs
            self.assertEqual(
                kwargs["output_token_policy"],
                OutputTokenPolicy.MODEL_MAX,
            )
            self.assertIs(kwargs["retry_response_contract_errors"], False)
            self.assertIs(kwargs["retry_transport_errors"], False)
            self.assertIs(kwargs["audit_checkpoint"], audit_checkpoint)
            self.assertIn("document_id", kwargs["audit_context"])
            self.assertNotIn("resume_checkpoint", kwargs)
            self.assertNotIn("max_continuations", kwargs)
        self.assertEqual(checkpoints, [])

    async def test_critic_preserves_arbitrarily_long_valid_structured_output(
        self,
    ) -> None:
        long_summary = "Содержательный вывод. " * 12_000
        review = {**_critic_review("pass"), "summary": long_summary}
        response = SimpleNamespace(
            parsed=review,
            text=json.dumps(review, ensure_ascii=False),
            usage={"total_tokens": 90_000},
        )
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as provider_mock,
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ),
        ):
            parsed, raw_text, usage = await review_analysis(
                {"site_profile": {"brand_name": "Example"}},
                iteration=1,
            )

        self.assertEqual(parsed["summary"], long_summary)
        self.assertEqual(json.loads(raw_text)["summary"], long_summary)
        self.assertEqual(usage["total_tokens"], 90_000)
        request = provider_mock.await_args.kwargs
        self.assertEqual(
            request["output_token_policy"],
            OutputTokenPolicy.MODEL_MAX,
        )
        self.assertIs(request["retry_response_contract_errors"], False)
        self.assertIs(request["retry_transport_errors"], False)
        self.assertNotIn("max_continuations", request)
        self.assertNotIn("max_completion_tokens", request)

    def test_map_reduce_verdict_floor_preserves_leaf_block(self) -> None:
        blocked = _critic_review(
            "block",
            anomalies=[
                {
                    "code": "fabricated_evidence",
                    "severity": "critical",
                    "finding": "Leaf обнаружил выдуманное доказательство.",
                    "answer_ids": [17],
                    "entities": ["Campaign 360"],
                }
            ],
        )
        reduced = _merge_reviews_preserving_material_findings(
            [blocked, _critic_review("pass")],
            _critic_review("pass"),
        )

        self.assertEqual(reduced["verdict"], "block")
        self.assertEqual(reduced["anomalies"][0]["answer_ids"], [17])

    def test_map_partition_rejects_duplicate_and_fragments_oversized_answer(
        self,
    ) -> None:
        duplicate = _large_critic_payload(answer_count=2, raw_chars=1_000)
        duplicate["answers"][1]["answer_id"] = 1
        with self.assertRaisesRegex(OpenRouterError, "duplicate answer_id=1"):
            _build_critic_leaf_payloads(
                duplicate,
                iteration=1,
                max_iterations=MAX_CRITIC_ITERATIONS,
                recovery_final=False,
                input_budget_bytes=65_536,
            )

        oversized = _large_critic_payload(answer_count=1, raw_chars=80_000)
        whole, fragments, manifest = _build_critic_map_plan(
            oversized,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=65_536,
        )
        self.assertEqual(whole, [])
        self.assertGreater(len(fragments), 1)
        self.assertEqual(manifest["fragmented_answer_ids"], [1])
        fragment_manifest = manifest["fragmented_answers"][0]
        self.assertTrue(fragment_manifest["exact_core_accounting"])
        self.assertEqual(
            fragment_manifest["submitted_core_chars"],
            fragment_manifest["source_chars"],
        )
        self.assertEqual(
            len(fragment_manifest["core_unit_ids"]),
            len(set(fragment_manifest["core_unit_ids"])),
        )

    def test_fragment_core_accounting_fails_on_missing_or_duplicate_unit(
        self,
    ) -> None:
        units, manifest = split_lossless_text(
            "x" * 2_000,
            document_id="critic-test",
            target_chars=500,
            context_overlap_chars=128,
        )
        manifest_json = manifest.as_dict()
        self.assertEqual(
            _verify_fragment_core_accounting(units, manifest_json),
            "x" * 2_000,
        )
        with self.assertRaisesRegex(OpenRouterError, "core unit accounting"):
            _verify_fragment_core_accounting(units[:-1], manifest_json)
        with self.assertRaisesRegex(OpenRouterError, "duplicate core unit ids"):
            _verify_fragment_core_accounting(
                [units[0], units[0], *units[1:]],
                manifest_json,
            )

    def test_map_plan_losslessly_partitions_oversized_shared_context(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=1, raw_chars=50)
        payload["site_profile"] = {
            **payload["site_profile"],
            "evidence": "X" * 120_000,
        }

        whole, tasks, manifest = _build_critic_map_plan(
            payload,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=50_000,
        )

        context_tasks = [
            task for task in tasks if task.get("kind") == "shared_context"
        ]
        self.assertTrue(manifest["shared_context_partitioned"])
        self.assertGreater(len(context_tasks), 1)
        self.assertEqual(
            manifest["shared_context_leaf_count"],
            len(context_tasks),
        )
        self.assertTrue(
            all(size <= 50_000 for size in manifest["leaf_request_utf8_bytes"])
        )
        expected = json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "answers"
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        reconstructed_parts: list[tuple[int, str]] = []
        for task in context_tasks:
            leaf = task["payload"]
            partition = leaf["critic_shared_context_partition"]
            fragment = leaf["shared_context_json_fragment"]
            core = fragment[
                partition["core_start_in_context"] : partition[
                    "core_end_in_context"
                ]
            ]
            reconstructed_parts.append((partition["unit_index"], core))
        reconstructed = "".join(
            core for _index, core in sorted(reconstructed_parts)
        )
        self.assertEqual(reconstructed, expected)
        self.assertEqual(
            manifest["shared_context"]["source_sha256"],
            hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(whole[0]["answers"][0]["raw_answer"], payload[
            "answers"
        ][0]["raw_answer"])
        self.assertNotIn("site_profile", whole[0])
        self.assertEqual(
            whole[0]["shared_context_digest"]["status"],
            "pending_lossless_answer_context_join",
        )

    async def test_shared_context_material_verdict_survives_answer_reduce(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=1, raw_chars=50)
        payload["site_profile"] = {
            **payload["site_profile"],
            "evidence": "X" * 120_000,
        }
        seen_answer_digest = False
        seen_semantic_facts: list[dict] = []
        request_sizes: list[int] = []

        async def provider(**kwargs):
            nonlocal seen_answer_digest, seen_semantic_facts
            request_sizes.append(
                len(json.dumps(kwargs["messages"], ensure_ascii=False).encode(
                    "utf-8"
                ))
            )
            command = json.loads(kwargs["messages"][1]["content"])
            if "critic_shared_context_partition" in command:
                partition = command["critic_shared_context_partition"]
                review = (
                    _critic_review(
                        "revise",
                        anomalies=[
                            {
                                "code": "scope_leakage",
                                "severity": "important",
                                "finding": "Контекст содержит material-проблему.",
                                "answer_ids": [],
                                "entities": ["Example"],
                            }
                        ],
                    )
                    if partition["unit_index"] == 0
                    else _critic_review("pass")
                )
            elif "critic_context_binding" in command:
                seen_answer_digest = bool(command.get("shared_context_receipt"))
                for unit in command["shared_context_facts"]:
                    if unit["mode"] == "complete_fact":
                        seen_semantic_facts.append(unit["fact"])
                self.assertNotIn("site_profile", command)
                review = _critic_review("pass")
            else:
                review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={"total_tokens": 1},
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 66_384,
                    "max_completion_tokens": 8_192,
                },
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=provider,
            ),
        ):
            review, _raw, usage = await review_analysis(payload, iteration=1)

        self.assertTrue(seen_answer_digest)
        facts_by_path = {
            fact["path"]: fact.get("value") for fact in seen_semantic_facts
        }
        self.assertEqual(facts_by_path["/site_profile/brand_name"], "Example")
        # Unmentioned entity records are reviewed once by the shared-context
        # tree, not repeated beside every answer/query leaf.
        self.assertNotIn(
            "Campaign 360",
            [
                value
                for path, value in facts_by_path.items()
                if path.endswith("/canonical_name")
            ],
        )
        self.assertEqual(review["verdict"], "revise")
        self.assertTrue(
            any(
                anomaly["finding"] == "Контекст содержит material-проблему."
                for anomaly in review["anomalies"]
            )
        )
        self.assertTrue(request_sizes)
        self.assertTrue(all(size <= 58_192 for size in request_sizes))
        provenance = usage["_aiv_critic_map_reduce"]
        self.assertTrue(provenance["shared_context_partitioned"])
        self.assertTrue(provenance["shared_context_digest_sha256"])

    def test_context_fact_units_keep_huge_fact_content_losslessly(self) -> None:
        decisive_suffix = "::КЛИЕНТ-ПРИНАДЛЕЖИТ-ЭТОМУ-САЙТУ::"
        huge_value = ("контекст " * 18_000) + decisive_suffix
        inventory = _shared_context_semantic_inventory(
            {
                "site_profile": {
                    "brand_name": "Example",
                    "market_evidence": huge_value,
                },
                "answers": [],
            }
        )

        units, manifest = _build_context_fact_units(
            inventory,
            per_call_reserve_bytes=12_000,
        )

        fact_index = next(
            index
            for index, fact in enumerate(inventory["facts"])
            if fact["path"] == "/site_profile/market_evidence"
        )
        fact_manifest = manifest["fact_manifests"][fact_index]
        by_id = {unit["unit_id"]: unit for unit in units}
        reconstructed = "".join(
            by_id[unit_id]["fact_json_fragment"][
                by_id[unit_id]["fragment"]["core_start_in_context"] :
                by_id[unit_id]["fragment"]["core_end_in_context"]
            ]
            for unit_id in fact_manifest["unit_ids"]
        )
        exact_fact = json.loads(reconstructed)
        self.assertEqual(exact_fact, inventory["facts"][fact_index])
        self.assertTrue(exact_fact["value"].endswith(decisive_suffix))
        self.assertGreater(len(fact_manifest["unit_ids"]), 100)
        serialized_units = json.dumps(units, ensure_ascii=False)
        self.assertNotIn("value_omitted", serialized_units)
        self.assertNotIn("hash_only", serialized_units)

    def test_context_join_relevance_index_avoids_paid_fact_cross_product(
        self,
    ) -> None:
        entities = [
            {
                "canonical_name": f"Продукт {index:03d}",
                "description": (f"Факт {index:03d} " + "z" * 480),
            }
            for index in range(180)
        ]
        inventory = _shared_context_semantic_inventory(
            {
                "site_profile": {"brand_name": "Example"},
                "entity_catalog": {"entities": entities},
                "answers": [],
            }
        )
        envelope = _model_envelope(
            context_length=32_768,
            max_completion_tokens=4_096,
        )

        tasks, manifest = _build_context_join_tasks(
            base_entries=[
                _context_join_base(1, "intent-a"),
                _context_join_base(2, "intent-b"),
            ],
            semantic_inventory=inventory,
            context_receipt=_context_receipt(),
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=28_416,
            context_envelope=envelope,
            per_call_reserve_bytes=12_000,
        )

        expected_units = manifest["fact_manifest"]["unit_ids"]
        self.assertGreater(len(expected_units), 300)
        self.assertLessEqual(manifest["task_count"], 2)
        self.assertEqual(manifest["base_count"], 2)
        self.assertEqual(len(tasks), manifest["task_count"])
        for base in manifest["base_manifests"]:
            self.assertNotEqual(base["covered_unit_ids"], expected_units)
            self.assertEqual(
                base["covered_unit_ids"],
                base["expected_context_unit_ids"],
            )
            self.assertTrue(base["exact_relevant_unit_accounting"])
            self.assertEqual(len(base["answer_query_bindings"]), 1)
        self.assertEqual(
            [
                base["answer_query_bindings"][0]["prompt_key"]
                for base in manifest["base_manifests"]
            ],
            ["intent-a", "intent-b"],
        )
        late_path = "/entity_catalog/entities/179/description"
        late_unit_ids = {
            unit_id
            for unit_id, unit in {
                unit["unit_id"]: unit
                for task in tasks
                for unit in task["payload"]["shared_context_facts"]
            }.items()
            if unit.get("fact", {}).get("path") == late_path
        }
        self.assertFalse(late_unit_ids)
        self.assertGreater(
            manifest["avoided_cross_product_unit_attachments"],
            500,
        )
        self.assertEqual(
            sorted(
                [
                    *manifest["joined_unique_unit_ids"],
                    *manifest["shared_context_only_unit_ids"],
                ]
            ),
            sorted(expected_units),
        )
        self.assertTrue(manifest["exact_global_fact_accounting"])
        self.assertTrue(
            manifest["paid_task_cost_bound"][
                "task_count_lte_relevant_unit_attachments"
            ]
        )
        self.assertTrue(
            all(
                task["physical_request_utf8_bytes"] <= 28_416
                and task["physical_preflight"]["fits_model_envelope"] is True
                and task["physical_preflight"][
                    "effective_max_completion_tokens"
                ] == 4_096
                for task in tasks
            )
        )

    def test_context_join_binds_exact_entity_owner_scope_to_same_answer(
        self,
    ) -> None:
        inventory = _shared_context_semantic_inventory(
            {
                "site_profile": {"brand_name": "Example"},
                "entity_catalog": {
                    "target_aliases": ["Example"],
                    "entities": [
                        {
                            "canonical_name": "Campaign 360",
                            "aliases": ["Campaign360"],
                        },
                        {
                            "canonical_name": "Sibling Service",
                            "aliases": ["Sibling"],
                        },
                    ],
                },
                # This global owner is context, not permission to attribute
                # every entity in the catalog to that owner.
                "attribution_owner_aliases": ["MR Group"],
                "entity_attribution_aliases": {
                    "campaign 360": ["Example", "MR Group"],
                    "sibling service": ["Example"],
                },
                "answers": [],
            }
        )
        campaign = _context_join_base(1, "campaign")
        campaign["payload"]["answers"][0]["raw_answer"] = (
            "Campaign 360 — продукт MR Group."
        )
        sibling = _context_join_base(2, "sibling")
        sibling["payload"]["answers"][0]["raw_answer"] = (
            "Sibling Service ошибочно назван продуктом MR Group."
        )
        tasks, manifest = _build_context_join_tasks(
            base_entries=[campaign, sibling],
            semantic_inventory=inventory,
            context_receipt=_context_receipt(),
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=28_416,
            context_envelope=_model_envelope(
                context_length=32_768,
                max_completion_tokens=4_096,
            ),
            per_call_reserve_bytes=12_000,
        )

        paths_by_answer: dict[int, set[str]] = {1: set(), 2: set()}
        values_by_answer: dict[int, dict[str, object]] = {1: {}, 2: {}}
        for task in tasks:
            answer_id = task["assigned_answer_ids"][0]
            for unit in task["payload"]["shared_context_facts"]:
                self.assertEqual(unit["mode"], "complete_fact")
                fact = unit["fact"]
                path = fact["path"]
                paths_by_answer[answer_id].add(path)
                values_by_answer[answer_id][path] = fact["value"]

        campaign_owner_path = (
            "/entity_attribution_aliases/campaign 360/1"
        )
        sibling_owner_path = (
            "/entity_attribution_aliases/sibling service/0"
        )
        self.assertEqual(values_by_answer[1][campaign_owner_path], "MR Group")
        self.assertIn(sibling_owner_path, paths_by_answer[2])
        self.assertNotIn(campaign_owner_path, paths_by_answer[2])
        self.assertFalse(
            any(
                path.startswith(
                    "/entity_attribution_aliases/sibling service/"
                )
                and value == "MR Group"
                for path, value in values_by_answer[2].items()
            )
        )
        # The owner remains globally visible, but the exact sibling allowlist
        # proves that global visibility does not authorize attribution.
        self.assertEqual(
            values_by_answer[2]["/attribution_owner_aliases/0"],
            "MR Group",
        )
        for base_manifest in manifest["base_manifests"]:
            selected_reasons = set(
                base_manifest["relevance_manifest"][
                    "selection_reasons"
                ].values()
            )
            self.assertIn("entity_attribution_binding", selected_reasons)

    def test_context_join_fails_closed_on_missing_cross_bound_or_tampered_data(
        self,
    ) -> None:
        inventory = _shared_context_semantic_inventory(
            {
                "site_profile": {"brand_name": "Example"},
                "entity_catalog": {
                    "entities": [
                        {
                            "canonical_name": f"Сущность {index}",
                            "evidence": "e" * 600,
                        }
                        for index in range(40)
                    ]
                },
                "answers": [],
            }
        )
        tasks, manifest = _build_context_join_tasks(
            base_entries=[
                _context_join_base(1, "intent-a"),
                _context_join_base(2, "intent-b"),
            ],
            semantic_inventory=inventory,
            context_receipt=_context_receipt(),
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=28_416,
            context_envelope=_model_envelope(
                context_length=32_768,
                max_completion_tokens=4_096,
            ),
            per_call_reserve_bytes=12_000,
        )
        results = [
            {
                "task_id": task["task_id"],
                "base_leaf_id": task["base_leaf_id"],
                "assigned_answer_ids": task["assigned_answer_ids"],
                "context_unit_ids": task["context_unit_ids"],
                "input_sha256": task["payload_sha256"],
                "review": _critic_review("pass"),
            }
            for task in tasks
        ]
        _validate_context_join_coverage(tasks, manifest, results)

        with self.assertRaisesRegex(OpenRouterError, "missing|reordered"):
            _validate_context_join_coverage(tasks[:-1], manifest)
        with self.assertRaisesRegex(OpenRouterError, "missing|reordered"):
            _validate_context_join_coverage(tasks, manifest, results[:-1])

        cross_bound = copy.deepcopy(results)
        cross_bound[0]["base_leaf_id"] = tasks[-1]["base_leaf_id"]
        with self.assertRaisesRegex(OpenRouterError, "cross-bound|tampered"):
            _validate_context_join_coverage(tasks, manifest, cross_bound)

        content_tamper = copy.deepcopy(tasks)
        first_unit = content_tamper[0]["payload"]["shared_context_facts"][0]
        if first_unit["mode"] == "complete_fact":
            first_unit["fact"]["value"] = "ПОДМЕНЕНО"
        else:
            first_unit["fact_json_fragment"] += "ПОДМЕНЕНО"
        with self.assertRaisesRegex(OpenRouterError, "mutated|tampered"):
            _validate_context_join_coverage(content_tamper, manifest)

        for field, replacement in (
            ("temperature", 0.9),
            ("system_prompt", "подменённый prompt"),
            ("schema_name", "foreign_schema"),
        ):
            physical_tamper = copy.deepcopy(tasks)
            request = physical_tamper[0]["physical_preflight"][
                "request_payload"
            ]
            if field == "temperature":
                request["temperature"] = replacement
            elif field == "system_prompt":
                request["messages"][0]["content"] = replacement
            else:
                request["response_format"]["json_schema"]["name"] = replacement
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    OpenRouterError,
                    "physical request",
                ):
                    _validate_context_join_coverage(physical_tamper, manifest)

    def test_critic_physical_preflight_hashes_exact_provider_body(self) -> None:
        payload = _context_join_base(7, "intent-exact")["payload"]
        preflight = _critic_physical_request_preflight(
            payload,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            schema_name="aiv_analysis_critic_exact",
            context_envelope=_model_envelope(),
            map_leaf=True,
            context_join_leaf=True,
        )
        body = preflight["request_payload"]
        exact_bytes = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(preflight["request_utf8_bytes"], len(exact_bytes))
        self.assertEqual(
            preflight["request_sha256"],
            hashlib.sha256(exact_bytes).hexdigest(),
        )
        self.assertEqual(preflight["input_token_upper_bound"], len(exact_bytes) + 256)
        self.assertEqual(body["model"], CRITIC_MODEL)
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["reasoning"]["effort"], CRITIC_REASONING_EFFORT)
        self.assertEqual(
            body["response_format"]["json_schema"]["name"],
            "aiv_analysis_critic_exact",
        )
        self.assertEqual(body["max_completion_tokens"], 8_192)
        command = json.loads(body["messages"][1]["content"])
        self.assertEqual(command["answers"], payload["answers"])
        self.assertEqual(command["iteration"], 1)

    async def test_per_answer_reducer_rejects_missing_or_duplicate_lineage(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=1, raw_chars=80_000)
        _whole, fragment_tasks, plan = _build_critic_map_plan(
            payload,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=65_536,
        )
        fragment_manifest = plan["fragmented_answers"][0]
        with self.assertRaisesRegex(OpenRouterError, "incomplete lineage"):
            await _reduce_fragmented_answer(
                payload,
                answer_id=1,
                fragment_results=[],
                fragment_manifest=fragment_manifest,
                iteration=1,
                recovery_final=False,
                input_budget_bytes=65_536,
            )

        first = {
            "unit_id": fragment_tasks[0]["unit_id"],
            "unit_index": fragment_tasks[0]["unit_index"],
            "review": _critic_review("pass"),
        }
        with self.assertRaisesRegex(OpenRouterError, "incomplete lineage"):
            await _reduce_fragmented_answer(
                payload,
                answer_id=1,
                fragment_results=[first, dict(first)],
                fragment_manifest=fragment_manifest,
                iteration=1,
                recovery_final=False,
                input_budget_bytes=65_536,
            )

    async def test_recovery_final_answer_reducer_uses_bounded_fan_in_tree(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=1, raw_chars=10)
        unit_ids = [f"unit-{index:03d}" for index in range(48)]
        fragment_results = [
            {
                "unit_id": unit_id,
                "unit_index": index,
                "review": (
                    _critic_review(
                        "revise",
                        anomalies=[
                            {
                                "code": "annotation_evidence_mismatch",
                                "severity": "important",
                                "finding": "Последний фрагмент требует правки.",
                                "answer_ids": [1],
                                "entities": ["Campaign 360"],
                            }
                        ],
                    )
                    if index == len(unit_ids) - 1
                    else _critic_review("pass")
                ),
            }
            for index, unit_id in enumerate(unit_ids)
        ]
        audited: list[dict] = []

        async def audit_sink(event: dict) -> None:
            audited.append(copy.deepcopy(event))

        response_review = _critic_review("pass")
        response = SimpleNamespace(
            parsed=response_review,
            text=json.dumps(response_review, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        budget = 10_000
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await _reduce_fragmented_answer(
                payload,
                answer_id=1,
                fragment_results=fragment_results,
                fragment_manifest={
                    "core_unit_ids": unit_ids,
                    "exact_core_accounting": True,
                },
                iteration=(
                    MAX_CRITIC_ITERATIONS
                    + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                ),
                recovery_final=True,
                input_budget_bytes=budget,
                audit_sink=audit_sink,
            )

        self.assertEqual(result["review"]["verdict"], "revise")
        self.assertGreater(len(result["provider_calls"]), 1)
        self.assertGreater(
            max(call["level"] for call in result["provider_calls"]),
            0,
        )
        self.assertTrue(
            all(
                call["request_utf8_bytes"] <= budget
                for call in result["provider_calls"]
            )
        )
        self.assertEqual(result["provider_calls"][-1]["lineage"], unit_ids)
        self.assertEqual(len(audited), len(result["provider_calls"]))
        self.assertTrue(
            all(event["kind"] == "answer_reducer" for event in audited)
        )
        self.assertTrue(all(event["status"] == "completed" for event in audited))

    async def test_recovery_final_corpus_reducer_uses_bounded_fan_in_tree(
        self,
    ) -> None:
        answer_ids = list(range(1, 49))
        payload = _large_critic_payload(
            answer_count=len(answer_ids),
            raw_chars=10,
        )
        corpus_results = [
            {
                "assigned_answer_ids": [answer_id],
                "review": (
                    _critic_review(
                        "block",
                        anomalies=[
                            {
                                "code": "fabricated_evidence",
                                "severity": "critical",
                                "finding": "Последний ответ содержит выдумку.",
                                "answer_ids": [answer_id],
                                "entities": ["Campaign 360"],
                            }
                        ],
                    )
                    if answer_id == answer_ids[-1]
                    else _critic_review("pass")
                ),
            }
            for answer_id in answer_ids
        ]
        audited: list[dict] = []

        async def audit_sink(event: dict) -> None:
            audited.append(copy.deepcopy(event))

        response_review = _critic_review("pass")
        response = SimpleNamespace(
            parsed=response_review,
            text=json.dumps(response_review, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        budget = 12_000
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await _reduce_corpus_reviews(
                payload,
                corpus_results=corpus_results,
                partition_manifest={
                    "complete_answer_ids": answer_ids,
                    "exact_accounting": True,
                },
                iteration=(
                    MAX_CRITIC_ITERATIONS
                    + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                ),
                recovery_final=True,
                input_budget_bytes=budget,
                audit_sink=audit_sink,
            )

        self.assertEqual(result["review"]["verdict"], "block")
        self.assertGreater(len(result["provider_calls"]), 1)
        self.assertGreater(
            max(call["level"] for call in result["provider_calls"]),
            0,
        )
        self.assertEqual(result["provider_calls"][-1]["lineage"], answer_ids)
        self.assertTrue(
            all(
                call["request_utf8_bytes"] <= budget
                for call in result["provider_calls"]
            )
        )
        self.assertEqual(len(audited), len(result["provider_calls"]))
        self.assertTrue(
            all(event["kind"] == "corpus_reducer" for event in audited)
        )

    async def test_recovery_final_reducer_preserves_paid_contract_result(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=1, raw_chars=10)
        paid_result = SimpleNamespace(
            parsed=None,
            text='{"verdict":"revise"',
            usage={"prompt_tokens": 40, "completion_tokens": 37},
        )
        audited: list[dict] = []

        async def audit_sink(event: dict) -> None:
            audited.append(copy.deepcopy(event))

        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            side_effect=OpenRouterResponseContractError(
                "synthetic incomplete reducer response",
                result=paid_result,
            ),
        ):
            with self.assertRaises(OpenRouterResponseContractError) as caught:
                await _reduce_corpus_reviews(
                    payload,
                    corpus_results=[
                        {
                            "assigned_answer_ids": [1],
                            "review": _critic_review("pass"),
                        }
                    ],
                    partition_manifest={
                        "complete_answer_ids": [1],
                        "exact_accounting": True,
                    },
                    iteration=(
                        MAX_CRITIC_ITERATIONS
                        + MAX_CRITIC_RECOVERY_FINAL_REVIEWS
                    ),
                    recovery_final=True,
                    input_budget_bytes=65_536,
                    audit_sink=audit_sink,
                )

        self.assertIs(caught.exception.result, paid_result)
        self.assertEqual(len(audited), 1)
        event = audited[0]
        self.assertEqual(event["kind"], "corpus_reducer")
        self.assertEqual(event["status"], "failed")
        self.assertTrue(event["provider_response_present"])
        self.assertEqual(event["raw_text"], paid_result.text)
        self.assertEqual(event["usage"]["prompt_tokens"], 40)
        self.assertEqual(event["usage"]["completion_tokens"], 37)

    async def test_corpus_reducer_waits_for_paid_sibling_audits_on_failure(
        self,
    ) -> None:
        answer_ids = list(range(1, 13))
        payload = _large_critic_payload(
            answer_count=len(answer_ids),
            raw_chars=10,
        )
        audited: list[dict] = []

        async def audit_sink(event: dict) -> None:
            audited.append(copy.deepcopy(event))

        async def reducer_response(**kwargs):
            command = json.loads(kwargs["messages"][1]["content"])
            group_index = command["reduction_tree"]["group_index"]
            if group_index == 0:
                raise OpenRouterError("synthetic corpus reducer failure")
            await asyncio.sleep(0.02)
            review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={"total_tokens": 1},
            )

        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            side_effect=reducer_response,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "after durable sibling audit",
            ):
                await _reduce_corpus_reviews(
                    payload,
                    corpus_results=[
                        {
                            "assigned_answer_ids": [answer_id],
                            "review": _critic_review("pass"),
                        }
                        for answer_id in answer_ids
                    ],
                    partition_manifest={
                        "complete_answer_ids": answer_ids,
                        "exact_accounting": True,
                    },
                    iteration=1,
                    recovery_final=False,
                    input_budget_bytes=12_000,
                    audit_sink=audit_sink,
                )

        self.assertTrue(any(event["status"] == "failed" for event in audited))
        self.assertTrue(
            any(event["status"] == "completed" for event in audited)
        )

    async def test_single_answer_beyond_context_keeps_boundary_fact_once(
        self,
    ) -> None:
        budget = 73_472
        payload = _large_critic_payload(answer_count=1, raw_chars=120_000)
        _whole, _fragments, initial_manifest = _build_critic_map_plan(
            payload,
            iteration=1,
            max_iterations=MAX_CRITIC_ITERATIONS,
            recovery_final=False,
            input_budget_bytes=budget,
        )
        first_core_end = initial_manifest["fragmented_answers"][0][
            "lossless_manifest"
        ]["units"][0]["end_char"]
        marker = "Example->Campaign360"
        marker_start = first_core_end - 8
        answer = payload["answers"][0]
        original = answer["raw_answer"]
        crossed = (
            original[:marker_start]
            + marker
            + original[marker_start + len(marker) :]
        )
        self.assertEqual(len(crossed), len(original))
        _source_units, source_manifest = split_lossless_text(
            crossed,
            document_id="1",
            target_chars=24_000,
        )
        source_sha = hashlib.sha256(crossed.encode("utf-8")).hexdigest()
        answer.update(
            {
                "raw_answer": crossed,
                "raw_answer_sha256": source_sha,
                "raw_answer_char_count": len(crossed),
                "raw_answer_manifest": source_manifest.as_dict(),
            }
        )
        answer["annotation"] = {
            **answer["annotation"],
            "_answer_sha256": source_sha,
        }

        contexts_with_marker: list[str] = []
        owning_units: list[str] = []

        async def critic_response(**kwargs):
            command = json.loads(kwargs["messages"][1]["content"])
            if command.get("critic_map_partition", {}).get("mode") == (
                "answer_fragment"
            ):
                fragment_answer = command["answers"][0]
                fragment = fragment_answer["critic_fragment"]
                context = fragment_answer["raw_answer"]
                marker_in_context = context.find(marker)
                if marker_in_context >= 0:
                    contexts_with_marker.append(fragment["unit_id"])
                owns_marker = bool(
                    fragment["core_start_in_context"]
                    <= marker_in_context
                    < fragment["core_end_in_context"]
                )
                if owns_marker:
                    owning_units.append(fragment["unit_id"])
                    review = _critic_review(
                        "revise",
                        anomalies=[
                            {
                                "code": "annotation_evidence_mismatch",
                                "severity": "important",
                                "finding": (
                                    "Связь Example->Campaign360 пересекает "
                                    "границу core и требует повторной разметки."
                                ),
                                "answer_ids": [1],
                                "entities": ["Campaign 360"],
                            }
                        ],
                        adjustments=[
                            {
                                "action": (
                                    "require_literal_attribution_evidence"
                                ),
                                "entity_name": "Campaign 360",
                                "alias": None,
                                "reason": (
                                    "Буквальная связь найдена на границе "
                                    "semantic-overlap фрагментов."
                                ),
                                "answer_ids": [1],
                            }
                        ],
                        guidance=(
                            "Повторно проверить полный непрерывный фрагмент "
                            "Example->Campaign360 в ответе № 1."
                        ),
                    )
                else:
                    review = _critic_review("pass")
            elif command.get("stage") == "fragmented_answer_reduce":
                # Both reducers try to weaken the decision. The code-owned
                # floor must preserve the fragment's material finding.
                review = _critic_review("pass")
            else:
                review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "_aiv_transport": {
                        "status": "succeeded",
                        "output_complete": True,
                    },
                },
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 81_920,
                    "max_completion_tokens": 8_192,
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=critic_response,
            ) as chat_mock,
        ):
            review, raw_text, usage = await review_analysis(
                payload,
                iteration=1,
            )

        self.assertEqual(review["verdict"], "revise")
        self.assertEqual(len(owning_units), 1)
        self.assertGreaterEqual(len(set(contexts_with_marker)), 2)
        provenance = usage["_aiv_critic_map_reduce"]
        self.assertEqual(provenance["complete_answer_ids"], [1])
        self.assertEqual(provenance["whole_answer_ids"], [])
        self.assertEqual(provenance["fragmented_answer_ids"], [1])
        fragments = provenance["fragmented_answers"][0]
        final_boundary = fragments["lossless_manifest"]["units"][0][
            "end_char"
        ]
        self.assertLess(marker_start, final_boundary)
        self.assertGreater(marker_start + len(marker), final_boundary)
        self.assertTrue(fragments["exact_core_accounting"])
        self.assertEqual(
            fragments["submitted_core_chars"],
            fragments["source_chars"],
        )
        self.assertGreater(fragments["overlap_chars_excluded_from_coverage"], 0)
        self.assertEqual(fragments["missing_core_unit_ids"], [])
        self.assertEqual(fragments["duplicate_core_unit_ids"], [])
        child_kinds = [
            child["kind"] for child in provenance["child_calls"]
        ]
        self.assertEqual(
            child_kinds.count("fragment_leaf"),
            fragments["core_unit_count"],
        )
        self.assertEqual(child_kinds.count("answer_reducer"), 1)
        self.assertEqual(child_kinds.count("reducer"), 1)
        raw_provenance = json.loads(raw_text)
        self.assertEqual(len(raw_provenance["answer_reducers"]), 1)
        self.assertEqual(
            len(raw_provenance["leaf_responses"]),
            fragments["core_unit_count"],
        )
        for call in chat_mock.await_args_list:
            self.assertEqual(
                call.kwargs["output_token_policy"],
                OutputTokenPolicy.MODEL_MAX,
            )
            self.assertNotIn("document_id", call.kwargs)
            self.assertNotIn("audit_checkpoint", call.kwargs)
            self.assertNotIn("resume_checkpoint", call.kwargs)
            self.assertNotIn("max_continuations", call.kwargs)

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

    async def test_recovery_final_uses_one_bounded_contract_repair(
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
            result = await _analysis_critic_artifact(
                "run-final-primary-only",
                iteration=3,
                payload=payload,
                recovery_final=True,
            )
        self.assertEqual(result["verdict"], "pass")
        repair.assert_awaited_once()

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
        self.assertEqual(
            request["output_token_policy"],
            OutputTokenPolicy.MODEL_MAX,
        )
        self.assertIs(request["retry_response_contract_errors"], False)
        self.assertIs(request["retry_transport_errors"], False)
        self.assertNotIn("document_id", request)
        self.assertNotIn("audit_checkpoint", request)
        self.assertNotIn("resume_checkpoint", request)
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
            compact["affected_answer_evidence"][0]["raw_answer"],
            "x" * 20_000,
        )
        self.assertFalse(
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

    async def test_repair_preserves_all_referenced_evidence_without_cap(
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
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ) as chat_mock:
            repaired, _raw, _usage = await repair_analysis_review(
                payload,
                incomplete,
                iteration=1,
                validation_errors=["incomplete"],
            )

        self.assertEqual(repaired["verdict"], "pass")
        request = json.loads(
            chat_mock.await_args.kwargs["messages"][1]["content"]
        )
        context = request["repair_context"]
        self.assertEqual(
            [
                item["answer_id"]
                for item in context["affected_answer_evidence"]
            ],
            answer_ids,
        )
        self.assertEqual(
            context["evidence_limits"]["max_included_answers"],
            None,
        )
        self.assertEqual(
            context["evidence_limits"]["omitted_referenced_answer_ids"],
            [],
        )

    async def test_repair_preserves_full_affected_raw_answer(
        self,
    ) -> None:
        raw_answer = "Evidence " * 1_000
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
                    "raw_answer": raw_answer,
                    "raw_answer_included": True,
                    "raw_answer_char_count": len(raw_answer),
                    "raw_answer_truncated": False,
                }
            ],
        }
        response = SimpleNamespace(
            parsed=_critic_review("pass"),
            text="{}",
            usage={},
        )
        with patch(
            "app.services.analysis_critic.chat",
            new_callable=AsyncMock,
            return_value=response,
        ) as chat_mock:
            repaired, _raw, _usage = await repair_analysis_review(
                payload,
                _critic_review("revise"),
                iteration=1,
                validation_errors=["incomplete"],
            )

        self.assertEqual(repaired["verdict"], "pass")
        request = json.loads(
            chat_mock.await_args.kwargs["messages"][1]["content"]
        )
        self.assertEqual(
            request["repair_context"]["affected_answer_evidence"][0][
                "raw_answer"
            ],
            raw_answer,
        )
        self.assertFalse(
            request["repair_context"]["affected_answer_evidence"][0][
                "repair_raw_truncated"
            ]
        )

    async def test_repair_map_reduce_losslessly_covers_several_huge_answers(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=3, raw_chars=90_000)
        answer_ids = [1, 2, 3]
        incomplete = _critic_review(
            "revise",
            anomalies=[
                {
                    "code": "annotation_evidence_mismatch",
                    "severity": "important",
                    "finding": "Нужно перепроверить три длинных ответа.",
                    "answer_ids": answer_ids,
                    "entities": ["Campaign 360"],
                }
            ],
        )
        raw_by_answer = {
            int(answer["answer_id"]): str(answer["raw_answer"])
            for answer in payload["answers"]
        }
        cores_by_answer: dict[int, list[tuple[int, str, str]]] = {
            answer_id: [] for answer_id in answer_ids
        }
        request_sizes: list[int] = []
        audited: list[dict] = []

        async def audit_sink(event: dict) -> None:
            audited.append(copy.deepcopy(event))

        async def repair_response(**kwargs):
            request_sizes.append(
                len(
                    json.dumps(
                        kwargs["messages"],
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
            )
            command = json.loads(kwargs["messages"][1]["content"])
            if command.get("repair_stage") == "fragment_leaf":
                evidence = command["repair_context"][
                    "affected_answer_evidence"
                ][0]
                fragment = evidence["repair_fragment"]
                context = evidence["raw_answer"]
                core = context[
                    fragment["core_start_in_context"] : fragment[
                        "core_end_in_context"
                    ]
                ]
                cores_by_answer[int(fragment["answer_id"])].append(
                    (
                        int(fragment["unit_index"]),
                        str(fragment["unit_id"]),
                        core,
                    )
                )
                self.assertEqual(
                    command["repair_context"]["repair_partition"][
                        "assigned_unit_id"
                    ],
                    fragment["unit_id"],
                )
                self.assertEqual(
                    len(
                        command["repair_context"][
                            "affected_answer_evidence"
                        ]
                    ),
                    1,
                )
            review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "_aiv_transport": {
                        "status": "succeeded",
                        "output_complete": True,
                    },
                },
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 81_920,
                    "max_completion_tokens": 8_192,
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=repair_response,
            ) as chat_mock,
        ):
            repaired, raw_text, usage = await repair_analysis_review(
                payload,
                incomplete,
                iteration=1,
                validation_errors=["synthetic long repair"],
                audit_sink=audit_sink,
            )

        self.assertEqual(repaired["verdict"], "pass")
        self.assertGreater(chat_mock.await_count, len(answer_ids))
        self.assertTrue(request_sizes)
        self.assertTrue(all(size <= 73_728 for size in request_sizes))
        for answer_id in answer_ids:
            ordered = sorted(cores_by_answer[answer_id])
            self.assertGreater(len(ordered), 1)
            self.assertEqual(
                "".join(core for _index, _unit_id, core in ordered),
                raw_by_answer[answer_id],
            )
            self.assertEqual(
                len({unit_id for _index, unit_id, _core in ordered}),
                len(ordered),
            )
        provenance = usage["_aiv_critic_repair_map_reduce"]
        self.assertEqual(provenance["status"], "completed")
        self.assertEqual(
            provenance["manifest"]["affected_answer_ids"],
            answer_ids,
        )
        self.assertTrue(
            provenance["manifest"]["exact_fragment_accounting"]
        )
        raw_ledger = json.loads(raw_text)
        self.assertEqual(raw_ledger["status"], "completed")
        self.assertEqual(
            len(raw_ledger["provider_calls"]),
            provenance["provider_call_count"],
        )
        self.assertTrue(audited)
        self.assertTrue(all(event["status"] == "completed" for event in audited))
        self.assertIn("repair_fragment_leaf", {event["kind"] for event in audited})
        self.assertIn(
            "repair_answer_reducer",
            {event["kind"] for event in audited},
        )
        self.assertIn(
            "repair_corpus_reducer",
            {event["kind"] for event in audited},
        )

    async def test_repair_map_failure_is_audited_and_returns_verdict_floor_block(
        self,
    ) -> None:
        payload = _large_critic_payload(answer_count=2, raw_chars=80_000)
        answer_ids = [1, 2]
        incomplete = _critic_review(
            "revise",
            anomalies=[
                {
                    "code": "annotation_evidence_mismatch",
                    "severity": "important",
                    "finding": "Исходная важная находка не должна исчезнуть.",
                    "answer_ids": answer_ids,
                    "entities": ["Campaign 360"],
                }
            ],
        )
        paid_partial = SimpleNamespace(
            parsed=None,
            text='{"verdict":"revise","summary":"paid partial"',
            usage={"prompt_tokens": 50, "completion_tokens": 40},
            transport={"status": "succeeded", "output_complete": False},
        )
        audited: list[dict] = []

        async def audit_sink(event: dict) -> None:
            audited.append(copy.deepcopy(event))

        async def repair_response(**kwargs):
            command = json.loads(kwargs["messages"][1]["content"])
            if command.get("repair_stage") == "fragment_leaf":
                fragment = command["repair_context"][
                    "affected_answer_evidence"
                ][0]["repair_fragment"]
                if (
                    int(fragment["answer_id"]) == 1
                    and int(fragment["unit_index"]) == 0
                ):
                    raise OpenRouterResponseContractError(
                        "synthetic paid repair failure",
                        result=paid_partial,
                    )
                await asyncio.sleep(0.01)
            review = _critic_review("pass")
            return SimpleNamespace(
                parsed=review,
                text=json.dumps(review, ensure_ascii=False),
                usage={"total_tokens": 15},
            )

        with (
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value={
                    "context_length": 81_920,
                    "max_completion_tokens": 8_192,
                    "resolution": "test",
                },
            ),
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=repair_response,
            ),
        ):
            repaired, raw_text, usage = await repair_analysis_review(
                payload,
                incomplete,
                iteration=1,
                validation_errors=["synthetic paid failure"],
                audit_sink=audit_sink,
            )

        self.assertEqual(repaired["verdict"], "block")
        self.assertTrue(
            any(
                anomaly["finding"]
                == "Исходная важная находка не должна исчезнуть."
                for anomaly in repaired["anomalies"]
            )
        )
        self.assertEqual(
            usage["_aiv_critic_repair_map_reduce"]["status"],
            "fail_closed",
        )
        self.assertTrue(any(event["status"] == "failed" for event in audited))
        self.assertTrue(
            any(event["status"] == "completed" for event in audited)
        )
        failed = next(event for event in audited if event["status"] == "failed")
        self.assertEqual(failed["raw_text"], paid_partial.text)
        self.assertEqual(failed["usage"]["prompt_tokens"], 50)
        ledger = json.loads(raw_text)
        self.assertEqual(ledger["status"], "fail_closed")
        self.assertTrue(
            any(
                call["raw_text"] == paid_partial.text
                for call in ledger["provider_calls"]
            )
        )

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

    async def test_output_limited_verdict_cannot_authorize_subset_tail_repair(
        self,
    ) -> None:
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
        early_anomaly = {
            "code": "generic_term_leakage",
            "severity": "important",
            "finding": "Ранняя находка присутствует в префиксе.",
            "answer_ids": [11],
            "entities": ["Campaign 360"],
        }
        tail_anomaly = {
            "code": "fabricated_evidence",
            "severity": "critical",
            "finding": "Материальная находка была в отрезанном хвосте.",
            "answer_ids": [12],
            "entities": ["Tail entity"],
        }
        early_policy = {
            "action": "require_target_attribution",
            "entity_name": "Campaign 360",
            "alias": None,
            "reason": "Ранняя policy присутствует в префиксе.",
            "answer_ids": [11],
        }
        tail_policy = {
            "action": "exclude_portfolio_entity",
            "entity_name": "Tail entity",
            "alias": None,
            "reason": "Материальная policy была в отрезанном хвосте.",
            "answer_ids": [12],
        }
        complete_intended_review = _critic_review(
            "revise",
            anomalies=[early_anomaly, tail_anomaly],
            adjustments=[early_policy, tail_policy],
            guidance="Переразметить обе затронутые сущности.",
        )
        complete_text = json.dumps(
            complete_intended_review,
            ensure_ascii=False,
        )
        tail_marker = '"code": "fabricated_evidence"'
        partial_text = complete_text[: complete_text.index(tail_marker)]
        self.assertIn("Ранняя находка", partial_text)
        self.assertNotIn("Материальная находка", partial_text)
        self.assertNotIn("Материальная policy", partial_text)
        limited = OpenRouterOutputLimitError(
            "OpenRouter response hit the output limit",
            result=SimpleNamespace(
                text=partial_text,
                usage=primary_usage,
                transport=transport,
            ),
        )
        unsafe_subset_repair = _critic_review(
            "revise",
            anomalies=[early_anomaly],
            adjustments=[early_policy],
            guidance="Переразметить только раннюю сущность.",
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
                return_value=(unsafe_subset_repair, "{}", {}),
            ) as repair_mock,
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ),
        ):
            parsed, raw, usage = await review_analysis(payload, iteration=1)

        self.assertEqual(parsed["verdict"], "block")
        self.assertEqual(parsed["anomalies"], [])
        self.assertEqual(parsed["policy_adjustments"], [])
        self.assertEqual(parsed["annotation_guidance"], "")
        self.assertEqual(chat_mock.await_count, 4)
        repair_mock.assert_not_awaited()
        primary_request = chat_mock.await_args.kwargs
        self.assertEqual(
            primary_request["output_token_policy"],
            OutputTokenPolicy.MODEL_MAX,
        )
        self.assertIs(primary_request["retry_response_contract_errors"], False)
        self.assertIs(primary_request["retry_transport_errors"], False)
        self.assertNotIn("document_id", primary_request)
        self.assertNotIn("audit_checkpoint", primary_request)
        self.assertNotIn("resume_checkpoint", primary_request)

        ledger = json.loads(raw)
        self.assertEqual(ledger["status"], "fail_closed")
        self.assertEqual(ledger["failure"], "output_limit")
        self.assertEqual(
            ledger["coverage"]["semantic_authority"],
            "none",
        )
        self.assertFalse(ledger["coverage"]["anomalies_complete"])
        self.assertFalse(
            ledger["coverage"]["policy_adjustments_complete"]
        )
        self.assertEqual(
            ledger["primary_partial"]["_partial_response"],
            partial_text,
        )
        self.assertEqual(
            ledger["primary_partial"]["_partial_response_sha256"],
            hashlib.sha256(partial_text.encode("utf-8")).hexdigest(),
        )
        self.assertFalse(
            ledger["primary_partial"]["_partial_response_truncated"]
        )
        self.assertEqual(usage["total_tokens"], 80_400)
        self.assertEqual(len(usage["_aiv_critic_attempts"]), 4)
        self.assertEqual(
            usage["_aiv_critic_attempts"][0]["kind"],
            "primary_output_limited",
        )
        self.assertTrue(
            all(
                item["kind"] == "output_limit_decision_shard"
                and item["status"] == "failed"
                for item in usage["_aiv_critic_attempts"][1:]
            )
        )
        self.assertEqual(
            usage["_aiv_critic_contract"]["recovered_from"],
            "output_limit_fail_closed",
        )
        self.assertEqual(
            usage["_aiv_critic_contract"]["semantic_verdict_status"],
            "pending_deterministic_validation",
        )
        self.assertEqual(
            usage["_aiv_critic_output_limit"]["semantic_authority"],
            "none",
        )

    async def test_output_limit_replans_into_complete_decision_shards(
        self,
    ) -> None:
        transport = {
            "status": "succeeded",
            "output_complete": False,
            "output_limited": True,
            "finish_reason": "length",
        }
        limited = OpenRouterOutputLimitError(
            "primary output limited",
            result=SimpleNamespace(
                text='{"verdict":"revise","anomalies":[',
                usage={"total_tokens": 100, "_aiv_transport": transport},
                transport=transport,
            ),
        )
        anomaly = {
            "code": "fabricated_evidence",
            "severity": "critical",
            "finding": "Хвостовая ошибка найдена отдельным shard.",
            "answer_ids": [11],
            "entities": ["Tail entity"],
        }
        adjustment = {
            "action": "exclude_portfolio_entity",
            "entity_name": "Tail entity",
            "alias": None,
            "reason": "Нет подтверждённой принадлежности.",
            "answer_ids": [11],
        }
        anomaly_review = _critic_review(
            "revise", anomalies=[anomaly]
        )
        policy_review = _critic_review(
            "revise", adjustments=[adjustment]
        )
        conclusion_review = _critic_review(
            "revise",
            guidance="Переразметить ответ 11.",
        )
        conclusion_review["acceptance_checks"] = [
            "Сущность исключена без доказанной связи."
        ]
        responses = [
            limited,
            SimpleNamespace(
                parsed=anomaly_review,
                text=json.dumps(anomaly_review, ensure_ascii=False),
                usage={"total_tokens": 10},
            ),
            SimpleNamespace(
                parsed=policy_review,
                text=json.dumps(policy_review, ensure_ascii=False),
                usage={"total_tokens": 11},
            ),
            SimpleNamespace(
                parsed=conclusion_review,
                text=json.dumps(conclusion_review, ensure_ascii=False),
                usage={"total_tokens": 12},
            ),
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
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                side_effect=responses,
            ) as chat_mock,
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ),
        ):
            parsed, raw, usage = await review_analysis(payload, iteration=1)

        self.assertEqual(chat_mock.await_count, 4)
        self.assertEqual(parsed["verdict"], "revise")
        self.assertEqual(parsed["anomalies"], [anomaly])
        self.assertEqual(parsed["policy_adjustments"], [adjustment])
        self.assertEqual(parsed["annotation_guidance"], "Переразметить ответ 11.")
        self.assertEqual(
            json.loads(raw)["status"],
            "recovered_by_complete_decision_shards",
        )
        self.assertEqual(usage["total_tokens"], 133)
        self.assertEqual(
            usage["_aiv_critic_output_limit"]["semantic_authority"],
            "complete_decision_shard_union",
        )
        self.assertTrue(
            usage["_aiv_critic_output_limit"]["decision_shards_complete"]
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
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ),
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

    def test_primary_critic_payload_contains_the_full_nonempty_raw_corpus(
        self,
    ) -> None:
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
        self.assertEqual(len(included), len(rows))
        self.assertEqual(
            payload["raw_evidence_selection"]["included_raw_count"],
            len(rows),
        )
        self.assertEqual(
            payload["raw_evidence_selection"]["included_answer_ids"],
            list(range(1, 82)),
        )
        self.assertEqual(
            payload["raw_evidence_selection"]["strategy"],
            "complete_raw_corpus_v4",
        )
        self.assertIsNone(
            payload["raw_evidence_selection"]["optional_char_window"]
        )
        self.assertIsNone(
            payload["raw_evidence_selection"]["max_raw_answers"]
        )
        self.assertEqual(
            payload["raw_evidence_selection"]["omitted_warning_answer_ids"],
            [],
        )
        full_raw_chars = sum(len(row["answer_text"]) for row in rows)
        sent_raw_chars = sum(
            len(answer["raw_answer"]) for answer in payload["answers"]
        )
        self.assertEqual(sent_raw_chars, full_raw_chars)
        self.assertEqual(
            sent_raw_chars,
            payload["raw_evidence_selection"]["included_raw_chars"],
        )
        self.assertEqual(
            [answer["raw_answer"] for answer in payload["answers"]],
            [row["answer_text"] for row in rows],
        )
        self.assertTrue(
            all(
                answer["raw_answer_manifest"]["source_sha256"]
                == answer["raw_answer_sha256"]
                for answer in payload["answers"]
            )
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

    def test_warning_linked_raw_is_never_omitted_from_large_corpus(self) -> None:
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
        self.assertEqual(selection["included_raw_count"], len(rows))
        self.assertEqual(
            selection["included_raw_chars"],
            sum(len(row["answer_text"]) for row in rows),
        )
        self.assertEqual(selection["omitted_warning_answer_ids"], [])
        self.assertEqual(selection["missing_warning_answer_ids"], [])
        self.assertFalse(selection["insufficient_warning_evidence_requires_block"])
        self.assertTrue(
            all(answer["raw_answer_included"] for answer in payload["answers"])
        )
        self.assertTrue(
            all(
                answer["raw_answer"] == row["answer_text"]
                for answer, row in zip(payload["answers"], rows, strict=True)
            )
        )
        errors = _critic_review_errors(
            _critic_review("pass"),
            payload=payload,
        )
        self.assertFalse(
            any("warning-linked raw answers" in error for error in errors)
        )

    def test_oversized_raw_is_partition_manifested_without_truncation(
        self,
    ) -> None:
        raw_answer = "x" * 24_001
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=[{**ROWS[0], "answer_text": raw_answer}],
            metrics=METRICS,
            policy_history=[],
        )

        answer = payload["answers"][0]
        self.assertEqual(answer["raw_answer"], raw_answer)
        self.assertTrue(answer["raw_answer_included"])
        self.assertFalse(answer["raw_answer_truncated"])
        self.assertGreater(answer["raw_answer_manifest"]["unit_count"], 1)
        self.assertEqual(
            answer["raw_answer_manifest"]["source_sha256"],
            hashlib.sha256(raw_answer.encode("utf-8")).hexdigest(),
        )
        self.assertFalse(
            _critic_review_errors(
                _critic_review("pass"),
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
        with (
            patch(
                "app.services.analysis_critic.chat",
                new_callable=AsyncMock,
                return_value=response,
            ),
            patch(
                "app.services.analysis_critic.model_output_envelope",
                new_callable=AsyncMock,
                return_value=_model_envelope(),
            ),
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
        self.assertEqual(
            result["fallback"]["kind"],
            "deterministic_degraded_advisory",
        )
        self.assertEqual(result["anomalies"], still_invalid["anomalies"])
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
            "deterministic_degraded_advisory",
        )
        self.assertEqual(result["anomalies"], incomplete["anomalies"])
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
            "deterministic_degraded_advisory",
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
        self.assertEqual(gate["quality_state"], "complete")
        self.assertEqual(gate["reason_codes"], [])
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

    async def test_critic_provider_failure_publishes_degraded_gate(self) -> None:
        gate = {
            "passed": True,
            "quality_state": "degraded",
            "reason_codes": ["critic_provider_schema_or_cache_unavailable"],
        }
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=OpenRouterError("provider 503"),
            ),
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
        ):
            _catalog, returned_rows, returned_metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-critic-provider-outage",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
            )

        self.assertIs(returned_rows, ROWS)
        self.assertIs(returned_metrics, METRICS)
        self.assertEqual(returned_gate, gate)
        self.assertTrue(gate_mock.await_args.kwargs["passed"])
        self.assertEqual(gate_mock.await_args.kwargs["quality_state"], "degraded")
        self.assertEqual(
            gate_mock.await_args.kwargs["reason_codes"],
            ["critic_provider_schema_or_cache_unavailable"],
        )
        annotate_mock.assert_not_awaited()

    async def test_critic_provider_failure_cannot_hide_confirmed_integrity_error(
        self,
    ) -> None:
        warning = {
            "code": "target_mention_false_negative",
            "severity": "important",
            "finding": "Raw evidence contains the exact target alias.",
            "answer_ids": [11],
            "entities": ["Example"],
        }
        with (
            patch(
                "app.services.analyzer._deterministic_annotation_warnings",
                return_value=[warning],
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=OpenRouterError("provider 503"),
            ),
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value={"passed": False},
            ) as gate_mock,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
        ):
            with self.assertRaises(_ConfirmedCriticIntegrityBlock) as caught:
                await _run_analysis_critic_loop(
                    "run-critic-provider-outage-confirmed-anomaly",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )

        self.assertIn(
            "annotation_integrity:target_mention_false_negative",
            caught.exception.reason_codes,
        )
        gate_mock.assert_awaited_once()
        self.assertFalse(gate_mock.await_args.kwargs["passed"])
        self.assertEqual(
            gate_mock.await_args.kwargs["critic_outcome"],
            "confirmed_integrity_block",
        )

    async def test_critic_provider_failure_cannot_hide_stale_annotation_lineage(
        self,
    ) -> None:
        stale_rows = copy.deepcopy(ROWS)
        stale_rows[0]["annotation"]["_answer_sha256"] = "0" * 64
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=OpenRouterError("provider 503"),
            ),
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value={"passed": False},
            ) as gate_mock,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
        ):
            with self.assertRaises(_ConfirmedCriticIntegrityBlock) as caught:
                await _run_analysis_critic_loop(
                    "run-critic-provider-outage-stale-lineage",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=stale_rows,
                    metrics=METRICS,
                )

        self.assertIn(
            "corpus_lineage:annotation_raw_hash_mismatch",
            caught.exception.reason_codes,
        )
        gate_mock.assert_awaited_once()
        self.assertFalse(gate_mock.await_args.kwargs["passed"])

    async def test_zero_eligible_evidence_still_blocks_degraded_gate(self) -> None:
        ineligible_rows = [copy.deepcopy(ROWS[0])]
        ineligible_rows[0]["metric_eligible"] = False
        ineligible_rows[0]["metric_evidence_state"] = "provider_limited_prefix"
        ineligible_rows[0]["metric_limitation"] = "provider_output_limit"
        with patch(
            "app.services.analyzer._save_artifact",
            new_callable=AsyncMock,
        ):
            with self.assertRaises(_ConfirmedCriticIntegrityBlock) as caught:
                await _save_critic_gate(
                    "run-zero-eligible",
                    passed=True,
                    iteration=1,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ineligible_rows,
                    metrics=METRICS,
                    policy_history=[],
                    reason="Critic unavailable.",
                    quality_state="degraded",
                    reason_codes=["critic_provider_unavailable"],
                )

        self.assertTrue(
            any(
                code.startswith("panel_integrity:")
                for code in caught.exception.reason_codes
            )
        )

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

    async def test_second_revise_publishes_degraded_without_a_second_repair(
        self,
    ) -> None:
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
                return_value={"passed": True, "quality_state": "degraded"},
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
            _catalog, returned_rows, returned_metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-second-revise",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
            )

        self.assertEqual(critic_mock.await_count, MAX_CRITIC_ITERATIONS)
        annotate_mock.assert_awaited_once()
        metrics_mock.assert_called_once()
        artifact_mock.assert_awaited_once()
        gate_mock.assert_awaited_once()
        self.assertTrue(gate_mock.await_args.kwargs["passed"])
        self.assertEqual(gate_mock.await_args.kwargs["quality_state"], "degraded")
        self.assertEqual(
            gate_mock.await_args.kwargs["critic_outcome"],
            "non_convergent",
        )
        self.assertEqual(
            gate_mock.await_args.kwargs["iteration"],
            MAX_CRITIC_ITERATIONS,
        )
        self.assertIs(returned_rows, ROWS)
        self.assertIs(returned_metrics, METRICS)
        self.assertTrue(returned_gate["passed"])

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

    async def test_r1_target_receipt_survives_different_r2_recovery_action(
        self,
    ) -> None:
        base_row = copy.deepcopy(MAKAR_ROW)
        base_row["annotation"].update(
            {
                "_annotation_version": ANNOTATION_VERSION,
                "_answer_sha256": hashlib.sha256(
                    MAKAR_RAW.encode("utf-8")
                ).hexdigest(),
                "_answer_model": MAKAR_ROW["model"],
                "_annotation_input_sha256": "base-makar-context",
            }
        )
        r1_adjustment = {
            "action": "require_literal_target_mention_evidence",
            "entity_name": "Makarska Tattoo & Piercing Studio",
            "alias": "Tattoo & Piercing Makarska",
            "reason": "Исправить доказанное ложное отрицание.",
            "answer_ids": [580],
        }
        r2_adjustment = {
            "action": "require_literal_brand_knowledge_evidence",
            "entity_name": "Makarska Tattoo & Piercing Studio",
            "alias": None,
            "reason": "Отдельно перепроверить факты знания бренда.",
            "answer_ids": [580],
        }
        r2_anomaly = {
            "code": "annotation_evidence_mismatch",
            "severity": "important",
            "finding": "Факты о бренде требуют повторной разметки.",
            "answer_ids": [580],
            "entities": ["Makarska Tattoo & Piercing Studio"],
        }
        recovered_row = copy.deepcopy(base_row)
        recovered_row["annotation"].update(
            {
                "target_mentioned": True,
                "target_role": "mentioned",
                "uncertainties": ["targeted recovery changed state"],
            }
        )
        recovered_metrics = {
            "providers": [{"provider_key": "perplexity"}],
            "target_visibility": {"web": {"mention_count": 1}},
        }
        required_checks = {
            CHECK_RAW_CORPUS_UNCHANGED,
            CHECK_DERIVED_METRICS_RECOMPUTED,
            CHECK_CRITIC_GATE_PASSED,
        }
        plan = SimpleNamespace(
            epoch=3,
            plan_digest="d" * 64,
            decision={
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "rationale": "Повторить разметку только строки 580.",
                "guidance": "Проверь буквальные факты только в answer_id 580.",
                "target_answer_ids": [580],
                "invalidate_artifact_keys": [],
                "acceptance_checks": sorted(required_checks),
            },
        )
        gate = {"passed": True, "iteration": 3}

        with (
            patch(
                "app.services.analyzer.settings.PIPELINE_ORCHESTRATOR_ENABLED",
                True,
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _critic_review(
                        "revise",
                        adjustments=[r1_adjustment],
                        guidance="Исправить упоминание exact target.",
                    ),
                    _critic_review(
                        "revise",
                        adjustments=[r2_adjustment],
                        anomalies=[r2_anomaly],
                    ),
                    _critic_review("pass"),
                ],
            ),
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
            ),
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value=gate,
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
                side_effect=[[base_row], [recovered_row]],
            ),
            patch(
                "app.services.analyzer._compute_metrics",
                side_effect=[{"providers": []}, recovered_metrics],
            ),
        ):
            _catalog, rows, metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-makar-receipt-recovery",
                    profile=MAKAR_PROFILE,
                    catalog=MAKAR_CATALOG,
                    rows=[base_row],
                    metrics={"providers": []},
                )
            )

        self.assertEqual(rows, [recovered_row])
        self.assertEqual(metrics, recovered_metrics)
        self.assertEqual(returned_gate, gate)
        self.assertEqual(annotate.await_count, 2)
        r1_receipts = annotate.await_args_list[0].kwargs[
            "target_mention_receipts"
        ]
        recovery_receipts = annotate.await_args_list[1].kwargs[
            "target_mention_receipts"
        ]
        self.assertEqual(len(r1_receipts), 1)
        self.assertEqual(recovery_receipts, r1_receipts)
        self.assertEqual(recovery_receipts[0]["answer_id"], 580)
        provenance = annotate.await_args_list[1].kwargs[
            "annotation_repair_provenance"
        ]
        self.assertEqual(
            provenance["target_mention_receipts_sha256"],
            stable_digest(recovery_receipts),
        )
        self.assertEqual(
            provenance["recovery_policy_step"][
                "target_mention_receipts_sha256"
            ],
            stable_digest(recovery_receipts),
        )

    async def test_r2_targeted_repair_keeps_recovered_state_when_final_critic_blocks(
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
                return_value={"passed": True, "quality_state": "degraded"},
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
            _catalog, returned_rows, returned_metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-fable-final-block",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
            )

        self.assertEqual(critic.await_count, 3)
        finish.assert_awaited_once()
        self.assertTrue(finish.await_args.kwargs["succeeded"])
        gate.assert_awaited_once()
        self.assertTrue(gate.await_args.kwargs["passed"])
        self.assertEqual(gate.await_args.kwargs["quality_state"], "degraded")
        self.assertEqual(gate.await_args.kwargs["rows"], recovered_rows)
        self.assertTrue(gate.await_args.kwargs["metrics"]["recovered"])
        self.assertEqual(returned_rows, recovered_rows)
        self.assertTrue(returned_metrics["recovered"])
        self.assertTrue(returned_gate["passed"])

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
                return_value={"passed": True, "quality_state": "degraded"},
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
            _catalog, returned_rows, returned_metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-fable-stop",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
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
        self.assertIs(returned_rows, ROWS)
        self.assertIs(returned_metrics, METRICS)
        self.assertTrue(returned_gate["passed"])
        self.assertTrue(gate.await_args.kwargs["passed"])
        self.assertEqual(gate.await_args.kwargs["quality_state"], "degraded")

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

    def test_targeted_recovery_payload_keeps_full_raw_beyond_old_limit(
        self,
    ) -> None:
        raw_answer = "x" * 24_001
        oversized_rows = [
            {
                **ROWS[0],
                "answer_text": raw_answer,
            }
        ]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=oversized_rows,
            metrics=METRICS,
            policy_history=[],
            mandatory_raw_answer_ids={11},
        )

        self.assertEqual(
            payload["raw_evidence_selection"]["mandatory_answer_ids"],
            [11],
        )
        self.assertEqual(
            payload["raw_evidence_selection"]["omitted_mandatory_answer_ids"],
            [],
        )
        self.assertEqual(payload["answers"][0]["raw_answer"], raw_answer)
        self.assertIsNone(
            payload["raw_evidence_selection"]["optional_char_window"]
        )

    def test_coverage_only_block_becomes_degraded_advisory(
        self,
    ) -> None:
        degraded_rows = [copy.deepcopy(ROWS[0]), copy.deepcopy(ROWS[0])]
        degraded_rows[1]["answer_id"] = 12
        degraded_rows[1]["provider_key"] = "gemini"
        degraded_rows[1]["model"] = "google/gemini-3.1-pro-preview"
        degraded_rows[1]["metric_eligible"] = False
        degraded_rows[1]["metric_evidence_state"] = "provider_limited_prefix"
        degraded_rows[1]["metric_limitation"] = "provider_output_limit"
        degraded_rows[1]["annotation"]["_answer_model"] = degraded_rows[1]["model"]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=degraded_rows,
            metrics=METRICS,
            policy_history=[],
        )
        review = _critic_review("block")
        warning_code = payload["panel_metric_coverage_admission"]["warning_codes"][0]
        review["acceptance_checks"] = [f"coverage_warning_ack:{warning_code}"]

        errors = _critic_review_validation_errors(review, payload=payload)
        fallback = _deterministic_critic_fallback_review(
            payload,
            review,
            validation_errors=errors,
        )

        self.assertTrue(payload["panel_metric_coverage_admission"]["allowed"])
        self.assertEqual(
            payload["panel_metric_coverage_admission"]["quality_state"],
            "degraded",
        )
        self.assertIn("block contains no critical/important anomalies", errors)
        self.assertEqual(fallback["verdict"], "block")
        self.assertEqual(
            fallback["fallback"]["kind"],
            "deterministic_degraded_advisory",
        )
        self.assertIn(
            "critic_unconfirmed_block",
            fallback["fallback"]["reason_codes"],
        )

    async def test_coverage_only_block_publishes_degraded_gate(self) -> None:
        degraded_rows = [copy.deepcopy(ROWS[0]), copy.deepcopy(ROWS[0])]
        degraded_rows[1]["answer_id"] = 12
        degraded_rows[1]["provider_key"] = "gemini"
        degraded_rows[1]["model"] = "google/gemini-3.1-pro-preview"
        degraded_rows[1]["metric_eligible"] = False
        degraded_rows[1]["metric_evidence_state"] = "provider_limited_prefix"
        degraded_rows[1]["metric_limitation"] = "provider_output_limit"
        degraded_rows[1]["annotation"]["_answer_model"] = degraded_rows[1]["model"]
        coverage_block = _critic_review("block")
        gate = {"passed": True, "quality_state": "degraded"}
        with (
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                return_value=coverage_block,
            ),
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
                return_value=gate,
            ) as gate_mock,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
        ):
            _catalog, returned_rows, _metrics, returned_gate = (
                await _run_analysis_critic_loop(
                    "run-coverage-only-advisory",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=degraded_rows,
                    metrics=METRICS,
                )
            )

        self.assertIs(returned_rows, degraded_rows)
        self.assertEqual(returned_gate, gate)
        self.assertTrue(gate_mock.await_args.kwargs["passed"])
        self.assertEqual(gate_mock.await_args.kwargs["quality_state"], "degraded")
        self.assertEqual(
            gate_mock.await_args.kwargs["critic_outcome"],
            "advisory_block",
        )

    async def test_coverage_only_block_gets_one_bounded_repair_to_pass(
        self,
    ) -> None:
        degraded_rows = [copy.deepcopy(ROWS[0]), copy.deepcopy(ROWS[0])]
        degraded_rows[1]["answer_id"] = 12
        degraded_rows[1]["provider_key"] = "gemini"
        degraded_rows[1]["model"] = "google/gemini-3.1-pro-preview"
        degraded_rows[1]["metric_eligible"] = False
        degraded_rows[1]["metric_evidence_state"] = "provider_limited_prefix"
        degraded_rows[1]["metric_limitation"] = "provider_output_limit"
        degraded_rows[1]["annotation"]["_answer_model"] = degraded_rows[1]["model"]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=degraded_rows,
            metrics=METRICS,
            policy_history=[],
        )
        warning_code = payload["panel_metric_coverage_admission"]["warning_codes"][0]
        invalid_block = _critic_review("block")
        invalid_block["acceptance_checks"] = [
            f"coverage_warning_ack:{warning_code}"
        ]
        repaired_pass = _critic_review("pass")
        repaired_pass["acceptance_checks"].append(
            f"coverage_warning_ack:{warning_code}"
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
                return_value=(invalid_block, "{}", {"total_tokens": 10}),
            ),
            patch(
                "app.services.analyzer.repair_analysis_review",
                new_callable=AsyncMock,
                return_value=(repaired_pass, "{}", {"total_tokens": 5}),
            ) as repair_mock,
        ):
            result = await _analysis_critic_artifact(
                "run-coverage-repair",
                iteration=1,
                payload=payload,
            )

        self.assertEqual(result, repaired_pass)
        repair_mock.assert_awaited_once()
        self.assertIn(
            "block contains no critical/important anomalies",
            repair_mock.await_args.kwargs["validation_errors"],
        )

    def test_malformed_non_coverage_block_becomes_degraded_advisory(self) -> None:
        degraded_rows = [copy.deepcopy(ROWS[0]), copy.deepcopy(ROWS[0])]
        degraded_rows[1]["answer_id"] = 12
        degraded_rows[1]["provider_key"] = "gemini"
        degraded_rows[1]["model"] = "google/gemini-3.1-pro-preview"
        degraded_rows[1]["metric_eligible"] = False
        degraded_rows[1]["metric_evidence_state"] = "provider_limited_prefix"
        degraded_rows[1]["metric_limitation"] = "provider_output_limit"
        degraded_rows[1]["annotation"]["_answer_model"] = degraded_rows[1]["model"]
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=degraded_rows,
            metrics=METRICS,
            policy_history=[],
        )
        review = _critic_review("block")
        errors = _critic_review_validation_errors(review, payload=payload)

        fallback = _deterministic_critic_fallback_review(
            payload,
            review,
            validation_errors=errors,
        )

        self.assertEqual(fallback["verdict"], "block")
        self.assertEqual(
            fallback.get("fallback", {}).get("kind"),
            "deterministic_degraded_advisory",
        )

    def test_model_only_missing_as_zero_cannot_hard_block(self) -> None:
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=[copy.deepcopy(ROWS[0])],
            metrics=METRICS,
            policy_history=[],
        )
        review = _critic_review(
            "block",
            anomalies=[
                {
                    "code": "missing_data_as_zero",
                    "severity": "critical",
                    "finding": (
                        "Недоступная ячейка попала в опубликованный числитель "
                        "как нулевое наблюдение."
                    ),
                    "answer_ids": [11],
                    "entities": ["Campaign 360"],
                }
            ],
        )

        self.assertIn(
            "block is not backed by a code-owned confirmed integrity anomaly",
            _critic_review_validation_errors(review, payload=payload),
        )

    def test_context_only_truncated_prefix_is_not_an_integrity_block(self) -> None:
        limited_rows = [copy.deepcopy(ROWS[0])]
        limited_rows[0]["metric_eligible"] = False
        limited_rows[0]["context_eligible"] = True
        limited_rows[0]["metric_evidence_state"] = "provider_limited_prefix"
        limited_rows[0]["metric_limitation"] = "provider_output_limit"
        payload = _critic_payload(
            profile=PROFILE,
            catalog=CATALOG,
            rows=limited_rows,
            metrics=METRICS,
            policy_history=[],
        )
        payload["answers"][0]["raw_answer_truncated"] = True

        errors = _critic_review_validation_errors(
            _critic_review("pass"),
            payload=payload,
        )

        self.assertFalse(
            any("truncated raw answers require block" in error for error in errors)
        )

    async def test_code_owned_annotation_anomaly_still_hard_blocks(self) -> None:
        warning = {
            "code": "target_mention_false_negative",
            "severity": "important",
            "finding": (
                "Код нашёл буквальный exact-target alias в raw-ответе, "
                "но текущая разметка не засчитала его."
            ),
            "answer_ids": [11],
            "entities": ["Example"],
        }
        review = _critic_review(
            "block",
            anomalies=[copy.deepcopy(warning)],
        )
        with (
            patch(
                "app.services.analyzer._deterministic_annotation_warnings",
                return_value=[warning],
            ),
            patch(
                "app.services.analyzer._analysis_critic_artifact",
                new_callable=AsyncMock,
                return_value=review,
            ),
            patch(
                "app.services.analyzer._save_critic_gate",
                new_callable=AsyncMock,
            ) as save_gate,
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
        ):
            with self.assertRaises(_ConfirmedCriticIntegrityBlock) as caught:
                await _run_analysis_critic_loop(
                    "run-confirmed-annotation-integrity",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )

        self.assertIn(
            "annotation_integrity:target_mention_false_negative",
            caught.exception.reason_codes,
        )
        self.assertFalse(save_gate.await_args.kwargs["passed"])
        self.assertEqual(
            save_gate.await_args.kwargs["critic_outcome"],
            "confirmed_integrity_block",
        )

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
                return_value={"passed": True, "quality_state": "degraded"},
            ) as save_gate,
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
            _catalog, returned_rows, returned_metrics, gate = (
                await _run_analysis_critic_loop(
                    "run-r2-block",
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=ROWS,
                    metrics=METRICS,
                )
            )
        planner.assert_not_awaited()
        self.assertIs(returned_rows, ROWS)
        self.assertIs(returned_metrics, METRICS)
        self.assertTrue(gate["passed"])
        self.assertEqual(save_gate.await_args.kwargs["quality_state"], "degraded")
        self.assertEqual(
            save_gate.await_args.kwargs["critic_outcome"],
            "advisory_block",
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

    def test_makarska_false_negative_compiles_answer_bound_receipt(self) -> None:
        warnings = _deterministic_annotation_warnings(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            rows=[copy.deepcopy(MAKAR_ROW)],
        )
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], "target_mention_false_negative")
        self.assertEqual(warnings[0]["answer_ids"], [580])

        review = _critic_review(
            "revise",
            anomalies=[
                {
                    "code": "target_mention_false_negative",
                    "severity": "important",
                    "finding": (
                        "Raw буквально называет exact target и связывает "
                        "его с официальным источником, но разметка дала false."
                    ),
                    "answer_ids": [580],
                    "entities": ["Makarska Tattoo & Piercing Studio"],
                }
            ],
            adjustments=[
                {
                    "action": "require_literal_target_mention_evidence",
                    "entity_name": "Makarska Tattoo & Piercing Studio",
                    "alias": "Tattoo & Piercing Makarska",
                    "reason": "Нужно исправить доказанное ложное отрицание.",
                    "answer_ids": [580],
                }
            ],
            guidance="Повторно разметить только ответ 580.",
        )
        payload = _critic_payload(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            rows=[copy.deepcopy(MAKAR_ROW)],
            metrics={"providers": []},
            policy_history=[],
        )
        self.assertEqual(payload["client_domain"], "makarskatattoo.com")
        self.assertEqual(
            payload["answers"][0]["citation_sources"][1]["host"],
            "makarskatattoo.com",
        )
        self.assertEqual(
            _critic_review_validation_errors(review, payload=payload),
            [],
        )

        original_catalog_digest = stable_digest(MAKAR_CATALOG)
        tightened, applied, guidance = _apply_critic_policy(
            MAKAR_CATALOG,
            review,
            valid_answer_ids={580},
            profile=MAKAR_PROFILE,
            answer_rows=[copy.deepcopy(MAKAR_ROW)],
        )
        self.assertEqual(stable_digest(tightened), original_catalog_digest)
        self.assertEqual(len(applied), 1)
        receipt = applied[0]["target_mention_receipts"][0]
        self.assertEqual(receipt["answer_id"], 580)
        self.assertEqual(receipt["official_citation_index"], 2)
        self.assertIn("не переносится в другие строки", guidance)

        pending = {
            "answer_id": 580,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": copy.deepcopy(MAKAR_ROW["citations"]),
            "response_annotations": copy.deepcopy(
                MAKAR_ROW["response_annotations"]
            ),
            "answer": MAKAR_RAW,
            "answer_sha256": hashlib.sha256(MAKAR_RAW.encode()).hexdigest(),
            "answer_model": MAKAR_ROW["model"],
        }
        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            MAKAR_PROFILE,
            MAKAR_CATALOG,
            annotation_input_sha256="receipt-context",
            target_mention_receipts=[receipt],
        )
        self.assertTrue(reconciled["target_mentioned"])
        self.assertEqual(reconciled["target_role"], "mentioned")
        self.assertEqual(reconciled["target_position"], None)
        self.assertEqual(
            reconciled["_critic_target_mention_receipts"],
            [receipt],
        )

        grounded_row = {
            **copy.deepcopy(MAKAR_ROW),
            "annotation": reconciled,
        }
        self.assertTrue(
            _row_target_mention_is_grounded(
                grounded_row,
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
            )
        )

    def test_critic_paths_cannot_weaken_code_owned_coreference_contract(
        self,
    ) -> None:
        profile = {
            "brand_name": "London Tattoo Studio",
            "brand_aliases": [],
            "entity_scope": [
                {
                    "canonical_name": "London Tattoo Studio",
                    "aliases": [],
                    "relationship": "self",
                    "entity_type": "primary_brand",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
            "offer_catalog": {
                "client_domain": "londontattoostudio.example",
                "accepted_offers": [],
            },
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "London Tattoo Studio",
                    "aliases": ["London Tattoo"],
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ]
        }
        row = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_id": 633,
            "scenario": "Какие тату-студии популярны?",
            "answer_text": "London Tattoo is a popular search term [1].",
            "citations_count": 1,
            "citations": [
                {
                    "url": "https://londontattoostudio.example/",
                    "title": "London Tattoo Studio",
                }
            ],
            "response_annotations": [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://londontattoostudio.example/",
                        "title": "London Tattoo Studio",
                        "start_index": 0,
                        "end_index": 0,
                    },
                }
            ],
        }
        self.assertEqual(
            _deterministic_annotation_warnings(
                profile=profile,
                catalog=catalog,
                rows=[row],
            ),
            [],
        )
        review = _critic_review(
            "revise",
            adjustments=[
                {
                    "action": "require_literal_target_mention_evidence",
                    "entity_name": "London Tattoo Studio",
                    "alias": "London Tattoo",
                    "reason": "Модель предложила считать строку упоминанием.",
                    "answer_ids": [633],
                }
            ],
            guidance="Перепроверить строку.",
        )
        _tightened, applied, guidance = _apply_critic_policy(
            catalog,
            review,
            valid_answer_ids={633},
            profile=profile,
            answer_rows=[row],
        )
        self.assertEqual(applied, [])
        self.assertEqual(guidance, "")

    def test_makarska_false_negative_is_reconciled_without_critic_choice(self) -> None:
        pending = {
            "answer_id": 580,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": copy.deepcopy(MAKAR_ROW["citations"]),
            "response_annotations": copy.deepcopy(
                MAKAR_ROW["response_annotations"]
            ),
            "answer": MAKAR_RAW,
            "answer_sha256": hashlib.sha256(MAKAR_RAW.encode()).hexdigest(),
            "answer_model": MAKAR_ROW["model"],
        }
        receipts = _code_owned_target_mention_receipts(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            row=pending,
        )
        self.assertEqual(len(receipts), 1)

        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            MAKAR_PROFILE,
            MAKAR_CATALOG,
            annotation_input_sha256="code-owned-identity-receipt",
            target_mention_receipts=receipts,
        )

        self.assertTrue(reconciled["target_mentioned"])
        self.assertEqual(reconciled["target_role"], "mentioned")
        self.assertIsNone(reconciled["target_position"])
        self.assertEqual(reconciled["sentiment"], "unknown")
        self.assertEqual(
            reconciled["_critic_target_mention_receipts"][0]["alias"],
            "Tattoo & Piercing Makarska",
        )
        self.assertTrue(
            _row_target_mention_is_grounded(
                {**copy.deepcopy(MAKAR_ROW), "annotation": reconciled},
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
            )
        )

    def test_generic_makarska_tattoo_search_term_stays_absent(self) -> None:
        raw = (
            "Ищите в соцсетях по ключевым словам `coverup makarska tattoo` "
            "и `rework tattoo makarska`."
        )
        pending = {
            "answer_id": 627,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": [],
            "response_annotations": [],
            "answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_model": "deepseek/deepseek-chat",
        }

        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            MAKAR_PROFILE,
            MAKAR_CATALOG,
            annotation_input_sha256="generic-search-term",
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertEqual(reconciled["target_role"], "absent")
        self.assertNotIn("_critic_target_mention_receipts", reconciled)

    def test_answer_derived_target_alias_alone_stays_absent(self) -> None:
        raw = "Tattoo & Piercing Makarska называют одной из местных студий."
        pending = {
            "answer_id": 628,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": [
                {"url": "https://directory.example/studios", "title": "Directory"}
            ],
            "response_annotations": [],
            "answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_model": "example/model",
        }

        self.assertEqual(
            _code_owned_target_mention_receipts(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                row=pending,
            ),
            [],
        )

        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            MAKAR_PROFILE,
            MAKAR_CATALOG,
            annotation_input_sha256="answer-derived-alias-only",
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertNotIn("_critic_target_mention_receipts", reconciled)

    def test_competitor_relationship_cannot_erase_independent_target_receipt(
        self,
    ) -> None:
        catalog = copy.deepcopy(MAKAR_CATALOG)
        catalog["entities"].append(
            {
                "canonical_name": "Tattoo Točka Rijeka",
                "aliases": ["Točka"],
                "category": "competitor",
                "target_relationship": "competitor",
                "commercially_relevant": True,
                "mention_policy": "standalone",
            }
        )
        raw = MAKAR_RAW + "\nTattoo Točka Rijeka — отдельная студия."
        pending = {
            "answer_id": 580,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": copy.deepcopy(MAKAR_ROW["citations"]),
            "response_annotations": copy.deepcopy(
                MAKAR_ROW["response_annotations"]
            ),
            "answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_model": MAKAR_ROW["model"],
        }
        annotation = copy.deepcopy(MAKAR_ROW["annotation"])
        annotation["entity_mentions"] = [
            {
                "canonical_name": "Tattoo Točka Rijeka",
                "position": 2,
                "role": "mentioned",
                "attributed_to_target": False,
                "evidence": "Tattoo Točka Rijeka",
            }
        ]

        receipts = _code_owned_target_mention_receipts(
            profile=MAKAR_PROFILE,
            catalog=catalog,
            row=pending,
        )
        self.assertEqual(len(receipts), 1)
        reconciled = _reconcile_annotation(
            annotation,
            pending,
            MAKAR_PROFILE,
            catalog,
            annotation_input_sha256="independent-target-and-competitor",
            target_mention_receipts=receipts,
        )

        self.assertTrue(reconciled["target_mentioned"])
        self.assertEqual(reconciled["target_role"], "mentioned")
        self.assertIn("_critic_target_mention_receipts", reconciled)
        competitor = next(
            mention
            for mention in reconciled["entity_mentions"]
            if mention["canonical_name"] == "Tattoo Točka Rijeka"
        )
        self.assertFalse(competitor["attributed_to_target"])

    def test_realweb_generic_catalog_alias_cannot_issue_target_receipt(self) -> None:
        profile = {
            "brand_name": "Realweb",
            "brand_aliases": ["Риалвеб"],
            "entity_scope": [
                {
                    "canonical_name": "Realweb",
                    "aliases": ["Риалвеб"],
                    "relationship": "self",
                    "entity_type": "primary_brand",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
            "offer_catalog": {
                "client_domain": "realweb.ru",
                "accepted_offers": [],
            },
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "Realweb",
                    "aliases": ["DOOH", "programmatic"],
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ]
        }
        raw = "DOOH и programmatic доступны рекламодателям.[1]"
        pending = {
            "answer_id": 629,
            "status": "completed",
            "metric_eligible": True,
            "scenario": "Какие рекламные технологии выбрать?",
            "scenario_role": "unbranded_discovery",
            "citations": [
                {"url": "https://realweb.ru/services", "title": "Realweb"}
            ],
            "response_annotations": [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://realweb.ru/services",
                        "title": "Realweb",
                        "start_index": 0,
                        "end_index": 0,
                    },
                }
            ],
            "answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_model": "example/model",
        }

        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            profile,
            catalog,
            annotation_input_sha256="realweb-generic-alias",
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertNotIn("_critic_target_mention_receipts", reconciled)

    def test_geographic_profile_subset_cannot_issue_target_receipt(self) -> None:
        profile = {
            "brand_name": "New York Tattoo Studio",
            "brand_aliases": [],
            "entity_scope": [
                {
                    "canonical_name": "New York Tattoo Studio",
                    "aliases": [],
                    "relationship": "self",
                    "entity_type": "primary_brand",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
            "offer_catalog": {
                "client_domain": "newyorktattoo.example",
                "accepted_offers": [],
            },
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "New York Tattoo Studio",
                    "aliases": ["New York"],
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ]
        }
        raw = "New York привлекает путешественников.[1]"
        pending = {
            "answer_id": 630,
            "status": "completed",
            "metric_eligible": True,
            "scenario": "Куда поехать в США?",
            "scenario_role": "unbranded_discovery",
            "citations": [
                {
                    "url": "https://newyorktattoo.example/about",
                    "title": "New York Tattoo Studio",
                }
            ],
            "response_annotations": [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://newyorktattoo.example/about",
                        "title": "New York Tattoo Studio",
                        "start_index": 0,
                        "end_index": 0,
                    },
                }
            ],
            "answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_model": "example/model",
        }

        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            profile,
            catalog,
            annotation_input_sha256="geographic-profile-subset",
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertNotIn("_critic_target_mention_receipts", reconciled)

    def test_three_token_geographic_prefix_cannot_issue_target_receipt(self) -> None:
        profile = {
            "brand_name": "New York City Studio",
            "brand_aliases": [],
            "entity_scope": [
                {
                    "canonical_name": "New York City Studio",
                    "aliases": [],
                    "relationship": "self",
                    "entity_type": "primary_brand",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
            "offer_catalog": {
                "client_domain": "newyorkcitystudio.example",
                "accepted_offers": [],
            },
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "New York City Studio",
                    "aliases": ["New York City"],
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ]
        }
        raw = "New York City привлекает путешественников.[1]"
        pending = {
            "answer_id": 631,
            "status": "completed",
            "metric_eligible": True,
            "scenario": "Куда поехать в США?",
            "scenario_role": "unbranded_discovery",
            "citations": [
                {
                    "url": "https://newyorkcitystudio.example/about",
                    "title": "New York City Studio",
                }
            ],
            "response_annotations": [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://newyorkcitystudio.example/about",
                        "title": "New York City Studio",
                        "start_index": 0,
                        "end_index": 0,
                    },
                }
            ],
            "answer": raw,
            "answer_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "answer_model": "example/model",
        }

        reconciled = _reconcile_annotation(
            copy.deepcopy(MAKAR_ROW["annotation"]),
            pending,
            profile,
            catalog,
            annotation_input_sha256="three-token-geographic-prefix",
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertNotIn("_critic_target_mention_receipts", reconciled)

    def test_target_receipt_rejects_echo_remote_source_and_transfer(self) -> None:
        entity = MAKAR_CATALOG["entities"][0]
        alias = "Tattoo & Piercing Makarska"
        receipt = _target_mention_receipt(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            entity=entity,
            alias=alias,
            row=copy.deepcopy(MAKAR_ROW),
        )
        self.assertIsNotNone(receipt)

        prompt_echo = {
            **copy.deepcopy(MAKAR_ROW),
            "scenario": MAKAR_ROW["scenario"] + " Tattoo & Piercing Makarska",
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=prompt_echo,
            )
        )

        url_prompt_echo = {
            **copy.deepcopy(MAKAR_ROW),
            "scenario": (
                "Суммируй страницу https://directory.example/vendors/"
                "tattoo-and-piercing-makarska"
            ),
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=url_prompt_echo,
            )
        )

        url_prompt_echo_without_connector = {
            **copy.deepcopy(MAKAR_ROW),
            "scenario": (
                "Суммируй https://directory.example/vendors/"
                "tattoo-piercing-makarska"
            ),
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=url_prompt_echo_without_connector,
            )
        )

        encoded_url_prompt_echo = {
            **copy.deepcopy(MAKAR_ROW),
            "scenario": (
                "Суммируй https://directory.example/vendors/"
                "tattoo-%2526-piercing-makarska"
            ),
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=encoded_url_prompt_echo,
            )
        )

        html_entity_prompt_echo = {
            **copy.deepcopy(MAKAR_ROW),
            "scenario": "Суммируй Tattoo &amp; Piercing Makarska",
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=html_entity_prompt_echo,
            )
        )

        for encoded_domain in (
            "https://makarska%74attoo.com",
            "https%3A%2F%2Fmakarska%2574attoo.com",
        ):
            with self.subTest(encoded_domain=encoded_domain):
                domain_prompt_echo = {
                    **copy.deepcopy(MAKAR_ROW),
                    "scenario": f"Какие студии? {encoded_domain}",
                }
                self.assertIsNone(
                    _target_mention_receipt(
                        profile=MAKAR_PROFILE,
                        catalog=MAKAR_CATALOG,
                        entity=entity,
                        alias=alias,
                        row=domain_prompt_echo,
                    )
                )

        cross_alias_prompt_echo = {
            **copy.deepcopy(MAKAR_ROW),
            "scenario": "Что известно о Makarska Tattoo?",
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=cross_alias_prompt_echo,
            )
        )

        remote_marker = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Здесь названа Tattoo & Piercing Makarska.\n\n"
                "Официальный источник приведён отдельно.[2]"
            ),
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=remote_marker,
            )
        )

        unrelated_same_line = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Tattoo & Piercing Makarska здесь означает лишь поисковую "
                "фразу. Отдельный факт о сетевой доступности подтверждён [2]."
            ),
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=unrelated_same_line,
            )
        )

        transferred = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_id": 581,
            "annotation": {
                **copy.deepcopy(MAKAR_ROW["annotation"]),
                "target_mentioned": True,
                "target_role": "mentioned",
                "_critic_target_mention_receipts": [receipt],
            },
        }
        self.assertFalse(
            _row_target_mention_is_grounded(
                transferred,
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
            )
        )

        changed_raw = copy.deepcopy(MAKAR_ROW)
        changed_raw["answer_text"] += " Изменённый хвост."
        changed_raw["annotation"] = {
            **copy.deepcopy(MAKAR_ROW["annotation"]),
            "target_mentioned": True,
            "target_role": "mentioned",
            "_critic_target_mention_receipts": [receipt],
        }
        self.assertFalse(
            _row_target_mention_is_grounded(
                changed_raw,
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
            )
        )

        external_only = copy.deepcopy(MAKAR_ROW)
        external_only["citations"] = [
            {"url": "https://example.org/2", "title": "External"},
            {"url": "https://example.org/3", "title": "External"},
        ]
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=external_only,
            )
        )

        deduplicated_sources = copy.deepcopy(MAKAR_ROW)
        deduplicated_sources["response_annotations"].insert(
            1,
            copy.deepcopy(deduplicated_sources["response_annotations"][1]),
        )
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=deduplicated_sources,
            )
        )

        evil_substring = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Рекомендую Tattoo & Piercing Makarska "
                "(https://evil-makarskatattoo.com/fake)."
            ),
            "response_annotations": [],
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=evil_substring,
            )
        )

        unicode_prefixed_domain = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Tattoo & Piercing Makarska (жmakarskatattoo.com)."
            ),
            "response_annotations": [],
        }
        self.assertIsNone(
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                entity=entity,
                alias=alias,
                row=unicode_prefixed_domain,
            )
        )

        repeated_alias = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Список: Tattoo & Piercing Makarska.\n\n"
                "Подробнее: Tattoo & Piercing Makarska [2] принимает заявки."
            ),
        }
        repeated_receipt = _target_mention_receipt(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            entity=entity,
            alias=alias,
            row=repeated_alias,
        )
        self.assertIsNotNone(repeated_receipt)
        self.assertEqual(
            repeated_receipt["alias_start_char"],
            repeated_alias["answer_text"].rindex(alias),
        )

    def test_code_owned_receipt_requires_identity_coreference_and_literal_use(
        self,
    ) -> None:
        london_profile = {
            "brand_name": "London Tattoo Studio",
            "brand_aliases": [],
            "entity_scope": [
                {
                    "canonical_name": "London Tattoo Studio",
                    "aliases": [],
                    "relationship": "self",
                    "entity_type": "primary_brand",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
            "offer_catalog": {
                "client_domain": "londontattoostudio.example",
                "accepted_offers": [],
            },
        }
        london_catalog = {
            "entities": [
                {
                    "canonical_name": "London Tattoo Studio",
                    "aliases": ["Tattoo Studio London", "London Tattoo"],
                    "category": "target",
                    "target_relationship": "exact_target",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ]
        }
        lone_reorder = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_id": 632,
            "answer_text": "Tattoo Studio London — популярный запрос.[1]",
            "scenario": "Какие студии популярны?",
            "citations": [
                {
                    "url": "https://londontattoostudio.example/",
                    "title": "London Tattoo Studio",
                }
            ],
            "response_annotations": [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://londontattoostudio.example/",
                        "title": "London Tattoo Studio",
                        "start_index": 0,
                        "end_index": 0,
                    },
                }
            ],
        }
        self.assertEqual(
            _code_owned_target_mention_receipts(
                profile=london_profile,
                catalog=london_catalog,
                row=lone_reorder,
            ),
            [],
        )

        rejected_lines = (
            "Ищите по запросу `Tattoo & Piercing Makarska "
            "(Makarska Tattoo)` [2].",
            "Tattoo & Piercing Makarska (Makarska Tattoo) — это не "
            "название студии [2].",
            "Tattoo & Piercing Makarska (Makarska Tattoo) — категория "
            "заведений [2].",
            "Введите Tattoo & Piercing Makarska (Makarska Tattoo) в строку "
            "поиска [2].",
            "Наберите Tattoo & Piercing Makarska (Makarska Tattoo) в Google [2].",
            "Tattoo & Piercing Makarska (Makarska Tattoo) здесь не означает "
            "название студии [2].",
            "Tattoo & Piercing Makarska (Makarska Tattoo) не относится к "
            "бренду [2].",
            "Tattoo & Piercing Makarska (Makarska Tattoo), а адрес другой "
            "студии подтверждён [2].",
            "Запрос: Tattoo & Piercing Makarska (Makarska Tattoo) [2].",
            "Поиск: Tattoo & Piercing Makarska (Makarska Tattoo) [2].",
            "Название для поиска: Tattoo & Piercing Makarska "
            "(Makarska Tattoo) [2].",
            "Результат поиска «Tattoo & Piercing Makarska "
            "(Makarska Tattoo)» [2].",
            "Запрос «Tattoo & Piercing Makarska (Makarska Tattoo)» [2].",
            "Ключевая фраза Tattoo & Piercing Makarska "
            "(Makarska Tattoo) [2].",
            "Не бренд, а Tattoo & Piercing Makarska "
            "(Makarska Tattoo) [2].",
            "Это не студия Tattoo & Piercing Makarska "
            "(Makarska Tattoo) [2].",
        )
        for answer_text in rejected_lines:
            with self.subTest(answer_text=answer_text):
                row = {**copy.deepcopy(MAKAR_ROW), "answer_text": answer_text}
                self.assertEqual(
                    _code_owned_target_mention_receipts(
                        profile=MAKAR_PROFILE,
                        catalog=MAKAR_CATALOG,
                        row=row,
                    ),
                    [],
                )

        generic_corroborator_catalog = copy.deepcopy(MAKAR_CATALOG)
        generic_corroborator_catalog["entities"][0]["aliases"].append(
            "Tattoo Studio"
        )
        generic_corroborator = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Tattoo & Piercing Makarska (Tattoo Studio) [2]."
            ),
        }
        self.assertEqual(
            _code_owned_target_mention_receipts(
                profile=MAKAR_PROFILE,
                catalog=generic_corroborator_catalog,
                row=generic_corroborator,
            ),
            [],
        )

        operational_negative = {
            **copy.deepcopy(MAKAR_ROW),
            "answer_text": (
                "Tattoo & Piercing Makarska (Makarska Tattoo) временно не "
                "принимает заявки [2]."
            ),
        }
        self.assertEqual(
            len(
                _code_owned_target_mention_receipts(
                    profile=MAKAR_PROFILE,
                    catalog=MAKAR_CATALOG,
                    row=operational_negative,
                )
            ),
            1,
        )

    def test_target_receipt_never_promotes_recommendation_or_position(self) -> None:
        receipt = _target_mention_receipt(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            entity=MAKAR_CATALOG["entities"][0],
            alias="Tattoo & Piercing Makarska",
            row=copy.deepcopy(MAKAR_ROW),
        )
        self.assertIsNotNone(receipt)
        pending = {
            "answer_id": 580,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": copy.deepcopy(MAKAR_ROW["citations"]),
            "response_annotations": copy.deepcopy(
                MAKAR_ROW["response_annotations"]
            ),
            "answer": MAKAR_RAW,
            "answer_sha256": hashlib.sha256(MAKAR_RAW.encode()).hexdigest(),
            "answer_model": MAKAR_ROW["model"],
        }
        overstated = copy.deepcopy(MAKAR_ROW["annotation"])
        overstated.update(
            {
                "target_mentioned": True,
                "target_role": "recommended",
                "target_position": 1,
                "sentiment": "positive",
            }
        )

        reconciled = _reconcile_annotation(
            overstated,
            pending,
            MAKAR_PROFILE,
            MAKAR_CATALOG,
            annotation_input_sha256="receipt-does-not-prove-role",
            target_mention_receipts=[receipt],
        )

        self.assertTrue(reconciled["target_mentioned"])
        self.assertEqual(reconciled["target_role"], "mentioned")
        self.assertIsNone(reconciled["target_position"])
        self.assertEqual(reconciled["sentiment"], "unknown")
        absent_control = copy.deepcopy(MAKAR_ROW)
        absent_control.update(
            {
                "answer_id": 581,
                "answer_text": "В ответе перечислены другие студии.",
                "annotation": {
                    **copy.deepcopy(MAKAR_ROW["annotation"]),
                    "target_mentioned": False,
                    "target_role": "absent",
                    "target_position": None,
                    "sentiment": "neutral",
                },
            }
        )
        known_top3 = copy.deepcopy(MAKAR_ROW)
        known_top3.update(
            {
                "answer_id": 582,
                "answer_text": "Целевую студию рекомендуют первой.",
                "annotation": {
                    **copy.deepcopy(MAKAR_ROW["annotation"]),
                    "target_mentioned": True,
                    "target_role": "recommended",
                    "target_position": 1,
                    "sentiment": "positive",
                },
            }
        )
        before = _visibility_slice(
            [copy.deepcopy(MAKAR_ROW), absent_control, known_top3],
            mode="web",
        )
        after = _visibility_slice(
            [
                {**copy.deepcopy(MAKAR_ROW), "annotation": reconciled},
                absent_control,
                known_top3,
            ],
            mode="web",
        )
        self.assertEqual((before["mention_count"], before["mention_rate"]), (1, 33.3))
        self.assertEqual((after["mention_count"], after["mention_rate"]), (2, 66.7))
        self.assertEqual(before["top3_denominator"], 3)
        self.assertEqual(after["top3_denominator"], 3)
        self.assertEqual(after["top3_count"], before["top3_count"])
        self.assertEqual(after["top3_rate"], before["top3_rate"])
        self.assertEqual(
            after["recommendation_count"],
            before["recommendation_count"],
        )
        self.assertEqual(
            after["recommendation_rate"],
            before["recommendation_rate"],
        )
        self.assertEqual(before["score"], 33.3)
        self.assertEqual(after["score"], 41.6)

    def test_target_receipt_does_not_expand_other_answers(self) -> None:
        receipt = _target_mention_receipt(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            entity=MAKAR_CATALOG["entities"][0],
            alias="Tattoo & Piercing Makarska",
            row=copy.deepcopy(MAKAR_ROW),
        )
        self.assertIsNotNone(receipt)
        pending = {
            "answer_id": 581,
            "status": "completed",
            "metric_eligible": True,
            "scenario": MAKAR_ROW["scenario"],
            "scenario_role": "unbranded_discovery",
            "citations": copy.deepcopy(MAKAR_ROW["citations"]),
            "response_annotations": copy.deepcopy(
                MAKAR_ROW["response_annotations"]
            ),
            "answer": MAKAR_RAW,
            "answer_sha256": hashlib.sha256(MAKAR_RAW.encode()).hexdigest(),
            "answer_model": MAKAR_ROW["model"],
        }
        with self.assertRaisesRegex(
            OrchestratorContractError,
            "invalid schema or scope",
        ):
            _reconcile_annotation(
                copy.deepcopy(MAKAR_ROW["annotation"]),
                pending,
                MAKAR_PROFILE,
                MAKAR_CATALOG,
                annotation_input_sha256="other-answer-context",
                target_mention_receipts=[receipt],
            )

    def test_receipt_manifest_is_rebound_after_catalog_change_or_fails_loudly(
        self,
    ) -> None:
        receipts = _code_owned_target_mention_receipts(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            row=copy.deepcopy(MAKAR_ROW),
        )
        self.assertEqual(len(receipts), 1)
        changed_catalog = copy.deepcopy(MAKAR_CATALOG)
        changed_catalog["entities"].append(
            {
                "canonical_name": "Unrelated Studio",
                "aliases": ["Unrelated"],
                "category": "competitor",
                "target_relationship": "competitor",
                "mention_policy": "standalone",
            }
        )

        with self.assertRaisesRegex(
            OrchestratorContractError,
            "deterministic revalidation",
        ):
            _validated_target_mention_receipt_manifest(
                receipts,
                rows=[copy.deepcopy(MAKAR_ROW)],
                profile=MAKAR_PROFILE,
                catalog=changed_catalog,
            )

        refreshed = _refresh_target_mention_receipt_manifest(
            receipts,
            rows=[copy.deepcopy(MAKAR_ROW)],
            profile=MAKAR_PROFILE,
            catalog=changed_catalog,
        )
        self.assertNotEqual(refreshed, receipts)
        self.assertEqual(
            refreshed[0]["catalog_sha256"],
            _stable_json_sha256(changed_catalog),
        )
        self.assertEqual(
            _validated_target_mention_receipt_manifest(
                refreshed,
                rows=[copy.deepcopy(MAKAR_ROW)],
                profile=MAKAR_PROFILE,
                catalog=changed_catalog,
            ),
            refreshed,
        )

    def test_malformed_or_duplicate_receipt_manifest_fails_loudly(self) -> None:
        receipt = _code_owned_target_mention_receipts(
            profile=MAKAR_PROFILE,
            catalog=MAKAR_CATALOG,
            row=copy.deepcopy(MAKAR_ROW),
        )[0]
        malformed_cases = [
            ["not-an-object"],
            [{**receipt, "answer_id": "580"}],
            [{**receipt, "grounding_mode": "unknown-mode"}],
            [receipt, copy.deepcopy(receipt)],
        ]
        for manifest in malformed_cases:
            with self.subTest(manifest=manifest):
                with self.assertRaises(OrchestratorContractError):
                    _validated_target_mention_receipt_manifest(
                        manifest,
                        rows=[copy.deepcopy(MAKAR_ROW)],
                        profile=MAKAR_PROFILE,
                        catalog=MAKAR_CATALOG,
                    )

        with self.assertRaisesRegex(
            OrchestratorContractError,
            "invalid schema or scope",
        ):
            _validated_target_mention_receipt_manifest(
                [receipt],
                rows=[copy.deepcopy(MAKAR_ROW)],
                profile=MAKAR_PROFILE,
                catalog=MAKAR_CATALOG,
                allowed_answer_ids={999},
            )

    def test_target_receipt_binds_final_catalog_after_other_adjustments(self) -> None:
        catalog = copy.deepcopy(MAKAR_CATALOG)
        catalog["entities"].append(
            {
                "canonical_name": "Booking",
                "aliases": ["appointment"],
                "category": "target",
                "target_relationship": "portfolio_entity",
                "commercially_relevant": True,
                "mention_policy": "standalone",
            }
        )
        review = _critic_review(
            "revise",
            adjustments=[
                {
                    "action": "require_literal_target_mention_evidence",
                    "entity_name": "Makarska Tattoo & Piercing Studio",
                    "alias": "Tattoo & Piercing Makarska",
                    "reason": "Исправить ложное отрицание.",
                    "answer_ids": [580],
                },
                {
                    "action": "require_target_attribution",
                    "entity_name": "Booking",
                    "alias": None,
                    "reason": "Общая услуга требует явной связи.",
                    "answer_ids": [580],
                },
            ],
            guidance="Применить обе проверки.",
            anomalies=[
                {
                    "code": "target_mention_false_negative",
                    "severity": "important",
                    "finding": "Нужно исправить ложное отрицание exact target.",
                    "answer_ids": [580],
                    "entities": ["Makarska Tattoo & Piercing Studio"],
                }
            ],
        )

        tightened, applied, _guidance = _apply_critic_policy(
            catalog,
            review,
            valid_answer_ids={580},
            profile=MAKAR_PROFILE,
            answer_rows=[copy.deepcopy(MAKAR_ROW)],
        )
        receipt_adjustment = next(
            item
            for item in applied
            if item["action"] == "require_literal_target_mention_evidence"
        )
        receipt = receipt_adjustment["target_mention_receipts"][0]
        final_target = next(
            entity
            for entity in tightened["entities"]
            if entity["canonical_name"]
            == "Makarska Tattoo & Piercing Studio"
        )
        self.assertEqual(
            receipt,
            _target_mention_receipt(
                profile=MAKAR_PROFILE,
                catalog=tightened,
                entity=final_target,
                alias="Tattoo & Piercing Makarska",
                row=copy.deepcopy(MAKAR_ROW),
                require_local_coreference=True,
            ),
        )

    def test_ungrounded_target_cannot_keep_recommendation_or_position(self) -> None:
        row = {
            **copy.deepcopy(ROWS[0]),
            "answer_text": "В ответе рекомендуется только OtherCo.",
            "citations": [],
            "annotation": {
                **copy.deepcopy(ROWS[0]["annotation"]),
                "target_mentioned": True,
                "target_position": 1,
                "target_role": "recommended",
                "entity_mentions": [],
            },
        }

        metrics = _compute_metrics([row], PROFILE, CATALOG)
        parent = metrics["parent_discovery"]["web"]

        self.assertEqual(parent["mention_count"], 0)
        self.assertEqual(parent["top3_count"], 0)
        self.assertEqual(parent["recommendation_count"], 0)
        self.assertEqual(parent["recommendation_rate"], 0.0)

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
        target_receipts_sha256 = stable_digest([])
        resume_annotation_digest = _annotation_context_sha256(
            PROFILE,
            scoped_catalog,
        )
        repair_annotation_digest = _annotation_context_sha256(
            PROFILE,
            scoped_catalog,
            plan["guidance"],
            repair_mode=ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
            target_mention_receipts=[],
        )
        recovery_step = {
            "iteration": MAX_CRITIC_ITERATIONS + 1,
            "kind": "orchestrated_targeted_annotation_repair",
            "orchestrator_epoch": epoch,
            "target_answer_ids": [11],
            "critic_adjustments": [],
            "annotation_guidance": plan["guidance"],
            "raw_corpus_sha256": "",
            "target_mention_receipts_sha256": target_receipts_sha256,
            "resume_target_mention_receipts_sha256": target_receipts_sha256,
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
            "repair_annotation_input_sha256": repair_annotation_digest,
            "resume_annotation_input_sha256": resume_annotation_digest,
            "guidance_sha256": hashlib.sha256(
                plan["guidance"].encode("utf-8")
            ).hexdigest(),
            "critic_review_sha256": stable_digest({}),
            "target_mention_receipts_sha256": target_receipts_sha256,
            "resume_target_mention_receipts_sha256": target_receipts_sha256,
            "recovery_policy_step": copy.deepcopy(recovery_step),
        }
        repaired_rows[0]["annotation"][
            "_annotation_input_sha256"
        ] = repair_annotation_digest
        repaired_rows[0]["annotation"]["uncertainties"] = [
            "targeted repair completed"
        ]
        state_digest = _critic_analysis_state_digest(
            repaired_rows,
            METRICS,
            profile=PROFILE,
            catalog=scoped_catalog,
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
        scoped_catalog = _scope_entity_catalog_to_profile(CATALOG, PROFILE)
        resume_annotation_context = _annotation_context_manifest(
            profile=PROFILE,
            catalog=scoped_catalog,
            rows=rows,
            research_guidance="",
            target_mention_receipts=[],
        )
        facts = {
            "site_profile": PROFILE,
            "entity_catalog": scoped_catalog,
            "analysis_state_sha256": before_digest,
            "raw_corpus_sha256": _raw_corpus_digest(rows),
            "critic_review": {},
            "prior_policy_history": [],
            "target_mention_receipts_sha256": plan.get(
                "target_mention_receipts_sha256",
                stable_digest([]),
            ),
            "resume_annotation_context": resume_annotation_context,
        }
        allowed_actions = {
            ACTION_STOP,
            ACTION_TARGETED_ANNOTATION_REPAIR,
        }
        permitted_answer_ids = set(plan["target_answer_ids"])
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions=allowed_actions,
            permitted_answer_ids=permitted_answer_ids,
            permitted_artifact_keys=set(),
        )
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    failure_fingerprint="f" * 64,
                    facts_digest=facts_digest,
                    status=status,
                    input_json={
                        "incident": {
                            "run_id": self.run_id,
                            "stage": "analysis_critic",
                            "failure_class": "repairable_semantic",
                            "code": "analysis_critic_non_convergent",
                            "fingerprint": "f" * 64,
                            "facts_digest": facts_digest,
                            "diagnostics": {},
                            "facts": facts,
                        },
                        "allowed_actions": sorted(allowed_actions),
                        "permitted_answer_ids": sorted(permitted_answer_ids),
                        "permitted_artifact_keys": [],
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

    def _executing_checkpoint_fixture(
        self,
        *,
        scoped_catalog: dict,
        repaired_rows: list[dict],
        plan: dict,
        recovery_step: dict,
    ) -> tuple[dict, dict, dict[int, str], dict[int, str], dict]:
        resume_context = _annotation_context_manifest(
            profile=PROFILE,
            catalog=scoped_catalog,
            rows=repaired_rows,
            research_guidance="",
            target_mention_receipts=[],
        )
        repair_context = _annotation_context_manifest(
            profile=PROFILE,
            catalog=scoped_catalog,
            rows=repaired_rows,
            research_guidance=recovery_step["annotation_guidance"],
            target_mention_receipts=[],
            repair_mode=ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
            allowed_answer_ids={11},
        )
        annotation_digests = _current_annotation_input_digests(repaired_rows)
        resume_annotation_digests = {
            answer_id: str(resume_context["annotation_input_sha256"])
            for answer_id in annotation_digests
        }
        annotation_repair_provenance = copy.deepcopy(
            repaired_rows[0]["annotation"]["_annotation_repair_provenance"]
        )
        checkpoint = {
            "version": ANALYSIS_CRITIC_RECOVERY_CHECKPOINT_VERSION,
            "phase": "prepared",
            "orchestrator_epoch": 1,
            "orchestrator_plan_digest": stable_digest(plan),
            "target_answer_ids": [11],
            "resume_annotation_context": resume_context,
            "repair_annotation_context": repair_context,
            "annotation_repair_provenance": annotation_repair_provenance,
            "resume_annotation_input_sha256_by_answer_id": {
                str(answer_id): digest
                for answer_id, digest in sorted(
                    resume_annotation_digests.items()
                )
            },
            "annotation_input_sha256_by_answer_id": {
                str(answer_id): digest
                for answer_id, digest in sorted(annotation_digests.items())
            },
        }
        return (
            resume_context,
            repair_context,
            resume_annotation_digests,
            annotation_digests,
            checkpoint,
        )

    async def _insert_executing_checkpoint_artifact(
        self,
        *,
        scoped_catalog: dict,
        checkpoint: dict,
        resume_context: dict,
    ) -> None:
        async with self.SessionLocal() as session:
            session.add(
                RunArtifact(
                    run_id=self.run_id,
                    stage_key="knowledge_gap",
                    artifact_key="analysis_critic_policy",
                    status="completed",
                    model=CRITIC_MODEL,
                    prompt_version=ANALYSIS_CRITIC_VERSION,
                    input_json={
                        "iteration": MAX_CRITIC_ITERATIONS + 1,
                        "executing_recovery_checkpoint_sha256": (
                            _stable_json_sha256(checkpoint)
                        ),
                    },
                    output_json={
                        "base_catalog_version": "fixture",
                        "policy_history": [],
                        "effective_profile": PROFILE,
                        "effective_catalog": scoped_catalog,
                        "annotation_context": resume_context,
                        "annotation_context_sha256": _stable_json_sha256(
                            resume_context
                        ),
                        "executing_recovery": checkpoint,
                    },
                )
            )
            await session.commit()

    async def test_executing_checkpoint_loader_reconstructs_mixed_state(
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
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
        )
        (
            resume_context,
            _repair_context,
            _resume_annotation_digests,
            annotation_digests,
            checkpoint,
        ) = self._executing_checkpoint_fixture(
            scoped_catalog=scoped_catalog,
            repaired_rows=repaired_rows,
            plan=plan,
            recovery_step=recovery_step,
        )
        await self._insert_executing_checkpoint_artifact(
            scoped_catalog=scoped_catalog,
            checkpoint=checkpoint,
            resume_context=resume_context,
        )

        with patch(
            "app.services.analyzer._metric_rows",
            new=AsyncMock(side_effect=[repaired_rows, repaired_rows]),
        ) as metric_rows:
            loaded = await _load_executing_analysis_critic_checkpoint(
                self.run_id,
                profile=PROFILE,
            )

        self.assertIsNotNone(loaded)
        loaded_catalog, loaded_rows, _loaded_metrics, loaded_context = loaded
        self.assertEqual(loaded_catalog, scoped_catalog)
        self.assertEqual(loaded_rows, repaired_rows)
        self.assertEqual(loaded_context, resume_context)
        self.assertEqual(metric_rows.await_count, 2)
        self.assertEqual(
            metric_rows.await_args_list[1].kwargs[
                "annotation_input_sha256_by_answer_id"
            ],
            annotation_digests,
        )

    async def test_executing_checkpoint_loader_completes_exact_pre_cas_state(
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
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=state_digest,
            rows=repaired_rows,
        )
        (
            resume_context,
            repair_context,
            resume_annotation_digests,
            annotation_digests,
            checkpoint,
        ) = self._executing_checkpoint_fixture(
            scoped_catalog=scoped_catalog,
            repaired_rows=repaired_rows,
            plan=plan,
            recovery_step=recovery_step,
        )
        await self._insert_executing_checkpoint_artifact(
            scoped_catalog=scoped_catalog,
            checkpoint=checkpoint,
            resume_context=resume_context,
        )
        resume_rows = copy.deepcopy(repaired_rows)
        for row in resume_rows:
            row["annotation"]["_annotation_input_sha256"] = str(
                resume_context["annotation_input_sha256"]
            )
            row["annotation"].pop("_annotation_repair_provenance", None)

        with (
            patch(
                "app.services.analyzer._metric_rows",
                new=AsyncMock(
                    side_effect=[
                        resume_rows,
                        resume_rows,
                        resume_rows,
                        repaired_rows,
                    ]
                ),
            ) as metric_rows,
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate,
        ):
            loaded = await _load_executing_analysis_critic_checkpoint(
                self.run_id,
                profile=PROFILE,
            )

        self.assertIsNotNone(loaded)
        loaded_catalog, loaded_rows, _loaded_metrics, loaded_context = loaded
        self.assertEqual(loaded_catalog, scoped_catalog)
        self.assertEqual(loaded_rows, repaired_rows)
        self.assertEqual(loaded_context, resume_context)
        self.assertEqual(metric_rows.await_count, 4)
        annotate.assert_awaited_once_with(
            self.run_id,
            PROFILE,
            scoped_catalog,
            research_guidance=repair_context["research_guidance"],
            target_answer_ids={11},
            repair_mode=ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
            annotation_repair_provenance=checkpoint[
                "annotation_repair_provenance"
            ],
            target_mention_receipts=[],
            _completion_attempt=ANNOTATION_COMPLETION_ATTEMPTS,
        )
        self.assertEqual(
            metric_rows.await_args_list[2].kwargs[
                "annotation_input_sha256_by_answer_id"
            ],
            resume_annotation_digests,
        )
        self.assertEqual(
            metric_rows.await_args_list[3].kwargs[
                "annotation_input_sha256_by_answer_id"
            ],
            annotation_digests,
        )

    async def test_pre_cas_replay_keeps_nonempty_target_receipt_manifest(
        self,
    ) -> None:
        profile = copy.deepcopy(MAKAR_PROFILE)
        catalog = _scope_entity_catalog_to_profile(
            copy.deepcopy(MAKAR_CATALOG),
            profile,
        )
        source_row = copy.deepcopy(MAKAR_ROW)
        receipts = _code_owned_target_mention_receipts(
            profile=profile,
            catalog=catalog,
            row=source_row,
        )
        self.assertEqual(len(receipts), 1)
        resume_context = _annotation_context_manifest(
            profile=profile,
            catalog=catalog,
            rows=[source_row],
            research_guidance="",
            target_mention_receipts=[],
        )
        targeted_guidance = "Исправить только подтверждённый answer_id 580."
        repair_context = _annotation_context_manifest(
            profile=profile,
            catalog=catalog,
            rows=[source_row],
            research_guidance=targeted_guidance,
            target_mention_receipts=receipts,
            repair_mode=ANALYSIS_CRITIC_TARGETED_REPAIR_MODE,
            allowed_answer_ids={580},
        )
        plan = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "target_answer_ids": [580],
            "guidance": targeted_guidance,
        }
        target_receipts_sha256 = _stable_json_sha256(receipts)
        resume_receipts_sha256 = _stable_json_sha256([])
        recovery_step = {
            "iteration": MAX_CRITIC_ITERATIONS + 1,
            "kind": "orchestrated_targeted_annotation_repair",
            "orchestrator_epoch": 1,
            "target_answer_ids": [580],
            "critic_adjustments": [],
            "annotation_guidance": targeted_guidance,
            "raw_corpus_sha256": _raw_corpus_digest([source_row]),
            "target_mention_receipts_sha256": target_receipts_sha256,
            "resume_target_mention_receipts_sha256": resume_receipts_sha256,
        }
        provenance = {
            "version": "analysis-critic-targeted-repair-v1",
            "orchestrator_epoch": 1,
            "orchestrator_plan_digest": stable_digest(plan),
            "target_answer_ids": [580],
            "repair_annotation_input_sha256": repair_context[
                "annotation_input_sha256"
            ],
            "resume_annotation_input_sha256": resume_context[
                "annotation_input_sha256"
            ],
            "guidance_sha256": hashlib.sha256(
                targeted_guidance.encode("utf-8")
            ).hexdigest(),
            "critic_review_sha256": stable_digest({}),
            "target_mention_receipts_sha256": target_receipts_sha256,
            "resume_target_mention_receipts_sha256": resume_receipts_sha256,
            "recovery_policy_step": recovery_step,
        }
        resume_row = copy.deepcopy(source_row)
        resume_row["annotation"].update(
            {
                "_annotation_version": ANNOTATION_VERSION,
                "_answer_sha256": hashlib.sha256(
                    MAKAR_RAW.encode("utf-8")
                ).hexdigest(),
                "_answer_model": source_row["model"],
                "_annotation_input_sha256": resume_context[
                    "annotation_input_sha256"
                ],
            }
        )
        repaired_row = copy.deepcopy(resume_row)
        repaired_row["annotation"].update(
            {
                "_annotation_input_sha256": repair_context[
                    "annotation_input_sha256"
                ],
                "_annotation_repair_provenance": provenance,
                "_critic_target_mention_receipts": receipts,
                "target_mentioned": True,
                "target_role": "mentioned",
            }
        )
        facts = {
            "site_profile": profile,
            "entity_catalog": catalog,
            "critic_review": {},
            "prior_policy_history": [],
            "resume_annotation_context": resume_context,
            "raw_corpus_sha256": _raw_corpus_digest([source_row]),
        }
        checkpoint = {
            "version": ANALYSIS_CRITIC_RECOVERY_CHECKPOINT_VERSION,
            "phase": "prepared",
            "orchestrator_epoch": 1,
            "orchestrator_plan_digest": stable_digest(plan),
            "target_answer_ids": [580],
            "resume_annotation_context": resume_context,
            "repair_annotation_context": repair_context,
            "annotation_repair_provenance": provenance,
            "resume_annotation_input_sha256_by_answer_id": {
                "580": resume_context["annotation_input_sha256"]
            },
            "annotation_input_sha256_by_answer_id": {
                "580": repair_context["annotation_input_sha256"]
            },
        }
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key=ANALYSIS_CRITIC_RECOVERY_STAGE,
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    failure_fingerprint="a" * 64,
                    facts_digest="b" * 64,
                    status="executing",
                    input_json={"incident": {"facts": facts}},
                    plan_json=plan,
                    plan_digest=stable_digest(plan),
                    outcome_json={
                        "execution_attempts": 1,
                        "stage_execution_attempts": 1,
                        "stage_execution_limit": 1,
                    },
                )
            )
            session.add(
                RunArtifact(
                    run_id=self.run_id,
                    stage_key="knowledge_gap",
                    artifact_key="analysis_critic_policy",
                    status="completed",
                    model=CRITIC_MODEL,
                    prompt_version=ANALYSIS_CRITIC_VERSION,
                    input_json={
                        "iteration": MAX_CRITIC_ITERATIONS + 1,
                        "executing_recovery_checkpoint_sha256": (
                            _stable_json_sha256(checkpoint)
                        ),
                    },
                    output_json={
                        "base_catalog_version": "fixture",
                        "policy_history": [],
                        "effective_profile": profile,
                        "effective_catalog": catalog,
                        "annotation_context": resume_context,
                        "annotation_context_sha256": _stable_json_sha256(
                            resume_context
                        ),
                        "executing_recovery": checkpoint,
                    },
                )
            )
            await session.commit()

        with (
            patch(
                "app.services.analyzer._metric_rows",
                new=AsyncMock(
                    side_effect=[
                        [resume_row],
                        [resume_row],
                        [resume_row],
                        [repaired_row],
                    ]
                ),
            ),
            patch(
                "app.services.analyzer._annotate_answers",
                new_callable=AsyncMock,
            ) as annotate,
        ):
            loaded = await _load_executing_analysis_critic_checkpoint(
                self.run_id,
                profile=profile,
            )

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded[1], [repaired_row])
        annotate.assert_awaited_once()
        self.assertEqual(
            annotate.await_args.kwargs["target_mention_receipts"],
            receipts,
        )
        self.assertEqual(
            annotate.await_args.kwargs["annotation_repair_provenance"],
            provenance,
        )

    async def test_paid_call_audits_are_appended_before_parent_failure(
        self,
    ) -> None:
        payload = {"site_profile": {"brand_name": "Example"}}
        leaf_raw = '{"verdict":"pass"}'
        leaf_usage = {
            "prompt_tokens": 9,
            "completion_tokens": 4,
            "total_tokens": 13,
        }

        reducer_raw = '{"verdict":"pass","summary":"reduced"}'
        reducer_usage = {
            "prompt_tokens": 14,
            "completion_tokens": 7,
            "total_tokens": 21,
        }

        async def fail_after_paid_calls(
            _payload,
            *,
            iteration,
            recovery_final,
            audit_sink,
            **_transport,
        ):
            self.assertEqual(iteration, 1)
            self.assertFalse(recovery_final)
            await audit_sink(
                {
                    "version": CRITIC_CALL_AUDIT_VERSION,
                    "attempt_id": "a" * 32,
                    "iteration": iteration,
                    "kind": "leaf",
                    "index": 0,
                    "status": "completed",
                    "model": CRITIC_MODEL,
                    "input": {"leaf": 0, "answers": [{"answer_id": 1}]},
                    "input_sha256": "b" * 64,
                    "lineage": {"assigned_answer_ids": [1]},
                    "output": _critic_review("pass"),
                    "raw_text": leaf_raw,
                    "raw_response_sha256": hashlib.sha256(
                        leaf_raw.encode("utf-8")
                    ).hexdigest(),
                    "raw_response_chars": len(leaf_raw),
                    "provider_response_present": True,
                    "usage": leaf_usage,
                    "error_type": None,
                    "error_message": None,
                }
            )
            await audit_sink(
                {
                    "version": CRITIC_CALL_AUDIT_VERSION,
                    "attempt_id": "c" * 32,
                    "iteration": iteration,
                    "kind": "corpus_reducer",
                    "index": 0,
                    "status": "completed",
                    "model": CRITIC_MODEL,
                    "input": {"reduction_tree": {"lineage": [1]}},
                    "input_sha256": "d" * 64,
                    "lineage": {
                        "sources": [1],
                        "tree_level": 0,
                    },
                    "output": _critic_review("pass"),
                    "raw_text": reducer_raw,
                    "raw_response_sha256": hashlib.sha256(
                        reducer_raw.encode("utf-8")
                    ).hexdigest(),
                    "raw_response_chars": len(reducer_raw),
                    "provider_response_present": True,
                    "usage": reducer_usage,
                    "error_type": None,
                    "error_message": None,
                }
            )
            raise OpenRouterError("synthetic reducer sibling failure")

        with patch(
            "app.services.analyzer.review_analysis",
            new_callable=AsyncMock,
            side_effect=fail_after_paid_calls,
        ):
            with bind_run_lease(self.run_id, "current-owner"):
                result = await _analysis_critic_artifact(
                    self.run_id,
                    iteration=1,
                    payload=payload,
                )

        self.assertEqual(result["verdict"], "block")
        self.assertEqual(
            result["fallback"]["kind"],
            "deterministic_degraded_advisory",
        )
        self.assertIn(
            "critic_provider_schema_or_cache_unavailable",
            result["fallback"]["reason_codes"],
        )

        async with self.SessionLocal() as session:
            artifacts = list(
                (
                    await session.execute(
                        select(RunArtifact)
                        .where(RunArtifact.run_id == self.run_id)
                        .order_by(RunArtifact.artifact_key)
                    )
                )
                .scalars()
                .all()
            )
        leaf_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.artifact_key.startswith("analysis_critic_r1_call_leaf_")
        ]
        self.assertEqual(len(leaf_artifacts), 1)
        leaf = leaf_artifacts[0]
        self.assertEqual(leaf.status, "completed")
        self.assertEqual(leaf.prompt_version, CRITIC_CALL_AUDIT_VERSION)
        self.assertEqual(leaf.raw_text, leaf_raw)
        self.assertEqual(leaf.usage_json, leaf_usage)
        self.assertEqual(leaf.output_json["status"], "completed")
        reducer = next(
            artifact
            for artifact in artifacts
            if artifact.artifact_key.startswith(
                "analysis_critic_r1_call_corpus_reducer_"
            )
        )
        self.assertEqual(reducer.status, "completed")
        self.assertEqual(reducer.raw_text, reducer_raw)
        self.assertEqual(reducer.usage_json, reducer_usage)
        self.assertEqual(reducer.output_json["lineage"]["sources"], [1])
        parent = next(
            artifact
            for artifact in artifacts
            if artifact.artifact_key == "analysis_critic_r1"
        )
        self.assertEqual(parent.status, "failed")
        self.assertIn("synthetic reducer sibling failure", parent.error_message)

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
                annotation_context=_annotation_context_manifest(
                    profile=PROFILE,
                    catalog=scoped_catalog,
                    rows=repaired_rows,
                    research_guidance="",
                    target_mention_receipts=[],
                ),
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
                annotation_context=_annotation_context_manifest(
                    profile=PROFILE,
                    catalog=scoped_catalog,
                    rows=repaired_rows,
                    research_guidance="",
                    target_mention_receipts=[],
                ),
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
            scoped_catalog,
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

    async def test_crash_resume_rejects_tampered_receipt_manifest_before_r3(
        self,
    ) -> None:
        (
            scoped_catalog,
            repaired_rows,
            plan,
            _recovery_step,
            before_digest,
            _state_digest,
        ) = self._recovered_fixture()
        repaired_rows[0]["annotation"]["_annotation_repair_provenance"][
            "target_mention_receipts_sha256"
        ] = "0" * 64
        tampered_state_digest = _critic_analysis_state_digest(
            repaired_rows,
            METRICS,
            profile=PROFILE,
            catalog=scoped_catalog,
        )
        await self._insert_recovery_epoch(
            status="executing",
            plan=plan,
            before_digest=before_digest,
            state_digest=tampered_state_digest,
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
            with self.assertRaisesRegex(
                _AnalysisCriticRecoveryBlocked,
                "receipt or annotation provenance",
            ):
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )

        critic.assert_not_awaited()
        progress.assert_not_awaited()

    async def test_crash_resume_rejects_tampered_facts_scope_before_r3(
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
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
            tampered_input = copy.deepcopy(epoch.input_json)
            tampered_input["incident"]["facts"]["prior_policy_history"] = [
                {"kind": "tampered-after-crash"}
            ]
            epoch.input_json = tampered_input
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
            with self.assertRaisesRegex(
                _AnalysisCriticRecoveryBlocked,
                "facts scope digest",
            ):
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )

        critic.assert_not_awaited()
        progress.assert_not_awaited()

    async def test_crash_during_r3_reservation_publishes_degraded_without_call(
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
            _catalog, returned_rows, returned_metrics, gate = (
                await _run_analysis_critic_loop(
                    self.run_id,
                    profile=PROFILE,
                    catalog=CATALOG,
                    rows=repaired_rows,
                    metrics=METRICS,
                )
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
            gate_output = (
                await session.execute(
                    select(RunArtifact.output_json).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "analysis_critic_gate",
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch_status, "failed")
        self.assertEqual(gate_status, "completed")
        self.assertEqual(gate_output["quality_state"], "degraded")
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["quality_state"], "degraded")
        self.assertEqual(returned_rows, repaired_rows)
        self.assertEqual(returned_metrics, METRICS)

    async def _insert_terminal_analysis_block(
        self,
        *,
        state_digest: str,
        confirmed: bool = True,
    ) -> None:
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
                            **(
                                {
                                    "terminal_analysis_critic_reason_code": (
                                        "confirmed_integrity_block"
                                    ),
                                    "terminal_integrity_codes": [
                                        "corpus_lineage:annotation_raw_hash_mismatch"
                                    ],
                                }
                                if confirmed
                                else {}
                            ),
                        },
                    },
                )
            )
            await session.commit()

    async def test_legacy_model_only_terminal_boolean_is_ignored(self) -> None:
        state_digest = _critic_analysis_state_digest(
            ROWS,
            METRICS,
            profile=PROFILE,
            catalog=_scope_entity_catalog_to_profile(CATALOG, PROFILE),
        )
        await self._insert_terminal_analysis_block(
            state_digest=state_digest,
            confirmed=False,
        )

        self.assertIsNone(
            await _terminal_analysis_critic_recovery_reason(
                self.run_id,
                state_digest=state_digest,
            )
        )

    async def test_terminal_post_repair_state_blocks_resume_before_r1(
        self,
    ) -> None:
        state_digest = _critic_analysis_state_digest(
            ROWS,
            METRICS,
            profile=PROFILE,
            catalog=_scope_entity_catalog_to_profile(CATALOG, PROFILE),
        )
        await self._insert_terminal_analysis_block(
            state_digest=state_digest,
        )

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

    async def test_terminal_latch_is_bound_to_profile_catalog_and_raw_corpus(
        self,
    ) -> None:
        scoped_catalog = _scope_entity_catalog_to_profile(CATALOG, PROFILE)
        baseline_state = _critic_analysis_state_digest(
            ROWS,
            METRICS,
            profile=PROFILE,
            catalog=scoped_catalog,
        )
        await self._insert_terminal_analysis_block(state_digest=baseline_state)

        changed_profile = copy.deepcopy(PROFILE)
        changed_profile["brand_name"] = "Corrected brand"
        changed_catalog = copy.deepcopy(scoped_catalog)
        changed_catalog["entities"][0]["aliases"] = ["Corrected alias"]
        changed_rows = copy.deepcopy(ROWS)
        changed_rows[0]["answer_text"] += " Corrected raw evidence."
        variants = (
            (changed_profile, scoped_catalog, ROWS),
            (PROFILE, changed_catalog, ROWS),
            (PROFILE, scoped_catalog, changed_rows),
        )

        for profile, catalog, rows in variants:
            changed_state = _critic_analysis_state_digest(
                rows,
                METRICS,
                profile=profile,
                catalog=catalog,
            )
            self.assertNotEqual(changed_state, baseline_state)
            self.assertIsNone(
                await _terminal_analysis_critic_recovery_reason(
                    self.run_id,
                    state_digest=changed_state,
                )
            )

    async def test_terminal_scope_upgrade_permits_a_fresh_r1(self) -> None:
        baseline_state = _critic_analysis_state_digest(
            ROWS,
            METRICS,
            profile=PROFILE,
            catalog=_scope_entity_catalog_to_profile(CATALOG, PROFILE),
        )
        await self._insert_terminal_analysis_block(
            state_digest=baseline_state,
        )
        changed_panel_contract = [
            {
                "prompt_id": 7,
                "provider_key": "openai",
                "model": "openai/gpt-chat-latest",
                "mode": "web",
            },
            {
                "prompt_id": 7,
                "provider_key": "gemini",
                "model": "google/gemini-next",
                "mode": "web",
            },
        ]
        variants = (
            (
                "critic_version",
                patch(
                    "app.services.analyzer.ANALYSIS_CRITIC_VERSION",
                    ANALYSIS_CRITIC_VERSION + "-next",
                ),
                None,
            ),
            (
                "critic_model",
                patch(
                    "app.services.analyzer.CRITIC_MODEL",
                    CRITIC_MODEL + "-next",
                ),
                None,
            ),
            (
                "map_reduce_policy",
                patch(
                    "app.services.analyzer.CRITIC_MAP_REDUCE_VERSION",
                    CRITIC_MAP_REDUCE_VERSION + "-next",
                ),
                None,
            ),
            (
                "recovery_policy",
                patch(
                    "app.services.analyzer."
                    "ANALYSIS_CRITIC_RECOVERY_POLICY_VERSION",
                    "aiv-analysis-critic-recovery-policy-next",
                ),
                None,
            ),
            (
                "expected_panel_contract",
                nullcontext(),
                changed_panel_contract,
            ),
        )

        for label, scope_patch, expected_cells in variants:
            with (
                self.subTest(scope_change=label),
                scope_patch,
                patch(
                    "app.services.analyzer._analysis_critic_artifact",
                    new_callable=AsyncMock,
                    return_value=_critic_review("pass"),
                ) as critic,
                patch(
                    "app.services.analyzer._save_critic_gate",
                    new_callable=AsyncMock,
                    return_value={"passed": True, "iteration": 1},
                ),
                patch(
                    "app.services.analyzer.update_progress",
                    new_callable=AsyncMock,
                ),
            ):
                _catalog, _rows, _metrics, gate = (
                    await _run_analysis_critic_loop(
                        self.run_id,
                        profile=PROFILE,
                        catalog=CATALOG,
                        rows=ROWS,
                        metrics=METRICS,
                        expected_corpus_cells=expected_cells,
                    )
                )

            self.assertTrue(gate["passed"])
            self.assertEqual(critic.await_count, 1)
            self.assertEqual(critic.await_args.kwargs["iteration"], 1)


if __name__ == "__main__":
    unittest.main()
