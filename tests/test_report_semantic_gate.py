import copy
import hashlib
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, func, select

from app.db import SessionLocal, init_db
from app.models import Run, RunArtifact, RunStatus
from app.services.analyzer import (
    _eligible_illustration_answer_context,
    _final_report,
    _load_final_semantic_part_checkpoint,
    _load_final_semantic_physical_result,
    _persist_final_semantic_audit_event,
    _select_final_answer_context,
)
from app.services.openrouter import (
    PHYSICAL_POST_AUDIT_VERSION,
    OpenRouterError,
    OutputTokenPolicy,
)
from app.services.report_semantic_gate import (
    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
    CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION,
    REPORT_SEMANTIC_MODEL,
    REPORT_SEMANTIC_PARTITION_VERSION,
    REPORT_SEMANTIC_REASONING_EFFORT,
    REPORT_SEMANTIC_REVIEW_SYSTEM,
    _reduce_semantic_receipts,
    _semantic_atomic_claim_spans,
    _semantic_evidence_path_contract,
    _semantic_disposition_manifest,
    _semantic_exact_output_utf8_bytes,
    _semantic_final_user_payload,
    _semantic_finding_ledger_entry,
    _semantic_finding_ledger_manifest,
    _semantic_json_sha256,
    _semantic_part_receipt,
    _semantic_partition_parts,
    _semantic_prepare_finding_shards,
    _semantic_reducer_user_payload,
    _semantic_required_summary_tokens,
    _semantic_summary_tokens,
    _semantic_validate_fragment_reconstruction,
    _validate_semantic_final_response,
    _validate_semantic_part_response,
    _validate_semantic_partition_coverage,
    _validate_semantic_reducer_result,
    deterministic_report_semantic_errors,
    metric_availability_contract,
    normalize_report_semantic_review,
    report_semantic_blockers,
    review_final_report_semantics,
    validate_report_semantic_review,
)


def _reduced_material_findings(user_payload: dict) -> list[dict]:
    if not user_payload["input_finding_manifest"]:
        return []
    nodes = {
        node["node_id"]: node for node in user_payload["source_nodes"]
    }
    source_ids: list[str] = []
    evidence_paths: list[str] = []
    semantic_literals: list[str] = []
    for item in user_payload["input_finding_manifest"]:
        source = nodes[item["source_node_id"]]["material_findings"][
            item["finding_index"]
        ]
        source_paths = (
            list(source.get("evidence_paths") or [])
            if isinstance(source, dict)
            else []
        )
        if isinstance(source, dict):
            for key in ("claim", "interpretation", "statement"):
                literal = source.get(key)
                if (
                    isinstance(literal, str)
                    and literal
                    and literal not in semantic_literals
                ):
                    semantic_literals.append(literal)
        elif isinstance(source, str) and source not in semantic_literals:
            semantic_literals.append(source)
        source_ids.append(item["source_finding_id"])
        for path in source_paths:
            if path not in evidence_paths:
                evidence_paths.append(path)
    return [
        {
            "source_finding_ids": source_ids,
            "statement": "\n".join(semantic_literals)
            or "Все входные material findings учтены без пропуска.",
            "evidence_paths": evidence_paths,
        }
    ]


def _grounded_reducer_summary(user_payload: dict) -> str:
    anchors: list[str] = []
    for node in user_payload.get("source_nodes") or []:
        for finding in node.get("material_findings") or []:
            tokens = sorted(_semantic_required_summary_tokens(finding))
            if tokens:
                anchors.extend(tokens)
    return " ".join(anchors) or "Нет материальных findings."


class FinalSemanticAuditStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_id = f"semantic-audit-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=self.run_id,
                    domain="semantic-audit.example",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id == self.run_id))
            await session.commit()

    async def test_physical_and_part_receipts_are_durable_and_resumable(
        self,
    ) -> None:
        request_payload = {
            "model": REPORT_SEMANTIC_MODEL,
            "messages": [{"role": "user", "content": "audit"}],
        }
        request_sha256 = hashlib.sha256(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        raw_text = json.dumps(_pass_review(), ensure_ascii=False)
        physical_event = {
            "version": PHYSICAL_POST_AUDIT_VERSION,
            "event_id": "a" * 32,
            "event_kind": "provider_post",
            "logical_call_id": "b" * 32,
            "document_id": "semantic:test",
            "sequence": 0,
            "attempt": 1,
            "status": "accepted",
            "model": REPORT_SEMANTIC_MODEL,
            "request_payload": request_payload,
            "request_sha256": request_sha256,
            "response": {},
            "raw_text": raw_text,
            "usage": {"prompt_tokens": 3},
            "transport": {},
            "resume_contract": None,
            "error": None,
            "partial_text": "",
            "manifest": None,
            "aggregate_usage": {},
            "call_records": [],
        }
        candidate_sha256 = "c" * 64
        manifest_sha256 = "d" * 64
        semantic_event = {
            "version": REPORT_SEMANTIC_PARTITION_VERSION,
            "kind": "semantic_part_accepted",
            "candidate_sha256": candidate_sha256,
            "source_part_receipts_sha256": manifest_sha256,
            "part_receipt": {"part_id": "part-1", "part_index": 0},
            "request_sha256": request_sha256,
            "parsed_review": _pass_review(),
            "semantic_receipt": {"summary": "Проверено.", "claims": []},
            "raw_text": raw_text,
            "usage": {"prompt_tokens": 3},
        }
        with patch(
            "app.services.analyzer.assert_run_lease",
            new=AsyncMock(),
        ):
            await _persist_final_semantic_audit_event(
                self.run_id,
                attempt=1,
                event=physical_event,
            )
            await _persist_final_semantic_audit_event(
                self.run_id,
                attempt=1,
                event=physical_event,
            )
            await _persist_final_semantic_audit_event(
                self.run_id,
                attempt=1,
                event=semantic_event,
            )
            resumed = await _load_final_semantic_physical_result(
                self.run_id,
                attempt=1,
                descriptor={
                    "version": REPORT_SEMANTIC_PARTITION_VERSION,
                    "kind": "atomic-a1",
                    "model": REPORT_SEMANTIC_MODEL,
                    "request_payload": request_payload,
                    "request_sha256": request_sha256,
                    "response_schema_sha256": "e" * 64,
                    "schema_name": "audit",
                },
            )
            part_checkpoint = await _load_final_semantic_part_checkpoint(
                self.run_id,
                attempt=1,
            )

        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.text, raw_text)
        self.assertEqual(resumed.usage["prompt_tokens"], 3)
        self.assertEqual(
            part_checkpoint["accepted_parts"], [semantic_event]
        )
        async with SessionLocal() as session:
            physical_count = (
                await session.execute(
                    select(func.count(RunArtifact.id)).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key >= "frsg_a1_post_",
                        RunArtifact.artifact_key < "frsg_a1_post_\uffff",
                    )
                )
            ).scalar_one()
        self.assertEqual(physical_count, 1)


class SemanticPromptCoverageTests(unittest.TestCase):
    def test_prompt_requires_every_material_violation_without_count_cap(
        self,
    ) -> None:
        self.assertIn(
            "не ограничивай их\nколичество",
            REPORT_SEMANTIC_REVIEW_SYSTEM,
        )
        self.assertNotIn(
            "не более 16",
            REPORT_SEMANTIC_REVIEW_SYSTEM.casefold(),
        )


def _unavailable_slice() -> dict[str, object]:
    return {
        "score": None,
        "specific_rate": None,
        "state": "unknown",
        "data_state": "unavailable",
        "expected_answers": 0,
        "completed_answers": 0,
        "valid_answers": 0,
    }


def _public_report_with_unavailable_memory() -> dict[str, object]:
    unavailable = _unavailable_slice()
    return {
        "brand": {"name": "Example"},
        "discovery": {
            "parent": {"memory": dict(unavailable)},
            "portfolio": {"memory": dict(unavailable)},
            "paired_web_lift": {
                "n_pairs": 0,
                "parent": {"score_lift": None},
                "portfolio": {"score_lift": None},
            },
        },
        "brand_knowledge": {"memory": dict(unavailable)},
    }


def _public_report_with_unavailable_portfolio() -> dict[str, object]:
    unavailable = {
        **_unavailable_slice(),
        "expected_answers": 6,
        "completed_answers": 6,
        "valid_answers": 6,
        "unavailable_reason": "target_portfolio_unconfirmed",
    }
    available = {
        "score": 40.0,
        "specific_rate": 40.0,
        "state": "visible",
        "data_state": "complete",
        "expected_answers": 6,
        "completed_answers": 6,
        "valid_answers": 6,
    }
    return {
        "brand": {"name": "Example"},
        "portfolio_scope": {
            "state": "unavailable",
            "candidate_entities": 10,
            "confirmed_entities": 0,
            "rejected_entities": 10,
            "reason": "target_portfolio_unconfirmed",
        },
        "discovery": {
            "parent": {"web": dict(available), "memory": dict(available)},
            "portfolio": {
                "web": dict(unavailable),
                "memory": dict(unavailable),
            },
            "portfolio_scope": {
                "state": "unavailable",
                "candidate_entities": 10,
                "confirmed_entities": 0,
                "rejected_entities": 10,
                "reason": "target_portfolio_unconfirmed",
            },
            "model_consistency": {
                "value": None,
                "state": "unknown",
                "data_state": "unavailable",
                "unavailable_reason": "target_portfolio_unconfirmed",
            },
            "paired_web_lift": {
                "n_pairs": 6,
                "parent": {
                    "web": dict(available),
                    "memory": dict(available),
                    "score_lift": 0.0,
                },
                "portfolio": {
                    "web": dict(unavailable),
                    "memory": dict(unavailable),
                    "score_lift": None,
                    "state": "unknown",
                    "data_state": "unavailable",
                    "unavailable_reason": "target_portfolio_unconfirmed",
                },
            },
        },
        "brand_knowledge": {"memory": dict(available)},
    }


def _public_report_with_observational_memory() -> dict[str, object]:
    web = {
        "score": 60.0,
        "state": "visible",
        "data_state": "complete",
        "expected_answers": 4,
        "completed_answers": 4,
        "valid_answers": 4,
        "evidence_state": "legacy_retrieval_confirmed",
        "observational_answers": 0,
    }
    memory = {
        "score": 25.0,
        "specific_rate": 50.0,
        "state": "weak",
        "data_state": "limited",
        "expected_answers": 4,
        "completed_answers": 4,
        "valid_answers": 4,
        "evidence_state": "legacy_observational",
        "observational_answers": 4,
        "strict_no_web_verified": False,
        "limitation_reason": "legacy_memory_request_not_enforced",
    }
    return {
        "brand": {"name": "Example"},
        "discovery": {
            "parent": {"web": dict(web), "memory": dict(memory)},
            "portfolio": {"web": dict(web), "memory": dict(memory)},
            "paired_web_lift": {
                "n_pairs": 4,
                "data_state": "limited",
                "causal_interpretation_allowed": False,
                "parent": {
                    "web": dict(web),
                    "memory": dict(memory),
                    "score_lift": None,
                    "observed_difference": 35.0,
                },
                "portfolio": {
                    "web": dict(web),
                    "memory": dict(memory),
                    "score_lift": None,
                    "observed_difference": 35.0,
                },
            },
        },
        "brand_knowledge": {"memory": dict(memory)},
    }


def _candidate(summary: str) -> dict[str, object]:
    return {
        "headline": "Что показала проверка",
        "headline_emphasis": [],
        "verdict": summary,
        "executive_summary": summary,
        "sections": [{"heading": "Результат", "body": summary}],
        "actions": [
            {
                "priority": "now",
                "title": "Уточнить данные",
                "why": "Нужен проверяемый вывод.",
                "step": "Повторить анализ сохранённых ответов.",
                "evidence": "Срез ограничен.",
            }
        ],
        "limitations": ["Это экспресс-снимок."],
    }


def _pass_review() -> dict[str, object]:
    return {"verdict": "pass", "summary": "Противоречий нет.", "violations": []}


def _part_semantic_claims(
    part: dict[str, object],
) -> list[dict[str, object]]:
    return [
        {
            "report_path": span["report_path"],
            "claim": span["claim"],
            "evidence_paths": ["/report_data"],
            "interpretation": "Фрагмент сохранён для глобальной сверки.",
        }
        for span in _semantic_atomic_claim_spans(part)
    ]


def _revise_review() -> dict[str, object]:
    return {
        "verdict": "revise",
        "summary": "Недоступный срез превращён в вывод.",
        "violations": [
            {
                "code": "unavailable_metric_claim",
                "severity": "critical",
                "report_path": "/executive_summary",
                "claim": "Модели не помнят бренд без веба.",
                "evidence_paths": [
                    "/report_data/brand_knowledge/memory/data_state"
                ],
                "finding": "Для memory нет валидных ответов.",
                "repair_instruction": "Заменить вывод ограничением данных.",
            }
        ],
    }


class ReportSemanticGateUnitTests(unittest.TestCase):
    def test_final_context_withholds_unattested_raw_answer_and_evidence(self) -> None:
        def answer(
            answer_id: int,
            provider: str,
            *,
            eligible: bool,
            reason: str,
        ) -> dict[str, object]:
            return {
                "answer_id": answer_id,
                "prompt_id": answer_id,
                "provider_key": provider,
                "mode": "memory",
                "metric_eligible": eligible,
                "intent_class": "I",
                "scenario_role": "brand_diagnostic",
                "scenario_sequence": answer_id,
                "scenario": "Что известно о бренде?",
                "answer_text": f"secret raw answer {answer_id}",
                "citations": [{"url": "https://source.example"}],
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_role": "primary",
                    "sentiment": "positive",
                    "evidence": ["secret evidence"],
                },
                "panel_evidence": {
                    "reason": reason,
                    "sha256": f"panel-{answer_id}",
                },
                "provenance": {
                    "raw_answer_sha256": f"raw-{answer_id}",
                    "annotation_sha256": f"annotation-{answer_id}",
                },
            }

        selected, manifest = _select_final_answer_context(
            [
                answer(
                    1,
                    "legacy",
                    eligible=False,
                    reason="legacy_memory_request_not_enforced",
                ),
                answer(
                    2,
                    "attested",
                    eligible=True,
                    reason="memory_request_enforced",
                ),
            ],
            corpus_manifest={
                "digest": "full-corpus",
                "critic_rows_sha256": "critic-rows",
            },
            max_answers=2,
        )

        by_id = {item["answer_id"]: item for item in selected}
        legacy = by_id[1]
        self.assertEqual(legacy["requested_mode"], "memory")
        self.assertIsNone(legacy["verified_mode"])
        self.assertEqual(legacy["context_access"], "metadata_only")
        self.assertEqual(
            legacy["content_withheld_reason"],
            "legacy_memory_request_not_enforced",
        )
        self.assertNotIn("answer_text", legacy)
        self.assertNotIn("annotation", legacy)
        self.assertNotIn("citations", legacy)
        self.assertEqual(by_id[2]["context_access"], "full_text")
        self.assertEqual(by_id[2]["verified_mode"], "memory")
        self.assertIn("answer_text", by_id[2])
        self.assertEqual(manifest["selected_full_text_count"], 1)
        self.assertEqual(manifest["selected_metadata_only_count"], 1)

    def test_contract_makes_unavailable_and_unpaired_states_explicit(self) -> None:
        contract = metric_availability_contract(
            _public_report_with_unavailable_memory()
        )
        by_path = {item["path"]: item for item in contract}

        self.assertFalse(by_path["/brand_knowledge/memory"]["available"])
        self.assertFalse(
            by_path["/discovery/paired_web_lift"]["available"]
        )
        self.assertEqual(
            by_path["/discovery/paired_web_lift"]["signals"]["n_pairs"],
            0,
        )

    def test_unavailable_memory_cannot_be_published_as_a_result(self) -> None:
        errors = deterministic_report_semantic_errors(
            _candidate("Без веб-поиска модели не помнят бренд: 0%."),
            _public_report_with_unavailable_memory(),
        )

        self.assertTrue(errors)
        self.assertTrue(any("представлен как ноль" in item for item in errors))

    def test_unavailable_portfolio_false_zero_and_absence_claims_fail(
        self,
    ) -> None:
        report_data = _public_report_with_unavailable_portfolio()
        claims = (
            "Продуктовая видимость равна нулю: 0%.",
            "Продукты не названы ни разу.",
            "Ни одна модель не обнаруживает продукты компании.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                candidate = _candidate(claim)
                candidate["limitations"] = [
                    CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION
                ]
                errors = deterministic_report_semantic_errors(
                    candidate,
                    report_data,
                )
                self.assertTrue(errors)
                self.assertTrue(
                    any("продукт" in error.casefold() for error in errors)
                )

    def test_honest_unavailable_portfolio_limitation_is_allowed(self) -> None:
        candidate = _candidate(
            "В отчёте используются только подтверждённые показатели."
        )
        candidate["limitations"] = [
            CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION
        ]

        errors = deterministic_report_semantic_errors(
            candidate,
            _public_report_with_unavailable_portfolio(),
        )

        self.assertEqual(errors, [])

    def test_technical_delivery_copy_is_not_a_portfolio_claim(self) -> None:
        candidate = _candidate(
            "Правила robots.txt открыты, стены авторизации нет: барьер "
            "лежит не в разрешениях, а в том, что физически приходит по "
            "проводу."
        )
        candidate["limitations"] = [
            CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION
        ]

        errors = deterministic_report_semantic_errors(
            candidate,
            _public_report_with_unavailable_portfolio(),
        )

        self.assertEqual(errors, [])

    def test_unavailable_portfolio_is_checked_for_illustration_copy(self) -> None:
        errors = deterministic_report_semantic_errors(
            {"headline": "Модели не находят продукты компании."},
            _public_report_with_unavailable_portfolio(),
            enforce_report_contract=False,
        )

        self.assertTrue(errors)

    def test_availability_contract_marks_unavailable_portfolio_derivatives(
        self,
    ) -> None:
        contract = metric_availability_contract(
            _public_report_with_unavailable_portfolio()
        )
        by_path = {item["path"]: item for item in contract}

        self.assertFalse(
            by_path["/discovery/model_consistency"]["available"]
        )
        self.assertFalse(
            by_path["/discovery/paired_web_lift/portfolio"]["available"]
        )
        self.assertFalse(
            by_path["/discovery/portfolio/web"]["available"]
        )

    def test_honest_unavailable_memory_limitation_is_allowed(self) -> None:
        candidate = _candidate("В отчёте используются подтверждённые данные.")
        candidate["limitations"] = [
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        ]
        errors = deterministic_report_semantic_errors(
            candidate,
            _public_report_with_unavailable_memory(),
        )

        self.assertEqual(errors, [])

    def test_observational_memory_requires_explicit_transport_limitation(
        self,
    ) -> None:
        claims = (
            "Без веб-поиска модели знают бренд в 50% ответов.",
            "Модели узнают бренд из своих обучающих данных в 50% ответов.",
            "Даже не обращаясь к интернету, модели знают бренд в 50% ответов.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                errors = deterministic_report_semantic_errors(
                    _candidate(claim),
                    _public_report_with_observational_memory(),
                )
                self.assertTrue(errors)
                self.assertTrue(
                    any("legacy-observational" in error for error in errors)
                )

    def test_observational_limit_does_not_authorize_epistemic_claims(
        self,
    ) -> None:
        claims = (
            "В моделях закреплены сведения об услугах и рынке.",
            "Модели ориентируются в продуктах бренда.",
            "ИИ знаком с услугами компании.",
            "Бренд уже присутствует в знаниях Gemini.",
            "Внутри модели закрепилась информация о компании.",
            "Gemini знает бренд.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                candidate = _candidate(claim)
                candidate["limitations"] = [
                    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
                ]
                errors = deterministic_report_semantic_errors(
                    candidate,
                    _public_report_with_observational_memory(),
                )
                self.assertTrue(
                    any("устойчивое знание" in error for error in errors)
                )

    def test_observational_caveat_cannot_mask_a_later_strict_claim(
        self,
    ) -> None:
        candidate = _candidate(
            "Техническое отключение веба в том запуске не аттестовано. "
            "Модели знают бренд."
        )
        candidate["limitations"] = [
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
        ]

        errors = deterministic_report_semantic_errors(
            candidate,
            _public_report_with_observational_memory(),
        )

        self.assertTrue(any("Модели знают бренд" in error for error in errors))

    def test_observational_memory_allows_exact_descriptive_wording(self) -> None:
        candidate = _candidate(
            "В историческом срезе, запрошенном без веб-поиска, модели "
            "назвали бренд в 1 из 4 ответов."
        )
        candidate["limitations"] = [
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
        ]
        errors = deterministic_report_semantic_errors(
            candidate,
            _public_report_with_observational_memory(),
        )

        self.assertEqual(errors, [])

    def test_observational_pair_cannot_be_written_as_web_effect(self) -> None:
        claims = (
            "Веб-поиск повысил видимость бренда на 35 пунктов.",
            "Веб-поиск снизил видимость бренда на 10 пунктов.",
            "Веб-поиск не изменил видимость бренда.",
            "После подключения поиска видимость выросла на 35 пунктов.",
            "После включения веб-поиска видимость стала выше.",
            "Без поиска было 25%, после его включения стало 60%.",
            "Без веба результат был хуже.",
            "Собственные знания моделей дали результат выше.",
            "Доступ к интернету повысил видимость на 35 пунктов.",
            "Поисковый режим поднял видимость бренда.",
            "Поиск прибавил 20 пунктов.",
            "После выхода в интернет бренд стали называть чаще.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                candidate = _candidate(claim)
                candidate["limitations"] = [
                    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
                ]
                errors = deterministic_report_semantic_errors(
                    candidate,
                    _public_report_with_observational_memory(),
                )
                self.assertTrue(errors)
                self.assertTrue(
                    any("причинный эффект" in error for error in errors)
                )

    def test_observational_pair_allows_noncausal_descriptive_difference(self) -> None:
        candidate = _candidate(
            "В сопоставленном историческом срезе, запрошенном без "
            "веб-поиска, доля упоминаний составила 25%, а с веб-поиском — "
            "60%: наблюдаемая разница 35 п.п."
        )
        candidate["limitations"] = [
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
        ]
        errors = deterministic_report_semantic_errors(
            candidate,
            _public_report_with_observational_memory(),
        )

        self.assertEqual(errors, [])

    def test_labeled_web_slice_with_citations_is_not_a_causal_claim(self) -> None:
        claims = (
            "В брендовых сценариях с веб-поиском 15 из 15 ответов "
            "оказались по существу, 15 из 15 дали конкретику, 15 из 15 "
            "привели ссылки.",
            "С веб-поиском модели привели ссылки, а затем перешли к "
            "результатам аудита.",
            "С веб-поиском модели привели примеры и цитаты.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                candidate = _candidate(claim)
                candidate["limitations"] = [
                    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
                ]

                errors = deterministic_report_semantic_errors(
                    candidate,
                    _public_report_with_observational_memory(),
                )

                self.assertEqual(errors, [])

    def test_web_search_led_to_metric_growth_remains_causal(self) -> None:
        claims = (
            "Веб-поиск привёл к росту доли упоминаний.",
            "Веб-поиск приводил к росту доли рекомендаций.",
            "Веб-поиск приведёт к увеличению видимости.",
            "Веб-поиск не привёл к росту доли рекомендаций.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                candidate = _candidate(claim)
                candidate["limitations"] = [
                    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
                ]

                errors = deterministic_report_semantic_errors(
                    candidate,
                    _public_report_with_observational_memory(),
                )

                self.assertTrue(
                    any("причинный эффект" in error for error in errors)
                )

    def test_adversative_clause_cannot_hide_unavailable_memory_claim(self) -> None:
        errors = deterministic_report_semantic_errors(
            _candidate(
                "Срез памяти не измерен, но модели не знают бренд без веба."
            ),
            _public_report_with_unavailable_memory(),
        )

        self.assertTrue(errors)
        self.assertTrue(any("не знают бренд" in item for item in errors))

    def test_disabled_search_and_own_knowledge_wording_is_caught(self) -> None:
        report_data = _public_report_with_unavailable_memory()

        disabled = deterministic_report_semantic_errors(
            _candidate("При отключённом поиске модели показывают 0%."),
            report_data,
        )
        own_knowledge = deterministic_report_semantic_errors(
            _candidate("Срез собственных знаний моделей показывает 0%."),
            report_data,
        )

        self.assertTrue(disabled)
        self.assertTrue(own_knowledge)

    def test_future_memory_measurement_is_not_treated_as_a_result(self) -> None:
        errors = deterministic_report_semantic_errors(
            _candidate("Провести повторный тест без веб-поиска."),
            _public_report_with_unavailable_memory(),
        )

        self.assertTrue(errors)
        self.assertFalse(any("представлен как ноль" in item for item in errors))

    def test_past_measurement_wording_is_not_exempted_as_future_action(self) -> None:
        report_data = _public_report_with_unavailable_memory()
        claims = (
            "Проверка без веб-поиска показала 0%.",
            "Измерение памяти показало 0%.",
            "Модели сделали ноль упоминаний без веба.",
            "Мы проверили память: бренд не известен моделям.",
        )

        for claim in claims:
            with self.subTest(claim=claim):
                self.assertTrue(
                    deterministic_report_semantic_errors(
                        _candidate(claim),
                        report_data,
                    )
                )

    def test_noncanonical_limitations_require_canonical_copy(self) -> None:
        report_data = _public_report_with_unavailable_memory()
        limitations = (
            "Мы проверили режим без веба: данных недостаточно для вывода.",
            "Срез без веба: не измерен.",
            "Нет подтверждения режима без веба.",
            "Чтобы проверить знания без веба, нужно провести отдельный тест.",
        )

        for limitation in limitations:
            with self.subTest(limitation=limitation):
                errors = deterministic_report_semantic_errors(
                    _candidate(limitation),
                    report_data,
                )
                self.assertTrue(errors)
                self.assertFalse(
                    any("представлен как ноль" in item for item in errors)
                )

    def test_leading_although_cannot_hide_unavailable_memory_outcome(self) -> None:
        errors = deterministic_report_semantic_errors(
            _candidate(
                "Хотя срез памяти не измерен, модели не знают бренд."
            ),
            _public_report_with_unavailable_memory(),
        )

        self.assertTrue(errors)

    def test_model_pass_cannot_override_deterministic_missing_data_gate(self) -> None:
        blockers = report_semantic_blockers(
            _candidate("Память моделей не знает бренд."),
            _public_report_with_unavailable_memory(),
            _pass_review(),
        )

        self.assertTrue(blockers)
        self.assertTrue(
            any("без доступного среза" in blocker for blocker in blockers)
        )

    def test_review_contract_rejects_pass_with_blocking_violation(self) -> None:
        review = _revise_review()
        review["verdict"] = "pass"

        errors = validate_report_semantic_review(review)

        self.assertIn(
            "Verdict pass несовместим с важными нарушениями.",
            errors,
        )

    def test_model_cannot_downgrade_inherently_blocking_code(self) -> None:
        review = _revise_review()
        review["verdict"] = "pass"
        review["violations"][0]["severity"] = "observation"

        errors = validate_report_semantic_review(review)

        self.assertIn(
            "Verdict pass несовместим с важными нарушениями.",
            errors,
        )

    def test_malformed_duplicate_is_removed_only_when_complete_finding_exists(
        self,
    ) -> None:
        review = _revise_review()
        duplicate = {
            "code": "unavailable_metric_claim",
            "severity": review["violations"][0]["severity"],
            "report_path": "/executive_summary",
            "claim": review["violations"][0]["claim"],
        }
        review["violations"].append(duplicate)

        normalized = normalize_report_semantic_review(review)

        self.assertEqual(len(normalized["violations"]), 1)
        self.assertEqual(validate_report_semantic_review(normalized), [])

        only_malformed = {
            "verdict": "revise",
            "summary": "Нужна правка.",
            "violations": [duplicate],
        }
        self.assertEqual(
            normalize_report_semantic_review(only_malformed)["violations"],
            [duplicate],
        )
        self.assertTrue(validate_report_semantic_review(only_malformed))

    def test_complete_violations_with_same_code_and_path_are_never_deduped(
        self,
    ) -> None:
        review = _revise_review()
        second = dict(review["violations"][0])
        second.update(
            {
                "severity": "critical",
                "claim": "Отдельное более серьёзное утверждение.",
                "finding": "Это самостоятельное критическое нарушение.",
            }
        )
        review["violations"].append(second)

        normalized = normalize_report_semantic_review(review)

        self.assertEqual(normalized["violations"], review["violations"])
        self.assertEqual(len(normalized["violations"]), 2)

    def test_malformed_same_path_but_different_claim_remains_fail_closed(
        self,
    ) -> None:
        review = _revise_review()
        incomplete_distinct_finding = {
            "code": review["violations"][0]["code"],
            "severity": review["violations"][0]["severity"],
            "report_path": review["violations"][0]["report_path"],
            "claim": "Другое потенциальное нарушение.",
        }
        review["violations"].append(incomplete_distinct_finding)

        normalized = normalize_report_semantic_review(review)

        self.assertEqual(len(normalized["violations"]), 2)
        self.assertTrue(validate_report_semantic_review(normalized))

    def test_honest_unavailable_limitations_override_critic_false_positives(
        self,
    ) -> None:
        report_data = _public_report_with_unavailable_memory()
        limitation_claim = (
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        )
        candidate = {"limitations": [limitation_claim]}
        review = {
            "verdict": "revise",
            "summary": "Нужна правка.",
            "violations": [
                {
                    "code": "unavailable_metric_claim",
                    "severity": "important",
                    "report_path": "/limitations/0",
                    "claim": limitation_claim,
                    "evidence_paths": [
                        "/report_data/brand_knowledge/memory sculpture"
                    ],
                    "finding": "Критик ошибочно счёл ограничение выводом.",
                    "repair_instruction": "Удалить ограничение.",
                }
            ],
        }
        evidence_document = {"report_data": report_data}

        normalized = normalize_report_semantic_review(
            review,
            evidence_document=evidence_document,
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "pass")
        self.assertEqual(normalized["violations"], [])
        self.assertEqual(
            validate_report_semantic_review(
                normalized,
                evidence_document=evidence_document,
                candidate_report=candidate,
            ),
            [],
        )

    def test_missing_repair_instruction_is_filled_without_losing_blocker(
        self,
    ) -> None:
        review = _revise_review()
        review["violations"][0].pop("repair_instruction")

        normalized = normalize_report_semantic_review(review)

        violation = normalized["violations"][0]
        self.assertTrue(violation["repair_instruction"])
        self.assertEqual(normalized["verdict"], "revise")
        self.assertEqual(validate_report_semantic_review(normalized), [])

    def test_missing_evidence_leaf_uses_deepest_existing_parent(self) -> None:
        candidate = {
            "sections": [
                {
                    "body": "Вывод требует уточнения.",
                }
            ]
        }
        evidence_document = {
            "report_data": {
                "technical": {
                    "findings": [
                        {
                            "title": "На главной мало серверного текста",
                            "detail": "Проверен исходный HTML.",
                        }
                    ]
                }
            }
        }
        review = {
            "verdict": "revise",
            "summary": "Нужна правка.",
            "violations": [
                {
                    "code": "scope_overreach",
                    "severity": "important",
                    "report_path": "/sections/0/body",
                    "claim": "Вывод требует уточнения.",
                    "evidence_paths": [
                        "/report_data/technical/findings/0/business_effect"
                    ],
                    "finding": "Эффект сформулирован шире наблюдения.",
                    "repair_instruction": "Ограничить вывод проверенным срезом.",
                }
            ],
        }

        normalized = normalize_report_semantic_review(
            review,
            evidence_document=evidence_document,
            candidate_report=candidate,
            report_data=evidence_document["report_data"],
        )

        self.assertEqual(
            normalized["violations"][0]["evidence_paths"],
            ["/report_data/technical/findings/0"],
        )
        self.assertEqual(
            validate_report_semantic_review(
                normalized,
                evidence_document=evidence_document,
                candidate_report=candidate,
            ),
            [],
        )

    def test_invented_evidence_branch_remains_invalid(self) -> None:
        candidate = {"sections": [{"body": "Вывод требует уточнения."}]}
        evidence_document = {"report_data": {"technical": {"findings": []}}}
        review = {
            "verdict": "revise",
            "summary": "Нужна правка.",
            "violations": [
                {
                    "code": "scope_overreach",
                    "severity": "important",
                    "report_path": "/sections/0/body",
                    "claim": "Вывод требует уточнения.",
                    "evidence_paths": [
                        "/report_data/invented/business_effect"
                    ],
                    "finding": "Нет такого доказательства.",
                    "repair_instruction": "Проверить источник.",
                }
            ],
        }

        normalized = normalize_report_semantic_review(
            review,
            evidence_document=evidence_document,
            candidate_report=candidate,
            report_data=evidence_document["report_data"],
        )

        self.assertEqual(
            normalized["violations"][0]["evidence_paths"],
            ["/report_data/invented/business_effect"],
        )
        self.assertTrue(
            validate_report_semantic_review(
                normalized,
                evidence_document=evidence_document,
                candidate_report=candidate,
            )
        )

    def test_content_free_placeholder_violation_is_ignored(self) -> None:
        review = {
            "verdict": "revise",
            "summary": "Критик вернул пустой второй элемент.",
            "violations": [
                {
                    "code": "causal_overreach",
                    "severity": "important",
                }
            ],
        }

        normalized = normalize_report_semantic_review(review)

        self.assertEqual(normalized["verdict"], "pass")
        self.assertEqual(normalized["violations"], [])
        self.assertEqual(validate_report_semantic_review(normalized), [])

    def test_explicit_noncausal_statement_overrides_critic_false_positive(
        self,
    ) -> None:
        report_data = _public_report_with_observational_memory()
        source = (
            "Это сопоставление двух наблюдений: в ответе имя юнита не "
            "связано с материнским брендом, а на сайте за юнитом закреплено "
            "только функциональное описание. Причинную связь такие данные "
            "не показывают."
        )
        candidate = _candidate(source)
        candidate["limitations"] = [
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
        ]
        review = {
            "verdict": "revise",
            "summary": "Критик ошибочно увидел причинный вывод.",
            "violations": [
                {
                    "code": "causal_overreach",
                    "severity": "important",
                    "report_path": "/candidate_report/sections/0/body",
                    "claim": source,
                    "evidence_paths": [
                        "/report_data/discovery/paired_web_lift/"
                        "causal_interpretation_allowed"
                    ],
                    "finding": "Причинная интерпретация запрещена.",
                }
            ],
        }
        evidence_document = {"report_data": report_data}

        normalized = normalize_report_semantic_review(
            review,
            evidence_document=evidence_document,
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "pass")
        self.assertEqual(normalized["violations"], [])
        self.assertEqual(
            validate_report_semantic_review(
                normalized,
                evidence_document=evidence_document,
                candidate_report=candidate,
            ),
            [],
        )

    def test_allowed_observational_aggregate_overrides_mode_false_positive(
        self,
    ) -> None:
        report_data = _public_report_with_observational_memory()
        source = (
            "Этот блок читаем строго как описание чисел. Срез запрашивался "
            "без веб-поиска: ссылок и сигналов обращения к веб-инструментам "
            "не обнаружено, однако техническое отключение веба в том запуске "
            "не аттестовано. Поэтому его агрегаты — наблюдение, а не строгое "
            "утверждение о знании модели."
        )
        claim = (
            "Поэтому его агрегаты — наблюдение, а не строгое утверждение о "
            "знании модели."
        )
        candidate = _candidate("Использованы подтверждённые данные.")
        candidate["sections"][0]["body"] = source
        candidate["limitations"] = [
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
        ]
        review = {
            "verdict": "revise",
            "summary": "Критик ошибочно потребовал скрыть агрегат.",
            "violations": [
                {
                    "code": "mode_substitution",
                    "severity": "important",
                    "report_path": "/candidate_report/sections/0/body",
                    "claim": claim,
                    "evidence_paths": [
                        "/report_data/brand_knowledge/memory/evidence_state"
                    ],
                    "finding": "Исторический режим не аттестован.",
                    "repair_instruction": "Скрыть числовой агрегат.",
                }
            ],
        }

        normalized = normalize_report_semantic_review(
            review,
            evidence_document={"report_data": report_data},
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "pass")
        self.assertEqual(normalized["violations"], [])

    def test_explicit_noncausal_and_nonepistemic_copy_passes_precheck(
        self,
    ) -> None:
        phrases = (
            "Все числа этого раздела описывают только срез с веб-поиском; "
            "они не сопоставляются с другим режимом запроса, и влияние "
            "веб-поиска на них отчёт не рассчитывает и не утверждает.",
            "Поэтому эти числа нельзя называть строгим показателем и нельзя "
            "объяснять наблюдаемую разницу эффектом веб-поиска.",
            "В историческом срезе, запрошенном без веб-поиска, доля "
            "упоминаний составила 0%, а с веб-поиском — тоже 0%. Это "
            "констатация двух величин: эффект веб-поиска не рассчитан, не "
            "доказан и из сопоставления не выводится.",
            "В историческом срезе, запрошенном без веб-поиска, конкретика "
            "есть в 4 ответах из 12. Ссылок и сигналов обращения к "
            "веб-инструментам не обнаружено, но техническое отключение веба "
            "не аттестовано, поэтому числа описывают только сам срез и не "
            "толкуются как знание или память модели.",
            "Этот блок читаем строго как описание чисел. Срез запрашивался "
            "без веб-поиска: ссылок и сигналов обращения к веб-инструментам "
            "не обнаружено, однако техническое отключение веба в том запуске "
            "не аттестовано. Поэтому его агрегаты — наблюдение, а не строгое "
            "утверждение о знании модели. Парное сопоставление: доля "
            "упоминаний 0 % в обоих срезах; наблюдаемая разница — 0 п. п. "
            "Это описательное сопоставление, эффектом веб-поиска его "
            "называть нельзя: расчётный прирост не определён.",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                candidate = _candidate("Использованы подтверждённые данные.")
                candidate["sections"][0]["body"] = phrase
                candidate["actions"][0]["step"] = (
                    "Передача медиабаинга на аутсорс с моделями "
                    "вознаграждения."
                )
                candidate["limitations"] = [
                    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION
                ]
                self.assertEqual(
                    deterministic_report_semantic_errors(
                        candidate,
                        _public_report_with_observational_memory(),
                    ),
                    [],
                )

    def test_boilerplate_critic_false_positives_do_not_block_safe_report(
        self,
    ) -> None:
        report_data = _public_report_with_observational_memory()
        action_source = (
            "Что делать. Связать каждое направление с зонтичным брендом. "
            "Тогда рекомендация сервиса начнёт работать на узнаваемость RW+."
        )
        address_claim = (
            "Тогда у любого читателя — человека или системы — появится один "
            "адрес, где числа и знаменатели заданы самим брендом."
        )
        technical_claim = (
            "В безбрендовом веб-срезе один ответ Gemini исключён из расчёта "
            "по технической причине «отсутствует подтверждение обращения к "
            "вебу» (legacy_web_evidence_missing), поэтому знаменатель равен "
            "29 учтённым ответам."
        )
        candidate = _candidate("Использованы подтверждённые данные.")
        candidate["sections"] = [
            {"heading": "Действие", "body": action_source},
            {"heading": "Факты", "body": address_claim},
        ]
        candidate["limitations"] = [
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
            technical_claim,
        ]
        review = {
            "verdict": "revise",
            "summary": "Критик неверно привязал безопасный текст к memory.",
            "violations": [
                {
                    "code": "causal_overreach",
                    "severity": "important",
                    "report_path": "/sections/0/body",
                    "claim": (
                        "Тогда рекомендация сервиса начнёт работать на "
                        "узнаваемость RW+."
                    ),
                    "evidence_paths": [
                        "/report_data/discovery/paired_web_lift"
                    ],
                    "finding": "Критик принял будущую рекомендацию за lift.",
                    "repair_instruction": "Удалить рекомендацию.",
                },
                {
                    "code": "mode_substitution",
                    "severity": "important",
                    "report_path": "/sections/1/body",
                    "claim": address_claim,
                    "evidence_paths": [
                        "/report_data/key_metrics/brand_knowledge"
                    ],
                    "finding": "Критик ошибочно увидел память модели.",
                    "repair_instruction": "Удалить слово «система».",
                },
                {
                    "code": "mode_substitution",
                    "severity": "important",
                    "report_path": "/limitations/1",
                    "claim": technical_claim,
                    "evidence_paths": [
                        "/selected_answer_context/0/panel_evidence/reason"
                    ],
                    "finding": "Критик ошибочно принял web-исключение за memory.",
                    "repair_instruction": "Переписать ограничение.",
                },
            ],
        }
        evidence_document = {
            "report_data": report_data,
            "selected_answer_context": [],
        }

        normalized = normalize_report_semantic_review(
            review,
            evidence_document=evidence_document,
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "pass")
        self.assertEqual(normalized["violations"], [])
        self.assertEqual(
            validate_report_semantic_review(
                normalized,
                evidence_document=evidence_document,
                candidate_report=candidate,
            ),
            [],
        )

    def test_real_causal_claim_is_not_hidden_by_a_later_disclaimer(self) -> None:
        report_data = _public_report_with_observational_memory()
        source = (
            "Веб-поиск повысил видимость бренда. Причинную связь такие "
            "данные не показывают."
        )
        candidate = _candidate(source)
        review = {
            "verdict": "revise",
            "summary": "Есть причинный вывод.",
            "violations": [
                {
                    "code": "causal_overreach",
                    "severity": "important",
                    "report_path": "/sections/0/body",
                    "claim": source,
                    "evidence_paths": [
                        "/report_data/discovery/paired_web_lift/"
                        "causal_interpretation_allowed"
                    ],
                    "finding": "Причинная интерпретация запрещена.",
                }
            ],
        }

        normalized = normalize_report_semantic_review(
            review,
            evidence_document={"report_data": report_data},
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "revise")
        self.assertEqual(len(normalized["violations"]), 1)
        self.assertTrue(normalized["violations"][0]["repair_instruction"])

    def test_real_unavailable_claim_is_kept_and_pointer_typo_is_repaired(
        self,
    ) -> None:
        report_data = _public_report_with_unavailable_memory()
        claim = "Без веба модели не помнят бренд: 0%."
        candidate = {"executive_summary": claim}
        review = {
            "verdict": "revise",
            "summary": "Недоступный срез превращён в ноль.",
            "violations": [
                {
                    "code": "unavailable_metric_claim",
                    "severity": "important",
                    "report_path": "/executive_summary",
                    "claim": claim,
                    "evidence_paths": [
                        "/report_data/brand_knowledge/memory sculpture"
                    ],
                    "finding": "Для memory нет валидных ответов.",
                    "repair_instruction": "Заменить вывод ограничением.",
                }
            ],
        }
        evidence_document = {"report_data": report_data}

        normalized = normalize_report_semantic_review(
            review,
            evidence_document=evidence_document,
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "revise")
        self.assertEqual(len(normalized["violations"]), 1)
        self.assertEqual(
            normalized["violations"][0]["evidence_paths"],
            ["/report_data/brand_knowledge/memory"],
        )
        self.assertTrue(
            report_semantic_blockers(
                candidate,
                report_data,
                normalized,
                evidence_document=evidence_document,
            )
        )

    def test_canonical_claim_cannot_hide_extra_outcome_in_source(self) -> None:
        report_data = _public_report_with_unavailable_memory()
        canonical = (
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        )
        tails = (
            " Поэтому бренд отсутствует в ответах.",
            " Значит бренд выпадает из ответов.",
            " Следовательно у бренда нет присутствия.",
        )
        evidence_document = {"report_data": report_data}

        for tail in tails:
            with self.subTest(tail=tail):
                candidate = {"limitations": [canonical + tail]}
                review = {
                    "verdict": "revise",
                    "summary": "Ограничение дополнено выводом.",
                    "violations": [
                        {
                            "code": "unavailable_metric_claim",
                            "severity": "important",
                            "report_path": "/limitations/0",
                            "claim": canonical,
                            "evidence_paths": [
                                "/report_data/brand_knowledge/memory"
                            ],
                            "finding": "После ограничения сделан вывод.",
                            "repair_instruction": "Удалить вывод.",
                        }
                    ],
                }

                normalized = normalize_report_semantic_review(
                    review,
                    evidence_document=evidence_document,
                    candidate_report=candidate,
                    report_data=report_data,
                )

                self.assertEqual(normalized["verdict"], "revise")
                self.assertEqual(len(normalized["violations"]), 1)

    def test_limitation_followed_by_memory_outcome_stays_blocking(self) -> None:
        report_data = _public_report_with_unavailable_memory()
        claims = (
            "Срез памяти не измерен, поэтому модели не знают бренд.",
            "Данных о памяти недостаточно, следовательно бренд моделям "
            "неизвестен.",
            "Срез без веба недоступен — модели не помнят бренд.",
            "Без веба данные не измерены и модели не знают бренд.",
            "Срез памяти не измерен, поэтому бренд отсутствует в знаниях "
            "моделей.",
            "Срез памяти не измерен, поэтому модели не располагают "
            "сведениями о бренде.",
            "Срез памяти недоступен — бренд выпадает из ответов моделей.",
            "Без веба данных недостаточно, значит у моделей нет сведений о "
            "бренде.",
            "Срез памяти не измерен, поэтому модели игнорируют бренд.",
            "Показатель знания бренда без веба равен 0% и не определён.",
            "Срез памяти показывает 0% и не вошёл: для него не удалось "
            "подтвердить отключение веба.",
            "Не считается разрыв памяти поэтому модели не знают бренд: "
            "сопоставимых пар систем нет.",
            "Вывод о памяти такой: модели не знают бренд и отрицать это "
            "нельзя.",
            "Нельзя утверждать что модели знают бренд без веба потому что "
            "они точно не знают его.",
            "Модели могут не знать бренд без веба и точно его не знают хотя "
            "данных недостаточно.",
        )

        for claim in claims:
            with self.subTest(claim=claim):
                candidate = {"executive_summary": claim}
                review = {
                    "verdict": "revise",
                    "summary": "Недоступный срез превращён в вывод.",
                    "violations": [
                        {
                            "code": "unavailable_metric_claim",
                            "severity": "important",
                            "report_path": "/executive_summary",
                            "claim": claim,
                            "evidence_paths": [
                                "/report_data/brand_knowledge/memory"
                            ],
                            "finding": "Для memory нет валидных ответов.",
                            "repair_instruction": "Оставить только ограничение.",
                        }
                    ],
                }
                evidence_document = {"report_data": report_data}

                normalized = normalize_report_semantic_review(
                    review,
                    evidence_document=evidence_document,
                    candidate_report=candidate,
                    report_data=report_data,
                )

                self.assertEqual(normalized["verdict"], "revise")
                self.assertEqual(len(normalized["violations"]), 1)
                self.assertTrue(
                    report_semantic_blockers(
                        candidate,
                        report_data,
                        normalized,
                        evidence_document=evidence_document,
                    )
                )

    def test_epistemic_uncertainty_is_not_treated_as_memory_outcome(self) -> None:
        report_data = _public_report_with_unavailable_memory()
        claims = (
            "Модели могли как знать бренд, так и не знать его: срез памяти "
            "не измерен.",
            "Модели могут знать бренд без веба, но данных недостаточно, чтобы "
            "это подтвердить.",
        )

        for claim in claims:
            with self.subTest(claim=claim):
                errors = deterministic_report_semantic_errors(
                    {"executive_summary": claim},
                    report_data,
                )
                self.assertTrue(errors)
                self.assertFalse(
                    any(
                        "ограничение данных смешано" in item
                        or "вывод о памяти моделей сделан" in item
                        for item in errors
                    )
                )

    def test_all_unavailable_memory_requires_one_canonical_limitation(
        self,
    ) -> None:
        report_data = _public_report_with_unavailable_memory()
        canonical = (
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        )
        neutral = _candidate("В отчёте используются подтверждённые данные.")
        variants = []

        missing = json.loads(json.dumps(neutral))
        missing["limitations"] = ["Это экспресс-снимок."]
        variants.append(missing)

        duplicated = json.loads(json.dumps(neutral))
        duplicated["limitations"] = [canonical, canonical]
        variants.append(duplicated)

        wrong_place = json.loads(json.dumps(neutral))
        wrong_place["executive_summary"] = canonical
        wrong_place["limitations"] = ["Это экспресс-снимок."]
        variants.append(wrong_place)

        section_heading = json.loads(json.dumps(neutral))
        section_heading["limitations"] = [canonical]
        section_heading["sections"][0]["heading"] = (
            "Без веба модели не знают бренд"
        )
        variants.append(section_heading)

        action_title = json.loads(json.dumps(neutral))
        action_title["limitations"] = [canonical]
        action_title["actions"][0]["title"] = "Память моделей равна 0%"
        variants.append(action_title)

        action_step = json.loads(json.dumps(neutral))
        action_step["limitations"] = [canonical]
        action_step["actions"][0]["step"] = (
            "Зафиксировать, что модели не помнят бренд без веба"
        )
        variants.append(action_step)

        for synonym in (
            "Модели не знают бренд без интернета.",
            "Модели не знают бренд в автономном режиме.",
            "Модели не знают бренд без подключения к интернету.",
            "Модели не знают бренд на основе только обучающих данных.",
            "Из параметрических знаний модели бренд не известен.",
            "Без интернета модели не знают бренд.",
            "В автономном режиме модели не знают бренд.",
            "Без подключения к интернету модели не знают бренд.",
            "Только на обучающих данных модели не знают бренд.",
        ):
            synonym_candidate = json.loads(json.dumps(neutral))
            synonym_candidate["limitations"] = [canonical]
            synonym_candidate["verdict"] = synonym
            variants.append(synonym_candidate)

        for candidate in variants:
            with self.subTest(candidate=candidate):
                self.assertTrue(
                    deterministic_report_semantic_errors(
                        candidate,
                        report_data,
                    )
                )

    def test_memory_contract_does_not_match_site_copy_homonyms(self) -> None:
        report_data = _public_report_with_unavailable_memory()
        canonical = (
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        )
        phrases = (
            "Подготовьте памятку для контент-команды.",
            "Добавьте памятный адрес страницы.",
            "Пользователь должен находить услугу без поиска по каталогу.",
            "Раздел работает без поиска по сайту.",
        )

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                candidate = _candidate("Использованы подтверждённые данные.")
                candidate["actions"][0]["step"] = phrase
                candidate["limitations"] = [canonical]
                self.assertEqual(
                    deterministic_report_semantic_errors(
                        candidate,
                        report_data,
                    ),
                    [],
                )

    def test_canonical_auto_drop_requires_all_memory_families_unavailable(
        self,
    ) -> None:
        report_data = _public_report_with_unavailable_memory()
        report_data["discovery"]["parent"]["memory"] = {
            "score": 50.0,
            "specific_rate": 50.0,
            "state": "measured",
            "data_state": "complete",
            "expected_answers": 6,
            "completed_answers": 6,
            "valid_answers": 6,
        }
        canonical = (
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        )
        candidate = {"limitations": [canonical]}
        review = {
            "verdict": "revise",
            "summary": "Обобщение не соответствует смешанному покрытию.",
            "violations": [
                {
                    "code": "unavailable_metric_claim",
                    "severity": "important",
                    "report_path": "/limitations/0",
                    "claim": canonical,
                    "evidence_paths": [
                        "/report_data/discovery/parent/memory"
                    ],
                    "finding": "Один memory-срез доступен.",
                    "repair_instruction": "Ограничить формулировку.",
                }
            ],
        }

        normalized = normalize_report_semantic_review(
            review,
            evidence_document={"report_data": report_data},
            candidate_report=candidate,
            report_data=report_data,
        )

        self.assertEqual(normalized["verdict"], "revise")
        self.assertEqual(len(normalized["violations"]), 1)

    def test_metadata_only_manifest_does_not_leak_annotation_signals(self) -> None:
        legacy = {
            "answer_id": 1,
            "prompt_id": 1,
            "provider_key": "legacy",
            "mode": "memory",
            "metric_eligible": False,
            "intent_class": "I",
            "scenario_role": "brand_diagnostic",
            "scenario_sequence": 1,
            "scenario": "Что известно о бренде?",
            "answer_text": "secret raw answer",
            "citations": [],
            "annotation": {
                "valid": True,
                "target_mentioned": True,
                "target_role": "primary",
                "sentiment": "positive",
            },
            "panel_evidence": {
                "reason": "legacy_memory_request_not_enforced",
                "sha256": "panel-1",
            },
            "provenance": {
                "raw_answer_sha256": "raw-1",
                "annotation_sha256": "annotation-1",
            },
        }

        selected, manifest = _select_final_answer_context(
            [legacy],
            corpus_manifest={"digest": "full", "critic_rows_sha256": "rows"},
            max_answers=1,
        )

        self.assertEqual(selected[0]["context_access"], "metadata_only")
        self.assertNotIn("target_role", manifest["observed_coverage"])
        self.assertNotIn("sentiment", manifest["selected_coverage"])
        self.assertNotIn("valid", manifest["selected_coverage"])
        self.assertNotIn("target_mentioned", manifest["selected_coverage"])

    def test_visual_context_withholds_unattested_raw_content(self) -> None:
        legacy = {
            "answer_id": 1,
            "provider_key": "legacy",
            "mode": "memory",
            "metric_eligible": False,
            "answer_text": "UNATTESTED-SENTINEL",
            "annotation": {"evidence": ["SECRET-EVIDENCE"]},
            "panel_evidence": {
                "reason": "legacy_memory_request_not_enforced"
            },
        }
        eligible = {
            "answer_id": 2,
            "provider_key": "current",
            "mode": "web",
            "metric_eligible": True,
            "answer_text": "eligible evidence",
            "annotation": {"evidence": ["public evidence"]},
            "panel_evidence": {"reason": "web_search_attested"},
        }

        selected, contract = _eligible_illustration_answer_context(
            [legacy, eligible]
        )
        serialized = json.dumps(selected, ensure_ascii=False)

        self.assertNotIn("UNATTESTED-SENTINEL", serialized)
        self.assertNotIn("SECRET-EVIDENCE", serialized)
        self.assertIn("eligible evidence", serialized)
        self.assertEqual(contract["withheld_metadata_only_count"], 1)
        self.assertFalse(contract["unattested_raw_content_included"])

    def test_visual_context_keeps_every_attested_long_answer(self) -> None:
        answers = [
            {
                "answer_id": index,
                "provider_key": "current",
                "mode": "web",
                "metric_eligible": True,
                "answer_text": f"answer-{index}-" + ("я" * 20_000),
                "annotation": {"evidence": [f"evidence-{index}"]},
                "panel_evidence": {"reason": "web_search_attested"},
            }
            for index in range(1, 22)
        ]

        selected, contract = _eligible_illustration_answer_context(answers)

        self.assertEqual(len(selected), len(answers))
        self.assertEqual(contract["selected_raw_context_count"], len(answers))
        self.assertEqual(
            contract["context_policy"],
            "all_attested_answers_no_local_cap_v1",
        )
        self.assertTrue(selected[-1]["answer_text"].endswith("я" * 20_000))


class SemanticPartitionCoverageTests(unittest.TestCase):
    def _partition_payload(self, summary: str) -> dict[str, object]:
        return {
            "evidence_document": {"report_data": {}},
            "model_evidence_context": {"report_data": {}},
            "metric_availability_contract": [],
            "candidate_report": _candidate(summary),
            "deterministic_precheck_errors": [],
        }

    def test_lossless_partition_covers_every_report_record_exactly(self) -> None:
        source = "Точный длинный фрагмент. " * 180
        payload = self._partition_payload(source)
        parts, manifest = _semantic_partition_parts(payload, target_chars=256)
        receipts = [
            _semantic_part_receipt(part, _pass_review()) for part in parts
        ]

        digest = _validate_semantic_partition_coverage(
            manifest,
            receipts,
            candidate_report=payload["candidate_report"],
            reviews=[_pass_review() for _part in parts],
        )

        self.assertEqual(len(digest), 64)
        self.assertEqual(manifest["part_count"], len(parts))
        self.assertEqual(
            manifest["record_count"],
            len({part["record_index"] for part in parts}),
        )
        section_parts = sorted(
            (
                part["unit_index"],
                part["context_text"][
                    part["core_start_in_context"] : part["core_end_in_context"]
                ],
            )
            for part in parts
            if part["report_path"] == "/sections/0"
        )
        reconstructed_section = "".join(
            text for _index, text in section_parts
        )
        self.assertIn(source, reconstructed_section)
        section_receipt = next(
            item
            for item in manifest["record_receipts"]
            if item["report_path"] == "/sections/0"
        )
        body_receipt = next(
            item
            for item in section_receipt["field_receipts"]
            if item["report_path"] == "/sections/0/body"
        )
        self.assertEqual(body_receipt["source_chars"], len(source))

    def test_part_receipt_cannot_cover_multi_clause_tail_with_one_token(
        self,
    ) -> None:
        source = (
            "A подтверждённая метрика равна 17%. "
            "TAIL_STATE остаётся unknown и не равен нулю."
        )
        payload = self._partition_payload(source)
        parts, _manifest = _semantic_partition_parts(payload, target_chars=512)
        part = next(
            item
            for item in parts
            if item["report_path"] == "/sections/0"
            and "TAIL_STATE" in item["context_text"]
        )
        exact_claims = _part_semantic_claims(part)
        self.assertTrue(
            any("TAIL_STATE" in str(item["claim"]) for item in exact_claims)
        )
        parsed = {
            "part_id": part["part_id"],
            "source_sha256": part["source_sha256"],
            "unit_sha256": part["unit_sha256"],
            "review": _pass_review(),
            "semantic_receipt": {
                "summary": "A",
                "claims": [
                    {
                        "report_path": "/sections/0/body",
                        "claim": "A",
                        "evidence_paths": ["/report_data"],
                        "interpretation": "A",
                    }
                ],
            },
        }
        with self.assertRaisesRegex(OpenRouterError, "atomic spans"):
            _validate_semantic_part_response(
                parsed,
                part=part,
                candidate_report=payload["candidate_report"],
                evidence_document=payload["evidence_document"],
            )

    def test_partition_rejects_missing_or_tampered_receipt(self) -> None:
        payload = self._partition_payload("Проверяемый текст. " * 80)
        parts, manifest = _semantic_partition_parts(payload, target_chars=256)
        receipts = [
            _semantic_part_receipt(part, _pass_review()) for part in parts
        ]

        with self.assertRaisesRegex(OpenRouterError, "coverage is incomplete"):
            _validate_semantic_partition_coverage(
                manifest,
                receipts[:-1],
                candidate_report=payload["candidate_report"],
            )

        tampered = [dict(item) for item in receipts]
        tampered[0]["unit_sha256"] = "0" * 64
        with self.assertRaisesRegex(OpenRouterError, "changed unit_sha256"):
            _validate_semantic_partition_coverage(
                manifest,
                tampered,
                candidate_report=payload["candidate_report"],
            )

        tampered_review = [dict(item) for item in receipts]
        tampered_review[0]["review_sha256"] = "f" * 64
        with self.assertRaisesRegex(OpenRouterError, "review digest mismatch"):
            _validate_semantic_partition_coverage(
                manifest,
                tampered_review,
                candidate_report=payload["candidate_report"],
                reviews=[_pass_review() for _part in parts],
            )

    def test_part_receipt_rejects_internal_digest_evidence_path(self) -> None:
        payload = self._partition_payload("Проверяемый текст. " * 20)
        parts, _manifest = _semantic_partition_parts(payload, target_chars=256)
        part = next(
            item
            for item in parts
            if item["report_path"] == "/sections/0"
            and item["unit_index"] == 0
        )
        claims = _part_semantic_claims(part)
        for claim in claims:
            claim["evidence_paths"] = ["/evidence_digest/fake"]
        parsed = {
            "part_id": part["part_id"],
            "source_sha256": part["source_sha256"],
            "unit_sha256": part["unit_sha256"],
            "review": _pass_review(),
            "semantic_receipt": {
                "summary": "Проверяется буквальное утверждение.",
                "claims": claims,
            },
        }

        with self.assertRaisesRegex(OpenRouterError, "evidence path is missing"):
            _validate_semantic_part_response(
                parsed,
                part=part,
                candidate_report=payload["candidate_report"],
                evidence_document=payload["evidence_document"],
            )

    def test_hierarchical_receipt_accepts_only_advertised_original_path(
        self,
    ) -> None:
        payload = self._partition_payload("Проверяемый текст. " * 20)
        payload["evidence_document"] = {
            "report_data": {"allowed": True, "hidden": False}
        }
        parts, _manifest = _semantic_partition_parts(payload, target_chars=256)
        part = next(
            item
            for item in parts
            if item["report_path"] == "/sections/0"
            and item["unit_index"] == 0
        )
        compact_evidence = {
            "long_input_contract": {
                "mode": "bounded_transitive_evidence_tree"
            },
            "evidence_digest": {
                "observations": [
                    {"source_paths": ["/report_data/allowed"]}
                ]
            },
        }
        path_contract = _semantic_evidence_path_contract(compact_evidence)
        self.assertEqual(
            path_contract["citation_rule"],
            "cite_only_original_source_paths",
        )
        claim_records = _part_semantic_claims(part)
        for claim_record in claim_records:
            claim_record["evidence_paths"] = ["/report_data/allowed"]
            claim_record["interpretation"] = (
                "Утверждение связано с исходным фактом."
            )
        parsed = {
            "part_id": part["part_id"],
            "source_sha256": part["source_sha256"],
            "unit_sha256": part["unit_sha256"],
            "review": _pass_review(),
            "semantic_receipt": {
                "summary": "Проверяется буквальное утверждение.",
                "claims": claim_records,
            },
        }

        validated = _validate_semantic_part_response(
            parsed,
            part=part,
            candidate_report=payload["candidate_report"],
            evidence_document=payload["evidence_document"],
            evidence_path_contract=path_contract,
        )
        self.assertEqual(
            validated["semantic_receipt"]["claims"][0]["evidence_paths"],
            ["/report_data/allowed"],
        )

        for claim_record in parsed["semantic_receipt"]["claims"]:
            claim_record["evidence_paths"] = ["/report_data/hidden"]
        with self.assertRaisesRegex(
            OpenRouterError, "outside the original hierarchical"
        ):
            _validate_semantic_part_response(
                parsed,
                part=part,
                candidate_report=payload["candidate_report"],
                evidence_document=payload["evidence_document"],
                evidence_path_contract=path_contract,
            )


    def test_oversized_single_finding_is_fragmented_and_reconstructed(self) -> None:
        node = {
            "node_id": "semantic-singleton",
            "level": 0,
            "source_part_ids": ["part-singleton"],
            "source_part_ids_sha256": _semantic_json_sha256(
                ["part-singleton"]
            ),
            "source_part_count": 1,
            "verdict": "pass",
            "summary": "Один очень длинный finding.",
            "material_findings": [
                {
                    "report_path": "/summary",
                    "claim": "A" * 30_000,
                    "interpretation": "B" * 30_000,
                    "evidence_paths": [],
                }
            ],
            "metric_availability_rows": [],
            "violations": [],
        }
        for input_window_bytes in (12_000, 100_000):
            with self.subTest(input_window_bytes=input_window_bytes):
                shards, coverage, audit = _semantic_prepare_finding_shards(
                    [node],
                    model_envelope={"max_completion_tokens": 32_000},
                    input_window_bytes=input_window_bytes,
                )
                decision_entries = [
                    entry
                    for shard in shards
                    for entry in coverage[shard["node_id"]]
                ]
                source_entries = [
                    entry
                    for shard in audit["source_shards"]
                    for entry in shard["entries"]
                ]
                self.assertEqual(audit["source_manifest"]["finding_count"], 1)
                self.assertGreater(audit["manifest"]["finding_count"], 1)
                self.assertGreater(len(shards), 1)
                self.assertTrue(
                    all(
                        _semantic_exact_output_utf8_bytes(
                            coverage[shard["node_id"]]
                        )
                        <= 16_384
                        for shard in shards
                    )
                )
                self.assertEqual(
                    _semantic_validate_fragment_reconstruction(
                        source_entries,
                        decision_entries,
                    ),
                    audit["reconstruction_manifest"],
                )
                tampered = copy.deepcopy(decision_entries)
                tampered[0]["finding"]["statement"] += "X"
                with self.assertRaisesRegex(OpenRouterError, "digest"):
                    _semantic_validate_fragment_reconstruction(
                        source_entries,
                        tampered,
                    )


class FinalReportSemanticGateIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_finding_ledger_shards_more_than_one_context_without_omission(
        self,
    ) -> None:
        findings: list[dict[str, object]] = []
        for index in range(90):
            if index % 2 == 0:
                claim = (
                    f"Метрика {index} равна {index}, знаменатель {index + 100}. "
                    + ("Числовой контекст. " * 22)
                )
                interpretation = (
                    "Числовой вывод сохраняет значение и знаменатель."
                )
            else:
                claim = (
                    f"Качественное наблюдение {index}: статус unknown не ноль. "
                    + ("Качественный контекст. " * 20)
                )
                interpretation = (
                    "Качественный вывод сохраняет data-state и ограничение."
                )
            findings.append(
                {
                    "report_path": "/summary",
                    "claim": claim,
                    "interpretation": interpretation,
                    "evidence_paths": [],
                }
            )
        node = {
            "node_id": "semantic-leaf-adversarial",
            "level": 0,
            "source_part_ids": ["part-adversarial"],
            "source_part_ids_sha256": hashlib.sha256(
                json.dumps(
                    ["part-adversarial"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_part_count": 1,
            "verdict": "pass",
            "summary": "Все числовые и качественные findings значимы.",
            "material_findings": findings,
            "metric_availability_rows": [
                {
                    "path": f"/report_data/metric_{index}",
                    "available": index % 3 != 0,
                    "signals": {
                        "data_state": (
                            "available" if index % 3 != 0 else "unavailable"
                        ),
                        "state": "known" if index % 3 != 0 else "unknown",
                        "value": index if index % 3 != 0 else None,
                        "context": "Точный metric-row контекст. " * 8,
                    },
                }
                for index in range(100)
            ],
            "violations": [
                {
                    "code": "other",
                    "severity": "important",
                    "report_path": "/summary",
                    "claim": "Проверяемый отчёт.",
                    "evidence_paths": [],
                    "finding": f"Исходное замечание {index}.",
                    "repair_instruction": f"Исправить замечание {index}.",
                }
                for index in range(100)
            ],
        }
        calls: list[dict[str, object]] = []
        raw_parts: list[dict[str, object]] = []

        async def fake_chat(**kwargs: object) -> SimpleNamespace:
            user_payload = json.loads(kwargs["messages"][1]["content"])
            source_nodes = user_payload["source_nodes"]
            parsed = {
                "source_node_ids": [
                    source_node["node_id"] for source_node in source_nodes
                ],
                "summary": _grounded_reducer_summary(user_payload),
                "material_findings": _reduced_material_findings(user_payload),
                "global_violations": [],
                "verdict": "pass",
            }
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        input_window_bytes = 23_744
        with patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fake_chat),
        ) as chat_mock:
            root, finding_audit = await _reduce_semantic_receipts(
                [node],
                model_envelope={
                    "context_length": 32_000,
                    "max_completion_tokens": 8_000,
                },
                input_window_bytes=input_window_bytes,
                candidate_report=_candidate("Проверяемый отчёт."),
                evidence_document={"report_data": {}},
                audit_checkpoint=None,
                calls=calls,
                raw_parts=raw_parts,
                manifest={
                    "candidate_sha256": "a" * 64,
                    "part_receipts_sha256": "b" * 64,
                    "part_receipts": [{"part_id": "part-adversarial"}],
                },
            )

        ledger_entries = [
            entry
            for shard in finding_audit["shards"]
            for entry in shard["entries"]
        ]
        self.assertGreater(finding_audit["shard_count"], 1)
        self.assertEqual(finding_audit["manifest"]["finding_count"], 290)
        self.assertEqual(
            finding_audit["manifest"]["item_kind_counts"],
            {
                "material_finding": 90,
                "metric_availability_row": 100,
                "semantic_violation": 100,
            },
        )
        self.assertEqual(len(ledger_entries), 290)
        self.assertEqual(
            [
                entry["finding"]
                for entry in ledger_entries
                if entry["item_kind"] == "material_finding"
            ],
            findings,
        )
        self.assertEqual(
            len({entry["finding_id"] for entry in ledger_entries}),
            290,
        )
        self.assertEqual(
            root["finding_ledger_manifest"], finding_audit["root_manifest"]
        )
        self.assertEqual(root["finding_ledger_manifest"]["finding_count"], 290)
        self.assertEqual(
            finding_audit["exact_disposition_manifest"]["disposition_count"],
            290,
        )
        self.assertEqual(len(finding_audit["exact_dispositions"]), 290)
        self.assertEqual(
            finding_audit["source_manifest"]["finding_count"], 290
        )
        self.assertEqual(
            root["finding_reconstruction_manifest"]["decision_atom_count"],
            290,
        )
        self.assertEqual(len(root["metric_availability_rows"]), 100)
        self.assertEqual(len(root["violations"]), 100)
        self.assertEqual(len(root["material_findings"]), 1)
        self.assertGreater(chat_mock.await_count, 1)
        self.assertTrue(
            all(
                int(call["request_utf8_bytes"]) <= input_window_bytes
                for call in calls
            )
        )
        self.assertTrue(
            all(
                int(call["minimum_response_utf8_bytes"]) <= 4_000
                for call in calls
            )
        )

    async def test_reducer_rejects_lineage_only_semantic_erasure(self) -> None:
        source_finding = {
            "report_path": "/summary",
            "claim": "TAIL_ALPHA равен 17.",
            "interpretation": "TAIL_BETA остаётся unknown, а не нулём.",
            "evidence_paths": [],
        }
        child = {
            "node_id": "child-semantic-tail",
            "level": 0,
            "source_part_ids": ["part-semantic-tail"],
            "source_part_ids_sha256": hashlib.sha256(
                json.dumps(
                    ["part-semantic-tail"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_part_count": 1,
            "verdict": "pass",
            "summary": "Два самостоятельных факта.",
            "material_findings": [source_finding],
            "metric_availability_rows": [],
            "violations": [],
        }
        source_finding_id = _semantic_reducer_user_payload(
            [child], level=1, group_index=0
        )["input_finding_manifest"][0]["source_finding_id"]

        with self.assertRaisesRegex(
            OpenRouterError, "dropped literal meaning"
        ):
            _validate_semantic_reducer_result(
                {
                    "source_node_ids": [child["node_id"]],
                    "summary": "Данные обработаны.",
                    "material_findings": [
                        {
                            "source_finding_ids": [source_finding_id],
                            "statement": "Данные обработаны.",
                            "evidence_paths": [],
                        }
                    ],
                    "global_violations": [],
                    "verdict": "pass",
                },
                children=[child],
                candidate_report=_candidate("TAIL_ALPHA равен 17."),
                evidence_document={"report_data": {}},
            )

    async def test_reducer_rejects_exact_findings_with_unrelated_summary(
        self,
    ) -> None:
        source_finding = {
            "report_path": "/limitations/0",
            "claim": "TAIL_STATE остаётся unknown.",
            "interpretation": "Unknown нельзя превращать в ноль.",
            "evidence_paths": ["/report_data/state"],
        }
        child = {
            "node_id": "child-exact-tail",
            "level": 0,
            "source_part_ids": ["part-exact-tail"],
            "source_part_ids_sha256": _semantic_json_sha256(
                ["part-exact-tail"]
            ),
            "source_part_count": 1,
            "verdict": "pass",
            "summary": "Состояние метрики проверено.",
            "material_findings": [source_finding],
            "metric_availability_rows": [],
            "violations": [],
        }
        source_id = _semantic_reducer_user_payload(
            [child], level=1, group_index=0
        )["input_finding_manifest"][0]["source_finding_id"]
        with self.assertRaisesRegex(OpenRouterError, "not grounded"):
            _validate_semantic_reducer_result(
                {
                    "source_node_ids": [child["node_id"]],
                    "summary": "ok",
                    "material_findings": [
                        {
                            "source_finding_ids": [source_id],
                            "statement": (
                                source_finding["claim"]
                                + "\n"
                                + source_finding["interpretation"]
                            ),
                            "evidence_paths": ["/report_data/state"],
                        }
                    ],
                    "global_violations": [],
                    "verdict": "pass",
                },
                children=[child],
                candidate_report=_candidate(source_finding["claim"]),
                evidence_document={"report_data": {"state": "unknown"}},
            )

    async def test_global_reducer_cannot_replace_receipt_with_one_token(
        self,
    ) -> None:
        source_finding = {
            "source_finding_ids": ["ledger-child"],
            "statement": (
                "TAIL_ALPHA равен 17%, а состояние TAIL_BETA остаётся unknown."
            ),
            "evidence_paths": [],
        }
        child = {
            "node_id": "sealed-child-tail",
            "level": 1,
            "source_part_ids": ["part-tail"],
            "source_part_ids_sha256": _semantic_json_sha256(["part-tail"]),
            "source_part_count": 1,
            "verdict": "pass",
            "summary": source_finding["statement"],
            "material_findings": [source_finding],
            "metric_availability_rows": [],
            "violations": [],
            "finding_decision_sealed": True,
        }
        reducer_payload = _semantic_reducer_user_payload(
            [child], level=2, group_index=0
        )
        source_id = reducer_payload["input_finding_manifest"][0][
            "source_finding_id"
        ]
        with self.assertRaisesRegex(
            OpenRouterError, "global reducer dropped bounded finding meaning"
        ):
            _validate_semantic_reducer_result(
                {
                    "source_node_ids": [child["node_id"]],
                    "summary": _grounded_reducer_summary(reducer_payload),
                    "material_findings": [
                        {
                            "source_finding_ids": [source_id],
                            "statement": "TAIL_ALPHA",
                            "evidence_paths": [],
                        }
                    ],
                    "global_violations": [],
                    "verdict": "pass",
                },
                children=[child],
                candidate_report=_candidate("TAIL_ALPHA равен 17%."),
                evidence_document={"report_data": {}},
            )

    async def test_compact_final_root_keeps_manifests_and_inherited_verdict(
        self,
    ) -> None:
        metric_row = {
            "path": "/report_data/metric_0",
            "available": False,
            "signals": {"state": "unknown", "value": None},
        }
        violation = {
            "code": "other",
            "severity": "important",
            "report_path": "/summary",
            "claim": "Проверяемый отчёт.",
            "evidence_paths": [],
            "finding": "Замечание 0.",
            "repair_instruction": "Исправить 0.",
        }
        finding_entries = [
            _semantic_finding_ledger_entry(
                source_node_id="source-1",
                finding_index=0,
                item_kind="metric_availability_row",
                finding={
                    "semantic_item_kind": "metric_availability_row",
                    "metric_row": metric_row,
                    "statement": _semantic_json_sha256(metric_row),
                    "evidence_paths": [metric_row["path"]],
                },
            ),
            _semantic_finding_ledger_entry(
                source_node_id="source-1",
                finding_index=0,
                item_kind="semantic_violation",
                finding={
                    "semantic_item_kind": "semantic_violation",
                    "violation": violation,
                    "statement": _semantic_json_sha256(violation),
                    "evidence_paths": [],
                },
            ),
        ]
        finding_manifest = _semantic_finding_ledger_manifest(finding_entries)
        reconstruction_manifest = _semantic_validate_fragment_reconstruction(
            finding_entries,
            finding_entries,
        )
        exact_dispositions = [
            {
                "finding_id": entry["finding_id"],
                "finding_sha256": entry["finding_sha256"],
                "source_finding_id": f"source-{index}",
                "statement": entry["finding"]["statement"],
                "statement_sha256": hashlib.sha256(
                    entry["finding"]["statement"].encode("utf-8")
                ).hexdigest(),
                "evidence_paths": list(
                    entry["finding"].get("evidence_paths") or []
                ),
            }
            for index, entry in enumerate(finding_entries)
        ]
        exact_disposition_manifest = _semantic_disposition_manifest(
            exact_dispositions
        )
        decision_receipt = {
            "source_node_ids": ["source-1"],
            "summary": "Проверяемый отчёт и unknown сохранены.",
            "verdict": "revise",
            "material_findings": [],
            "metric_availability_rows": [metric_row],
            "violations": [],
            "finding_ledger_manifest": finding_manifest,
            "exact_dispositions": exact_dispositions,
            "exact_disposition_manifest": exact_disposition_manifest,
        }
        decision_shard = {
            "stage": "source_decision",
            "node_id": "decision-1",
            "receipt_sha256": _semantic_json_sha256(decision_receipt),
            "receipt": decision_receipt,
        }
        decision_manifest = {
            "decision_shard_count": 1,
            "decision_receipts_sha256": _semantic_json_sha256(
                [
                    {
                        "stage": "source_decision",
                        "node_id": "decision-1",
                        "receipt_sha256": decision_shard["receipt_sha256"],
                    }
                ]
            ),
            "exact_disposition_manifest": exact_disposition_manifest,
            "coverage_complete": True,
        }
        finding_ledger_audit = {
            "manifest": finding_manifest,
            "source_manifest": finding_manifest,
            "source_shards": [
                {
                    "source_node_id": "source-1",
                    "manifest": finding_manifest,
                    "entries": finding_entries,
                }
            ],
            "reconstruction_manifest": reconstruction_manifest,
            "shards": [
                {
                    "manifest": finding_manifest,
                    "entries": finding_entries,
                }
            ],
            "decision_shards": [decision_shard],
            "decision_manifest": decision_manifest,
            "exact_dispositions": exact_dispositions,
            "exact_disposition_manifest": exact_disposition_manifest,
        }
        semantic_root = {
            "node_id": "semantic-root-compact",
            "level": 2,
            "source_part_ids": ["part-1"],
            "source_part_count": 1,
            "source_part_ids_sha256": "a" * 64,
            "finding_ledger_manifest": finding_manifest,
            "source_finding_manifest": finding_manifest,
            "finding_reconstruction_manifest": reconstruction_manifest,
            "decision_manifest": decision_manifest,
            "verdict": "revise",
            "summary": "Все decision shards проверены; verdict унаследован.",
            "material_findings": [
                {
                    "source_finding_ids": ["ledger-root"],
                    "statement": "Проверены числовые и качественные выводы.",
                    "evidence_paths": [],
                }
            ],
            "metric_availability_rows": [metric_row],
            "violations": [violation],
        }
        provider_payload = {
            "evidence_document": {"report_data": {}},
            "metric_availability_contract": [],
            "deterministic_precheck_errors": [],
        }
        compact = _semantic_final_user_payload(
            provider_payload,
            manifest={
                "candidate_sha256": "e" * 64,
                "candidate_utf8_bytes": 1,
                "record_count": 1,
                "part_count": 1,
                "section_count": 0,
                "action_count": 0,
                "limitation_count": 0,
                "record_receipts_sha256": "f" * 64,
                "part_receipts_sha256": "1" * 64,
            },
            receipts=[
                {
                    "verdict": "revise",
                    "violation_count": 1,
                    "blocking_violation_count": 1,
                }
            ],
            receipts_sha256="2" * 64,
            verdict_floor="revise",
            semantic_root=semantic_root,
            attempt=1,
            include_exact_ledgers=False,
        )

        self.assertEqual(compact["semantic_root"]["metric_availability_rows"], [])
        self.assertEqual(compact["semantic_root"]["violations"], [])
        self.assertEqual(
            compact["semantic_root"]["finding_ledger_manifest"],
            semantic_root["finding_ledger_manifest"],
        )
        self.assertEqual(
            compact["semantic_root"]["decision_manifest"],
            semantic_root["decision_manifest"],
        )
        inherited = _validate_semantic_final_response(
            {
                "review": {
                    "verdict": "revise",
                    "summary": "Строгий verdict сохранён из decision ledger.",
                    "violations": [],
                },
                "candidate_sha256": "e" * 64,
                "part_receipts_sha256": "2" * 64,
                "coverage_complete": True,
            },
            candidate_sha256="e" * 64,
            receipts_sha256="2" * 64,
            verdict_floor="revise",
            candidate_report=_candidate("Проверяемый отчёт."),
            evidence_document={"report_data": {}},
            semantic_root=semantic_root,
            finding_ledger_audit=finding_ledger_audit,
        )
        self.assertEqual(inherited["verdict"], "revise")
        tampered_root = dict(semantic_root)
        tampered_root["violations"] = []
        with self.assertRaisesRegex(OpenRouterError, "violation union"):
            _validate_semantic_final_response(
                {
                    "review": {
                        "verdict": "revise",
                        "summary": "Попытка скрыть локальное нарушение.",
                        "violations": [],
                    },
                    "candidate_sha256": "e" * 64,
                    "part_receipts_sha256": "2" * 64,
                    "coverage_complete": True,
                },
                candidate_sha256="e" * 64,
                receipts_sha256="2" * 64,
                verdict_floor="revise",
                candidate_report=_candidate("Проверяемый отчёт."),
                evidence_document={"report_data": {}},
                semantic_root=tampered_root,
                finding_ledger_audit=finding_ledger_audit,
            )

    async def test_reducer_cannot_acknowledge_child_and_drop_its_finding(
        self,
    ) -> None:
        child = {
            "node_id": "child-1",
            "level": 0,
            "source_part_ids": ["part-1"],
            "verdict": "pass",
            "material_findings": [
                {
                    "report_path": "/summary",
                    "claim": "Содержательный TAIL_MARKER.",
                    "interpretation": "Материальный качественный вывод.",
                    "evidence_paths": [],
                }
            ],
            "metric_availability_rows": [],
            "violations": [],
        }
        with self.assertRaisesRegex(
            OpenRouterError,
            "omitted, reordered, or duplicated",
        ):
            _validate_semantic_reducer_result(
                {
                    "source_node_ids": ["child-1"],
                    "summary": "Узел формально покрыт.",
                    "material_findings": [],
                    "global_violations": [],
                    "verdict": "pass",
                },
                children=[child],
                candidate_report=_candidate("Содержательный TAIL_MARKER."),
                evidence_document={"report_data": {}},
            )

    async def test_huge_report_uses_lossless_parts_and_one_receipt_root(
        self,
    ) -> None:
        summary = "Подтверждённый содержательный вывод. " * 1_000
        payload = {
            "evidence_document": {"report_data": {}},
            "model_evidence_context": {"report_data": {}},
            "metric_availability_contract": [],
            "candidate_report": _candidate(summary),
            "deterministic_precheck_errors": [],
        }
        observed_cores: dict[str, list[tuple[int, str]]] = {}
        audit_checkpoint = AsyncMock()

        async def fake_chat(**kwargs: object) -> SimpleNamespace:
            messages = kwargs["messages"]
            user_payload = json.loads(messages[1]["content"])
            if "candidate_part" in user_payload:
                part = user_payload["candidate_part"]
                core = part["context_text"][
                    part["core_start_in_context"] : part["core_end_in_context"]
                ]
                observed_cores.setdefault(part["report_path"], []).append(
                    (part["unit_index"], core)
                )
                parsed = {
                    "part_id": part["part_id"],
                    "source_sha256": part["source_sha256"],
                    "unit_sha256": part["unit_sha256"],
                    "review": _pass_review(),
                    "semantic_receipt": {
                        "summary": "Фрагмент не содержит противоречий.",
                        "claims": _part_semantic_claims(part),
                    },
                }
                return SimpleNamespace(
                    parsed=parsed,
                    text=json.dumps(parsed, ensure_ascii=False),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            if "source_nodes" in user_payload:
                nodes = user_payload["source_nodes"]
                parsed = {
                    "source_node_ids": [node["node_id"] for node in nodes],
                    "summary": _grounded_reducer_summary(user_payload),
                    "material_findings": _reduced_material_findings(
                        user_payload
                    ),
                    "global_violations": [],
                    "verdict": "pass",
                }
                return SimpleNamespace(
                    parsed=parsed,
                    text=json.dumps(parsed, ensure_ascii=False),
                    usage={"prompt_tokens": 1, "completion_tokens": 1},
                )
            receipt_manifest = user_payload["audit_receipts_manifest"]
            candidate_manifest = user_payload["candidate_manifest"]
            parsed = {
                "review": {
                    "verdict": user_payload["global_invariants"][
                        "verdict_floor"
                    ],
                    "summary": "Покрытие подтверждено.",
                    "violations": [],
                },
                "candidate_sha256": candidate_manifest["candidate_sha256"],
                "part_receipts_sha256": receipt_manifest[
                    "part_receipts_sha256"
                ],
                "coverage_complete": True,
            }
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        with patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 32_000,
                    "max_completion_tokens": 8_000,
                }
            ),
        ), patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fake_chat),
        ) as chat_mock:
            review, raw_text, usage = await review_final_report_semantics(
                payload,
                attempt=1,
                audit_checkpoint=audit_checkpoint,
            )

        self.assertEqual(review, _pass_review() | {"summary": "Покрытие подтверждено."})
        partition = usage["_aiv_semantic_partition"]
        self.assertGreater(partition["manifest"]["part_count"], 1)
        self.assertEqual(
            partition["manifest"]["physical_input_window_utf8_bytes"],
            23_744,
        )
        self.assertGreater(chat_mock.await_count, partition["manifest"]["part_count"])
        self.assertEqual(audit_checkpoint.await_count, chat_mock.await_count)
        self.assertEqual(
            audit_checkpoint.await_args_list[-1].args[0]["kind"],
            "semantic_receipt_root_accepted",
        )
        self.assertTrue(partition["coverage_complete"])
        self.assertTrue(json.loads(raw_text)["coverage_complete"])
        self.assertIn(
            summary,
            "".join(
                text
                for _index, text in sorted(observed_cores["/sections/0"])
            ),
        )
        self.assertTrue(
            all(
                call["request_utf8_bytes"] <= 23_744
                for call in partition["provider_calls"]
            )
        )

    async def test_physical_receipts_resume_parts_reducers_and_root_without_post(
        self,
    ) -> None:
        payload = {
            "evidence_document": {"report_data": {}},
            "model_evidence_context": {"report_data": {}},
            "metric_availability_contract": [],
            "candidate_report": _candidate("Проверяемый вывод. " * 1_000),
            "deterministic_precheck_errors": [],
        }
        lookup_kinds: list[str] = []
        accepted_events: list[str] = []

        async def checkpoint(event: dict[str, object]) -> None:
            accepted_events.append(str(event.get("kind") or ""))

        async def lookup(descriptor: dict[str, object]) -> SimpleNamespace:
            kind = str(descriptor["kind"])
            lookup_kinds.append(kind)
            request_payload = descriptor["request_payload"]
            user_payload = json.loads(
                request_payload["messages"][1]["content"]
            )
            if "candidate_part" in user_payload:
                part = user_payload["candidate_part"]
                parsed = {
                    "part_id": part["part_id"],
                    "source_sha256": part["source_sha256"],
                    "unit_sha256": part["unit_sha256"],
                    "review": _pass_review(),
                    "semantic_receipt": {
                        "summary": "Фрагмент проверен по сохранённому ответу.",
                        "claims": _part_semantic_claims(part),
                    },
                }
            elif "source_nodes" in user_payload:
                nodes = user_payload["source_nodes"]
                parsed = {
                    "source_node_ids": [node["node_id"] for node in nodes],
                    "summary": _grounded_reducer_summary(user_payload),
                    "material_findings": _reduced_material_findings(
                        user_payload
                    ),
                    "global_violations": [],
                    "verdict": "pass",
                }
            else:
                parsed = {
                    "review": _pass_review()
                    | {"summary": "Физический root-ответ восстановлен."},
                    "candidate_sha256": user_payload["candidate_manifest"][
                        "candidate_sha256"
                    ],
                    "part_receipts_sha256": user_payload[
                        "audit_receipts_manifest"
                    ]["part_receipts_sha256"],
                    "coverage_complete": True,
                }
            return SimpleNamespace(
                text=json.dumps(parsed, ensure_ascii=False),
                usage={"prompt_tokens": 17},
            )

        setattr(checkpoint, "lookup_completed", lookup)
        with patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 32_000,
                    "max_completion_tokens": 8_000,
                }
            ),
        ), patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(
                side_effect=AssertionError("provider POST must not run")
            ),
        ) as chat_mock:
            review, _raw, usage = await review_final_report_semantics(
                payload,
                attempt=1,
                audit_checkpoint=checkpoint,
            )

        self.assertEqual(review["verdict"], "pass")
        self.assertEqual(chat_mock.await_count, 0)
        self.assertTrue(any(kind.startswith("part-") for kind in lookup_kinds))
        self.assertTrue(
            any(kind.startswith("reduce-") for kind in lookup_kinds)
        )
        self.assertIn("receipt-root", lookup_kinds)
        self.assertIn("semantic_part_accepted", accepted_events)
        self.assertIn("semantic_reduce_accepted", accepted_events)
        self.assertIn("semantic_receipt_root_accepted", accepted_events)
        self.assertNotIn("prompt_tokens", usage)
        self.assertTrue(
            all(
                str(call["kind"]).endswith("_physical_resumed")
                for call in usage["_aiv_semantic_partition"][
                    "provider_calls"
                ]
            )
        )

    async def test_global_root_detects_cross_field_unknown_as_zero(self) -> None:
        candidate = _candidate("Фоновый проверяемый контекст. " * 1_000)
        candidate["sections"] = [
            {
                "heading": "Метрика равна 0%.",
                "body": "Состояние этой метрики — unknown.",
            }
        ]
        metric_row = {
            "path": "/report_data/metric",
            "available": False,
            "signals": {
                "data_state": "unavailable",
                "state": "unknown",
                "score": None,
                "valid_answers": 0,
            },
        }
        evidence = {
            "report_data": {
                "metric": {
                    "data_state": "unavailable",
                    "state": "unknown",
                    "score": None,
                    "valid_answers": 0,
                }
            },
            "metric_availability_contract": [metric_row],
        }
        payload = {
            "evidence_document": evidence,
            "model_evidence_context": evidence,
            "metric_availability_contract": [metric_row],
            "candidate_report": candidate,
            "deterministic_precheck_errors": [],
        }
        global_violation = {
            "code": "missing_data_as_zero",
            "severity": "critical",
            "report_path": "/sections/0/heading",
            "claim": "Метрика равна 0%.",
            "evidence_paths": ["/report_data/metric/data_state"],
            "finding": "Unknown ошибочно представлен как измеренный ноль.",
            "repair_instruction": "Заменить ноль явным статусом недоступности.",
        }
        exact_metric_row_seen = False
        exact_metric_row_reached_final = False

        async def fake_chat(**kwargs: object) -> SimpleNamespace:
            nonlocal exact_metric_row_reached_final, exact_metric_row_seen
            user_payload = json.loads(kwargs["messages"][1]["content"])
            if "candidate_part" in user_payload:
                part = user_payload["candidate_part"]
                claims = _part_semantic_claims(part)
                for claim_record in claims:
                    if str(claim_record["report_path"]).startswith(
                        "/sections/0/"
                    ):
                        claim_record["evidence_paths"] = [
                            "/report_data/metric/data_state"
                        ]
                        claim_record["interpretation"] = (
                            "Сопоставление статуса и опубликованного значения."
                        )
                parsed = {
                    "part_id": part["part_id"],
                    "source_sha256": part["source_sha256"],
                    "unit_sha256": part["unit_sha256"],
                    "review": _pass_review(),
                    "semantic_receipt": {
                        "summary": "Проверен смысл связанного контейнера.",
                        "claims": claims,
                    },
                }
            elif "source_nodes" in user_payload:
                nodes = user_payload["source_nodes"]
                serialized = json.dumps(nodes, ensure_ascii=False)
                exact_metric_row_seen = exact_metric_row_seen or (
                    '"data_state": "unavailable"' in serialized
                    and '"available": false' in serialized
                )
                has_conflict = (
                    "Метрика равна 0%." in serialized
                    and "unknown" in serialized
                    and '"data_state": "unavailable"' in serialized
                ) or (
                    "missing_data_as_zero" in serialized
                    and "/report_data/metric/data_state" in serialized
                )
                violations = [global_violation] if has_conflict else []
                parsed = {
                    "source_node_ids": [node["node_id"] for node in nodes],
                    "summary": _grounded_reducer_summary(user_payload),
                    "material_findings": _reduced_material_findings(
                        user_payload
                    ),
                    "global_violations": violations,
                    "verdict": (
                        "revise"
                        if violations
                        or any(node["verdict"] != "pass" for node in nodes)
                        else "pass"
                    ),
                }
            else:
                semantic_root = user_payload["semantic_root"]
                exact_metric_row_reached_final = metric_row in semantic_root[
                    "metric_availability_rows"
                ]
                receipt_manifest = user_payload["audit_receipts_manifest"]
                candidate_manifest = user_payload["candidate_manifest"]
                parsed = {
                    "review": {
                        "verdict": "revise",
                        "summary": "Финальный арбитр сохранил строгий verdict.",
                        # The provider deliberately omits the exact local
                        # violation. Code must re-open the sealed ledger and
                        # put it back into the publication review.
                        "violations": [],
                    },
                    "candidate_sha256": candidate_manifest[
                        "candidate_sha256"
                    ],
                    "part_receipts_sha256": receipt_manifest[
                        "part_receipts_sha256"
                    ],
                    "coverage_complete": True,
                }
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={},
            )

        with patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 32_000,
                    "max_completion_tokens": 8_000,
                }
            ),
        ), patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fake_chat),
        ):
            review, _raw_text, _usage = await review_final_report_semantics(
                payload,
                attempt=1,
            )

        self.assertEqual(review["verdict"], "revise")
        self.assertTrue(
            any(
                item["code"] == "missing_data_as_zero"
                for item in review["violations"]
            )
        )
        self.assertTrue(exact_metric_row_seen)
        self.assertTrue(exact_metric_row_reached_final)

    async def test_mid_run_failure_resumes_without_rebilling_accepted_part(
        self,
    ) -> None:
        payload = {
            "evidence_document": {"report_data": {}},
            "model_evidence_context": {"report_data": {}},
            "metric_availability_contract": [],
            "candidate_report": _candidate("Длинный отчёт. " * 1_000),
            "deterministic_precheck_errors": [],
        }
        checkpoint = AsyncMock()
        physical_call = 0

        def part_result(part: dict[str, object]) -> SimpleNamespace:
            parsed = {
                "part_id": part["part_id"],
                "source_sha256": part["source_sha256"],
                "unit_sha256": part["unit_sha256"],
                "review": _pass_review(),
                "semantic_receipt": {
                    "summary": "Фрагмент проверен.",
                    "claims": _part_semantic_claims(part),
                },
            }
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={"prompt_tokens": 1},
            )

        async def fail_second_part(**kwargs: object) -> SimpleNamespace:
            nonlocal physical_call
            physical_call += 1
            user_payload = json.loads(kwargs["messages"][1]["content"])
            if physical_call == 2:
                error = OpenRouterError("provider interrupted")
                error.result = SimpleNamespace(
                    text="failed-raw",
                    usage={"prompt_tokens": 2},
                )
                raise error
            return part_result(user_payload["candidate_part"])

        envelope = {
            "context_length": 32_000,
            "max_completion_tokens": 8_000,
        }
        with patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(return_value=envelope),
        ), patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fail_second_part),
        ):
            with self.assertRaisesRegex(
                OpenRouterError, "provider interrupted"
            ) as failure:
                await review_final_report_semantics(
                    payload,
                    attempt=1,
                    audit_checkpoint=checkpoint,
                )

        prefix = failure.exception.result.usage[
            "_aiv_semantic_partition_failure_prefix"
        ]
        self.assertEqual(prefix["accepted_call_count"], 1)
        accepted = [
            call.args[0]
            for call in checkpoint.await_args_list
            if call.args[0]["kind"] == "semantic_part_accepted"
        ]
        self.assertEqual(len(accepted), 1)
        resume_checkpoint = {
            "version": accepted[0]["version"],
            "candidate_sha256": accepted[0]["candidate_sha256"],
            "source_part_receipts_sha256": accepted[0][
                "source_part_receipts_sha256"
            ],
            "accepted_parts": accepted,
        }

        async def finish_chat(**kwargs: object) -> SimpleNamespace:
            user_payload = json.loads(kwargs["messages"][1]["content"])
            if "candidate_part" in user_payload:
                return part_result(user_payload["candidate_part"])
            if "source_nodes" in user_payload:
                nodes = user_payload["source_nodes"]
                parsed = {
                    "source_node_ids": [node["node_id"] for node in nodes],
                    "summary": _grounded_reducer_summary(user_payload),
                    "material_findings": _reduced_material_findings(
                        user_payload
                    ),
                    "global_violations": [],
                    "verdict": "pass",
                }
            else:
                parsed = {
                    "review": _pass_review(),
                    "candidate_sha256": user_payload["candidate_manifest"][
                        "candidate_sha256"
                    ],
                    "part_receipts_sha256": user_payload[
                        "audit_receipts_manifest"
                    ]["part_receipts_sha256"],
                    "coverage_complete": True,
                }
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={"prompt_tokens": 1},
            )

        with patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(return_value=envelope),
        ), patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=finish_chat),
        ) as resumed_chat:
            review, _raw, usage = await review_final_report_semantics(
                payload,
                attempt=1,
                resume_checkpoint=resume_checkpoint,
            )

        self.assertEqual(review["verdict"], "pass")
        provider_calls = usage["_aiv_semantic_partition"]["provider_calls"]
        self.assertEqual(provider_calls[0]["kind"], "part_resumed")
        self.assertEqual(resumed_chat.await_count, len(provider_calls) - 1)
        self.assertEqual(usage["prompt_tokens"], len(provider_calls) - 1)

    async def test_semantic_reviewer_has_bounded_nontruncating_contract(
        self,
    ) -> None:
        response = SimpleNamespace(
            parsed={
                "verdict": "pass",
                "summary": "Смысловых противоречий не найдено.",
                "violations": [],
            },
            text="{}",
            usage={},
        )
        checkpoint = AsyncMock()
        with patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(return_value=response),
        ) as chat_mock, patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 1_000_000,
                    "max_completion_tokens": 100_000,
                }
            ),
        ):
            review, _text, _usage = await review_final_report_semantics(
                {
                    "evidence_document": {"report_data": {}},
                    "model_evidence_context": {"report_data": {}},
                    "metric_availability_contract": [],
                    "candidate_report": {},
                    "deterministic_precheck_errors": [],
                },
                attempt=1,
                audit_checkpoint=checkpoint,
                resume_checkpoint={"document_id": "must-not-be-forwarded"},
            )

        self.assertEqual(review["verdict"], "pass")
        kwargs = chat_mock.await_args.kwargs
        self.assertEqual(
            kwargs["reasoning_effort"],
            REPORT_SEMANTIC_REASONING_EFFORT,
        )
        self.assertEqual(
            kwargs["output_token_policy"],
            OutputTokenPolicy.MODEL_MAX,
        )
        self.assertIs(kwargs["retry_response_contract_errors"], False)
        self.assertIs(kwargs["retry_transport_errors"], False)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertNotIn("document_id", kwargs)
        self.assertIs(kwargs["audit_checkpoint"], checkpoint)
        self.assertEqual(kwargs["audit_context"]["sequence"], 0)
        self.assertNotIn("resume_checkpoint", kwargs)
        self.assertNotIn("max_continuations", kwargs)

    async def test_atomic_semantic_reviewer_uses_bounded_context_and_full_corpus_audit(
        self,
    ) -> None:
        async def fake_chat(**kwargs: object) -> SimpleNamespace:
            user_payload = json.loads(kwargs["messages"][1]["content"])
            if str(kwargs["schema_name"]).startswith(
                "aiv_semantic_evidence_"
            ):
                claim_batch = user_payload["claim_batch"]
                parsed = {
                    "task_id": user_payload["task_id"],
                    "evidence_shard_id": user_payload["evidence_shard"][
                        "evidence_shard_id"
                    ],
                    "claim_batch_sha256": claim_batch[
                        "claim_batch_sha256"
                    ],
                    "coverage_complete": True,
                    "dispositions": [
                        {
                            "claim_id": claim["claim_id"],
                            "status": "clear",
                            "evidence_paths": [],
                            "evidence_quote": "",
                            "explanation": (
                                "В этом фрагменте нет противоречия."
                            ),
                        }
                        for claim in claim_batch["claims"]
                    ],
                }
            else:
                parsed = _pass_review()
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={},
            )

        payload = {
            "evidence_document": {
                "report_data": {"full_only": "FULL-EVIDENCE-SENTINEL"},
            },
            "model_evidence_context": {
                "evidence_digest": {"bounded": "BOUNDED-CONTEXT-SENTINEL"},
            },
            "metric_availability_contract": [],
            "candidate_report": _candidate("Проверяемый вывод."),
            "deterministic_precheck_errors": [],
        }
        with patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fake_chat),
        ) as chat_mock, patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 1_000_000,
                    "max_completion_tokens": 100_000,
                }
            ),
        ):
            review, _text, usage = await review_final_report_semantics(
                payload,
                attempt=1,
            )

        self.assertEqual(review["verdict"], "pass")
        self.assertGreater(chat_mock.await_count, 1)
        provider_user_payload = json.loads(
            chat_mock.await_args_list[0].kwargs["messages"][1]["content"]
        )
        serialized = json.dumps(provider_user_payload, ensure_ascii=False)
        self.assertNotIn("model_evidence_context", provider_user_payload)
        self.assertNotIn("FULL-EVIDENCE-SENTINEL", serialized)
        self.assertIn("BOUNDED-CONTEXT-SENTINEL", serialized)
        self.assertEqual(
            provider_user_payload["evidence_document"],
            payload["model_evidence_context"],
        )
        full_corpus_calls = [
            json.loads(call.kwargs["messages"][1]["content"])
            for call in chat_mock.await_args_list[1:]
            if str(call.kwargs["schema_name"]).startswith(
                "aiv_semantic_evidence_"
            )
        ]
        self.assertTrue(full_corpus_calls)
        self.assertIn(
            "FULL-EVIDENCE-SENTINEL",
            json.dumps(full_corpus_calls, ensure_ascii=False),
        )
        coverage = usage["_aiv_semantic_evidence_coverage"]["manifest"]
        self.assertTrue(coverage["coverage_complete"])
        self.assertEqual(
            coverage["reviewed_pair_count"],
            coverage["expected_pair_count"],
        )

    async def test_material_tail_contradiction_is_seen_by_independent_reviewer(
        self,
    ) -> None:
        tail_fact = "Фактический показатель равен 12%."
        evidence = {
            "report_data": {
                "corpus": ("Промежуточные исходные данные. " * 2_000)
                + tail_fact,
            }
        }
        payload = {
            "evidence_document": evidence,
            "model_evidence_context": {
                "evidence_digest": {
                    "summary": "Сжатый корневой контекст без хвоста."
                }
            },
            "metric_availability_contract": [],
            "candidate_report": _candidate("Показатель равен 100%."),
            "deterministic_precheck_errors": [],
        }
        saw_tail = False

        async def fake_chat(**kwargs: object) -> SimpleNamespace:
            nonlocal saw_tail
            user_payload = json.loads(kwargs["messages"][1]["content"])
            if not str(kwargs["schema_name"]).startswith(
                "aiv_semantic_evidence_"
            ):
                parsed = _pass_review()
            else:
                shard_text = json.dumps(
                    user_payload["evidence_shard"],
                    ensure_ascii=False,
                )
                tail_visible = tail_fact in shard_text
                saw_tail = saw_tail or tail_visible
                dispositions = []
                for claim in user_payload["claim_batch"]["claims"]:
                    is_conflict = (
                        tail_visible
                        and claim["report_path"] == "/executive_summary"
                    )
                    dispositions.append(
                        {
                            "claim_id": claim["claim_id"],
                            "status": (
                                "contradiction" if is_conflict else "clear"
                            ),
                            "evidence_paths": (
                                ["/report_data/corpus"]
                                if is_conflict
                                else []
                            ),
                            "evidence_quote": tail_fact if is_conflict else "",
                            "explanation": (
                                "Полный хвост корпуса противоречит числу в отчёте."
                                if is_conflict
                                else "В этом фрагменте нет противоречия."
                            ),
                        }
                    )
                parsed = {
                    "task_id": user_payload["task_id"],
                    "evidence_shard_id": user_payload["evidence_shard"][
                        "evidence_shard_id"
                    ],
                    "claim_batch_sha256": user_payload["claim_batch"][
                        "claim_batch_sha256"
                    ],
                    "coverage_complete": True,
                    "dispositions": dispositions,
                }
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={},
            )

        with patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fake_chat),
        ), patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 60_000,
                    "max_completion_tokens": 8_000,
                }
            ),
        ):
            review, _text, usage = await review_final_report_semantics(
                payload,
                attempt=1,
            )

        self.assertTrue(saw_tail)
        self.assertEqual(review["verdict"], "revise")
        self.assertTrue(
            any(
                violation["report_path"] == "/executive_summary"
                and tail_fact in violation["finding"]
                for violation in review["violations"]
            )
        )
        coverage = usage["_aiv_semantic_evidence_coverage"]["manifest"]
        self.assertGreater(coverage["evidence_manifest"]["unit_count"], 1)
        self.assertGreater(coverage["evidence_shard_count"], 1)
        self.assertEqual(
            coverage["reviewed_pair_count"],
            coverage["expected_pair_count"],
        )

    async def test_incomplete_full_corpus_dispositions_block_after_bounded_loop(
        self,
    ) -> None:
        payload = {
            "evidence_document": {
                "report_data": {"fact": "Полный исходный факт."},
            },
            "model_evidence_context": {
                "evidence_digest": {"summary": "Сжатый контекст."},
            },
            "metric_availability_contract": [],
            "candidate_report": _candidate("Проверяемый вывод."),
            "deterministic_precheck_errors": [],
        }

        async def fake_chat(**kwargs: object) -> SimpleNamespace:
            user_payload = json.loads(kwargs["messages"][1]["content"])
            if str(kwargs["schema_name"]).startswith(
                "aiv_semantic_evidence_"
            ):
                parsed = {
                    "task_id": user_payload["task_id"],
                    "evidence_shard_id": user_payload["evidence_shard"][
                        "evidence_shard_id"
                    ],
                    "claim_batch_sha256": user_payload["claim_batch"][
                        "claim_batch_sha256"
                    ],
                    "coverage_complete": True,
                    "dispositions": [],
                }
            else:
                parsed = _pass_review()
            return SimpleNamespace(
                parsed=parsed,
                text=json.dumps(parsed, ensure_ascii=False),
                usage={},
            )

        with patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(side_effect=fake_chat),
        ) as chat_mock, patch(
            "app.services.report_semantic_gate.model_output_envelope",
            new=AsyncMock(
                return_value={
                    "context_length": 1_000_000,
                    "max_completion_tokens": 100_000,
                }
            ),
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "exhausted its bounded protocol-repair loop",
            ):
                await review_final_report_semantics(payload, attempt=1)

        coverage_calls = [
            call
            for call in chat_mock.await_args_list
            if str(call.kwargs["schema_name"]).startswith(
                "aiv_semantic_evidence_"
            )
        ]
        self.assertEqual(len(coverage_calls), 2)
        self.assertTrue(coverage_calls[0].kwargs["schema_name"].endswith("_r1"))
        self.assertTrue(coverage_calls[1].kwargs["schema_name"].endswith("_r2"))

    async def test_semantic_reviewer_rejects_unpreflighted_evidence(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            OpenRouterError,
            "no preflighted evidence context",
        ):
            await review_final_report_semantics(
                {
                    "evidence_document": {"report_data": {}},
                    "metric_availability_contract": [],
                    "candidate_report": {},
                    "deterministic_precheck_errors": [],
                },
                attempt=1,
            )

    async def test_one_repair_is_reviewed_before_publication(self) -> None:
        rejected = _candidate("Память моделей не знает бренд.")
        repaired = _candidate("В отчёте используются подтверждённые данные.")
        repaired["limitations"] = [
            "Срез без веб-поиска не измерен: вывод о памяти моделей не "
            "формируется."
        ]
        chat_results = [
            SimpleNamespace(parsed=rejected, text="rejected", usage={}),
            SimpleNamespace(parsed=repaired, text="repaired", usage={}),
        ]
        payload = {
            "report_data": _public_report_with_unavailable_memory(),
            "answer_selection_manifest": {"digest": "selection"},
            "answer_corpus_manifest": {"digest": "corpus"},
            "selected_full_answers": [],
        }
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
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
                "app.services.analyzer.chat_continuable_structured",
                new_callable=AsyncMock,
                side_effect=chat_results,
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[_revise_review(), _pass_review()],
            ) as semantic_review,
            patch(
                "app.services.analyzer._edit_final_report_language",
                new=AsyncMock(return_value=repaired),
            ),
        ):
            result = await _final_report(
                "run-id",
                _public_report_with_unavailable_memory(),
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, repaired)
        self.assertEqual(final_chat.await_count, 2)
        self.assertEqual(semantic_review.await_count, 2)
        system_prompt = final_chat.await_args_list[0].kwargs["messages"][0][
            "content"
        ]
        self.assertIn(
            "answer_count и answer_rate означают ответы по",
            system_prompt,
        )
        self.assertIn(
            "существу, а не конкретику",
            system_prompt,
        )
        self.assertIn(
            "только specific_count и specific_rate",
            system_prompt,
        )
        repair_payload = json.loads(
            final_chat.await_args_list[1].kwargs["messages"][1]["content"]
        )
        self.assertEqual(repair_payload["rejected_report"], rejected)
        self.assertEqual(
            repair_payload["semantic_review_to_fix"]["verdict"],
            "revise",
        )
        final_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "final_report"
        ]
        self.assertEqual(final_writes[-1]["status"], "completed")
        self.assertEqual(final_writes[-1]["output_json"], repaired)

    async def test_third_semantic_rejection_fails_closed_without_looping(self) -> None:
        rejected = _candidate("Без веб-поиска память моделей равна 0%.")
        payload = {
            "report_data": _public_report_with_unavailable_memory(),
            "answer_selection_manifest": {"digest": "selection"},
            "answer_corpus_manifest": {"digest": "corpus"},
            "selected_full_answers": [],
        }
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
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
                "app.services.analyzer.chat_continuable_structured",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(parsed=rejected, text="one", usage={}),
                    SimpleNamespace(parsed=rejected, text="two", usage={}),
                    SimpleNamespace(parsed=rejected, text="three", usage={}),
                ],
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    _revise_review(),
                    _revise_review(),
                    _revise_review(),
                ],
            ) as semantic_review,
        ):
            with self.assertRaises(OpenRouterError):
                await _final_report(
                    "run-id",
                    _public_report_with_unavailable_memory(),
                    {"manifest": {"digest": "corpus"}, "answers": [{}]},
                )

        self.assertEqual(final_chat.await_count, 3)
        self.assertEqual(semantic_review.await_count, 3)
        final_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "final_report"
        ]
        self.assertEqual(final_writes[-1]["status"], "failed")
        self.assertFalse(
            any(
                item.get("status") == "completed"
                and item.get("output_json") == rejected
                for item in final_writes
            )
        )
