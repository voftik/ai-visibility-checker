import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.analyzer import (
    _eligible_illustration_answer_context,
    _final_report,
    _select_final_answer_context,
)
from app.services.openrouter import OpenRouterError
from app.services.report_semantic_gate import (
    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
    CANONICAL_UNAVAILABLE_PORTFOLIO_LIMITATION,
    REPORT_SEMANTIC_MAX_TOKENS,
    REPORT_SEMANTIC_REASONING_EFFORT,
    deterministic_report_semantic_errors,
    metric_availability_contract,
    normalize_report_semantic_review,
    report_semantic_blockers,
    review_final_report_semantics,
    validate_report_semantic_review,
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


class FinalReportSemanticGateIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
        with patch(
            "app.services.report_semantic_gate.chat",
            new=AsyncMock(return_value=response),
        ) as chat_mock:
            review, _text, _usage = await review_final_report_semantics(
                {
                    "evidence_document": {"report_data": {}},
                    "metric_availability_contract": [],
                    "candidate_report": {},
                    "deterministic_precheck_errors": [],
                },
                attempt=1,
            )

        self.assertEqual(review["verdict"], "pass")
        kwargs = chat_mock.await_args.kwargs
        self.assertEqual(
            kwargs["reasoning_effort"],
            REPORT_SEMANTIC_REASONING_EFFORT,
        )
        self.assertEqual(kwargs["max_tokens"], REPORT_SEMANTIC_MAX_TOKENS)

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
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=chat_results,
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[_revise_review(), _pass_review()],
            ) as semantic_review,
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
                "app.services.analyzer.chat",
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
