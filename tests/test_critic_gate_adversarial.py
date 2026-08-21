import unittest

from app.services.analysis_critic import _normalize_review
from app.services.analyzer import (
    _critic_review_validation_errors,
    _deterministic_critic_fallback_review,
    _scope_leakage_warning_machine_resolved,
)


def _pass_review() -> dict:
    return {
        "verdict": "pass",
        "summary": "Все опубликованные метрики сверены с исходными ответами.",
        "anomalies": [],
        "policy_adjustments": [],
        "annotation_guidance": "",
        "acceptance_checks": [
            "Проверены числители, знаменатели и буквальные доказательства."
        ],
    }


def _payload(*, deterministic_warnings: list[dict] | None = None) -> dict:
    return {
        "answers": [],
        "entity_catalog": {"target_aliases": [], "entities": []},
        "deterministic_warnings": deterministic_warnings or [],
    }


def _validation_errors(review: dict, *, payload: dict | None = None) -> list[str]:
    # Live model responses pass through this normalization before the gate.
    # Regressions must therefore exercise the same path instead of validating
    # only the unnormalized fixture.
    return _critic_review_validation_errors(
        _normalize_review(review),
        payload=payload or _payload(),
    )


class CriticGateAdversarialTests(unittest.TestCase):
    @staticmethod
    def _limited_scope_payload() -> dict:
        zero_slice = {
            "data_state": "limited",
            "expected_answers": 2,
            "completed_answers": 1,
            "annotated_answers": 1,
            "valid_answers": 1,
            "coverage_rate": 50.0,
            "mention_count": 0,
            "mention_rate": 0.0,
        }
        return {
            "site_profile": {
                "brand_name": "Example",
                "brand_aliases": [],
                "entity_scope": [],
            },
            "entity_catalog": {
                "target_aliases": ["Example"],
                "entities": [],
            },
            "candidate_metrics": {
                "providers": [
                    {
                        "name": "ChatGPT",
                        "parent_discovery": dict(zero_slice),
                        "portfolio_capture": dict(zero_slice),
                    }
                ]
            },
            "deterministic_warnings": [
                {
                    "code": "scope_leakage",
                    "severity": "important",
                    "finding": "Совпали независимо рассчитанные числители.",
                    "providers": ["ChatGPT"],
                }
            ],
            "answers": [
                {
                    "answer_id": 1,
                    "mode": "web",
                    "scenario_role": "unbranded_discovery",
                    "provider": "openai",
                    "status": "completed",
                    "metric_eligible": True,
                    "annotation_state": "current",
                    "raw_answer_truncated": False,
                    "raw_answer": "В ответе перечислены другие компании.",
                    "annotation": {
                        "valid": True,
                        "target_mentioned": False,
                        "entity_mentions": [],
                    },
                },
                {
                    "answer_id": 2,
                    "mode": "web",
                    "scenario_role": "unbranded_discovery",
                    "provider": "openai",
                    "status": "completed",
                    "metric_eligible": False,
                    "annotation_state": "current",
                    "raw_answer_truncated": False,
                    "raw_answer": "Веб-режим этого ответа не подтверждён.",
                    "annotation": {
                        "valid": True,
                        "target_mentioned": False,
                        "entity_mentions": [],
                    },
                },
            ],
        }

    def test_well_formed_pass_is_accepted(self) -> None:
        self.assertEqual(_validation_errors(_pass_review()), [])

    def test_missing_required_pass_fields_are_rejected_after_normalization(
        self,
    ) -> None:
        for field in (
            "verdict",
            "summary",
            "anomalies",
            "policy_adjustments",
            "annotation_guidance",
            "acceptance_checks",
        ):
            with self.subTest(field=field):
                review = _pass_review()
                review.pop(field)
                self.assertTrue(
                    _validation_errors(review),
                    f"missing required field {field!r} opened the critic gate",
                )

    def test_null_required_pass_fields_are_rejected_after_normalization(
        self,
    ) -> None:
        for field in (
            "verdict",
            "summary",
            "anomalies",
            "policy_adjustments",
            "annotation_guidance",
            "acceptance_checks",
        ):
            with self.subTest(field=field):
                review = _pass_review()
                review[field] = None
                self.assertTrue(
                    _validation_errors(review),
                    f"null required field {field!r} opened the critic gate",
                )

    def test_malformed_anomaly_items_are_rejected(self) -> None:
        malformed_items = (
            "not-an-object",
            {
                "code": "provider_uniformity",
                "severity": "observation",
                # Missing finding, answer_ids, and entities.
            },
            {
                "code": "not_a_supported_code",
                "severity": "observation",
                "finding": "Unknown anomaly code must not be accepted.",
                "answer_ids": [],
                "entities": [],
            },
            {
                "code": "provider_uniformity",
                "severity": "minor",
                "finding": "Unknown severity must not be accepted.",
                "answer_ids": [],
                "entities": [],
            },
            {
                "code": "provider_uniformity",
                "severity": "observation",
                "finding": "answer_ids has the wrong type.",
                "answer_ids": "11",
                "entities": [],
            },
        )
        for anomaly in malformed_items:
            with self.subTest(anomaly=anomaly):
                review = _pass_review()
                review["anomalies"] = [anomaly]
                self.assertTrue(
                    _validation_errors(review),
                    "malformed anomaly item opened the critic gate",
                )

    def test_pass_requires_at_least_one_nonblank_acceptance_check(self) -> None:
        for acceptance_checks in ([], [""], ["   "], ["", "\n\t"]):
            with self.subTest(acceptance_checks=acceptance_checks):
                review = _pass_review()
                review["acceptance_checks"] = acceptance_checks
                self.assertTrue(
                    _validation_errors(review),
                    "pass without a concrete acceptance check opened the gate",
                )

    def test_important_deterministic_warning_cannot_be_silently_ignored(
        self,
    ) -> None:
        payload = _payload(
            deterministic_warnings=[
                {
                    "code": "scope_leakage",
                    "severity": "important",
                    "finding": (
                        "У нескольких систем совпали числители материнского "
                        "бренда и портфеля."
                    ),
                    "providers": ["ChatGPT", "Gemini", "Claude"],
                }
            ]
        )
        review = _pass_review()
        review["acceptance_checks"] = [
            "Проверено, что в ответах нет пустых строк."
        ]

        self.assertTrue(
            _validation_errors(review, payload=payload),
            "pass silently ignored an important deterministic warning",
        )

    def test_scope_resolver_is_missing_aware_without_turning_gap_into_zero(
        self,
    ) -> None:
        payload = self._limited_scope_payload()
        warning = payload["deterministic_warnings"][0]

        self.assertTrue(
            _scope_leakage_warning_machine_resolved(payload, warning)
        )

    def test_malformed_critic_can_fallback_only_after_machine_resolution(
        self,
    ) -> None:
        payload = self._limited_scope_payload()
        incomplete = _pass_review()
        incomplete.pop("acceptance_checks")

        fallback = _deterministic_critic_fallback_review(
            payload,
            incomplete,
            validation_errors=["acceptance_checks missing"],
        )

        self.assertEqual(fallback["verdict"], "pass")
        self.assertEqual(
            fallback["fallback"]["kind"],
            "deterministic_safe_pass",
        )
        self.assertEqual(
            _critic_review_validation_errors(fallback, payload=payload),
            [],
        )

    def test_malformed_critic_fallback_blocks_unresolved_warning(self) -> None:
        payload = self._limited_scope_payload()
        payload["deterministic_warnings"][0]["code"] = (
            "generic_term_leakage"
        )
        incomplete = _pass_review()
        incomplete.pop("acceptance_checks")

        fallback = _deterministic_critic_fallback_review(
            payload,
            incomplete,
            validation_errors=["acceptance_checks missing"],
        )

        self.assertEqual(fallback["verdict"], "block")
        self.assertEqual(
            fallback["fallback"]["kind"],
            "deterministic_block",
        )


if __name__ == "__main__":
    unittest.main()
