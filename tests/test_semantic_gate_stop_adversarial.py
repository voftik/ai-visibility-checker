import unittest
from typing import Any

from app.services.analysis_critic import _normalize_review
from app.services.analyzer import (
    _critic_review_validation_errors,
    _reconcile_annotation,
    _scope_leakage_warning_machine_resolved,
    _scope_entity_catalog_to_profile,
)


def _blank_annotation() -> dict[str, Any]:
    return {
        "answer_id": 1,
        "valid": True,
        "target_mentioned": False,
        "target_position": None,
        "target_role": "absent",
        "sentiment": "unknown",
        "entity_mentions": [],
        "brand_answer": {
            "directness": "not_applicable",
            "specificity": "not_applicable",
            "supported_facets": [],
            "contradictions": [],
        },
        "evidence": [],
        "uncertainties": [],
    }


def _reconcile(
    raw: str,
    profile: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    scoped_catalog = _scope_entity_catalog_to_profile(catalog, profile)
    return _reconcile_annotation(
        _blank_annotation(),
        {
            "answer": raw,
            "answer_sha256": "raw-sha256",
            "answer_model": "provider/model",
        },
        profile,
        scoped_catalog,
    )


def _attributed(
    annotation: dict[str, Any],
    canonical_name: str,
) -> bool:
    return any(
        mention.get("canonical_name") == canonical_name
        and mention.get("attributed_to_target") is True
        for mention in annotation.get("entity_mentions") or []
    )


class SemanticGateStopAdversarialTests(unittest.TestCase):
    def test_jois_answer_255_nested_condition_uses_real_canonical(self) -> None:
        canonical = "Рассрочка 0% для ЖК «Джойс»"
        raw = (
            "#### 3. MR Group (сегмент MR Premium)\n"
            "* **JOIS** (Хорошёво-Мнёвники), **SLAVA** "
            "(м. Белорусская), **МИRА** (Алексеевская):\n"
            "  * *Условия:* Рассрочка 0% с ПВ от 5–30% и периодом "
            "выплат до 20 месяцев (до завершения строительства)."
        )
        profile = {
            "brand_name": "ЖК «Джойс»",
            "brand_aliases": ["JOIS", "Джойс"],
            "products": ["рассрочка 0% с ПВ 30%"],
            "entity_scope": [
                {
                    "canonical_name": "ЖК «Джойс»",
                    "aliases": ["JOIS", "Джойс"],
                    "entity_type": "primary_brand",
                    "relationship": "self",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": "MR Group",
                    "aliases": ["МР Групп"],
                    "entity_type": "business_unit",
                    "relationship": "operated_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": "Рассрочка 0%",
                    "aliases": ["рассрочка с фиксированными платежами"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
            ],
        }
        catalog = {
            "target_aliases": ["ЖК «Джойс»", "JOIS"],
            "entities": [
                {
                    "canonical_name": canonical,
                    "aliases": [
                        {
                            "value": "рассрочка 0%",
                            "match_policy": "requires_target_attribution",
                        },
                        {
                            "value": "беспроцентная рассрочка",
                            "match_policy": "requires_target_attribution",
                        },
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ],
        }

        annotation = _reconcile(raw, profile, catalog)

        self.assertTrue(_attributed(annotation, canonical))

    def test_profile_product_does_not_confirm_extended_catalog_canonical(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["Campaign 360"],
            "entity_scope": [],
        }
        catalog = {
            "target_aliases": ["Example"],
            "entities": [
                {
                    "canonical_name": "Campaign 360 Invented Suite",
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ],
        }

        entity = _scope_entity_catalog_to_profile(
            catalog,
            profile,
        )["entities"][0]

        self.assertFalse(entity["_profile_membership_confirmed"])
        self.assertEqual(entity["category"], "other")
        self.assertEqual(entity["target_relationship"], "unrelated")

    def test_long_profile_name_does_not_confirm_one_token_suffix(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["Campaign 360 Enterprise Analytics"],
            "entity_scope": [],
        }
        catalog = {
            "target_aliases": ["Example"],
            "entities": [
                {
                    "canonical_name": (
                        "Campaign 360 Enterprise Analytics Invented"
                    ),
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ],
        }

        entity = _scope_entity_catalog_to_profile(
            catalog,
            profile,
        )["entities"][0]

        self.assertFalse(entity["_profile_membership_confirmed"])
        self.assertEqual(entity["category"], "other")

    def test_owner_word_suffix_does_not_bypass_membership(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["Campaign 360 Enterprise Analytics"],
            "entity_scope": [],
        }
        for suffix in (
            "Group",
            "Группа",
            "MR",
            "МР",
            "AI",
            "ＭＲ",
            "ΑΙ",
            "グループ",
            "人工智能",
        ):
            with self.subTest(suffix=suffix):
                catalog = {
                    "target_aliases": ["Example"],
                    "entities": [
                        {
                            "canonical_name": (
                                "Campaign 360 Enterprise Analytics "
                                f"{suffix}"
                            ),
                            "aliases": [],
                            "category": "target",
                            "target_relationship": "portfolio_entity",
                            "commercially_relevant": True,
                            "mention_policy": "standalone",
                        }
                    ],
                }

                entity = _scope_entity_catalog_to_profile(
                    catalog,
                    profile,
                )["entities"][0]

                self.assertFalse(entity["_profile_membership_confirmed"])
                self.assertEqual(entity["category"], "other")

    def test_generic_tail_does_not_inherit_long_product_membership(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["Campaign 360 Enterprise Analytics"],
            "entity_scope": [],
        }
        catalog = {
            "target_aliases": ["Example"],
            "entities": [
                {
                    "canonical_name": "Enterprise Analytics",
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ],
        }

        entity = _scope_entity_catalog_to_profile(
            catalog,
            profile,
        )["entities"][0]

        self.assertFalse(entity["_profile_membership_confirmed"])
        self.assertEqual(entity["category"], "other")

    def test_catalog_owner_alias_cannot_donate_city_bay_feature_to_jois(
        self,
    ) -> None:
        canonical = "Квартиры с террасами JOIS"
        profile = {
            "brand_name": "JOIS",
            "brand_aliases": [],
            "products": [canonical],
            "entity_scope": [
                {
                    "canonical_name": "JOIS",
                    "aliases": [],
                    "entity_type": "primary_brand",
                    "relationship": "self",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": "MR Group",
                    "aliases": ["МР Групп"],
                    "entity_type": "business_unit",
                    "relationship": "operated_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": canonical,
                    "aliases": ["квартиры с террасами"],
                    "entity_type": "product",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
            ],
        }
        catalog = {
            "target_aliases": ["JOIS"],
            "entities": [
                {
                    "canonical_name": canonical,
                    "aliases": [
                        {
                            "value": "квартиры с террасами",
                            "match_policy": "requires_target_attribution",
                        },
                        {
                            "value": "MR Group",
                            "match_policy": "requires_target_attribution",
                        },
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ],
        }

        annotation = _reconcile(
            "MR Group предлагает квартиры с террасами в проекте City Bay.",
            profile,
            catalog,
        )

        self.assertFalse(_attributed(annotation, canonical))

    def test_catalog_owner_canonical_cannot_donate_city_bay_feature(
        self,
    ) -> None:
        profile_name = "Квартиры с зелёными террасами JOIS"
        poisoned_name = profile_name + " MR Group"
        profile = {
            "brand_name": "JOIS",
            "brand_aliases": [],
            "products": [profile_name],
            "entity_scope": [
                {
                    "canonical_name": "JOIS",
                    "aliases": [],
                    "entity_type": "primary_brand",
                    "relationship": "self",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": "MR Group",
                    "aliases": ["МР Групп"],
                    "entity_type": "business_unit",
                    "relationship": "operated_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": profile_name,
                    "aliases": ["квартиры с зелёными террасами"],
                    "entity_type": "product",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
            ],
        }
        catalog = {
            "target_aliases": ["JOIS"],
            "entities": [
                {
                    "canonical_name": poisoned_name,
                    "aliases": [
                        {
                            "value": "квартиры с зелёными террасами",
                            "match_policy": "requires_target_attribution",
                        }
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ],
        }

        annotation = _reconcile(
            (
                "MR Group предлагает квартиры с зелёными террасами "
                "в проекте City Bay."
            ),
            profile,
            catalog,
        )

        self.assertFalse(_attributed(annotation, poisoned_name))

    def test_competitor_colon_card_closes_target_heading_scope(self) -> None:
        profile, catalog, canonical = self._white_box_context()
        for raw in (
            "### JOIS\n**ПИК:**\n- White Box доступен.",
            "### JOIS\nПИК:\n- White Box доступен.",
        ):
            with self.subTest(raw=raw):
                annotation = _reconcile(raw, profile, catalog)
                self.assertFalse(_attributed(annotation, canonical))

    def test_inline_competitor_label_cannot_inherit_target_heading(self) -> None:
        profile, catalog, canonical = self._white_box_context()
        for raw in (
            "### JOIS\n- ПИК: White Box доступен.",
            "### JOIS\n- **ПИК:** White Box доступен.",
            "### JOIS\n- **ПИК** — White Box доступен.",
            "### JOIS\n- ПИК / White Box доступен.",
            "### JOIS\n- ПИК - White Box доступен.",
            "### JOIS\n- ПИК · White Box доступен.",
            "### JOIS\n- ПИК → White Box доступен.",
            "### JOIS\n- ПИК ⇒ White Box доступен.",
            "### JOIS\n- ПИК ⟶ White Box доступен.",
            "### JOIS\n- ПИК: отделка White Box доступна.",
        ):
            with self.subTest(raw=raw):
                annotation = _reconcile(raw, profile, catalog)
                self.assertFalse(_attributed(annotation, canonical))

    def test_nested_competitor_path_cannot_inherit_jois_scope(self) -> None:
        profile, catalog, canonical = self._white_box_context()
        for raw in (
            "* JOIS:\n  * ПИК:\n    * White Box доступен.",
            (
                "* JOIS:\n"
                "  * Застройщик: ПИК\n"
                "  * White Box доступен."
            ),
            "* JOIS:\n  * Условия: У ПИК доступен White Box.",
        ):
            with self.subTest(raw=raw):
                annotation = _reconcile(raw, profile, catalog)
                self.assertFalse(_attributed(annotation, canonical))

    @staticmethod
    def _white_box_context() -> tuple[dict[str, Any], dict[str, Any], str]:
        canonical = "White Box JOIS"
        profile = {
            "brand_name": "JOIS",
            "brand_aliases": [],
            "products": [canonical],
            "entity_scope": [
                {
                    "canonical_name": "JOIS",
                    "aliases": [],
                    "entity_type": "primary_brand",
                    "relationship": "self",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": canonical,
                    "aliases": ["White Box"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
            ],
        }
        catalog = {
            "target_aliases": ["JOIS"],
            "entities": [
                {
                    "canonical_name": canonical,
                    "aliases": [
                        {
                            "value": "White Box",
                            "match_policy": "requires_target_attribution",
                        }
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ],
        }
        return profile, catalog, canonical

    def test_critic_cannot_close_warning_with_empty_observation(self) -> None:
        payload = {
            "answers": [],
            "entity_catalog": {"entities": []},
            "deterministic_warnings": [
                {
                    "code": "scope_leakage",
                    "severity": "important",
                    "finding": "Материальная утечка scope.",
                    "answer_ids": [42],
                }
            ],
        }
        review = _normalize_review(
            {
                "verdict": "pass",
                "summary": "Проверка завершена.",
                "anomalies": [
                    {
                        "code": "scope_leakage",
                        "severity": "observation",
                        "finding": "Проверено.",
                        "answer_ids": [],
                        "entities": [],
                    }
                ],
                "policy_adjustments": [],
                "annotation_guidance": "",
                "acceptance_checks": ["Проверены метрики."],
            }
        )

        errors = _critic_review_validation_errors(review, payload=payload)

        self.assertTrue(errors)

    def test_scope_leakage_pass_needs_and_accepts_independent_hit_vectors(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["Campaign 360"],
            "entity_scope": [],
        }
        catalog = _scope_entity_catalog_to_profile(
            {
                "target_aliases": ["Example"],
                "entities": [
                    {
                        "canonical_name": "Campaign 360",
                        "aliases": [],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    }
                ],
            },
            profile,
        )
        complete_slice = {
            "mention_count": 1,
            "mention_rate": 100.0,
            "expected_answers": 1,
            "completed_answers": 1,
            "annotated_answers": 1,
            "valid_answers": 1,
            "coverage_rate": 100.0,
            "data_state": "complete",
        }
        warning = {
            "code": "scope_leakage",
            "severity": "important",
            "finding": "Два среза совпали и требуют независимой проверки.",
            "providers": ["ChatGPT"],
        }
        payload = {
            "site_profile": profile,
            "entity_catalog": catalog,
            "candidate_metrics": {
                "providers": [
                    {
                        "name": "ChatGPT",
                        "parent_discovery": dict(complete_slice),
                        "portfolio_capture": dict(complete_slice),
                    }
                ]
            },
            "deterministic_warnings": [warning],
            "answers": [
                {
                    "answer_id": 7,
                    "mode": "web",
                    "provider": "openai",
                    "scenario_role": "unbranded_discovery",
                    "status": "completed",
                    "metric_eligible": True,
                    "annotation_state": "current",
                    "raw_answer_truncated": False,
                    "raw_answer": "Example предлагает Campaign 360.",
                    "annotation": {
                        "valid": True,
                        "target_mentioned": True,
                        "entity_mentions": [
                            {
                                "canonical_name": "Campaign 360",
                                "position": None,
                                "role": "mentioned",
                                "attributed_to_target": False,
                                "evidence": "Campaign 360",
                            }
                        ],
                    },
                }
            ],
        }
        review = _normalize_review(
            {
                "verdict": "pass",
                "summary": "Независимые трассы пересчитаны.",
                "anomalies": [
                    {
                        "code": "scope_leakage",
                        "severity": "observation",
                        "finding": (
                            "Брендовый и продуктовый векторы независимо "
                            "восстановлены из разных полей канонической разметки."
                        ),
                        "answer_ids": [],
                        "entities": [],
                    }
                ],
                "policy_adjustments": [],
                "annotation_guidance": "",
                "acceptance_checks": [
                    "Независимо сверены брендовый и продуктовый answer_id-векторы."
                ],
            }
        )

        self.assertTrue(
            _scope_leakage_warning_machine_resolved(payload, warning)
        )
        self.assertEqual(
            _critic_review_validation_errors(review, payload=payload),
            [],
        )

        payload["answers"][0]["raw_answer_truncated"] = True
        self.assertFalse(
            _scope_leakage_warning_machine_resolved(payload, warning)
        )
        self.assertTrue(
            _critic_review_validation_errors(review, payload=payload)
        )


if __name__ == "__main__":
    unittest.main()
