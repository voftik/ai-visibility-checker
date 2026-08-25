import unittest

from app.services.analyzer import (
    _literal_target_attribution_evidence,
    _reconcile_annotation,
)


def _profile() -> dict:
    return {
        "brand_name": "Realweb",
        "brand_aliases": ["Реалвеб"],
        "products": [
            "стратегия и продвижение (performance, медиабаинг)",
            "креативные услуги: брейнштормы, спецпроекты",
            "исследования и аналитика, data-driven решения",
        ],
        "entity_scope": [
            {
                "canonical_name": "DOOH Realweb",
                "aliases": ["Programmatic DOOH", "INDOOR"],
                "entity_type": "service",
                "relationship": "offered_by",
                "commercially_relevant": True,
                "confidence": "high",
            }
        ],
    }


def _catalog() -> dict:
    def contextual(canonical_name: str, aliases: list[str]) -> dict:
        return {
            "canonical_name": canonical_name,
            "aliases": [
                {
                    "value": alias,
                    "match_policy": "requires_target_attribution",
                }
                for alias in aliases
            ],
            "category": "target",
            "target_relationship": "portfolio_entity",
            "commercially_relevant": True,
            "mention_policy": "requires_target_attribution",
        }

    return {
        "target_aliases": ["Realweb", "Реалвеб", "Риалвеб"],
        "entities": [
            contextual(
                "DOOH Realweb",
                ["DOOH", "programmatic DOOH", "INDOOR"],
            ),
            contextual(
                "Стратегия и продвижение Realweb",
                [
                    "performance",
                    "перформанс",
                    "performance-маркетинг",
                    "медиабаинг",
                ],
            ),
            contextual(
                "Исследования и аналитика Realweb",
                ["аналитика", "сквозная аналитика"],
            ),
            contextual(
                "Креативные услуги Realweb",
                ["креатив", "креативные услуги", "спецпроекты"],
            ),
        ],
    }


def _annotation(answer_id: int = 1) -> dict:
    return {
        "answer_id": answer_id,
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


def _reconcile(raw: str, *, role: str = "unbranded_discovery") -> dict:
    return _reconcile_annotation(
        _annotation(),
        {
            "answer": raw,
            "answer_sha256": "fixture-sha",
            "answer_model": "fixture/model",
            "scenario_role": role,
        },
        _profile(),
        _catalog(),
    )


def _mentions(report: dict) -> dict[str, dict]:
    return {
        str(item.get("canonical_name")): item
        for item in report.get("entity_mentions") or []
    }


class RealwebAttributionRegressionTests(unittest.TestCase):
    def test_long_markdown_claim_cannot_hide_competing_product_owner(self) -> None:
        detail = (
            "с подробным описанием условий размещения, интеграции, "
            "аналитики и сопровождения "
            * 12
        )
        competitor_claim = (
            f"## Realweb\n- programmatic {detail[:700]} — продукт OtherAgency"
        )
        target_claim = (
            f"## Realweb\n- programmatic {detail[:700]} — продукт Realweb"
        )
        separate_claim = (
            f"## Realweb\n- programmatic {detail[:700]}.\n"
            "## OtherAgency\n- отдельный продукт"
        )

        self.assertFalse(
            _literal_target_attribution_evidence(
                competitor_claim,
                ["programmatic"],
                ["realweb"],
                direct_target_aliases=["realweb"],
            )
        )
        self.assertTrue(
            _literal_target_attribution_evidence(
                target_claim,
                ["programmatic"],
                ["realweb"],
                direct_target_aliases=["realweb"],
            )
        )
        self.assertTrue(
            _literal_target_attribution_evidence(
                separate_claim,
                ["programmatic"],
                ["realweb"],
                direct_target_aliases=["realweb"],
            )
        )

    def test_long_same_sentence_programmatic_binding_is_kept(self) -> None:
        qualifiers = (
            "после подробного сопоставления каналов, аудиторий, инвентаря, "
            "географии, требований к brand safety и измерению результата "
            * 4
        )
        raw = (
            f"Realweb {qualifiers}предлагает programmatic DOOH "
            "для рекламодателей."
        )
        self.assertGreater(
            raw.index("programmatic DOOH") - (raw.index("Realweb") + 7),
            320,
        )

        mention = _mentions(_reconcile(raw))["DOOH Realweb"]

        self.assertTrue(mention["attributed_to_target"])
        self.assertEqual(mention["evidence"], raw)

    def test_literal_single_line_dooh_binding_is_kept(self) -> None:
        raw = "- **Realweb** — закупка DOOH через рекламные платформы."
        mention = _mentions(_reconcile(raw))["DOOH Realweb"]

        self.assertTrue(mention["attributed_to_target"])
        self.assertEqual(mention["evidence"], raw)

    def test_numbered_card_evidence_does_not_cross_into_competitor(self) -> None:
        raw = (
            "1. Realweb (Санкт-Петербург, Москва)\n"
            "- Сильная сторона: универсальное digital-агентство. "
            "Хорошо сочетает performance-маркетинг, медиазакупку, "
            "аналитику и стратегическую экспертизу.\n\n"
            "2. Nectarin (Москва)\n"
            "- Сильная сторона: full-service, креатив и аналитика."
        )
        mentions = _mentions(_reconcile(raw))

        for canonical in (
            "Стратегия и продвижение Realweb",
            "Исследования и аналитика Realweb",
        ):
            with self.subTest(canonical=canonical):
                mention = mentions[canonical]
                self.assertTrue(mention["attributed_to_target"])
                self.assertIn(mention["evidence"], raw)
                self.assertIn("Realweb", mention["evidence"])
                self.assertNotIn("Nectarin", mention["evidence"])

    def test_hash_card_restores_literal_strategy_and_analytics(self) -> None:
        raw = (
            "### 1. Realweb (Риалвеб)\n"
            "* **География:** Санкт-Петербург / Москва\n"
            "* **В чём сила:** Крупный игрок по объёмам медиазакупок и "
            "перформанс-маркетингу. Глубоко прорабатывает сквозную "
            "аналитику и бизнес-стратегии.\n\n"
            "### 2. MGCom\n"
            "* **В чём сила:** performance-маркетинг и аналитика."
        )
        mentions = _mentions(_reconcile(raw))

        for canonical in (
            "Стратегия и продвижение Realweb",
            "Исследования и аналитика Realweb",
        ):
            with self.subTest(canonical=canonical):
                mention = mentions[canonical]
                self.assertTrue(mention["attributed_to_target"])
                self.assertIn("Realweb", mention["evidence"])
                self.assertNotIn("MGCom", mention["evidence"])

    def test_same_line_realweb_analytics_relation_is_kept(self) -> None:
        raw = (
            "- **Сильный комплекс именно как партнёр «разработка + "
            "маркетинг + аналитика»**: в рейтинге СПб‑агентств Realweb "
            "упоминается как омниканальный игрок с полной связкой "
            "dev+media+CRM+аналитика."
        )
        mention = _mentions(_reconcile(raw))[
            "Исследования и аналитика Realweb"
        ]

        self.assertTrue(mention["attributed_to_target"])
        self.assertIn(mention["evidence"], raw)

    def test_nested_tender_card_keeps_only_literal_realweb_services(self) -> None:
        raw = (
            "- Realweb\n"
            "  - Подходит для: 360° digital, performance, media, "
            "retail media, аналитика, креатив.\n"
            "  - Как приглашать: через форму на сайте.\n\n"
            "- iConText\n"
            "  - Подходит для: спецпроекты."
        )
        mentions = _mentions(_reconcile(raw))

        for canonical in (
            "Стратегия и продвижение Realweb",
            "Исследования и аналитика Realweb",
            "Креативные услуги Realweb",
        ):
            with self.subTest(canonical=canonical):
                mention = mentions[canonical]
                self.assertTrue(mention["attributed_to_target"])
                self.assertIn("Realweb", mention["evidence"])
                self.assertNotIn("iConText", mention["evidence"])

    def test_intro_creative_term_does_not_leak_into_realweb_card(self) -> None:
        raw = (
            "Creative Digital / SMM: спецпроекты, креативные стратегии.\n\n"
            "### 1. Крупнейшие Performance & Media агентства\n\n"
            "* **Realweb (Риалвеб)**\n"
            "  * **Специализация:** Performance-маркетинг, медийная "
            "реклама, ритейл-медиа, аналитика, SMM, мобайл.\n"
            "* **MGCom**\n"
            "  * **Специализация:** performance и спецпроекты."
        )
        mentions = _mentions(_reconcile(raw))

        self.assertTrue(
            mentions["Стратегия и продвижение Realweb"][
                "attributed_to_target"
            ]
        )
        self.assertTrue(
            mentions["Исследования и аналитика Realweb"][
                "attributed_to_target"
            ]
        )
        self.assertNotIn("Креативные услуги Realweb", mentions)
        for canonical in (
            "Стратегия и продвижение Realweb",
            "Исследования и аналитика Realweb",
        ):
            self.assertNotIn("MGCom", mentions[canonical]["evidence"])

    def test_brand_diagnostic_not_applicable_is_repaired_from_raw(self) -> None:
        raw = (
            "### Кто такой Realweb сейчас\n"
            "Realweb — digital-агентство полного цикла.\n"
            "- Есть отдельный продуктовый раздел: DOOH в Realweb — "
            "быстрый запуск кампаний и аналитический подход."
        )
        report = _reconcile(raw, role="brand_diagnostic")

        self.assertEqual(report["brand_answer"]["directness"], "partial")
        self.assertEqual(report["brand_answer"]["specificity"], "specific")
        self.assertIn("offering", report["brand_answer"]["supported_facets"])

    def test_unconfirmed_contextual_alias_cannot_rename_product(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "products": ["Campaign 360"],
            "entity_scope": [],
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "Campaign 360",
                    "aliases": [
                        {
                            "value": "analytics",
                            "match_policy": "requires_target_attribution",
                        }
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ]
        }
        reconciled = _reconcile_annotation(
            _annotation(),
            {
                "answer": "Example предлагает analytics.",
                "answer_sha256": "fixture-sha",
                "answer_model": "fixture/model",
                "scenario_role": "unbranded_discovery",
            },
            profile,
            catalog,
        )

        self.assertEqual(reconciled["entity_mentions"], [])


if __name__ == "__main__":
    unittest.main()
