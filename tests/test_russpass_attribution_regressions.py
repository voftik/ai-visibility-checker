import unittest

from app.services.analyzer import (
    _has_markdown_scoped_target_attribution,
    _reconcile_annotation,
)


class RusspassAttributionRegressionTests(unittest.TestCase):
    """Regressions for list-owned service cards in RUSSPASS answers."""

    TARGET_ALIASES = ["RUSSPASS", "Russpass", "Русспасс"]

    @staticmethod
    def _reconcile_service(raw: str) -> dict[str, object]:
        canonical = "Бронирование отелей RUSSPASS"
        profile = {
            "brand_name": "RUSSPASS",
            "brand_aliases": ["Русспасс"],
            "products": [canonical],
            "entity_scope": [
                {
                    "canonical_name": "RUSSPASS",
                    "aliases": ["Russpass", "Русспасс"],
                    "entity_type": "primary_brand",
                    "relationship": "self",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": canonical,
                    "aliases": ["бронирование отелей"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
            ],
        }
        catalog = {
            "target_aliases": ["RUSSPASS", "Russpass", "Русспасс"],
            "entities": [
                {
                    "canonical_name": canonical,
                    "aliases": [
                        {
                            "value": "бронирование отелей",
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
        item = {
            "answer_id": 1,
            "valid": True,
            "target_mentioned": True,
            "target_position": 1,
            "target_role": "recommended",
            "sentiment": "positive",
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
        return _reconcile_annotation(
            item,
            {
                "answer": raw,
                "answer_sha256": "raw-sha256",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )

    def test_bold_list_owner_scopes_nested_card_fields(self) -> None:
        raw = (
            "*   **RUSSPASS (Руспасс)**\n"
            "    *   **Что это:** Национальный "
            "туристический "
            "сервис.\n"
            "    *   **Возможности:** Можно "
            "забронировать отели, "
            "купить билеты в музеи и на экскурсии.\n"
            "    *   **Оплата:** Российскими "
            "картами и СБП.\n"
            "*   **COMPETITOR**\n"
            "    *   **Возможности:** Другой набор услуг."
        )

        for entity_alias in (
            "отели",
            "билеты в музеи",
            "экскурсии",
        ):
            with self.subTest(entity_alias=entity_alias):
                self.assertTrue(
                    _has_markdown_scoped_target_attribution(
                        raw,
                        [entity_alias],
                        self.TARGET_ALIASES,
                        direct_target_aliases=self.TARGET_ALIASES,
                    )
                )

    def test_peer_bold_list_owner_closes_target_scope(self) -> None:
        raw = (
            "* **RUSSPASS**\n"
            "  * **Что это:** Туристический сервис.\n"
            "  * **Возможности:** Маршруты и экскурсии.\n"
            "* **COMPETITOR**\n"
            "  * **Что это:** Другой сервис.\n"
            "  * **Возможности:** Кэшбэк баллами "
            "и единый "
            "кошелёк."
        )

        self.assertFalse(
            _has_markdown_scoped_target_attribution(
                raw,
                ["кэшбэк баллами"],
                self.TARGET_ALIASES,
                direct_target_aliases=self.TARGET_ALIASES,
            )
        )

    def test_allowlisted_service_forms_match_only_inside_target_scope(
        self,
    ) -> None:
        for service_phrase in (
            "забронировать отель",
            "бронировать отели",
        ):
            raw = (
                "* **RUSSPASS**\n"
                "  * **Возможности:** Можно "
                f"{service_phrase} и купить билеты.\n"
                "* **COMPETITOR**\n"
                "  * **Возможности:** Кэшбэк."
            )
            with self.subTest(service_phrase=service_phrase):
                self.assertTrue(
                    _has_markdown_scoped_target_attribution(
                        raw,
                        ["бронирование отелей"],
                        self.TARGET_ALIASES,
                        direct_target_aliases=self.TARGET_ALIASES,
                    )
                )
                reconciled = self._reconcile_service(raw)
                mentions = reconciled["entity_mentions"]
                self.assertEqual(len(mentions), 1)
                self.assertTrue(mentions[0]["attributed_to_target"])
                self.assertIn(mentions[0]["evidence"], raw)

    def test_allowlisted_service_forms_do_not_cross_peer_owner(self) -> None:
        raw = (
            "* **RUSSPASS**\n"
            "  * **Возможности:** Маршруты и экскурсии.\n"
            "* **COMPETITOR**\n"
            "  * **Возможности:** Можно забронировать "
            "отель."
        )

        self.assertFalse(
            _has_markdown_scoped_target_attribution(
                raw,
                ["бронирование отелей"],
                self.TARGET_ALIASES,
                direct_target_aliases=self.TARGET_ALIASES,
            )
        )
        self.assertEqual(
            self._reconcile_service(raw)["entity_mentions"],
            [],
        )

    def test_allowlisted_action_does_not_fuzzy_match_another_object(
        self,
    ) -> None:
        raw = (
            "* **RUSSPASS**\n"
            "  * **Возможности:** Можно "
            "забронировать ресторан."
        )

        self.assertFalse(
            _has_markdown_scoped_target_attribution(
                raw,
                ["бронирование отелей"],
                self.TARGET_ALIASES,
                direct_target_aliases=self.TARGET_ALIASES,
            )
        )

    def test_allowlisted_form_rejects_competing_actor_in_target_field(
        self,
    ) -> None:
        raw = (
            "* **RUSSPASS**\n"
            "  * **Возможности:** COMPETITOR предлагает "
            "забронировать отель."
        )

        self.assertFalse(
            _has_markdown_scoped_target_attribution(
                raw,
                ["бронирование отелей"],
                self.TARGET_ALIASES,
                direct_target_aliases=self.TARGET_ALIASES,
            )
        )

    def test_allowlisted_form_does_not_turn_absence_into_an_offering(
        self,
    ) -> None:
        for negative_claim in (
            "Не забронировать отель.",
            "Нельзя забронировать отель.",
            "Забронировать отель нельзя.",
            "Забронировать отель невозможно.",
            "Невозможно забронировать отель.",
            "Без возможности забронировать отель.",
        ):
            raw = (
                "* **RUSSPASS**\n"
                f"  * **Возможности:** {negative_claim}"
            )
            with self.subTest(negative_claim=negative_claim):
                self.assertEqual(
                    self._reconcile_service(raw)["entity_mentions"],
                    [],
                )

    def test_allowlisted_form_is_not_enabled_for_plain_prose(self) -> None:
        for raw in (
            "RUSSPASS предлагает забронировать отель.",
            (
                "RUSSPASS\n"
                "- Возможности: можно "
                "забронировать отель."
            ),
        ):
            with self.subTest(raw=raw):
                self.assertFalse(
                    _has_markdown_scoped_target_attribution(
                        raw,
                        ["бронирование отелей"],
                        self.TARGET_ALIASES,
                        direct_target_aliases=self.TARGET_ALIASES,
                    )
                )
                self.assertEqual(
                    self._reconcile_service(raw)["entity_mentions"],
                    [],
                )


if __name__ == "__main__":
    unittest.main()
