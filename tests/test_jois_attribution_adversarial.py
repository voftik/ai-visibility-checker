import unittest
from typing import Any

from app.services.analyzer import _reconcile_annotation


class JoisAttributionAdversarialTests(unittest.TestCase):
    """Regression matrix for exact JOIS raw-answer attribution scopes."""

    @staticmethod
    def _entity(
        canonical_name: str,
        aliases: list[str],
    ) -> dict[str, Any]:
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

    @staticmethod
    def _profile_scope(
        canonical_name: str,
        aliases: list[str],
    ) -> dict[str, Any]:
        return {
            "canonical_name": canonical_name,
            "aliases": aliases,
            "entity_type": "service",
            "relationship": "offered_by",
            "commercially_relevant": True,
            "confidence": "high",
        }

    @classmethod
    def _context(
        cls,
        entities: list[tuple[str, list[str]]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        profile = {
            "brand_name": "ЖК «Джойс»",
            "brand_aliases": ["JOIS", "ЖК JOIS", "Джойс"],
            "products": [canonical for canonical, _aliases in entities],
            "entity_scope": [
                {
                    "canonical_name": "ЖК «Джойс»",
                    "aliases": ["JOIS", "ЖК JOIS", "Джойс"],
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
                *[
                    cls._profile_scope(canonical, aliases)
                    for canonical, aliases in entities
                ],
            ],
        }
        catalog = {
            "target_aliases": ["ЖК «Джойс»", "JOIS", "ЖК JOIS"],
            "entities": [
                cls._entity(canonical, aliases)
                for canonical, aliases in entities
            ],
        }
        return profile, catalog

    @staticmethod
    def _blank_item(
        seeded_mentions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "answer_id": 1,
            "valid": True,
            "target_mentioned": False,
            "target_position": None,
            "target_role": "absent",
            "sentiment": "unknown",
            "entity_mentions": seeded_mentions or [],
            "brand_answer": {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
            "evidence": [],
            "uncertainties": [],
        }

    @classmethod
    def _reconcile(
        cls,
        raw: str,
        entities: list[tuple[str, list[str]]],
        *,
        seed_attribution_for: str | None = None,
    ) -> dict[str, Any]:
        profile, catalog = cls._context(entities)
        seeded_mentions: list[dict[str, Any]] = []
        if seed_attribution_for is not None:
            seeded_mentions.append(
                {
                    "canonical_name": seed_attribution_for,
                    "position": None,
                    "role": "mentioned",
                    "attributed_to_target": True,
                    "evidence": raw,
                }
            )
        return _reconcile_annotation(
            cls._blank_item(seeded_mentions),
            {
                "answer": raw,
                "answer_sha256": "raw-sha256",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )

    @staticmethod
    def _mention(
        reconciled: dict[str, Any],
        canonical_name: str,
    ) -> dict[str, Any] | None:
        return next(
            (
                mention
                for mention in reconciled["entity_mentions"]
                if mention["canonical_name"] == canonical_name
            ),
            None,
        )

    def assertAttributed(
        self,
        reconciled: dict[str, Any],
        canonical_name: str,
        raw: str,
    ) -> None:
        mention = self._mention(reconciled, canonical_name)
        self.assertIsNotNone(mention, canonical_name)
        assert mention is not None
        self.assertTrue(mention["attributed_to_target"], canonical_name)
        self.assertIn(mention["evidence"], raw)

    def assertNotAttributed(
        self,
        reconciled: dict[str, Any],
        canonical_name: str,
    ) -> None:
        mention = self._mention(reconciled, canonical_name)
        self.assertFalse(
            mention is not None and mention.get("attributed_to_target") is True,
            canonical_name,
        )

    def test_answer_244_terraces_true_penthouses_false(self) -> None:
        raw = (
            "- JOIS (MR Group, район Хорошёво-Мнёвники). В продаже есть "
            "квартиры с приватными террасами на различных уровнях башен. "
            "([newnovostroy.ru](https://newnovostroy.ru/news/"
            "otkryla-prodazhi-terras-jois.html?utm_source=openai))"
        )
        terraces = "Квартиры с приватными террасами ЖК «Джойс»"
        penthouses = "Двухуровневые пентхаусы ЖК «Джойс»"
        reconciled = self._reconcile(
            raw,
            [
                (terraces, ["квартиры с приватными террасами"]),
                (penthouses, ["двухуровневые пентхаусы"]),
            ],
        )

        self.assertAttributed(reconciled, terraces, raw)
        self.assertNotAttributed(reconciled, penthouses)

    def test_answer_247_terraces_and_penthouses_true_parking_false(self) -> None:
        raw = (
            "**1. ЖК «Климашкина 7/11» (MR Group)**\n"
            "- Локация: Пресненский район, ЦАО\n"
            "- Камерный клубный дом (7–9 этажей, всего 46 резиденций), "
            "архитектор Сергей Чобан\n"
            "- Квартиры с приватными террасами **площадью до 37 м²**, "
            "три пентхауса с приватными патио на крыше\n"
            "- Высота потолков — до 7 метров, не более 4 квартир на этаже\n"
            "- Подземный паркинг, фитнес и SPA, библиотека-бар, "
            "скандинавский сад во дворе\n\n"
            "**2. ЖК «Джойс» (JOIS)**\n"
            "- Локация: 10 минут от Москва-Сити\n"
            "- Уникальный формат: квартиры с **зелёными террасами**, "
            "где предусмотрены зоны для джакузи и барбекю\n"
            "- На всех террасах выполнено частичное озеленение "
            "(уход можно доверить сервисной службе)\n"
            "- Двухуровневые пентхаусы с потолками до 6 метров\n"
            "- Ландшафтный парк >2 га с парящим мостом и парком "
            "скульптур Cosmoscow"
        )
        terraces = "Зелёные террасы ЖК «Джойс»"
        penthouses = "Двухуровневые пентхаусы ЖК «Джойс»"
        parking = "Подземный паркинг ЖК «Джойс»"
        reconciled = self._reconcile(
            raw,
            [
                (terraces, ["зелёными террасами"]),
                (penthouses, ["двухуровневые пентхаусы"]),
                (parking, ["Подземный паркинг"]),
            ],
        )

        self.assertAttributed(reconciled, terraces, raw)
        self.assertAttributed(reconciled, penthouses, raw)
        self.assertNotAttributed(reconciled, parking)

    def test_answer_250_white_box_true_beyond_eighty_word_scope(self) -> None:
        raw = (
            "### 3. ЖК JOIS (Застройщик: MR Group)\n"
            "* **Локация:** 3-й Силикатный проезд / 3-я Хорошёвская ул. "
            "(в 5–7 минутах от Сити).\n"
            "* **Цена за м²:** ~650 000 – 810 000 руб./м².\n"
            "* **Стоимость 3-комнатных (80–110 м²):** ~60–80 млн руб.\n"
            "* **Срок сдачи:** IV квартал 2027 года.\n"
            "* **Репутация застройщика:** **MR Group** — лидер высотного "
            "строительства бизнес- и премиум-класса в Москве с "
            "безупречным портфелем реализованных проектов "
            "(City Bay, SLT, Famous, Hide).\n\n"
            "**Почему выигрывает:**\n"
            "* **Архитектура и видовые характеристики:** Футуристичные "
            "башни ступенчатой формы с каскадными террасами и прямыми "
            "видами на небоскребы Москва-Сити.\n"
            "* **Качество отделки:** Квартиры сдаются в предчистовой "
            "отделке White Box, что существенно экономит время и бюджет "
            "на ремонт."
        )
        self.assertGreater(len(raw.split()), 80)
        self.assertLess(len(raw), 1200)
        mr_base = "MR Base"
        reconciled = self._reconcile(
            raw,
            [(mr_base, ["предчистовая отделка White Box", "White Box"])],
        )

        self.assertAttributed(reconciled, mr_base, raw)

    def test_answer_255_installment_inherits_nested_jois_scope(self) -> None:
        raw = (
            "#### 3. MR Group (сегмент MR Premium)\n"
            "* **JOIS** (Хорошёво-Мнёвники), **SLAVA** "
            "(м. Белорусская), **МИRА** (Алексеевская):\n"
            "  * *Условия:* Рассрочка 0% с ПВ от 5–30% и периодом "
            "выплат до 20 месяцев (до завершения строительства). "
            "Возможны фиксированные ежемесячные платежи с окончательным "
            "расчетом перед сдачей корпуса."
        )
        installment = "Рассрочка 0% от MR Group"
        reconciled = self._reconcile(
            raw,
            [(installment, ["Рассрочка 0%", "беспроцентная рассрочка"])],
        )

        self.assertAttributed(reconciled, installment, raw)

    def test_answer_271_online_purchase_true(self) -> None:
        raw = (
            "- **MR Group** — у компании есть собственный сценарий "
            "«онлайн-покупки» через **мобильное приложение**, где "
            "доступны выбор квартиры, бронирование, сбор документов, "
            "получение ЭЦП, оплата на карту или через **аккредитив** и "
            "подписание документов; в описании прямо сказано, что весь "
            "процесс можно провести полностью удалённо, а регистрация "
            "в Росреестре занимает до 10 рабочих дней.[3][4]"
        )
        online_purchase = "Онлайн-покупка MR Group"
        reconciled = self._reconcile(
            raw,
            [
                (
                    online_purchase,
                    ["онлайн-покупка", "мобильное приложение"],
                )
            ],
        )

        self.assertAttributed(reconciled, online_purchase, raw)

    def test_owner_service_limited_to_sibling_project_is_not_attributed(
        self,
    ) -> None:
        online_purchase = "Онлайн-покупка MR Group"
        for raw in (
            "MR Group предлагает онлайн-покупку только в проекте City Bay.",
            "MR Group предлагает онлайн-покупку лишь в проекте City Bay.",
            "MR Group предлагает онлайн-покупку лишь в City Bay.",
            "MR Group предлагает онлайн-покупку в проекте City Bay.",
            "MR Group предлагает онлайн-покупку для City Bay.",
            (
                "MR Group предлагает онлайн-покупку в жилом квартале "
                "City Bay."
            ),
            "MR Group предлагает онлайн-покупку в комплексе City Bay.",
            (
                "MR Group предлагает онлайн-покупку для City Bay, "
                "но не для JOIS."
            ),
            (
                "MR Group предлагает онлайн-покупку для City Bay; "
                "для JOIS сервис недоступен."
            ),
            (
                "MR Group предлагает онлайн-покупку. Однако для JOIS "
                "сервис недоступен."
            ),
            (
                "MR Group предлагает онлайн-покупку для City Bay; "
                "для JOIS она недоступна."
            ),
            (
                "MR Group предлагает онлайн-покупку для City Bay, тогда "
                "как для JOIS она недоступна."
            ),
            (
                "MR Group предлагает онлайн-покупку.\n\nОднако для JOIS "
                "сервис недоступен."
            ),
            (
                "MR Group предлагает онлайн-покупку.\n\n\nОднако для "
                "JOIS сервис недоступен."
            ),
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(online_purchase, ["онлайн-покупка"])],
                    seed_attribution_for=online_purchase,
                )
                self.assertNotAttributed(reconciled, online_purchase)

    def test_direct_target_negation_outweighs_positive_cue(self) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "JOIS пока не предлагает White Box.",
            "JOIS ещё не предлагает White Box.",
            "JOIS больше не предлагает White Box.",
            "JOIS никогда не предлагал White Box.",
            "JOIS предлагает White Box не для своего проекта.",
            "### JOIS\n- White Box доступен только у ПИК.",
            "White Box доступен, не включая JOIS.",
            "White Box доступен за исключением JOIS.",
            "### JOIS\n- White Box доступен от ПИК.",
            "### JOIS\n- White Box предлагается компанией ПИК.",
            "### JOIS\n- White Box — продукт ПИК.",
            "### JOIS\n- White Box — решение ПИК.",
            "### JOIS\n- White Box — сервис ПИК.",
            "### JOIS\n- White Box принадлежит ПИК.",
            "### JOIS\n- White Box от компании ПИК.",
            "White Box доступен, но для JOIS её нет.",
            "White Box на JOIS не распространяется.",
            "White Box доступен кроме проекта JOIS.",
            "### JOIS\n- Не White Box.",
            "### JOIS\n- Без White Box.",
            "### JOIS\n- Качество отделки: не White Box.",
            "### JOIS\n- Качество отделки: без White Box.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                    seed_attribution_for=white_box,
                )
                self.assertNotAttributed(reconciled, white_box)

    def test_short_structured_field_remains_attributed(self) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "### JOIS\n- Качество отделки: White Box.",
            "### JOIS\n- **Качество отделки:** White Box.",
            "### JOIS\n- Архитектура: White Box доступен.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                )
                self.assertAttributed(reconciled, white_box, raw)

    def test_allowed_multiword_owner_is_preserved(self) -> None:
        online_purchase = "Онлайн-покупка MR Group"
        raw = (
            "MR Group предлагает онлайн-покупку в проекте JOIS."
        )
        reconciled = self._reconcile(
            raw,
            [(online_purchase, ["онлайн-покупка"])],
        )

        self.assertAttributed(reconciled, online_purchase, raw)

    def test_explicit_direct_project_binding_allows_neutral_entity(self) -> None:
        mr_base = "MR Base"
        for raw in (
            "MR Group предлагает White Box в проекте JOIS.",
            "MR Group предлагает White Box исключительно в ЖК JOIS.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(mr_base, ["White Box"])],
                )
                self.assertAttributed(reconciled, mr_base, raw)

    def test_independent_competitor_claim_does_not_poison_target_claim(
        self,
    ) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "JOIS предлагает White Box. City Bay предлагает White Box.",
            "JOIS предлагает White Box; City Bay предлагает White Box.",
            "JOIS предлагает White Box.\nCity Bay предлагает White Box.",
            "JOIS предлагает White Box.\n- City Bay предлагает White Box.",
            (
                "### JOIS\n- Качество отделки: White Box.\n"
                "- ПИК: White Box."
            ),
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                )
                self.assertAttributed(reconciled, white_box, raw)

    def test_later_scope_qualification_cancels_owner_claim(self) -> None:
        online_purchase = "Онлайн-покупка MR Group"
        for raw in (
            (
                "MR Group предлагает онлайн-покупку.\n"
                "Условия уточняются.\n"
                "Для JOIS сервис недоступен."
            ),
            (
                "MR Group предлагает онлайн-покупку.\n\n"
                "Однако услуга доступна только в City Bay."
            ),
            (
                "MR Group предлагает онлайн-покупку.\n"
                "Однако сервис доступен только у ПИК."
            ),
            (
                "MR Group предлагает онлайн-покупку.\n"
                "Для JOIS она не предусмотрена."
            ),
            (
                "MR Group предлагает онлайн-покупку.\n"
                "Для JOIS она не реализована."
            ),
            (
                "MR Group предлагает онлайн-покупку: "
                "только City Bay и Symphony 34."
            ),
            (
                "Только City Bay и Symphony 34 поддерживают "
                "онлайн-покупку MR Group."
            ),
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(online_purchase, ["онлайн-покупка"])],
                    seed_attribution_for=online_purchase,
                )
                self.assertNotAttributed(reconciled, online_purchase)

    def test_contrastive_competing_owner_cancels_owner_claim(self) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "JOIS предлагает White Box.\nНо только City Bay предлагает White Box.",
            "JOIS предлагает White Box.\nНо White Box предлагает только City Bay.",
            "JOIS предлагает White Box.\nОднако White Box предлагает лишь ПИК.",
            "JOIS предлагает White Box.\nНо White Box — эксклюзив City Bay.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                    seed_attribution_for=white_box,
                )
                self.assertNotAttributed(reconciled, white_box)

    def test_passive_competing_owner_is_not_target_attribution(self) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "### JOIS\n- White Box разработан компанией ПИК.",
            "### JOIS\n- White Box создан компанией ПИК.",
            "### JOIS\n- White Box — разработка ПИК.",
            "### JOIS\n- White Box, разработанный ПИК, доступен.",
            "### JOIS\n- White Box, предлагаемый ПИК, доступен.",
            "### JOIS\n- White Box, спроектированный ПИК, доступен.",
            "### JOIS\n- White Box (ПИК).",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                )
                self.assertNotAttributed(reconciled, white_box)

    def test_explicit_direct_target_binding_overrides_upstream_author(self) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "ПИК разработал White Box специально для JOIS.",
            "ПИК разработал White Box для JOIS.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                )
                self.assertAttributed(reconciled, white_box, raw)

    def test_joint_exclusive_subject_keeps_explicit_target(self) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "Только JOIS и City Bay поддерживают White Box.",
            "JOIS и City Bay используют White Box.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                )
                self.assertAttributed(reconciled, white_box, raw)

    def test_nested_structured_field_value_inherits_target_heading(
        self,
    ) -> None:
        white_box = "White Box JOIS"
        for raw in (
            "### JOIS\n- Качество отделки:\n  - White Box.",
            "### JOIS\n- **Качество отделки:**\n  - White Box.",
            "### JOIS\nКачество отделки:\n- White Box.",
            "### JOIS\nКачество отделки:\nWhite Box.",
            "### JOIS\n**Качество отделки:**\nWhite Box.",
        ):
            with self.subTest(raw=raw):
                reconciled = self._reconcile(
                    raw,
                    [(white_box, ["White Box"])],
                )
                self.assertAttributed(reconciled, white_box, raw)

    def test_direct_actor_colon_with_positive_cue_is_attributed(self) -> None:
        white_box = "White Box JOIS"
        raw = "### Сравнение\n- JOIS: White Box доступен."
        reconciled = self._reconcile(
            raw,
            [(white_box, ["White Box"])],
        )

        self.assertAttributed(reconciled, white_box, raw)

    def test_hash_competitor_heading_at_same_or_deeper_level_ends_scope(
        self,
    ) -> None:
        terraces = "Квартиры с приватными террасами ЖК «Джойс»"
        for competitor_heading in (
            "### ЖК Rival (Застройщик: Rival Development)",
            "#### ЖК Rival (Застройщик: Rival Development)",
        ):
            with self.subTest(competitor_heading=competitor_heading):
                raw = (
                    "### ЖК JOIS\n"
                    f"{competitor_heading}\n"
                    "- Квартиры с приватными террасами доступны в продаже."
                )
                reconciled = self._reconcile(
                    raw,
                    [(terraces, ["квартиры с приватными террасами"])],
                    seed_attribution_for=terraces,
                )

                self.assertNotAttributed(reconciled, terraces)

    def test_numbered_competitor_heading_ends_hash_target_scope(self) -> None:
        raw = (
            "### ЖК JOIS\n"
            "**1. ЖК Rival (Застройщик: Rival Development)**\n"
            "- Квартиры с приватными террасами доступны в продаже."
        )
        terraces = "Квартиры с приватными террасами ЖК «Джойс»"
        reconciled = self._reconcile(
            raw,
            [(terraces, ["квартиры с приватными террасами"])],
            seed_attribution_for=terraces,
        )

        self.assertNotAttributed(reconciled, terraces)

    def test_hash_competitor_heading_ends_numbered_target_scope(self) -> None:
        raw = (
            "**2. ЖК JOIS**\n"
            "### ЖК Rival (Застройщик: Rival Development)\n"
            "- Квартиры с приватными террасами доступны в продаже."
        )
        terraces = "Квартиры с приватными террасами ЖК «Джойс»"
        reconciled = self._reconcile(
            raw,
            [(terraces, ["квартиры с приватными террасами"])],
            seed_attribution_for=terraces,
        )

        self.assertNotAttributed(reconciled, terraces)

    def test_repeated_competitor_owner_field_ends_target_scope(self) -> None:
        raw = (
            "### ЖК JOIS (Застройщик: MR Group)\n"
            "* **Застройщик:** Rival Development\n"
            "* **Качество отделки:** Квартиры сдаются в предчистовой "
            "отделке White Box."
        )
        mr_base = "MR Base"
        reconciled = self._reconcile(
            raw,
            [(mr_base, ["предчистовая отделка White Box", "White Box"])],
            seed_attribution_for=mr_base,
        )

        self.assertNotAttributed(reconciled, mr_base)

    def test_prose_comparison_does_not_donate_competitor_feature(self) -> None:
        raw = (
            "JOIS сравнивают с Rival Development, который предлагает "
            "квартиры с приватными террасами."
        )
        terraces = "Квартиры с приватными террасами ЖК «Джойс»"
        reconciled = self._reconcile(
            raw,
            [(terraces, ["квартиры с приватными террасами"])],
            seed_attribution_for=terraces,
        )

        self.assertNotAttributed(reconciled, terraces)

    def test_same_line_competitor_owner_does_not_donate_feature(self) -> None:
        raw = (
            "- JOIS сравнивают с Rival Development: компания "
            "Rival Development предлагает квартиры с приватными террасами."
        )
        terraces = "Квартиры с приватными террасами ЖК «Джойс»"
        reconciled = self._reconcile(
            raw,
            [(terraces, ["квартиры с приватными террасами"])],
            seed_attribution_for=terraces,
        )

        self.assertNotAttributed(reconciled, terraces)


if __name__ == "__main__":
    unittest.main()
