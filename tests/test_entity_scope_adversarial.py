import unittest

from app.services.analyzer import (
    _entity_alias_entries,
    _entity_attribution_aliases,
    _reconcile_annotation,
    _scope_entity_catalog_to_profile,
    _target_aliases,
)


def _blank_annotation() -> dict:
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


class EntityScopeAdversarialTests(unittest.TestCase):
    def test_catalog_alias_cannot_confirm_invented_portfolio_entity(self) -> None:
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
                    "canonical_name": "Invented Suite",
                    "aliases": ["Campaign 360"],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                }
            ],
        }

        scoped = _scope_entity_catalog_to_profile(catalog, profile)
        entity = scoped["entities"][0]

        self.assertFalse(entity["_profile_membership_confirmed"])
        self.assertEqual(entity["_profile_confirmed_match_aliases"], [])
        self.assertEqual(entity["category"], "other")
        self.assertEqual(entity["target_relationship"], "unrelated")

    def test_poisoned_target_duplicate_cannot_displace_competitor(self) -> None:
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
                    "canonical_name": "Invented Suite",
                    "aliases": ["Campaign 360"],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
                {
                    "canonical_name": "Invented Suite",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
            ],
        }

        scoped = _scope_entity_catalog_to_profile(catalog, profile)

        self.assertEqual(len(scoped["entities"]), 1)
        self.assertEqual(scoped["entities"][0]["category"], "competitor")
        self.assertEqual(
            scoped["entities"][0]["target_relationship"],
            "competitor",
        )

    def test_catalog_target_alias_and_generic_product_aliases_fail_closed(
        self,
    ) -> None:
        profile = {
            "brand_name": "JOIS",
            "brand_aliases": [],
            "products": ["MR Base"],
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
                    "canonical_name": "MR Base",
                    "aliases": ["MR BASE"],
                    "entity_type": "product",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
            ],
        }
        entity = {
            "canonical_name": "MR Base",
            "aliases": ["Group", "White Box", "Онлайн", "покупка"],
            "category": "target",
            "target_relationship": "portfolio_entity",
            "commercially_relevant": True,
            "mention_policy": "standalone",
        }
        catalog = {
            "target_aliases": ["JOIS", "Group"],
            "entities": [entity],
        }

        self.assertEqual(_target_aliases(profile, catalog), ["JOIS"])
        entries = _entity_alias_entries(
            entity,
            profile,
            excluded_aliases=_entity_attribution_aliases(
                profile,
                catalog,
                entity,
            ),
        )
        self.assertEqual(
            entries,
            [
                ("MR Base", "standalone"),
                ("White Box", "requires_target_attribution"),
            ],
        )

        for raw in (
            "ПИК Group предлагает ипотеку.",
            "JOIS предлагает покупку онлайн.",
        ):
            with self.subTest(raw=raw):
                reconciled = _reconcile_annotation(
                    _blank_annotation(),
                    {
                        "answer": raw,
                        "answer_sha256": "sha",
                        "answer_model": "provider/model",
                    },
                    profile,
                    catalog,
                )
                self.assertEqual(reconciled["entity_mentions"], [])
                if "Group" in raw:
                    self.assertFalse(reconciled["target_mentioned"])

        contextual = _reconcile_annotation(
            _blank_annotation(),
            {
                "answer": "JOIS предлагает White Box.",
                "answer_sha256": "sha",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )
        self.assertTrue(contextual["entity_mentions"][0]["attributed_to_target"])

        standalone = _reconcile_annotation(
            _blank_annotation(),
            {
                "answer": "MR Base доступен.",
                "answer_sha256": "sha",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )
        self.assertEqual(
            standalone["entity_mentions"][0]["canonical_name"],
            "MR Base",
        )


if __name__ == "__main__":
    unittest.main()
