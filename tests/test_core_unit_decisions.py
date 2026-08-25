from __future__ import annotations

import copy
import json
import unittest

from app.services.analyzer import (
    CORE_UNIT_DECISION_LEDGER_VERSION,
    ENTITY_CATALOG_LEAF_SCHEMA,
    SITE_PROFILE_LEAF_SCHEMA,
    _attach_core_decisions,
    _attach_core_decision_head,
    _core_decision_pointer,
    _core_decisions_from_inputs,
    _core_unit_claims,
    _deterministic_entity_catalog_union,
    _deterministic_site_profile_union,
    _model_payload_view,
    _normalize_core_dispositions,
    _preserve_entity_catalog_core_decisions,
    _preserve_site_profile_core_decisions,
    _validate_final_core_decisions,
    _validate_reducer_core_decisions,
)
from app.services.long_response import partition_text_records
from app.services.openrouter import OpenRouterError


def _blank_profile() -> dict[str, object]:
    return {
        "brand_name": "",
        "brand_aliases": [],
        "site_type": "",
        "category": "",
        "topics": [],
        "market": "",
        "business_model": "",
        "products": [],
        "audiences": [],
        "customer_jobs": [],
        "decision_criteria": [],
        "geography": [],
        "language": "",
        "positioning": "",
        "entity_scope": [],
        "evidence": [],
        "uncertainties": [],
        "confidence": "low",
    }


def _decision(
    claim: dict[str, object],
    *,
    disposition: str,
    quote: str,
    reason: str,
) -> dict[str, object]:
    return {
        "claim_id": claim["claim_id"],
        "unit_id": claim["unit_id"],
        "core_sha256": claim["core_sha256"],
        "disposition": disposition,
        "evidence_quote": quote,
        "reason": reason,
    }


class CoreUnitDecisionContractTests(unittest.TestCase):
    def _one_site_unit(self, text: str = "Example продаёт Atlas One") -> tuple[
        dict[str, object], dict[str, object]
    ]:
        units, _manifests = partition_text_records(
            [{"url": "https://example.test/", "main_text": text}],
            text_key="main_text",
            id_key="url",
            target_chars=1_000,
        )
        claim = _core_unit_claims(units)[0]
        return units[0], claim

    def test_leaf_schemas_require_explicit_core_dispositions(self) -> None:
        self.assertEqual(
            set(SITE_PROFILE_LEAF_SCHEMA["required"]),
            {"profile", "core_disposition"},
        )
        self.assertEqual(
            set(ENTITY_CATALOG_LEAF_SCHEMA["required"]),
            {"catalog", "core_dispositions"},
        )

    def test_grounded_blank_or_unquoted_profile_fails_closed(self) -> None:
        _unit, claim = self._one_site_unit()
        grounded = _decision(
            claim,
            disposition="grounded_fact",
            quote="Atlas One",
            reason="В core буквально назван продукт Atlas One.",
        )
        with self.assertRaisesRegex(OpenRouterError, "blank"):
            _normalize_core_dispositions(
                [grounded],
                expected_claims=[claim],
                analytic_output=_blank_profile(),
                output_kind="site_profile",
            )

        unquoted = _blank_profile()
        unquoted["brand_name"] = "Example"
        with self.assertRaisesRegex(OpenRouterError, "absent"):
            _normalize_core_dispositions(
                [grounded],
                expected_claims=[claim],
                analytic_output=unquoted,
                output_kind="site_profile",
            )

        generic = _blank_profile()
        generic["site_type"] = "website"
        generic["language"] = "ru"
        generic["evidence"] = ["Atlas One"]
        with self.assertRaisesRegex(OpenRouterError, "blank"):
            _normalize_core_dispositions(
                [grounded],
                expected_claims=[claim],
                analytic_output=generic,
                output_kind="site_profile",
            )

        grounded_profile = _blank_profile()
        grounded_profile["brand_name"] = "Atlas One"
        grounded_profile["evidence"] = ["Atlas One"]
        receipts = _normalize_core_dispositions(
            [grounded],
            expected_claims=[claim],
            analytic_output=grounded_profile,
            output_kind="site_profile",
        )
        with self.assertRaisesRegex(OpenRouterError, "blank or generic"):
            _validate_final_core_decisions(
                generic,
                receipts,
                output_kind="site_profile",
            )

        general_root = _blank_profile()
        general_root["category"] = "Услуги"
        general_root["market"] = "Цифровой рынок"
        with self.assertRaisesRegex(OpenRouterError, "general summary"):
            _validate_reducer_core_decisions(
                general_root,
                receipts,
                output_kind="site_profile",
                source_inputs=[grounded_profile],
            )

    def test_generic_no_fact_and_invented_reducer_fail_closed(self) -> None:
        _unit, claim = self._one_site_unit("Cookie settings and navigation")
        with self.assertRaisesRegex(OpenRouterError, "generic"):
            _normalize_core_dispositions(
                [
                    _decision(
                        claim,
                        disposition="explicit_no_fact",
                        quote="",
                        reason="Нет фактов",
                    )
                ],
                expected_claims=[claim],
                analytic_output=_blank_profile(),
                output_kind="site_profile",
            )

        receipts = _normalize_core_dispositions(
            [
                _decision(
                    claim,
                    disposition="explicit_no_fact",
                    quote="",
                    reason=(
                        "Core содержит только настройки cookie и "
                        "служебную "
                        "навигацию."
                    ),
                )
            ],
            expected_claims=[claim],
            analytic_output=_blank_profile(),
            output_kind="site_profile",
        )
        invented = _blank_profile()
        invented["brand_name"] = "Invented"
        with self.assertRaisesRegex(OpenRouterError, "invented material"):
            _validate_reducer_core_decisions(
                invented,
                receipts,
                output_kind="site_profile",
            )

    def test_receipt_tamper_duplicate_and_reorder_fail_closed(self) -> None:
        units, _manifests = partition_text_records(
            [
                {"url": "https://example.test/a", "main_text": "Alpha product"},
                {"url": "https://example.test/b", "main_text": "Beta product"},
            ],
            text_key="main_text",
            id_key="url",
            target_chars=1_000,
        )
        claims = _core_unit_claims(units)
        leaves: list[dict[str, object]] = []
        for claim in claims:
            quote = str(claim["core_text"])
            profile = _blank_profile()
            profile["brand_name"] = quote
            profile["evidence"] = [quote]
            receipts = _normalize_core_dispositions(
                [
                    _decision(
                        claim,
                        disposition="grounded_fact",
                        quote=quote,
                        reason=(
                            "В core буквально назван "
                            "отдельный продукт."
                        ),
                    )
                ],
                expected_claims=[claim],
                analytic_output=profile,
                output_kind="site_profile",
            )
            leaves.append(
                _attach_core_decisions(profile, receipts, [str(claim["unit_id"])])
            )

        with self.assertRaisesRegex(OpenRouterError, "coverage/order"):
            _core_decisions_from_inputs(
                list(reversed(leaves)),
                expected_claims=claims,
            )
        with self.assertRaisesRegex(OpenRouterError, "duplicates"):
            _core_decisions_from_inputs(
                [leaves[0], leaves[0]],
            )
        tampered = copy.deepcopy(leaves[0])
        tampered["_aiv_core_unit_decision_shards"][0][
            "evidence_quote_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(OpenRouterError, "digest mismatch"):
            _core_decisions_from_inputs([tampered])

    def test_ancestor_decision_head_never_embeds_linear_receipts(self) -> None:
        units, _manifests = partition_text_records(
            [
                {
                    "url": f"https://example.test/{index:04d}",
                    "main_text": f"Brand-{index:04d}",
                }
                for index in range(1_000)
            ],
            text_key="main_text",
            id_key="url",
            target_chars=1_000,
        )
        claims = _core_unit_claims(units)
        receipts: list[dict[str, object]] = []
        for claim in claims:
            quote = str(claim["core_text"])
            profile = _blank_profile()
            profile["brand_name"] = quote
            profile["evidence"] = [quote]
            receipts.extend(
                _normalize_core_dispositions(
                    [
                        _decision(
                            claim,
                            disposition="grounded_fact",
                            quote=quote,
                            reason=(
                                "В core буквально назван "
                                "отдельный бренд."
                            ),
                        )
                    ],
                    expected_claims=[claim],
                    analytic_output=profile,
                    output_kind="site_profile",
                )
            )

        head = _core_decision_pointer(receipts)  # type: ignore[arg-type]
        ancestor = _attach_core_decision_head(
            {"brand_name": "Merged"},
            receipts,  # type: ignore[arg-type]
            [str(claim["unit_id"]) for claim in claims],
        )
        self.assertLess(len(json.dumps(head)), 512)
        self.assertNotIn(
            "_aiv_core_unit_decision_shards",
            ancestor,
        )
        self.assertEqual(
            ancestor["_aiv_core_unit_decision_head"],
            head,
        )
        provider_view = _model_payload_view(ancestor)
        self.assertNotIn("_aiv_source_unit_ids", provider_view)
        self.assertLess(len(json.dumps(provider_view)), 1_024)


class CoreUnitDecisionScaleTests(unittest.TestCase):
    def test_large_site_corpus_preserves_adversarial_tail_exactly(self) -> None:
        marker = "TAIL_MARKER_SITE_9XZ"
        pages = [
            {
                "url": f"https://example.test/page-{index:04d}",
                "main_text": f"Navigation row {index:04d} without profile facts",
            }
            for index in range(1_000)
        ]
        pages.append(
            {
                "url": "https://example.test/tail",
                "main_text": f"Example launches named product {marker}",
            }
        )
        units, _manifests = partition_text_records(
            pages,
            text_key="main_text",
            id_key="url",
            target_chars=1_000,
        )
        claims = _core_unit_claims(units)
        leaves: list[dict[str, object]] = []
        for claim in claims:
            profile = _blank_profile()
            if marker in str(claim["core_text"]):
                profile["brand_name"] = "Example"
                profile["products"] = [marker]
                profile["evidence"] = [marker]
                raw = _decision(
                    claim,
                    disposition="grounded_fact",
                    quote=marker,
                    reason=(
                        "В tail core буквально назван "
                        "продукт клиента."
                    ),
                )
            else:
                raw = _decision(
                    claim,
                    disposition="explicit_no_fact",
                    quote="",
                    reason=(
                        "Core содержит только нумерованную "
                        "служебную строку "
                        "навигации."
                    ),
                )
            receipts = _normalize_core_dispositions(
                [raw],
                expected_claims=[claim],
                analytic_output=profile,
                output_kind="site_profile",
            )
            leaves.append(
                _attach_core_decisions(profile, receipts, [str(claim["unit_id"])])
            )

        receipts = _core_decisions_from_inputs(
            leaves,
            expected_claims=claims,
        )
        merged = _deterministic_site_profile_union(leaves)
        merged = _preserve_site_profile_core_decisions(merged, receipts)
        _validate_final_core_decisions(
            merged,
            receipts,
            output_kind="site_profile",
        )

        self.assertEqual(len(receipts), 1_001)
        self.assertEqual(
            {item["version"] for item in receipts},
            {CORE_UNIT_DECISION_LEDGER_VERSION},
        )
        self.assertIn(marker, merged["products"])
        self.assertIn(marker, merged["evidence"])

    def test_large_entity_corpus_preserves_every_core_and_tail(self) -> None:
        marker = "TAIL_MARKER_ENTITY_7QK"
        answers = [
            {
                "answer_id": index + 1,
                "answer": (
                    f"Named competitor Competitor-{index:04d}"
                    if index < 599
                    else f"Named competitor {marker}"
                ),
            }
            for index in range(600)
        ]
        units, _manifests = partition_text_records(
            answers,
            text_key="answer",
            id_key="answer_id",
            target_chars=1_000,
        )
        claims = _core_unit_claims(units)
        leaves: list[dict[str, object]] = []
        for index, claim in enumerate(claims):
            name = marker if marker in str(claim["core_text"]) else f"Competitor-{index:04d}"
            quote = str(claim["core_text"])
            catalog = {
                "target_aliases": [],
                "entities": [
                    {
                        "canonical_name": name,
                        "aliases": [],
                        "category": "competitor",
                        "target_relationship": "competitor",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                        "evidence": quote,
                    }
                ],
                "uncertainties": [],
            }
            receipts = _normalize_core_dispositions(
                [
                    _decision(
                        claim,
                        disposition="grounded_fact",
                        quote=quote,
                        reason=(
                            "В core буквально назван коммерческий "
                            "конкурент."
                        ),
                    )
                ],
                expected_claims=[claim],
                analytic_output=catalog,
                output_kind="entity_catalog",
            )
            leaves.append(
                _attach_core_decisions(catalog, receipts, [str(claim["unit_id"])])
            )

        receipts = _core_decisions_from_inputs(
            leaves,
            expected_claims=claims,
        )
        merged = _deterministic_entity_catalog_union(leaves)
        merged = _preserve_entity_catalog_core_decisions(merged, receipts)
        _validate_final_core_decisions(
            merged,
            receipts,
            output_kind="entity_catalog",
        )

        self.assertEqual(len(receipts), 600)
        self.assertEqual(len(merged["entities"]), 600)
        self.assertIn(
            marker,
            {item["canonical_name"] for item in merged["entities"]},
        )


if __name__ == "__main__":
    unittest.main()
