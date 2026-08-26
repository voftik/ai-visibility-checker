from __future__ import annotations

import copy
import hashlib
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
    _stable_json_sha256,
    _validate_core_decision_receipt,
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

    def test_unique_markdown_emphasis_quote_is_repaired_losslessly(self) -> None:
        core_text = (
            "Среди альтернатив названы *ST Tattoo*, *Tattoo Roko* и другие."
        )
        units, _manifests = partition_text_records(
            [{"answer_id": 614, "answer": core_text}],
            text_key="answer",
            id_key="answer_id",
            target_chars=1_000,
        )
        claim = _core_unit_claims(units)[0]
        self.assertEqual(claim["unit_id"], "614:000000")
        submitted_quote = "ST Tattoo, Tattoo Roko"
        catalog = {
            "target_aliases": [],
            "entities": [
                {
                    "canonical_name": "ST Tattoo",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "«ST Tattoo, Tattoo Roko и другие»",
                },
                {
                    "canonical_name": "Tattoo Roko",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                    "evidence": "«ST Tattoo, Tattoo Roko и другие»",
                },
            ],
            "uncertainties": [],
        }
        receipts = _normalize_core_dispositions(
            [
                _decision(
                    claim,
                    disposition="grounded_fact",
                    quote=submitted_quote,
                    reason="В core дословно перечислены две альтернативы.",
                )
            ],
            expected_claims=[claim],
            analytic_output=catalog,
            output_kind="entity_catalog",
        )

        receipt = receipts[0]
        exact_source_quote = "*ST Tattoo*, *Tattoo Roko*"
        source_start = core_text.index(exact_source_quote)
        repair = receipt["evidence_quote_repair"]
        self.assertEqual(receipt["evidence_quote"], exact_source_quote)
        self.assertEqual(
            receipt["evidence_quote_sha256"],
            hashlib.sha256(exact_source_quote.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(repair["submitted_quote"], submitted_quote)
        self.assertEqual(
            repair["submitted_quote_sha256"],
            hashlib.sha256(submitted_quote.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            repair["canonical_quote_sha256"],
            hashlib.sha256(submitted_quote.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            repair["source_quote_sha256"],
            hashlib.sha256(exact_source_quote.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(repair["core_start_char"], source_start)
        self.assertEqual(
            repair["core_end_char"],
            source_start + len(exact_source_quote),
        )
        self.assertEqual(repair["source_start_char"], source_start)
        self.assertEqual(
            repair["source_end_char"],
            source_start + len(exact_source_quote),
        )

        coordinate_tamper = copy.deepcopy(receipt)
        coordinate_tamper["evidence_quote_repair"]["core_start_char"] += 1
        coordinate_tamper.pop("decision_sha256")
        coordinate_tamper["decision_sha256"] = _stable_json_sha256(
            coordinate_tamper
        )
        with self.assertRaisesRegex(OpenRouterError, "coordinates"):
            _validate_core_decision_receipt(coordinate_tamper)

        # The immutable receipt survives attachment validation, while the
        # exact source spelling is restored to the analytic result before the
        # terminal evidence gate.
        attached = _attach_core_decisions(
            catalog,
            receipts,
            [str(claim["unit_id"])],
        )
        validated_receipts = _core_decisions_from_inputs(
            [attached],
            expected_claims=[claim],
        )
        restored = _preserve_entity_catalog_core_decisions(
            catalog,
            validated_receipts,
        )
        self.assertIn(exact_source_quote, restored["uncertainties"][-1])
        _validate_final_core_decisions(
            restored,
            validated_receipts,
            output_kind="entity_catalog",
        )

    def test_markdown_quote_repair_keeps_absolute_partition_offsets(self) -> None:
        answer = (
            ("Служебный контекст без сущностей. " * 40)
            + "Альтернативы: *ST Tattoo*, *Tattoo Roko*."
        )
        units, _manifests = partition_text_records(
            [{"answer_id": 614, "answer": answer}],
            text_key="answer",
            id_key="answer_id",
            target_chars=256,
        )
        unit = next(
            item for item in units if "ST Tattoo" in str(item["_lr_core_text"])
        )
        claim = _core_unit_claims([unit])[0]
        self.assertGreater(claim["start_char"], 0)
        submitted_quote = "ST Tattoo, Tattoo Roko"
        profile = _blank_profile()
        profile["brand_name"] = "ST Tattoo"
        profile["evidence"] = [submitted_quote]
        receipt = _normalize_core_dispositions(
            [
                _decision(
                    claim,
                    disposition="grounded_fact",
                    quote=submitted_quote,
                    reason="В core перечислены две альтернативы.",
                )
            ],
            expected_claims=[claim],
            analytic_output=profile,
            output_kind="site_profile",
        )[0]
        repair = receipt["evidence_quote_repair"]
        self.assertEqual(
            repair["source_start_char"],
            claim["start_char"] + repair["core_start_char"],
        )
        self.assertEqual(
            repair["source_end_char"],
            claim["start_char"] + repair["core_end_char"],
        )
        self.assertEqual(
            answer[
                repair["source_start_char"] : repair["source_end_char"]
            ],
            receipt["evidence_quote"],
        )

    def test_markdown_quote_repair_rejects_ambiguity_and_invention(self) -> None:
        repeated = (
            "*ST Tattoo*, *Tattoo Roko*; "
            "*ST Tattoo*, *Tattoo Roko*"
        )
        _unit, repeated_claim = self._one_site_unit(repeated)
        profile = _blank_profile()
        profile["brand_name"] = "ST Tattoo"
        profile["evidence"] = ["ST Tattoo, Tattoo Roko"]
        with self.assertRaisesRegex(OpenRouterError, "ambiguous"):
            _normalize_core_dispositions(
                [
                    _decision(
                        repeated_claim,
                        disposition="grounded_fact",
                        quote="ST Tattoo, Tattoo Roko",
                        reason="В core перечислены альтернативы.",
                    )
                ],
                expected_claims=[repeated_claim],
                analytic_output=profile,
                output_kind="site_profile",
            )

        _unit, claim = self._one_site_unit(
            "Альтернативы: *ST Tattoo*, *Tattoo Roko*."
        )
        for invented_quote in (
            "ST Tattoo, Tattoo Moko",
            "ST tattoo, Tattoo Roko",
            "ST Tattoo — Tattoo Roko",
        ):
            with self.subTest(quote=invented_quote):
                invented_profile = _blank_profile()
                invented_profile["brand_name"] = "ST Tattoo"
                invented_profile["evidence"] = [invented_quote]
                with self.assertRaisesRegex(
                    OpenRouterError,
                    "not an exact core substring",
                ):
                    _normalize_core_dispositions(
                        [
                            _decision(
                                claim,
                                disposition="grounded_fact",
                                quote=invented_quote,
                                reason=(
                                    "Ответ модели содержит изменённую цитату."
                                ),
                            )
                        ],
                        expected_claims=[claim],
                        analytic_output=invented_profile,
                        output_kind="site_profile",
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
