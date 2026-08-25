from __future__ import annotations

from dataclasses import replace
import json
import unittest

from app.services.claim_ledger import (
    ClaimLedgerManifest,
    SourceClaim,
    build_claim_ledger,
    claim_coverage_references,
    reconstruct_claim_ledger,
    validate_claim_coverage,
    validate_claim_ledger,
)


class ClaimLedgerRoundtripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "brand.name": "Реалвеб 🤖",
            "metrics": {
                "visible": True,
                "rate": 66.7,
                "rank": 0,
                "note": None,
            },
            "answers": [
                "Первый ответ.\nВторая строка с 漢字 и 👩‍💻.",
                "",
            ],
            'ключ с "кавычкой"': "точное значение",
        }

    def test_nested_unicode_value_reconstructs_exactly(self) -> None:
        claims, manifest = build_claim_ledger(
            self.source,
            document_id="run:unicode",
            target_fragment_utf8_bytes=24,
        )

        self.assertEqual(reconstruct_claim_ledger(claims, manifest), self.source)
        validate_claim_ledger(claims, manifest)
        self.assertEqual(manifest.scalar_count, 8)
        self.assertGreater(manifest.claim_count, manifest.scalar_count)
        self.assertTrue(
            any(
                claim.json_path == '$["ключ с \\"кавычкой\\""]'
                for claim in claims
            )
        )

    def test_ids_and_manifest_are_deterministic(self) -> None:
        first_claims, first_manifest = build_claim_ledger(
            self.source,
            document_id="run:stable",
            target_fragment_utf8_bytes=19,
        )
        second_claims, second_manifest = build_claim_ledger(
            self.source,
            document_id="run:stable",
            target_fragment_utf8_bytes=19,
        )

        self.assertEqual(
            [claim.claim_id for claim in first_claims],
            [claim.claim_id for claim in second_claims],
        )
        self.assertEqual(first_manifest.as_dict(), second_manifest.as_dict())

    def test_json_database_roundtrip_preserves_validation(self) -> None:
        claims, manifest = build_claim_ledger(
            self.source,
            document_id="run:persisted",
            target_fragment_utf8_bytes=23,
        )
        serialized_claims = json.loads(
            json.dumps([claim.as_dict() for claim in claims], ensure_ascii=False)
        )
        serialized_manifest = json.loads(
            json.dumps(manifest.as_dict(), ensure_ascii=False)
        )

        self.assertEqual(
            reconstruct_claim_ledger(serialized_claims, serialized_manifest),
            self.source,
        )
        self.assertEqual(
            ClaimLedgerManifest.from_dict(serialized_manifest).as_dict(),
            manifest.as_dict(),
        )
        self.assertEqual(
            SourceClaim.from_dict(serialized_claims[0]).as_dict(),
            claims[0].as_dict(),
        )

    def test_serialized_manifest_is_detached_from_frozen_manifest(self) -> None:
        claims, manifest = build_claim_ledger(
            self.source,
            document_id="run:detached-manifest",
            target_fragment_utf8_bytes=23,
        )
        serialized = manifest.as_dict()
        serialized["structure"]["entries"][0]["key"] = "changed"

        self.assertEqual(reconstruct_claim_ledger(claims, manifest), self.source)

    def test_claim_input_order_does_not_change_reconstruction(self) -> None:
        claims, manifest = build_claim_ledger(
            self.source,
            document_id="run:reordered-input",
            target_fragment_utf8_bytes=20,
        )

        self.assertEqual(reconstruct_claim_ledger(list(reversed(claims)), manifest), self.source)

    def test_empty_string_still_has_a_claim(self) -> None:
        claims, manifest = build_claim_ledger(
            "",
            document_id="run:empty",
            target_fragment_utf8_bytes=1,
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].excerpt, "")
        self.assertEqual(claims[0].source_utf8_length, 0)
        self.assertEqual(reconstruct_claim_ledger(claims, manifest), "")


class ClaimLedgerFragmentTests(unittest.TestCase):
    def test_offsets_cover_every_utf8_byte_without_gap_or_overlap(self) -> None:
        source = "Абзац один.\n\nАбзац два 👩‍💻 — данные, числа 42.\n" * 40
        claims, manifest = build_claim_ledger(
            {"answer": source},
            document_id="answer:offsets",
            target_fragment_utf8_bytes=71,
        )
        answer_claims = [claim for claim in claims if claim.json_path == '$["answer"]']

        self.assertEqual("".join(claim.excerpt for claim in answer_claims), source)
        self.assertEqual(answer_claims[0].source_utf8_offset, 0)
        for previous, following in zip(answer_claims, answer_claims[1:], strict=False):
            self.assertEqual(previous.source_utf8_end, following.source_utf8_offset)
        self.assertEqual(
            answer_claims[-1].source_utf8_end,
            len(source.encode("utf-8")),
        )
        self.assertEqual(reconstruct_claim_ledger(claims, manifest)["answer"], source)

    def test_target_smaller_than_unicode_codepoint_keeps_codepoint_intact(self) -> None:
        source = "🤖🧠🚀"
        claims, manifest = build_claim_ledger(
            source,
            document_id="answer:codepoint",
            target_fragment_utf8_bytes=1,
        )

        self.assertEqual([claim.excerpt for claim in claims], list(source))
        self.assertTrue(all(claim.source_utf8_length == 4 for claim in claims))
        self.assertEqual(reconstruct_claim_ledger(claims, manifest), source)

    def test_long_text_stress_preserves_unique_tail_marker(self) -> None:
        tail_marker = "\nTAIL_MARKER::RW+AIV::НЕ_ПОТЕРЯТЬ::🧿"
        source = (
            "Содержательный ответ про AI visibility, сущности и цитирование. "
            "URL=https://example.test/раздел; score=66,7%.\n"
            * 8_000
        ) + tail_marker
        claims, manifest = build_claim_ledger(
            {"raw_answer": source},
            document_id="answer:stress-tail",
            target_fragment_utf8_bytes=257,
        )

        self.assertGreater(len(claims), 2_000)
        reconstructed = reconstruct_claim_ledger(claims, manifest)
        self.assertEqual(reconstructed["raw_answer"], source)
        self.assertTrue(reconstructed["raw_answer"].endswith(tail_marker))
        self.assertEqual(
            sum(claim.excerpt.count("TAIL_MARKER") for claim in claims),
            1,
        )

    def test_many_scalars_have_no_total_claim_count_cap(self) -> None:
        source = {"rows": [{"id": index, "ok": True} for index in range(5_001)]}
        claims, manifest = build_claim_ledger(
            source,
            document_id="answer:many-scalars",
            target_fragment_utf8_bytes=32,
        )

        self.assertEqual(manifest.scalar_count, 10_002)
        self.assertEqual(manifest.claim_count, 10_002)
        self.assertEqual(reconstruct_claim_ledger(claims, manifest), source)


class ClaimLedgerFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.claims, self.manifest = build_claim_ledger(
            {"answer": "Длинный точный ответ " * 30, "score": 17},
            document_id="run:fail-closed",
            target_fragment_utf8_bytes=48,
        )

    def test_tampered_excerpt_fails_closed(self) -> None:
        tampered = list(self.claims)
        tampered[0] = replace(tampered[0], excerpt=tampered[0].excerpt + "подмена")

        with self.assertRaisesRegex(ValueError, "length|digest|tampered"):
            validate_claim_ledger(tampered, self.manifest)

    def test_recomputed_excerpt_digest_still_breaks_claim_identity(self) -> None:
        tampered = list(self.claims)
        original = tampered[0]
        changed = original.excerpt + "!"
        import hashlib

        tampered[0] = replace(
            original,
            excerpt=changed,
            excerpt_sha256=hashlib.sha256(changed.encode("utf-8")).hexdigest(),
            source_utf8_length=len(changed.encode("utf-8")),
        )

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_claim_ledger(tampered, self.manifest)

    def test_missing_unknown_and_duplicate_claims_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Claim count|Missing"):
            validate_claim_ledger(self.claims[:-1], self.manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate claim"):
            validate_claim_ledger([*self.claims, self.claims[0]], self.manifest)

        foreign_claims, _foreign_manifest = build_claim_ledger(
            "чужой ответ",
            document_id="run:foreign",
            target_fragment_utf8_bytes=48,
        )
        replaced = [*self.claims[:-1], foreign_claims[0]]
        with self.assertRaisesRegex(ValueError, "Unknown claims|Missing claims"):
            validate_claim_ledger(replaced, self.manifest)

    def test_manifest_tamper_fails_closed(self) -> None:
        manifest_dict = self.manifest.as_dict()
        manifest_dict["structure"]["entries"][0]["key"] = "changed"

        with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
            validate_claim_ledger(self.claims, manifest_dict)

    def test_non_json_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "string keys"):
            build_claim_ledger(
                {1: "bad"},  # type: ignore[dict-item]
                document_id="run:bad-key",
            )
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            build_claim_ledger(float("nan"), document_id="run:nan")


class ClaimCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.claims, self.manifest = build_claim_ledger(
            {
                "answer": "Realweb упомянут в ответе. " * 20,
                "brand_mentioned": True,
                "position": 2,
            },
            document_id="run:coverage",
            target_fragment_utf8_bytes=52,
        )

    def test_complete_exact_references_validate(self) -> None:
        references = claim_coverage_references(self.claims)
        report = validate_claim_coverage(
            self.claims,
            references,
            manifest=self.manifest,
        )

        self.assertTrue(report.coverage_complete)
        self.assertEqual(report.covered_claim_count, len(self.claims))
        self.assertEqual(report.as_dict()["missing_claim_ids"], [])

    def test_partial_coverage_is_reported_only_when_explicitly_allowed(self) -> None:
        references = claim_coverage_references(self.claims)[:-1]

        with self.assertRaisesRegex(ValueError, "Incomplete claim coverage"):
            validate_claim_coverage(self.claims, references, manifest=self.manifest)
        report = validate_claim_coverage(
            self.claims,
            references,
            manifest=self.manifest,
            require_complete=False,
        )
        self.assertFalse(report.coverage_complete)
        self.assertEqual(len(report.missing_claim_ids), 1)

    def test_unknown_duplicate_and_tampered_references_fail_closed(self) -> None:
        references = claim_coverage_references(self.claims)
        unknown = [*references, {"claim_id": "clm_unknown", "excerpt_sha256": "0" * 64}]
        with self.assertRaisesRegex(ValueError, "Unknown coverage claim"):
            validate_claim_coverage(
                self.claims,
                unknown,
                manifest=self.manifest,
                require_complete=False,
            )

        duplicate = [references[0], references[0]]
        with self.assertRaisesRegex(ValueError, "Duplicate coverage claim"):
            validate_claim_coverage(
                self.claims,
                duplicate,
                manifest=self.manifest,
                require_complete=False,
            )

        tampered = [dict(reference) for reference in references]
        tampered[0]["excerpt_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Coverage digest mismatch"):
            validate_claim_coverage(
                self.claims,
                tampered,
                manifest=self.manifest,
            )

    def test_coverage_does_not_claim_semantic_understanding(self) -> None:
        references = [
            {
                **reference,
                "observation": "generic acknowledgement without interpretation",
            }
            for reference in claim_coverage_references(self.claims)
        ]

        report = validate_claim_coverage(
            self.claims,
            references,
            manifest=self.manifest,
        )
        self.assertTrue(report.coverage_complete)


if __name__ == "__main__":
    unittest.main()
