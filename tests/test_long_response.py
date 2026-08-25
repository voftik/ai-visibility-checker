from __future__ import annotations

from dataclasses import replace
import json
import unittest

from app.services.long_response import (
    StructuredContinuationLedger,
    exact_boundary_join,
    json_prefix_cursor,
    partition_text_records,
    split_lossless_text,
    text_sha256,
    verify_units,
)


class LosslessTextPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            "# AI-visibility: проверка 🤖\n\n"
            "- Бренд: «Реалвеб»\n"
            "- URL: https://example.test/путь?x=1&y=2\n"
            "- Цитата: **точный Markdown** и `inline_code` 🚀\n\n"
            "Абзац с составным emoji 👩‍💻, кириллицей, Latin и 漢字.\n"
        ) * 24

    def test_unicode_emoji_and_markdown_reconstruct_byte_for_byte(self) -> None:
        units, manifest = split_lossless_text(
            self.source,
            document_id="report:unicode",
            target_chars=320,
        )

        self.assertGreater(len(units), 1)
        self.assertEqual(verify_units(units, manifest), self.source)
        self.assertEqual(manifest.source_sha256, text_sha256(self.source))
        self.assertEqual(
            manifest.source_utf8_bytes,
            len(self.source.encode("utf-8")),
        )
        self.assertEqual(
            sum(unit.utf8_bytes for unit in units),
            len(self.source.encode("utf-8")),
        )
        self.assertEqual("".join(unit.text for unit in units), self.source)

    def test_tampered_unit_fails_closed(self) -> None:
        units, manifest = split_lossless_text(
            self.source,
            document_id="report:tamper",
            target_chars=300,
        )
        tampered = list(units)
        tampered[1] = replace(tampered[1], text=tampered[1].text + "!")

        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verify_units(tampered, manifest)

    def test_missing_unit_fails_closed(self) -> None:
        units, manifest = split_lossless_text(
            self.source,
            document_id="report:missing",
            target_chars=300,
        )

        with self.assertRaises(ValueError):
            verify_units([*units[:1], *units[2:]], manifest)

    def test_reordered_units_fail_closed(self) -> None:
        units, manifest = split_lossless_text(
            self.source,
            document_id="report:reordered",
            target_chars=300,
        )
        reordered = list(units)
        reordered[0], reordered[1] = reordered[1], reordered[0]

        with self.assertRaisesRegex(ValueError, "order|index"):
            verify_units(reordered, manifest)

    def test_manifest_unit_identity_tamper_fails_closed(self) -> None:
        units, manifest = split_lossless_text(
            self.source,
            document_id="report:identity",
            target_chars=300,
        )
        tampered = list(units)
        tampered[0] = replace(tampered[0], unit_id="other-document:000000")

        with self.assertRaisesRegex(ValueError, "identity|manifest|unit"):
            verify_units(tampered, manifest)

    def test_manifest_is_stable_across_json_database_roundtrip(self) -> None:
        _units, manifest = split_lossless_text(
            self.source,
            document_id="report:json-roundtrip",
            target_chars=300,
        )

        serialized = manifest.as_dict()
        roundtripped = json.loads(
            json.dumps(serialized, ensure_ascii=False)
        )

        self.assertEqual(roundtripped, serialized)
        self.assertIsInstance(serialized["units"], list)

    def test_semantic_overlap_keeps_a_boundary_spanning_owner_relation(self) -> None:
        prefix = "Вводный текст. " * 24
        relation = (
            "\n## Realweb\n"
            "- Сервис: programmatic-реклама для федеральных клиентов\n"
        )
        suffix = "Следующий раздел. " * 24
        source = prefix + relation + suffix

        units, manifest = split_lossless_text(
            source,
            document_id="answer:boundary-relation",
            target_chars=300,
            context_overlap_chars=120,
        )

        self.assertGreater(len(units), 1)
        self.assertEqual(verify_units(units, manifest), source)
        relation_units = [
            unit for unit in units if "programmatic-реклама" in unit.context_text
        ]
        self.assertTrue(relation_units)
        self.assertTrue(
            any("## Realweb" in unit.context_text for unit in relation_units)
        )
        self.assertEqual("".join(unit.text for unit in units), source)


class PartitionedRecordTests(unittest.TestCase):
    def test_records_expand_to_complete_code_owned_units(self) -> None:
        records = [
            {
                "answer_id": 17,
                "provider": "gemini",
                "answer": ("## Ответ 🎯\n\n- пункт\n" * 80),
            },
            {
                "answer_id": 18,
                "provider": "claude",
                "answer": ("Второй ответ без потерь. " * 70),
            },
        ]

        expanded, manifests = partition_text_records(
            records,
            text_key="answer",
            id_key="answer_id",
            target_chars=280,
        )

        self.assertEqual([item["document_id"] for item in manifests], ["17", "18"])
        self.assertGreater(len(expanded), len(records))
        for record, manifest in zip(records, manifests, strict=True):
            parts = sorted(
                (
                    item
                    for item in expanded
                    if str(item["answer_id"]) == manifest["document_id"]
                ),
                key=lambda item: item["_lr_unit_index"],
            )
            self.assertEqual(
                "".join(item["_lr_core_text"] for item in parts),
                record["answer"],
            )
            self.assertEqual(len(parts), manifest["unit_count"])
            self.assertEqual(
                [item["_lr_unit_id"] for item in parts],
                [unit["unit_id"] for unit in manifest["units"]],
            )
            self.assertTrue(
                all(item["_lr_source_sha256"] == manifest["source_sha256"] for item in parts)
            )


class ExactBoundaryJoinTests(unittest.TestCase):
    def test_exact_unicode_markdown_overlap_is_removed_once(self) -> None:
        overlap = "## Секция 🧠\n\n- **Точный** boundary\n"
        previous = "Начало отчёта.\n\n" + overlap
        following = overlap + "Продолжение 🚀\n"

        joined, overlap_chars = exact_boundary_join(
            previous,
            following,
            minimum=32,
        )

        self.assertEqual(joined, previous + "Продолжение 🚀\n")
        self.assertEqual(overlap_chars, len(overlap))
        self.assertEqual(joined.count(overlap), 1)

    def test_tampered_or_missing_overlap_fails_closed(self) -> None:
        previous = "A" * 80 + "🚀 exact Markdown **tail**"
        following = "🚀 exact Markdown *changed*" + "B" * 80

        with self.assertRaisesRegex(ValueError, "exact boundary overlap"):
            exact_boundary_join(previous, following, minimum=16)


class StructuredContinuationLedgerTests(unittest.TestCase):
    def test_literal_parts_are_hash_chained_in_sequence(self) -> None:
        initial = '{"items":[{"name":"А"},'
        ledger = StructuredContinuationLedger(
            document_id="report:structured",
            text=initial,
            overlap_chars=12,
        )
        first_cursor = ledger.cursor()
        first = first_cursor["expected_overlap"] + '{"name":"Б"},'
        first_part = ledger.append(first, sequence=1)
        second_cursor = ledger.cursor()
        second = second_cursor["expected_overlap"] + '{"name":"В"}]}'
        second_part = ledger.append(second, sequence=2)

        self.assertEqual(
            ledger.text,
            '{"items":[{"name":"А"},{"name":"Б"},{"name":"В"}]}',
        )
        self.assertEqual(
            first_part["previous_document_sha256"],
            text_sha256(initial),
        )
        self.assertEqual(
            second_part["previous_document_sha256"],
            first_part["document_sha256"],
        )
        manifest = ledger.manifest(complete=True)
        self.assertEqual(manifest["continuation_count"], 2)
        self.assertEqual(manifest["part_count"], 3)
        self.assertEqual(manifest["document_sha256"], text_sha256(ledger.text))

    def test_missing_overlap_fails_without_mutating_document(self) -> None:
        initial = '{"items":[1,'
        ledger = StructuredContinuationLedger(
            document_id="report:missing-overlap",
            text=initial,
            overlap_chars=8,
        )

        with self.assertRaisesRegex(ValueError, "exact literal boundary"):
            ledger.append('"new root":true}', sequence=1)

        self.assertEqual(ledger.text, initial)
        self.assertEqual(ledger.continuation_count, 0)

    def test_duplicate_or_no_progress_fragment_fails_closed(self) -> None:
        initial = '{"items":[1,'
        ledger = StructuredContinuationLedger(
            document_id="report:duplicate",
            text=initial,
            overlap_chars=len(initial),
        )

        with self.assertRaisesRegex(ValueError, "already seen|progress"):
            ledger.append(initial, sequence=1)

        self.assertEqual(ledger.manifest(complete=False)["part_count"], 1)

    def test_out_of_order_sequence_fails_closed(self) -> None:
        ledger = StructuredContinuationLedger(
            document_id="report:sequence",
            text='{"items":[1,',
            overlap_chars=8,
        )
        continuation = ledger.cursor()["expected_overlap"] + "2]}"

        with self.assertRaisesRegex(ValueError, "sequence mismatch"):
            ledger.append(continuation, sequence=2)

    def test_json_cursor_rejects_impossible_bracket_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "impossible closing bracket"):
            json_prefix_cursor('{"items":]}')

    def test_ledger_rejects_impossible_initial_prefix_immediately(self) -> None:
        with self.assertRaisesRegex(ValueError, "impossible closing bracket"):
            StructuredContinuationLedger(
                document_id="report:invalid-initial",
                text='{"items":]',
                overlap_chars=8,
            )


if __name__ == "__main__":
    unittest.main()
