from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

from app.services.fact_binding import (
    AtomicFactBinding,
    FactBindingError,
    audit_statement_bindings,
    binding_references,
    extract_fact_bindings,
    validate_binding_integrity,
    validate_statement_bindings,
)


ENTITIES = {
    "chatgpt": ("ChatGPT", "OpenAI ChatGPT"),
    "gemini": ("Gemini", "Google Gemini"),
    "realweb": ("Realweb", "Риалвеб"),
}


class FactBindingExtractionTests(unittest.TestCase):
    def test_extracts_russian_and_english_atomic_facts(self) -> None:
        source = (
            "ChatGPT: mention rate 66.7% (4 of 6 answers), rank 2. "
            "Gemini: доля рекомендаций 50,0% "
            "(3 из 6 ответов), 1-е место. "
            "Realweb: доступность — не измерена."
        )

        bindings = extract_fact_bindings(
            source,
            child_id="child:model-summary",
            claim_id="claim:exact-text",
            entity_aliases=ENTITIES,
        )

        signatures = {binding.semantic_signature() for binding in bindings}
        self.assertIn(
            (
                "chatgpt",
                "mention_rate",
                "metric_value",
                None,
                "66.7",
                "percent",
                None,
                None,
                "unsigned",
                "neutral",
                None,
                None,
            ),
            signatures,
        )
        self.assertIn(
            (
                "chatgpt",
                "mention_rate",
                "metric_value",
                None,
                "4",
                "ratio",
                "4",
                "6",
                "unsigned",
                "neutral",
                None,
                None,
            ),
            signatures,
        )
        self.assertIn(
            (
                "gemini",
                "rank",
                "order",
                None,
                "1",
                "rank",
                None,
                None,
                "unsigned",
                "neutral",
                "rank",
                "1",
            ),
            signatures,
        )
        self.assertIn(
            (
                "realweb",
                "availability",
                "metric_state",
                "unavailable",
                None,
                None,
                None,
                None,
                "not_applicable",
                "neutral",
                None,
                None,
            ),
            signatures,
        )
        for binding in bindings:
            self.assertEqual(
                binding.source_excerpt,
                source[binding.source_char_start : binding.source_char_end],
            )
            self.assertEqual(
                binding.fact_lexeme,
                source[binding.fact_char_start : binding.fact_char_end],
            )
            validate_binding_integrity(binding, source_text=source)

    def test_direction_and_unit_are_typed_separately(self) -> None:
        source = (
            "ChatGPT: recommendation rate increased by +12.5 percentage points. "
            "Gemini: recommendation rate fell by -4,0%."
        )
        bindings = extract_fact_bindings(
            source,
            child_id="child:delta",
            entity_aliases=ENTITIES,
        )

        by_entity = {binding.entity: binding for binding in bindings}
        self.assertEqual(by_entity["chatgpt"].value, "12.5")
        self.assertEqual(by_entity["chatgpt"].unit, "percentage_point")
        self.assertEqual(by_entity["chatgpt"].sign, "positive")
        self.assertEqual(by_entity["chatgpt"].direction, "positive")
        self.assertEqual(by_entity["gemini"].value, "-4")
        self.assertEqual(by_entity["gemini"].unit, "percent")
        self.assertEqual(by_entity["gemini"].sign, "negative")
        self.assertEqual(by_entity["gemini"].direction, "negative")

    def test_direction_is_bound_locally_in_a_multi_entity_sentence(self) -> None:
        bindings = extract_fact_bindings(
            "ChatGPT: mention rate rose 12%, Gemini: mention rate fell 4%.",
            child_id="child:two-directions",
            entity_aliases=ENTITIES,
        )

        by_entity = {binding.entity: binding for binding in bindings}
        self.assertEqual(by_entity["chatgpt"].direction, "positive")
        self.assertEqual(by_entity["gemini"].direction, "negative")

    def test_defaults_support_already_typed_table_cells(self) -> None:
        bindings = extract_fact_bindings(
            "66,7% (4 из 6 ответов)",
            child_id="cell:chatgpt:mentions",
            entity_aliases={"chatgpt": ("ChatGPT",)},
            default_entity="chatgpt",
            default_metric="mention_rate",
        )

        self.assertEqual(len(bindings), 2)
        self.assertTrue(all(binding.entity == "chatgpt" for binding in bindings))
        self.assertTrue(all(binding.metric == "mention_rate" for binding in bindings))

    def test_ambiguous_aliases_fail_before_extraction(self) -> None:
        with self.assertRaisesRegex(FactBindingError, "Ambiguous alias"):
            extract_fact_bindings(
                "A: score 1.",
                child_id="ambiguous",
                entity_aliases={"first": ("A",), "second": ("a",)},
            )

    def test_binding_is_frozen_deterministic_and_json_roundtrippable(self) -> None:
        source = "Realweb: доля упоминаний 60,0%."
        first = extract_fact_bindings(
            source,
            child_id="child:stable",
            entity_aliases=ENTITIES,
        )
        second = extract_fact_bindings(
            source,
            child_id="child:stable",
            entity_aliases=ENTITIES,
        )

        self.assertEqual(first, second)
        with self.assertRaises(FrozenInstanceError):
            first[0].value = "99"  # type: ignore[misc]
        serialized = json.loads(json.dumps(first[0].as_dict(), ensure_ascii=False))
        self.assertEqual(AtomicFactBinding.from_dict(serialized), first[0])

    def test_tampering_fails_content_identity_check(self) -> None:
        binding = extract_fact_bindings(
            "Realweb: mention rate 60%.",
            child_id="child:tamper",
            entity_aliases=ENTITIES,
        )[0]

        with self.assertRaisesRegex(FactBindingError, "identity mismatch"):
            validate_binding_integrity(replace(binding, value="90"))
        with self.assertRaisesRegex(FactBindingError, "source digest mismatch"):
            validate_binding_integrity(binding, source_text="Realweb: mention rate 90%.")

    def test_no_binding_count_cap_and_tail_fact_survives(self) -> None:
        rows = [f"ChatGPT: mention rate {index}%\n" for index in range(1_001)]
        rows.append("Gemini: mention rate 77,7%\n")
        source = "".join(rows)

        bindings = extract_fact_bindings(
            source,
            child_id="child:long",
            entity_aliases=ENTITIES,
        )

        self.assertEqual(len(bindings), 1_002)
        self.assertEqual(bindings[-1].entity, "gemini")
        self.assertEqual(bindings[-1].value, "77.7")
        self.assertEqual(bindings[-1].source_order, 1_001)


class FactBindingStatementValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            "ChatGPT: mention rate 66.7% (4 of 6 answers), rank 2. "
            "Gemini: mention rate 50% (3 of 6 answers), rank 1."
        )
        self.bindings = extract_fact_bindings(
            self.source,
            child_id="child:comparison-table",
            entity_aliases=ENTITIES,
        )
        self.required_ids = [binding.binding_id for binding in self.bindings]

    def test_equivalent_decimal_spelling_validates(self) -> None:
        statement = (
            "ChatGPT: доля упоминаний 66,7% "
            "(4 из 6 ответов), 2-е место. "
            "Gemini: доля упоминаний 50,0% "
            "(3 из 6 ответов), 1-е место."
        )

        report = validate_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=self.required_ids,
        )

        self.assertTrue(report.valid)
        self.assertEqual(len(report.matches), len(self.bindings))

    def test_cross_bound_entity_values_fail_closed(self) -> None:
        statement = (
            "ChatGPT: mention rate 50% (3 of 6 answers), rank 2. "
            "Gemini: mention rate 66.7% (4 of 6 answers), rank 1."
        )

        report = audit_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=self.required_ids,
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.missing_binding_ids), 4)
        self.assertEqual(len(report.unexpected_statement_binding_ids), 4)
        with self.assertRaises(FactBindingError):
            validate_statement_bindings(
                statement,
                bindings=self.bindings,
                required_binding_ids=self.required_ids,
            )

    def test_cross_bound_metric_values_fail_closed(self) -> None:
        source_bindings = extract_fact_bindings(
            "ChatGPT: mention rate 66.7%. ChatGPT: recommendation rate 16.7%.",
            child_id="child:two-metrics",
            entity_aliases=ENTITIES,
        )

        report = audit_statement_bindings(
            "ChatGPT: mention rate 16.7%. ChatGPT: recommendation rate 66.7%.",
            bindings=source_bindings,
            required_binding_ids=[binding.binding_id for binding in source_bindings],
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.missing_binding_ids), 2)
        self.assertEqual(len(report.unexpected_statement_binding_ids), 2)

    def test_global_bag_of_words_cannot_fake_per_entity_binding(self) -> None:
        statement = (
            "ChatGPT and Gemini: mention rate values are 66.7%, 50%, "
            "4 of 6 answers, 3 of 6 answers, rank 2 and rank 1."
        )

        report = audit_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=self.required_ids,
        )

        self.assertFalse(report.valid)
        self.assertGreater(len(report.missing_binding_ids), 0)

    def test_generic_summary_that_drops_required_facts_fails(self) -> None:
        report = audit_statement_bindings(
            "ChatGPT и Gemini показывают заметную "
            "видимость бренда.",
            bindings=self.bindings,
            required_binding_ids=self.required_ids,
        )

        self.assertFalse(report.valid)
        self.assertEqual(set(report.missing_binding_ids), set(self.required_ids))

    def test_source_order_change_inside_one_child_fails(self) -> None:
        statement = (
            "Gemini: mention rate 50% (3 of 6 answers), rank 1. "
            "ChatGPT: mention rate 66.7% (4 of 6 answers), rank 2."
        )

        report = audit_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=self.required_ids,
        )

        self.assertFalse(report.valid)
        self.assertEqual(report.order_violations, ("child:comparison-table",))
        relaxed = audit_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=self.required_ids,
            enforce_child_order=False,
        )
        self.assertTrue(relaxed.valid)

    def test_sign_change_fails(self) -> None:
        source_bindings = extract_fact_bindings(
            "ChatGPT: recommendation rate increased by +12.5 percentage points.",
            child_id="child:delta",
            entity_aliases=ENTITIES,
        )

        report = audit_statement_bindings(
            "ChatGPT: recommendation rate decreased by -12.5 percentage points.",
            bindings=source_bindings,
            required_binding_ids=[source_bindings[0].binding_id],
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.missing_binding_ids), 1)

    def test_unit_change_fails(self) -> None:
        source_bindings = extract_fact_bindings(
            "ChatGPT: recommendation rate increased by +12.5 percentage points.",
            child_id="child:unit",
            entity_aliases=ENTITIES,
        )

        report = audit_statement_bindings(
            "ChatGPT: recommendation rate increased by +12.5%.",
            bindings=source_bindings,
            required_binding_ids=[source_bindings[0].binding_id],
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.unexpected_statement_binding_ids), 1)

    def test_state_change_fails(self) -> None:
        source_bindings = extract_fact_bindings(
            "Realweb: availability is unavailable.",
            child_id="child:state",
            entity_aliases=ENTITIES,
        )

        report = audit_statement_bindings(
            "Realweb: availability is available.",
            bindings=source_bindings,
            required_binding_ids=[source_bindings[0].binding_id],
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.missing_binding_ids), 1)

    def test_same_fact_from_two_children_requires_two_assertions(self) -> None:
        first = extract_fact_bindings(
            "ChatGPT: mention rate 50%.",
            child_id="child:first",
            entity_aliases=ENTITIES,
        )[0]
        second = extract_fact_bindings(
            "ChatGPT: mention rate 50%.",
            child_id="child:second",
            entity_aliases=ENTITIES,
        )[0]

        report = audit_statement_bindings(
            "ChatGPT: mention rate 50%.",
            bindings=(first, second),
            required_binding_ids=(first.binding_id, second.binding_id),
        )

        self.assertFalse(report.valid)
        self.assertEqual(len(report.matches), 1)
        self.assertEqual(len(report.missing_binding_ids), 1)

    def test_unbound_extra_typed_fact_fails_by_default(self) -> None:
        required = self.bindings[0]
        statement = "ChatGPT: mention rate 66.7%. Gemini: mention rate 90%."

        strict = audit_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=[required.binding_id],
        )
        permissive = audit_statement_bindings(
            statement,
            bindings=self.bindings,
            required_binding_ids=[required.binding_id],
            reject_unbound_facts=False,
        )

        self.assertFalse(strict.valid)
        self.assertEqual(len(strict.unexpected_statement_binding_ids), 1)
        self.assertTrue(permissive.valid)

    def test_binding_references_preserve_child_and_typed_relation(self) -> None:
        references = binding_references(self.bindings)

        self.assertEqual(len(references), len(self.bindings))
        self.assertEqual(references[0]["binding_id"], self.bindings[0].binding_id)
        self.assertEqual(references[0]["child_id"], "child:comparison-table")
        self.assertEqual(references[0]["value"], "66.7")
        self.assertEqual(references[0]["unit"], "percent")


if __name__ == "__main__":
    unittest.main()
