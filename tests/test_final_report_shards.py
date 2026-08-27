from __future__ import annotations

import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.analyzer import (
    _batch_final_report_source_records,
    _final_answer_accounting_manifest,
    _final_direct_answer_accounting_manifest,
    _final_input_claim_ledger,
    _final_input_analysis_dimension,
    _execute_final_report_adaptive_shards,
    _final_report_capsule_record,
    _final_report_shard_contracts,
    _final_report_source_claim_rows,
    _final_report_shard_source_records,
    _final_report_structured_attempt,
    _flatten_final_input_payload,
    _merge_final_report_shards,
    _minimum_final_report_shard_output_utf8_bytes,
    _prepare_final_model_payload,
)
from app.services.openrouter import (
    ChatResult,
    OpenRouterError,
    OpenRouterOutputLimitError,
    OpenRouterResponseContractError,
    OpenRouterStructuredContinuationError,
)
from app.services.sharded_document import GeneratedShard, ProviderAuditBinding


class FinalReportShardProjectionTests(unittest.TestCase):
    @staticmethod
    def _payload(observation_count: int = 2) -> dict:
        return {
            "long_input_contract": {
                "mode": "hierarchical_evidence_tree",
                "coverage_complete": True,
            },
            "evidence_digest": {
                "observations": [
                    {
                        "category": "visibility",
                        "statement": f"Наблюдение {index}",
                        "importance": "important",
                    }
                    for index in range(observation_count)
                ],
                "uncertainties": ["Оговорка"],
                "report_focus": ["Фокус"],
            },
            "deterministic_passthrough": {
                "values": [
                    {
                        "source_path": "/report_data/value",
                        "value": 17,
                    }
                ]
            },
        }

    @staticmethod
    def _value_for_source(source: dict, index: int) -> dict:
        claim_dispositions = [
            {
                "claim_id": contract["claim_id"],
                "excerpt_sha256": contract["excerpt_sha256"],
                "disposition": "material_observation",
                "evidence_excerpt": contract["excerpt"],
            }
            for contract in source["claim_contracts"]
        ]
        fact_dispositions = []
        visible_facts: list[str] = []
        for contract in source["fact_contracts"]:
            candidates = contract["assertion_candidates"]
            assertion = str(candidates[0]) if candidates else "Факт подтверждён"
            visible_facts.append(assertion)
            fact_dispositions.append(
                {
                    "fact_ref": contract["fact_ref"],
                    "disposition": "asserted",
                    "assertion": assertion,
                }
            )
        visible_claims = [
            str(contract["excerpt"]) for contract in source["claim_contracts"]
        ]
        body = "\n".join(
            [f"Текст {index}", *visible_claims, *visible_facts]
        )
        return {
            "source_shard_id": source["source_shard_id"],
            "source_sha256": source["source_sha256"],
            "core": (
                {
                    "headline": "Итог",
                    "headline_emphasis": [],
                    "verdict": "Вердикт",
                    "executive_summary": "Резюме",
                }
                if index == 0
                else None
            ),
            "section": {
                "heading": f"Раздел {index}",
                "body": body,
            },
            "actions": [
                {
                    "priority": "now",
                    "title": f"Шаг {index}",
                    "why": "Причина",
                    "step": "Сделать",
                    "evidence": "Основание",
                }
            ],
            "limitations": ["Одна оговорка"],
            "claim_dispositions": claim_dispositions,
            "fact_dispositions": fact_dispositions,
        }

    def test_projection_has_no_record_count_cap(self) -> None:
        records = _final_report_shard_source_records(
            self._payload(observation_count=2_000)
        )

        self.assertEqual(len(records), 2_004)
        self.assertEqual(records[0]["kind"], "executive_core")
        self.assertEqual(records[-1]["kind"], "exact_scalar")
        self.assertEqual(
            len({row["source_shard_id"] for row in records}),
            len(records),
        )

    def test_bounded_root_projection_preserves_every_semantic_item(self) -> None:
        payload = {
            "long_input_contract": {"mode": "bounded_transitive_evidence_tree"},
            "evidence_digest": {
                "format": "aiv-final-input-bounded-root-v4",
                "root_nodes": [
                    {
                        "source_node_id": "node-a",
                        "summary": {
                            "observations": [{"statement": "Факт A"}],
                            "uncertainties": [{"text": "Оговорка A"}],
                            "report_focus": [{"text": "Фокус A"}],
                        },
                    },
                    {
                        "source_node_id": "node-b",
                        "summary": {
                            "observations": [{"statement": "Факт B"}],
                            "uncertainties": [],
                            "report_focus": [],
                        },
                    },
                ],
            },
        }

        records = _final_report_shard_source_records(payload)

        self.assertEqual(
            [row["kind"] for row in records],
            [
                "executive_core",
                "observation",
                "uncertainty",
                "report_focus",
                "observation",
            ],
        )
        self.assertEqual(records[-1]["lineage"]["root_node_id"], "node-b")

    def test_merge_keeps_all_sections_actions_and_source_bindings(self) -> None:
        records = _final_report_shard_source_records(self._payload())
        values = [
            self._value_for_source(source, index)
            for index, source in enumerate(records)
        ]

        merged = _merge_final_report_shards(
            tuple(values),
            source_records=records,
        )

        self.assertEqual(len(merged["sections"]), len(records))
        self.assertEqual(len(merged["actions"]), len(records))
        self.assertEqual(merged["limitations"], ["Одна оговорка"])

        tampered = copy.deepcopy(values)
        tampered[1]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(OpenRouterError, "source binding"):
            _merge_final_report_shards(
                tuple(tampered),
                source_records=records,
            )

    def test_exact_qualitative_claim_requires_visible_tail_marker(self) -> None:
        units, _manifest = _flatten_final_input_payload(
            {
                "qualitative": (
                    "Клиент объясняет выбор через доверие к экспертам и "
                    "проверку результата, а не через цену. TAIL_MARKER_9XZ"
                )
            },
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(units)
        records = _final_report_shard_source_records(
            self._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        exact_index = next(
            index
            for index, record in enumerate(records)
            if record["kind"] == "exact_claim"
        )
        values = [
            self._value_for_source(source, index)
            for index, source in enumerate(records)
        ]

        accepted = _merge_final_report_shards(
            tuple(values),
            source_records=records,
        )
        self.assertIn(
            "TAIL_MARKER_9XZ",
            accepted["sections"][exact_index]["body"],
        )

        lost_tail = copy.deepcopy(values)
        lost_tail[exact_index]["section"]["body"] = (
            "Клиент объясняет выбор через доверие к экспертам."
        )
        with self.assertRaisesRegex(
            OpenRouterError,
            "material qualitative anchors",
        ):
            _merge_final_report_shards(
                tuple(lost_tail),
                source_records=records,
            )

    def test_generic_unknown_cannot_hide_unique_claim_meaning(self) -> None:
        unique = (
            "CRITICAL_UNIQUE_FINDING_7QZ market access is limited and the "
            "measurement state is unknown"
        )
        units, _manifest = _flatten_final_input_payload(
            {"qualitative": unique},
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(units)
        records = _final_report_shard_source_records(
            self._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        exact_index = next(
            index
            for index, record in enumerate(records)
            if record["kind"] == "exact_claim"
        )
        contract = records[exact_index]["claim_contracts"][0]
        self.assertTrue(contract["visibility_clauses"])
        self.assertIn(
            "CRITICAL_UNIQUE_FINDING_7QZ",
            contract["visibility_clauses"][0]["required_anchors"],
        )
        values = [
            self._value_for_source(source, index)
            for index, source in enumerate(records)
        ]
        values[exact_index]["section"]["body"] = (
            "Статус остаётся unknown."
        )

        with self.assertRaisesRegex(
            OpenRouterError,
            "material qualitative anchors",
        ):
            _merge_final_report_shards(
                tuple(values),
                source_records=records,
            )

    def test_output_floor_repartitions_before_provider_post(self) -> None:
        payload = {
            "qualitative": (
                ("FIRST_MATERIAL_ANCHOR_1Q evidence context. " * 130)
                + ("SECOND_MATERIAL_ANCHOR_2Q evidence context. " * 130)
            )
        }
        units, _manifest = _flatten_final_input_payload(
            payload,
            target_chars=32_000,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(units)
        atomic = _final_report_shard_source_records(
            self._payload(observation_count=0),
            claim_rows=claim_rows,
        )

        def request_bytes(record: dict) -> int:
            return 8_000 + len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        exact_records = [
            record for record in atomic if record["kind"] == "exact_claim"
        ]
        self.assertGreaterEqual(len(exact_records), 2)
        one_record_floor = max(
            _minimum_final_report_shard_output_utf8_bytes(record)
            for record in exact_records
        )
        output_window = one_record_floor + 256
        capsules, plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=80_000,
            output_window_utf8_bytes=output_window,
            request_utf8_bytes=request_bytes,
        )

        claim_capsules = [
            capsule
            for capsule in capsules
            if capsule.get("kind") == "domain_capsule"
            and capsule.get("claim_contracts")
        ]
        self.assertGreaterEqual(len(claim_capsules), 2)
        self.assertTrue(
            all(
                _minimum_final_report_shard_output_utf8_bytes(capsule)
                <= output_window
                for capsule in capsules
            )
        )
        self.assertEqual(
            plan["packing_rule"],
            "exact_physical_input_and_minimum_output_envelopes",
        )

    def test_numeric_corpus_larger_than_context_stays_per_claim_bounded(
        self,
    ) -> None:
        source = {
            "metrics": [
                (
                    f"metric-{index}: {index}% из {index + 1}; "
                    + "контрольное числовое свидетельство " * 18
                )
                for index in range(600)
            ]
        }
        units, _manifest = _flatten_final_input_payload(
            source,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(units)
        records = _final_report_shard_source_records(
            self._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        exact_records = [
            record for record in records if record["kind"] == "exact_claim"
        ]
        fact_records = [
            record for record in records if record["kind"] == "exact_fact"
        ]
        encoded_sizes = [
            len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for record in records
        ]

        self.assertEqual(len(exact_records), len(claim_rows))
        self.assertGreater(sum(encoded_sizes), 192_000)
        self.assertLess(max(encoded_sizes), 64_000)
        self.assertNotIn(
            "mandatory_fact_table",
            json.dumps(records[0], ensure_ascii=False),
        )
        self.assertTrue(
            all(len(record["claim_contracts"]) == 1 for record in exact_records)
        )
        self.assertTrue(
            all(
                "mandatory_fact_table" not in record["value"]
                and not record["fact_contracts"]
                for record in exact_records
            )
        )
        self.assertTrue(
            all(
                len(record["claim_contracts"]) == 0
                and len(record["fact_contracts"]) == 1
                and len(
                    record["value"]["mandatory_fact_table"]["rows"]
                )
                == 1
                for record in fact_records
            )
        )
        fact_refs = [
            contract["fact_ref"]
            for record in fact_records
            for contract in record["fact_contracts"]
        ]
        self.assertEqual(len(fact_refs), len(set(fact_refs)))

    def test_thousand_atomic_facts_become_bounded_domain_capsules(self) -> None:
        raw = "\n".join(
            f"Метрика {index}: {index}% из {index + 1}."
            for index in range(1_000)
        ) + "\nRARE_TAIL_UNKNOWN_71Z: unknown"
        source_payload = {
            "selected_full_answers": [
                {
                    "answer_id": 7,
                    "prompt_id": 3,
                    "prompt_key": "u-e",
                    "scenario": "Какие варианты стоит сравнить?",
                    "scenario_role": "unbranded_discovery",
                    "intent_class": "E",
                    "provider_key": "provider-a",
                    "model": "model-a",
                    "mode": "web",
                    "requested_mode": "web",
                    "verified_mode": "web",
                    "context_access": "full_text",
                    "metric_eligible": True,
                    "context_eligible": True,
                    "metric_evidence_state": "attested",
                    "citations": [],
                    "annotation": {
                        "valid": True,
                        "target_mentioned": True,
                        "target_role": "recommended",
                        "sentiment": "positive",
                    },
                    "answer_text": raw,
                    "provenance": {
                        "raw_answer_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest()
                    },
                }
            ]
        }
        units, _manifest = _flatten_final_input_payload(
            source_payload,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=source_payload,
        )
        atomic = _final_report_shard_source_records(
            self._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        input_window = 80_000
        output_window = 32_000

        def request_bytes(record: dict) -> int:
            return 9_000 + len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        capsules, plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=input_window,
            output_window_utf8_bytes=output_window,
            request_utf8_bytes=request_bytes,
        )

        self.assertTrue(plan["coverage_complete"])
        self.assertIsNone(plan["local_record_or_capsule_count_cap"])
        self.assertLess(len(capsules), len(atomic) // 4)
        self.assertTrue(
            any(
                len(capsule["lineage"].get("member_receipts") or []) > 10
                for capsule in capsules[1:]
            )
        )
        self.assertTrue(
            all(request_bytes(capsule) <= input_window for capsule in capsules)
        )
        self.assertTrue(
            all(
                _minimum_final_report_shard_output_utf8_bytes(capsule)
                <= output_window
                for capsule in capsules
            )
        )
        atomic_receipts = {
            (item["source_shard_id"], item["source_sha256"])
            for item in atomic[1:]
        }
        capsule_receipts = [
            (item["source_shard_id"], item["source_sha256"])
            for capsule in capsules[1:]
            for item in capsule["lineage"]["member_receipts"]
        ]
        self.assertEqual(set(capsule_receipts), atomic_receipts)
        self.assertEqual(len(capsule_receipts), len(atomic_receipts))
        values = [
            self._value_for_source(source, index)
            for index, source in enumerate(capsules)
        ]
        merged = _merge_final_report_shards(
            tuple(values),
            source_records=capsules,
        )
        self.assertEqual(len(merged["sections"]), len(capsules))
        self.assertTrue(
            any(
                "RARE_TAIL_UNKNOWN_71Z" in section["body"]
                for section in merged["sections"]
            )
        )

    def test_report_array_indexes_do_not_create_singleton_capsules(self) -> None:
        source_payload = {
            "report_data": {
                "metrics": [
                    {
                        "label": f"Метрика {index}",
                        "value": index,
                        "state": (
                            "unknown" if index == 119 else "measured"
                        ),
                    }
                    for index in range(120)
                ]
            }
        }
        units, _manifest = _flatten_final_input_payload(
            source_payload,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=source_payload,
        )
        report_claims = [
            row
            for row in claim_rows
            if row["source_path"].startswith("/report_data/metrics/")
        ]
        self.assertEqual(
            {row["domain_context"]["group_path"] for row in report_claims},
            {"/report_data/metrics"},
        )
        atomic = _final_report_shard_source_records(
            self._payload(observation_count=0),
            claim_rows=claim_rows,
        )

        def request_bytes(record: dict) -> int:
            return 8_000 + len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        capsules, _plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=80_000,
            output_window_utf8_bytes=32_000,
            request_utf8_bytes=request_bytes,
        )
        self.assertLess(len(capsules), len(atomic) // 3)
        self.assertTrue(
            any(
                len(capsule["lineage"].get("member_receipts") or []) > 20
                for capsule in capsules[1:]
            )
        )


class FinalAnswerAccountingTests(unittest.TestCase):
    @staticmethod
    def _answer(index: int, raw: str) -> dict:
        return {
            "answer_id": index + 1,
            "prompt_id": index + 101,
            "prompt_key": f"intent-{index}",
            "scenario": f"Как выбрать решение для задачи {index}?",
            "scenario_role": "unbranded_discovery",
            "intent_class": ("I", "E", "T", "NB", "NAV", "TR")[
                index % 6
            ],
            "provider_key": f"provider-{index % 5}",
            "model": f"model-{index % 5}",
            "mode": "web" if index % 2 == 0 else "memory",
            "requested_mode": "web" if index % 2 == 0 else "memory",
            "verified_mode": "web" if index % 2 == 0 else "memory",
            "context_access": "full_text",
            "metric_eligible": True,
            "context_eligible": True,
            "metric_evidence_state": "attested",
            "citations": [f"https://example.test/{index}"],
            "annotation": {
                "valid": True,
                "target_mentioned": index % 3 == 0,
                "target_role": "recommended" if index % 4 == 0 else "listed",
                "sentiment": "positive",
            },
            "answer_text": raw,
            "provenance": {
                "raw_answer_sha256": hashlib.sha256(
                    raw.encode("utf-8")
                ).hexdigest()
            },
        }

    def test_direct_route_has_content_addressed_exact_once_receipt(self) -> None:
        answers = [
            self._answer(
                index,
                f"Полный ответ {index}; RARE_DIRECT_{index}_UNKNOWN",
            )
            for index in range(12)
        ]
        payload = {
            "report_data": {"state": "ready"},
            "selected_full_answers": answers,
        }

        accounting = _final_direct_answer_accounting_manifest(payload)

        self.assertEqual(accounting["route"], "direct_single_provider_post")
        self.assertEqual(accounting["answer_count"], 12)
        self.assertEqual(accounting["accounted_answer_count"], 12)
        self.assertEqual(accounting["raw_text_answer_count"], 12)
        self.assertEqual(
            accounting["request_payload_occurrences_per_raw_answer"],
            1,
        )
        self.assertEqual(
            len({row["row_sha256"] for row in accounting["answer_rows"]}),
            12,
        )
        self.assertEqual(
            accounting["answer_rows"][-1]["raw_answer_sha256"],
            hashlib.sha256(
                answers[-1]["answer_text"].encode("utf-8")
            ).hexdigest(),
        )


    def test_many_long_answers_have_exact_once_domain_accounting(self) -> None:
        answers = [
            self._answer(
                index,
                (f"Ответ {index}. " * 1_800)
                + f"RARE_TAIL_{index}_UNKNOWN unknown",
            )
            for index in range(40)
        ]
        payload = {
            "report_data": {"quality": {"state": "limited"}},
            "selected_full_answers": answers,
        }
        units, _manifest = _flatten_final_input_payload(
            payload,
            target_chars=8_192,
            context_overlap_chars=256,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=payload,
        )
        accounting = _final_answer_accounting_manifest(payload, claim_rows)

        self.assertEqual(accounting["answer_count"], 40)
        self.assertEqual(accounting["accounted_answer_count"], 40)
        self.assertEqual(accounting["raw_text_answer_count"], 40)
        self.assertTrue(accounting["raw_reconstruction_complete"])
        self.assertTrue(accounting["claim_assignment_complete"])
        self.assertEqual(
            len(
                {
                    row["domain_context_id"]
                    for row in accounting["answer_rows"]
                }
            ),
            40,
        )
        for index, answer in enumerate(answers):
            raw_claims = sorted(
                (
                    row
                    for row in claim_rows
                    if row["source_path"]
                    == f"/selected_full_answers/{index}/answer_text"
                ),
                key=lambda row: (
                    row["source_core_start_char"],
                    row["source_utf8_offset"],
                ),
            )
            self.assertEqual(
                "".join(row["excerpt"] for row in raw_claims),
                answer["answer_text"],
            )
            self.assertIn(
                f"RARE_TAIL_{index}_UNKNOWN",
                raw_claims[-1]["excerpt"],
            )

    def test_prompt_and_answer_share_join_key_but_keep_dimensions(self) -> None:
        raw = "Бренд рекомендован для выбора. TAIL_CONTEXT_X9"
        payload = {"selected_full_answers": [self._answer(0, raw)]}
        units, _manifest = _flatten_final_input_payload(
            payload,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=payload,
        )
        prompt_claim = next(
            row
            for row in claim_rows
            if row["source_path"].endswith("/scenario")
        )
        answer_claim = next(
            row
            for row in claim_rows
            if row["source_path"].endswith("/answer_text")
        )
        self.assertEqual(
            prompt_claim["domain_context_id"],
            answer_claim["domain_context_id"],
        )
        self.assertEqual(prompt_claim["analysis_dimension"], "prompt_context")
        self.assertEqual(answer_claim["analysis_dimension"], "answer_response")
        self.assertEqual(
            _final_input_analysis_dimension(
                "/report_data/discovery/portfolio_visibility/web"
            ),
            "portfolio_visibility",
        )
        atomic = _final_report_shard_source_records(
            FinalReportShardProjectionTests._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        capsules, _plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=80_000,
            output_window_utf8_bytes=32_000,
            request_utf8_bytes=lambda record: 8_000
            + len(json.dumps(record, ensure_ascii=False).encode("utf-8")),
        )
        answer_capsule = next(
            capsule
            for capsule in capsules
            if capsule.get("kind") == "domain_capsule"
            and capsule["value"].get("analysis_dimension")
            == "model_answer_bundle"
        )
        self.assertIn(
            "prompt_context", answer_capsule["value"]["analysis_dimensions"]
        )
        self.assertIn(
            "answer_response", answer_capsule["value"]["analysis_dimensions"]
        )
        member_paths = [
            member["value"].get("source_path")
            for member in answer_capsule["value"]["members"]
            if member.get("kind") == "exact_claim"
        ]
        self.assertLess(
            member_paths.index("/selected_full_answers/0/scenario"),
            member_paths.index("/selected_full_answers/0/answer_text"),
        )

    def test_every_long_answer_capsule_gets_exact_prompt_join_context(
        self,
    ) -> None:
        raw = (
            "LONG_RESPONSE_EVIDENCE market and entity context. " * 800
        ) + "RARE_FINAL_9QZ unknown"
        answer = self._answer(0, raw)
        payload = {"selected_full_answers": [answer]}
        units, _manifest = _flatten_final_input_payload(
            payload,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=payload,
        )
        atomic = _final_report_shard_source_records(
            FinalReportShardProjectionTests._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        capsules, _plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=50_000,
            output_window_utf8_bytes=20_000,
            request_utf8_bytes=lambda record: 8_000
            + len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        )
        answer_capsules = [
            capsule
            for capsule in capsules
            if capsule.get("kind") == "domain_capsule"
            and any(
                str(
                    (
                        member.get("value")
                        if isinstance(member.get("value"), dict)
                        else {}
                    ).get("source_path")
                    or ""
                ).endswith("/answer_text")
                for member in capsule["value"]["members"]
            )
        ]
        self.assertGreater(len(answer_capsules), 2)
        for capsule in answer_capsules:
            join = capsule["value"].get("prompt_join_context")
            self.assertIsInstance(join, dict)
            self.assertEqual(
                join["contract"],
                "exact_non_disposition_join_context",
            )
            self.assertEqual(
                "".join(
                    fragment["excerpt"]
                    for fragment in join["selected_fragments"]
                ),
                answer["scenario"],
            )
            self.assertTrue(join["full_prompt_included"])

        prompt_member_receipts = [
            receipt
            for capsule in capsules[1:]
            for receipt in capsule["lineage"]["member_receipts"]
            if receipt["source_shard_id"]
            in {
                record["source_shard_id"]
                for record in atomic
                if record.get("kind") == "exact_claim"
                and str(record["value"].get("source_path") or "").endswith(
                    "/scenario"
                )
            }
        ]
        self.assertEqual(len(prompt_member_receipts), 1)

    def test_physically_long_prompt_uses_exact_endpoint_fragment_ledger(
        self,
    ) -> None:
        answer = self._answer(0, "Краткий ответ с проверяемым выводом.")
        answer["scenario"] = (
            "PROMPT_OPENING_3Q условия выбора и критерии. " * 750
        ) + ("PROMPT_FINAL_TASK_8Z сформулируй рекомендацию. " * 750)
        payload = {"selected_full_answers": [answer]}
        units, _manifest = _flatten_final_input_payload(
            payload,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=payload,
        )
        atomic = _final_report_shard_source_records(
            FinalReportShardProjectionTests._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        capsules, _plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=50_000,
            output_window_utf8_bytes=20_000,
            request_utf8_bytes=lambda record: 8_000
            + len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        )
        answer_capsule = next(
            capsule
            for capsule in capsules
            if capsule.get("kind") == "domain_capsule"
            and any(
                isinstance(member.get("value"), dict)
                and str(
                    member["value"].get("source_path") or ""
                ).endswith("/answer_text")
                for member in capsule["value"]["members"]
            )
        )
        join = answer_capsule["value"]["prompt_join_context"]
        self.assertFalse(join["full_prompt_included"])
        self.assertIn(join["selected_fragment_count"], {1, 2})
        self.assertGreater(join["prompt_fragment_count"], 2)
        selected_text = "\n".join(
            fragment["excerpt"]
            for fragment in join["selected_fragments"]
        )
        self.assertIn("PROMPT_FINAL_TASK_8Z", selected_text)
        if join["selected_fragment_count"] == 2:
            self.assertIn("PROMPT_OPENING_3Q", selected_text)
        self.assertRegex(join["all_prompt_receipts_sha256"], r"^[0-9a-f]{64}$")

    def test_middle_long_prompt_product_clause_is_joined_to_answer(self) -> None:
        answer = self._answer(
            0,
            "Ответ связывает LateService с подтверждённым предложением клиента.",
        )
        answer["scenario"] = (
            "PROMPT_OPEN_ONLY критерии выбора. "
            + ("a" * 6_000)
            + " MIDDLE_PRODUCT_SENTINEL_77 услуга LateService; источник "
            "https://client.example/services/late; проверь связь с ответом. "
            + ("b" * 6_000)
            + " FINAL_TASK_SENTINEL_88 сформулируй вывод. "
            + ("c" * 6_000)
        )
        payload = {"selected_full_answers": [answer]}
        units, _manifest = _flatten_final_input_payload(
            payload,
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, _ledger = _final_input_claim_ledger(
            units,
            source_payload=payload,
        )
        atomic = _final_report_shard_source_records(
            FinalReportShardProjectionTests._payload(observation_count=0),
            claim_rows=claim_rows,
        )
        capsules, plan = _batch_final_report_source_records(
            atomic,
            input_window_utf8_bytes=50_000,
            output_window_utf8_bytes=20_000,
            request_utf8_bytes=lambda record: 8_000
            + len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        )

        middle_capsule = next(
            capsule
            for capsule in capsules
            if capsule.get("kind") == "domain_capsule"
            and any(
                "MIDDLE_PRODUCT_SENTINEL_77"
                in str(
                    (
                        member.get("value")
                        if isinstance(member.get("value"), dict)
                        else {}
                    ).get("excerpt")
                    or ""
                )
                for member in capsule["value"]["members"]
                if isinstance(member, dict)
            )
        )
        answer_join = middle_capsule["value"].get("answer_join_context")
        self.assertIsInstance(answer_join, dict)
        joined_answer = "".join(
            fragment["excerpt"]
            for fragment in answer_join["selected_fragments"]
        )
        self.assertIn("LateService", joined_answer)
        middle_prompt = "".join(
            str(
                (
                    member.get("value")
                    if isinstance(member.get("value"), dict)
                    else {}
                ).get("excerpt")
                or ""
            )
            for member in middle_capsule["value"]["members"]
        )
        self.assertIn("https://client.example/services/late", middle_prompt)
        self.assertTrue(plan["prompt_answer_join_coverage_complete"])
        self.assertEqual(plan["joined_model_answer_bundle_count"], 1)
        self.assertGreater(plan["joined_prompt_fragment_count"], 2)
        self.assertGreaterEqual(plan["joined_answer_fragment_count"], 1)


class FinalDirectPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_preflight_persists_answer_accounting_receipt(
        self,
    ) -> None:
        raw = "Прямой полный ответ; DIRECT_TAIL_UNKNOWN"
        payload = {
            "selected_full_answers": [
                FinalAnswerAccountingTests._answer(0, raw)
            ]
        }
        save = AsyncMock()
        with (
            patch(
                "app.services.analyzer._final_model_input_window",
                new=AsyncMock(
                    return_value={
                        "input_utf8_window": 20_000,
                        "model_envelope": {},
                        "version": "test",
                    }
                ),
            ),
            patch("app.services.analyzer._save_artifact", new=save),
        ):
            prepared, plan = await _prepare_final_model_payload(
                "run-direct",
                payload=payload,
                system="author",
                final_request_utf8_bytes=lambda _payload, _envelope: 1_000,
            )

        self.assertIs(prepared, payload)
        self.assertEqual(plan["mode"], "direct")
        self.assertEqual(plan["answer_accounting"]["answer_count"], 1)
        self.assertEqual(
            plan["answer_accounting"][
                "request_payload_occurrences_per_raw_answer"
            ],
            1,
        )
        save.assert_awaited_once()
        saved_manifest = save.await_args.kwargs["output_json"]
        self.assertEqual(
            saved_manifest["answer_rows"][0]["raw_answer_sha256"],
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )


class FinalReportClaimLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_only_the_exact_content_addressed_claim_ledger(self) -> None:
        units, _manifest = _flatten_final_input_payload(
            {"qualitative": "Полный исходный текст TAIL_MARKER_EXACT"},
            target_chars=4_096,
            context_overlap_chars=0,
        )
        claim_rows, _claims, _ids, ledger = _final_input_claim_ledger(units)
        payload = {
            "long_input_contract": {
                "mode": "hierarchical_evidence_tree",
                "claim_ledger": {
                    "artifact_key": "claims-1",
                    "sha256": ledger["ledger_sha256"],
                    "claim_ids_sha256": ledger["claim_ids_sha256"],
                },
            }
        }
        with patch(
            "app.services.analyzer._artifact_output",
            new=AsyncMock(return_value=ledger),
        ):
            loaded = await _final_report_source_claim_rows("run-1", payload)
        self.assertEqual(loaded, claim_rows)

        tampered = copy.deepcopy(ledger)
        tampered["claims"][0]["excerpt"] += " подмена"
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=tampered),
            ),
            self.assertRaisesRegex(OpenRouterError, "integrity"),
        ):
            await _final_report_source_claim_rows("run-1", payload)


class FinalReportShardRoutingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _provider_result() -> SimpleNamespace:
        return SimpleNamespace(text="{\"partial\":", usage={})

    async def test_physical_continuation_failure_routes_to_shards(self) -> None:
        failure = OpenRouterStructuredContinuationError(
            "whole document no longer fits",
            result=self._provider_result(),
            manifest={"complete": False},
        )
        fallback = SimpleNamespace(
            parsed={"headline": "ok"},
            text="{}",
            usage={"_aiv_sharded_document": {"complete": True}},
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(),
            ) as save,
            patch(
                "app.services.analyzer._durable_structured_transport",
                new=AsyncMock(return_value=(AsyncMock(), None)),
            ),
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new=AsyncMock(side_effect=failure),
            ),
            patch(
                "app.services.analyzer._final_report_sharded_attempt",
                new=AsyncMock(return_value=fallback),
            ) as sharded,
        ):
            result = await _final_report_structured_attempt(
                "run-1",
                system="system",
                user_payload={"evidence_digest": {}},
                artifact_role="author",
                attempt=0,
            )

        self.assertIs(result, fallback)
        sharded.assert_awaited_once()
        failed_saves = [
            call
            for call in save.await_args_list
            if call.kwargs.get("status") == "failed"
        ]
        self.assertEqual(len(failed_saves), 1)
        self.assertEqual(failed_saves[0].kwargs["raw_text"], '{"partial":')

    async def test_prepared_hierarchical_input_routes_to_exact_claim_shards(
        self,
    ) -> None:
        fallback = SimpleNamespace(
            parsed={"headline": "ok"},
            text="{}",
            usage={"_aiv_sharded_document": {"complete": True}},
        )
        payload = {
            "long_input_contract": {
                "mode": "bounded_transitive_evidence_tree",
                "claim_ledger": {
                    "artifact_key": "claims",
                    "sha256": "1" * 64,
                    "claim_ids_sha256": "2" * 64,
                },
            },
            "evidence_digest": {"root_nodes": []},
        }
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(),
            ) as save,
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new=AsyncMock(),
            ) as monolith,
            patch(
                "app.services.analyzer._final_report_sharded_attempt",
                new=AsyncMock(return_value=fallback),
            ) as sharded,
        ):
            result = await _final_report_structured_attempt(
                "run-1",
                system="system",
                user_payload=payload,
                artifact_role="author",
                attempt=0,
            )

        self.assertIs(result, fallback)
        sharded.assert_awaited_once()
        monolith.assert_not_awaited()
        save.assert_not_awaited()

    async def test_non_continuation_contract_error_fails_closed(self) -> None:
        failure = OpenRouterResponseContractError(
            "schema mismatch",
            result=self._provider_result(),
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(),
            ),
            patch(
                "app.services.analyzer._durable_structured_transport",
                new=AsyncMock(return_value=(AsyncMock(), None)),
            ),
            patch(
                "app.services.analyzer.chat_continuable_structured",
                new=AsyncMock(side_effect=failure),
            ),
            patch(
                "app.services.analyzer._final_report_sharded_attempt",
                new=AsyncMock(),
            ) as sharded,
        ):
            with self.assertRaises(OpenRouterResponseContractError):
                await _final_report_structured_attempt(
                    "run-1",
                    system="system",
                    user_payload={"evidence_digest": {}},
                    artifact_role="author",
                    attempt=0,
                )

        sharded.assert_not_awaited()


class FinalAdaptiveShardExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_output_limit_splits_leaf_and_resumes_prefix(self) -> None:
        base = _final_report_shard_source_records(
            FinalReportShardProjectionTests._payload(observation_count=0)
        )[0]
        members = []
        for ordinal in range(2):
            value = {"statement": f"Материальный вывод {ordinal}"}
            digest = hashlib.sha256(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            members.append(
                {
                    "kind": "observation",
                    "ordinal": ordinal,
                    "value": value,
                    "lineage": {"domain_context_id": "shared-context"},
                    "claim_contracts": [],
                    "fact_contracts": [],
                    "source_shard_id": f"observation-{ordinal}-{digest[:16]}",
                    "source_sha256": digest,
                }
            )
        capsule = _final_report_capsule_record(members, ordinal=1)
        source_records = [base, capsule]
        shard_system = "Напиши проверяемый фрагмент."
        generation_contract, merge_contract = _final_report_shard_contracts(
            shard_system=shard_system
        )
        generated_sources: list[str] = []

        def provider_binding(raw: str, request_sha256: str) -> ProviderAuditBinding:
            raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            return ProviderAuditBinding(
                event_id="event",
                receipt_ref="receipt",
                receipt_sha256="a" * 64,
                physical_request_sha256="b" * 64,
                logical_request_sha256=request_sha256,
                raw_text_sha256=raw_sha,
            )

        class FakeStore:
            def __init__(self, plan: object) -> None:
                self.plan = plan
                self.source = plan.shards[0].payload
                self.request = SimpleNamespace(request_sha256="c" * 64)

            def planned_requests(self) -> tuple[SimpleNamespace, ...]:
                return (self.request,)

            def provider_request_utf8_bytes(
                self,
                _request: object,
                *,
                max_completion_tokens: int | None,
            ) -> bytes:
                self.assert_max = max_completion_tokens
                return b"{}"

            async def promote_accepted_provider_response(
                self, _request: object
            ) -> GeneratedShard | None:
                if self.source["kind"] != "executive_core":
                    return None
                value = FinalReportShardProjectionTests._value_for_source(
                    self.source,
                    int(self.source["ordinal"]),
                )
                raw = json.dumps(value, ensure_ascii=False)
                return GeneratedShard(
                    value=value,
                    raw_text=raw,
                    provider_audit=provider_binding(raw, "c" * 64),
                )

            def provider_chat_arguments(self, _request: object) -> dict:
                return {}

            async def generate_or_resume(
                self, _request: object, _provider_call: object
            ) -> GeneratedShard:
                generated_sources.append(self.source["source_shard_id"])
                member_count = len(
                    (self.source.get("value") or {}).get("members") or []
                )
                if member_count > 1:
                    partial = ChatResult(
                        text='{"source_shard_id":"partial"}',
                        parsed=None,
                        citations=[],
                        usage={},
                        annotations=[],
                        request_policy={},
                        web_attestation={},
                        router_metadata={},
                        transport={
                            "output_limited": True,
                            "output_complete": False,
                        },
                    )
                    raise OpenRouterOutputLimitError(
                        "physical output limit",
                        result=partial,
                    )
                value = FinalReportShardProjectionTests._value_for_source(
                    self.source,
                    int(self.source["ordinal"]),
                )
                raw = json.dumps(value, ensure_ascii=False)
                return GeneratedShard(
                    value=value,
                    raw_text=raw,
                    provider_audit=provider_binding(raw, "c" * 64),
                )

            async def verify_provider_audit(
                self, _binding: object, _request: object
            ) -> None:
                return None

        with patch(
            "app.services.analyzer.create_sharded_artifact_store",
            side_effect=lambda **kwargs: FakeStore(kwargs["plan"]),
        ):
            document, terminal, manifest = (
                await _execute_final_report_adaptive_shards(
                    "run-adaptive",
                    source_records=source_records,
                    system=shard_system,
                    generation_contract=generation_contract,
                    merge_contract=merge_contract,
                    artifact_role="author",
                    attempt=0,
                    source_digest="d" * 64,
                    input_window_utf8_bytes=100_000,
                    max_completion_tokens=32_000,
                )
            )

        self.assertEqual(len(terminal), 3)
        self.assertEqual(len(document["sections"]), 3)
        self.assertTrue(manifest["complete"])
        self.assertTrue(manifest["coverage_complete"])
        self.assertEqual(manifest["replan_event_count"], 1)
        self.assertEqual(
            manifest["replan_events"][0]["strategy"],
            "split_domain_capsule_members",
        )
        self.assertEqual(manifest["resumed_shards"], 1)
        self.assertEqual(manifest["generated_shards"], 2)
        self.assertEqual(len(generated_sources), 3)
        self.assertFalse(
            any(source.startswith("executive_core") for source in generated_sources)
        )


if __name__ == "__main__":
    unittest.main()
