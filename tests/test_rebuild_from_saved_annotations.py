from __future__ import annotations

import unittest
import uuid
from unittest.mock import AsyncMock, patch

from app.services import analyzer
from scripts import rebuild_from_saved_annotations as rebuild_cli


PROFILE = {"brand_name": "Example", "brand_aliases": []}
CATALOG = {"target_aliases": ["Example"], "entities": []}
ROWS = [
    {
        "answer_id": 1,
        "mode": "web",
        "provider_key": "openai",
        "prompt_id": 1,
        "prompt_key": "u-1",
        "scenario": "Какие решения выбрать?",
        "role": "unbranded_discovery",
        "intent_class": "I",
        "status": "completed",
        "model": "test/model",
        "answer_text": "Example назван в ответе.",
        "annotation": {
            "valid": True,
            "target_mentioned": True,
        },
    }
]
METRICS = {
    "parent_discovery": {
        "web": {
            "mention_count": 1,
            "valid_answers": 1,
        }
    }
}


class SavedAnnotationRebuildGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_gate_context_drives_exact_metric_reload(self) -> None:
        class StopAfterMetricReload(RuntimeError):
            pass

        run_id = str(uuid.uuid4())
        effective_profile = {
            **PROFILE,
            "_offer_identity_policy": {
                "version": "fixture-policy-v1",
                "output_digest": "f" * 64,
            },
        }
        context = analyzer._annotation_context_manifest(
            profile=effective_profile,
            catalog=CATALOG,
            rows=ROWS,
            research_guidance="Persisted guidance, not policy reconstruction.",
            target_mention_receipts=[],
        )
        gate = {
            "passed": True,
            "policy_history": [],
            "annotation_context": context,
            "annotation_context_sha256": analyzer._stable_json_sha256(context),
            "annotation_input_sha256_by_answer_id": {"1": "a" * 64},
        }

        async def artifact(_run_id: str, key: str) -> dict:
            self.assertEqual(_run_id, run_id)
            return {
                "site_profile": PROFILE,
                "entity_catalog": CATALOG,
                "analysis_critic_gate": gate,
            }[key]

        with (
            patch.object(
                rebuild_cli,
                "_validate_saved_inputs",
                new_callable=AsyncMock,
            ),
            patch.object(
                rebuild_cli,
                "_artifact_dict",
                new=AsyncMock(side_effect=artifact),
            ),
            patch.object(
                rebuild_cli,
                "_optional_artifact_dict",
                new=AsyncMock(
                    return_value={
                        "policy_history": [],
                        "effective_profile": effective_profile,
                        "effective_catalog": CATALOG,
                        "annotation_context": context,
                        "annotation_context_sha256": (
                            analyzer._stable_json_sha256(context)
                        ),
                    }
                ),
            ),
            patch.object(
                rebuild_cli.analyzer,
                "_metric_rows",
                new=AsyncMock(
                    side_effect=[ROWS, StopAfterMetricReload]
                ),
            ) as metric_rows,
        ):
            with self.assertRaises(StopAfterMetricReload):
                await rebuild_cli.rebuild_from_saved_annotations(run_id)

        self.assertEqual(metric_rows.await_count, 2)
        self.assertEqual(
            metric_rows.await_args_list[1].kwargs,
            {
                "annotation_input_sha256": context[
                    "annotation_input_sha256"
                ],
                "annotation_input_sha256_by_answer_id": {1: "a" * 64},
            },
        )

    async def test_tampered_explicit_gate_context_is_rejected_before_reload(
        self,
    ) -> None:
        run_id = str(uuid.uuid4())
        context = analyzer._annotation_context_manifest(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            research_guidance="Persisted guidance.",
            target_mention_receipts=[],
        )
        gate = {
            "passed": True,
            "policy_history": [],
            "annotation_context": context,
            "annotation_context_sha256": "0" * 64,
        }

        async def artifact(_run_id: str, key: str) -> dict:
            return {
                "site_profile": PROFILE,
                "entity_catalog": CATALOG,
                "analysis_critic_gate": gate,
            }[key]

        with (
            patch.object(
                rebuild_cli,
                "_validate_saved_inputs",
                new_callable=AsyncMock,
            ),
            patch.object(
                rebuild_cli,
                "_artifact_dict",
                new=AsyncMock(side_effect=artifact),
            ),
            patch.object(
                rebuild_cli,
                "_optional_artifact_dict",
                new=AsyncMock(
                    return_value={
                        "policy_history": [],
                        "effective_profile": PROFILE,
                        "effective_catalog": CATALOG,
                        "annotation_context": context,
                        "annotation_context_sha256": "0" * 64,
                    }
                ),
            ),
            patch.object(
                rebuild_cli.analyzer,
                "_metric_rows",
                new=AsyncMock(return_value=ROWS),
            ) as metric_rows,
        ):
            with self.assertRaisesRegex(
                rebuild_cli.RebuildGuardError,
                "Digest annotation context",
            ):
                await rebuild_cli.rebuild_from_saved_annotations(run_id)

        self.assertEqual(metric_rows.await_count, 1)

    async def _assert_rebuild_rejects(
        self,
        *,
        gate: dict,
        rows: list[dict],
    ) -> None:
        run_id = str(uuid.uuid4())

        async def artifact(_run_id: str, key: str) -> dict:
            self.assertEqual(_run_id, run_id)
            return {
                "site_profile": PROFILE,
                "entity_catalog": CATALOG,
                "analysis_critic_gate": gate,
            }[key]

        with (
            patch.object(
                rebuild_cli,
                "_validate_saved_inputs",
                new_callable=AsyncMock,
            ),
            patch.object(
                rebuild_cli,
                "_artifact_dict",
                new=AsyncMock(side_effect=artifact),
            ),
            patch.object(
                rebuild_cli,
                "_optional_artifact_dict",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                rebuild_cli.analyzer,
                "_metric_rows",
                new=AsyncMock(return_value=rows),
            ) as metric_rows,
            patch.object(
                rebuild_cli.analyzer,
                "_compute_metrics",
                return_value=METRICS,
            ),
            patch.object(
                rebuild_cli.analyzer,
                "_technical_summary",
                new_callable=AsyncMock,
            ) as technical,
        ):
            with self.assertRaisesRegex(
                rebuild_cli.RebuildGuardError,
                "изменились после critic gate",
            ):
                await rebuild_cli.rebuild_from_saved_annotations(run_id)

        self.assertEqual(metric_rows.await_count, 2)
        self.assertEqual(
            metric_rows.await_args_list[0].kwargs,
            {
                "annotation_input_sha256": (
                    "saved-rebuild-raw-receipt-validation"
                )
            },
        )
        self.assertEqual(
            metric_rows.await_args_list[1].kwargs,
            {
                "annotation_input_sha256": (
                    analyzer._annotation_context_sha256(PROFILE, CATALOG)
                ),
                "annotation_input_sha256_by_answer_id": None,
            },
        )
        technical.assert_not_awaited()

    async def test_same_metrics_do_not_hide_changed_raw_annotation_or_scenario(
        self,
    ) -> None:
        provenance = analyzer._critic_provenance_digests(
            profile=PROFILE,
            catalog=CATALOG,
            rows=ROWS,
            metrics=METRICS,
            policy_history=[],
        )
        gate = {
            "passed": True,
            "policy_history": [],
            "metrics_sha256": provenance["metrics_sha256"],
            "provenance": provenance,
        }
        changed_cases = [
            [{**ROWS[0], "answer_text": "Raw-ответ изменился."}],
            [{**ROWS[0], "scenario": "Сценарий изменился."}],
            [
                {
                    **ROWS[0],
                    "annotation": {
                        **ROWS[0]["annotation"],
                        "target_mentioned": False,
                    },
                }
            ],
        ]

        for rows in changed_cases:
            with self.subTest(rows=rows):
                await self._assert_rebuild_rejects(gate=gate, rows=rows)

    async def test_legacy_metrics_only_gate_is_rejected(self) -> None:
        await self._assert_rebuild_rejects(
            gate={
                "passed": True,
                "policy_history": [],
                "metrics_sha256": analyzer._stable_json_sha256(METRICS),
            },
            rows=ROWS,
        )


if __name__ == "__main__":
    unittest.main()
