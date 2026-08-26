from __future__ import annotations

import unittest

from app.services.panel_coverage import (
    PanelMetricCoverageError,
    build_panel_metric_coverage_admission,
    require_panel_metric_coverage,
)


WEB_PROVIDERS = ("openai", "gemini", "perplexity", "deepseek", "claude")
MEMORY_PROVIDERS = ("openai", "gemini", "deepseek", "claude")


def _expected_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for prompt_id in range(1, 10):
        for mode, providers in (
            ("web", WEB_PROVIDERS),
            ("memory", MEMORY_PROVIDERS),
        ):
            for provider in providers:
                cells.append(
                    {
                        "prompt_id": prompt_id,
                        "provider_key": provider,
                        "mode": mode,
                        "model": f"{provider}/model",
                    }
                )
    return cells


def _observed_cells() -> list[dict[str, object]]:
    return [
        {
            **cell,
            "status": "completed",
            "metric_eligible": True,
            "metric_evidence_state": "strict_verified",
            "metric_limitation": None,
        }
        for cell in _expected_cells()
    ]


class PanelMetricCoverageAdmissionTests(unittest.TestCase):
    def test_complete_81_cell_panel_is_admitted(self) -> None:
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=_observed_cells(),
        )

        self.assertTrue(admission["allowed"])
        self.assertEqual(admission["quality_state"], "complete")
        self.assertEqual(admission["reason_codes"], [])
        self.assertEqual(admission["warning_codes"], [])
        self.assertEqual(admission["expected_cell_count"], 81)
        self.assertEqual(admission["eligible_cell_count"], 81)
        self.assertEqual(len(admission["expected_cell_manifest_sha256"]), 64)
        require_panel_metric_coverage(admission)

    def test_expected_grid_digest_binds_model_topology(self) -> None:
        expected = _expected_cells()
        original = build_panel_metric_coverage_admission(
            expected_cells=expected,
            observed_rows=_observed_cells(),
        )
        changed = [dict(cell) for cell in expected]
        changed[0]["model"] = "openai/replacement-model"
        rebound = build_panel_metric_coverage_admission(
            expected_cells=changed,
            observed_rows=_observed_cells(),
        )

        self.assertNotEqual(
            original["expected_cell_manifest_sha256"],
            rebound["expected_cell_manifest_sha256"],
        )
        self.assertFalse(rebound["allowed"])
        self.assertEqual(rebound["quality_state"], "blocked")
        self.assertIn("panel_model_grid_mismatch", rebound["reason_codes"])

    def test_wholly_unavailable_provider_is_explicit_but_not_partial_noise(
        self,
    ) -> None:
        rows = _observed_cells()
        for row in rows:
            if row["mode"] == "web" and row["provider_key"] == "perplexity":
                row["status"] = "failed"
                row["metric_eligible"] = False
                row["metric_evidence_state"] = "excluded"
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertTrue(admission["allowed"])
        self.assertEqual(admission["quality_state"], "degraded")
        self.assertIn(
            "web_perplexity_provider_unavailable",
            admission["warning_codes"],
        )
        self.assertEqual(
            admission["unavailable_provider_count_by_mode"]["web"],
            1,
        )

    def test_tiny_partial_provider_slice_is_rejected(self) -> None:
        rows = _observed_cells()
        for row in rows:
            if (
                row["mode"] == "web"
                and row["provider_key"] == "gemini"
                and row["prompt_id"] > 2
            ):
                row["metric_eligible"] = False
                row["metric_evidence_state"] = "provider_limited_prefix"
                row["metric_limitation"] = "provider_output_limit"
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertFalse(admission["allowed"])
        self.assertIn(
            "web_gemini_partial_slice_below_minimum",
            admission["reason_codes"],
        )
        with self.assertRaises(PanelMetricCoverageError):
            require_panel_metric_coverage(admission)

    def test_one_to_four_wholly_unavailable_web_providers_are_degraded(
        self,
    ) -> None:
        for unavailable_count in range(1, 5):
            with self.subTest(unavailable_count=unavailable_count):
                unavailable = set(WEB_PROVIDERS[:unavailable_count])
                rows = _observed_cells()
                for row in rows:
                    if row["mode"] == "web" and row["provider_key"] in unavailable:
                        row["metric_eligible"] = False
                        row["metric_evidence_state"] = "excluded"
                admission = build_panel_metric_coverage_admission(
                    expected_cells=_expected_cells(),
                    observed_rows=rows,
                )

                self.assertTrue(admission["allowed"])
                self.assertEqual(admission["quality_state"], "degraded")
                self.assertEqual(
                    admission["unavailable_provider_count_by_mode"]["web"],
                    unavailable_count,
                )
                require_panel_metric_coverage(admission)

    def test_too_many_provider_outages_are_warnings(self) -> None:
        rows = _observed_cells()
        for row in rows:
            if row["mode"] == "web" and row["provider_key"] in {
                "openai",
                "gemini",
                "perplexity",
            }:
                row["metric_eligible"] = False
                row["metric_evidence_state"] = "excluded"
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertTrue(admission["allowed"])
        self.assertEqual(admission["quality_state"], "degraded")
        self.assertIn(
            "web_too_many_unavailable_providers",
            admission["warning_codes"],
        )
        require_panel_metric_coverage(admission)

    def test_prompt_without_a_cross_model_quorum_is_degraded(self) -> None:
        rows = _observed_cells()
        for row in rows:
            if (
                row["mode"] == "memory"
                and row["prompt_id"] == 4
                and row["provider_key"] != "openai"
            ):
                row["metric_eligible"] = False
                row["metric_evidence_state"] = "provider_limited_prefix"
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertTrue(admission["allowed"])
        self.assertEqual(admission["quality_state"], "degraded")
        self.assertIn(
            "memory_prompt_4_coverage_below_minimum",
            admission["warning_codes"],
        )
        require_panel_metric_coverage(admission)

    def test_fully_absent_mode_is_degraded_when_other_mode_is_complete(self) -> None:
        rows = _observed_cells()
        for row in rows:
            if row["mode"] == "web":
                row["metric_eligible"] = False
                row["metric_evidence_state"] = "excluded"
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertTrue(admission["allowed"])
        self.assertEqual(admission["quality_state"], "degraded")
        self.assertIn(
            "web_mode_coverage_below_minimum",
            admission["warning_codes"],
        )
        self.assertIn("overall_coverage_below_minimum", admission["warning_codes"])
        require_panel_metric_coverage(admission)

    def test_zero_eligible_panel_evidence_is_blocked(self) -> None:
        rows = _observed_cells()
        for row in rows:
            row["metric_eligible"] = False
            row["metric_evidence_state"] = "excluded"
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertFalse(admission["allowed"])
        self.assertEqual(admission["quality_state"], "blocked")
        self.assertIn("zero_eligible_panel_evidence", admission["reason_codes"])
        with self.assertRaises(PanelMetricCoverageError):
            require_panel_metric_coverage(admission)

    def test_tampered_disallowed_receipt_cannot_bypass_require(self) -> None:
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=_observed_cells(),
        )
        admission["allowed"] = False

        with self.assertRaises(PanelMetricCoverageError):
            require_panel_metric_coverage(admission)

    def test_missing_whole_provider_is_structural_loss_not_model_mismatch(self) -> None:
        rows = [
            row
            for row in _observed_cells()
            if not (row["mode"] == "web" and row["provider_key"] == "gemini")
        ]
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertFalse(admission["allowed"])
        self.assertEqual(admission["quality_state"], "blocked")
        self.assertIn("missing_panel_cells", admission["reason_codes"])
        self.assertNotIn("panel_model_grid_mismatch", admission["reason_codes"])
        with self.assertRaises(PanelMetricCoverageError):
            require_panel_metric_coverage(admission)

    def test_admission_digest_is_deterministic_and_rebinds_quality(self) -> None:
        expected = _expected_cells()
        observed = _observed_cells()
        original = build_panel_metric_coverage_admission(
            expected_cells=expected,
            observed_rows=observed,
        )
        repeated = build_panel_metric_coverage_admission(
            expected_cells=reversed(expected),
            observed_rows=reversed(observed),
        )
        self.assertEqual(original["admission_sha256"], repeated["admission_sha256"])

        observed[0]["metric_eligible"] = False
        observed[0]["metric_evidence_state"] = "excluded"
        rebound = build_panel_metric_coverage_admission(
            expected_cells=expected,
            observed_rows=observed,
        )
        self.assertNotEqual(original["admission_sha256"], rebound["admission_sha256"])

    def test_wrong_model_and_unknown_cell_are_rejected(self) -> None:
        rows = _observed_cells()
        rows[0]["model"] = "unexpected/model"
        rows.append(
            {
                "prompt_id": 1,
                "provider_key": "intruder",
                "mode": "web",
                "model": "intruder/model",
                "status": "completed",
                "metric_eligible": True,
            }
        )
        admission = build_panel_metric_coverage_admission(
            expected_cells=_expected_cells(),
            observed_rows=rows,
        )

        self.assertFalse(admission["allowed"])
        self.assertIn("panel_model_grid_mismatch", admission["reason_codes"])
        self.assertIn("unexpected_panel_cells", admission["reason_codes"])


if __name__ == "__main__":
    unittest.main()
