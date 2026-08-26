"""Deterministic admission for a report built from a partially available panel.

Transport success is not the same thing as analytic evidence.  In particular,
an answer cut off at the provider's physical output boundary is valuable raw
context, but it cannot support an absence metric.  This module keeps the
decision independent of SQL and model calls so the exact policy is replayable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

PANEL_METRIC_COVERAGE_ADMISSION_VERSION = "aiv-panel-metric-coverage-v4"
PANEL_METRIC_MINIMUM_RATE = 0.60
PANEL_METRIC_MAX_UNAVAILABLE_PROVIDERS = {"web": 2, "memory": 1}


class PanelMetricCoverageError(RuntimeError):
    """The saved panel is too incomplete for a defensible public report."""


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cell_key(value: Mapping[str, Any]) -> tuple[int, str, str]:
    prompt_id = value.get("prompt_id")
    if isinstance(prompt_id, bool) or not isinstance(prompt_id, int):
        raise PanelMetricCoverageError("Panel cell prompt_id is invalid")
    provider = str(value.get("provider_key") or "").strip()
    mode = str(value.get("mode") or "").strip()
    if not provider or mode not in {"web", "memory"}:
        raise PanelMetricCoverageError("Panel cell provider or mode is invalid")
    return prompt_id, provider, mode


def _rate(eligible: int, expected: int) -> float:
    return round(eligible / expected, 6) if expected else 0.0


def _required_count(expected: int, minimum_rate: float) -> int:
    return math.ceil(expected * minimum_rate)


def build_panel_metric_coverage_admission(
    *,
    expected_cells: Iterable[Mapping[str, Any]],
    observed_rows: Iterable[Mapping[str, Any]],
    minimum_rate: float = PANEL_METRIC_MINIMUM_RATE,
) -> dict[str, Any]:
    """Return a content-addressed allow/block decision for panel coverage.

    Coverage is checked globally, per mode, per prompt and per provider.  A
    provider may be wholly or partly unavailable (and therefore shown with its
    actual denominator and a limitation) without blocking every other valid
    slice. Only structural corruption or a corpus with no eligible evidence
    vetoes publication.
    """

    if not 0 < minimum_rate <= 1:
        raise PanelMetricCoverageError("minimum_rate must be in (0, 1]")
    expected = [dict(value) for value in expected_cells]
    observed = [dict(value) for value in observed_rows]
    if not expected:
        raise PanelMetricCoverageError("Panel coverage has no expected cells")

    expected_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cell in expected:
        key = _cell_key(cell)
        if key in expected_by_key:
            raise PanelMetricCoverageError("Panel expected grid has duplicates")
        model = str(cell.get("model") or "").strip()
        if not model:
            raise PanelMetricCoverageError("Panel expected model is empty")
        expected_by_key[key] = {**cell, "model": model}
    expected_manifest = sorted(
        (
            {
                "prompt_id": key[0],
                "provider_key": key[1],
                "mode": key[2],
                "model": str(cell["model"]),
            }
            for key, cell in expected_by_key.items()
        ),
        key=lambda cell: (
            int(cell["prompt_id"]),
            str(cell["provider_key"]),
            str(cell["model"]),
            str(cell["mode"]),
        ),
    )

    observed_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    unknown_cells: list[dict[str, Any]] = []
    for row in observed:
        key = _cell_key(row)
        if key in observed_by_key:
            raise PanelMetricCoverageError("Panel observed grid has duplicates")
        observed_by_key[key] = row
        if key not in expected_by_key:
            unknown_cells.append(
                {
                    "prompt_id": key[0],
                    "provider_key": key[1],
                    "mode": key[2],
                    "model": str(row.get("model") or ""),
                }
            )

    rows: list[dict[str, Any]] = []
    for key, cell in sorted(expected_by_key.items(), key=lambda item: item[0]):
        observed_row = observed_by_key.get(key)
        expected_model = str(cell["model"])
        actual_model = (
            str(observed_row.get("model") or "").strip()
            if observed_row is not None
            else ""
        )
        observed_cell = observed_row is not None
        # A missing row is kept distinct from a model mismatch so diagnostics
        # remain truthful.  It is still a hard structural loss below: provider
        # outages must be persisted as terminal failed rows, never disappear
        # from the immutable expected grid.
        model_matches = not observed_cell or actual_model == expected_model
        metric_eligible = bool(
            observed_cell
            and model_matches
            and observed_row.get("metric_eligible") is True
        )
        rows.append(
            {
                "prompt_id": key[0],
                "provider_key": key[1],
                "mode": key[2],
                "expected_model": expected_model,
                "actual_model": actual_model or None,
                "status": (
                    str(observed_row.get("status") or "missing")
                    if observed_row is not None
                    else "missing"
                ),
                "metric_eligible": metric_eligible,
                "metric_evidence_state": (
                    observed_row.get("metric_evidence_state")
                    if observed_row is not None
                    else "missing"
                ),
                "metric_limitation": (
                    observed_row.get("metric_limitation")
                    if observed_row is not None
                    else "missing_cell"
                ),
                "observed": observed_cell,
                "model_matches": model_matches,
            }
        )

    blockers: list[str] = []
    warnings: list[str] = []
    if unknown_cells:
        blockers.append("unexpected_panel_cells")
    if any(not row["observed"] for row in rows):
        blockers.append("missing_panel_cells")
    if any(row["observed"] and not row["model_matches"] for row in rows):
        blockers.append("panel_model_grid_mismatch")

    def grouped_receipts(
        group_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in group_fields)].append(row)
        receipts: list[dict[str, Any]] = []
        for key, members in sorted(groups.items(), key=lambda item: item[0]):
            eligible = sum(row["metric_eligible"] is True for row in members)
            expected_count = len(members)
            receipts.append(
                {
                    **dict(zip(group_fields, key, strict=True)),
                    "expected_cells": expected_count,
                    "eligible_cells": eligible,
                    "minimum_eligible_cells": _required_count(
                        expected_count,
                        minimum_rate,
                    ),
                    "coverage_rate": _rate(eligible, expected_count),
                }
            )
        return receipts

    mode_receipts = grouped_receipts(("mode",))
    prompt_mode_receipts = grouped_receipts(("mode", "prompt_id"))
    provider_mode_receipts = grouped_receipts(("mode", "provider_key"))

    for receipt in mode_receipts:
        if receipt["eligible_cells"] < receipt["minimum_eligible_cells"]:
            warnings.append(f"{receipt['mode']}_mode_coverage_below_minimum")
    for receipt in prompt_mode_receipts:
        if receipt["eligible_cells"] < receipt["minimum_eligible_cells"]:
            warnings.append(
                f"{receipt['mode']}_prompt_{receipt['prompt_id']}_coverage_below_minimum"
            )

    unavailable_by_mode: dict[str, int] = defaultdict(int)
    for receipt in provider_mode_receipts:
        if receipt["eligible_cells"] == 0:
            unavailable_by_mode[str(receipt["mode"])] += 1
            warnings.append(
                f"{receipt['mode']}_{receipt['provider_key']}_provider_unavailable"
            )
        elif receipt["eligible_cells"] < receipt["minimum_eligible_cells"]:
            # A sparse provider slice is an availability limitation, not an
            # integrity failure.  Keep its eligible observations, mark the
            # slice degraded and let the report show its actual denominator.
            # Blocking here made one verbose or flaky provider veto an
            # otherwise complete 81-cell research run.
            warnings.append(
                f"{receipt['mode']}_{receipt['provider_key']}_partial_slice_below_minimum"
            )
    for mode, count in sorted(unavailable_by_mode.items()):
        allowed_unavailable = PANEL_METRIC_MAX_UNAVAILABLE_PROVIDERS.get(mode, 0)
        if count > allowed_unavailable:
            warnings.append(f"{mode}_too_many_unavailable_providers")

    total_eligible = sum(row["metric_eligible"] is True for row in rows)
    total_expected = len(rows)
    if total_eligible == 0:
        blockers.append("zero_eligible_panel_evidence")
    elif total_eligible < _required_count(total_expected, minimum_rate):
        warnings.append("overall_coverage_below_minimum")
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    quality_state = "blocked" if blockers else "degraded" if warnings else "complete"
    core = {
        "version": PANEL_METRIC_COVERAGE_ADMISSION_VERSION,
        "minimum_rate": minimum_rate,
        "allowed": not blockers,
        "quality_state": quality_state,
        # ``reason_codes`` remains the compatibility field for hard blockers.
        # Availability and aggregate coverage limitations are explicit warnings.
        "reason_codes": blockers,
        "warning_codes": warnings,
        "expected_cell_count": total_expected,
        "eligible_cell_count": total_eligible,
        "coverage_rate": _rate(total_eligible, total_expected),
        "mode_receipts": mode_receipts,
        "prompt_mode_receipts": prompt_mode_receipts,
        "provider_mode_receipts": provider_mode_receipts,
        "unavailable_provider_count_by_mode": dict(sorted(unavailable_by_mode.items())),
        "unexpected_cells": sorted(
            unknown_cells,
            key=lambda row: (
                row["mode"],
                row["provider_key"],
                row["prompt_id"],
            ),
        ),
        "expected_cell_manifest_sha256": _digest(expected_manifest),
        "cell_manifest_sha256": _digest(rows),
    }
    return {**core, "admission_sha256": _digest(core)}


def require_panel_metric_coverage(admission: Mapping[str, Any]) -> None:
    reasons = admission.get("reason_codes")
    quality_state = admission.get("quality_state")
    if (
        admission.get("allowed") is True
        and quality_state in {None, "complete", "degraded"}
        and not reasons
    ):
        return
    reason_text = ", ".join(str(value) for value in reasons or [])
    raise PanelMetricCoverageError(
        "Panel metric coverage admission failed"
        + (f": {reason_text}" if reason_text else "")
    )


__all__ = [
    "PANEL_METRIC_COVERAGE_ADMISSION_VERSION",
    "PANEL_METRIC_MAX_UNAVAILABLE_PROVIDERS",
    "PANEL_METRIC_MINIMUM_RATE",
    "PanelMetricCoverageError",
    "build_panel_metric_coverage_admission",
    "require_panel_metric_coverage",
]
