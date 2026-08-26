"""Rebuild report illustrations from saved concepts and report data only."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json

from sqlalchemy import select

import app.services.analyzer as analyzer
from app.db import SessionLocal
from app.models import Run, RunArtifact, RunStatus
from app.services.publication_contract import (
    PublicationContractError,
    ensure_publication_contract,
    publication_snapshot,
    publication_snapshot_digest,
    replace_completed_publication,
)


async def _quiet_progress(*args: object, **kwargs: object) -> None:
    del args, kwargs


async def rebuild_visuals(run_id: str) -> list[dict[str, object]]:
    """Regenerate images without crawling, prompt design or panel requests."""

    analyzer.update_progress = _quiet_progress
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run is None:
            raise RuntimeError("Run not found")
        if run.status != RunStatus.completed:
            raise RuntimeError("Visual rebuild requires a completed run")
        if not isinstance(run.report_json, dict) or not isinstance(
            run.analysis_markdown,
            str,
        ):
            raise RuntimeError("Completed run has no publishable report snapshot")
        try:
            receipt = await ensure_publication_contract(
                session,
                run,
                allow_legacy_baseline=False,
            )
        except PublicationContractError as exc:
            raise RuntimeError(
                "Current report has no valid publication receipt; "
                "use the full saved-run reprocess path"
            ) from exc
        if not isinstance(receipt, dict) or receipt.get("legacy_baseline") is True:
            raise RuntimeError(
                "Legacy report has no reader-copy provenance; "
                "use the full saved-run reprocess path"
            )
        report = copy.deepcopy(run.report_json)
        markdown = run.analysis_markdown
        expected_snapshot_digest = publication_snapshot_digest(
            publication_snapshot(
                report_json=report,
                analysis_markdown=markdown,
            )
        )
        concept_artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "illustration_concepts",
                    RunArtifact.status == "completed",
                )
            )
        ).scalar_one_or_none()
        illustration_copy_artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "illustration_copy_editorial",
                    RunArtifact.status == "completed",
                )
            )
        ).scalar_one_or_none()
        final_artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "final_report_editorial",
                    RunArtifact.status == "completed",
                )
            )
        ).scalar_one_or_none()
        if (
            concept_artifact is None
            or illustration_copy_artifact is None
            or final_artifact is None
        ):
            raise RuntimeError(
                "Required completed editorial artifacts are missing; "
                "use the full saved-run reprocess path"
            )
        raw_concepts = list(
            (concept_artifact.output_json or {}).get("illustrations") or []
        )
        copy_document = (illustration_copy_artifact.output_json or {}).get("copy")
        final_report = (final_artifact.output_json or {}).get("report")
        if not isinstance(copy_document, dict) or not isinstance(
            final_report,
            dict,
        ):
            raise RuntimeError("Completed editorial artifacts have invalid output")
        try:
            concepts = analyzer._merge_illustration_copy(
                raw_concepts,
                copy_document,
            )
        except Exception as exc:
            raise RuntimeError(
                "Saved illustration concepts do not match their editorial receipt"
            ) from exc
        public_report = copy.deepcopy(report)
        for presentation_key in ("narrative", "illustrations", "site_preview"):
            public_report.pop(presentation_key, None)
        public_report_sha256 = analyzer._stable_json_sha256(public_report)
        expected_narrative = {
            "headline": final_report.get("headline"),
            "headline_emphasis": final_report.get("headline_emphasis") or [],
            "verdict": final_report.get("verdict"),
            "executive_summary": final_report.get("executive_summary"),
            "actions": final_report.get("actions") or [],
        }
        if (
            report.get("narrative") != expected_narrative
            or analyzer._render_markdown(final_report) != markdown
            or not isinstance(final_artifact.input_json, dict)
            or final_artifact.input_json.get("public_report_sha256")
            != public_report_sha256
            or not isinstance(illustration_copy_artifact.input_json, dict)
            or illustration_copy_artifact.input_json.get("public_report_sha256")
            != public_report_sha256
            or not isinstance(concept_artifact.input_json, dict)
            or concept_artifact.input_json.get("report_data") != public_report
        ):
            raise RuntimeError(
                "Saved editorial and illustration artifacts are not bound to "
                "the current report; use the full saved-run reprocess path"
            )
        brand_name = str(
            (public_report.get("brand") or {}).get("name") or run.domain
        )

    result = await analyzer._generate_illustrations(
        run_id,
        brand_name=brand_name,
        concepts=concepts,
        public_report=public_report,
    )

    updated_report = copy.deepcopy(report)
    updated_report["illustrations"] = result
    reader_copy_manifest = await analyzer._save_reader_copy_manifest(
        run_id,
        final_report=final_report,
        public_report=public_report,
        illustrations=result,
        analysis_markdown=markdown,
        report_json=updated_report,
    )
    await replace_completed_publication(
        run_id=run_id,
        expected_snapshot_digest=expected_snapshot_digest,
        report_json=updated_report,
        analysis_markdown=markdown,
        reader_copy_manifest=reader_copy_manifest,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    result = asyncio.run(rebuild_visuals(args.run_id))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
