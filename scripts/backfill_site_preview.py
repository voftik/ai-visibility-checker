"""Attach a real first-screen snapshot to an already completed report."""
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import SessionLocal
from app.models import Run, RunArtifact, RunStatus, SitePage
from app.services import analyzer
from app.services.crawler import _validate_public_url
from app.services.publication_contract import (
    PublicationContractError,
    ensure_publication_contract,
    publication_snapshot,
    publication_snapshot_digest,
    replace_completed_publication,
)
from app.services.site_preview import capture_site_preview


async def backfill(run_id: str) -> str:
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run is None or not run.domain:
            raise RuntimeError("Run not found or has no domain")
        if run.status != RunStatus.completed:
            raise RuntimeError("Site-preview backfill requires a completed run")
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
        final_artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "final_report_editorial",
                    RunArtifact.status == "completed",
                )
            )
        ).scalar_one_or_none()
        final_report = (
            (final_artifact.output_json or {}).get("report")
            if final_artifact is not None
            else None
        )
        if not isinstance(final_report, dict):
            raise RuntimeError(
                "Completed final-report editorial artifact is missing; "
                "use the full saved-run reprocess path"
            )
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
        ):
            raise RuntimeError(
                "Saved final-report artifact is not bound to the current report; "
                "use the full saved-run reprocess path"
            )
        illustrations = copy.deepcopy(report.get("illustrations") or [])
        page = (
            await session.execute(
                select(SitePage)
                .where(
                    SitePage.run_id == run_id,
                    SitePage.page_kind == "home",
                )
                .order_by(SitePage.id)
            )
        ).scalars().first()
        source_url = page.url if page is not None else f"https://{run.domain}/"
        domain = run.domain

    preview = await capture_site_preview(
        run_id,
        domain=domain,
        source_url=source_url,
        validate_url=_validate_public_url,
    )
    if preview is None:
        raise RuntimeError("Site preview capture failed")

    updated_report = copy.deepcopy(report)
    updated_report["site_preview"] = preview
    reader_copy_manifest = await analyzer._save_reader_copy_manifest(
        run_id,
        final_report=final_report,
        public_report=public_report,
        illustrations=illustrations,
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
    return str(preview["file_url"])


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    return parser.parse_args()


if __name__ == "__main__":
    print(asyncio.run(backfill(_args().run_id)))
