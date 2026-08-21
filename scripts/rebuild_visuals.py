"""Rebuild report illustrations from saved concepts and report data only."""

from __future__ import annotations

import argparse
import asyncio
import json

from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

import app.services.analyzer as analyzer
from app.db import SessionLocal
from app.models import Run, RunArtifact


async def _quiet_progress(*args: object, **kwargs: object) -> None:
    del args, kwargs


async def rebuild_visuals(run_id: str) -> list[dict[str, object]]:
    """Regenerate images without crawling, prompt design or panel requests."""

    analyzer.update_progress = _quiet_progress
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one()
        report = dict(run.report_json or {})
        artifact = (
            await session.execute(
                select(RunArtifact).where(
                    RunArtifact.run_id == run_id,
                    RunArtifact.artifact_key == "illustration_concepts",
                )
            )
        ).scalar_one()
        concepts = list(
            (artifact.output_json or {}).get("illustrations") or []
        )
        brand_name = str(
            (report.get("brand") or {}).get("name") or run.domain
        )

    result = await analyzer._generate_illustrations(
        run_id,
        brand_name=brand_name,
        concepts=concepts,
        public_report=report,
    )

    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one()
        updated_report = dict(run.report_json or {})
        updated_report["illustrations"] = result
        run.report_json = updated_report
        flag_modified(run, "report_json")
        await session.commit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    result = asyncio.run(rebuild_visuals(args.run_id))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
