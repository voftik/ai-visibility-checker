"""Attach a real first-screen snapshot to an already completed report."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import SessionLocal
from app.models import Run, SitePage
from app.services.crawler import _validate_public_url
from app.services.site_preview import capture_site_preview


async def backfill(run_id: str) -> str:
    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one_or_none()
        if run is None or not run.domain:
            raise RuntimeError("Run not found or has no domain")
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

    async with SessionLocal() as session:
        run = (
            await session.execute(select(Run).where(Run.id == run_id))
        ).scalar_one()
        report = dict(run.report_json or {})
        report["site_preview"] = preview
        run.report_json = report
        await session.commit()
    return str(preview["file_url"])


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    return parser.parse_args()


if __name__ == "__main__":
    print(asyncio.run(backfill(_args().run_id)))
