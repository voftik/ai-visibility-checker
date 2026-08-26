from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal, init_db
from app.main import app
from app.models import ReportIllustration, Run, RunArtifact, RunStatus
from app.services.analyzer import (
    _immutable_editorial_cache_proof,
    _revalidate_reused_illustration_assets,
)
from app.services.live_russian_policy import LIVE_RUSSIAN_POLICY_MANIFEST
from app.services.publication_contract import (
    IMMUTABLE_ILLUSTRATION_QA_PREFIX,
    IMMUTABLE_READER_COPY_PREFIX,
    LEGACY_PUBLICATION_BASELINE_VERSION,
    PUBLICATION_RECEIPT_PREFIX,
    PUBLICATION_RECEIPT_VERSION,
    READER_COPY_MANIFEST_VERSION,
    PublicationContractError,
    build_publication_receipt,
    ensure_publication_contract,
    publication_snapshot,
    publication_snapshot_digest,
    replace_completed_publication,
    stable_json_sha256,
    stage_publication_receipt,
)
from app.services.reader_copy_registry import READER_COPY_REGISTRY_MANIFEST
from app.services.report_editor import (
    edit_report,
    illustration_copy_narrative_paths,
)
from app.services.run_coordinator import (
    SAVED_ANSWERS_ONLY_MARKER_KEY,
    SAVED_ANSWERS_ONLY_MARKER_VERSION,
    SAVED_ANSWERS_ONLY_MODE,
)
from app.services.site_preview import site_preview_asset_receipt

TEST_ILLUSTRATION_POLICY = {
    "publish_only_verified_assets": True,
    "verified_subset_allowed": True,
    "zero_assets_allowed": True,
}


async def _valid_editorial_receipt() -> dict:
    source = {
        "illustrations": [
            {
                "role": "technical_access",
                "core_claim": "Сервер отдаёт основной текст краулеру.",
                "title": "Как краулер получает текст",
                "caption": "Сервер передаёт основной текст в HTML.",
                "alt_text": "Схема передачи текста сайта краулеру.",
                "evidence_paths": ["/technical/score"],
            }
        ]
    }
    prose_paths = illustration_copy_narrative_paths(source)

    async def editor(payload: dict) -> dict:
        return {
            "source_unit_id": payload["source_unit_id"],
            "source_sha256": payload["source_sha256"],
            "edited_text": str(payload["core_text"]),
            "claim_receipts": [
                {
                    "claim_sha256": claim["claim_sha256"],
                    "preserved": True,
                    "target_excerpt": claim["source_excerpt"],
                    "note": "Смысл сохранён.",
                }
                for claim in payload["source_claims"]
            ],
            "new_claims": [],
        }

    async def critic(payload: dict) -> dict:
        return {
            "verdict": "pass",
            "issues": [],
            "claim_checks": [
                {
                    "claim_sha256": claim["claim_sha256"],
                    "meaning_preserved": True,
                    "actor_preserved": True,
                    "scope_preserved": True,
                    "numbers_preserved": True,
                    "actor_or_mechanism_explicit": True,
                    "number_carrier_explicit": True,
                    "active_voice": True,
                    "no_slogan_or_meta": True,
                    "no_mechanical_triad": True,
                    "reason": "Смысл совпадает.",
                }
                for claim in payload["source_claims"]
            ],
            "new_claims": [],
        }

    result, audit = await edit_report(
        source,
        editor_call=editor,
        critic_call=critic,
        prose_paths=prose_paths,
    )
    proof, revalidated = _immutable_editorial_cache_proof(
        source=source,
        result=result,
        audit=audit,
        prose_paths=prose_paths,
        protected_terms=[],
        source_artifact_key="test_editorial_source",
    )
    if not revalidated or not isinstance(proof, dict):
        raise AssertionError("Test editorial fixture is not publication-valid")
    return {
        "accepted": True,
        "artifact_key": "test_editorial_result",
        "prompt_version": "test-editorial-v1",
        "audit_sha256": audit["audit_sha256"],
        "result_report_sha256": stable_json_sha256(result),
        "cache_revalidated": True,
        "cache_proof": proof,
        "cache_proof_sha256": proof["proof_sha256"],
        "reasons": [],
    }


async def _reader_copy_manifest(
    report_json: dict,
    markdown: str,
    asset_receipts: list[dict],
    site_preview_receipt: dict | None = None,
) -> dict:
    published_count = len(report_json.get("illustrations") or [])
    editorial_receipt = await _valid_editorial_receipt()
    publication = {
        "checks": {"exact_public_snapshot": True},
        "blocking_reasons": [],
        "report_json_sha256": stable_json_sha256(report_json),
        "analysis_markdown_sha256": hashlib.sha256(
            markdown.encode("utf-8")
        ).hexdigest(),
    }
    core = {
        "version": READER_COPY_MANIFEST_VERSION,
        "canonical_policy": LIVE_RUSSIAN_POLICY_MANIFEST.as_dict(),
        "code_owned_copy_registry": READER_COPY_REGISTRY_MANIFEST.as_dict(),
        "lint": {
            "policy_version": LIVE_RUSSIAN_POLICY_MANIFEST.version,
            "policy_sha256": LIVE_RUSSIAN_POLICY_MANIFEST.sha256,
            "blocking": False,
            "issues": [],
            "omitted_issue_count": 0,
        },
        "publication_contract": publication,
        "decision": "pass",
        "blocking_reasons": [],
        "quality_complete": True,
        "illustration_asset_receipts": asset_receipts,
        "illustration_asset_receipts_sha256": stable_json_sha256(asset_receipts),
        "site_preview_asset_receipt": site_preview_receipt,
        "site_preview_asset_receipt_sha256": (
            site_preview_receipt.get("receipt_sha256")
            if isinstance(site_preview_receipt, dict)
            else None
        ),
        "editorial_receipts": {
            "final_report": copy.deepcopy(editorial_receipt),
            "technical_review": copy.deepcopy(editorial_receipt),
            "illustrations": {
                **(
                    copy.deepcopy(editorial_receipt)
                    if published_count
                    else {
                        "accepted": True,
                        "artifact_key": None,
                        "prompt_version": None,
                        "audit_sha256": None,
                        "result_report_sha256": None,
                        "cache_revalidated": None,
                        "cache_proof": None,
                        "cache_proof_sha256": None,
                        "reasons": [],
                    }
                ),
                "state": "published" if published_count else "not_published",
                "published_count": published_count,
                "publication_policy": TEST_ILLUSTRATION_POLICY,
            },
            "illustration_assets": {
                "accepted": True,
                "published_count": published_count,
                "verified_count": published_count,
                "publication_policy": TEST_ILLUSTRATION_POLICY,
            },
        },
    }
    return {**core, "manifest_sha256": stable_json_sha256(core)}


def _qa_receipt(
    *,
    sequence: int,
    file_url: str,
    image_content: bytes,
    prompt_version: str = "illustration-qa-test-v1",
) -> tuple[dict, RunArtifact]:
    image_sha256 = hashlib.sha256(image_content).hexdigest()
    source_artifact_key = f"illustration_qa_{sequence}_{image_sha256[:16]}"
    qa_input = {"image_sha256": image_sha256, "sequence": sequence}
    qa_output = {"usable": True, "facts_grounded": True}
    qa_core = {
        "input": qa_input,
        "output": qa_output,
        "prompt_version": prompt_version,
    }
    qa_receipt_sha256 = stable_json_sha256(qa_core)
    artifact_key = f"{IMMUTABLE_ILLUSTRATION_QA_PREFIX}{sequence}_{qa_receipt_sha256}"
    receipt = {
        "sequence": sequence,
        "file_url": file_url,
        "image_sha256": image_sha256,
        "image_bytes": len(image_content),
        "qa_verified": True,
        "qa_artifact_key": artifact_key,
        "qa_source_artifact_key": source_artifact_key,
        "qa_receipt_sha256": qa_receipt_sha256,
        "content_addressed_filename": True,
    }
    artifact = RunArtifact(
        stage_key="publication",
        artifact_key=artifact_key,
        status="completed",
        prompt_version=prompt_version,
        input_json={
            "sequence": sequence,
            "file_url": file_url,
            "image_sha256": image_sha256,
            "source_artifact_key": source_artifact_key,
            "qa_receipt_sha256": qa_receipt_sha256,
        },
        output_json={
            **qa_core,
            "source_artifact_key": source_artifact_key,
            "receipt_sha256": qa_receipt_sha256,
        },
    )
    return receipt, artifact


class PublicationContractIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def _delete_run(self, run_id: str) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id == run_id))
            await session.commit()

    async def test_legacy_completed_report_gets_explicit_baseline_then_fails_on_tamper(
        self,
    ) -> None:
        run_id = f"legacy-publication-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="legacy.example",
                    status=RunStatus.completed,
                    config_json={},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    analysis_markdown="# Старый отчёт",
                    report_json={
                        "narrative": {"headline": "Старый отчёт"},
                        "illustrations": [],
                    },
                )
            )
            await session.commit()
        try:
            first = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(first.status_code, 200)
            async with SessionLocal() as session:
                receipt = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.artifact_key.like(
                                f"{PUBLICATION_RECEIPT_PREFIX}%"
                            ),
                        )
                    )
                ).scalar_one()
                self.assertEqual(
                    receipt.prompt_version,
                    LEGACY_PUBLICATION_BASELINE_VERSION,
                )
                self.assertTrue(receipt.output_json["legacy_baseline"])
                self.assertFalse(receipt.output_json["historical_pipeline_provenance"])

                run = await session.get(Run, run_id)
                run.report_json = {
                    "narrative": {"headline": "Подменённый отчёт"},
                    "illustrations": [],
                }
                await session.commit()

            tampered = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(tampered.status_code, 409)
        finally:
            await self._delete_run(run_id)

    async def test_legacy_baseline_unique_race_revalidates_plain_snapshot(
        self,
    ) -> None:
        run_id = f"legacy-publication-race-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="legacy-race.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown="# Старый отчёт",
                    report_json={
                        "narrative": {"headline": "Старый отчёт"},
                        "illustrations": [],
                    },
                )
            )
            await session.commit()
        try:
            async with SessionLocal() as session:
                run = await session.get(Run, run_id)
                real_commit = session.commit

                async def committed_race_loser() -> None:
                    # Persist the identical row as if a concurrent reader won,
                    # then expire the ORM run and surface the UNIQUE race that
                    # the production branch handles with rollback + reload.
                    await real_commit()
                    session.expire(run)
                    raise IntegrityError("simulated unique race", {}, Exception())

                with patch.object(session, "commit", committed_race_loser):
                    receipt = await ensure_publication_contract(session, run)
                self.assertIsNotNone(receipt)
                self.assertTrue(receipt["legacy_baseline"])
        finally:
            await self._delete_run(run_id)

    async def test_unfinished_run_never_exposes_or_shares_report_snapshot(
        self,
    ) -> None:
        run_id = f"unfinished-publication-{uuid.uuid4()}"
        share_token = f"unfinished-{uuid.uuid4().hex}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="unfinished.example",
                    status=RunStatus.analyzing,
                    config_json={},
                    share_token=share_token,
                    analysis_markdown="# Не опубликовано",
                    report_json={
                        "narrative": {"headline": "Не опубликовано"},
                        "illustrations": [
                            {
                                "sequence": 1,
                                "title": "Черновик",
                                "caption": "Черновик",
                                "alt_text": "Черновик",
                                "file_url": "/static/generated/draft.png",
                            }
                        ],
                    },
                )
            )
            await session.commit()
        try:
            detail = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(detail.status_code, 200)
            self.assertIsNone(detail.json()["analysis_markdown"])
            self.assertIsNone(detail.json()["report_json"])
            self.assertEqual(detail.json()["illustrations"], [])
            create_share = await self.client.post(f"/api/runs/{run_id}/share")
            self.assertEqual(create_share.status_code, 409)
            shared = await self.client.get(f"/api/shared/{share_token}")
            self.assertEqual(shared.status_code, 409)
        finally:
            await self._delete_run(run_id)

    async def test_saved_answer_reprocess_keeps_previous_receipt_public(
        self,
    ) -> None:
        run_id = str(uuid.uuid4())
        share_token = f"reprocess-{uuid.uuid4().hex}"
        report_json = {
            "narrative": {"headline": "Опубликованный отчёт"},
            "illustrations": [],
        }
        markdown = "# Опубликованный отчёт"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="reprocess-public.example",
                    status=RunStatus.completed,
                    config_json={"page_limit": 6},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    share_token=share_token,
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.commit()
        try:
            self.assertEqual(
                (await self.client.get(f"/api/runs/{run_id}")).status_code,
                200,
            )
            marker = {
                "version": SAVED_ANSWERS_ONLY_MARKER_VERSION,
                "mode": SAVED_ANSWERS_ONLY_MODE,
                "run_id": run_id,
                "owner": "test-operator",
                "attempt_count": 1,
                "raw_answers_sha256": "a" * 64,
                "previous_config_json": {"page_limit": 6},
                "previous_terminal_state": {
                    "status": RunStatus.completed.value,
                    "progress_current": 100,
                    "progress_total": 100,
                    "progress_percent": 100,
                    "stage_key": "report",
                    "stage_label": "Собираем отчёт и иллюстрации",
                    "stage_detail": "Отчёт готов.",
                    "eta_seconds": 0,
                    "error_message": None,
                    "checkpointed_at": None,
                    "finished_at": None,
                },
            }
            async with SessionLocal() as session:
                run = await session.get(Run, run_id)
                run.status = RunStatus.analyzing
                run.config_json = {
                    "page_limit": 6,
                    SAVED_ANSWERS_ONLY_MARKER_KEY: marker,
                }
                run.execution_slot = 1
                run.lease_owner = "test-operator"
                await session.commit()

            detail = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json()["status"], "analyzing")
            self.assertEqual(detail.json()["report_json"], report_json)
            self.assertEqual(
                detail.json()["analysis_markdown"],
                markdown,
            )
            shared = await self.client.get(f"/api/shared/{share_token}")
            self.assertEqual(shared.status_code, 200)
            self.assertEqual(shared.json()["report_json"], report_json)
        finally:
            await self._delete_run(run_id)

    async def test_site_preview_bytes_are_bound_to_new_and_shared_publication(
        self,
    ) -> None:
        run_id = str(uuid.uuid4())
        share_token = f"preview-{uuid.uuid4().hex}"
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        static_dir = Path(temporary_directory.name) / "static"
        generated_dir = static_dir / "generated"
        run_dir = generated_dir / run_id
        run_dir.mkdir(parents=True)
        image_content = b"\xff\xd8\xffverified-site-preview"
        image_sha256 = hashlib.sha256(image_content).hexdigest()
        filename = f"site-preview-{image_sha256[:12]}.jpg"
        image_path = run_dir / filename
        image_path.write_bytes(image_content)
        preview = {
            "state": "captured",
            "file_url": f"/static/generated/{run_id}/{filename}",
            "source_domain": "preview.example",
            "width": 1440,
            "height": 900,
            "captured_at": "2026-08-26T10:00:00+00:00",
            "sha256": image_sha256,
        }
        report_json = {
            "narrative": {"headline": "Отчёт со снимком"},
            "illustrations": [],
            "site_preview": preview,
        }
        markdown = "# Отчёт со снимком"
        with (
            patch(
                "app.services.publication_contract.STATIC_DIR",
                static_dir,
            ),
            patch(
                "app.services.publication_contract.GENERATED_DIR",
                generated_dir,
            ),
            patch("app.services.site_preview.STATIC_DIR", static_dir),
            patch("app.services.site_preview.GENERATED_DIR", generated_dir),
        ):
            preview_receipt = site_preview_asset_receipt(run_id, preview)
            reader_manifest = await _reader_copy_manifest(
                report_json,
                markdown,
                [],
                preview_receipt,
            )
            async with SessionLocal() as session:
                session.add(
                    Run(
                        id=run_id,
                        domain="preview.example",
                        status=RunStatus.completed,
                        config_json={},
                        share_token=share_token,
                        analysis_markdown=markdown,
                        report_json=report_json,
                    )
                )
                await session.flush()
                receipt = await stage_publication_receipt(
                    session,
                    run_id=run_id,
                    report_json=report_json,
                    analysis_markdown=markdown,
                    reader_copy_manifest=reader_manifest,
                )
                await session.commit()
            try:
                self.assertEqual(
                    receipt["site_preview_asset_receipt"],
                    preview_receipt,
                )
                self.assertEqual(
                    (await self.client.get(f"/api/runs/{run_id}")).status_code,
                    200,
                )
                self.assertEqual(
                    (await self.client.get(f"/api/shared/{share_token}")).status_code,
                    200,
                )

                image_path.write_bytes(b"\xff\xd8\xfftampered-site-preview")
                self.assertEqual(
                    (await self.client.get(f"/api/runs/{run_id}")).status_code,
                    409,
                )
                self.assertEqual(
                    (await self.client.get(f"/api/shared/{share_token}")).status_code,
                    409,
                )
            finally:
                await self._delete_run(run_id)

    async def test_legacy_site_preview_baseline_binds_exact_bytes(self) -> None:
        run_id = str(uuid.uuid4())
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        static_dir = Path(temporary_directory.name) / "static"
        generated_dir = static_dir / "generated"
        run_dir = generated_dir / run_id
        run_dir.mkdir(parents=True)
        image_content = b"\xff\xd8\xfflegacy-site-preview"
        image_sha256 = hashlib.sha256(image_content).hexdigest()
        filename = f"site-preview-{image_sha256[:12]}.jpg"
        image_path = run_dir / filename
        image_path.write_bytes(image_content)
        report_json = {
            "narrative": {"headline": "Старый отчёт со снимком"},
            "illustrations": [],
            "site_preview": {
                "state": "captured",
                "file_url": f"/static/generated/{run_id}/{filename}",
                "source_domain": "legacy-preview.example",
                "width": 1440,
                "height": 900,
                "captured_at": "2026-08-26T10:00:00+00:00",
                "sha256": image_sha256,
            },
        }
        with (
            patch(
                "app.services.publication_contract.STATIC_DIR",
                static_dir,
            ),
            patch(
                "app.services.publication_contract.GENERATED_DIR",
                generated_dir,
            ),
            patch("app.services.site_preview.STATIC_DIR", static_dir),
            patch("app.services.site_preview.GENERATED_DIR", generated_dir),
        ):
            async with SessionLocal() as session:
                session.add(
                    Run(
                        id=run_id,
                        domain="legacy-preview.example",
                        status=RunStatus.completed,
                        config_json={},
                        analysis_markdown="# Старый отчёт со снимком",
                        report_json=report_json,
                    )
                )
                await session.commit()
            try:
                self.assertEqual(
                    (await self.client.get(f"/api/runs/{run_id}")).status_code,
                    200,
                )
                image_path.write_bytes(b"\xff\xd8\xfftampered-legacy-preview")
                self.assertEqual(
                    (await self.client.get(f"/api/runs/{run_id}")).status_code,
                    409,
                )
            finally:
                await self._delete_run(run_id)

    async def test_new_receipt_binds_report_markdown_and_asset_bytes(self) -> None:
        run_id = f"new-publication-{uuid.uuid4()}"
        share_token = f"share-{uuid.uuid4().hex}"
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        static_dir = Path(temporary_directory.name) / "static"
        generated_dir = static_dir / "generated"
        run_dir = generated_dir / run_id
        run_dir.mkdir(parents=True)
        image_content = b"verified-public-image"
        image_sha256 = hashlib.sha256(image_content).hexdigest()
        filename = f"01-{image_sha256}.png"
        (run_dir / filename).write_bytes(image_content)
        file_url = f"/static/generated/{run_id}/{filename}"
        report_json = {
            "narrative": {"headline": "Проверенный отчёт"},
            "illustrations": [
                {
                    "sequence": 1,
                    "title": "Проверенная схема",
                    "caption": "Подпись",
                    "alt_text": "Описание",
                    "file_url": file_url,
                }
            ],
        }
        markdown = "# Проверенный отчёт"
        asset_receipt, qa_artifact = _qa_receipt(
            sequence=1,
            file_url=file_url,
            image_content=image_content,
        )
        asset_receipts = [asset_receipt]
        reader_manifest = await _reader_copy_manifest(
            report_json,
            markdown,
            asset_receipts,
        )
        with (
            patch(
                "app.services.publication_contract.STATIC_DIR",
                static_dir,
            ),
            patch(
                "app.services.publication_contract.GENERATED_DIR",
                generated_dir,
            ),
        ):
            async with SessionLocal() as session:
                session.add(
                    Run(
                        id=run_id,
                        domain="new.example",
                        status=RunStatus.completed,
                        config_json={},
                        progress_current=100,
                        progress_total=100,
                        progress_percent=100,
                        share_token=share_token,
                        analysis_markdown=markdown,
                        report_json=report_json,
                    )
                )
                qa_artifact.run_id = run_id
                session.add(qa_artifact)
                session.add(
                    RunArtifact(
                        run_id=run_id,
                        stage_key="illustration",
                        artifact_key=asset_receipt["qa_source_artifact_key"],
                        status="completed",
                        prompt_version="mutable-source-qa-v1",
                        input_json={"image_sha256": image_sha256},
                        output_json={"usable": True},
                    )
                )
                await session.flush()
                await stage_publication_receipt(
                    session,
                    run_id=run_id,
                    report_json=report_json,
                    analysis_markdown=markdown,
                    reader_copy_manifest=reader_manifest,
                )
                await session.commit()

            try:
                published = await self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(published.status_code, 200)
                shared = await self.client.get(f"/api/shared/{share_token}")
                self.assertEqual(shared.status_code, 200)
                async with SessionLocal() as session:
                    receipt = (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.artifact_key.like(
                                    f"{PUBLICATION_RECEIPT_PREFIX}%"
                                ),
                            )
                        )
                    ).scalar_one()
                    self.assertEqual(
                        receipt.prompt_version,
                        PUBLICATION_RECEIPT_VERSION,
                    )
                    self.assertFalse(receipt.output_json["legacy_baseline"])
                    self.assertEqual(
                        receipt.output_json["reader_copy_manifest_artifact_key"],
                        IMMUTABLE_READER_COPY_PREFIX
                        + reader_manifest["manifest_sha256"],
                    )

                async with SessionLocal() as session:
                    mutable_source = (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.artifact_key
                                == asset_receipt["qa_source_artifact_key"],
                            )
                        )
                    ).scalar_one()
                    mutable_source.output_json = {"usable": False}
                    await session.commit()
                source_changed = await self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(source_changed.status_code, 200)

                async with SessionLocal() as session:
                    qa_row = (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.artifact_key
                                == asset_receipt["qa_artifact_key"],
                            )
                        )
                    ).scalar_one()
                    original_qa_output = dict(qa_row.output_json or {})
                    qa_row.output_json = {"usable": False}
                    await session.commit()
                qa_tampered = await self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(qa_tampered.status_code, 409)
                async with SessionLocal() as session:
                    qa_row = (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.artifact_key
                                == asset_receipt["qa_artifact_key"],
                            )
                        )
                    ).scalar_one()
                    qa_row.output_json = original_qa_output
                    await session.commit()
                restored = await self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(restored.status_code, 200)

                (run_dir / filename).write_bytes(b"tampered")
                tampered = await self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(tampered.status_code, 409)
                shared_tampered = await self.client.get(f"/api/shared/{share_token}")
                self.assertEqual(shared_tampered.status_code, 409)
            finally:
                await self._delete_run(run_id)

    async def test_verified_one_of_three_and_two_of_three_subsets_publish(
        self,
    ) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        static_dir = Path(temporary_directory.name) / "static"
        generated_dir = static_dir / "generated"
        with (
            patch(
                "app.services.publication_contract.STATIC_DIR",
                static_dir,
            ),
            patch(
                "app.services.publication_contract.GENERATED_DIR",
                generated_dir,
            ),
        ):
            for sequences in ((2,), (1, 3)):
                with self.subTest(sequences=sequences):
                    run_id = f"subset-publication-{uuid.uuid4()}"
                    run_dir = generated_dir / run_id
                    run_dir.mkdir(parents=True)
                    rows: list[dict] = []
                    receipts: list[dict] = []
                    artifacts: list[RunArtifact] = []
                    for sequence in sequences:
                        content = f"image-{sequence}".encode("utf-8")
                        image_sha256 = hashlib.sha256(content).hexdigest()
                        filename = f"{sequence:02d}-{image_sha256}.png"
                        (run_dir / filename).write_bytes(content)
                        file_url = f"/static/generated/{run_id}/{filename}"
                        rows.append(
                            {
                                "sequence": sequence,
                                "title": f"Схема {sequence}",
                                "caption": "Подпись",
                                "alt_text": "Описание",
                                "file_url": file_url,
                            }
                        )
                        receipt, artifact = _qa_receipt(
                            sequence=sequence,
                            file_url=file_url,
                            image_content=content,
                        )
                        artifact.run_id = run_id
                        receipts.append(receipt)
                        artifacts.append(artifact)
                    report_json = {
                        "narrative": {"headline": "Частичный набор"},
                        "illustrations": rows,
                    }
                    markdown = "# Частичный набор"
                    manifest = await _reader_copy_manifest(
                        report_json,
                        markdown,
                        receipts,
                    )
                    async with SessionLocal() as session:
                        session.add(
                            Run(
                                id=run_id,
                                domain="subset.example",
                                status=RunStatus.completed,
                                config_json={},
                                analysis_markdown=markdown,
                                report_json=report_json,
                            )
                        )
                        session.add_all(artifacts)
                        await session.flush()
                        await stage_publication_receipt(
                            session,
                            run_id=run_id,
                            report_json=report_json,
                            analysis_markdown=markdown,
                            reader_copy_manifest=manifest,
                        )
                        await session.commit()
                    try:
                        response = await self.client.get(f"/api/runs/{run_id}")
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(
                            [
                                row["sequence"]
                                for row in response.json()["illustrations"]
                            ],
                            list(sequences),
                        )
                    finally:
                        await self._delete_run(run_id)

    async def test_fake_or_missing_reader_manifest_blocks_public_read(
        self,
    ) -> None:
        run_id = f"missing-reader-manifest-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        fake_sha256 = "f" * 64
        receipt = build_publication_receipt(
            run_id=run_id,
            report_json=report_json,
            analysis_markdown=markdown,
            reader_copy_manifest_artifact_key=(
                IMMUTABLE_READER_COPY_PREFIX + fake_sha256
            ),
            reader_copy_manifest_sha256=fake_sha256,
            illustration_asset_receipts=[],
        )
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="missing-manifest.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="publication",
                    artifact_key=(
                        PUBLICATION_RECEIPT_PREFIX + receipt["receipt_sha256"]
                    ),
                    status="completed",
                    prompt_version=PUBLICATION_RECEIPT_VERSION,
                    output_json=receipt,
                )
            )
            await session.commit()
        try:
            response = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 409)
        finally:
            await self._delete_run(run_id)

    async def test_immutable_reader_manifest_survives_constant_key_overwrite_but_not_tamper(
        self,
    ) -> None:
        run_id = f"immutable-reader-manifest-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        immutable_key = IMMUTABLE_READER_COPY_PREFIX + manifest["manifest_sha256"]
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="immutable-manifest.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.flush()
            await stage_publication_receipt(
                session,
                run_id=run_id,
                report_json=report_json,
                analysis_markdown=markdown,
                reader_copy_manifest=manifest,
            )
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="report",
                    artifact_key="reader_copy_manifest",
                    status="completed",
                    prompt_version="overwritten-later",
                    output_json={"decision": "block"},
                )
            )
            await session.commit()
        try:
            unaffected = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(unaffected.status_code, 200)
            async with SessionLocal() as session:
                immutable = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.artifact_key == immutable_key,
                        )
                    )
                ).scalar_one()
                immutable.output_json = {
                    **manifest,
                    "decision": "block",
                }
                await session.commit()
            tampered = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(tampered.status_code, 409)
        finally:
            await self._delete_run(run_id)

    async def test_trusted_archived_reader_contract_survives_current_version_bump(
        self,
    ) -> None:
        run_id = f"archived-reader-contract-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="archived-reader.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.flush()
            await stage_publication_receipt(
                session,
                run_id=run_id,
                report_json=report_json,
                analysis_markdown=markdown,
                reader_copy_manifest=manifest,
            )
            await session.commit()

        next_policy = {
            **LIVE_RUSSIAN_POLICY_MANIFEST.as_dict(),
            "version": "live-russian-next",
            "sha256": "1" * 64,
        }
        next_registry = {
            **READER_COPY_REGISTRY_MANIFEST.as_dict(),
            "version": "aiv-reader-copy-registry-ru-v2",
            "file_sha256": "2" * 64,
            "document_sha256": "3" * 64,
            "live_russian_policy": next_policy,
        }
        try:
            with (
                patch(
                    "app.services.publication_contract.READER_COPY_MANIFEST_VERSION",
                    "aiv-reader-copy-manifest-v6",
                ),
                patch(
                    "app.services.publication_contract.LIVE_RUSSIAN_POLICY_MANIFEST",
                    SimpleNamespace(
                        version=next_policy["version"],
                        sha256=next_policy["sha256"],
                        as_dict=lambda: next_policy,
                    ),
                ),
                patch(
                    "app.services.publication_contract.READER_COPY_REGISTRY_MANIFEST",
                    SimpleNamespace(as_dict=lambda: next_registry),
                ),
                patch(
                    "app.services.report_editor.REPORT_EDITOR_HARNESS_VERSION",
                    "aiv-report-editor-lossless-v7",
                ),
                patch(
                    "app.services.report_editor.REPORT_EDITOR_POLICY_VERSION",
                    "aiv-ru-editorial-policy-v5",
                ),
                patch(
                    "app.services.report_editor.REPORT_EDITOR_BOUNDARY_VERSION",
                    "aiv-editor-code-owned-boundary-v2",
                ),
            ):
                response = await self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(response.status_code, 200)

                async with SessionLocal() as session:
                    with self.assertRaises(PublicationContractError) as caught:
                        await stage_publication_receipt(
                            session,
                            run_id=run_id,
                            report_json=report_json,
                            analysis_markdown=markdown,
                            reader_copy_manifest=manifest,
                        )
                    self.assertIn(
                        "editorial_cache_invalid",
                        str(caught.exception),
                    )
                    await session.rollback()
        finally:
            await self._delete_run(run_id)

    async def test_unknown_resealed_historical_registry_cannot_be_read(self) -> None:
        run_id = f"unknown-reader-contract-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        unknown_manifest = copy.deepcopy(manifest)
        unknown_manifest["code_owned_copy_registry"]["version"] = (
            "aiv-reader-copy-registry-unknown"
        )
        unknown_core = {
            key: value
            for key, value in unknown_manifest.items()
            if key != "manifest_sha256"
        }
        unknown_manifest["manifest_sha256"] = stable_json_sha256(unknown_core)
        manifest_sha256 = unknown_manifest["manifest_sha256"]
        manifest_key = IMMUTABLE_READER_COPY_PREFIX + manifest_sha256
        receipt = build_publication_receipt(
            run_id=run_id,
            report_json=report_json,
            analysis_markdown=markdown,
            reader_copy_manifest_artifact_key=manifest_key,
            reader_copy_manifest_sha256=manifest_sha256,
            illustration_asset_receipts=[],
        )
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="unknown-reader.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            session.add_all(
                [
                    RunArtifact(
                        run_id=run_id,
                        stage_key="publication",
                        artifact_key=manifest_key,
                        status="completed",
                        model=None,
                        prompt_version=unknown_manifest["version"],
                        input_json={
                            "report_json_sha256": stable_json_sha256(report_json),
                            "analysis_markdown_sha256": hashlib.sha256(
                                markdown.encode("utf-8")
                            ).hexdigest(),
                            "site_preview_asset_receipt_sha256": None,
                        },
                        output_json=unknown_manifest,
                    ),
                    RunArtifact(
                        run_id=run_id,
                        stage_key="publication",
                        artifact_key=(
                            PUBLICATION_RECEIPT_PREFIX + receipt["receipt_sha256"]
                        ),
                        status="completed",
                        model=None,
                        prompt_version=PUBLICATION_RECEIPT_VERSION,
                        input_json={
                            "snapshot_digest": receipt["snapshot_digest"],
                            "reader_copy_manifest_artifact_key": manifest_key,
                            "reader_copy_manifest_sha256": manifest_sha256,
                            "site_preview_asset_receipt_sha256": None,
                        },
                        output_json=receipt,
                    ),
                ]
            )
            await session.commit()
        try:
            response = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 409)
        finally:
            await self._delete_run(run_id)

    async def test_new_receipt_rejects_report_or_markdown_field_tamper(
        self,
    ) -> None:
        run_id = f"snapshot-field-tamper-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="snapshot-tamper.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.flush()
            await stage_publication_receipt(
                session,
                run_id=run_id,
                report_json=report_json,
                analysis_markdown=markdown,
                reader_copy_manifest=manifest,
            )
            await session.commit()
        try:
            async with SessionLocal() as session:
                run = await session.get(Run, run_id)
                run.analysis_markdown = "# Подмена"
                await session.commit()
            markdown_tamper = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(markdown_tamper.status_code, 409)

            async with SessionLocal() as session:
                run = await session.get(Run, run_id)
                run.analysis_markdown = markdown
                run.report_json = {
                    "narrative": {"headline": "Подмена"},
                    "illustrations": [],
                }
                await session.commit()
            report_tamper = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(report_tamper.status_code, 409)
        finally:
            await self._delete_run(run_id)

    async def test_stage_rejects_fake_reader_manifest_self_digest(self) -> None:
        run_id = f"fake-reader-manifest-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        manifest["manifest_sha256"] = "0" * 64
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="fake-manifest.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.flush()
            with self.assertRaises(PublicationContractError):
                await stage_publication_receipt(
                    session,
                    run_id=run_id,
                    report_json=report_json,
                    analysis_markdown=markdown,
                    reader_copy_manifest=manifest,
                )
            await session.rollback()

    async def test_stage_rejects_resealed_stale_reader_policy(self) -> None:
        report_json = {
            "narrative": {"headline": "Отчёт"},
            "illustrations": [],
        }
        markdown = "# Отчёт"
        for field in ("version", "canonical_policy", "code_owned_copy_registry"):
            with self.subTest(field=field):
                manifest = await _reader_copy_manifest(report_json, markdown, [])
                if field == "version":
                    manifest["version"] = "aiv-reader-copy-manifest-stale"
                elif field == "canonical_policy":
                    manifest["canonical_policy"] = {
                        **LIVE_RUSSIAN_POLICY_MANIFEST.as_dict(),
                        "version": "stale-policy",
                    }
                else:
                    manifest["code_owned_copy_registry"] = {
                        **READER_COPY_REGISTRY_MANIFEST.as_dict(),
                        "version": "stale-registry",
                    }
                core = {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
                manifest["manifest_sha256"] = stable_json_sha256(core)
                async with SessionLocal() as session:
                    with self.assertRaises(PublicationContractError):
                        await stage_publication_receipt(
                            session,
                            run_id=f"stale-reader-policy-{uuid.uuid4()}",
                            report_json=report_json,
                            analysis_markdown=markdown,
                            reader_copy_manifest=manifest,
                        )
                    await session.rollback()

    async def test_stage_revalidates_editorial_cache_from_frozen_inputs(
        self,
    ) -> None:
        report_json = {
            "narrative": {"headline": "Отчёт"},
            "illustrations": [],
        }
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        receipt = manifest["editorial_receipts"]["final_report"]
        proof = receipt["cache_proof"]
        self.assertTrue(proof["audit"]["quality_complete"])
        # Keep the old, correctly self-sealed audit and all positive receipt
        # flags, but swap the frozen source and consistently reseal every outer
        # digest.  Only a fresh validate_editorial_cache call can detect that
        # the audit no longer belongs to these source bytes.
        proof["source"]["illustrations"][0]["caption"] = "Подменённый исходный текст."
        proof["source_sha256"] = stable_json_sha256(proof["source"])
        proof_core = {
            key: value for key, value in proof.items() if key != "proof_sha256"
        }
        proof["proof_sha256"] = stable_json_sha256(proof_core)
        receipt["cache_proof_sha256"] = proof["proof_sha256"]
        manifest_core = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = stable_json_sha256(manifest_core)
        async with SessionLocal() as session:
            with self.assertRaises(PublicationContractError):
                await stage_publication_receipt(
                    session,
                    run_id=f"invalid-editorial-cache-{uuid.uuid4()}",
                    report_json=report_json,
                    analysis_markdown=markdown,
                    reader_copy_manifest=manifest,
                )
            await session.rollback()

    async def test_zero_illustrations_require_explicit_publication_policy(
        self,
    ) -> None:
        run_id = f"zero-policy-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Отчёт"}, "illustrations": []}
        markdown = "# Отчёт"
        manifest = await _reader_copy_manifest(report_json, markdown, [])
        manifest["editorial_receipts"]["illustration_assets"]["publication_policy"].pop(
            "zero_assets_allowed"
        )
        core = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = stable_json_sha256(core)
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="zero-policy.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.flush()
            with self.assertRaises(PublicationContractError):
                await stage_publication_receipt(
                    session,
                    run_id=run_id,
                    report_json=report_json,
                    analysis_markdown=markdown,
                    reader_copy_manifest=manifest,
                )
            await session.rollback()

    async def test_operator_replacement_is_atomic_and_compare_and_swap_guarded(
        self,
    ) -> None:
        run_id = f"operator-publication-{uuid.uuid4()}"
        old_report = {"narrative": {"headline": "Было"}, "illustrations": []}
        old_markdown = "# Было"
        old_manifest = await _reader_copy_manifest(old_report, old_markdown, [])
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="operator.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=old_markdown,
                    report_json=old_report,
                )
            )
            await session.flush()
            await stage_publication_receipt(
                session,
                run_id=run_id,
                report_json=old_report,
                analysis_markdown=old_markdown,
                reader_copy_manifest=old_manifest,
            )
            await session.commit()
        expected = publication_snapshot_digest(
            publication_snapshot(
                report_json=old_report,
                analysis_markdown=old_markdown,
            )
        )
        new_report = {"narrative": {"headline": "Стало"}, "illustrations": []}
        new_markdown = "# Стало"
        new_manifest = await _reader_copy_manifest(new_report, new_markdown, [])
        try:
            receipt = await replace_completed_publication(
                run_id=run_id,
                expected_snapshot_digest=expected,
                report_json=new_report,
                analysis_markdown=new_markdown,
                reader_copy_manifest=new_manifest,
            )
            self.assertEqual(
                receipt["snapshot"]["report_json_sha256"],
                stable_json_sha256(new_report),
            )
            published = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(published.status_code, 200)
            self.assertEqual(
                published.json()["report_json"]["narrative"]["headline"],
                "Стало",
            )

            with self.assertRaises(PublicationContractError):
                await replace_completed_publication(
                    run_id=run_id,
                    expected_snapshot_digest=expected,
                    report_json=old_report,
                    analysis_markdown=old_markdown,
                    reader_copy_manifest=old_manifest,
                )
            still_published = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(
                still_published.json()["report_json"]["narrative"]["headline"],
                "Стало",
            )
        finally:
            await self._delete_run(run_id)

    async def test_operator_replacement_rejects_legacy_baseline(self) -> None:
        run_id = f"legacy-operator-publication-{uuid.uuid4()}"
        report_json = {"narrative": {"headline": "Старый"}, "illustrations": []}
        markdown = "# Старый"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="legacy-operator.example",
                    status=RunStatus.completed,
                    config_json={},
                    analysis_markdown=markdown,
                    report_json=report_json,
                )
            )
            await session.commit()
        try:
            baseline = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(baseline.status_code, 200)
            expected = publication_snapshot_digest(
                publication_snapshot(
                    report_json=report_json,
                    analysis_markdown=markdown,
                )
            )
            with self.assertRaises(PublicationContractError):
                await replace_completed_publication(
                    run_id=run_id,
                    expected_snapshot_digest=expected,
                    report_json={
                        "narrative": {"headline": "Подмена"},
                        "illustrations": [],
                    },
                    analysis_markdown="# Подмена",
                    reader_copy_manifest=await _reader_copy_manifest(
                        {
                            "narrative": {"headline": "Подмена"},
                            "illustrations": [],
                        },
                        "# Подмена",
                        [],
                    ),
                )
        finally:
            await self._delete_run(run_id)

    async def test_reused_legacy_image_is_copied_to_content_addressed_path(
        self,
    ) -> None:
        run_id = f"content-address-reuse-{uuid.uuid4()}"
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        static_dir = Path(temporary_directory.name) / "static"
        generated_dir = static_dir / "generated"
        run_dir = generated_dir / run_id
        run_dir.mkdir(parents=True)
        image_content = b"legacy-image-bytes"
        old_path = run_dir / "01.png"
        old_path.write_bytes(image_content)
        old_url = f"/static/generated/{run_id}/01.png"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="reuse.example",
                    status=RunStatus.completed,
                    config_json={},
                )
            )
            session.add(
                ReportIllustration(
                    run_id=run_id,
                    sequence=1,
                    title="Старая схема",
                    caption="Старая подпись",
                    alt_text="Старое описание",
                    file_url=old_url,
                    generation_prompt="saved prompt",
                    model="test/image-model",
                    usage_json={},
                )
            )
            await session.commit()
        accepted_review = {
            "usable": True,
            "facts_grounded": True,
            "claim_readable": True,
            "scores": {"context_specificity": 5},
            "unsupported_assertions": [],
            "hard_blockers": [],
            "visible_text_problems": [],
        }
        with (
            patch("app.services.analyzer.STATIC_DIR", static_dir),
            patch("app.services.analyzer.GENERATED_DIR", generated_dir),
            patch(
                "app.services.analyzer._review_illustration",
                new=AsyncMock(return_value=accepted_review),
            ),
        ):
            published = await _revalidate_reused_illustration_assets(
                run_id,
                saved_illustrations=[
                    {
                        "sequence": 1,
                        "title": "Старая схема",
                        "caption": "Старая подпись",
                        "alt_text": "Старое описание",
                        "file_url": old_url,
                    }
                ],
                refreshed_concepts=[
                    {
                        "title": "Новая схема",
                        "caption": "Новая подпись",
                        "alt_text": "Новое описание",
                        "evidence_paths": [],
                    }
                ],
                public_report={"brand": {"name": "Example"}},
            )
        try:
            image_sha256 = hashlib.sha256(image_content).hexdigest()
            expected_url = f"/static/generated/{run_id}/01-{image_sha256}.png"
            self.assertEqual(published[0]["file_url"], expected_url)
            self.assertTrue((run_dir / f"01-{image_sha256}.png").is_file())
            self.assertTrue(old_path.is_file())
            async with SessionLocal() as session:
                row = (
                    await session.execute(
                        select(ReportIllustration).where(
                            ReportIllustration.run_id == run_id,
                            ReportIllustration.sequence == 1,
                        )
                    )
                ).scalar_one()
                self.assertEqual(row.file_url, expected_url)
        finally:
            await self._delete_run(run_id)


if __name__ == "__main__":
    unittest.main()
