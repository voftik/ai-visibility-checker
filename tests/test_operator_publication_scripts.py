from __future__ import annotations

import ast
import copy
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch

from app.models import RunStatus
from app.services.publication_contract import (
    publication_snapshot,
    publication_snapshot_digest,
)
from scripts import backfill_site_preview, rebuild_from_saved_annotations
from scripts import rebuild_visuals


class _Result:
    def __init__(self, value: object) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object:
        return self.value

    def scalars(self) -> _Result:
        return self

    def first(self) -> object:
        return self.value


class _Session:
    def __init__(self, *values: object) -> None:
        self._values = list(values)

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def execute(self, _query: object) -> _Result:
        if not self._values:
            raise AssertionError("Unexpected database query")
        return _Result(self._values.pop(0))


def _run(report: dict, *, markdown: str = "# Report") -> SimpleNamespace:
    return SimpleNamespace(
        id="11111111-1111-4111-8111-111111111111",
        domain="example.com",
        status=RunStatus.completed,
        report_json=copy.deepcopy(report),
        analysis_markdown=markdown,
    )


class OperatorPublicationScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_rebuild_publishes_exact_candidate_via_cas(self) -> None:
        original = {
            "brand": {"name": "Example"},
            "technical": {"score": 80},
            "narrative": {"headline": "Before"},
            "illustrations": [{"sequence": 1, "file_url": "/old.png"}],
            "site_preview": {"file_url": "/preview.png"},
        }
        run = _run(original)
        raw_concepts = [{"role": "technical_access"}]
        edited_concepts = [{"role": "technical_access", "title": "Edited"}]
        generated = [
            {
                "sequence": 1,
                "title": "Edited",
                "caption": "Caption",
                "alt_text": "Alt",
                "file_url": "/static/generated/run/01-hash.png",
            }
        ]
        public_report = {
            "brand": {"name": "Example"},
            "technical": {"score": 80},
        }
        public_report_sha256 = rebuild_visuals.analyzer._stable_json_sha256(
            public_report
        )
        concept_artifact = SimpleNamespace(
            input_json={"report_data": public_report},
            output_json={"illustrations": raw_concepts},
        )
        copy_artifact = SimpleNamespace(
            input_json={"public_report_sha256": public_report_sha256},
            output_json={"copy": {"illustrations": []}},
        )
        final_report = {"headline": "Before", "sections": [], "actions": []}
        original["narrative"] = {
            "headline": "Before",
            "headline_emphasis": [],
            "verdict": None,
            "executive_summary": None,
            "actions": [],
        }
        run.report_json = copy.deepcopy(original)
        run.analysis_markdown = rebuild_visuals.analyzer._render_markdown(
            final_report
        )
        final_artifact = SimpleNamespace(
            input_json={"public_report_sha256": public_report_sha256},
            output_json={"report": final_report},
        )
        session = _Session(
            run,
            concept_artifact,
            copy_artifact,
            final_artifact,
        )
        manifest = {"manifest_sha256": "reader-manifest"}

        with (
            patch.object(rebuild_visuals, "SessionLocal", return_value=session),
            patch.object(
                rebuild_visuals,
                "ensure_publication_contract",
                new=AsyncMock(return_value={"legacy_baseline": False}),
            ),
            patch.object(
                rebuild_visuals.analyzer,
                "_merge_illustration_copy",
                return_value=edited_concepts,
            ),
            patch.object(
                rebuild_visuals.analyzer,
                "_generate_illustrations",
                new=AsyncMock(return_value=generated),
            ) as generate,
            patch.object(
                rebuild_visuals.analyzer,
                "_save_reader_copy_manifest",
                new=AsyncMock(return_value=manifest),
            ) as save_manifest,
            patch.object(
                rebuild_visuals,
                "replace_completed_publication",
                new=AsyncMock(return_value={}),
            ) as replace,
        ):
            result = await rebuild_visuals.rebuild_visuals(run.id)

        self.assertEqual(result, generated)
        self.assertEqual(run.report_json, original)
        generate.assert_awaited_once_with(
            run.id,
            brand_name="Example",
            concepts=edited_concepts,
            public_report=public_report,
        )
        expected_report = {**copy.deepcopy(original), "illustrations": generated}
        save_manifest.assert_awaited_once_with(
            run.id,
            final_report=final_report,
            public_report=public_report,
            illustrations=generated,
            analysis_markdown=run.analysis_markdown,
            report_json=expected_report,
        )
        replace.assert_awaited_once_with(
            run_id=run.id,
            expected_snapshot_digest=publication_snapshot_digest(
                publication_snapshot(
                    report_json=original,
                    analysis_markdown=run.analysis_markdown,
                )
            ),
            report_json=expected_report,
            analysis_markdown=run.analysis_markdown,
            reader_copy_manifest=manifest,
        )

    async def test_visual_rebuild_rejects_legacy_before_generation(self) -> None:
        run = _run({"brand": {"name": "Example"}})
        generate = AsyncMock()
        with (
            patch.object(
                rebuild_visuals,
                "SessionLocal",
                return_value=_Session(run),
            ),
            patch.object(
                rebuild_visuals,
                "ensure_publication_contract",
                new=AsyncMock(return_value={"legacy_baseline": True}),
            ),
            patch.object(
                rebuild_visuals.analyzer,
                "_generate_illustrations",
                new=generate,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Legacy report"):
                await rebuild_visuals.rebuild_visuals(run.id)
        generate.assert_not_awaited()

    async def test_preview_backfill_publishes_exact_candidate_via_cas(self) -> None:
        original = {
            "brand": {"name": "Example"},
            "technical": {"score": 80},
            "narrative": {"headline": "Before"},
            "illustrations": [],
            "site_preview": {"file_url": "/old-preview.png"},
        }
        run = _run(original)
        final_report = {"headline": "Before", "sections": [], "actions": []}
        original["narrative"] = {
            "headline": "Before",
            "headline_emphasis": [],
            "verdict": None,
            "executive_summary": None,
            "actions": [],
        }
        run.report_json = copy.deepcopy(original)
        run.analysis_markdown = backfill_site_preview.analyzer._render_markdown(
            final_report
        )
        public_report = {
            "brand": {"name": "Example"},
            "technical": {"score": 80},
        }
        final_artifact = SimpleNamespace(
            input_json={
                "public_report_sha256": (
                    backfill_site_preview.analyzer._stable_json_sha256(
                        public_report
                    )
                )
            },
            output_json={"report": final_report},
        )
        page = SimpleNamespace(url="https://example.com/")
        session = _Session(run, final_artifact, page)
        preview = {
            "file_url": "/static/generated/run/site-preview-hash.webp",
            "source_url": "https://example.com/",
        }
        manifest = {"manifest_sha256": "reader-manifest"}

        with (
            patch.object(
                backfill_site_preview,
                "SessionLocal",
                return_value=session,
            ),
            patch.object(
                backfill_site_preview,
                "ensure_publication_contract",
                new=AsyncMock(return_value={"legacy_baseline": False}),
            ),
            patch.object(
                backfill_site_preview,
                "capture_site_preview",
                new=AsyncMock(return_value=preview),
            ) as capture,
            patch.object(
                backfill_site_preview.analyzer,
                "_save_reader_copy_manifest",
                new=AsyncMock(return_value=manifest),
            ) as save_manifest,
            patch.object(
                backfill_site_preview,
                "replace_completed_publication",
                new=AsyncMock(return_value={}),
            ) as replace,
        ):
            result = await backfill_site_preview.backfill(run.id)

        self.assertEqual(result, preview["file_url"])
        self.assertEqual(run.report_json, original)
        capture.assert_awaited_once_with(
            run.id,
            domain="example.com",
            source_url="https://example.com/",
            validate_url=ANY,
        )
        expected_report = {**copy.deepcopy(original), "site_preview": preview}
        save_manifest.assert_awaited_once_with(
            run.id,
            final_report=final_report,
            public_report=public_report,
            illustrations=[],
            analysis_markdown=run.analysis_markdown,
            report_json=expected_report,
        )
        replace.assert_awaited_once_with(
            run_id=run.id,
            expected_snapshot_digest=publication_snapshot_digest(
                publication_snapshot(
                    report_json=original,
                    analysis_markdown=run.analysis_markdown,
                )
            ),
            report_json=expected_report,
            analysis_markdown=run.analysis_markdown,
            reader_copy_manifest=manifest,
        )

    async def test_saved_annotation_rebuild_rejects_legacy_receipt(self) -> None:
        run = _run({"brand": {"name": "Example"}})
        with (
            patch.object(
                rebuild_from_saved_annotations,
                "SessionLocal",
                return_value=_Session(run),
            ),
            patch.object(
                rebuild_from_saved_annotations,
                "ensure_publication_contract",
                new=AsyncMock(return_value={"legacy_baseline": True}),
            ),
        ):
            with self.assertRaisesRegex(
                rebuild_from_saved_annotations.RebuildGuardError,
                "Legacy-отчёт",
            ):
                await rebuild_from_saved_annotations._validate_saved_inputs(run.id)

    def test_operator_scripts_have_no_direct_public_snapshot_writes(self) -> None:
        for module in (
            rebuild_from_saved_annotations,
            rebuild_visuals,
            backfill_site_preview,
        ):
            with self.subTest(module=module.__name__):
                source_path = Path(inspect.getsourcefile(module) or "")
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                direct_targets: list[str] = []
                commits = 0
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                        for target in targets:
                            if isinstance(target, ast.Attribute) and target.attr in {
                                "report_json",
                                "analysis_markdown",
                            }:
                                direct_targets.append(target.attr)
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "commit"
                    ):
                        commits += 1
                self.assertEqual(direct_targets, [])
                self.assertEqual(commits, 0)
                source = source_path.read_text(encoding="utf-8")
                self.assertIn("allow_legacy_baseline=False", source)
                self.assertIn("_save_reader_copy_manifest", source)
                self.assertIn("replace_completed_publication", source)


if __name__ == "__main__":
    unittest.main()
