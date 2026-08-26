from __future__ import annotations

import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from app.services.analyzer import _reader_copy_document, _render_markdown
from app.services.live_russian_policy import lint_reader_copy_tree
from app.services.reader_copy_registry import (
    READER_COPY_REGISTRY_FILE_SHA256,
    READER_COPY_REGISTRY_MANIFEST,
    READER_COPY_REGISTRY_VERSION,
    assert_reader_copy_registry_integrity,
    load_reader_copy_registry,
    reader_copy_registry_document,
    reader_copy_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ASSET = REPO_ROOT / READER_COPY_REGISTRY_MANIFEST.as_dict()["asset"]
FRONTEND = REPO_ROOT / "static" / "index.html"


class ReaderCopyRegistryTests(unittest.TestCase):
    def tearDown(self) -> None:
        load_reader_copy_registry.cache_clear()

    def test_browser_asset_is_the_exact_linted_backend_registry(self) -> None:
        payload = REGISTRY_ASSET.read_bytes()
        document = reader_copy_registry_document()

        self.assertEqual(sha256(payload).hexdigest(), READER_COPY_REGISTRY_FILE_SHA256)
        self.assertEqual(document["version"], READER_COPY_REGISTRY_VERSION)
        self.assertEqual(
            assert_reader_copy_registry_integrity().as_dict(),
            READER_COPY_REGISTRY_MANIFEST.as_dict(),
        )
        lint = lint_reader_copy_tree(
            document["copy"],
            excluded_subtrees=frozenset(),
            excluded_keys=frozenset(),
        )
        self.assertEqual(lint.issues, ())
        self.assertEqual(lint.skipped_paths, ())
        self.assertEqual(lint.omitted_issue_count, 0)

    def test_loader_does_not_depend_on_working_directory(self) -> None:
        original_cwd = Path.cwd()
        load_reader_copy_registry.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                document = load_reader_copy_registry()
        finally:
            os.chdir(original_cwd)

        self.assertEqual(document["language"], "ru")
        self.assertEqual(
            reader_copy_value("technical_ui.finding_basis_missing"),
            "Нет подтверждения",
        )

    def test_file_drift_fails_before_copy_can_reach_a_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tampered = Path(temp_dir) / REGISTRY_ASSET.name
            tampered.write_bytes(
                REGISTRY_ASSET.read_bytes().replace(
                    "Нет подтверждения".encode(),
                    "Не указано".encode(),
                )
            )
            load_reader_copy_registry.cache_clear()
            with patch(
                "app.services.reader_copy_registry._registry_path",
                return_value=tampered,
            ), self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                load_reader_copy_registry()

    def test_frontend_consumes_registry_for_technical_fallback_copy(self) -> None:
        source = FRONTEND.read_text(encoding="utf-8")

        self.assertIn(
            'src="/static/reader-copy-registry.ru.v2026-08-26.js"',
            source,
        )
        self.assertIn(
            'integrity="sha256-vofwWlBVDFfMnBuJrbEZRwFOXbJvfqUSemMERTgAdDU="',
            source,
        )
        for path in (
            "technical_ui.pages_missing",
            "technical_ui.findings_empty",
            "technical_ui.finding_title_missing",
            "technical_ui.finding_basis_missing",
            "technical_ui.finding_effect_missing",
            "technical_ui.finding_action_missing",
            "technical_ui.matrix_families_missing",
            "technical_ui.matrix_coverage_missing",
            "technical_ui.matrix_pages_missing",
            "technical_ui.matrix_auth_form_found",
            "technical_ui.matrix_auth_wall_absent",
            "technical_ui.matrix_schema_types_checked",
            "technical_ui.matrix_url_list_empty",
            "technical_ui.access_matrix_missing",
            "technical_ui.limitations_empty",
            "technical_ui.methodology_fallback",
        ):
            self.assertIn(f'readerCopy("{path}")', source)
        self.assertNotIn('finding?.evidence || "Не указано"', source)

    def test_markdown_and_manifest_document_consume_the_same_registry(self) -> None:
        report = {
            "headline": "Сайт доступен моделям",
            "verdict": "Сервер отдаёт основной текст.",
            "executive_summary": "Проверка охватила шесть страниц.",
            "sections": [],
            "actions": [
                {
                    "priority": "now",
                    "title": "Добавить разметку",
                    "why": "Сущности не связаны.",
                    "step": "Опубликовать Schema.org.",
                    "evidence": "Разметка не найдена.",
                }
            ],
            "limitations": ["Вывод относится к проверенным страницам."],
        }

        markdown = _render_markdown(report)
        document = _reader_copy_document(
            final_report=report,
            public_report={},
            illustrations=[],
        )

        self.assertIn("**Вердикт.**", markdown)
        self.assertIn("## Что изменить в первую очередь", markdown)
        self.assertIn("**Основание.** Разметка не найдена.", markdown)
        self.assertEqual(
            document["code_owned_copy_registry"],
            reader_copy_registry_document()["copy"],
        )

    def test_raw_evidence_exclusions_remain_path_aware(self) -> None:
        document = {
            "rendered_technical_basis_copy": "Факт — требует правки",
            "evidence": {"quote": "Источник — цитируем дословно"},
            "citations": [{"title": "Название — из источника"}],
        }

        lint = lint_reader_copy_tree(document)

        self.assertEqual(
            {issue.path for issue in lint.issues},
            {"$.rendered_technical_basis_copy"},
        )
        self.assertEqual(set(lint.skipped_paths), {"$.evidence", "$.citations"})


if __name__ == "__main__":
    unittest.main()
