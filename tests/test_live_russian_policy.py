from __future__ import annotations

import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from app.services.live_russian_policy import (
    LIVE_RUSSIAN_POLICY_MANIFEST,
    LIVE_RUSSIAN_POLICY_SHA256,
    LIVE_RUSSIAN_POLICY_VERSION,
    assert_live_russian_policy_integrity,
    build_live_russian_policy_prompt,
    has_blocking_copy_issues,
    lint_reader_copy_tree,
    lint_russian_copy,
    load_live_russian_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / LIVE_RUSSIAN_POLICY_MANIFEST.snapshot


class LiveRussianPolicyTests(unittest.TestCase):
    def test_snapshot_is_pinned_by_exact_sha256(self) -> None:
        payload = SNAPSHOT.read_bytes()

        self.assertEqual(sha256(payload).hexdigest(), LIVE_RUSSIAN_POLICY_SHA256)
        self.assertEqual(
            assert_live_russian_policy_integrity().as_dict(),
            LIVE_RUSSIAN_POLICY_MANIFEST.as_dict(),
        )
        self.assertIn("## 0.1. Правила служат читателю", payload.decode("utf-8"))

    def test_loader_does_not_depend_on_working_directory_or_yandex_disk(self) -> None:
        original_cwd = Path.cwd()
        load_live_russian_policy.cache_clear()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                policy = load_live_russian_policy()
        finally:
            os.chdir(original_cwd)
            load_live_russian_policy.cache_clear()

        self.assertIn("# Живой русский язык: промпт для LLM-агентов", policy)
        self.assertNotIn("/Users/", policy)

    def test_prompt_carries_version_genre_and_fact_safety_contract(self) -> None:
        prompt = build_live_russian_policy_prompt(context="report")

        self.assertIn(LIVE_RUSSIAN_POLICY_VERSION, prompt)
        self.assertIn(LIVE_RUSSIAN_POLICY_SHA256, prompt)
        self.assertIn("Жанр: аналитический отчёт", prompt)
        self.assertIn("Не меняй числа, имена, URL, роли", prompt)
        self.assertIn("не использовать длинные тире никогда", prompt)

    def test_linter_reports_only_explicit_defects_and_keeps_offsets(self) -> None:
        text = 'Таким образом, "этот сервис" — это не просто решение...'

        report = lint_russian_copy(text)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(
            {
                "summary_thus",
                "straight_quotes",
                "long_dash",
                "not_just",
                "three_dots",
            }.issubset(codes)
        )
        self.assertEqual(report.checked_characters, len(text))
        self.assertEqual(report.checked_fields, 1)
        self.assertTrue(report.blocking)
        self.assertTrue(has_blocking_copy_issues(report))

    def test_clear_analytical_sentence_passes_deterministic_lint(self) -> None:
        text = "Система сравнила 81 ответ модели с каталогом услуг Realweb."

        report = lint_russian_copy(text)

        self.assertEqual(report.issues, ())
        self.assertFalse(has_blocking_copy_issues(text))

    def test_tree_lint_records_reader_paths_and_classifies_raw_evidence(self) -> None:
        document = {
            "headline": "Ось начинается с нуля.",
            "summary": ["Система проверила 81 ответ."],
            "source_url": "https://example.test/?q=важно отметить",
            "evidence": {"quote": "Важно отметить, что источник говорит иначе."},
        }

        report = lint_reader_copy_tree(document)

        self.assertEqual(report.checked_fields, 2)
        self.assertEqual({issue.path for issue in report.issues}, {"$.headline"})
        self.assertEqual({issue.code for issue in report.issues}, {"obvious_zero_axis"})
        self.assertEqual(
            set(report.skipped_paths),
            {"$.source_url", "$.evidence"},
        )

    def test_tree_lint_excludes_literal_identity_by_path_not_field_name(self) -> None:
        document = {
            "published_report": {
                "brand": {"name": "Клиент — официальный"},
                "competitors": [{"name": "Конкурент — группа"}],
                "technical": {
                    "review": {
                        "findings": [
                            {"title": "Авторский вывод — требует правки"}
                        ]
                    }
                },
            },
            "illustrations": [
                {"title": "Авторская схема — требует правки"}
            ],
            "finding": {"name": "Авторское имя — требует правки"},
        }

        report = lint_reader_copy_tree(document)
        issue_paths = {issue.path for issue in report.issues}

        self.assertNotIn("$.published_report.brand.name", issue_paths)
        self.assertNotIn("$.published_report.competitors[0].name", issue_paths)
        self.assertIn(
            "$.published_report.technical.review.findings[0].title",
            issue_paths,
        )
        self.assertIn("$.illustrations[0].title", issue_paths)
        self.assertIn("$.finding.name", issue_paths)
        self.assertIn("$.published_report.brand.name", report.skipped_paths)

    def test_linter_scans_full_input_even_when_issue_output_is_bounded(self) -> None:
        text = ("Проверенный факт. " * 20_000) + "Финальная фраза — дефект."

        report = lint_russian_copy(text, max_issues=1)

        self.assertEqual(report.checked_characters, len(text))
        self.assertEqual(len(report.issues), 1)
        self.assertEqual(report.issues[0].code, "long_dash")
        self.assertGreater(report.issues[0].start, 300_000)
        self.assertEqual(report.omitted_issue_count, 0)

    def test_issue_output_limit_is_explicit(self) -> None:
        report = lint_russian_copy("— — —", max_issues=2)

        self.assertEqual(len(report.issues), 2)
        self.assertEqual(report.omitted_issue_count, 1)

    def test_hidden_issue_cannot_hide_blocking_state(self) -> None:
        report = lint_russian_copy(
            "Таким образом сначала идёт предупреждение, а потом — ошибка.",
            max_issues=1,
        )

        self.assertEqual([issue.code for issue in report.issues], ["summary_thus"])
        self.assertEqual(report.blocking_count, 1)
        self.assertTrue(report.blocking)
        self.assertEqual(report.omitted_issue_count, 1)

    def test_unknown_prompt_context_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Russian policy context"):
            build_live_russian_policy_prompt(context="marketing")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
