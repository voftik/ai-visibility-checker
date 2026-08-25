from __future__ import annotations

import copy
import unittest
from unittest.mock import AsyncMock, patch

from app.services.analyzer import (
    _edit_technical_review_language,
    _technical_editorial_shape_is_safe,
)

from app.services.report_editor import (
    REPORT_EDITOR_HARNESS_VERSION,
    REPORT_EDITOR_POLICY_VERSION,
    build_editorial_units,
    edit_report,
    reader_narrative_paths,
    technical_review_narrative_paths,
    validate_critic_result,
    validate_editor_result,
)


def _report(body: str = "Сайт доступен на 50% проверенных страниц.") -> dict:
    return {
        "headline": "Важно отметить, Acme виден в ответах",
        "headline_emphasis": ["Acme"],
        "verdict": "Acme назвали в 3 из 6 ответов.",
        "executive_summary": "Проверка относится к шести страницам.",
        "sections": [{"heading": "Что показала проверка", "body": body}],
        "actions": [
            {
                "priority": "now",
                "title": "Добавить описание",
                "why": "Моделям не хватает связи услуги с Acme.",
                "step": "Опубликовать страницу услуги.",
                "evidence": "Точная цитата остаётся неизменной.",
            }
        ],
        "limitations": ["Экспресс-снимок не описывает весь рынок."],
    }


def _technical_review() -> dict:
    return {
        "overall_conclusion": (
            "Ниже представлено: краулер получает HTML "
            "на 2 из 3 проверенных страниц."
        ),
        "render_conclusion": (
            "Сервер отдаёт основной текст без JavaScript."
        ),
        "findings": [
            {
                "severity": "important",
                "title": "Одна страница зависит от клиентского рендеринга",
                "evidence": "HTTP 200 — точный evidence остаётся неизменным.",
                "business_effect": (
                    "Краулер не получит основной текст этой страницы."
                ),
                "action": (
                    "Команда разработки перенесёт основной текст в HTML."
                ),
            }
        ],
        "limitations": ["Краулер проверил 3 страницы сайта."],
    }


class ReportEditorUnitTests(unittest.TestCase):
    def test_technical_editorial_shape_allows_only_narrative_changes(self) -> None:
        source = _technical_review()
        edited = copy.deepcopy(source)
        edited["overall_conclusion"] = "Краулер получает HTML на двух страницах."
        edited["render_conclusion"] = "Сервер передаёт основной текст в HTML."
        edited["findings"][0]["title"] = "Одна страница требует JavaScript"
        edited["findings"][0]["business_effect"] = (
            "Без JavaScript краулер пропустит текст этой страницы."
        )
        edited["findings"][0]["action"] = (
            "Команда разработки добавит основной текст в серверный HTML."
        )
        edited["limitations"][0] = "Проверка охватила три страницы сайта."

        self.assertTrue(_technical_editorial_shape_is_safe(source, edited))

        evidence_drift = copy.deepcopy(edited)
        evidence_drift["findings"][0]["evidence"] = "Другое доказательство"
        self.assertFalse(
            _technical_editorial_shape_is_safe(source, evidence_drift)
        )

        severity_drift = copy.deepcopy(edited)
        severity_drift["findings"][0]["severity"] = "critical"
        self.assertFalse(
            _technical_editorial_shape_is_safe(source, severity_drift)
        )

        structural_drift = copy.deepcopy(edited)
        structural_drift["findings"].append(copy.deepcopy(edited["findings"][0]))
        self.assertFalse(
            _technical_editorial_shape_is_safe(source, structural_drift)
        )

    def test_only_reader_prose_is_selected(self) -> None:
        paths = reader_narrative_paths(_report())
        self.assertIn("/headline", paths)
        self.assertIn("/sections/0/body", paths)
        self.assertIn("/actions/0/step", paths)
        self.assertNotIn("/actions/0/evidence", paths)
        self.assertNotIn("/limitations/0", paths)

    def test_technical_review_selects_prose_but_not_evidence_or_enum(self) -> None:
        review = _technical_review()

        paths = technical_review_narrative_paths(review)

        self.assertEqual(
            paths,
            [
                "/overall_conclusion",
                "/render_conclusion",
                "/findings/0/title",
                "/findings/0/business_effect",
                "/findings/0/action",
                "/limitations/0",
            ],
        )
        self.assertNotIn("/findings/0/evidence", paths)
        self.assertNotIn("/findings/0/severity", paths)

    def test_explicit_json_pointer_paths_drive_the_lossless_manifest(self) -> None:
        review = _technical_review()
        paths = technical_review_narrative_paths(review)

        units, manifest = build_editorial_units(
            review,
            prose_paths=paths,
            target_chars=256,
        )

        self.assertEqual(manifest["version"], REPORT_EDITOR_HARNESS_VERSION)
        self.assertEqual(manifest["path_selection"], "explicit_json_pointer")
        self.assertEqual(manifest["prose_paths"], paths)
        self.assertEqual({item.path for item in units}, set(paths))
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            build_editorial_units(
                review,
                prose_paths=["/findings/0/missing"],
            )

    def test_long_report_has_lossless_unbounded_unit_manifest(self) -> None:
        tail = "TAIL-EDITOR-41"
        report = _report(("Подтверждённый факт. " * 3_000) + tail)
        units, manifest = build_editorial_units(report, target_chars=512)
        body_units = [item for item in units if item.path == "/sections/0/body"]
        self.assertGreater(len(body_units), 100)
        self.assertEqual("".join(item.source_text for item in body_units), report["sections"][0]["body"])
        self.assertTrue(body_units[-1].source_text.endswith(tail))
        self.assertTrue(manifest["coverage_complete"])
        self.assertEqual(manifest["unit_count"], len(units))

    def test_validator_rejects_changed_metric_and_url(self) -> None:
        report = _report("Acme: 50% на https://acme.example/a.")
        units, _manifest = build_editorial_units(
            report,
            protected_terms=["Acme"],
        )
        unit = next(item for item in units if item.path == "/sections/0/body")
        result = {
            "source_unit_id": unit.unit_id,
            "source_sha256": unit.source_sha256,
            "edited_text": "Acme: 60% на https://acme.example/b.",
            "claim_receipts": [
                {
                    "claim_sha256": item["claim_sha256"],
                    "preserved": True,
                    "target_excerpt": "Acme: 60% на https://acme.example/b.",
                    "note": "",
                }
                for item in unit.claims
            ],
            "new_claims": [],
        }
        errors = validate_editor_result(unit, result)
        self.assertIn("url_set_changed", errors)
        self.assertIn("number_or_unit_set_changed", errors)

    def test_validator_rejects_machine_like_russian(self) -> None:
        body = (
            "Важно отметить: данные анализируются. "
            "Быстро, надёжно, удобно. "
            "72% уже используют сервис — это очевидно. "
            "Это не просто проверка, а новый уровень."
        )
        report = _report(body)
        units, _manifest = build_editorial_units(report)
        unit = next(item for item in units if item.path == "/sections/0/body")
        result = {
            "source_unit_id": unit.unit_id,
            "source_sha256": unit.source_sha256,
            "edited_text": body,
            "claim_receipts": [
                {
                    "claim_sha256": item["claim_sha256"],
                    "preserved": True,
                    "target_excerpt": body,
                    "note": "",
                }
                for item in unit.claims
            ],
            "new_claims": [],
        }

        errors = validate_editor_result(unit, result)

        self.assertIn("forbidden_editorial_boilerplate", errors)
        self.assertIn("avoidable_passive_voice", errors)
        self.assertIn("slogan_or_hollow_antithesis", errors)
        self.assertIn("mechanical_triad", errors)
        self.assertIn("number_carrier_missing", errors)
        self.assertIn("long_dash_forbidden", errors)
        self.assertEqual(REPORT_EDITOR_POLICY_VERSION, "aiv-ru-editorial-policy-v2")

    def test_critic_must_confirm_actor_number_carrier_and_natural_style(
        self,
    ) -> None:
        report = _report()
        units, _manifest = build_editorial_units(report)
        unit = next(item for item in units if item.path == "/sections/0/body")
        critic = {
            "verdict": "pass",
            "issues": ["У числа не назван носитель."],
            "claim_checks": [
                {
                    "claim_sha256": item["claim_sha256"],
                    "meaning_preserved": True,
                    "actor_preserved": True,
                    "scope_preserved": True,
                    "numbers_preserved": True,
                    "actor_or_mechanism_explicit": True,
                    "number_carrier_explicit": False,
                    "active_voice": True,
                    "no_slogan_or_meta": True,
                    "no_mechanical_triad": True,
                    "reason": "Носитель не назван.",
                }
                for item in unit.claims
            ],
            "new_claims": [],
        }

        errors = validate_critic_result(unit, critic)

        self.assertIn("critic_found_live_russian_defect", errors)
        self.assertIn("critic_reported_issues", errors)


class ReportEditorWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_technical_editorial_pass_returns_edited_review_losslessly(
        self,
    ) -> None:
        source = _technical_review()
        edited = copy.deepcopy(source)
        edited["overall_conclusion"] = (
            "Краулер получает основной HTML на двух из трёх страниц."
        )
        edited["findings"][0]["business_effect"] = (
            "Краулер пропустит текст одной страницы без JavaScript."
        )
        audit: dict = {}
        artifact_output = AsyncMock(return_value=None)
        save_artifact = AsyncMock()
        editor = AsyncMock(return_value=(edited, audit))

        with (
            patch(
                "app.services.analyzer._artifact_output",
                artifact_output,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                save_artifact,
            ),
            patch(
                "app.services.analyzer.edit_report",
                editor,
            ),
        ):
            result = await _edit_technical_review_language(
                "run-editorial",
                review=source,
                profile={
                    "brand_name": "Acme",
                    "brand_aliases": ["Acme Group"],
                    "products": ["Acme Cloud"],
                },
            )

        self.assertEqual(result, edited)
        self.assertEqual(
            result["findings"][0]["evidence"],
            source["findings"][0]["evidence"],
        )
        self.assertEqual(
            result["findings"][0]["severity"],
            source["findings"][0]["severity"],
        )
        self.assertEqual(
            editor.await_args.kwargs["prose_paths"],
            technical_review_narrative_paths(source),
        )
        self.assertEqual(
            editor.await_args.kwargs["protected_terms"],
            ["Acme", "Acme Group", "Acme Cloud"],
        )
        self.assertEqual(save_artifact.await_count, 2)
        completed_call = save_artifact.await_args_list[-1]
        self.assertEqual(completed_call.kwargs["status"], "completed")
        self.assertEqual(
            completed_call.kwargs["output_json"]["review"],
            edited,
        )
        self.assertEqual(
            completed_call.kwargs["output_json"]["audit"]["semantic_fallback"],
            {"used": False, "errors": []},
        )

    async def test_successful_edit_preserves_facts_and_clears_emphasis(self) -> None:
        source = _report()

        async def editor(payload: dict) -> dict:
            edited = str(payload["core_text"])
            if payload["path"] == "/headline":
                edited = "Acme виден в ответах"
            return {
                "source_unit_id": payload["source_unit_id"],
                "source_sha256": payload["source_sha256"],
                "edited_text": edited,
                "claim_receipts": [
                    {
                        "claim_sha256": item["claim_sha256"],
                        "preserved": True,
                        "target_excerpt": edited,
                        "note": "Смысл сохранён.",
                    }
                    for item in payload["source_claims"]
                ],
                "new_claims": [],
            }

        async def critic(payload: dict) -> dict:
            return {
                "verdict": "pass",
                "issues": [],
                "claim_checks": [
                    {
                        "claim_sha256": item["claim_sha256"],
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
                    for item in payload["source_claims"]
                ],
                "new_claims": [],
            }

        edited, audit = await edit_report(
            source,
            editor_call=editor,
            critic_call=critic,
            protected_terms=["Acme"],
        )
        self.assertEqual(edited["headline"], "Acme виден в ответах")
        self.assertEqual(edited["headline_emphasis"], [])
        self.assertEqual(edited["actions"][0]["evidence"], source["actions"][0]["evidence"])
        self.assertEqual(edited["limitations"], source["limitations"])
        self.assertTrue(audit["coverage_complete"])
        self.assertFalse(audit["fallback_units"])

    async def test_technical_review_edit_keeps_evidence_and_enum_exact(self) -> None:
        source = _technical_review()
        prose_paths = technical_review_narrative_paths(source)

        async def editor(payload: dict) -> dict:
            edited = str(payload["core_text"])
            if payload["path"] == "/overall_conclusion":
                edited = "Краулер получает HTML на 2 из 3 проверенных страниц."
            return {
                "source_unit_id": payload["source_unit_id"],
                "source_sha256": payload["source_sha256"],
                "edited_text": edited,
                "claim_receipts": [
                    {
                        "claim_sha256": item["claim_sha256"],
                        "preserved": True,
                        "target_excerpt": edited,
                        "note": "Смысл сохранён.",
                    }
                    for item in payload["source_claims"]
                ],
                "new_claims": [],
            }

        async def critic(payload: dict) -> dict:
            return {
                "verdict": "pass",
                "issues": [],
                "claim_checks": [
                    {
                        "claim_sha256": item["claim_sha256"],
                        "meaning_preserved": True,
                        "actor_preserved": True,
                        "scope_preserved": True,
                        "numbers_preserved": True,
                        "actor_or_mechanism_explicit": True,
                        "number_carrier_explicit": True,
                        "active_voice": True,
                        "no_slogan_or_meta": True,
                        "no_mechanical_triad": True,
                        "reason": "Смысл и стиль прошли проверку.",
                    }
                    for item in payload["source_claims"]
                ],
                "new_claims": [],
            }

        edited, audit = await edit_report(
            source,
            editor_call=editor,
            critic_call=critic,
            prose_paths=prose_paths,
        )

        self.assertEqual(
            edited["overall_conclusion"],
            "Краулер получает HTML на 2 из 3 проверенных страниц.",
        )
        self.assertEqual(
            edited["findings"][0]["evidence"],
            source["findings"][0]["evidence"],
        )
        self.assertEqual(
            edited["findings"][0]["severity"],
            source["findings"][0]["severity"],
        )
        self.assertNotIn("headline_emphasis", edited)
        self.assertEqual(
            audit["source_manifest"]["path_selection"],
            "explicit_json_pointer",
        )
        self.assertFalse(audit["fallback_units"])

    async def test_changed_number_falls_back_to_exact_source_unit(self) -> None:
        source = _report()

        async def bad_editor(payload: dict) -> dict:
            edited = str(payload["core_text"]).replace("50%", "90%").replace("3 из 6", "6 из 6")
            return {
                "source_unit_id": payload["source_unit_id"],
                "source_sha256": payload["source_sha256"],
                "edited_text": edited,
                "claim_receipts": [
                    {
                        "claim_sha256": item["claim_sha256"],
                        "preserved": True,
                        "target_excerpt": edited,
                        "note": "",
                    }
                    for item in payload["source_claims"]
                ],
                "new_claims": [],
            }

        async def critic(_payload: dict) -> dict:
            raise AssertionError("numeric drift must fail before critic")

        edited, audit = await edit_report(
            source,
            editor_call=bad_editor,
            critic_call=critic,
            protected_terms=["Acme"],
        )
        self.assertEqual(edited["verdict"], source["verdict"])
        self.assertEqual(edited["sections"][0]["body"], source["sections"][0]["body"])
        self.assertTrue(audit["fallback_units"])


if __name__ == "__main__":
    unittest.main()
