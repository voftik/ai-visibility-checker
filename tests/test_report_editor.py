from __future__ import annotations

import copy
import unittest
from unittest.mock import AsyncMock, patch

from app.services.analyzer import (
    _edit_final_report_language,
    _edit_technical_review_language,
    _technical_editorial_shape_is_safe,
)

from app.services.report_editor import (
    REPORT_EDITOR_BOUNDARY_VERSION,
    REPORT_EDITOR_HARNESS_VERSION,
    REPORT_EDITOR_POLICY_VERSION,
    build_editorial_units,
    edit_report,
    illustration_copy_immutable_passthrough_paths,
    illustration_copy_narrative_paths,
    illustration_copy_rendered_string_paths,
    reader_immutable_passthrough_paths,
    reader_narrative_paths,
    reader_rendered_string_paths,
    seal_editorial_audit,
    technical_review_immutable_passthrough_paths,
    technical_review_narrative_paths,
    technical_review_rendered_string_paths,
    validate_critic_result,
    validate_editorial_cache,
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
        self.assertIn("/limitations/0", paths)
        self.assertNotIn("/actions/0/evidence", paths)

        immutable = reader_immutable_passthrough_paths(_report())
        self.assertIn("/headline_emphasis/0", immutable)
        self.assertIn("/actions/0/priority", immutable)
        self.assertIn("/actions/0/evidence", immutable)
        self.assertEqual(
            set(reader_rendered_string_paths(_report())),
            set(paths) | set(immutable),
        )

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
        immutable = technical_review_immutable_passthrough_paths(review)
        self.assertEqual(
            immutable,
            ["/findings/0/severity", "/findings/0/evidence"],
        )
        self.assertEqual(
            set(technical_review_rendered_string_paths(review)),
            set(paths) | set(immutable),
        )

    def test_illustration_copy_registers_every_rendered_string(self) -> None:
        document = {
            "illustrations": [
                {
                    "role": "technical_access",
                    "core_claim": "Сервер отдаёт основной текст краулеру.",
                    "title": "Как сайт читают ИИ-системы",
                    "caption": "Краулер получает основной текст в HTML.",
                    "alt_text": "Схема пути текста сайта к ИИ-системе.",
                    "evidence_paths": ["/technical/score"],
                },
                {
                    "role": "competitive_visibility",
                    "core_claim": "Бренд появляется рядом с альтернативами.",
                    "title": "Где появляется бренд",
                    "caption": "Модели сравнивают бренд с альтернативами.",
                    "alt_text": "Схема поля бренда и альтернатив.",
                    "evidence_paths": ["/discovery/parent/web/score"],
                },
            ]
        }

        paths = illustration_copy_narrative_paths(document)
        units, manifest = build_editorial_units(document, prose_paths=paths)

        self.assertEqual(manifest["document_kind"], "illustration_copy")
        self.assertTrue(manifest["coverage_complete"])
        self.assertEqual(
            set(illustration_copy_rendered_string_paths(document)),
            set(paths)
            | set(illustration_copy_immutable_passthrough_paths(document)),
        )
        self.assertEqual({unit.path for unit in units}, set(paths))

        unsupported = copy.deepcopy(document)
        unsupported["illustrations"][0]["hidden_copy"] = "Скрытый текст"
        with self.assertRaisesRegex(ValueError, "invalid editorial document shape"):
            build_editorial_units(unsupported)

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
        self.assertTrue(manifest["coverage_complete"])
        self.assertEqual(
            {item["path"] for item in manifest["immutable_passthrough"]},
            {"/findings/0/severity", "/findings/0/evidence"},
        )
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            build_editorial_units(
                review,
                prose_paths=["/findings/0/missing"],
            )

    def test_omitted_reader_path_fails_exact_coverage(self) -> None:
        review = _technical_review()
        paths = technical_review_narrative_paths(review)

        _units, manifest = build_editorial_units(
            review,
            prose_paths=[path for path in paths if path != "/limitations/0"],
        )

        self.assertFalse(manifest["coverage_complete"])
        self.assertEqual(manifest["missing_paths"], ["/limitations/0"])

    def test_partial_schema_is_not_admitted_as_an_editorial_document(self) -> None:
        for missing_key in ("headline_emphasis", "limitations"):
            with self.subTest(document="final_report", missing_key=missing_key):
                partial = _report()
                partial.pop(missing_key)
                with self.assertRaisesRegex(
                    ValueError,
                    "unsupported editorial document shape",
                ):
                    build_editorial_units(partial)

        partial_review = _technical_review()
        partial_review.pop("limitations")
        with self.assertRaisesRegex(
            ValueError,
            "unsupported editorial document shape",
        ):
            build_editorial_units(partial_review)

    def test_long_report_has_lossless_unbounded_unit_manifest(self) -> None:
        tail = "TAIL-EDITOR-41"
        report = _report(("Подтверждённый факт. " * 3_000) + tail)
        units, manifest = build_editorial_units(report, target_chars=512)
        body_units = [item for item in units if item.path == "/sections/0/body"]
        self.assertGreater(len(body_units), 100)
        self.assertEqual("".join(item.source_text for item in body_units), report["sections"][0]["body"])
        self.assertTrue(body_units[-1].source_text.endswith(tail))
        self.assertTrue(
            all(
                item.source_text
                == item.code_owned_prefix
                + item.editable_text
                + item.code_owned_suffix
                for item in body_units
            )
        )
        self.assertTrue(
            any(item.code_owned_suffix for item in body_units[:-1])
        )
        self.assertEqual(
            manifest["boundary_contract"]["version"],
            REPORT_EDITOR_BOUNDARY_VERSION,
        )
        self.assertEqual(
            REPORT_EDITOR_HARNESS_VERSION,
            "aiv-report-editor-lossless-v6",
        )
        self.assertTrue(manifest["coverage_complete"])
        self.assertEqual(manifest["unit_count"], len(units))

    def test_validator_rejects_model_owned_edge_whitespace(self) -> None:
        report = _report("Подтверждённый факт. " * 80)
        units, _manifest = build_editorial_units(report, target_chars=256)
        unit = next(
            item
            for item in units
            if item.path == "/sections/0/body" and item.code_owned_suffix
        )
        edited = unit.editable_text + " "
        candidate = {
            "source_unit_id": unit.unit_id,
            "source_sha256": unit.source_sha256,
            "edited_text": edited,
            "claim_receipts": [
                {
                    "claim_sha256": claim["claim_sha256"],
                    "preserved": True,
                    "target_excerpt": claim["source_excerpt"],
                    "note": "Смысл сохранён.",
                }
                for claim in unit.claims
            ],
            "new_claims": [],
        }

        self.assertIn(
            "code_owned_boundary_changed",
            validate_editor_result(unit, candidate),
        )

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
        self.assertEqual(REPORT_EDITOR_POLICY_VERSION, "aiv-ru-editorial-policy-v4")

    def test_validator_rejects_actor_value_and_url_swaps(self) -> None:
        report = _report(
            "OpenAI: 4 из 6, https://openai.example/fact. "
            "Gemini: 2 из 6, https://gemini.example/fact."
        )
        units, _manifest = build_editorial_units(
            report,
            protected_terms=["OpenAI", "Gemini"],
        )
        unit = next(item for item in units if item.path == "/sections/0/body")
        edited = (
            "OpenAI: 2 из 6, https://gemini.example/fact. "
            "Gemini: 4 из 6, https://openai.example/fact."
        )
        result = {
            "source_unit_id": unit.unit_id,
            "source_sha256": unit.source_sha256,
            "edited_text": edited,
            "claim_receipts": [
                {
                    "claim_sha256": item["claim_sha256"],
                    "preserved": True,
                    "target_excerpt": edited,
                    "note": "",
                }
                for item in unit.claims
            ],
            "new_claims": [],
        }

        errors = validate_editor_result(unit, result)

        self.assertNotIn("number_or_unit_set_changed", errors)
        self.assertNotIn("url_set_changed", errors)
        self.assertNotIn("protected_name_set_changed", errors)
        self.assertIn("actor_value_binding_changed", errors)
        self.assertIn("actor_url_binding_changed", errors)

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
    async def test_final_editor_cache_key_binds_current_semantic_evidence(
        self,
    ) -> None:
        artifact_output = AsyncMock(return_value=None)
        save_artifact = AsyncMock()

        async def unchanged(source: dict, **_kwargs) -> tuple[dict, dict]:
            return copy.deepcopy(source), {}

        with (
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch("app.services.analyzer.edit_report", side_effect=unchanged),
            patch("app.services.analyzer._save_artifact", save_artifact),
        ):
            for suffix in ("old", "new"):
                await _edit_final_report_language(
                    "run-semantic-editor-cache",
                    report=_report(),
                    public_report={"brand": {"name": "Acme"}},
                    selected_answer_context=[
                        {"answer_id": "answer-1", "answer_text": suffix}
                    ],
                    answer_selection_manifest={"digest": suffix},
                    semantic_evidence_document={"evidence": suffix},
                )

        first_input = artifact_output.await_args_list[0].kwargs["input_json"]
        second_input = artifact_output.await_args_list[1].kwargs["input_json"]
        for key in (
            "selected_answer_context_sha256",
            "answer_selection_manifest_sha256",
            "semantic_evidence_document_sha256",
        ):
            self.assertRegex(first_input[key], r"^[0-9a-f]{64}$")
            self.assertNotEqual(first_input[key], second_input[key])
        self.assertIn("semantic_gate_version", first_input)

    async def test_partial_final_preflight_returns_source_without_model_call(
        self,
    ) -> None:
        partial = _report()
        partial.pop("limitations")
        expected = copy.deepcopy(partial)
        expected["headline_emphasis"] = []
        artifact_output = AsyncMock()
        editor = AsyncMock()
        save_artifact = AsyncMock()

        with (
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch("app.services.analyzer.edit_report", editor),
            patch("app.services.analyzer._save_artifact", save_artifact),
        ):
            result = await _edit_final_report_language(
                "run-partial-final-editorial",
                report=partial,
                public_report={"brand": {"name": "Acme"}},
                selected_answer_context=[],
                answer_selection_manifest={},
                semantic_evidence_document={},
            )

        self.assertEqual(result, expected)
        artifact_output.assert_not_awaited()
        editor.assert_not_awaited()
        save_artifact.assert_awaited_once()
        saved = save_artifact.await_args.kwargs
        self.assertEqual(saved["status"], "completed")
        self.assertFalse(saved["output_json"]["audit"]["coverage_complete"])
        self.assertIn("audit_sha256", saved["output_json"]["audit"])

    async def test_unsupported_technical_preflight_returns_source_without_model_call(
        self,
    ) -> None:
        unsupported = {"unexpected": "Сохранённый технический вывод"}
        artifact_output = AsyncMock()
        editor = AsyncMock()
        save_artifact = AsyncMock()

        with (
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch("app.services.analyzer.edit_report", editor),
            patch("app.services.analyzer._save_artifact", save_artifact),
        ):
            result = await _edit_technical_review_language(
                "run-unsupported-technical-editorial",
                review=unsupported,
                profile={"brand_name": "Acme"},
            )

        self.assertEqual(result, unsupported)
        artifact_output.assert_not_awaited()
        editor.assert_not_awaited()
        save_artifact.assert_awaited_once()
        saved = save_artifact.await_args.kwargs
        self.assertEqual(saved["status"], "completed")
        self.assertFalse(saved["output_json"]["audit"]["coverage_complete"])
        self.assertIn("audit_sha256", saved["output_json"]["audit"])

    async def test_final_action_without_evidence_never_reaches_cache_or_models(
        self,
    ) -> None:
        partial = _report()
        partial["actions"][0].pop("evidence")
        expected = copy.deepcopy(partial)
        expected["headline_emphasis"] = []
        artifact_output = AsyncMock()
        structured_artifact = AsyncMock()
        editor = AsyncMock()
        save_artifact = AsyncMock()

        with (
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch("app.services.analyzer._structured_artifact", structured_artifact),
            patch("app.services.analyzer.edit_report", editor),
            patch("app.services.analyzer._save_artifact", save_artifact),
        ):
            result = await _edit_final_report_language(
                "run-action-without-evidence",
                report=partial,
                public_report={"brand": {"name": "Acme"}},
                selected_answer_context=[],
                answer_selection_manifest={},
                semantic_evidence_document={},
            )

        self.assertEqual(result, expected)
        artifact_output.assert_not_awaited()
        structured_artifact.assert_not_awaited()
        editor.assert_not_awaited()
        save_artifact.assert_awaited_once()
        audit = save_artifact.await_args.kwargs["output_json"]["audit"]
        self.assertFalse(audit["coverage_complete"])
        self.assertIn("audit_sha256", audit)

    async def test_technical_finding_without_evidence_never_reaches_cache_or_models(
        self,
    ) -> None:
        partial = _technical_review()
        partial["findings"][0].pop("evidence")
        artifact_output = AsyncMock()
        structured_artifact = AsyncMock()
        editor = AsyncMock()
        save_artifact = AsyncMock()

        with (
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch("app.services.analyzer._structured_artifact", structured_artifact),
            patch("app.services.analyzer.edit_report", editor),
            patch("app.services.analyzer._save_artifact", save_artifact),
        ):
            result = await _edit_technical_review_language(
                "run-finding-without-evidence",
                review=partial,
                profile={"brand_name": "Acme"},
            )

        self.assertEqual(result, partial)
        artifact_output.assert_not_awaited()
        structured_artifact.assert_not_awaited()
        editor.assert_not_awaited()
        save_artifact.assert_awaited_once()
        audit = save_artifact.await_args.kwargs["output_json"]["audit"]
        self.assertFalse(audit["coverage_complete"])
        self.assertIn("audit_sha256", audit)

    async def test_incomplete_reader_contract_fails_before_model_calls(self) -> None:
        source = _technical_review()
        editor = AsyncMock()
        critic = AsyncMock()

        edited, audit = await edit_report(
            source,
            editor_call=editor,
            critic_call=critic,
            prose_paths=["/overall_conclusion"],
        )

        self.assertEqual(edited, source)
        self.assertFalse(audit["coverage_complete"])
        self.assertIn(
            "/limitations/0",
            audit["source_manifest"]["missing_paths"],
        )
        editor.assert_not_awaited()
        critic.assert_not_awaited()

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

    async def test_incomplete_cached_review_is_ignored_for_fresh_pass(self) -> None:
        source = _technical_review()
        poisoned = copy.deepcopy(source)
        poisoned["overall_conclusion"] = "Старый неполный результат"
        fresh = copy.deepcopy(source)
        fresh["overall_conclusion"] = "Краулер получает HTML на двух страницах."
        artifact_output = AsyncMock(
            return_value={
                "review": poisoned,
                "audit": {
                    "version": REPORT_EDITOR_HARNESS_VERSION,
                    "policy_version": REPORT_EDITOR_POLICY_VERSION,
                    "coverage_complete": False,
                },
            }
        )
        save_artifact = AsyncMock()
        editor = AsyncMock(return_value=(fresh, {}))

        with (
            patch("app.services.analyzer._artifact_output", artifact_output),
            patch("app.services.analyzer._save_artifact", save_artifact),
            patch("app.services.analyzer.edit_report", editor),
        ):
            result = await _edit_technical_review_language(
                "run-poisoned-editorial-cache",
                review=source,
                profile={"brand_name": "Acme"},
            )

        self.assertEqual(result, fresh)
        editor.assert_awaited_once()
        self.assertEqual(save_artifact.await_count, 2)

    async def test_successful_edit_preserves_facts_and_passthrough_receipts(self) -> None:
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
        self.assertEqual(edited["headline_emphasis"], source["headline_emphasis"])
        self.assertEqual(edited["actions"][0]["evidence"], source["actions"][0]["evidence"])
        self.assertEqual(edited["limitations"], source["limitations"])
        self.assertTrue(audit["coverage_complete"])
        self.assertTrue(audit["path_coverage_complete"])
        self.assertTrue(audit["quality_complete"])
        self.assertFalse(audit["fallback_units"])
        self.assertTrue(
            validate_editorial_cache(
                source,
                edited,
                audit,
                protected_terms=["Acme"],
            )
        )

        incomplete = copy.deepcopy(audit)
        incomplete["coverage_complete"] = False
        incomplete = seal_editorial_audit(incomplete)
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                incomplete,
                protected_terms=["Acme"],
            )
        )

        poisoned_result = copy.deepcopy(edited)
        poisoned_result["actions"][0]["evidence"] = "Подменённое основание"
        self.assertFalse(
            validate_editorial_cache(
                source,
                poisoned_result,
                audit,
                protected_terms=["Acme"],
            )
        )

        poisoned_source = copy.deepcopy(source)
        poisoned_source["verdict"] = "Другой исходный отчёт"
        self.assertFalse(
            validate_editorial_cache(
                poisoned_source,
                edited,
                audit,
                protected_terms=["Acme"],
            )
        )

        tampered_audit = copy.deepcopy(audit)
        tampered_audit["changed_paths"] = []
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                tampered_audit,
                protected_terms=["Acme"],
            )
        )

        poisoned_audit = copy.deepcopy(audit)
        poisoned_audit["policy_sha256"] = "0" * 64
        poisoned_audit = seal_editorial_audit(poisoned_audit)
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                poisoned_audit,
                protected_terms=["Acme"],
            )
        )

    async def test_long_edit_keeps_code_owned_separators_when_model_strips(
        self,
    ) -> None:
        source = _report("Первый подтверждённый факт. " * 1_100)
        source["headline"] = "Acme виден в ответах"
        seen_payloads: list[dict] = []

        async def stripping_editor(payload: dict) -> dict:
            seen_payloads.append(payload)
            edited = str(payload["core_text"]).strip()
            return {
                "source_unit_id": payload["source_unit_id"],
                "source_sha256": payload["source_sha256"],
                "edited_text": edited,
                "claim_receipts": [
                    {
                        "claim_sha256": item["claim_sha256"],
                        "preserved": True,
                        "target_excerpt": item["source_excerpt"],
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
            editor_call=stripping_editor,
            critic_call=critic,
            protected_terms=["Acme"],
        )

        body_payloads = [
            payload
            for payload in seen_payloads
            if payload["path"] == "/sections/0/body"
        ]
        self.assertGreater(len(body_payloads), 1)
        self.assertTrue(
            all(
                payload["core_text"] == payload["core_text"].strip()
                for payload in body_payloads
            )
        )
        self.assertEqual(
            edited["sections"][0]["body"],
            source["sections"][0]["body"],
        )
        self.assertNotIn("факт.Первый", edited["sections"][0]["body"])
        self.assertTrue(audit["boundary_integrity_complete"])
        self.assertTrue(audit["quality_complete"])
        self.assertEqual(len(audit["boundary_receipts"]), audit["unit_count"])
        self.assertTrue(
            validate_editorial_cache(
                source,
                edited,
                audit,
                protected_terms=["Acme"],
            )
        )

        tampered_audit = copy.deepcopy(audit)
        tampered_audit["boundary_receipts"][0][
            "code_owned_suffix_sha256"
        ] = "0" * 64
        tampered_audit = seal_editorial_audit(tampered_audit)
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                tampered_audit,
                protected_terms=["Acme"],
            )
        )

    async def test_source_fallback_is_path_complete_but_not_cacheable(self) -> None:
        source = _report()
        editor = AsyncMock(side_effect=RuntimeError("editor unavailable"))
        critic = AsyncMock()

        edited, audit = await edit_report(
            source,
            editor_call=editor,
            critic_call=critic,
            protected_terms=["Acme"],
        )

        self.assertEqual(edited, source)
        self.assertTrue(audit["coverage_complete"])
        self.assertTrue(audit["path_coverage_complete"])
        self.assertFalse(audit["quality_complete"])
        self.assertTrue(audit["fallback_units"])
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                audit,
                protected_terms=["Acme"],
            )
        )
        critic.assert_not_awaited()

    async def test_critic_and_arbiter_failure_keeps_retryable_source(self) -> None:
        source = _report()
        source["headline"] = "Acme виден в ответах"

        async def editor(payload: dict) -> dict:
            edited = str(payload["core_text"])
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

        critic = AsyncMock(side_effect=RuntimeError("critic unavailable"))
        arbiter = AsyncMock(side_effect=RuntimeError("arbiter unavailable"))

        edited, audit = await edit_report(
            source,
            editor_call=editor,
            critic_call=critic,
            arbiter_call=arbiter,
            protected_terms=["Acme"],
        )

        self.assertEqual(edited, source)
        self.assertTrue(audit["path_coverage_complete"])
        self.assertFalse(audit["quality_complete"])
        self.assertTrue(audit["fallback_units"])
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                audit,
                protected_terms=["Acme"],
            )
        )
        self.assertGreater(critic.await_count, 0)
        self.assertGreater(arbiter.await_count, 0)

    async def test_semantic_fallback_invalidates_editorial_quality_cache(self) -> None:
        source = _report()
        source["headline"] = "Acme виден в ответах"

        async def editor(payload: dict) -> dict:
            edited = str(payload["core_text"])
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
        self.assertTrue(audit["quality_complete"])

        semantic_pass = copy.deepcopy(audit)
        semantic_pass["semantic_fallback"] = {"used": False, "errors": []}
        semantic_pass = seal_editorial_audit(semantic_pass)
        self.assertTrue(semantic_pass["quality_complete"])
        self.assertTrue(
            validate_editorial_cache(
                source,
                edited,
                semantic_pass,
                protected_terms=["Acme"],
            )
        )

        audit["semantic_fallback"] = {
            "used": True,
            "errors": ["semantic drift"],
        }
        audit = seal_editorial_audit(audit)

        self.assertFalse(audit["quality_complete"])
        self.assertFalse(
            validate_editorial_cache(
                source,
                edited,
                audit,
                protected_terms=["Acme"],
            )
        )

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
