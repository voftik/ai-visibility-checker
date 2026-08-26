import re
import subprocess
import unittest
from pathlib import Path


class FrontendStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = Path("static/index.html").read_text(encoding="utf-8")
        cls.pages = Path("app/routes/pages.py").read_text(encoding="utf-8")
        cls.analyzer = Path("app/services/analyzer.py").read_text(encoding="utf-8")
        cls.analysis_critic = Path("app/services/analysis_critic.py").read_text(
            encoding="utf-8"
        )
        cls.deepseek_icon = Path("static/brand/providers/deepseek.svg").read_text(
            encoding="utf-8"
        )

    def test_homepage_has_one_domain_input_and_no_analysis_controls(self) -> None:
        self.assertIn('id="domain-input"', self.html)
        self.assertIn("JSON.stringify({ domain })", self.html)
        self.assertNotIn("setsLoading", self.html)
        self.assertNotIn('name="user_agents"', self.html)
        self.assertNotIn('name="concurrency"', self.html)
        self.assertNotIn("Выберите домены", self.html)
        self.assertNotIn("rwplus-aiv-run-ids", self.html)
        self.assertIn('fetchJSON("/api/runs/lookup"', self.html)
        self.assertIn("fetchJSON(`/api/runs?${params.toString()}`)", self.html)

    def test_progress_has_six_named_public_stages(self) -> None:
        for label in (
            "Изучаем сайт",
            "Проверяем доступ для ИИ",
            "Формируем сценарии выбора",
            "Сравниваем ответы ИИ-систем",
            "Сопоставляем режимы исследования",
            "Собираем отчёт и иллюстрации",
        ):
            self.assertIn(label, self.html)
        self.assertNotIn("proxy_address", self.html)
        self.assertNotIn("body_sample", self.html)
        self.assertNotIn("response_headers", self.html)

    def test_progress_distinguishes_queue_running_and_recovery(self) -> None:
        for marker in (
            "const runLifecycleState =",
            '["queued", "running", "recovering", "completed", "failed"]',
            "const progressPresentation =",
            'lifecycle === "queued"',
            'lifecycle === "recovering"',
            'presentation.lifecycle === "running"',
            "Позиция в очереди:",
            "Сохранённые результаты не пропали.",
            "Одна проверка за раз",
            "Сейчас вы ${positionLabel} в очереди.",
            "queue_position",
            "queue_total",
        ):
            self.assertIn(marker, self.html)
        progress_renderer = self.html[
            self.html.index("const renderProgress = () =>") : self.html.index(
                "const metricHTML ="
            )
        ]
        self.assertIn("const currentIndex = stages.findIndex(", progress_renderer)
        self.assertIn("const hasActiveStage = (", progress_renderer)
        self.assertNotIn("const currentIndex = Math.max(", progress_renderer)
        self.assertNotIn("исполнительный слот", self.html)

    def test_terminal_review_state_is_not_presented_as_retryable(self) -> None:
        for marker in (
            "const operatorReviewStages = new Set([",
            '"source_review_required"',
            '"integrity_review_required"',
            '"panel_review_required"',
            "const runNeedsOperatorReview =",
            "Автоматическое продолжение отключено",
            'if (runNeedsOperatorReview(run)) return "Подробнее";',
            "const needsReview = runNeedsOperatorReview(run);",
            '${needsReview ? "" : `',
        ):
            self.assertIn(marker, self.html)
        self.assertIn("Нужна проверка источников", self.html)
        self.assertIn("Нужна проверка данных", self.html)
        self.assertIn("Нужна проверка ответов", self.html)

    def test_progress_never_invents_an_eta(self) -> None:
        eta_formatter = self.html[
            self.html.index("const formatEta =") : self.html.index(
                "const runLifecycleState ="
            )
        ]
        self.assertIn("const value = finiteNumber(seconds);", eta_formatter)
        self.assertIn(
            "if (value === null || value <= 0) return null;",
            eta_formatter,
        )
        self.assertNotIn('return "меньше минуты"', eta_formatter)
        self.assertIn('eta || "Время уточняется"', self.html)
        self.assertIn(
            "eta_seconds: finiteNumber(payload.eta_seconds)",
            self.html,
        )
        self.assertGreaterEqual(self.html.count('"Время уточняется"'), 3)

    def test_live_run_updates_are_revision_guarded(self) -> None:
        for marker in (
            "const runRevision =",
            "const canApplyRunUpdate =",
            "return nextRevision >= currentRevision;",
            "const mergeRunListsByRevision =",
            "if (state.run && !canApplyRunUpdate(state.run, run)) return false;",
            "if (!canApplyRunUpdate(state.run, eventUpdate)) return;",
            "state_revision: payload.state_revision",
        ):
            self.assertIn(marker, self.html)
        watch_block = self.html[
            self.html.index("const watchRun =") : self.html.index(
                "const refreshHistory ="
            )
        ]
        self.assertGreaterEqual(
            watch_block.count(
                "if (!canApplyRunUpdate(state.run, eventUpdate)) return;"
            ),
            2,
        )
        refresh_block = self.html[
            self.html.index("const refreshRun =") : self.html.index("const watchRun =")
        ]
        self.assertIn("updateRunView(run);", refresh_block)

    def test_rw_plus_visual_tokens_are_present(self) -> None:
        self.assertIn("--canvas: #cdd5de", self.html.lower())
        self.assertIn("--accent: #ff324b", self.html.lower())
        self.assertIn("--ink: #101011", self.html.lower())
        self.assertIn("family=Manrope", self.html)
        self.assertIn("/static/brand/logo-rw-plus.svg", self.html)
        self.assertIn('class="brand-logo"', self.html)
        self.assertIn('alt="RW+"', self.html)
        self.assertNotIn('class="brand-name"', self.html)

    def test_homepage_rotates_the_subject_with_gsap_and_keeps_the_question(
        self,
    ) -> None:
        self.assertIn("gsap@3.15.0/dist/gsap.min.js", self.html)
        self.assertIn('data-words="бренд,сайт,продукт"', self.html)
        self.assertIn('class="rotating-word-current">бренд</span>', self.html)
        self.assertIn('class="hero-question"', self.html)
        self.assertIn("initHeadlineRotation", self.html)
        self.assertIn("window.gsap.timeline", self.html)
        self.assertIn("window.gsap.delayedCall", self.html)
        self.assertIn('"(prefers-reduced-motion: reduce)"', self.html)
        self.assertIn('aria-label="ваш бренд, сайт или продукт?"', self.html)

    def test_report_integrates_three_illustrations_and_metrics(self) -> None:
        self.assertIn("illustrationHTML(illustrations[0]", self.html)
        self.assertIn("illustrationHTML(illustrations[1]", self.html)
        self.assertIn("illustrationHTML(illustrations[2]", self.html)
        self.assertIn("metric-rail", self.html)
        self.assertIn("intent-grid", self.html)
        self.assertIn("compare-grid", self.html)

    def test_report_uses_real_interactive_charts_with_table_fallbacks(self) -> None:
        self.assertIn("echarts@6.1.0/dist/echarts.min.js", self.html)
        self.assertIn("window.echarts.init", self.html)
        self.assertIn('{ renderer: "svg" }', self.html)
        self.assertIn("disposeReportCharts", self.html)
        self.assertIn("ResizeObserver", self.html)
        for chart_id in (
            "chart-competitors",
            "chart-intents",
        ):
            self.assertIn(f'id="{chart_id}"', self.html)
        self.assertNotIn('id="chart-technical-families"', self.html)
        self.assertIn('class="family-status-panel"', self.html)
        self.assertNotIn('id="chart-providers"', self.html)
        self.assertNotIn('mountReportChart("chart-providers"', self.html)
        self.assertNotIn('id="chart-web-memory"', self.html)
        self.assertIn('class="data-table paired-comparison-table"', self.html)
        self.assertIn("chartDataDetailsHTML", self.html)
        self.assertIn('aria-busy="true"', self.html)
        self.assertIn('"(prefers-reduced-motion: reduce)"', self.html)

    def test_entity_ranking_explains_scope_formula_and_meaning(self) -> None:
        for marker in (
            "Кого ИИ называет, когда бренда",
            "и его вариантов в запросах",
            "competitorChartExplainerHTML(viewModel, competitorRows)",
            "Пользовательские задачи без названия",
            "Ответы из режима с веб-поиском.",
            "Незавершённые и невалидные",
            "доля, где сущность встретилась хотя бы один раз",
            "Это не доля рынка и не обязательно рекомендация.",
            "competitor-chart-tooltip__context",
            "Исследуемый бренд не назван в вопросе · режим с веб-поиском",
            "<strong>Как считается:</strong>",
            "Процент равен",
            "Повтор сущности внутри одного ответа не увеличивает числитель.",
            "<strong>Что это значит:</strong>",
            "Упоминание не означает рекомендацию, высокую позицию или долю рынка.",
            "диаграммы не обязаны складываться в 100%.",
            "Доля валидных ответов",
            'triggerOn: "mousemove|click"',
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn(
            '<h3 class="chart-title">Целевой бренд, продукты группы и альтернативы</h3>',
            self.html,
        )

    def test_provider_and_web_memory_comparisons_are_plainly_readable(self) -> None:
        for marker in (
            "Как отвечает каждая ИИ-система",
            "providerComparisonMatrixHTML(providers",
            "Одинаковый процент может скрывать разные базы",
            "Безбрендовые запросы · с веб-поиском",
            "knowledgePresentation.columnLabel.toLowerCase()",
            "pairedComparisonTableHTML",
            "Что ищем в ответах",
            "Наблюдаемая разница",
            "Одинаковые запросы и ИИ-системы в двух режимах",
            "paired-delta-pill",
            "Показана наблюдаемая разница долей, а не доказанный причинный эффект",
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn("Точечное сравнение ИИ-систем", self.html)
        self.assertNotIn("Веб-поиск повышает вероятность", self.html)
        self.assertNotIn("собственные знания моделей дали результат выше", self.html)
        self.assertIn("visualMap: {", self.html)
        self.assertIn("show: false", self.html)

    def test_observational_memory_is_visible_but_never_presented_as_attested(
        self,
    ) -> None:
        for marker in (
            "const observationalMemoryLimitation =",
            "const isObservationalMemoryMetric =",
            "const memorySlicePresentation =",
            "metric.strict_no_web_verified === true",
            '"legacy_observational", "mixed"',
            '=== "legacy_memory_request_not_enforced"',
            '? "is-observational"',
            'parts.push(compact ? "исторический срез" : "Исторический срез")',
            'class="comparison-observation-note"',
            '"обращения к веб-инструментам не обнаружено, но техническое "',
            '"отключение веба в том запуске не аттестовано."',
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn("allowKnowledgeWebFallback", self.html)
        self.assertNotIn("usesWebFallback", self.html)

    def test_provider_comparison_is_an_accessible_evidence_matrix(self) -> None:
        for marker in (
            "const providerComparisonMatrixHTML =",
            'class="provider-comparison-table"',
            'aria-label="Сравнение результатов ИИ-систем"',
            "<caption>Доля валидных ответов, соответствующих критерию</caption>",
            'scope="colgroup"',
            'scope="row"',
            'data-label="${escapeHTML(definition.label)}"',
            "providerMetricDenominator",
            "metric.valid_answers",
            "top3_count",
            "recommendation_count",
            "contradiction_count",
            "Неполная база:",
            "Для этого среза нет валидных ответов, поэтому ноль не подставляется.",
            "Показатель отражает конкретность ответа",
            "providerIdentityHTML",
            'alt=""',
            'aria-hidden="true"',
            ".provider-comparison-table thead",
            "content: attr(data-label)",
        ):
            self.assertIn(marker, self.html)
        for obsolete in (
            'id="chart-providers"',
            'mountReportChart("chart-providers"',
            "provider-metric-legend",
            "providerAxisRich",
            "providerSeries",
            "providerIcon",
            "providerBadge",
        ):
            self.assertNotIn(obsolete, self.html)

    def test_report_supports_new_visibility_contract_and_legacy_alias(self) -> None:
        for field in (
            "const normalizeReport =",
            'const contract = hasV2Shape ? "v2" : "legacy"',
            "discovery.parent",
            "discovery.portfolio",
            "report.portfolio_visibility",
            "report.brand_knowledge",
            "legacyVisibility.providers",
            "provider?.parent_discovery",
            "provider?.portfolio_capture",
            "provider?.brand_knowledge",
        ):
            self.assertIn(field, self.html)
        self.assertIn("target.parent = normalizeMetricObject", self.html)
        self.assertIn("target.portfolio = normalizeMetricObject", self.html)
        self.assertIn("finiteNumber(metric.value) === null", self.html)
        self.assertIn("Материнский бренд без подсказки", self.html)
        self.assertIn("Продукты и направления без подсказки", self.html)
        self.assertIn("Ответ по существу о названном бренде", self.html)

    def test_metric_percentages_show_evidence_or_explicit_data_state(self) -> None:
        self.assertIn("const metricPresentation = (", self.html)
        self.assertIn('rateKeys = ["mention_rate"]', self.html)
        self.assertIn('dataState === "limited"', self.html)
        self.assertIn('dataState === "unavailable"', self.html)
        self.assertIn('primary: "Неполные данные"', self.html)
        self.assertIn('primary: "Нет данных"', self.html)
        self.assertIn('countKey: "mention_count"', self.html)
        self.assertIn('countKey: "specific_count"', self.html)
        self.assertIn('countLabel: "Упоминаний"', self.html)
        self.assertIn('countLabel: "Конкретных ответов"', self.html)
        self.assertIn("metric?.annotated_answers", self.html)
        self.assertIn("metric?.valid_answers", self.html)
        self.assertIn("metric?.coverage_rate", self.html)
        self.assertIn("providerMetricCellHTML(", self.html)
        self.assertIn('data-label="${escapeHTML(metricLabel)}"', self.html)
        self.assertIn('title="${escapeHTML(accessibleLabel)}"', self.html)
        self.assertIn('aria-label="${escapeHTML(accessibleLabel)}"', self.html)

    def test_unavailable_portfolio_is_explained_and_not_drawn_as_zero(
        self,
    ) -> None:
        for marker in (
            "const portfolioScope = (",
            "discovery.portfolio_scope",
            "const portfolioScopeNotice = (",
            '=== "target_portfolio_unconfirmed"',
            "Состав продуктового портфеля не подтверждён данными сайта",
            "продуктовые показатели не рассчитаны и не заменены нулём",
            "portfolioUnavailable: Boolean(portfolioScopeNotice)",
            "portfolioUnavailable ? 1 : 2",
            "...(!portfolioScopeNotice",
            "...(!portfolioUnavailable",
            "{ includePortfolio: !portfolioScopeNotice }",
            "value: [intentIndex, entry.rowIndex, value]",
            "const hasMatchedPairs = pairCount !== null && pairCount > 0;",
            "Нет подтверждённых пар для сравнения",
            "Продуктовый показатель не рассчитан, и ноль не подставляется.",
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn(
            "value: [intentIndex, entry.rowIndex, value === null ? 0 : value]",
            self.html,
        )

    def test_mentions_use_rates_and_composite_scores_are_named_as_indices(self) -> None:
        for marker in (
            'rateKeys: ["mention_rate"]',
            "row.observedDifferenceMetric",
            '=== "mention_rate_percentage_points"',
            "finiteNumber(row.observedDifference)",
            'const indexValue = metricRate(metric, ["score"])',
            "Индекс ${safeNumber(indexValue, indexValue % 1 ? 1 : 0)}/100",
            "Индекс 0–100",
            "Индекс видимости бренда",
            "Индекс видимости продуктов",
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn('rateKeys: ["score"]', self.html)
        self.assertNotIn('["score", "mention_rate"]', self.html)
        self.assertNotIn("score_lift", self.html)
        self.assertNotIn("webRate - memoryRate", self.html)
        self.assertNotIn("calculatedLift", self.html)

    def test_report_avoids_obvious_interface_meta_narration(self) -> None:
        self.assertNotIn("\u2014", self.html)
        for redundant_copy in (
            "Три независимых среза",
            "Доля ответов и фактический числитель",
            "Две строки разделяют материнский бренд и продукты",
            "Не общий балл ради балла",
            "Длина полос относительная",
            "Краткий итог каждого раздела остаётся на виду",
            "Откройте карточку",
            "Сравниваем ${safeNumber(paired.nPairs)} одинаковых пар",
            "Два отдельных среза",
            "Здесь разведены три разных вопроса",
            "Они показаны отдельно",
            "Раздел без названия",
            "Пояснение для этого пункта не сформировано",
            "Пояснение для этого раздела не сформировано",
            "Ось начинается с нуля",
            "Целевой бренд остаётся на диаграмме",
            "подсказка показывает число ответов",
        ):
            self.assertNotIn(redundant_copy, self.html)
        self.assertIn(
            "Одна модель может назвать несколько компаний",
            self.html,
        )
        self.assertIn(
            "называют материнский бренд и подтверждённые продукты",
            self.html,
        )
        self.assertIn(
            "Каждый запрос сопоставлен с той же ИИ-системой",
            self.html,
        )

    def test_technical_audit_is_first_detailed_report_section(self) -> None:
        active_renderer = self.html[self.html.index("const renderReport = () =>") :]
        audit_index = active_renderer.index(
            '<h2 class="section-title" id="technical-title">Технический аудит</h2>'
        )
        visibility_index = active_renderer.index(
            '<h2 class="section-title" id="visibility-title">Кого модели упоминают сами</h2>'
        )
        self.assertLess(audit_index, visibility_index)
        self.assertEqual(self.html.count('id="technical-title">Технический аудит'), 1)
        self.assertIn('class="audit-scoreline"', self.html)
        self.assertIn("technicalAuditMatrixHTML(technical)", self.html)
        self.assertIn("technicalAccessMatrixHTML(technical)", self.html)
        self.assertIn("technicalPagesHTML(technical.pages)", self.html)
        self.assertIn("technicalFindingsHTML(technical)", self.html)
        self.assertIn("const reviewedFindings =", self.html)
        self.assertIn("const candidates = reviewedFindings.length", self.html)
        self.assertIn('class="data-table audit-matrix-table"', self.html)
        self.assertIn('class="data-table page-audit-table"', self.html)
        self.assertIn('class="data-table audit-actions-table"', self.html)
        self.assertIn(
            "const contentPages = pages.filter((page) => page?.is_utility !== true);",
            self.html,
        )
        self.assertIn(
            "const isUtility = page?.is_utility === true;",
            self.html,
        )
        self.assertGreaterEqual(self.html.count('"Не проверяем"'), 3)
        self.assertIn('"Не оценивается"', self.html)

    def test_page_audit_table_has_readable_columns_and_accessible_tooltips(
        self,
    ) -> None:
        for marker in (
            "min-width: 900px",
            ".page-audit-table col:nth-child(7)",
            "word-break: normal",
            ".audit-tooltip--header",
            "white-space: nowrap",
            "const auditTooltipHTML =",
            "const auditHeaderTooltipHTML =",
            'role="tooltip"',
            'aria-describedby="${escapeHTML(id)}"',
            "data-audit-tooltip-trigger",
            "data-audit-tooltip-bubble",
            "bindAuditTooltips()",
            'trigger.addEventListener("focus"',
            'event.pointerType === "touch"',
            'event.key === "Escape"',
            "httpTooltipCopy",
            "renderingTooltipCopy",
            "accessTooltipCopy",
            "checksTooltipCopy",
            "schemaStateForPage",
            "schemaStatePresentation",
            "scoreInclusionTooltipCopy",
            'data-label="Schema.org"',
            '"Schema.org"',
        ):
            self.assertIn(marker, self.html)
        self.assertIn("<colgroup>", self.html)
        self.assertNotIn("Schema.or<br", self.html)

    def test_schema_state_matrix_fails_closed_for_incomplete_pages(self) -> None:
        def declaration_for(name: str) -> str:
            marker = f"      const {name} ="
            start = self.html.find(marker)
            self.assertNotEqual(start, -1)
            end = self.html.find("\n\n      const ", start + len(marker))
            self.assertNotEqual(end, -1)
            return self.html[start:end].strip()

        script = "\n".join(
            (
                '"use strict";',
                'const assert = require("node:assert/strict");',
                declaration_for("schemaStateForPage"),
                declaration_for("schemaStatePresentation"),
                r"""
const state = (page) => schemaStateForPage(page);
assert.equal(
  state({is_utility: true, structured_data_types: ["WebPage"]}),
  "excluded"
);
assert.equal(state({schema_evaluation_excluded: true}), "excluded");
assert.equal(state({is_excluded: true}), "excluded");
assert.equal(
  state({
    structured_data_types: ["Organization"],
    body_truncated: true,
    structured_data_complete: false
  }),
  "found"
);
assert.equal(
  state({
    structured_data_types: [],
    body_truncated: false,
    structured_data_complete: true
  }),
  "absent"
);
assert.equal(
  state({
    structured_data_types: [],
    body_truncated: true,
    structured_data_complete: true
  }),
  "unknown"
);
assert.equal(
  state({
    structured_data_types: [],
    body_truncated: false,
    structured_data_complete: false
  }),
  "unknown"
);
assert.equal(state({structured_data_types: []}), "unknown");
assert.equal(
  state({body_truncated: false, structured_data_complete: true}),
  "absent"
);

const found = schemaStatePresentation({
  structured_data_types: ["Organization"],
  body_truncated: false,
  structured_data_complete: true
});
assert.equal(found.label, "Найдена");
assert.match(found.copy, /Organization/);

const absent = schemaStatePresentation({
  structured_data_types: [],
  body_truncated: false,
  structured_data_complete: true
});
assert.equal(absent.label, "Не найдена");
assert.match(absent.copy, /проверены полностью/);

const truncated = schemaStatePresentation({
  structured_data_types: [],
  body_truncated: true,
  structured_data_complete: false
});
assert.equal(truncated.label, "Данных недостаточно");
assert.match(truncated.copy, /Получен только фрагмент HTML/);
assert.match(truncated.copy, /отсутствие Schema\.org подтвердить нельзя/);

const unknown = schemaStatePresentation({structured_data_types: []});
assert.equal(unknown.label, "Данных недостаточно");
assert.match(unknown.copy, /Полнота HTML.+не подтверждена/);

const excluded = schemaStatePresentation({is_utility: true});
assert.equal(excluded.label, "Не проверяем");
assert.match(excluded.copy, /Служебные страницы/);
""",
            )
        )
        completed = subprocess.run(
            ["node", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"Schema state runtime matrix failed:\n{completed.stderr}",
        )

    def test_schema_cell_explains_found_absent_unknown_and_excluded_states(
        self,
    ) -> None:
        for marker in (
            'return "excluded";',
            'return "found";',
            'return "absent";',
            'return "unknown";',
            'label: "Найдена"',
            'label: "Не найдена"',
            'label: "Данных недостаточно"',
            'label: "Не проверяем"',
            "schemaPresentation.copy",
            "schemaPresentation.schemaTypes.join",
            "поэтому отсутствие Schema.org подтвердить нельзя",
        ):
            self.assertIn(marker, self.html)

    def test_completed_report_does_not_render_unexplained_missing_metrics(self) -> None:
        self.assertIn('let formatted = "Нет данных";', self.html)
        self.assertIn('metric.state = "unavailable"', self.html)
        self.assertIn("completedRequiredMissing", self.html)
        self.assertIn(
            "Завершённый отчёт пришёл без обязательной части данных.", self.html
        )
        self.assertIn("data-hard-refresh", self.html)
        self.assertIn('cache: options.cache || "no-store"', self.html)
        self.assertIn('const uiBuildId = "2026-07-31.29"', self.html)
        self.assertIn('"2026-07-v3"', self.html)
        self.assertIn("checkUIVersion", self.html)

    def test_html_shell_is_never_reused_after_a_ui_contract_change(self) -> None:
        self.assertIn('UI_BUILD_ID = "2026-07-31.29"', self.pages)
        self.assertIn('"Cache-Control": "no-store, max-age=0"', self.pages)
        self.assertIn('"X-AIV-UI-Version": UI_BUILD_ID', self.pages)
        self.assertIn('@router.get("/api/ui-version"', self.pages)

    def test_family_access_uses_binary_lamps_and_provider_visuals(self) -> None:
        for marker in (
            "providerChartVisuals",
            "providerChartVisual",
            "/static/brand/providers/openai.svg",
            "/static/brand/providers/anthropic.svg",
            "/static/brand/providers/perplexity.svg",
            "/static/brand/providers/gemini.svg",
            "/static/brand/providers/deepseek.svg",
            "familyAccessStatus",
            "familyStatusPanelHTML",
            "family-status-grid",
            "family-status-card",
            "family-status-lamp-bezel",
            "family-status-lamp-glass",
            "family-status-lamp-filament",
            'label: "Доступ открыт"',
            'label: "Доступ закрыт"',
            'label: "Не подтверждён"',
            'label: "Не проверен"',
            "Лампа горит: доступ открыт",
            'modifier: "family-status-tooltip"',
            "Как проверен доступ для ${row.name}",
            "Индикатор показывает",
            "но не\\u00a0качество их содержания",
            "family-status-card:not(.is-available) .family-status-lamp-filament",
            "family-status-card.is-available .family-status-lamp-glass",
            ".family-status-card:hover,",
            "animation: none;",
            "chart-title-domain",
            "Доступ сайта",
            "для ИИ-краулеров",
            "${escapeHTML(technicalCoverage.detail)} · открыт / закрыт",
            'role="list"',
            'role="listitem"',
            'tabindex="0"',
        ):
            self.assertIn(marker, self.html)
        for obsolete in (
            'id="chart-technical-families"',
            'mountReportChart("chart-technical-families"',
            "familyAxisRich",
            "familyChartDataTableHTML",
            "· 0–100%",
        ):
            self.assertNotIn(obsolete, self.html)
        self.assertIn('viewBox="0 0 56.25 41.36"', self.deepseek_icon)
        self.assertIn('fill="#4D6BFE"', self.deepseek_icon)

    def test_technical_score_is_explicitly_scoped_to_checked_pages(self) -> None:
        for marker in (
            "Техническая готовность проверенного среза",
            "const technicalCoveragePresentation =",
            "technicalCoverage.evaluated_pages",
            "technicalCoverage.discovered_pages",
            'badge: "Ограниченный срез"',
            'badge: "Полный охват"',
            'badge: "Охват не определён"',
            "Вывод по проверенному срезу:",
            "coverage-badge",
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn(
            'report.technical?.score,\n                "Техническая доступность"',
            self.html,
        )

    def test_long_technical_tables_use_native_accessible_disclosures(self) -> None:
        for marker in (
            "const technicalTableDisclosureHTML =",
            'class="audit-section audit-disclosure"',
            "data-technical-table-disclosure",
            'role="heading" aria-level="3"',
            ".audit-disclosure > summary",
            ".audit-disclosure[open] > summary::after",
            "Статус каждого сигнала, подтверждение, влияние на ИИ и следующий шаг.",
            "HTTP, объём текста, способ рендеринга, доступ ИИ и Schema.org по каждому URL.",
            "Какие семейства ИИ-краулеров смогли открыть каждую проверенную страницу.",
            "Приоритеты, доказательства, влияние на видимость и конкретные исправления.",
        ):
            self.assertIn(marker, self.html)
        self.assertEqual(self.html.count("technicalTableDisclosureHTML({"), 4)
        disclosure_helper = self.html[
            self.html.index("const technicalTableDisclosureHTML =") : self.html.index(
                "const illustrationHTML ="
            )
        ]
        self.assertNotIn("chart-technical-families", disclosure_helper)
        self.assertNotIn(" open", disclosure_helper)

    def test_homepage_explains_the_check_in_the_users_words(self) -> None:
        self.assertIn(
            "Давайте посмотрим, насколько ваши продукты, услуги, бренд и сайт",
            self.html,
        )
        self.assertIn(
            "в целом доступны для ИИ-поиска и можно ли с этим что-то сделать.",
            self.html,
        )
        self.assertNotIn("AIV читает сайт как краулер", self.html)

    def test_history_is_a_separate_responsive_routed_screen(self) -> None:
        home_renderer = self.html[
            self.html.index("const renderHome = () =>") : self.html.index(
                "const renderHistory = () =>"
            )
        ]
        self.assertNotIn('id="history"', home_renderer)
        for marker in (
            "data-history-link",
            "const renderHistory = () =>",
            'document.title = "История проверок · RW+"',
            'class="history-table"',
            '<th scope="col">Сайт</th>',
            '<th scope="col">Запущена</th>',
            '<th scope="col">Статус</th>',
            "history-open",
            'class="empty-history-copy"',
            "Проверок пока нет",
            'history.pushState({}, "", "/history")',
            'window.location.pathname === "/history"',
            'window.location.hash === "#history"',
            'window.addEventListener("popstate", scheduleLocationRoute)',
            'window.addEventListener("hashchange", scheduleLocationRoute)',
            "const HISTORY_PAGE_SIZE = 100",
            "before_created_at",
            "data-load-more-history",
            "Показать более ранние проверки",
            'fetchJSON("/api/runs/lookup"',
            "Здесь собраны все проверки сервиса.",
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn("rwplus-aiv-run-ids", self.html)
        self.assertNotIn("state.runs.slice(0, 8)", self.html)
        self.assertIn('else if (state.view === "history") renderHistory()', self.html)
        self.assertIn('@router.get("/history"', self.pages)
        self.assertIn("@media (max-width: 700px)", self.html)
        self.assertIn(".history-table tbody tr", self.html)

    def test_history_has_live_loading_and_error_states(self) -> None:
        for marker in (
            "historyLoading: false",
            'historyError: ""',
            "historyPollTimer: null",
            "const refreshHistory = async",
            "const startHistoryWatch =",
            "mergeRunListsByRevision(state.runs, runs)",
            "Загружаем историю",
            "История пока не загрузилась",
            "Не удалось обновить статусы.",
            "data-retry-history",
            "startHistoryWatch();",
            "runStatusDetail(run)",
            "Позиция в очереди:",
            "Готово ${Math.max(0, Math.min(100, Math.round(percent)))}%",
        ):
            self.assertIn(marker, self.html)
        loader = self.html[
            self.html.index("const loadPublicRuns =") : self.html.index(
                "const runStatusLabel ="
            )
        ]
        self.assertNotIn("catch", loader)
        self.assertNotIn("return []", loader)

    def test_full_opus_analysis_is_preserved_but_not_duplicated_in_main_flow(
        self,
    ) -> None:
        for marker in (
            "sanitizeMarkdownFragment",
            "analysisDocumentSections",
            "analysisSectionLead",
            "analysisDisclosureHTML",
            "appendAnalysisBody",
            "analysisIllustrationSVG",
            'className = "analysis-card-grid"',
            'className = "analysis-card-visual"',
            'title.textContent = "Выводы и доказательства"',
            "${analysisDisclosure}",
        ):
            self.assertIn(marker, self.html)
        self.assertNotIn('<details class="full-analysis"', self.html)
        self.assertIn('"TABLE", "THEAD"', self.html)
        self.assertIn('"TBODY", "TFOOT", "TR", "TH", "TD", "CAPTION"', self.html)
        self.assertIn('fallback.className = "analysis-raw-fallback"', self.html)
        self.assertIn("details.dataset.analysisKind = kind", self.html)
        self.assertIn(
            "if (sections.length === 1 && index === 0) details.open = true",
            self.html,
        )
        self.assertNotIn("localizeAnalysisEvidence", self.html)
        self.assertNotIn("localizeAnalysisText", self.html)
        self.assertIn("deck.textContent = lead", self.html)
        self.assertNotIn("final-v6", self.analyzer)
        self.assertIn("final-v28", self.analyzer)

    def test_intent_zeroes_show_scope_and_evidence(self) -> None:
        self.assertIn("const intentMetricPresentation =", self.html)
        self.assertIn("Цель не названа ни в одном", self.html)
        self.assertIn(
            "с веб-поиском · один ответ каждой ИИ-системы",
            self.html,
        )
        self.assertIn("Во всех шести безбрендовых сценариях", self.html)
        self.assertNotIn("0 из 5 в каждом сценарии", self.html)
        self.assertNotIn("В строке «Материнский бренд»", self.html)
        self.assertNotIn("Верхняя строка — 0 из 5", self.html)
        self.assertIn(
            "назвали «${escapeHTML(viewModel.brandName)}»",
            self.html,
        )
        self.assertNotIn("но имя RW+ в них не появилось", self.html)
        self.assertIn("metric.mentioned_entities", self.html)
        self.assertIn("${entity.name}:", self.html)
        self.assertIn("countLabel: presentation.countLabel", self.html)
        self.assertIn("{count|${params.data.countLabel}}", self.html)

    def test_analysis_prompts_do_not_assume_a_specific_client(self) -> None:
        self.assertIn(
            "Для сайта rw.plus исследуемый бренд может совпадать",
            self.analyzer,
        )
        self.assertNotIn("RW+ здесь —\nиздатель отчёта", self.analyzer)
        self.assertNotIn('"realweb",', self.analyzer)
        for client_example in ("Garpun", "DataGo", "Centra", "Campaign 360"):
            self.assertNotIn(client_example, self.analysis_critic)

    def test_intent_labels_follow_the_canonical_methodology(self) -> None:
        intent_block = self.html[
            self.html.index("const canonicalIntentNames =") : self.html.index(
                "const state ="
            )
        ]
        for marker in (
            'E: "Сравнение"',
            'T: "Решение"',
            'NB: "Потребность"',
            'NAV: "Источник"',
            'TR: "Тренд"',
            "Задача, боль, ограничение или контекст использования",
            "Источник, площадка, обзор, агрегатор или точка входа",
            "Тренды, новизна, популярность или меняющееся поведение",
            'methodology?.intent_taxonomy_version === "canonical-v1"',
            "usesCanonicalIntentTaxonomy(methodology)",
        ):
            self.assertIn(marker, intent_block)
        self.assertIn('NB: "Категория"', intent_block)
        self.assertIn('TR: "Доверие"', intent_block)

    def test_report_headline_segments_are_safe_and_lossless(self) -> None:
        self.assertIn("narrative?.headline_segments", self.html)
        self.assertIn("narrative?.headline_emphasis", self.html)
        self.assertIn("normalizeHeadline", self.html)
        self.assertIn("return null;", self.html)
        self.assertIn("escapeHTML(segment.text)", self.html)
        self.assertIn("headline-segment--accent", self.html)
        self.assertIn("headline-segment--support", self.html)

    def test_report_headline_uses_uniform_viewport_aware_type(self) -> None:
        segment_rule = re.search(
            r"\.headline-segment\s*\{([^}]*)\}",
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(segment_rule)
        for declaration in (
            "color: inherit;",
            "font-size: inherit;",
            "font-weight: inherit;",
            "letter-spacing: inherit;",
            "line-height: inherit;",
        ):
            self.assertIn(declaration, segment_rule.group(1))

        modifier_rules = self.html[
            self.html.index(".headline-segment--accent,") : self.html.index(
                ".report-verdict"
            )
        ]
        self.assertNotRegex(modifier_rules, r"font-(?:size|weight)\s*:")
        self.assertNotRegex(modifier_rules, r"letter-spacing\s*:")

        report_title_rules = re.findall(
            r"\.report-hero \.report-title\s*\{([^}]*)\}",
            self.html,
            re.DOTALL,
        )
        self.assertGreaterEqual(len(report_title_rules), 2)
        for rule in (report_title_rules[0], report_title_rules[-1]):
            size = re.search(r"font-size:\s*([^;]+);", rule)
            self.assertIsNotNone(size)
            for marker in ("clamp(", "min(", "vw", "svh"):
                self.assertIn(marker, size.group(1))
        self.assertIn("font-weight: 600;", report_title_rules[0])

    def test_mobile_report_tables_keep_readable_type(self) -> None:
        self.assertIn(".data-table td::before", self.html)
        self.assertIn("content: attr(data-label)", self.html)
        self.assertIn("@media (max-width: 460px)", self.html)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", self.html)
        self.assertIn('.data-table tbody th[scope="row"]', self.html)
        self.assertIn(".data-table.audit-matrix-table tbody td", self.html)
        self.assertIn("max-width: 100%;", self.html)
        self.assertIn("overflow-wrap: break-word;", self.html)
        self.assertIn("word-break: normal;", self.html)
        self.assertIn(".matrix-value::before", self.html)
        self.assertIn(".matrix-value {\n        font-size: 12px;", self.html)
        self.assertIn(".matrix-name {\n        grid-column: 1 / -1;", self.html)
        self.assertNotIn(".matrix-header {\n        font-size: 8px;", self.html)

    def test_report_navigation_does_not_float_over_report_content(self) -> None:
        self.assertIn('body[data-view="report"] .topbar-wrap', self.html)
        self.assertIn(
            'document.body.dataset.view = state.view || "home";',
            self.html,
        )

    def test_competitor_axis_is_compact_and_readable_on_mobile(self) -> None:
        layout_block = self.html[
            self.html.index("const competitorChartLayout =") : self.html.index(
                "const competitorChartExplainerHTML ="
            )
        ]
        chart_block = self.html[
            self.html.index(
                'const competitorChart = mountReportChart("chart-competitors"'
            ) : self.html.index("const intents = orderedIntentRows")
        ]
        chart_setup = self.html[
            self.html.index(
                "const initReportCharts = (viewModel) =>"
            ) : self.html.index(
                'const competitorChart = mountReportChart("chart-competitors"'
            )
        ]
        for marker in (
            'const competitorChartMedia = window.matchMedia(\n            "(max-width: 700px)"',
            "const currentCompetitorChartLayout = () => competitorChartLayout(",
            "window.innerWidth,\n            competitors.length",
            "const initialCompetitorChartLayout = currentCompetitorChartLayout();",
            "syncCompetitorChartHeight(initialCompetitorChartLayout);",
            "competitorChartElement.style.minHeight = height;",
        ):
            self.assertIn(marker, chart_setup)
        for marker in (
            "axisLabel: initialCompetitorChartLayout.axisLabel",
            'competitorChartMedia.addEventListener(\n              "change"',
            "const layout = currentCompetitorChartLayout();",
            "competitorChart.setOption({",
            "yAxis: { axisLabel: layout.axisLabel }",
            "reportChartDisposers.push(() => {",
        ):
            self.assertIn(marker, chart_block)
        for marker in (
            "const compact = viewportWidth <= 700;",
            "const count = Math.max(1, Math.floor(finiteNumber(rowCount) || 1));",
            "compact ? 280 : 320",
            "count * (compact ? 40 : 36) + (compact ? 68 : 78)",
            'overflow: "truncate"',
            'ellipsis: "…"',
            "interval: 0",
            "hideOverlap: false",
        ):
            self.assertIn(marker, layout_block)

        def expected_layout(viewport_width: int, row_count: int) -> tuple[bool, int]:
            compact = viewport_width <= 700
            height = max(
                280 if compact else 320,
                max(1, row_count) * (40 if compact else 36) + (68 if compact else 78),
            )
            return compact, height

        expected_cases = {
            (699, 1): (True, 280),
            (699, 5): (True, 280),
            (699, 12): (True, 548),
            (699, 30): (True, 1268),
            (701, 1): (False, 320),
            (701, 5): (False, 320),
            (701, 12): (False, 510),
            (701, 30): (False, 1158),
        }
        self.assertEqual(
            {case: expected_layout(*case) for case in expected_cases},
            expected_cases,
        )
        self.assertIn("const reportChartDisposers = [];", self.html)
        self.assertIn("while (reportChartDisposers.length)", self.html)
        self.assertIn(
            '<strong>${escapeHTML(row.name || "Без названия")}</strong>',
            self.html,
        )
        self.assertIn("competitorChartDataTableHTML(competitorRows)", self.html)

    def test_static_screen_assets_have_fixed_roles(self) -> None:
        self.assertIn("/static/brand/rw-plus-loader.png", self.html)
        self.assertIn("/static/brand/rw-plus-separated-blocks.png", self.html)
        self.assertIn("/static/brand/generated/aiv-progress-map.webp", self.html)
        self.assertIn("/static/brand/generated/aiv-report-cover.webp", self.html)
        self.assertIn(
            "/static/brand/generated/aiv-hero-desktop-v2.webp",
            self.html,
        )
        self.assertIn(
            "/static/brand/generated/aiv-hero-mobile-v2.webp",
            self.html,
        )
        self.assertIn('class="progress-visual"', self.html)
        self.assertIn('class="report-cover${safePreviewURL', self.html)
        self.assertNotIn("static.tildacdn.com", self.html)

    def test_report_hero_prefers_a_real_saved_site_viewport(self) -> None:
        self.assertIn("report.site_preview", self.html)
        self.assertIn("sitePreview:", self.html)
        self.assertIn("const reportCoverHTML =", self.html)
        self.assertIn("site-preview-[0-9a-f]{12}", self.html)
        self.assertIn("Первый экран сайта", self.html)
        self.assertIn("report-cover--site", self.html)
        self.assertIn("object-position: top center", self.html)
        self.assertIn("data-report-cover-image", self.html)
        self.assertIn("bindReportCoverFallback", self.html)
        self.assertIn("/static/brand/generated/aiv-report-cover.webp", self.html)

    def test_accessibility_and_safe_markdown_guards_exist(self) -> None:
        self.assertIn(":focus-visible", self.html)
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn("sanitizeMarkdown", self.html)
        self.assertIn("const allowed = new Set([", self.html)
        self.assertIn('["http:", "https:"]', self.html)
        self.assertIn('aria-label="Поделиться отчётом"', self.html)
        self.assertIn('aria-label="Начать новую проверку"', self.html)


if __name__ == "__main__":
    unittest.main()
