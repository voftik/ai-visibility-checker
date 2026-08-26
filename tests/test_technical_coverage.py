import unittest
import uuid

from sqlalchemy import delete, select

from app.db import SessionLocal, init_db
from app.models import (
    DomainProbe,
    ProbeType,
    RobotsRule,
    Run,
    RunArtifact,
    RunStatus,
    SitePage,
)
from app.services import crawler
from app.services.analyzer import (
    PROMPT_SET_VERSION,
    SITE_PAGE_MANIFEST_VERSION,
    _build_public_report,
    _entity_structured_data_types,
    _technical_page_coverage,
    _technical_summary,
    _uses_canonical_intent_taxonomy,
)


def _minimal_visibility_metrics() -> dict:
    return {
        "parent_discovery": {
            "web": {"score": 20, "state": "weak"},
        },
        "portfolio_visibility": {
            "web": {"score": 30, "state": "weak"},
        },
        "brand_knowledge": {
            "memory": {"specific_rate": 40, "state": "weak"},
        },
        "paired_web_lift": {},
        "model_consistency": {},
        "providers": [],
        "intents": [],
        "sentiment": {},
        "web": {},
        "memory": {},
        "competitors": [],
        "metric_note": "Тестовый контракт метрик.",
        "quality": {},
    }


class TechnicalPageCoverageContractTests(unittest.TestCase):
    def test_navigation_schema_does_not_count_as_entity_explanation(self) -> None:
        self.assertEqual(
            _entity_structured_data_types(["BreadcrumbList", "ListItem"]),
            [],
        )
        self.assertEqual(
            _entity_structured_data_types(
                ["BreadcrumbList", "Organization", "Product"]
            ),
            ["Organization", "Product"],
        )

    def test_unknown_denominator_never_becomes_full_coverage(self) -> None:
        coverage = _technical_page_coverage(2, None)

        self.assertEqual(coverage["evaluated_pages"], 2)
        self.assertIsNone(coverage["discovered_pages"])
        self.assertIsNone(coverage["coverage_rate"])
        self.assertEqual(coverage["coverage_state"], "unknown")
        self.assertEqual(
            coverage["scope_label"],
            "2 проверенные страницы · охват сайта не определён",
        )

    def test_known_denominator_distinguishes_limited_and_complete_scope(self) -> None:
        limited = _technical_page_coverage(
            2,
            {
                "pages": [
                    {"url": "https://example.com/"},
                    {"url": "https://example.com/services"},
                ],
                "discovered_count": 5,
                "selected_count": 2,
            },
        )
        complete = _technical_page_coverage(
            2,
            {
                "pages": [
                    {"url": "https://example.com/"},
                    {"url": "https://example.com/services"},
                ],
                "discovered_count": 2,
                "selected_count": 2,
            },
        )

        self.assertEqual(limited["coverage_rate"], 40.0)
        self.assertEqual(limited["coverage_state"], "limited")
        self.assertEqual(complete["coverage_rate"], 100.0)
        self.assertEqual(complete["coverage_state"], "complete")

    def test_inconsistent_manifest_is_reported_as_unknown(self) -> None:
        coverage = _technical_page_coverage(
            2,
            {
                "pages": [
                    {"url": "https://example.com/"},
                    {"url": "https://example.com/services"},
                ],
                "discovered_count": 1,
                "selected_count": 2,
                "coverage_state": "complete",
            },
        )

        self.assertIsNone(coverage["discovered_pages"])
        self.assertIsNone(coverage["coverage_rate"])
        self.assertEqual(coverage["coverage_state"], "unknown")

    def test_public_metric_carries_scope_instead_of_site_wide_claim(self) -> None:
        coverage = _technical_page_coverage(
            2,
            {
                "pages": [
                    {"url": "https://example.com/"},
                    {"url": "https://example.com/services"},
                ],
                "discovered_count": 5,
                "selected_count": 2,
            },
        )
        report = _build_public_report(
            profile={"brand_name": "Example"},
            technical={
                "score": 95,
                "state": "available",
                "coverage": coverage,
            },
            technical_review={},
            metrics=_minimal_visibility_metrics(),
        )

        metric = report["key_metrics"]["technical_access"]
        self.assertEqual(
            metric["label"],
            "Техническая готовность проверенного среза",
        )
        self.assertEqual(metric["state"], "limited")
        self.assertEqual(metric["access_state"], "available")
        self.assertEqual(metric["evaluated_pages"], 2)
        self.assertEqual(metric["discovered_pages"], 5)
        self.assertEqual(metric["coverage_rate"], 40.0)
        self.assertEqual(metric["coverage_state"], "limited")
        self.assertEqual(metric["coverage_label"], "Ограниченный срез")

    def test_public_report_marks_only_the_taxonomy_that_produced_answers(
        self,
    ) -> None:
        common = {
            "profile": {"brand_name": "Example"},
            "technical": {"score": 95, "state": "available"},
            "technical_review": {},
            "metrics": _minimal_visibility_metrics(),
        }

        legacy = _build_public_report(**common)
        canonical = _build_public_report(
            **common,
            canonical_intent_taxonomy=True,
        )

        self.assertEqual(
            legacy["methodology"]["intent_taxonomy_version"],
            "legacy-v1",
        )
        self.assertEqual(legacy["methodology"]["intent_definitions"], {})
        self.assertEqual(
            canonical["methodology"]["intent_taxonomy_version"],
            "canonical-v1",
        )
        self.assertIn("NB", canonical["methodology"]["intent_definitions"])
        self.assertIn(
            "задача",
            canonical["methodology"]["intent_definitions"]["NB"].lower(),
        )


class TechnicalSummaryCoverageIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_id = f"test-{uuid.uuid4()}"
        current_policy = crawler._body_read_policy(crawler.ProbeResult())
        home = SitePage(
            run_id=self.run_id,
            url="https://example.com/",
            page_kind="home",
            http_status=200,
            main_text="Home page",
            text_length=100,
            content_signals={
                "render_strategy": "server_rendered",
                "_body_read_policy": current_policy,
                "_source_body_sha256": "home-source",
            },
        )
        services = SitePage(
            run_id=self.run_id,
            url="https://example.com/services",
            page_kind="product",
            http_status=200,
            main_text="Services page",
            text_length=100,
            content_signals={
                "render_strategy": "server_rendered",
                "_body_read_policy": current_policy,
                "_source_body_sha256": "services-source",
            },
        )
        stale = SitePage(
            run_id=self.run_id,
            url="https://example.com/stale",
            page_kind="other",
            http_status=200,
            main_text="",
            text_length=0,
            content_signals={"render_strategy": "client_rendered_shell"},
        )
        pages = [
            ("https://example.com/", "home"),
            ("https://example.com/services", "product"),
        ]
        manifest_input = crawler._site_page_manifest_input("example.com")
        page_receipt = crawler._site_page_receipt(
            pages,
            {home.url: home, services.url: services},
        )
        relevance_receipt = crawler._selection_relevance_receipt(
            homepage_url=pages[0][0],
            candidates=[
                crawler._candidate_evidence_record(
                    pages[1][0],
                    source="test_fixture",
                )
            ],
            target=crawler.AUDIT_PAGE_DEFAULT,
            proposed=pages,
            attempts=[
                crawler._candidate_attempt(page, outcome="usable")
                for page in pages
            ],
            selected=pages,
        )
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=self.run_id,
                    domain="example.com",
                    status=RunStatus.completed,
                    config_json={},
                )
            )
            session.add_all([home, services, stale])
            session.add(
                RunArtifact(
                    run_id=self.run_id,
                    stage_key="site_discovery",
                    artifact_key="site_page_manifest",
                    status="completed",
                    prompt_version=SITE_PAGE_MANIFEST_VERSION,
                    input_json=manifest_input,
                    output_json={
                        "pages": crawler._selected_page_records(pages),
                        "expected_page_count": 2,
                        "discovered_count": 2,
                        "discovered_candidate_count": 2,
                        "selected_count": 2,
                        "selected_pages_sha256": (
                            crawler._selected_pages_sha256(pages)
                        ),
                        "page_scope": crawler.PAGE_SCOPE,
                        "selection_policy": manifest_input["selection_policy"],
                        "selection_exhausted": True,
                        "verified_exhaustion": True,
                        "legacy_snapshot": False,
                        "discovery_state": "complete",
                        "coverage_state": "complete",
                        "site_page_receipt": page_receipt,
                        "commercial_relevance_receipt": (
                            crawler._selection_relevance_projection(
                                relevance_receipt
                            )
                        ),
                    },
                )
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id == self.run_id))
            await session.commit()

    async def test_summary_uses_completed_manifest_and_current_page_set(
        self,
    ) -> None:
        summary = await _technical_summary(self.run_id)

        self.assertEqual(summary["evaluated_pages"], 2)
        self.assertEqual(summary["discovered_pages"], 2)
        self.assertEqual(summary["coverage_rate"], 100.0)
        self.assertEqual(summary["coverage_state"], "complete")
        self.assertEqual(
            {page["url"] for page in summary["pages"]},
            {
                "https://example.com/",
                "https://example.com/services",
            },
        )
        self.assertEqual(summary["summary"]["evaluated_pages"], 2)
        self.assertEqual(summary["summary"]["discovered_pages"], 2)

    async def test_bounded_manifest_uses_selected_pages_not_candidate_frontier(
        self,
    ) -> None:
        async with SessionLocal() as session:
            artifact = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "site_page_manifest",
                    )
                )
            ).scalar_one()
            output = dict(artifact.output_json)
            output.update(
                {
                    "discovered_count": 9,
                    "discovered_candidate_count": 9,
                    "selection_exhausted": True,
                    "coverage_state": "bounded",
                }
            )
            artifact.output_json = output
            await session.commit()

        summary = await _technical_summary(self.run_id)

        self.assertEqual(summary["evaluated_pages"], 2)
        self.assertEqual(summary["discovered_pages"], 9)
        self.assertEqual(summary["coverage_rate"], 22.2)
        self.assertEqual(summary["coverage_state"], "limited")
        self.assertEqual(
            {page["url"] for page in summary["pages"]},
            {
                "https://example.com/",
                "https://example.com/services",
            },
        )

    async def test_unknown_rendering_stays_unknown_without_crashing(self) -> None:
        async with SessionLocal() as session:
            pages = list(
                (
                    await session.execute(
                        select(SitePage).where(
                            SitePage.run_id == self.run_id,
                            SitePage.url.in_(
                                (
                                    "https://example.com/",
                                    "https://example.com/services",
                                )
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )
            for page in pages:
                page.content_signals = {
                    "render_strategy": "unknown",
                    "render_strategy_confidence": "low",
                }
            await session.commit()

        summary = await _technical_summary(self.run_id)
        server_html = next(
            fact
            for fact in summary["summary"]["facts"]
            if fact["key"] == "server_html"
        )

        self.assertIsNone(summary["rendering"]["server_readable_share"])
        self.assertEqual(server_html["value"], "2 из 2 не определено")
        self.assertEqual(server_html["state"], "unknown")

    async def test_jsonld_parse_failure_is_unknown_not_schema_absence(self) -> None:
        parse_evidence = {
            "script_count": 2,
            "parsed_count": 1,
            "failed_count": 1,
            "errors": [
                {
                    "script_index": 1,
                    "error_type": "json_decode_error",
                    "message": "Expecting value",
                }
            ],
            "state": "partial",
        }
        async with SessionLocal() as session:
            home = (
                await session.execute(
                    select(SitePage).where(
                        SitePage.run_id == self.run_id,
                        SitePage.url == "https://example.com/",
                    )
                )
            ).scalar_one()
            home.content_signals = {
                "render_strategy": "server_rendered",
                "structured_data_complete": False,
                "structured_data_types": ["Organization"],
                "jsonld": parse_evidence,
            }
            services = (
                await session.execute(
                    select(SitePage).where(
                        SitePage.run_id == self.run_id,
                        SitePage.url == "https://example.com/services",
                    )
                )
            ).scalar_one()
            services.content_signals = {
                "render_strategy": "server_rendered",
                "structured_data_complete": False,
                "structured_data_types": [],
                "jsonld": parse_evidence,
            }
            await session.commit()

        summary = await _technical_summary(self.run_id)
        home_result = next(
            page
            for page in summary["pages"]
            if page["url"] == "https://example.com/"
        )

        self.assertIs(home_result["structured_data_complete"], False)
        self.assertEqual(home_result["structured_data_types"], ["Organization"])
        self.assertEqual(home_result["jsonld"], parse_evidence)
        self.assertEqual(summary["structured_data"]["evaluated_pages"], 1)
        self.assertEqual(summary["structured_data"]["unknown_pages"], 1)
        self.assertEqual(summary["structured_data"]["entity_pages"], 1)

    async def test_taxonomy_marker_comes_from_saved_prompt_artifact(self) -> None:
        self.assertFalse(
            await _uses_canonical_intent_taxonomy(self.run_id)
        )
        async with SessionLocal() as session:
            session.add(
                RunArtifact(
                    run_id=self.run_id,
                    stage_key="scenario_design",
                    artifact_key="prompt_set",
                    status="completed",
                    prompt_version=PROMPT_SET_VERSION,
                    input_json={},
                    output_json={"prompts": []},
                )
            )
            await session.commit()

        self.assertTrue(
            await _uses_canonical_intent_taxonomy(self.run_id)
        )

    async def test_stale_manifest_version_cannot_claim_known_coverage(
        self,
    ) -> None:
        async with SessionLocal() as session:
            artifact = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "site_page_manifest",
                    )
                )
            ).scalar_one()
            artifact.prompt_version = "stale-site-page-manifest-v0"
            await session.commit()

        summary = await _technical_summary(self.run_id)

        self.assertEqual(summary["evaluated_pages"], 3)
        self.assertIsNone(summary["discovered_pages"])
        self.assertIsNone(summary["coverage_rate"])
        self.assertEqual(summary["coverage_state"], "unknown")

    async def test_foreign_domain_manifest_is_ignored(self) -> None:
        async with SessionLocal() as session:
            artifact = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == self.run_id,
                        RunArtifact.artifact_key == "site_page_manifest",
                    )
                )
            ).scalar_one()
            artifact.input_json = {
                "domain": "other.example",
                "page_scope": "complete_discovered_frontier_v1",
                "semantic_page_count_cap": None,
            }
            artifact.output_json = {
                "pages": [
                    {
                        "url": "https://other.example/",
                        "page_kind": "home",
                    }
                ],
                "discovered_count": 1,
                "selected_count": 1,
                "page_scope": "complete_discovered_frontier_v1",
                "semantic_page_count_cap": None,
                "coverage_state": "complete",
            }
            await session.commit()

        summary = await _technical_summary(self.run_id)

        self.assertEqual(summary["evaluated_pages"], 3)
        self.assertIsNone(summary["discovered_pages"])
        self.assertEqual(summary["coverage_state"], "unknown")

    async def test_transport_errors_are_excluded_from_access_denominators(
        self,
    ) -> None:
        async with SessionLocal() as session:
            session.add_all(
                [
                    DomainProbe(
                        run_id=self.run_id,
                        domain="example.com",
                        user_agent_label="GPTBot",
                        user_agent_string="GPTBot",
                        target_url="https://example.com/",
                        probe_type=ProbeType.main_page,
                        error_class="connect_timeout",
                        challenge_detected=False,
                        body_looks_empty=True,
                    ),
                    DomainProbe(
                        run_id=self.run_id,
                        domain="example.com",
                        user_agent_label="ClaudeBot",
                        user_agent_string="ClaudeBot",
                        target_url="https://example.com/",
                        probe_type=ProbeType.main_page,
                        http_status=200,
                        content_extractable_text_length=500,
                        content_signals={},
                        challenge_detected=False,
                        body_looks_empty=False,
                    ),
                ]
            )
            await session.commit()

        technical = await _technical_summary(self.run_id)
        summary = technical["summary"]
        families = {family["name"]: family for family in technical["families"]}
        home = next(
            page
            for page in technical["pages"]
            if page["url"] == "https://example.com/"
        )

        self.assertEqual(summary["passed_checks"], 1)
        self.assertEqual(summary["total_checks"], 1)
        self.assertEqual(summary["unknown_checks"], 1)
        self.assertEqual(summary["expected_checks"], 2)
        self.assertEqual(home["access_rate"], 100.0)
        self.assertEqual(home["unknown_checks"], 1)
        self.assertEqual(home["unknown_families"], ["OpenAI"])
        self.assertIsNone(families["OpenAI"]["access_rate"])
        self.assertEqual(families["OpenAI"]["total_count"], 0)
        self.assertEqual(families["OpenAI"]["unknown_count"], 1)
        self.assertEqual(families["OpenAI"]["state"], "unknown")
        self.assertEqual(families["Anthropic"]["access_rate"], 100)
        self.assertEqual(families["Anthropic"]["state"], "available")

    async def test_partial_robots_rules_never_look_globally_open(self) -> None:
        async with SessionLocal() as session:
            session.add(
                RobotsRule(
                    run_id=self.run_id,
                    domain="example.com",
                    bot_name="GPTBot",
                    rule="partial",
                    raw_directives=(
                        "User-agent: GPTBot\n"
                        "Disallow: /private\n"
                    ),
                )
            )
            await session.commit()

        technical = await _technical_summary(self.run_id)

        self.assertEqual(technical["robots"]["state"], "partial")
        self.assertEqual(technical["robots"]["partial_rules"], ["GPTBot"])
        robots_fact = next(
            fact
            for fact in technical["summary"]["facts"]
            if fact["key"] == "robots"
        )
        self.assertEqual(robots_fact["value"], "Открыты частично")
        self.assertEqual(robots_fact["state"], "warning")
