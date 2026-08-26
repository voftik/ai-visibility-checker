import json
import unittest
import uuid
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, select

from app.db import SessionLocal, init_db
from app.models import DomainProbe, ProbeType, Run, RunArtifact, RunStatus, SitePage
from app.services import crawler


def _probe(
    url: str,
    *,
    status: int = 200,
    body: str = "",
) -> crawler.ProbeResult:
    return crawler.ProbeResult(
        http_status=status,
        final_url=url,
        full_text=body,
        content_type="text/html; charset=utf-8",
    )


class SitePageManifestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id.in_(self.run_ids)))
            await session.commit()

    async def _create_run(
        self,
        *,
        artifact: RunArtifact | None = None,
        artifacts: list[RunArtifact] | None = None,
        page_urls: list[tuple[str, str]] | None = None,
    ) -> str:
        run_id = f"test-manifest-{uuid.uuid4()}"
        self.run_ids.append(run_id)
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.crawling,
                    config_json={},
                )
            )
            if artifact is not None:
                artifact.run_id = run_id
                session.add(artifact)
            for persisted in artifacts or []:
                persisted.run_id = run_id
                session.add(persisted)
            for url, page_kind in page_urls or []:
                session.add(
                    SitePage(
                        run_id=run_id,
                        url=url,
                        page_kind=page_kind,
                        http_status=200,
                        main_text="Saved page",
                        text_length=10,
                        content_signals={
                            "_body_truncated": False,
                            "_body_read_policy": {
                                "version": crawler.BODY_READ_POLICY_VERSION,
                                "response_size_limit_bytes": None,
                                "result": "complete_eof",
                                "terminal": True,
                            },
                            "_source_body_sha256": "saved-source",
                        },
                    )
                )
            await session.commit()
        return run_id

    def _completed_artifacts(
        self,
        pages: list[tuple[str, str]],
    ) -> list[RunArtifact]:
        receipt = {
            "expected_page_count": len(pages),
            "usable_page_count": len(pages),
            "pages": [
                {
                    "ordinal": ordinal,
                    "url": url,
                    "page_kind": page_kind,
                    "http_status": 200,
                    "text_length": 10,
                    "content_sha256": crawler.hashlib.sha256(b"Saved page").hexdigest(),
                    "source_body_sha256": "saved-source",
                    "body_policy_version": crawler.BODY_READ_POLICY_VERSION,
                    "usable": True,
                }
                for ordinal, (url, page_kind) in enumerate(pages)
            ],
            "retryable_urls": [],
            "unusable_urls": [],
            "legacy_snapshot": False,
            "complete": True,
        }
        receipt["receipt_sha256"] = crawler._json_sha256(receipt)
        candidates = [
            crawler._candidate_evidence_record(url, source="test_fixture")
            for url, _page_kind in pages[1:]
        ]
        attempts = [
            crawler._candidate_attempt(page, outcome="usable") for page in pages
        ]
        relevance_receipt = crawler._selection_relevance_receipt(
            homepage_url=pages[0][0],
            candidates=candidates,
            target=crawler.AUDIT_PAGE_DEFAULT,
            proposed=pages,
            attempts=attempts,
            selected=pages,
        )
        manifest = RunArtifact(
            run_id="replaced-by-helper",
            stage_key="site_discovery",
            artifact_key=crawler.SITE_PAGE_MANIFEST_KEY,
            status="completed",
            prompt_version=crawler.SITE_PAGE_MANIFEST_VERSION,
            input_json=crawler._site_page_manifest_input("example.com"),
            output_json={
                "pages": crawler._selected_page_records(pages),
                "expected_page_count": len(pages),
                "discovered_count": len(pages),
                "discovered_candidate_count": len(pages),
                "selected_count": len(pages),
                "selected_pages_sha256": crawler._selected_pages_sha256(pages),
                "page_scope": crawler.PAGE_SCOPE,
                "selection_policy": crawler._site_page_manifest_input("example.com")[
                    "selection_policy"
                ],
                "selection_exhausted": True,
                "verified_exhaustion": len(pages) < crawler.AUDIT_PAGE_MIN,
                "legacy_snapshot": False,
                "discovery_state": "complete",
                "coverage_state": "complete",
                "site_page_receipt": receipt,
                "commercial_relevance_receipt": (
                    crawler._selection_relevance_projection(relevance_receipt)
                ),
            },
        )
        relevance = RunArtifact(
            run_id="replaced-by-helper",
            stage_key="site_discovery",
            artifact_key=crawler.PAGE_RELEVANCE_ARTIFACT_KEY,
            status="completed",
            prompt_version=crawler.COMMERCIAL_RELEVANCE_POLICY_VERSION,
            input_json={
                "domain": "example.com",
                "policy_version": crawler.COMMERCIAL_RELEVANCE_POLICY_VERSION,
            },
            output_json=relevance_receipt,
        )
        return [manifest, relevance]

    def test_manifest_rejects_pages_from_another_domain(self) -> None:
        manifest = {
            "pages": [
                {
                    "ordinal": 0,
                    "url": "https://other.example/",
                    "page_kind": "home",
                },
            ],
            "expected_page_count": 1,
            "discovered_count": 1,
            "discovered_candidate_count": 1,
            "selected_count": 1,
            "page_scope": crawler.PAGE_SCOPE,
            "coverage_state": "complete",
        }
        self.assertIsNone(
            crawler._manifest_pages(
                manifest,
                domain="example.com",
            )
        )

    async def test_site_page_transient_probe_budget_becomes_terminal(self) -> None:
        run_id = await self._create_run()
        url = "https://example.com/"
        for attempt in range(1, crawler.SITE_PAGE_MAX_PROBE_ATTEMPTS + 1):
            await crawler._store_site_page(
                run_id,
                url=url,
                page_kind="home",
                probe=_probe(url, status=503, body="Unavailable"),
            )
            async with SessionLocal() as session:
                page = (
                    await session.execute(
                        select(SitePage).where(
                            SitePage.run_id == run_id,
                            SitePage.url == url,
                        )
                    )
                ).scalar_one()
            self.assertEqual(
                page.content_signals["_probe_attempts"],
                attempt,
            )
            self.assertEqual(
                crawler._site_page_is_retryable(page),
                attempt < crawler.SITE_PAGE_MAX_PROBE_ATTEMPTS,
            )

    def test_manifest_rejects_legacy_sampled_page_contract(self) -> None:
        self.assertIsNone(
            crawler._manifest_pages(
                {
                    "pages": [{"url": "https://example.com/", "page_kind": "home"}],
                    "discovered_count": 20,
                    "selected_count": 1,
                    "selection_limit": 6,
                    "coverage_state": "limited",
                },
                domain="example.com",
            )
        )

    async def test_cross_domain_redirect_body_is_not_used_as_site_content(
        self,
    ) -> None:
        external = crawler.ProbeResult(
            http_status=200,
            final_url="https://other.example/landing",
            full_text="<main>Чужой сайт</main>",
            body_sample="<main>Чужой сайт</main>",
        )
        with (
            patch.object(crawler, "get_pool", return_value=None),
            patch.object(
                crawler,
                "_do_probe",
                AsyncMock(return_value=external),
            ),
        ):
            result, _transport, _proxy = await crawler._probe_with_transport(
                url="https://example.com/",
                user_agent="Test",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(result.error_class, "cross_domain_redirect")
        self.assertEqual(result.full_text, "")
        self.assertEqual(result.body_sample, "")
        self.assertTrue(result.body_looks_empty)
        self.assertEqual(result.final_url, "https://other.example/landing")

    async def test_completed_manifest_resumes_missing_pages_and_is_idempotent(
        self,
    ) -> None:
        pages = [
            ("https://example.com/", "home"),
            ("https://example.com/services", "product"),
            ("https://example.com/about", "about"),
        ]
        run_id = await self._create_run(
            artifacts=self._completed_artifacts(pages),
            page_urls=[
                ("https://example.com/", "other"),
                ("https://example.com/stale", "other"),
            ],
        )

        async def probe_missing(**kwargs):
            return _probe(kwargs["url"], body="<main>Recovered</main>"), {}, None

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=probe_missing),
            ) as probe_call,
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            selected = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )
        self.assertEqual(selected, pages)
        self.assertEqual(
            [call.kwargs["url"] for call in probe_call.await_args_list],
            ["https://example.com/services", "https://example.com/about"],
        )

        async with SessionLocal() as session:
            stored_pages = list(
                await session.execute(
                    select(SitePage.url, SitePage.page_kind)
                    .where(SitePage.run_id == run_id)
                    .order_by(SitePage.url)
                )
            )
        self.assertEqual(stored_pages, sorted(pages))

        with (
            patch.object(crawler, "_probe_with_transport", AsyncMock()) as no_probe,
            patch.object(crawler, "update_progress", AsyncMock()),
        ):
            second = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )
        self.assertEqual(second, pages)
        no_probe.assert_not_awaited()

        jobs = crawler._build_jobs(
            ["example.com"],
            ["GPTBot", "ClaudeBot"],
            second,
        )
        self.assertEqual(len(jobs), 1 + len(pages) * 2)
        self.assertEqual(jobs[0].probe_type, ProbeType.robots_txt)
        main_pairs = {
            (job.target_url, job.user_agent_label)
            for job in jobs
            if job.probe_type is ProbeType.main_page
        }
        self.assertEqual(
            main_pairs,
            {
                (url, user_agent)
                for url, _ in pages
                for user_agent in ("GPTBot", "ClaudeBot")
            },
        )

    async def test_absent_incomplete_and_mismatched_manifests_rediscover(
        self,
    ) -> None:
        exact_input = crawler._site_page_manifest_input("example.com")
        cases = {
            "absent": None,
            "incomplete": RunArtifact(
                run_id="replaced-by-helper",
                stage_key="site_discovery",
                artifact_key=crawler.SITE_PAGE_MANIFEST_KEY,
                status="running",
                prompt_version=crawler.SITE_PAGE_MANIFEST_VERSION,
                input_json=exact_input,
            ),
            "mismatched": RunArtifact(
                run_id="replaced-by-helper",
                stage_key="site_discovery",
                artifact_key=crawler.SITE_PAGE_MANIFEST_KEY,
                status="completed",
                prompt_version="old-site-page-manifest",
                input_json=exact_input,
                output_json={
                    "pages": [{"url": "https://example.com/", "page_kind": "home"}],
                    "discovered_count": 1,
                    "selected_count": 1,
                    "page_scope": "complete_discovered_frontier_v1",
                    "semantic_page_count_cap": None,
                    "coverage_state": "complete",
                },
            ),
        }

        for label, artifact in cases.items():
            with self.subTest(label=label):
                run_id = await self._create_run(
                    artifact=artifact,
                    page_urls=[("https://example.com/", "home")],
                )
                persisted_after_completion: list[str] = []
                original_store = crawler._store_site_page

                async def assert_manifest_then_store(*args, **kwargs):
                    async with SessionLocal() as session:
                        current = (
                            await session.execute(
                                select(RunArtifact).where(
                                    RunArtifact.run_id == run_id,
                                    RunArtifact.artifact_key
                                    == crawler.SITE_PAGE_MANIFEST_KEY,
                                )
                            )
                        ).scalar_one()
                        persisted_after_completion.append(current.status)
                    await original_store(*args, **kwargs)

                async def discovery_probe(**kwargs):
                    url = kwargs["url"]
                    if url == "https://example.com/":
                        return (
                            _probe(
                                url,
                                body='<main>Home</main><a href="/services">Services</a>',
                            ),
                            {},
                            None,
                        )
                    if url == "https://example.com/sitemap.xml":
                        return _probe(url, status=404), {}, None
                    if url == "https://example.com/robots.txt":
                        return _probe(url, status=404), {}, None
                    if url == "https://example.com/services":
                        return _probe(url, body="<main>Services</main>"), {}, None
                    self.fail(f"Unexpected URL: {url}")

                with (
                    patch.object(
                        crawler,
                        "_probe_with_transport",
                        AsyncMock(side_effect=discovery_probe),
                    ) as probe_call,
                    patch.object(
                        crawler,
                        "_store_site_page",
                        new=assert_manifest_then_store,
                    ),
                    patch.object(crawler, "update_progress", AsyncMock()),
                    patch.object(crawler.asyncio, "sleep", AsyncMock()),
                ):
                    selected = await crawler.discover_site_pages(
                        run_id,
                        "example.com",
                        timeout_seconds=5,
                        concurrency=1,
                    )

                self.assertEqual(
                    [call.kwargs["url"] for call in probe_call.await_args_list],
                    [
                        "https://example.com/",
                        "https://example.com/robots.txt",
                        "https://example.com/sitemap.xml",
                        "https://example.com/services",
                    ],
                )
                self.assertEqual(
                    selected,
                    [
                        ("https://example.com/", "home"),
                        ("https://example.com/services", "product"),
                    ],
                )
                self.assertEqual(
                    persisted_after_completion,
                    ["running", "running"],
                )

                async with SessionLocal() as session:
                    manifest = (
                        await session.execute(
                            select(RunArtifact).where(
                                RunArtifact.run_id == run_id,
                                RunArtifact.artifact_key
                                == crawler.SITE_PAGE_MANIFEST_KEY,
                            )
                        )
                    ).scalar_one()
                    self.assertEqual(manifest.status, "completed")
                    self.assertEqual(
                        manifest.prompt_version,
                        crawler.SITE_PAGE_MANIFEST_VERSION,
                    )
                    self.assertEqual(manifest.input_json, exact_input)
                    self.assertEqual(
                        manifest.output_json["pages"],
                        crawler._selected_page_records(selected),
                    )
                    self.assertEqual(manifest.output_json["discovered_count"], 2)
                    self.assertEqual(manifest.output_json["selected_count"], 2)
                    self.assertEqual(
                        manifest.output_json["page_scope"], crawler.PAGE_SCOPE
                    )
                    self.assertEqual(manifest.output_json["expected_page_count"], 2)
                    self.assertEqual(
                        manifest.output_json["selected_pages_sha256"],
                        crawler._selected_pages_sha256(selected),
                    )
                    self.assertEqual(manifest.output_json["coverage_state"], "bounded")
                    self.assertFalse(
                        manifest.output_json["sitemap_discovery"]["complete"]
                    )

    async def test_urlset_materializes_complete_semantic_frontier_without_cap(
        self,
    ) -> None:
        run_id = await self._create_run()
        sitemap = """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/</loc></url>
              <url><loc>https://example.com/services</loc></url>
              <url><loc>https://example.com/about</loc></url>
              <url><loc>https://example.com/contact</loc></url>
            </urlset>
        """

        async def discovery_probe(**kwargs):
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body="<main>Home</main>"), {}, None
            if url == "https://example.com/sitemap.xml":
                return _probe(url, body=sitemap), {}, None
            if url == "https://example.com/robots.txt":
                return _probe(url, status=404), {}, None
            if url in {
                "https://example.com/services",
                "https://example.com/about",
                "https://example.com/contact",
            }:
                return _probe(url, body=f"<main>{url}</main>"), {}, None
            self.fail(f"Unexpected URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ),
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            selected = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(
            selected,
            [
                ("https://example.com/", "home"),
                ("https://example.com/services", "product"),
                ("https://example.com/about", "about"),
                ("https://example.com/contact", "contact"),
            ],
        )
        async with SessionLocal() as session:
            manifest = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                    )
                )
            ).scalar_one()
        self.assertEqual(manifest.output_json["discovered_count"], 4)
        self.assertEqual(manifest.output_json["selected_count"], 4)
        self.assertEqual(manifest.output_json["expected_page_count"], 4)
        self.assertEqual(manifest.output_json["coverage_state"], "complete")

    def test_sitemap_tail_after_six_hundred_is_considered_for_selection(
        self,
    ) -> None:
        locations = "".join(
            f"<url><loc>https://example.com/page-{index}</loc></url>"
            for index in range(600)
        )
        sitemap = (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + locations
            + "<url><loc>https://example.com/pricing</loc></url></urlset>"
        )
        links = crawler._links_from_sitemap(
            sitemap,
            "https://example.com/sitemap.xml",
            "example.com",
        )
        self.assertEqual(len(links), 601)
        self.assertIn("https://example.com/pricing", links)
        selected = crawler._semantic_frontier_urls(
            "https://example.com/",
            links,
        )
        self.assertEqual(len(selected), crawler.AUDIT_PAGE_DEFAULT)
        self.assertEqual(selected[0], ("https://example.com/", "home"))
        self.assertIn(("https://example.com/pricing", "pricing"), selected)

    def test_opaque_slug_is_ranked_from_source_grounded_link_evidence(
        self,
    ) -> None:
        opaque = crawler._candidate_evidence_record(
            "https://example.com/p/7f3a91",
            source="rendered_navigation",
            anchor_text="Enterprise AI analytics platform",
            title="Book a demo",
            primary_snippet="Product platform for revenue teams. Request a demo.",
        )

        relevance = crawler._commercial_relevance(opaque)

        self.assertEqual(relevance["page_kind"], "product")
        self.assertTrue(relevance["commercially_relevant"])
        self.assertGreaterEqual(
            relevance["commercial_relevance_score"],
            crawler.COMMERCIAL_RELEVANCE_THRESHOLD,
        )

    def test_specific_noncommercial_anchor_is_not_poisoned_by_parent_copy(
        self,
    ) -> None:
        about = crawler._candidate_evidence_record(
            "https://example.com/x/7f3a91",
            source="server_navigation",
            anchor_text="About us",
            primary_snippet=(
                "Products and services. Book a demo. About us. Contact sales."
            ),
        )

        relevance = crawler._commercial_relevance(about)

        self.assertEqual(relevance["page_kind"], "about")
        self.assertFalse(relevance["commercially_relevant"])

    def test_relevance_receipt_rejects_rewritten_anchor_even_with_new_digest(
        self,
    ) -> None:
        homepage = "https://example.com/"
        candidate = crawler._candidate_evidence_record(
            "https://example.com/p/opaque",
            source="rendered_navigation",
            anchor_text="Analytics product",
            title="Book a demo",
        )
        pages = [
            (homepage, "home"),
            ("https://example.com/p/opaque", "product"),
        ]
        receipt = crawler._selection_relevance_receipt(
            homepage_url=homepage,
            candidates=[candidate],
            target=crawler.AUDIT_PAGE_DEFAULT,
            proposed=pages,
            attempts=[
                crawler._candidate_attempt(page, outcome="usable") for page in pages
            ],
            selected=pages,
        )
        self.assertTrue(
            crawler._selection_relevance_receipt_is_valid(
                receipt,
                domain="example.com",
            )
        )

        tampered = deepcopy(receipt)
        tampered["candidates"][0]["discoveries"][0]["anchor_text"] = "About us"
        tampered_body = dict(tampered)
        tampered_body.pop("receipt_sha256")
        tampered["receipt_sha256"] = crawler._json_sha256(tampered_body)

        self.assertFalse(
            crawler._selection_relevance_receipt_is_valid(
                tampered,
                domain="example.com",
            )
        )

    async def test_client_shell_with_eight_ssr_links_admits_opaque_js_product(
        self,
    ) -> None:
        run_id = await self._create_run()
        generic_slugs = [
            "about",
            "contact",
            "blog",
            "news",
            "faq",
            "team",
            "careers",
            "partners",
        ]
        generic_urls = {f"https://example.com/{slug}" for slug in generic_slugs}
        links = "".join(
            f'<a href="/{slug}">{slug.title()}</a>' for slug in generic_slugs
        )
        homepage_html = (
            "<html><head><title>Example</title></head><body>"
            f'<div id="root">{links}</div>'
            "<script></script><script></script><script></script>"
            "</body></html>"
        )
        opaque_url = "https://example.com/p/7f3a91"
        opaque_evidence = [
            crawler._candidate_evidence_record(
                opaque_url,
                source="rendered_navigation",
                anchor_text="Enterprise AI analytics platform",
                title="Book a demo",
                primary_snippet="Product platform for revenue teams. Request a demo.",
            )
        ]
        rendered_receipt = {
            "state": "completed",
            "candidate_count": 1,
            "candidate_urls_sha256": crawler._json_sha256([opaque_url]),
            "candidate_evidence": opaque_evidence,
            "candidate_evidence_sha256": crawler._json_sha256(opaque_evidence),
        }

        async def discovery_probe(**kwargs):
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body=homepage_html), {}, None
            if url in {
                "https://example.com/robots.txt",
                "https://example.com/sitemap.xml",
            }:
                return _probe(url, status=404), {}, None
            if url == opaque_url or url in generic_urls:
                return (
                    _probe(
                        url,
                        body=f"<main>Readable content for {url}</main>",
                    ),
                    {},
                    None,
                )
            self.fail(f"Unexpected URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ),
            patch.object(
                crawler,
                "_links_from_rendered_homepage",
                AsyncMock(return_value=([opaque_url], rendered_receipt)),
            ) as rendered,
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            selected = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        rendered.assert_awaited_once()
        self.assertEqual(len(selected), crawler.AUDIT_PAGE_DEFAULT)
        self.assertLessEqual(len(selected), crawler.AUDIT_PAGE_HARD_MAX)
        self.assertIn((opaque_url, "product"), selected)

        async with SessionLocal() as session:
            artifacts = {
                artifact.artifact_key: artifact
                for artifact in (
                    await session.execute(
                        select(RunArtifact).where(RunArtifact.run_id == run_id)
                    )
                ).scalars()
            }
        manifest = artifacts[crawler.SITE_PAGE_MANIFEST_KEY]
        relevance_artifact = artifacts[crawler.PAGE_RELEVANCE_ARTIFACT_KEY]
        self.assertNotIn(
            "candidate_evidence",
            manifest.output_json["rendered_navigation_discovery"],
        )
        self.assertNotIn(
            "Enterprise AI analytics platform",
            json.dumps(manifest.output_json, ensure_ascii=False),
        )
        opaque_record = next(
            item
            for item in relevance_artifact.output_json["candidates"]
            if item["url"] == opaque_url
        )
        self.assertEqual(opaque_record["page_kind"], "product")
        self.assertEqual(
            opaque_record["discoveries"][0]["anchor_text"],
            "Enterprise AI analytics platform",
        )
        self.assertTrue(
            crawler._selection_relevance_receipt_is_valid(
                relevance_artifact.output_json,
                domain="example.com",
            )
        )
        self.assertTrue(
            crawler._relevance_artifact_matches_projection(
                relevance_artifact,
                manifest.output_json["commercial_relevance_receipt"],
                domain="example.com",
            )
        )
        tampered_projection = deepcopy(
            manifest.output_json["commercial_relevance_receipt"]
        )
        opaque_projection = next(
            item
            for item in tampered_projection["selected_evidence"]
            if item["url"] == opaque_url
        )
        opaque_projection["commercial_relevance_score"] += 1
        projection_body = dict(tampered_projection)
        projection_body.pop("projection_sha256")
        tampered_projection["projection_sha256"] = crawler._json_sha256(projection_body)
        self.assertFalse(
            crawler._relevance_artifact_matches_projection(
                relevance_artifact,
                tampered_projection,
                domain="example.com",
            )
        )

    async def test_rendered_navigation_worker_uses_versioned_mode(self) -> None:
        payload = {
            "ok": True,
            "worker_protocol_version": crawler.RENDERED_NAVIGATION_WORKER_PROTOCOL,
            "mode": "navigation",
            "final_url": "https://example.com/",
            "anchors": [
                {
                    "href": "https://example.com/p/opaque",
                    "anchor_text": "Analytics platform",
                    "title": "Book a demo",
                    "primary_snippet": "Product for revenue teams",
                }
            ],
            "anchor_count": 1,
            "truncated": False,
        }
        process = SimpleNamespace(
            communicate=AsyncMock(
                return_value=(json.dumps(payload).encode("utf-8"), b"")
            ),
            returncode=0,
        )
        with (
            patch.object(
                crawler.settings,
                "SITE_PREVIEW_WORKER_COMMAND",
                "/usr/local/bin/aiv-site-preview",
            ),
            patch.object(
                crawler.asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=process),
            ) as spawn,
            patch.object(crawler, "_validate_public_url", AsyncMock()),
        ):
            links, receipt = await crawler._links_from_rendered_homepage(
                "https://example.com/",
                domain="example.com",
                timeout_seconds=20,
            )

        command = spawn.await_args.args
        self.assertIn("--mode", command)
        self.assertEqual(command[command.index("--mode") + 1], "navigation")
        self.assertEqual(links, ["https://example.com/p/opaque"])
        self.assertEqual(receipt["state"], "completed")
        self.assertEqual(
            receipt["worker_protocol_version"],
            crawler.RENDERED_NAVIGATION_WORKER_PROTOCOL,
        )
        self.assertEqual(
            receipt["candidate_evidence"][0]["discoveries"][0]["title"],
            "Book a demo",
        )

    async def test_failed_rendered_read_cannot_claim_complete_discovery(
        self,
    ) -> None:
        run_id = await self._create_run()
        slugs = [
            "about",
            "contact",
            "blog",
            "news",
            "faq",
            "team",
            "careers",
            "partners",
        ]
        urls = [f"https://example.com/{slug}" for slug in slugs]
        links = "".join(f'<a href="/{slug}">{slug.title()}</a>' for slug in slugs)
        homepage_html = (
            "<html><head><title>Example</title></head><body>"
            f'<div id="root">{links}</div>'
            "<script></script><script></script><script></script>"
            "</body></html>"
        )
        sitemap = (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://example.com/</loc></url>"
            + "".join(f"<url><loc>{url}</loc></url>" for url in urls)
            + "</urlset>"
        )

        async def discovery_probe(**kwargs):
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body=homepage_html), {}, None
            if url == "https://example.com/robots.txt":
                return _probe(url, status=404), {}, None
            if url == "https://example.com/sitemap.xml":
                return _probe(url, body=sitemap), {}, None
            if url in urls:
                return _probe(url, body=f"<main>Content for {url}</main>"), {}, None
            self.fail(f"Unexpected URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ),
            patch.object(
                crawler,
                "_links_from_rendered_homepage",
                AsyncMock(
                    return_value=(
                        [],
                        {
                            "state": "failed_non_blocking",
                            "candidate_count": 0,
                            "error_class": "RuntimeError",
                        },
                    )
                ),
            ),
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        async with SessionLocal() as session:
            manifest = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                    )
                )
            ).scalar_one()
        self.assertTrue(manifest.output_json["sitemap_discovery"]["complete"])
        self.assertEqual(
            manifest.output_json["rendered_navigation_discovery"]["state"],
            "failed_non_blocking",
        )
        self.assertEqual(manifest.output_json["discovery_state"], "terminal_partial")
        self.assertEqual(manifest.output_json["coverage_state"], "bounded")

    async def test_more_than_eight_semantic_pages_are_bounded_in_manifest(
        self,
    ) -> None:
        run_id = await self._create_run()
        urls = [f"https://example.com/services/{index}" for index in range(18)]
        sitemap = (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(f"<url><loc>{url}</loc></url>" for url in urls)
            + "</urlset>"
        )

        async def discovery_probe(**kwargs):
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body="<main>Home</main>"), {}, None
            if url == "https://example.com/robots.txt":
                return _probe(url, status=404), {}, None
            if url == "https://example.com/sitemap.xml":
                return _probe(url, body=sitemap), {}, None
            if url in urls:
                return _probe(url, body=f"<main>{url}</main>"), {}, None
            self.fail(f"Unexpected discovery URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ),
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            selected = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(len(selected), crawler.AUDIT_PAGE_DEFAULT)
        self.assertEqual(selected[0], ("https://example.com/", "home"))
        async with SessionLocal() as session:
            manifest = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                    )
                )
            ).scalar_one()
        self.assertEqual(
            manifest.output_json["selected_count"],
            crawler.AUDIT_PAGE_DEFAULT,
        )
        self.assertEqual(manifest.output_json["discovered_count"], 19)
        self.assertEqual(
            manifest.output_json["selection_policy"]["max_page_count"],
            crawler.AUDIT_PAGE_HARD_MAX,
        )

    async def test_sitemap_index_is_walked_once_with_complete_manifest(
        self,
    ) -> None:
        run_id = await self._create_run()
        root_index = """
            <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <sitemap><loc>https://example.com/nested.xml</loc></sitemap>
              <sitemap><loc>https://example.com/nested.xml</loc></sitemap>
            </sitemapindex>
        """
        nested = """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/</loc></url>
              <url><loc>https://example.com/services</loc></url>
              <url><loc>https://example.com/about</loc></url>
            </urlset>
        """

        async def discovery_probe(**kwargs):
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body="<main>Home</main>"), {}, None
            if url == "https://example.com/sitemap.xml":
                return _probe(url, body=root_index), {}, None
            if url == "https://example.com/robots.txt":
                return _probe(url, status=404), {}, None
            if url == "https://example.com/nested.xml":
                return _probe(url, body=nested), {}, None
            if url == "https://example.com/services":
                return _probe(url, body="<main>Services</main>"), {}, None
            if url == "https://example.com/about":
                return _probe(url, body="<main>About</main>"), {}, None
            self.fail(f"Unexpected URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ) as probe_call,
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            selected = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(
            selected,
            [
                ("https://example.com/", "home"),
                ("https://example.com/services", "product"),
                ("https://example.com/about", "about"),
            ],
        )
        requested = [call.kwargs["url"] for call in probe_call.await_args_list]
        self.assertEqual(requested.count("https://example.com/nested.xml"), 1)
        async with SessionLocal() as session:
            manifest = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                    )
                )
            ).scalar_one()
        sitemap_manifest = manifest.output_json["sitemap_discovery"]
        self.assertTrue(sitemap_manifest["complete"])
        self.assertEqual(sitemap_manifest["document_count"], 2)
        self.assertEqual(sitemap_manifest["candidate_count"], 3)
        self.assertEqual(manifest.output_json["discovered_count"], 3)
        self.assertEqual(manifest.output_json["coverage_state"], "complete")

    async def test_incomplete_sitemap_graph_resumes_successful_documents(
        self,
    ) -> None:
        run_id = await self._create_run()
        root_index = """
            <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <sitemap><loc>https://example.com/child.xml</loc></sitemap>
            </sitemapindex>
        """
        recovered_child = """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/</loc></url>
              <url><loc>https://example.com/TAIL_MARKER</loc></url>
            </urlset>
        """
        child_attempts = 0

        async def discovery_probe(**kwargs):
            nonlocal child_attempts
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body="<main>Home</main>"), {}, None
            if url == "https://example.com/robots.txt":
                return _probe(url, status=404), {}, None
            if url == "https://example.com/sitemap.xml":
                return _probe(url, body=root_index), {}, None
            if url == "https://example.com/child.xml":
                child_attempts += 1
                if child_attempts == 1:
                    return _probe(url, status=503), {}, None
                return _probe(url, body=recovered_child), {}, None
            if url == "https://example.com/TAIL_MARKER":
                return _probe(url, body="<main>Tail</main>"), {}, None
            self.fail(f"Unexpected URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ) as probe_call,
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            with self.assertRaises(crawler.SitemapFrontierIncomplete):
                await crawler.discover_site_pages(
                    run_id,
                    "example.com",
                    timeout_seconds=5,
                    concurrency=1,
                )

            async with SessionLocal() as session:
                partial_artifact = (
                    await session.execute(
                        select(RunArtifact).where(
                            RunArtifact.run_id == run_id,
                            RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                        )
                    )
                ).scalar_one()
                partial_pages = list(
                    (
                        await session.execute(
                            select(SitePage).where(SitePage.run_id == run_id)
                        )
                    ).scalars()
                )
            self.assertEqual(partial_artifact.status, "running")
            self.assertTrue(
                partial_artifact.output_json["sitemap_discovery"]["resume_required"]
            )
            self.assertEqual(partial_pages, [])

            second = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(
            second,
            [
                ("https://example.com/", "home"),
                ("https://example.com/TAIL_MARKER", "other"),
            ],
        )
        requested = [call.kwargs["url"] for call in probe_call.await_args_list]
        self.assertEqual(requested.count("https://example.com/sitemap.xml"), 1)
        self.assertEqual(requested.count("https://example.com/child.xml"), 2)
        async with SessionLocal() as session:
            artifact = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                    )
                )
            ).scalar_one()
        self.assertEqual(artifact.status, "completed")
        self.assertTrue(artifact.output_json["sitemap_discovery"]["complete"])
        self.assertEqual(
            artifact.output_json["sitemap_discovery"]["candidate_count"],
            2,
        )

    async def test_incomplete_frontier_never_reaches_analysis(self) -> None:
        run_id = await self._create_run()
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
            run.status = RunStatus.pending
            await session.commit()

        with (
            patch.object(
                crawler,
                "discover_site_pages",
                AsyncMock(
                    side_effect=crawler.SitemapFrontierIncomplete(
                        "saved retryable frontier"
                    )
                ),
            ),
            patch("app.services.analyzer.analyze_run", AsyncMock()) as analyze,
            patch.object(crawler, "fail_run", AsyncMock()) as fail,
            patch.object(crawler, "update_progress", AsyncMock()),
        ):
            await crawler._run_crawl_impl(run_id)

        analyze.assert_not_awaited()
        fail.assert_not_awaited()
        async with SessionLocal() as session:
            run = (
                await session.execute(select(Run).where(Run.id == run_id))
            ).scalar_one()
        self.assertEqual(run.status, RunStatus.crawling)

    async def test_robots_advertised_sitemap_replaces_default_fallback(
        self,
    ) -> None:
        run_id = await self._create_run()
        custom = """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/</loc></url>
              <url><loc>https://example.com/pricing</loc></url>
            </urlset>
        """

        async def discovery_probe(**kwargs):
            url = kwargs["url"]
            if url == "https://example.com/":
                return _probe(url, body="<main>Home</main>"), {}, None
            if url == "https://example.com/robots.txt":
                return (
                    _probe(
                        url,
                        body="Sitemap: https://example.com/custom.xml\n",
                    ),
                    {},
                    None,
                )
            if url == "https://example.com/custom.xml":
                return _probe(url, body=custom), {}, None
            if url == "https://example.com/pricing":
                return _probe(url, body="<main>Pricing</main>"), {}, None
            self.fail(f"Unexpected URL: {url}")

        with (
            patch.object(
                crawler,
                "_probe_with_transport",
                AsyncMock(side_effect=discovery_probe),
            ) as probe_call,
            patch.object(crawler, "update_progress", AsyncMock()),
            patch.object(crawler.asyncio, "sleep", AsyncMock()),
        ):
            selected = await crawler.discover_site_pages(
                run_id,
                "example.com",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(
            selected,
            [
                ("https://example.com/", "home"),
                ("https://example.com/pricing", "pricing"),
            ],
        )
        requested = [call.kwargs["url"] for call in probe_call.await_args_list]
        self.assertIn("https://example.com/custom.xml", requested)
        self.assertNotIn("https://example.com/sitemap.xml", requested)
        async with SessionLocal() as session:
            artifact = (
                await session.execute(
                    select(RunArtifact).where(
                        RunArtifact.run_id == run_id,
                        RunArtifact.artifact_key == crawler.SITE_PAGE_MANIFEST_KEY,
                    )
                )
            ).scalar_one()
        robots_manifest = artifact.output_json["robots_sitemap_discovery"]
        self.assertEqual(robots_manifest["advertised_sitemap_count"], 1)
        self.assertFalse(robots_manifest["used_default_sitemap"])
        self.assertTrue(artifact.output_json["sitemap_discovery"]["complete"])

    async def test_completed_keys_retry_latest_transient_and_5xx_results(
        self,
    ) -> None:
        run_id = await self._create_run()
        target_url = "https://example.com/"

        def probe(
            label: str,
            *,
            status: int | None = None,
            error_class: str | None = None,
        ) -> DomainProbe:
            return DomainProbe(
                run_id=run_id,
                domain="example.com",
                user_agent_label=label,
                user_agent_string=label,
                target_url=target_url,
                probe_type=ProbeType.main_page,
                http_status=status,
                error_class=error_class,
                content_signals={
                    "_body_read_policy": {
                        "version": crawler.BODY_READ_POLICY_VERSION,
                        "response_size_limit_bytes": None,
                        "result": "complete_eof",
                        "terminal": True,
                    }
                },
                challenge_detected=False,
                body_looks_empty=True,
            )

        async with SessionLocal() as session:
            session.add_all(
                [
                    probe("GPTBot", status=200),
                    probe("GPTBot", status=503),
                    probe("ClaudeBot", error_class="connect_timeout"),
                    probe("ClaudeBot", status=200),
                    probe("PerplexityBot", status=404),
                    probe("Googlebot-desktop", status=429),
                    probe("DeepSeekBot", error_class="redirect_loop"),
                ]
            )
            await session.commit()

        completed = await crawler._completed_probe_keys(run_id)

        self.assertEqual(
            completed,
            {
                (target_url, "ClaudeBot", ProbeType.main_page.value),
                (target_url, "PerplexityBot", ProbeType.main_page.value),
                (target_url, "DeepSeekBot", ProbeType.main_page.value),
            },
        )


if __name__ == "__main__":
    unittest.main()
