import unittest
import uuid
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
            for url, page_kind in page_urls or []:
                session.add(
                    SitePage(
                        run_id=run_id,
                        url=url,
                        page_kind=page_kind,
                        text_length=10,
                    )
                )
            await session.commit()
        return run_id

    def _completed_artifact(self, pages: list[tuple[str, str]]) -> RunArtifact:
        return RunArtifact(
            run_id="replaced-by-helper",
            stage_key="site_discovery",
            artifact_key=crawler.SITE_PAGE_MANIFEST_KEY,
            status="completed",
            prompt_version=crawler.SITE_PAGE_MANIFEST_VERSION,
            input_json={
                "domain": "example.com",
                "selection_limit": len(pages),
            },
            output_json={
                "pages": [
                    {"url": url, "page_kind": page_kind}
                    for url, page_kind in pages
                ],
                "discovered_count": len(pages),
                "selected_count": len(pages),
                "selection_limit": len(pages),
                "coverage_state": "complete",
            },
        )

    def test_manifest_rejects_pages_from_another_domain(self) -> None:
        manifest = {
            "pages": [
                {"url": "https://other.example/", "page_kind": "home"},
            ],
            "discovered_count": 1,
            "selected_count": 1,
            "selection_limit": 1,
            "coverage_state": "complete",
        }
        self.assertIsNone(
            crawler._manifest_pages(
                manifest,
                domain="example.com",
                limit=1,
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
            artifact=self._completed_artifact(pages),
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
                limit=3,
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
                limit=3,
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
        exact_input = {"domain": "example.com", "selection_limit": 2}
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
                    "pages": [
                        {"url": "https://example.com/", "page_kind": "home"}
                    ],
                    "discovered_count": 1,
                    "selected_count": 1,
                    "selection_limit": 2,
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
                        limit=2,
                        timeout_seconds=5,
                        concurrency=1,
                    )

                self.assertEqual(
                    [call.kwargs["url"] for call in probe_call.await_args_list],
                    [
                        "https://example.com/",
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
                self.assertEqual(persisted_after_completion, ["completed", "completed"])

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
                        manifest.output_json,
                        {
                            "pages": [
                                {
                                    "url": "https://example.com/",
                                    "page_kind": "home",
                                },
                                {
                                    "url": "https://example.com/services",
                                    "page_kind": "product",
                                },
                            ],
                            "discovered_count": None,
                            "selected_count": 2,
                            "selection_limit": 2,
                            "coverage_state": "unknown",
                        },
                    )

    async def test_bounded_urlset_records_limited_discovery_coverage(self) -> None:
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
            if url == "https://example.com/services":
                return _probe(url, body="<main>Services</main>"), {}, None
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
                limit=2,
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertEqual(
            selected,
            [
                ("https://example.com/", "home"),
                ("https://example.com/services", "product"),
            ],
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
        self.assertEqual(manifest.output_json["discovered_count"], 4)
        self.assertEqual(manifest.output_json["selected_count"], 2)
        self.assertEqual(manifest.output_json["selection_limit"], 2)
        self.assertEqual(manifest.output_json["coverage_state"], "limited")

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
