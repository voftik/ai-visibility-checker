import copy
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, select

from app.db import SessionLocal, init_db
from app.models import DomainProbe, ProbeType, Run, RunArtifact, RunStatus, SitePage
from app.services import crawler
from app.services.analyzer import _site_context


def _policy() -> dict:
    return crawler._body_read_policy(crawler.ProbeResult())


class CrawlAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_ids: list[str] = []

    async def asyncTearDown(self) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id.in_(self.run_ids)))
            await session.commit()

    async def _run(self, *, config: dict | None = None) -> str:
        run_id = f"test-admission-{uuid.uuid4()}"
        self.run_ids.append(run_id)
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.completed,
                    config_json=config or {},
                )
            )
            await session.commit()
        return run_id

    async def _page(
        self,
        run_id: str,
        url: str,
        page_kind: str,
        *,
        legacy: bool = False,
    ) -> SitePage:
        signals = {"_body_truncated": False}
        if not legacy:
            signals["_body_read_policy"] = _policy()
            signals["_source_body_sha256"] = f"source-{url}"
        page = SitePage(
            run_id=run_id,
            url=url,
            page_kind=page_kind,
            http_status=200,
            main_text=f"Full text for {url}",
            text_length=100,
            content_signals=signals,
        )
        async with SessionLocal() as session:
            session.add(page)
            await session.commit()
            await session.refresh(page)
        return page

    async def _probe(
        self,
        run_id: str,
        url: str,
        label: str,
        *,
        status: int = 200,
        legacy: bool = False,
    ) -> None:
        async with SessionLocal() as session:
            session.add(
                DomainProbe(
                    run_id=run_id,
                    domain="example.com",
                    user_agent_label=label,
                    user_agent_string=label,
                    target_url=url,
                    page_kind=crawler._page_kind(url),
                    probe_type=ProbeType.main_page,
                    http_status=status,
                    content_signals={} if legacy else {"_body_read_policy": _policy()},
                    challenge_detected=False,
                    body_looks_empty=False,
                )
            )
            await session.commit()

    async def _manifest(
        self,
        run_id: str,
        pages: list[tuple[str, str]],
    ) -> None:
        async with SessionLocal() as session:
            stored = {
                page.url: page
                for page in (
                    await session.execute(
                        select(SitePage).where(SitePage.run_id == run_id)
                    )
                ).scalars()
            }
        receipt = crawler._site_page_receipt(pages, stored)
        input_json = crawler._site_page_manifest_input("example.com")
        candidates = [
            crawler._candidate_evidence_record(url, source="test_fixture")
            for url, _page_kind in pages[1:]
        ]
        relevance_receipt = crawler._selection_relevance_receipt(
            homepage_url=pages[0][0],
            candidates=candidates,
            target=crawler.AUDIT_PAGE_DEFAULT,
            proposed=pages,
            attempts=[
                crawler._candidate_attempt(page, outcome="usable")
                for page in pages
            ],
            selected=pages,
        )
        await crawler._save_page_relevance_artifact(
            run_id,
            domain="example.com",
            receipt=relevance_receipt,
        )
        await crawler._save_site_page_manifest(
            run_id,
            status="completed",
            input_json=input_json,
            output_json={
                "pages": crawler._selected_page_records(pages),
                "expected_page_count": len(pages),
                "selected_count": len(pages),
                "selected_pages_sha256": crawler._selected_pages_sha256(pages),
                "discovered_candidate_count": len(pages),
                "discovered_count": len(pages),
                "page_scope": crawler.PAGE_SCOPE,
                "selection_policy": input_json["selection_policy"],
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

    def test_selection_is_commercially_diverse_and_hard_bounded(self) -> None:
        candidates = [
            "https://example.com/privacy",
            "https://example.com/success",
            *[
                f"https://example.com/services/offer-{index}"
                for index in range(20)
            ],
            "https://example.com/about",
            "https://example.com/cases/client",
            "https://example.com/contact",
        ]

        selected = crawler._semantic_frontier_urls(
            "https://example.com/",
            candidates,
            limit=99,
        )

        self.assertEqual(len(selected), crawler.AUDIT_PAGE_HARD_MAX)
        self.assertEqual(selected[0], ("https://example.com/", "home"))
        self.assertNotIn("https://example.com/privacy", {url for url, _ in selected})
        self.assertNotIn("https://example.com/success", {url for url, _ in selected})
        self.assertGreaterEqual(
            sum(1 for _url, kind in selected if kind == "product"),
            6,
        )

    async def test_missing_matrix_cell_blocks_then_exact_receipt_admits(self) -> None:
        run_id = await self._run()
        pages = [
            ("https://example.com/", "home"),
            ("https://example.com/services", "product"),
        ]
        for url, kind in pages:
            await self._page(run_id, url, kind)
        await self._manifest(run_id, pages)
        for url, label in [
            (pages[0][0], "GPTBot"),
            (pages[0][0], "ClaudeBot"),
            (pages[1][0], "GPTBot"),
        ]:
            await self._probe(run_id, url, label)

        incomplete = await crawler.save_technical_matrix_receipt(
            run_id,
            domain="example.com",
            pages=pages,
            user_agents=["GPTBot", "ClaudeBot"],
        )
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["expected_cell_count"], 4)
        self.assertEqual(incomplete["missing_cell_count"], 1)
        with self.assertRaises(crawler.CrawlAdmissionIncomplete):
            await crawler.require_crawl_admission(
                run_id,
                domain="example.com",
                user_agents=["GPTBot", "ClaudeBot"],
            )

        await self._probe(run_id, pages[1][0], "ClaudeBot")
        complete = await crawler.save_technical_matrix_receipt(
            run_id,
            domain="example.com",
            pages=pages,
            user_agents=["GPTBot", "ClaudeBot"],
        )
        admission = await crawler.require_crawl_admission(
            run_id,
            domain="example.com",
            user_agents=["GPTBot", "ClaudeBot"],
        )
        self.assertTrue(complete["complete"])
        self.assertEqual(admission["expected_cell_count"], 4)
        self.assertEqual(admission["terminal_cell_count"], 4)
        self.assertEqual(admission["page_count"], 2)
        self.assertRegex(
            admission["commercial_relevance_receipt_sha256"],
            r"^[0-9a-f]{64}$",
        )

    async def test_three_transient_attempts_become_explicit_terminal_cell(self) -> None:
        run_id = await self._run()
        pages = [("https://example.com/", "home")]
        await self._page(run_id, pages[0][0], "home")
        for _ in range(crawler.TECHNICAL_MAX_PROBE_ATTEMPTS):
            await self._probe(run_id, pages[0][0], "GPTBot", status=503)

        receipt = await crawler.save_technical_matrix_receipt(
            run_id,
            domain="example.com",
            pages=pages,
            user_agents=["GPTBot"],
        )

        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["terminal_blocked_cell_count"], 1)
        self.assertEqual(receipt["cells"][0]["attempt_count"], 3)
        self.assertEqual(receipt["cells"][0]["terminal_reason"], "retry_exhausted")

    async def test_every_consumed_site_page_field_is_bound_before_analysis(
        self,
    ) -> None:
        run_id = await self._run()
        pages = [("https://example.com/", "home")]
        page = await self._page(run_id, pages[0][0], "home")
        async with SessionLocal() as session:
            stored = await session.get(SitePage, page.id)
            stored.title = "Example home"
            stored.meta_description = "Original description"
            await session.commit()
        await self._manifest(run_id, pages)
        await self._probe(run_id, pages[0][0], "GPTBot")
        await crawler.save_technical_matrix_receipt(
            run_id,
            domain="example.com",
            pages=pages,
            user_agents=["GPTBot"],
        )
        await crawler.require_crawl_admission(
            run_id,
            domain="example.com",
            user_agents=["GPTBot"],
        )

        async with SessionLocal() as session:
            original = await session.get(SitePage, page.id)
            original_values = {
                "title": original.title,
                "meta_description": original.meta_description,
                "content_signals": copy.deepcopy(original.content_signals),
            }

        mutations = {
            "title": "Different identity",
            "meta_description": "Different offer description",
            "content_signals": {
                **original_values["content_signals"],
                "structured_data_types": ["Product"],
            },
        }
        for field, changed_value in mutations.items():
            with self.subTest(field=field):
                async with SessionLocal() as session:
                    stored = await session.get(SitePage, page.id)
                    setattr(stored, field, copy.deepcopy(changed_value))
                    await session.commit()
                with self.assertRaisesRegex(
                    crawler.CrawlAdmissionIncomplete,
                    "SitePage rows no longer match manifest",
                ):
                    await crawler.require_crawl_admission(
                        run_id,
                        domain="example.com",
                        user_agents=["GPTBot"],
                    )
                with self.assertRaisesRegex(
                    crawler.CrawlAdmissionIncomplete,
                    "SitePage lineage no longer matches manifest",
                ):
                    await _site_context(run_id)
                async with SessionLocal() as session:
                    stored = await session.get(SitePage, page.id)
                    setattr(stored, field, copy.deepcopy(original_values[field]))
                    await session.commit()
                await crawler.require_crawl_admission(
                    run_id,
                    domain="example.com",
                    user_agents=["GPTBot"],
                )

    async def test_terminal_partial_discovery_admits_bounded_technical_corpus(
        self,
    ) -> None:
        run_id = await self._run()
        pages = [
            ("https://example.com/", "home"),
            ("https://example.com/services", "product"),
        ]
        for url, kind in pages:
            await self._page(run_id, url, kind)
        await self._manifest(run_id, pages)
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
            output = dict(manifest.output_json)
            output["discovery_state"] = "terminal_partial"
            output["coverage_state"] = "bounded"
            manifest.output_json = output
            await session.commit()
        for url, _kind in pages:
            await self._probe(run_id, url, "GPTBot")
        await crawler.save_technical_matrix_receipt(
            run_id,
            domain="example.com",
            pages=pages,
            user_agents=["GPTBot"],
        )

        admission = await crawler.require_crawl_admission(
            run_id,
            domain="example.com",
            user_agents=["GPTBot"],
        )

        self.assertEqual(admission["coverage_state"], "bounded")
        self.assertEqual(admission["page_count"], 2)

    async def test_legacy_bootstrap_is_network_free_and_visibly_limited(self) -> None:
        run_id = await self._run(config={"user_agents": ["GPTBot", "ClaudeBot"]})
        url = "https://example.com/"
        await self._page(run_id, url, "home", legacy=True)
        await self._probe(run_id, url, "GPTBot", legacy=True)
        await self._probe(run_id, url, "ClaudeBot", legacy=True)

        with patch.object(crawler, "_probe_with_transport", AsyncMock()) as network:
            admission = await crawler.bootstrap_legacy_crawl_admission(
                run_id,
                domain="example.com",
            )

        network.assert_not_awaited()
        self.assertTrue(admission["legacy_snapshot"])
        self.assertEqual(admission["coverage_state"], "limited")
        self.assertEqual(admission["page_count"], 1)

    async def test_legacy_bootstrap_preserves_historical_page_kind_lineage(
        self,
    ) -> None:
        run_id = await self._run(config={"user_agents": ["GPTBot"]})
        home_url = "https://example.com/"
        service_url = "https://example.com/services"
        privacy_url = "https://example.com/privacy"
        historical_service_kind = "other"
        self.assertNotEqual(
            crawler._page_kind(service_url),
            historical_service_kind,
        )
        self.assertEqual(crawler._page_kind(privacy_url), "utility")
        await self._page(run_id, home_url, "home", legacy=True)
        await self._page(
            run_id,
            service_url,
            historical_service_kind,
            legacy=True,
        )
        await self._page(
            run_id,
            privacy_url,
            "content",
            legacy=True,
        )
        await self._probe(run_id, home_url, "GPTBot", legacy=True)
        await self._probe(run_id, service_url, "GPTBot", legacy=True)
        await self._probe(run_id, privacy_url, "GPTBot", legacy=True)

        with patch.object(crawler, "_probe_with_transport", AsyncMock()) as network:
            admission = await crawler.bootstrap_legacy_crawl_admission(
                run_id,
                domain="example.com",
            )
        context = await _site_context(run_id)

        network.assert_not_awaited()
        self.assertTrue(admission["legacy_snapshot"])
        self.assertLessEqual(admission["page_count"], crawler.AUDIT_PAGE_HARD_MAX)
        self.assertEqual(
            [(page["url"], page["page_kind"]) for page in context["pages"]],
            [
                (home_url, "home"),
                (service_url, historical_service_kind),
                (privacy_url, "content"),
            ],
        )
        self.assertEqual(
            [
                (page["url"], page["page_kind"])
                for page in context["selected_pages_manifest"]["pages"]
            ],
            [
                (home_url, "home"),
                (service_url, historical_service_kind),
                (privacy_url, "content"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
