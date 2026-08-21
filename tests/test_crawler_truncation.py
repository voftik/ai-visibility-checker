import unittest
import uuid
from unittest.mock import AsyncMock, patch

from sqlalchemy import delete, select

from app.db import SessionLocal, init_db
from app.models import DomainProbe, ProbeType, Run, RunStatus, SitePage
from app.services import crawler


class _StreamingResponse:
    is_redirect = False
    status_code = 200
    url = "https://example.com/"
    headers = {"content-type": "text/html; charset=utf-8"}

    async def aiter_bytes(self):
        yield b"a" * crawler.BODY_LIMIT_BYTES
        yield b"b"

    async def aclose(self) -> None:
        return None


class _StreamingClient:
    def __init__(self, response: _StreamingResponse) -> None:
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def build_request(self, method: str, url: str, headers: dict):
        return method, url, headers

    async def send(self, request, *, stream: bool):
        del request, stream
        return self.response


class ProbeBodyTruncationTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_result_marks_an_oversized_stream_as_truncated(
        self,
    ) -> None:
        self.assertFalse(crawler.ProbeResult().body_truncated)
        response = _StreamingResponse()
        client = _StreamingClient(response)

        with (
            patch.object(crawler, "_validate_public_url", AsyncMock()),
            patch.object(crawler.httpx, "AsyncClient", return_value=client),
        ):
            result = await crawler._do_probe(
                url="https://example.com/",
                user_agent="TestBot",
                timeout_seconds=5,
                concurrency=1,
            )

        self.assertTrue(result.body_truncated)
        self.assertEqual(result.response_size_bytes, crawler.BODY_LIMIT_BYTES)
        self.assertEqual(len(result.body_bytes or b""), crawler.BODY_LIMIT_BYTES)


class PersistedTruncationContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.run_id = f"test-truncation-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=self.run_id,
                    domain="example.com",
                    status=RunStatus.crawling,
                    config_json={},
                )
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        async with SessionLocal() as session:
            await session.execute(delete(Run).where(Run.id == self.run_id))
            await session.commit()

    @staticmethod
    def _stored_probe(
        run_id: str,
        label: str,
        *,
        response_size_bytes: int,
        content_signals: dict | None = None,
    ) -> DomainProbe:
        return DomainProbe(
            run_id=run_id,
            domain="example.com",
            user_agent_label=label,
            user_agent_string=label,
            target_url="https://example.com/",
            probe_type=ProbeType.main_page,
            http_status=200,
            response_size_bytes=response_size_bytes,
            content_signals=content_signals,
            challenge_detected=False,
            body_looks_empty=False,
        )

    async def test_completed_keys_reject_legacy_and_explicit_truncation(
        self,
    ) -> None:
        async with SessionLocal() as session:
            session.add_all(
                [
                    self._stored_probe(
                        self.run_id,
                        "legacy-cap",
                        response_size_bytes=crawler.LEGACY_BODY_LIMIT_BYTES,
                    ),
                    self._stored_probe(
                        self.run_id,
                        "explicit-truncation",
                        response_size_bytes=1_234,
                        content_signals={"_body_truncated": True},
                    ),
                    self._stored_probe(
                        self.run_id,
                        "complete",
                        response_size_bytes=crawler.LEGACY_BODY_LIMIT_BYTES - 1,
                        content_signals={"_body_truncated": False},
                    ),
                ]
            )
            await session.commit()

        completed = await crawler._completed_probe_keys(self.run_id)

        self.assertEqual(
            completed,
            {
                (
                    "https://example.com/",
                    "complete",
                    ProbeType.main_page.value,
                )
            },
        )

    async def test_truncation_signal_is_persisted_and_rendering_stays_unknown(
        self,
    ) -> None:
        text = " ".join(
            ["Серверный текст страницы остаётся содержательным."] * 80
        )
        html = f"""
        <html>
          <head>
            <title>Example</title>
            <script type="application/ld+json">
              {{"@context":"https://schema.org","@type":"Organization"}}
            </script>
          </head>
          <body><main><article><h1>Example</h1><p>{text}</p></article></main></body>
        </html>
        """
        probe = crawler.ProbeResult(
            http_status=200,
            final_url="https://example.com/",
            full_text=html,
            content_type="text/html; charset=utf-8",
            response_size_bytes=crawler.BODY_LIMIT_BYTES,
            body_truncated=True,
            body_looks_empty=False,
        )
        job = crawler._Job(
            domain="example.com",
            user_agent_label="Chrome-control",
            user_agent_string="Chrome-control",
            target_url="https://example.com/",
            probe_type=ProbeType.main_page,
            page_kind="home",
        )

        await crawler._persist_probe(
            self.run_id,
            job,
            probe,
            {"proxy_used": False},
        )
        await crawler._store_site_page(
            self.run_id,
            url="https://example.com/",
            page_kind="home",
            probe=probe,
        )

        async with SessionLocal() as session:
            stored_probe = (
                await session.execute(
                    select(DomainProbe).where(
                        DomainProbe.run_id == self.run_id,
                        DomainProbe.user_agent_label == "Chrome-control",
                    )
                )
            ).scalar_one()
            page = (
                await session.execute(
                    select(SitePage).where(
                        SitePage.run_id == self.run_id,
                        SitePage.url == "https://example.com/",
                    )
                )
            ).scalar_one()

        probe_signals = stored_probe.content_signals or {}
        page_signals = page.content_signals or {}
        self.assertIs(probe_signals["_body_truncated"], True)
        self.assertIs(page_signals["_body_truncated"], True)
        self.assertGreater(page.text_length, 1_000)
        self.assertEqual(page_signals["render_strategy"], "unknown")
        self.assertEqual(page_signals["render_strategy_confidence"], "low")
        self.assertIs(page_signals["structured_data_complete"], False)
        self.assertEqual(
            page_signals["structured_data_types"],
            ["Organization"],
        )


if __name__ == "__main__":
    unittest.main()
