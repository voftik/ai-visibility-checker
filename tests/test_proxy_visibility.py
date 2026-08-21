import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from httpx import ASGITransport, AsyncClient

from app.main import app
from app.routes.runs import _public_event
from app.services import crawler
from app.services.proxy_pool import Proxy


class PublicEventPrivacyTests(unittest.TestCase):
    def test_operational_events_are_not_public(self) -> None:
        self.assertIsNone(
            _public_event(
                {
                    "type": "log",
                    "message": "GET via 203.0.113.1:8080",
                    "model": "provider/model",
                }
            )
        )
        self.assertIsNone(
            _public_event(
                {
                    "type": "probe_done",
                    "proxy_address": "203.0.113.1:8080",
                    "user_agent_string": "secret detail",
                }
            )
        )

    def test_stage_event_keeps_only_public_fields(self) -> None:
        event = _public_event(
            {
                "type": "stage",
                "stage": "technical_access",
                "label": "Проверяем доступ для ИИ",
                "detail": "Сравниваем ответы сайта.",
                "percent": 20,
                "eta_seconds": 900,
                "proxy_address": "203.0.113.1:8080",
                "model": "provider/model",
            }
        )
        self.assertEqual(
            event,
            {
                "type": "stage",
                "stage": "technical_access",
                "label": "Проверяем доступ для ИИ",
                "detail": "Сравниваем ответы сайта.",
                "percent": 20,
                "eta_seconds": 900,
            },
        )


class TransportFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_failure_falls_back_without_exposing_address(self) -> None:
        proxy = Proxy(
            id="proxy-1",
            address="198.51.100.20",
            port=8080,
            username="user",
            password="password",
            country="DE",
        )
        pool = Mock()
        pool.random_proxy.return_value = proxy
        first = crawler.ProbeResult(
            error_class="connect_timeout",
            error_message="timeout",
        )
        second = crawler.ProbeResult(
            http_status=200,
            final_url="https://example.com/",
            tls_ok=True,
            full_text="<main>Readable</main>",
        )
        with (
            patch.object(crawler, "get_pool", return_value=pool),
            patch.object(
                crawler,
                "_do_probe",
                new=AsyncMock(side_effect=[first, second]),
            ) as do_probe,
        ):
            result, transport, selected_proxy = await crawler._probe_with_transport(
                url="https://example.com/",
                user_agent="GPTBot",
                timeout_seconds=10,
                concurrency=2,
            )

        self.assertEqual(result.http_status, 200)
        self.assertIs(selected_proxy, proxy)
        self.assertTrue(transport["proxy_used"])
        self.assertTrue(transport["fallback_direct"])
        self.assertEqual(transport["country"], "DE")
        self.assertNotIn("address", transport)
        self.assertNotIn("username", transport)
        self.assertNotIn("password", transport)
        pool.mark_bad.assert_called_once_with(proxy.url)
        self.assertEqual(do_probe.await_count, 2)


class HealthEndpointPrivacyTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_endpoint_does_not_expose_infrastructure(self) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    async def test_interactive_api_documentation_is_not_public(self) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            responses = await asyncio.gather(
                client.get("/docs"),
                client.get("/redoc"),
                client.get("/openapi.json"),
            )
        self.assertTrue(all(response.status_code == 404 for response in responses))


if __name__ == "__main__":
    unittest.main()
