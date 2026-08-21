import unittest
from typing import Any
from unittest.mock import patch

from app.services.openrouter import (
    OpenRouterPolicyError,
    WebSearchPolicy,
    chat,
)


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any]):
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeClient:
    def __init__(self, body: dict[str, Any]):
        self.body = body
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _FakeResponse:
        self.requests.append({"headers": headers, "json": json})
        return _FakeResponse(self.body)


def _body(
    *,
    model: str = "openai/gpt-5.4",
    web_search_requests: int = 0,
    with_citation: bool = False,
) -> dict[str, Any]:
    annotations = (
        [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://source.example",
                    "title": "Source",
                    "content": "Fact",
                },
            }
        ]
        if with_citation
        else []
    )
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Непустой ответ.",
                    "annotations": annotations,
                }
            }
        ],
        "usage": {
            "server_tool_use": {
                "web_search_requests": web_search_requests,
            }
        },
        "openrouter_metadata": {"pipeline": []},
    }


class OpenRouterPolicyRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_forbidden_policy_is_present_in_the_actual_request(self) -> None:
        client = _FakeClient(_body())
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
        ):
            result = await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что ты знаешь?"}],
                web_policy=WebSearchPolicy.FORBIDDEN,
            )

        payload = client.requests[0]["json"]
        self.assertEqual(
            payload["plugins"],
            [{"id": "web", "enabled": False}],
        )
        self.assertEqual(payload["tool_choice"], "none")
        self.assertNotIn("tools", payload)
        self.assertNotIn(":online", payload["model"])
        self.assertEqual(
            client.requests[0]["headers"]["X-OpenRouter-Metadata"],
            "enabled",
        )
        self.assertTrue(result.web_attestation["metric_eligible"])

    async def test_required_policy_forces_the_only_server_tool(self) -> None:
        client = _FakeClient(
            _body(web_search_requests=1, with_citation=True)
        )
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
        ):
            result = await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        payload = client.requests[0]["json"]
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(
            payload["tools"][0]["type"],
            "openrouter:web_search",
        )
        self.assertEqual(
            result.usage["_aiv_web_attestation"]["web_search_requests"],
            1,
        )
        self.assertEqual(len(result.usage["_aiv_response_annotations"]), 1)
        self.assertEqual(len(result.citations), 1)

    async def test_unattested_required_response_raises_with_saved_evidence(
        self,
    ) -> None:
        client = _FakeClient(
            _body(web_search_requests=0, with_citation=False)
        )
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
            patch(
                "app.services.openrouter.asyncio.sleep",
                return_value=None,
            ),
            self.assertRaises(OpenRouterPolicyError) as raised,
        ):
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        self.assertEqual(len(client.requests), 3)
        attestation = raised.exception.result.web_attestation
        self.assertFalse(attestation["metric_eligible"])
        self.assertEqual(attestation["web_search_requests"], 0)
        self.assertIn(
            "web_search_requests_not_confirmed",
            attestation["violations"],
        )

    async def test_account_default_web_plugin_override_fails_closed(
        self,
    ) -> None:
        body = _body()
        body["openrouter_metadata"] = {
            "pipeline": [
                {
                    "type": "plugin",
                    "name": "web-search",
                    "data": {"results": 2},
                }
            ]
        }
        client = _FakeClient(body)
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
            patch(
                "app.services.openrouter.asyncio.sleep",
                return_value=None,
            ),
            self.assertRaises(OpenRouterPolicyError) as raised,
        ):
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что ты знаешь?"}],
                web_policy=WebSearchPolicy.FORBIDDEN,
            )

        self.assertIn(
            "router_retrieval_stage_while_forbidden",
            raised.exception.result.web_attestation["violations"],
        )
