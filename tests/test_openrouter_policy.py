import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from app.services.openrouter import (
    _NON_CITING_PROVIDERS,
    OpenRouterError,
    OpenRouterOutputLimitError,
    OpenRouterPolicyError,
    OpenRouterResponseContractError,
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
        # Копия: chat() переиспользует один payload между попытками, и без
        # снимка запись первой попытки меняется задним числом.
        self.requests.append(
            {"headers": dict(headers), "json": deepcopy(json)}
        )
        return _FakeResponse(self.body)


class _SequenceClient(_FakeClient):
    """Отдаёт заранее заданные ответы по одному на попытку."""

    def __init__(self, bodies: list[dict[str, Any]]):
        super().__init__(bodies[0])
        self._bodies = bodies

    async def post(
        self,
        _url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _FakeResponse:
        self.requests.append(
            {"headers": dict(headers), "json": deepcopy(json)}
        )
        index = min(len(self.requests) - 1, len(self._bodies) - 1)
        return _FakeResponse(self._bodies[index])


def _body(
    *,
    model: str = "openai/gpt-5.4",
    web_search_requests: int = 0,
    with_citation: bool = False,
    provider: str = "Test Provider",
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
        "provider": provider,
        "choices": [
            {
                "finish_reason": "stop",
                "native_finish_reason": "stop",
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
    def setUp(self) -> None:
        # Память о провайдерах живёт в модуле и иначе течёт между тестами.
        _NON_CITING_PROVIDERS.clear()
        self.addCleanup(_NON_CITING_PROVIDERS.clear)

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
        self.assertEqual(result.transport["status"], "succeeded")
        self.assertTrue(result.transport["output_complete"])
        self.assertEqual(
            result.usage["_aiv_transport"],
            result.transport,
        )

    async def test_required_policy_offers_the_only_server_tool(self) -> None:
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
        self.assertEqual(payload["tool_choice"], "auto")
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

    async def test_empty_pause_turn_response_names_the_stop_reason(
        self,
    ) -> None:
        # Пустой content при native_finish_reason="pause_turn" три недели
        # выглядел как безымянная «Model returned an empty response».
        body = _body(web_search_requests=1, with_citation=True)
        body["choices"] = [
            {
                "finish_reason": "stop",
                "native_finish_reason": "pause_turn",
                "message": {"role": "assistant", "content": ""},
            }
        ]
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
            self.assertRaises(OpenRouterError) as raised,
        ):
            await chat(
                model="anthropic/claude-opus-5",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        message = str(raised.exception)
        self.assertIn("pause_turn", message)
        self.assertIn("finish_reason=stop", message)

    async def test_output_limit_is_not_accepted_or_retried(self) -> None:
        body = _body()
        body["id"] = "generation-123"
        body["choices"][0]["finish_reason"] = "length"
        body["choices"][0]["native_finish_reason"] = "MAX_TOKENS"
        body["choices"][0]["message"]["content"] = '{"verdict":"pass"}'
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
            self.assertRaises(OpenRouterOutputLimitError) as raised,
        ):
            await chat(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Проверь расчёт."}],
                response_schema={"type": "object"},
                max_tokens=20_000,
                retry_response_contract_errors=False,
            )

        self.assertEqual(len(client.requests), 1)
        transport = raised.exception.result.transport
        self.assertEqual(transport["status"], "succeeded")
        self.assertFalse(transport["output_complete"])
        self.assertTrue(transport["output_limited"])
        self.assertEqual(transport["finish_reason"], "length")
        self.assertEqual(transport["native_finish_reason"], "MAX_TOKENS")
        self.assertEqual(transport["response_id"], "generation-123")
        self.assertIn("max_tokens=20000", str(raised.exception))

    async def test_nonfinal_finish_reasons_fail_closed_with_evidence(
        self,
    ) -> None:
        for finish_reason in ("content_filter", "error", ""):
            with self.subTest(finish_reason=finish_reason):
                body = _body()
                body["choices"][0]["finish_reason"] = finish_reason
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
                    self.assertRaises(
                        OpenRouterResponseContractError
                    ) as raised,
                ):
                    await chat(
                        model="google/gemini-3.6-flash",
                        messages=[
                            {"role": "user", "content": "Верни JSON."}
                        ],
                        response_schema={"type": "object"},
                        retry_response_contract_errors=False,
                    )

                self.assertEqual(len(client.requests), 1)
                transport = raised.exception.result.transport
                self.assertFalse(transport["output_complete"])
                self.assertFalse(transport["output_limited"])
                self.assertTrue(transport["output_incomplete_reason"])

    async def test_invalid_structured_output_preserves_transport_evidence(
        self,
    ) -> None:
        body = _body()
        body["choices"][0]["message"]["content"] = "{unfinished"
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
            self.assertRaises(OpenRouterResponseContractError) as raised,
        ):
            await chat(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни JSON."}],
                response_schema={"type": "object"},
                retry_response_contract_errors=False,
            )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(raised.exception.result.transport["status"], "succeeded")
        self.assertTrue(
            raised.exception.result.transport["output_complete"]
        )
        self.assertIn("incomplete JSON", str(raised.exception))

    async def test_complete_structured_response_returns_parsed_result(
        self,
    ) -> None:
        body = _body()
        body["choices"][0]["message"]["content"] = '{"value":"ok"}'
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
        ):
            result = await chat(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни JSON."}],
                response_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                retry_response_contract_errors=False,
            )

        self.assertEqual(result.parsed, {"value": "ok"})
        self.assertTrue(result.transport["output_complete"])
        self.assertEqual(len(client.requests), 1)

    async def test_policy_retry_tells_the_model_why_it_was_rejected(
        self,
    ) -> None:
        client = _SequenceClient(
            [
                _body(web_search_requests=0, with_citation=False),
                _body(web_search_requests=1, with_citation=True),
            ]
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
        ):
            result = await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        self.assertEqual(len(client.requests), 2)
        first = client.requests[0]["json"]["messages"]
        second = client.requests[1]["json"]["messages"]
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        self.assertIn("веб-поиск", second[-1]["content"])
        self.assertIn(
            "web_search_requests_not_confirmed",
            second[-1]["content"],
        )
        self.assertTrue(result.web_attestation["metric_eligible"])

    async def test_policy_retry_reminder_does_not_accumulate(self) -> None:
        client = _SequenceClient(
            [_body(web_search_requests=0, with_citation=False)]
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
            self.assertRaises(OpenRouterPolicyError),
        ):
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        self.assertEqual(len(client.requests), 3)
        self.assertEqual(len(client.requests[2]["json"]["messages"]), 2)

    async def test_endpoint_without_citations_is_routed_around(self) -> None:
        # Замер 2026-08-21: Anthropic и Claude Platform on AWS выполняют поиск,
        # но не отдают url_citation; Amazon Bedrock и Azure отдают.
        client = _SequenceClient(
            [
                _body(
                    web_search_requests=3,
                    with_citation=False,
                    provider="Claude Platform on AWS",
                ),
                _body(
                    web_search_requests=3,
                    with_citation=True,
                    provider="Amazon Bedrock",
                ),
            ]
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
        ):
            result = await chat(
                model="anthropic/claude-opus-5",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        self.assertEqual(len(client.requests), 2)
        self.assertNotIn("provider", client.requests[0]["json"])
        self.assertEqual(
            client.requests[1]["json"]["provider"],
            {"ignore": ["Claude Platform on AWS"]},
        )
        # Модель ни при чём — корректирующего сообщения быть не должно.
        self.assertEqual(len(client.requests[1]["json"]["messages"]), 1)
        self.assertTrue(result.web_attestation["metric_eligible"])
        self.assertEqual(
            _NON_CITING_PROVIDERS["anthropic/claude-opus-5"],
            {"Claude Platform on AWS"},
        )

    async def test_known_non_citing_endpoint_is_avoided_from_the_first_try(
        self,
    ) -> None:
        _NON_CITING_PROVIDERS["anthropic/claude-opus-5"] = {"Anthropic"}
        client = _FakeClient(
            _body(
                web_search_requests=3,
                with_citation=True,
                provider="Amazon Bedrock",
            )
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
            await chat(
                model="anthropic/claude-opus-5",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        self.assertEqual(
            client.requests[0]["json"]["provider"],
            {"ignore": ["Anthropic"]},
        )

    async def test_memory_only_calls_are_never_rerouted(self) -> None:
        _NON_CITING_PROVIDERS["openai/gpt-5.4"] = {"Anthropic"}
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
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "Что ты знаешь?"}],
                web_policy=WebSearchPolicy.FORBIDDEN,
            )

        self.assertNotIn("provider", client.requests[0]["json"])

    async def test_missing_search_is_corrected_by_prompt_not_by_routing(
        self,
    ) -> None:
        client = _SequenceClient(
            [
                _body(
                    web_search_requests=0,
                    with_citation=False,
                    provider="Amazon Bedrock",
                ),
                _body(
                    web_search_requests=2,
                    with_citation=True,
                    provider="Amazon Bedrock",
                ),
            ]
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
        ):
            await chat(
                model="anthropic/claude-opus-5",
                messages=[{"role": "user", "content": "Что нового?"}],
                web_policy=WebSearchPolicy.REQUIRED,
            )

        self.assertNotIn("provider", client.requests[1]["json"])
        self.assertEqual(len(client.requests[1]["json"]["messages"]), 2)
        self.assertEqual(_NON_CITING_PROVIDERS, {})
