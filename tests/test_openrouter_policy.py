import asyncio
import hashlib
import inspect
import json as json_lib
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import httpx

from app.services.openrouter import (
    _NON_CITING_PROVIDERS,
    PHYSICAL_POST_AUDIT_DELTA_VERSION,
    PHYSICAL_POST_AUDIT_VERSION,
    STRUCTURED_AUDIT_DELTA_CAPABILITY,
    STRUCTURED_AUDIT_EVENT_VERSION_ATTR,
    STRUCTURED_CHECKPOINT_DELTA_VERSION,
    STRUCTURED_CHECKPOINT_VERSION,
    OpenRouterAuditCheckpointError,
    OpenRouterError,
    OpenRouterOutputLimitError,
    OpenRouterPolicyError,
    OpenRouterResponseContractError,
    OpenRouterStructuredContinuationCancelled,
    OpenRouterStructuredContinuationError,
    OutputTokenPolicy,
    WebSearchPolicy,
    chat,
    chat_continuable_structured,
    promote_provider_post_to_structured_checkpoint,
    restore_completed_chat_provider_event,
    restore_completed_structured_checkpoint,
    structured_resume_contract,
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
        content: bytes,
    ) -> _FakeResponse:
        # Копия: chat() переиспользует один payload между попытками, и без
        # снимка запись первой попытки меняется задним числом.
        payload = json_lib.loads(content.decode("utf-8"))
        self.requests.append(
            {"headers": dict(headers), "json": payload, "content": content}
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
        content: bytes,
    ) -> _FakeResponse:
        payload = json_lib.loads(content.decode("utf-8"))
        self.requests.append(
            {"headers": dict(headers), "json": payload, "content": content}
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
        search_parameters = payload["tools"][0]["parameters"]
        self.assertEqual(search_parameters, {"engine": "auto"})
        for local_ceiling in (
            "max_results",
            "max_total_results",
            "max_uses",
            "search_context_size",
        ):
            self.assertNotIn(local_ceiling, search_parameters)
        self.assertNotIn("max_tool_calls", payload)
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

        self.assertEqual(len(client.requests), 1)
        attestation = raised.exception.result.web_attestation
        self.assertFalse(attestation["metric_eligible"])
        self.assertEqual(attestation["web_search_requests"], 0)
        self.assertIn(
            "web_search_requests_not_confirmed",
            attestation["violations"],
        )
        attempts = raised.exception.result.usage["_aiv_call_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "rejected")

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
                retry_response_contract_errors=True,
                retry_transport_errors=True,
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
                max_completion_tokens=20_000,
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
        self.assertIn("max_completion_tokens=20000", str(raised.exception))

    async def test_transport_retries_can_be_disabled_for_expensive_calls(
        self,
    ) -> None:
        client = _FakeClient({"error": {"message": "rate limited"}})
        with (
            patch.object(_FakeResponse, "status_code", 429),
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
            ) as sleep_mock,
            self.assertRaises(OpenRouterError),
        ):
            await chat(
                model="anthropic/claude-fable-5",
                messages=[{"role": "user", "content": "Верни план."}],
                retry_response_contract_errors=False,
                retry_transport_errors=False,
            )

        self.assertEqual(len(client.requests), 1)
        sleep_mock.assert_not_awaited()

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
        self.assertIn(
            "not exactly one complete JSON document",
            str(raised.exception),
        )

    async def test_structured_output_rejects_json_substring_salvage(
        self,
    ) -> None:
        for content in (
            'Explanation before JSON. {"value":"ok"}',
            '{"value":"ok"}\nImportant caveat after JSON.',
            '{"value":"ok"}{"ignored":"second document"}',
            '```json\n{"value":"ok"}\n```',
            '{"value":NaN}',
        ):
            with self.subTest(content=content):
                body = _body()
                body["choices"][0]["message"]["content"] = content
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
                        response_schema={
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"}
                            },
                            "required": ["value"],
                        },
                        retry_response_contract_errors=False,
                    )

                self.assertEqual(len(client.requests), 1)
                self.assertEqual(raised.exception.result.text, content)
                self.assertIn(
                    "not exactly one complete JSON document",
                    str(raised.exception),
                )

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
                retry_response_contract_errors=True,
                retry_transport_errors=True,
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
        self.assertEqual(
            [item["status"] for item in result.usage["_aiv_call_attempts"]],
            ["rejected", "accepted"],
        )

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
                retry_response_contract_errors=True,
                retry_transport_errors=True,
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
                retry_response_contract_errors=True,
                retry_transport_errors=True,
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
                retry_response_contract_errors=True,
                retry_transport_errors=True,
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
                retry_response_contract_errors=True,
                retry_transport_errors=True,
            )

        self.assertNotIn("provider", client.requests[1]["json"])
        self.assertEqual(len(client.requests[1]["json"]["messages"]), 2)
        self.assertEqual(_NON_CITING_PROVIDERS, {})

    async def test_model_max_is_bounded_by_exact_request_headroom(self) -> None:
        client = _FakeClient(_body())
        envelope = {
            "version": "test-envelope",
            "policy": "model_max_available",
            "requested_model": "openai/gpt-5.4",
            "resolution": "test",
            "context_length": 2_000,
            "max_completion_tokens": 1_500,
        }
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
                "app.services.openrouter.model_output_envelope",
                return_value=envelope,
            ),
        ):
            result = await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "x" * 700}],
                output_token_policy=OutputTokenPolicy.MODEL_MAX,
            )

        requested = client.requests[0]["json"]["max_completion_tokens"]
        estimate = result.usage["_aiv_output_envelope"]["request_estimate"]
        self.assertEqual(
            requested,
            2_000 - estimate["input_token_upper_bound"],
        )
        self.assertLess(requested, 1_500)
        self.assertEqual(
            result.usage["_aiv_output_envelope"][
                "effective_max_completion_tokens"
            ],
            requested,
        )

    async def test_empty_choices_post_is_checkpointed_before_failure(
        self,
    ) -> None:
        body = {
            "id": "paid-empty-choice",
            "choices": [],
            "usage": {"total_tokens": 30, "cost": 0.5},
            "provider": "Test Provider",
        }
        client = _FakeClient(body)
        events: list[dict[str, Any]] = []
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch(
                "app.services.openrouter._headers",
                return_value={"Authorization": "Bearer test"},
            ),
            self.assertRaises(OpenRouterError) as raised,
        ):
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "x"}],
                audit_checkpoint=events.append,
            )

        self.assertFalse(hasattr(raised.exception, "result"))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["status"], "response_error")
        self.assertEqual(event["response"]["body_json"], body)
        self.assertEqual(event["error"]["type"], "OpenRouterError")
        self.assertTrue(event["request_payload"])

    async def test_http_failure_is_checkpointed_before_transport_retry(
        self,
    ) -> None:
        class StatusResponse(_FakeResponse):
            def __init__(self, body: dict[str, Any], status: int):
                super().__init__(body)
                self.status_code = status

        class StatusSequenceClient(_FakeClient):
            def __init__(self) -> None:
                super().__init__(_body())
                self.index = 0

            async def post(self, _url: str, *, headers: dict[str, str], content: bytes) -> _FakeResponse:
                payload = json_lib.loads(content.decode("utf-8"))
                self.requests.append({"headers": dict(headers), "json": payload})
                self.index += 1
                if self.index == 1:
                    return StatusResponse(
                        {"error": {"message": "temporary"}, "usage": {"cost": 0.1}},
                        500,
                    )
                return StatusResponse(_body(), 200)

        client = StatusSequenceClient()
        events: list[dict[str, Any]] = []
        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=client),
            patch("app.services.openrouter._headers", return_value={}),
            patch("app.services.openrouter.asyncio.sleep", return_value=None),
        ):
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "x"}],
                retry_transport_errors=True,
                audit_checkpoint=events.append,
            )

        self.assertEqual([event["status"] for event in events], ["http_error", "accepted"])
        self.assertEqual(events[0]["response"]["http_status"], 500)
        self.assertEqual(len(client.requests), 2)

    async def test_exhausted_transport_retry_preserves_final_post_event(
        self,
    ) -> None:
        class StatusResponse(_FakeResponse):
            status_code = 500

        class Client(_FakeClient):
            async def post(
                self,
                _url: str,
                *,
                headers: dict[str, str],
                content: bytes,
            ) -> _FakeResponse:
                payload = json_lib.loads(content.decode("utf-8"))
                self.requests.append(
                    {"headers": dict(headers), "json": payload}
                )
                return StatusResponse(
                    {
                        "error": {"message": "still down"},
                        "usage": {"cost": 0.25},
                    }
                )

        client = Client(_body())
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.asyncio.sleep",
                return_value=None,
            ),
            self.assertRaises(OpenRouterError) as raised,
        ):
            await chat(
                model="openai/gpt-5.4",
                messages=[{"role": "user", "content": "x"}],
                retry_transport_errors=True,
            )

        self.assertEqual(len(client.requests), 3)
        event = raised.exception.audit_event
        self.assertEqual(event["attempt"], 3)
        self.assertEqual(event["status"], "http_error")
        self.assertEqual(event["response"]["body_json"]["usage"]["cost"], 0.25)


class OpenRouterStructuredContinuationTests(unittest.IsolatedAsyncioTestCase):
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    envelope = {
        "version": "test-envelope",
        "policy": "model_max_available",
        "requested_model": "google/gemini-3.6-flash",
        "resolution": "test",
        "context_length": 1_000_000,
        "max_completion_tokens": 65_536,
    }

    def _part(self, text: str, *, limited: bool) -> dict[str, Any]:
        body = _body(model="google/gemini-3.6-flash")
        body["choices"][0]["message"]["content"] = text
        if limited:
            body["choices"][0]["finish_reason"] = "length"
            body["choices"][0]["native_finish_reason"] = "MAX_TOKENS"
        return body

    async def _run(
        self,
        bodies: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[Any, _SequenceClient]:
        client = _SequenceClient(bodies)
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
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
        ):
            result = await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                retry_transport_errors=False,
                **kwargs,
            )
        return result, client

    async def test_successful_multi_part_document_is_joined_and_validated(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"},'
        after_first = initial + '{"name":"Б"},'
        first = initial[-12:] + '{"name":"Б"},'
        second = after_first[-12:] + '{"name":"В"}]}'

        result, client = await self._run(
            [
                self._part(initial, limited=True),
                self._part(first, limited=True),
                self._part(second, limited=False),
            ]
        )

        self.assertEqual(
            result.parsed,
            {"items": [{"name": "А"}, {"name": "Б"}, {"name": "В"}]},
        )
        self.assertEqual(len(client.requests), 3)
        self.assertIn("response_format", client.requests[0]["json"])
        self.assertNotIn("response_format", client.requests[1]["json"])
        self.assertNotIn("response_format", client.requests[2]["json"])
        self.assertEqual(
            client.requests[1]["json"]["messages"][-2],
            {"role": "assistant", "content": initial},
        )
        self.assertEqual(
            client.requests[2]["json"]["messages"][-2],
            {"role": "assistant", "content": after_first},
        )
        self.assertTrue(
            all(
                request["json"]["max_completion_tokens"] == 65_536
                for request in client.requests
            )
        )
        manifest = result.usage["_aiv_structured_continuation"]
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["continuation_count"], 2)
        self.assertEqual([part["sequence"] for part in manifest["parts"]], [0, 1, 2])
        self.assertEqual([call["sequence"] for call in manifest["calls"]], [0, 1, 2])
        self.assertTrue(result.transport["output_complete"])
        self.assertFalse(result.transport["output_limited"])

    async def test_resume_contract_binds_generation_and_protocol_policy(
        self,
    ) -> None:
        contract = structured_resume_contract(
            model="google/gemini-3.6-flash",
            messages=[{"role": "user", "content": "Верни документ."}],
            schema_name="aiv_continuable_document",
            response_schema=self.schema,
            document_id="contract-v2-test",
            reasoning_effort="high",
            temperature=0.1,
            overlap_chars=12,
        )

        self.assertEqual(
            contract["version"],
            "aiv-structured-resume-contract-v2",
        )
        self.assertEqual(contract["reasoning_effort"], "high")
        self.assertEqual(contract["temperature"], 0.1)
        self.assertEqual(contract["overlap_chars"], 12)
        self.assertEqual(
            contract["continuation_protocol_version"],
            "aiv-structured-continuation-v2",
        )
        self.assertEqual(
            contract["long_response_harness_version"],
            "aiv-long-response-harness-v2",
        )

    async def test_full_prefix_context_overflow_fails_with_checkpoint(
        self,
    ) -> None:
        initial = '{"items":[{"name":"' + ("а" * 8_000)
        events: list[dict[str, Any]] = []
        client = _SequenceClient([self._part(initial, limited=True)])
        constrained_envelope = {
            **self.envelope,
            "context_length": 5_000,
        }
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
                "app.services.openrouter.model_output_envelope",
                return_value=constrained_envelope,
            ),
            self.assertRaisesRegex(
                OpenRouterStructuredContinuationError,
                "no completion headroom",
            ) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                audit_checkpoint=events.append,
            )

        self.assertEqual(len(client.requests), 1)
        self.assertEqual(
            raised.exception.manifest["accepted_document_text"],
            initial,
        )
        checkpoints = [
            event
            for event in events
            if event["event_kind"] == "structured_continuation_checkpoint"
        ]
        self.assertEqual(
            [event["status"] for event in checkpoints],
            ["partial", "failed"],
        )
        self.assertEqual(checkpoints[-1]["partial_text"], initial)

    async def test_completed_resume_rejects_generation_policy_change(
        self,
    ) -> None:
        complete = '{"items":[{"name":"Политика"}]}'
        events: list[dict[str, Any]] = []
        await self._run(
            [self._part(complete, limited=False)],
            audit_checkpoint=events.append,
            reasoning_effort="high",
            temperature=0.1,
        )
        checkpoint = next(
            event
            for event in reversed(events)
            if event["event_kind"] == "structured_continuation_checkpoint"
        )

        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("contract mismatch must be offline"),
            ) as client_factory,
            self.assertRaisesRegex(OpenRouterError, "contract"),
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                reasoning_effort="high",
                temperature=0.2,
                resume_checkpoint=checkpoint,
                document_id=checkpoint["document_id"],
            )
        client_factory.assert_not_called()

    async def test_structured_reasoning_identity_is_canonical_on_wire(
        self,
    ) -> None:
        events: list[dict[str, Any]] = []
        result, client = await self._run(
            [self._part('{"items":[]}', limited=False)],
            audit_checkpoint=events.append,
            reasoning_effort=" high ",
        )

        self.assertEqual(result.parsed, {"items": []})
        self.assertEqual(
            client.requests[0]["json"]["reasoning"],
            {"effort": "high", "exclude": True},
        )
        provider = next(
            event for event in events if event["event_kind"] == "provider_post"
        )
        self.assertEqual(
            provider["request_sha256"],
            hashlib.sha256(client.requests[0]["content"]).hexdigest(),
        )
        self.assertEqual(
            provider["request_body_utf8_bytes"],
            len(client.requests[0]["content"]),
        )
        restored_atomic = restore_completed_chat_provider_event(
            provider,
            model="google/gemini-3.6-flash",
            messages=[{"role": "user", "content": "Верни документ."}],
            web_policy=WebSearchPolicy.FORBIDDEN,
            temperature=0.2,
            response_schema=self.schema,
            schema_name="aiv_continuable_document",
            reasoning_effort="high",
        )
        self.assertEqual(restored_atomic.parsed, {"items": []})
        self.assertEqual(restored_atomic.text, result.text)
        self.assertEqual(
            provider["resume_contract"]["reasoning_effort"],
            "high",
        )
        tampered_encoding = deepcopy(provider)
        tampered_encoding["request_body_encoding"] = "unknown"
        with self.assertRaisesRegex(OpenRouterError, "encoding mismatch"):
            restore_completed_chat_provider_event(
                tampered_encoding,
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                web_policy=WebSearchPolicy.FORBIDDEN,
                temperature=0.2,
                response_schema=self.schema,
                schema_name="aiv_continuable_document",
                reasoning_effort="high",
            )
        checkpoint = next(
            event
            for event in reversed(events)
            if event["event_kind"] == "structured_continuation_checkpoint"
        )
        restored = restore_completed_structured_checkpoint(
            checkpoint,
            "google/gemini-3.6-flash",
            [{"role": "user", "content": "Верни документ."}],
            "aiv_continuable_document",
            self.schema,
            checkpoint["document_id"],
            overlap_chars=12,
            reasoning_effort=" high ",
        )
        self.assertEqual(restored.parsed, {"items": []})

    async def test_capability_negotiates_compact_delta_audit_events(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"},'
        continuation = initial[-12:] + '{"name":"Б"}]}'
        events: list[dict[str, Any]] = []

        async def delta_sink(event: dict[str, Any]) -> None:
            events.append(event)

        setattr(
            delta_sink,
            STRUCTURED_AUDIT_EVENT_VERSION_ATTR,
            STRUCTURED_AUDIT_DELTA_CAPABILITY,
        )
        await self._run(
            [
                self._part(initial, limited=True),
                self._part(continuation, limited=False),
            ],
            audit_checkpoint=delta_sink,
        )

        provider_events = [
            event for event in events if event["event_kind"] == "provider_post"
        ]
        checkpoints = [
            event
            for event in events
            if event["event_kind"] == "structured_continuation_checkpoint"
        ]
        self.assertEqual(
            {event["version"] for event in provider_events},
            {PHYSICAL_POST_AUDIT_DELTA_VERSION},
        )
        self.assertEqual(
            {event["version"] for event in checkpoints},
            {STRUCTURED_CHECKPOINT_DELTA_VERSION},
        )
        self.assertTrue(
            all(
                key not in event
                for event in provider_events
                for key in (
                    "partial_text",
                    "manifest",
                    "aggregate_usage",
                    "call_records",
                )
            )
        )
        self.assertEqual(provider_events[0]["predecessor"]["latest_sequence"], -1)
        self.assertEqual(provider_events[1]["predecessor"]["latest_sequence"], 0)
        self.assertTrue(
            all(
                key not in event
                for event in checkpoints
                for key in (
                    "partial_text",
                    "manifest",
                    "aggregate_usage",
                    "call_records",
                )
            )
        )
        self.assertEqual([event["sequence"] for event in checkpoints], [0, 1])
        self.assertTrue(checkpoints[-1]["complete"])
        self.assertEqual(checkpoints[-1]["head"]["part_count"], 2)
        self.assertEqual(
            checkpoints[-1]["accepted_fragment"]["sequence"],
            1,
        )
        self.assertNotIn(
            "raw_text",
            checkpoints[-1]["accepted_fragment"],
        )

    async def test_plain_audit_sink_keeps_snapshot_v1_wire_format(self) -> None:
        events: list[dict[str, Any]] = []
        await self._run(
            [self._part('{"items":[]}', limited=False)],
            audit_checkpoint=events.append,
        )

        provider = next(
            event for event in events if event["event_kind"] == "provider_post"
        )
        checkpoint = next(
            event
            for event in events
            if event["event_kind"] == "structured_continuation_checkpoint"
        )
        self.assertEqual(provider["version"], PHYSICAL_POST_AUDIT_VERSION)
        self.assertEqual(checkpoint["version"], STRUCTURED_CHECKPOINT_VERSION)
        self.assertIn("call_records", provider)
        self.assertIn("call_records", checkpoint)

    async def test_unknown_audit_capability_fails_before_provider_post(
        self,
    ) -> None:
        async def unsupported_sink(_event: dict[str, Any]) -> None:
            pass

        setattr(
            unsupported_sink,
            STRUCTURED_AUDIT_EVENT_VERSION_ATTR,
            "aiv-structured-audit-delta-v999",
        )
        with self.assertRaisesRegex(
            OpenRouterError,
            "Unsupported structured audit event capability",
        ):
            await self._run(
                [self._part('{"items":[]}', limited=False)],
                audit_checkpoint=unsupported_sink,
            )

    async def test_complete_json_with_length_signal_needs_no_continuation(
        self,
    ) -> None:
        complete = '{"items":[{"name":"Уже завершено"}]}'
        result, client = await self._run(
            [self._part(complete, limited=True)]
        )

        self.assertEqual(
            result.parsed,
            {"items": [{"name": "Уже завершено"}]},
        )
        self.assertEqual(len(client.requests), 1)
        manifest = result.usage["_aiv_structured_continuation"]
        self.assertEqual(manifest["continuation_count"], 0)
        self.assertEqual(
            manifest["parts"][0]["kind"],
            "initial_complete_with_limit_signal",
        )

    async def test_continuation_complete_with_length_signal_stops(self) -> None:
        initial = '{"items":[{"name":"А"},'
        continuation = initial[-12:] + '{"name":"Б"}]}'

        result, client = await self._run(
            [
                self._part(initial, limited=True),
                self._part(continuation, limited=True),
            ]
        )

        self.assertEqual(
            result.parsed,
            {"items": [{"name": "А"}, {"name": "Б"}]},
        )
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(
            result.usage["_aiv_structured_continuation"]["parts"][-1][
                "kind"
            ],
            "literal_continuation_complete_with_limit_signal",
        )

    async def test_completed_checkpoint_is_restored_without_provider_post(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"},'
        continuation = initial[-12:] + '{"name":"Б"}]}'
        events: list[dict[str, Any]] = []
        initial_body = self._part(initial, limited=True)
        initial_body["usage"].update(
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        continuation_body = self._part(continuation, limited=False)
        continuation_body["usage"].update(
            {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        )
        original, client = await self._run(
            [initial_body, continuation_body],
            audit_checkpoint=events.append,
        )
        checkpoint = next(
            event
            for event in reversed(events)
            if event["event_kind"] == "structured_continuation_checkpoint"
            and event["status"] == "completed"
        )

        with patch(
            "app.services.openrouter.httpx.AsyncClient",
            side_effect=AssertionError("restore must not use the network"),
        ) as client_factory:
            restored = restore_completed_structured_checkpoint(
                checkpoint,
                "google/gemini-3.6-flash",
                [{"role": "user", "content": "Верни документ."}],
                "aiv_continuable_document",
                self.schema,
                checkpoint["document_id"],
                overlap_chars=12,
            )

        client_factory.assert_not_called()
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(restored.text, original.text)
        self.assertEqual(restored.parsed, original.parsed)
        self.assertEqual(
            restored.usage["total_tokens"],
            original.usage["total_tokens"],
        )
        self.assertTrue(
            restored.usage["_aiv_structured_continuation"]["complete"]
        )
        with patch(
            "app.services.openrouter.httpx.AsyncClient",
            side_effect=AssertionError("completed resume must not use network"),
        ) as resume_client_factory:
            resumed = await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                document_id=checkpoint["document_id"],
                overlap_chars=12,
                resume_checkpoint=checkpoint,
            )
        resume_client_factory.assert_not_called()
        self.assertEqual(resumed.text, original.text)
        self.assertEqual(resumed.parsed, original.parsed)

    async def test_tampered_completed_checkpoint_fails_before_network(
        self,
    ) -> None:
        complete = '{"items":[{"name":"Готово"}]}'
        events: list[dict[str, Any]] = []
        await self._run(
            [self._part(complete, limited=False)],
            audit_checkpoint=events.append,
        )
        checkpoint = deepcopy(
            next(
                event
                for event in reversed(events)
                if event["event_kind"]
                == "structured_continuation_checkpoint"
                and event["status"] == "completed"
            )
        )
        checkpoint["call_records"][0]["raw_text"] += "tampered"

        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError("tamper path must not use network"),
            ) as client_factory,
            self.assertRaisesRegex(OpenRouterError, "ledger mismatch|digest"),
        ):
            restore_completed_structured_checkpoint(
                checkpoint,
                "google/gemini-3.6-flash",
                [{"role": "user", "content": "Верни документ."}],
                "aiv_continuable_document",
                self.schema,
                checkpoint["document_id"],
                overlap_chars=12,
            )

        client_factory.assert_not_called()
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                side_effect=AssertionError(
                    "tampered completed resume must not use network"
                ),
            ) as resume_client_factory,
            self.assertRaises(OpenRouterError),
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                document_id=checkpoint["document_id"],
                overlap_chars=12,
                resume_checkpoint=checkpoint,
            )
        resume_client_factory.assert_not_called()

    async def test_initial_output_limited_provider_gap_is_promoted(self) -> None:
        initial = '{"items":[{"name":"А"},'
        bad_following = '{"name":"нет overlap"}]}'
        events: list[dict[str, Any]] = []
        with self.assertRaises(OpenRouterStructuredContinuationError):
            await self._run(
                [
                    self._part(initial, limited=True),
                    self._part(bad_following, limited=False),
                ],
                audit_checkpoint=events.append,
            )
        provider_event = next(
            event
            for event in events
            if event["event_kind"] == "provider_post"
            and event["sequence"] == 0
        )

        promoted = promote_provider_post_to_structured_checkpoint(
            provider_event,
            None,
            "google/gemini-3.6-flash",
            [{"role": "user", "content": "Верни документ."}],
            "aiv_continuable_document",
            self.schema,
            provider_event["document_id"],
            overlap_chars=12,
        )

        contract = structured_resume_contract(
            model="google/gemini-3.6-flash",
            messages=[{"role": "user", "content": "Верни документ."}],
            schema_name="aiv_continuable_document",
            response_schema=self.schema,
            document_id=provider_event["document_id"],
            overlap_chars=12,
        )
        self.assertEqual(provider_event["resume_contract"], contract)
        self.assertEqual(len(contract["sha256"]), 64)
        self.assertEqual(provider_event["status"], "rejected")
        self.assertEqual(promoted["status"], "partial")
        self.assertEqual(promoted["partial_text"], initial)
        self.assertEqual(promoted["manifest"]["part_count"], 1)
        self.assertEqual(
            promoted["promoted_from_provider_event_id"],
            provider_event["event_id"],
        )

    async def test_continuation_provider_gap_is_promoted(self) -> None:
        initial = '{"items":[{"name":"А"},'
        following = initial[-12:] + '{"name":"Б"},'
        bad_final = '{"items":[]}'
        events: list[dict[str, Any]] = []
        with self.assertRaises(OpenRouterStructuredContinuationError):
            await self._run(
                [
                    self._part(initial, limited=True),
                    self._part(following, limited=True),
                    self._part(bad_final, limited=False),
                ],
                audit_checkpoint=events.append,
            )
        predecessor = next(
            event
            for event in events
            if event["event_kind"] == "structured_continuation_checkpoint"
            and event["sequence"] == 0
            and event["status"] == "partial"
        )
        provider_event = next(
            event
            for event in events
            if event["event_kind"] == "provider_post"
            and event["sequence"] == 1
        )

        promoted = promote_provider_post_to_structured_checkpoint(
            provider_event,
            predecessor,
            "google/gemini-3.6-flash",
            [{"role": "user", "content": "Верни документ."}],
            "aiv_continuable_document",
            self.schema,
            provider_event["document_id"],
            overlap_chars=12,
        )

        self.assertEqual(promoted["status"], "partial")
        self.assertEqual(promoted["sequence"], 1)
        self.assertEqual(promoted["partial_text"], initial + '{"name":"Б"},')
        self.assertEqual(promoted["manifest"]["part_count"], 2)

    async def test_final_complete_provider_gap_is_promoted(self) -> None:
        initial = '{"items":[{"name":"А"},'
        final = initial[-12:] + '{"name":"Б"}]}'
        events: list[dict[str, Any]] = []
        original, _client = await self._run(
            [
                self._part(initial, limited=True),
                self._part(final, limited=False),
            ],
            audit_checkpoint=events.append,
        )
        predecessor = next(
            event
            for event in events
            if event["event_kind"] == "structured_continuation_checkpoint"
            and event["sequence"] == 0
            and event["status"] == "partial"
        )
        provider_event = next(
            event
            for event in events
            if event["event_kind"] == "provider_post"
            and event["sequence"] == 1
        )

        promoted = promote_provider_post_to_structured_checkpoint(
            provider_event,
            predecessor,
            "google/gemini-3.6-flash",
            [{"role": "user", "content": "Верни документ."}],
            "aiv_continuable_document",
            self.schema,
            provider_event["document_id"],
            overlap_chars=12,
        )
        restored = restore_completed_structured_checkpoint(
            promoted,
            "google/gemini-3.6-flash",
            [{"role": "user", "content": "Верни документ."}],
            "aiv_continuable_document",
            self.schema,
            provider_event["document_id"],
            overlap_chars=12,
        )

        self.assertEqual(promoted["status"], "completed")
        self.assertTrue(promoted["manifest"]["complete"])
        self.assertEqual(restored.parsed, original.parsed)

    async def test_provider_gap_tamper_and_unusable_rejection_fail(self) -> None:
        initial = '{"items":[{"name":"А"},'
        bad_following = '{"name":"нет overlap"}]}'
        events: list[dict[str, Any]] = []
        with self.assertRaises(OpenRouterStructuredContinuationError):
            await self._run(
                [
                    self._part(initial, limited=True),
                    self._part(bad_following, limited=False),
                ],
                audit_checkpoint=events.append,
            )
        provider_event = next(
            event
            for event in events
            if event["event_kind"] == "provider_post"
            and event["sequence"] == 0
        )
        args = (
            None,
            "google/gemini-3.6-flash",
            [{"role": "user", "content": "Верни документ."}],
            "aiv_continuable_document",
            self.schema,
            provider_event["document_id"],
        )

        tampered = deepcopy(provider_event)
        tampered["raw_text"] += "x"
        with self.assertRaisesRegex(OpenRouterError, "raw response mismatch"):
            promote_provider_post_to_structured_checkpoint(
                tampered,
                *args,
                overlap_chars=12,
            )

        unusable = deepcopy(provider_event)
        unusable["error"]["type"] = "OpenRouterResponseContractError"
        with self.assertRaisesRegex(OpenRouterError, "not a usable limit part"):
            promote_provider_post_to_structured_checkpoint(
                unusable,
                *args,
                overlap_chars=12,
            )

    async def test_short_valid_part_is_not_a_production_length_cap(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"'
        after_short = initial + "}"
        short = initial[-12:] + "}"
        final = after_short[-12:] + "]}"

        result, client = await self._run(
            [
                self._part(initial, limited=True),
                self._part(short, limited=True),
                self._part(final, limited=False),
            ]
        )

        self.assertEqual(result.parsed, {"items": [{"name": "А"}]})
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(
            result.usage["_aiv_structured_continuation"]["parts"][1][
                "appended_chars"
            ],
            1,
        )

    async def test_missing_overlap_is_rejected_with_partial_audit(self) -> None:
        initial = '{"items":[{"name":"А"},'
        client = _SequenceClient(
            [
                self._part(initial, limited=True),
                self._part('{"name":"Б"}]}', limited=False),
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
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaises(OpenRouterStructuredContinuationError) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                retry_transport_errors=False,
            )

        self.assertEqual(len(client.requests), 2)
        self.assertFalse(raised.exception.manifest["complete"])
        self.assertEqual(raised.exception.manifest["part_count"], 1)
        self.assertIn("rejected_part", raised.exception.manifest)
        self.assertEqual(
            raised.exception.result.text,
            '{"name":"Б"}]}',
        )
        self.assertEqual(
            raised.exception.manifest["rejected_part"]["raw_text"],
            '{"name":"Б"}]}',
        )
        self.assertEqual(
            raised.exception.manifest["accepted_document_text"],
            initial,
        )
        self.assertIn("boundary", str(raised.exception))

    async def test_continuation_usage_is_aggregated_across_every_post(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"},'
        continuation = initial[-12:] + '{"name":"Б"}]}'
        first = self._part(initial, limited=True)
        first["usage"].update(
            {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0.01}
        )
        second = self._part(continuation, limited=False)
        second["usage"].update(
            {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40, "cost": 0.02}
        )

        result, _client = await self._run([first, second])

        self.assertEqual(result.usage["prompt_tokens"], 35)
        self.assertEqual(result.usage["completion_tokens"], 25)
        self.assertEqual(result.usage["total_tokens"], 60)
        self.assertAlmostEqual(result.usage["cost"], 0.03)
        calls = result.usage["_aiv_structured_continuation"]["calls"]
        self.assertEqual([call["usage"]["total_tokens"] for call in calls], [20, 40])
        self.assertEqual([call["raw_text"] for call in calls], [initial, continuation])
        self.assertTrue(
            all("_aiv_call_attempts" not in call["usage"] for call in calls)
        )
        self.assertTrue(
            all(call["prior_transport_attempts"] == [] for call in calls)
        )

    async def test_duplicate_continuation_is_rejected(self) -> None:
        initial = '{"items":['
        client = _SequenceClient(
            [
                self._part(initial, limited=True),
                self._part(initial, limited=True),
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
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaisesRegex(
                OpenRouterStructuredContinuationError,
                "already seen|progress",
            ),
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=len(initial),
                retry_transport_errors=False,
            )

        self.assertEqual(len(client.requests), 2)

    async def test_no_local_continuation_count_ceiling(self) -> None:
        assembled = '{"items":['
        bodies = [self._part(assembled, limited=True)]
        for index, name in enumerate(("А", "Б", "В", "Г")):
            addition = ("," if index else "") + f'{{"name":"{name}"}}'
            bodies.append(
                self._part(assembled[-12:] + addition, limited=True)
            )
            assembled += addition
        bodies.append(self._part(assembled[-12:] + "]}", limited=False))

        result, client = await self._run(bodies)

        self.assertEqual(
            result.parsed,
            {
                "items": [
                    {"name": "А"},
                    {"name": "Б"},
                    {"name": "В"},
                    {"name": "Г"},
                ]
            },
        )
        self.assertEqual(len(client.requests), 6)
        self.assertEqual(
            result.usage["_aiv_structured_continuation"][
                "continuation_count"
            ],
            5,
        )

    def test_transport_api_has_no_length_or_total_time_stop_parameters(
        self,
    ) -> None:
        parameters = inspect.signature(chat_continuable_structured).parameters
        self.assertNotIn("max_continuations", parameters)
        self.assertNotIn("min_limited_progress_chars", parameters)
        self.assertNotIn("continuation_deadline_seconds", parameters)

    async def test_final_schema_violation_is_never_accepted(self) -> None:
        initial = '{"items":[{"name":"А"},'
        continuation = initial[-12:] + '{"wrong":"Б"}]}'
        client = _SequenceClient(
            [
                self._part(initial, limited=True),
                self._part(continuation, limited=False),
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
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaisesRegex(
                OpenRouterStructuredContinuationError,
                "schema|unusable|required",
            ) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                retry_transport_errors=False,
            )

        self.assertEqual(len(client.requests), 2)
        self.assertIsNone(raised.exception.result.parsed)
        self.assertFalse(raised.exception.manifest["complete"])

    async def test_initial_impossible_json_prefix_keeps_result_and_manifest(
        self,
    ) -> None:
        invalid = '{"items":]'
        events: list[dict[str, Any]] = []
        client = _SequenceClient([self._part(invalid, limited=True)])
        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=client),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaises(OpenRouterStructuredContinuationError) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=8,
                audit_checkpoint=events.append,
            )

        self.assertEqual(raised.exception.result.text, invalid)
        self.assertEqual(
            raised.exception.manifest["rejected_part"]["raw_text"],
            invalid,
        )
        self.assertTrue(
            any(event["status"] == "failed" for event in events)
        )

    async def test_transport_failure_after_partial_is_wrapped_with_ledger(
        self,
    ) -> None:
        class StatusResponse(_FakeResponse):
            def __init__(self, body: dict[str, Any], status: int):
                super().__init__(body)
                self.status_code = status

        initial = '{"items":[{"name":"А"},'

        class Client(_FakeClient):
            def __init__(self) -> None:
                super().__init__(_body())
                self.index = 0

            async def post(self, _url: str, *, headers: dict[str, str], content: bytes) -> _FakeResponse:
                payload = json_lib.loads(content.decode("utf-8"))
                self.requests.append({"headers": dict(headers), "json": payload})
                self.index += 1
                if self.index == 1:
                    return StatusResponse(
                        OpenRouterStructuredContinuationTests()._part(
                            initial,
                            limited=True,
                        ),
                        200,
                    )
                return StatusResponse({"error": {"message": "down"}}, 500)

        client = Client()
        events: list[dict[str, Any]] = []
        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=client),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaises(OpenRouterStructuredContinuationError) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                audit_checkpoint=events.append,
            )

        self.assertEqual(raised.exception.manifest["accepted_document_text"], initial)
        self.assertEqual(raised.exception.manifest["part_count"], 1)
        self.assertTrue(hasattr(raised.exception, "result"))
        provider_events = [
            event for event in events if event["event_kind"] == "provider_post"
        ]
        self.assertEqual(
            [event["status"] for event in provider_events],
            ["rejected", "http_error"],
        )

    async def test_provider_read_timeout_preserves_failed_checkpoint(self) -> None:
        class InactiveClient(_FakeClient):
            async def post(
                self,
                _url: str,
                *,
                headers: dict[str, str],
                content: bytes,
            ) -> _FakeResponse:
                payload = json_lib.loads(content.decode("utf-8"))
                self.requests.append(
                    {"headers": dict(headers), "json": payload}
                )
                raise httpx.ReadTimeout("provider read inactivity")

        client = InactiveClient(_body())
        events: list[dict[str, Any]] = []
        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaisesRegex(
                OpenRouterStructuredContinuationError,
                "provider POST",
            ) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                audit_checkpoint=events.append,
            )

        self.assertEqual(len(client.requests), 1)
        self.assertFalse(raised.exception.manifest["complete"])
        self.assertEqual(
            [event["status"] for event in events],
            ["transport_error", "failed"],
        )

    async def test_each_post_has_self_contained_checkpoint_and_safe_resume(
        self,
    ) -> None:
        initial = '{"items":[{"name":"А"},'
        after_first = initial + '{"name":"Б"},'
        first = initial[-12:] + '{"name":"Б"},'
        final = after_first[-12:] + '{"name":"В"}]}'
        events: list[dict[str, Any]] = []
        first_client = _SequenceClient(
            [self._part(initial, limited=True), self._part(first, limited=True)]
        )

        def stop_after_durable_partial(event: dict[str, Any]) -> None:
            events.append(event)
            if (
                event.get("event_kind")
                == "structured_continuation_checkpoint"
                and event.get("status") == "partial"
                and event.get("sequence") == 1
            ):
                raise RuntimeError("test interruption after durable partial")

        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=first_client),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaises(OpenRouterStructuredContinuationError),
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                audit_checkpoint=stop_after_durable_partial,
            )

        post_events = [
            event for event in events if event["event_kind"] == "provider_post"
        ]
        self.assertEqual(len(post_events), 2)
        for event in post_events:
            self.assertTrue(event["document_id"])
            self.assertIsNotNone(event["manifest"])
            self.assertIn("call_records", event)
            self.assertTrue(event["resume_contract"])
            self.assertIn("body_json", event["response"])

        resume = next(
            event
            for event in reversed(events)
            if event["event_kind"] == "structured_continuation_checkpoint"
            and event["status"] == "partial"
            and event["sequence"] == 1
        )
        resumed_events: list[dict[str, Any]] = []
        resumed_client = _SequenceClient([self._part(final, limited=False)])
        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=resumed_client),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
        ):
            result = await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                audit_checkpoint=resumed_events.append,
                resume_checkpoint=resume,
            )

        self.assertEqual(len(resumed_client.requests), 1)
        self.assertEqual(
            result.parsed,
            {"items": [{"name": "А"}, {"name": "Б"}, {"name": "В"}]},
        )
        self.assertEqual(
            result.usage["_aiv_structured_continuation"]["continuation_count"],
            2,
        )

        tampered = deepcopy(resume)
        tampered["call_records"][1]["raw_text"] += "x"
        no_post_client = _SequenceClient([self._part(final, limited=False)])
        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=no_post_client),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaisesRegex(OpenRouterError, "ledger mismatch|digest"),
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                resume_checkpoint=tampered,
            )
        self.assertEqual(len(no_post_client.requests), 0)

        nested_tamper = deepcopy(resume)
        nested = deepcopy(nested_tamper["call_records"][0])
        nested.pop("sequence", None)
        nested["attempt"] = 1
        nested["text_sha256"] = "0" * 64
        nested_tamper["call_records"][0]["prior_transport_attempts"] = [
            nested
        ]
        nested_tamper["manifest"]["calls"] = deepcopy(
            nested_tamper["call_records"]
        )
        with self.assertRaisesRegex(OpenRouterError, "prior attempt.*digest"):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                overlap_chars=12,
                resume_checkpoint=nested_tamper,
            )

    async def test_cancelled_initial_post_preserves_checkpoint(self) -> None:
        class CancelClient(_FakeClient):
            async def post(self, _url: str, *, headers: dict[str, str], content: bytes) -> _FakeResponse:
                payload = json_lib.loads(content.decode("utf-8"))
                self.requests.append({"headers": dict(headers), "json": payload})
                raise asyncio.CancelledError()

        client = CancelClient(_body())
        events: list[dict[str, Any]] = []
        with (
            patch("app.services.openrouter.httpx.AsyncClient", return_value=client),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaises(
                OpenRouterStructuredContinuationCancelled
            ) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                audit_checkpoint=events.append,
            )

        self.assertTrue(hasattr(raised.exception, "result"))
        self.assertFalse(raised.exception.manifest["complete"])
        self.assertTrue(any(event["status"] == "cancelled" for event in events))

    async def test_cancel_during_successful_audit_keeps_accepted_head(
        self,
    ) -> None:
        complete = '{"items":[{"name":"Сохранено"}]}'
        client = _SequenceClient([self._part(complete, limited=False)])
        sink_entered = asyncio.Event()
        release_sink = asyncio.Event()
        persisted: list[dict[str, Any]] = []

        async def sink(event: dict[str, Any]) -> None:
            if not persisted and event["event_kind"] == "provider_post":
                sink_entered.set()
                await release_sink.wait()
            persisted.append(event)

        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
        ):
            task = asyncio.create_task(
                chat_continuable_structured(
                    model="google/gemini-3.6-flash",
                    messages=[
                        {"role": "user", "content": "Верни документ."}
                    ],
                    response_schema=self.schema,
                    audit_checkpoint=sink,
                )
            )
            await sink_entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            release_sink.set()
            with self.assertRaises(
                OpenRouterStructuredContinuationCancelled
            ) as raised:
                await task

        provider_events = [
            event
            for event in persisted
            if event["event_kind"] == "provider_post"
        ]
        self.assertEqual(
            [event["status"] for event in provider_events],
            ["accepted"],
        )
        self.assertEqual(raised.exception.result.text, complete)
        self.assertEqual(
            raised.exception.manifest["accepted_document_text"],
            complete,
        )
        self.assertEqual(len(raised.exception.manifest["calls"]), 1)
        self.assertEqual(persisted[-1]["status"], "cancelled")

    async def test_structured_checkpoint_sink_failure_keeps_result_manifest(
        self,
    ) -> None:
        complete = '{"items":[{"name":"Готово"}]}'
        client = _SequenceClient([self._part(complete, limited=False)])
        persisted: list[dict[str, Any]] = []

        async def sink(event: dict[str, Any]) -> None:
            persisted.append(event)
            if event["event_kind"] == "structured_continuation_checkpoint":
                raise RuntimeError("database unavailable")

        with (
            patch(
                "app.services.openrouter.httpx.AsyncClient",
                return_value=client,
            ),
            patch("app.services.openrouter._headers", return_value={}),
            patch(
                "app.services.openrouter.model_output_envelope",
                return_value=dict(self.envelope),
            ),
            self.assertRaises(OpenRouterAuditCheckpointError) as raised,
        ):
            await chat_continuable_structured(
                model="google/gemini-3.6-flash",
                messages=[{"role": "user", "content": "Верни документ."}],
                response_schema=self.schema,
                audit_checkpoint=sink,
            )

        self.assertEqual(
            [event["event_kind"] for event in persisted],
            ["provider_post", "structured_continuation_checkpoint"],
        )
        self.assertEqual(raised.exception.result.text, complete)
        self.assertTrue(raised.exception.manifest["complete"])
