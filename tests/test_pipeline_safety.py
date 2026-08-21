import asyncio
import base64
import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, func, select, text

from app.config import settings
from app.db import SessionLocal, init_db
from app.main import app
from app.models import (
    AnswerAnnotation,
    DomainProbe,
    ModelAnswer,
    ProbeType,
    ReportIllustration,
    RobotsRule,
    Run,
    RunArtifact,
    RunStatus,
    SitePage,
    VisibilityPrompt,
)
from app.services import crawler
from app.services.openrouter import (
    WEB_ATTESTATION_VERSION,
    OpenRouterError,
    PanelModel,
    WebSearchPolicy,
    attest_web_response,
    panel_models,
    web_request_policy,
)
from app.services.report_semantic_gate import (
    CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
)
from app.services.analyzer import (
    ANALYSIS_MODEL,
    ANNOTATION_VERSION,
    ENTITY_CATALOG_CHUNK_VERSION,
    ENTITY_CATALOG_VERSION,
    FINAL_REPORT_VERSION,
    FINAL_CONTEXT_MAX_ANSWERS,
    FINAL_REPORT_AUTHOR_ARTIFACT_KEY,
    FINAL_REPORT_SCHEMA,
    ILLUSTRATION_CONCEPTS_SCHEMA,
    ILLUSTRATION_GENERATION_VERSION,
    ILLUSTRATION_CONCEPT_MODEL,
    ILLUSTRATION_QA_SCHEMA,
    ILLUSTRATION_ROLE_CONCURRENCY,
    LEGACY_PANEL_CONTRACT_VERSION,
    LEGACY_PANEL_EVIDENCE_VERSION,
    LEGACY_MEMORY_OBSERVATION_REASON,
    LEGACY_MEMORY_MODELS,
    PROCESSING_BATCH_CONCURRENCY,
    PROCESSING_MODEL,
    PANEL_CONTRACT_VERSION,
    MARKET_RESEARCH_SCHEMA,
    MARKET_RESEARCH_VERSION,
    METRICS_VERSION,
    PROMPT_SET_REVIEW_SCHEMA,
    PROMPT_SET_SCHEMA,
    PROMPT_SET_VERSION,
    SITE_PROFILE_SCHEMA,
    _annotate_answers,
    _annotation_context_sha256,
    _attribution_owner_aliases,
    _answers_for_catalog,
    _artifact_cache_matches,
    _build_public_report,
    _classify_site,
    _compute_metrics,
    _deterministic_annotation_warnings,
    _entity_catalog,
    _entity_alias_entries,
    _evidence_contains_complete_alias,
    _evidence_is_literal,
    _ensure_answer_rows,
    _expected_corpus_cells,
    _final_corpus_manifest,
    _final_input_preflight,
    _final_report,
    _final_report_payload,
    _full_answer_context,
    _full_answer_corpus_items,
    _generate_illustrations,
    _generate_prompt_set,
    _generate_reviewed_image,
    _illustration_cache_matches,
    _illustration_concepts,
    _illustration_generation_concept,
    _illustration_prompt,
    _illustration_review_errors,
    _legacy_panel_request_sha256,
    _legacy_panel_run_contract,
    _panel_metric_access,
    _market_research,
    _market_research_sufficiency,
    _metric_rows,
    _normalize_unpaired_memory_illustration,
    _panel_answer_attestation,
    _panel_request_sha256,
    _panel_web_policy,
    _persist_prompts,
    _portfolio_entity_is_grounded,
    _portfolio_mention_policy,
    _prompt_review_errors,
    _processing_artifact,
    _probe_access_outcome,
    _reconcile_annotation,
    reprocess_saved_answers,
    _rendering_assessment,
    _review_prompt_set_semantics,
    _review_illustration,
    _reuse_saved_illustration_assets,
    _scope_entity_catalog_to_profile,
    _run_report_branches,
    _run_reused_report_branches,
    _rows_from_full_answer_models,
    _sanitize_headline_emphasis,
    _select_final_answer_context,
    _site_context,
    _unannotated_answers,
    _validate_final_report,
    _validate_illustration_concepts,
    _validate_prompt_set,
    MarketResearchGateError,
)
from app.services.content_extractor import extract_text_signals
from app.services.robots_parser import parse_robots, robots_path_allowed


def _rules_by_bot(value: str) -> dict[str, tuple[str, str]]:
    return {bot: (rule, raw) for bot, rule, raw in parse_robots(value)}


def _attested_panel_usage(
    *,
    prompt_text: str,
    mode: str,
    provider_key: str,
    model: str,
) -> tuple[dict[str, object], list[dict[str, str]] | None]:
    policy = _panel_web_policy(mode, provider_key)
    _fields, request_policy = web_request_policy(
        model=model,
        policy=policy,
    )
    has_retrieval = policy is not WebSearchPolicy.FORBIDDEN
    citations = (
        [{"url": "https://source.example", "title": "Source", "content": "Fact"}]
        if has_retrieval
        else []
    )
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
        if has_retrieval
        else []
    )
    raw_usage: dict[str, object] = {
        "server_tool_use": {
            "web_search_requests": (
                1 if policy is WebSearchPolicy.REQUIRED else 0
            )
        }
    }
    attestation = attest_web_response(
        requested_model=model,
        response_model=model,
        policy=policy,
        usage=raw_usage,
        annotations=annotations,
        citations=citations,
        router_metadata={},
    )
    usage: dict[str, object] = {
        **raw_usage,
        "_aiv_request_policy": request_policy,
        "_aiv_response_annotations": annotations,
        "_aiv_router_metadata": {},
        "_aiv_web_attestation": attestation,
        "_aiv_panel_contract": {
            "version": PANEL_CONTRACT_VERSION,
            "request_sha256": _panel_request_sha256(
                prompt_text=prompt_text,
                mode=mode,
                provider_key=provider_key,
                model=model,
            ),
            "request_policy_sha256": request_policy["sha256"],
            "web_policy": request_policy["policy"],
            "attestation_version": WEB_ATTESTATION_VERSION,
            "web_attestation": attestation,
        },
    }
    return usage, citations or None


def _ready_market_research(
    *,
    brand_name: str = "Example",
    first_job: str = "Сравнить поставщиков аналитики",
) -> dict[str, object]:
    attestation = {
        "version": WEB_ATTESTATION_VERSION,
        "policy": WebSearchPolicy.REQUIRED.value,
        "state": "verified",
        "metric_eligible": True,
        "web_search_requests": 1,
        "violations": [],
    }
    source_urls = [
        "https://research.example/market",
        "https://industry.example/criteria",
    ]
    dimensions = (
        "market",
        "topics",
        "geography",
        "audiences",
        "customer_jobs",
        "decision_criteria",
        "terminology",
    )
    return {
        "status": "ready",
        "site_confirmed": {
            "primary_brand": brand_name,
            "brand_aliases": [],
            "site_type": "Сайт продукта",
            "category": "Аналитика",
            "products": ["Платформа"],
            "topics": ["AI visibility"],
            "geography": ["Россия"],
            "evidence": [
                {
                    "claim": "Сайт представляет платформу Example.",
                    "url": "https://example.com/",
                    "excerpt": "Example — аналитическая платформа.",
                }
            ],
        },
        "external_market_research": {
            "market": "B2B-аналитика",
            "topics": ["Измерение видимости"],
            "geography": ["Россия"],
            "audiences": ["Руководители маркетинга"],
            "customer_jobs": [first_job],
            "decision_criteria": ["Точность", "Скорость"],
            "terminology": [
                {
                    "term": "AI visibility",
                    "meaning": "Обнаружение бренда в ответах ИИ.",
                }
            ],
            "evidence": [
                {
                    "dimension": dimension,
                    "claim": f"Подтверждено измерение {dimension}.",
                    "source_urls": source_urls,
                    "evidence": f"Источник описывает {dimension}.",
                    "confidence": "high",
                }
                for dimension in dimensions
            ],
        },
        "sources": [
            {
                "url": url,
                "title": f"Источник {index}",
                "publisher": f"Издатель {index}",
                "evidence": "Исследование рынка.",
                "confidence": "high",
            }
            for index, url in enumerate(source_urls, start=1)
        ],
        "uncertainties": [],
        "confidence": "high",
        "sufficiency": {
            "status": "ready",
            "blocking_issues": [],
            "limited_issues": [],
            "required_dimensions": list(dimensions),
            "evidenced_dimensions": list(dimensions),
            "confirmed_source_urls": source_urls,
            "confirmed_external_source_urls": source_urls,
            "confirmed_site_claims": 1,
        },
        "web_evidence": {
            "attestation": attestation,
            "citations": [
                {
                    "url": url,
                    "title": f"Источник {index}",
                    "content": "Подтверждённая выдержка.",
                }
                for index, url in enumerate(source_urls, start=1)
            ],
        },
    }


class RobotsParserTests(unittest.TestCase):
    def test_wildcard_group_applies_to_known_bots(self) -> None:
        rules = _rules_by_bot("User-agent: *\nDisallow: /\n")
        self.assertEqual(rules["GPTBot"][0], "disallow_all")
        self.assertEqual(rules["ClaudeBot"][0], "disallow_all")
        self.assertIn("User-agent: *", rules["GPTBot"][1])

    def test_specific_bot_group_overrides_wildcard(self) -> None:
        rules = _rules_by_bot(
            "User-agent: *\nDisallow: /\n\n"
            "User-agent: GPTBot\nAllow: /\n"
        )
        self.assertEqual(rules["GPTBot"][0], "allow_all")
        self.assertEqual(rules["ClaudeBot"][0], "disallow_all")

    def test_repeated_specific_groups_are_combined(self) -> None:
        rules = _rules_by_bot(
            "User-agent: GPTBot\nDisallow: /private\n\n"
            "User-agent: GPTBot\nAllow: /private/public\n"
        )
        rule, raw = rules["GPTBot"]
        self.assertEqual(rule, "partial")
        self.assertFalse(
            robots_path_allowed(raw, "https://example.com/private/account")
        )
        self.assertTrue(
            robots_path_allowed(raw, "https://example.com/private/public")
        )

    def test_longest_path_rule_and_allow_tie_are_respected(self) -> None:
        raw = (
            "User-agent: *\n"
            "Disallow: /catalog/*\n"
            "Allow: /catalog/public\n"
        )
        self.assertFalse(
            robots_path_allowed(raw, "https://example.com/catalog/secret")
        )
        self.assertTrue(
            robots_path_allowed(raw, "https://example.com/catalog/public")
        )


class DomainAndJobSafetyTests(unittest.TestCase):
    def test_domain_normalization_accepts_urls_and_idna(self) -> None:
        self.assertEqual(
            crawler.normalize_domain(" https://WWW.Example.com/path?q=1 "),
            "example.com",
        )
        self.assertEqual(
            crawler.normalize_domain("https://пример.рф/"),
            "xn--e1afmkfd.xn--p1ai",
        )

    def test_domain_normalization_rejects_local_and_ip_targets(self) -> None:
        for value in ("localhost", "127.0.0.1", "10.0.0.1", "example", ""):
            self.assertEqual(crawler.normalize_domain(value), "")

    def test_build_jobs_uses_representative_pages_and_fixed_identity(self) -> None:
        jobs = crawler._build_jobs(
            ["example.com", "https://www.example.com/path"],
            ["GPTBot", "unknown"],
            [
                ("https://example.com/", "home"),
                ("https://example.com/services", "product"),
            ],
        )
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].probe_type, ProbeType.robots_txt)
        self.assertEqual(jobs[1].page_kind, "home")
        self.assertEqual(jobs[2].page_kind, "product")

    def test_default_headers_do_not_advertise_unsupported_brotli(self) -> None:
        encodings = {
            token.strip().lower()
            for token in crawler.DEFAULT_HEADERS["Accept-Encoding"].split(",")
        }
        self.assertNotIn("br", encodings)

    def test_sitemap_xml_is_parsed_without_html_fallback_noise(self) -> None:
        links = crawler._links_from_sitemap(
            """
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/products</loc></url>
              <url><loc>https://other.example/private</loc></url>
            </urlset>
            """,
            "https://example.com/sitemap.xml",
            "example.com",
        )
        self.assertEqual(links, ["https://example.com/products"])

    def test_confirmation_urls_do_not_enter_the_content_sample(self) -> None:
        self.assertIsNone(
            crawler._canonical_candidate(
                "https://example.com/success",
                "example.com",
            )
        )
        selected = crawler._select_representative_urls(
            "https://example.com/",
            [
                "https://example.com/success",
                "https://example.com/services",
            ],
            3,
        )
        self.assertNotIn(
            "https://example.com/success",
            [url for url, _kind in selected],
        )


class PanelRoutingTests(unittest.TestCase):
    def test_perplexity_uses_its_native_search_endpoint(self) -> None:
        self.assertIs(
            _panel_web_policy("web", "perplexity"),
            WebSearchPolicy.NATIVE_REQUIRED,
        )
        self.assertIs(
            _panel_web_policy("web", "openai"),
            WebSearchPolicy.REQUIRED,
        )
        self.assertIs(
            _panel_web_policy("memory", "openai"),
            WebSearchPolicy.FORBIDDEN,
        )

    def test_required_web_without_observed_search_is_not_eligible(self) -> None:
        attestation = attest_web_response(
            requested_model="openai/gpt-5.4",
            response_model="openai/gpt-5.4",
            policy=WebSearchPolicy.REQUIRED,
            usage={"server_tool_use": {"web_search_requests": 0}},
            annotations=[
                {
                    "type": "url_citation",
                    "url_citation": {"url": "https://source.example"},
                }
            ],
            citations=[{"url": "https://source.example"}],
            router_metadata={},
        )

        self.assertFalse(attestation["metric_eligible"])
        self.assertIn(
            "web_search_requests_not_confirmed",
            attestation["violations"],
        )

    def test_required_web_with_usage_and_annotations_is_verified(self) -> None:
        attestation = attest_web_response(
            requested_model="openai/gpt-5.4",
            response_model="openai/gpt-5.4",
            policy=WebSearchPolicy.REQUIRED,
            usage={"server_tool_use": {"web_search_requests": 2}},
            annotations=[
                {
                    "type": "url_citation",
                    "url_citation": {"url": "https://source.example"},
                }
            ],
            citations=[{"url": "https://source.example"}],
            router_metadata={},
        )

        self.assertTrue(attestation["metric_eligible"])
        self.assertEqual(attestation["web_search_requests"], 2)
        self.assertEqual(attestation["citation_annotations_count"], 1)

    def test_required_web_reads_server_tool_use_details(self) -> None:
        attestation = attest_web_response(
            requested_model="openai/gpt-5.4",
            response_model="openai/gpt-5.4",
            policy=WebSearchPolicy.REQUIRED,
            usage={
                "server_tool_use_details": {"web_search_requests": 3},
            },
            annotations=[
                {
                    "type": "url_citation",
                    "url_citation": {"url": "https://source.example"},
                }
            ],
            citations=[{"url": "https://source.example"}],
            router_metadata={},
        )

        self.assertTrue(attestation["metric_eligible"])
        self.assertEqual(attestation["web_search_requests"], 3)

    def test_memory_request_explicitly_forbids_all_web_access(self) -> None:
        fields, policy = web_request_policy(
            model="openai/gpt-5.4",
            policy=WebSearchPolicy.FORBIDDEN,
        )

        self.assertEqual(
            fields["plugins"],
            [{"id": "web", "enabled": False}],
        )
        self.assertEqual(fields["tool_choice"], "none")
        self.assertNotIn("tools", fields)
        self.assertNotIn(":online", policy["model"])
        with self.assertRaises(OpenRouterError):
            web_request_policy(
                model="openai/gpt-5.4:online",
                policy=WebSearchPolicy.FORBIDDEN,
            )

    def test_legacy_online_answer_fails_closed_without_aborting_rebuild(
        self,
    ) -> None:
        answer = ModelAnswer(
            run_id="legacy-run",
            prompt_id=1,
            provider_key="openai",
            model="openai/gpt-5.4:online",
            mode="web",
            status="completed",
            response_text="Сохранённый ответ.",
            usage_json={
                "_aiv_panel_contract": {},
                "_aiv_request_policy": {},
                "_aiv_web_attestation": {},
                "_aiv_response_annotations": [],
            },
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text="Что выбрать?",
        )

        self.assertFalse(verified)
        self.assertEqual(reason, "unsupported_legacy_web_contract")

    def test_legacy_web_requires_positive_retrieval_evidence(self) -> None:
        answer = ModelAnswer(
            run_id="legacy-run",
            prompt_id=1,
            provider_key="gemini",
            model="google/gemini-3.6-flash",
            mode="web",
            status="completed",
            response_text="Сохранённый ответ.",
            citations_json=[{"url": "https://source.example/article"}],
            usage_json={
                "server_tool_use_details": {"web_search_requests": 2},
            },
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text="Что выбрать?",
            legacy_allowed=True,
        )

        self.assertTrue(verified)
        self.assertEqual(reason, "legacy_web_retrieval_confirmed")

        answer.citations_json = [{"url": "not-a-url"}]
        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text="Что выбрать?",
            legacy_allowed=True,
        )
        self.assertFalse(verified)
        self.assertEqual(reason, "legacy_web_evidence_missing")

    def test_legacy_memory_is_context_only_without_enforced_request(self) -> None:
        answer = ModelAnswer(
            run_id="legacy-run",
            prompt_id=1,
            provider_key="openai",
            model="openai/gpt-chat-latest",
            mode="memory",
            status="completed",
            response_text="Сохранённый ответ.",
            citations_json=[],
            usage_json={},
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text="Что известно о бренде?",
            legacy_allowed=True,
        )

        self.assertFalse(verified)
        self.assertEqual(reason, "legacy_memory_request_not_enforced")

        access = _panel_metric_access(
            answer,
            transport_attested=verified,
            attestation_reason=reason,
            legacy_memory_observation_allowed=True,
        )
        self.assertTrue(access["metric_eligible"])
        self.assertFalse(access["context_eligible"])
        self.assertEqual(
            access["metric_evidence_state"],
            "legacy_observational",
        )

        selected, _manifest = _select_final_answer_context(
            [
                {
                    "answer_id": 1,
                    "prompt_id": 1,
                    "provider_key": "openai",
                    "mode": "memory",
                    "metric_eligible": True,
                    "context_eligible": False,
                    "metric_evidence_state": "legacy_observational",
                    "scenario_role": "brand_diagnostic",
                    "intent_class": "I",
                    "scenario_sequence": 1,
                    "scenario": "Что известно о бренде?",
                    "answer_text": "Секретный сохранённый ответ.",
                    "citations": [],
                    "annotation": {
                        "valid": True,
                        "target_mentioned": True,
                        "target_role": "mentioned",
                        "sentiment": "positive",
                    },
                    "panel_evidence": {
                        "reason": LEGACY_MEMORY_OBSERVATION_REASON,
                        "sha256": "panel-1",
                    },
                    "provenance": {
                        "raw_answer_sha256": "raw-1",
                        "annotation_sha256": "annotation-1",
                    },
                }
            ],
            corpus_manifest={"digest": "full", "critic_rows_sha256": "rows"},
            max_answers=1,
        )
        self.assertEqual(selected[0]["context_access"], "metadata_only")
        self.assertIsNone(selected[0]["verified_mode"])
        self.assertNotIn("answer_text", selected[0])
        self.assertNotIn("annotation", selected[0])
        self.assertNotIn("citations", selected[0])

    def test_legacy_memory_observation_fails_closed_on_any_trace(self) -> None:
        cases = (
            ({"citations_json": [{"url": "not-a-url"}]}, "malformed citation"),
            (
                {"usage_json": {"server_tool_use": {"web_search_requests": False}}},
                "boolean search counter",
            ),
            (
                {"usage_json": {"server_tool_use": {"web_fetch_requests": "0"}}},
                "string fetch counter",
            ),
            (
                {"usage_json": {"tool_calls": [{"name": "search"}]}},
                "tool call",
            ),
            (
                {"usage_json": {"_aiv_router_metadata": {"web": True}}},
                "router signal",
            ),
            ({"model": "openai/gpt-chat-latest:online"}, "online model"),
            ({"model": "openai/another-model"}, "wrong model"),
        )
        for overrides, label in cases:
            with self.subTest(label=label):
                fields = {
                    "run_id": "legacy-run",
                    "prompt_id": 1,
                    "provider_key": "openai",
                    "model": "openai/gpt-chat-latest",
                    "mode": "memory",
                    "status": "completed",
                    "response_text": "Сохранённый ответ.",
                    "citations_json": [],
                    "usage_json": {"total_tokens": 12},
                }
                fields.update(overrides)
                answer = ModelAnswer(**fields)
                verified, reason = _panel_answer_attestation(
                    answer,
                    prompt_text="Что известно о бренде?",
                    legacy_allowed=True,
                )
                access = _panel_metric_access(
                    answer,
                    transport_attested=verified,
                    attestation_reason=reason,
                    legacy_memory_observation_allowed=True,
                )
                self.assertFalse(access["metric_eligible"])
                self.assertFalse(access["context_eligible"])

    def test_strict_memory_remains_metric_and_context_eligible(self) -> None:
        prompt_text = "Что известно о бренде?"
        usage, citations = _attested_panel_usage(
            prompt_text=prompt_text,
            mode="memory",
            provider_key="openai",
            model="openai/gpt-5.4",
        )
        answer = ModelAnswer(
            run_id="strict-run",
            prompt_id=1,
            provider_key="openai",
            model="openai/gpt-5.4",
            mode="memory",
            status="completed",
            response_text="Сохранённый ответ.",
            citations_json=citations,
            usage_json=usage,
        )
        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
        )
        access = _panel_metric_access(
            answer,
            transport_attested=verified,
            attestation_reason=reason,
            legacy_memory_observation_allowed=False,
        )
        self.assertTrue(verified)
        self.assertEqual(reason, "verified")
        self.assertTrue(access["metric_eligible"])
        self.assertTrue(access["context_eligible"])
        self.assertEqual(access["metric_evidence_state"], "strict_verified")
        self.assertTrue(
            LEGACY_PANEL_EVIDENCE_VERSION.endswith("evidence-v2")
        )

    def test_panel_v1_hash_is_verified_without_mutating_usage(self) -> None:
        prompt_text = "Какие сервисы выбрать?"
        request_sha256 = _legacy_panel_request_sha256(
            prompt_text=prompt_text,
            mode="web",
            provider_key="openai",
            model="openai/gpt-chat-latest",
        )
        usage = {
            "_aiv_panel_contract": {
                "version": LEGACY_PANEL_CONTRACT_VERSION,
                "request_sha256": request_sha256,
            },
            "server_tool_use_details": {"web_search_requests": 1},
        }
        answer = ModelAnswer(
            run_id="legacy-run",
            prompt_id=1,
            provider_key="openai",
            model="openai/gpt-chat-latest",
            mode="web",
            status="completed",
            response_text="Сохранённый ответ.",
            citations_json=[{"url": "https://source.example"}],
            usage_json=usage,
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text=prompt_text,
            legacy_allowed=True,
        )

        self.assertTrue(verified)
        self.assertEqual(reason, "legacy_web_retrieval_confirmed")
        self.assertEqual(answer.usage_json, usage)

    def test_partial_current_contract_never_downgrades_to_legacy(self) -> None:
        answer = ModelAnswer(
            run_id="run-id",
            prompt_id=1,
            provider_key="openai",
            model="openai/gpt-chat-latest",
            mode="web",
            status="completed",
            response_text="Ответ.",
            citations_json=[{"url": "https://source.example"}],
            usage_json={
                "_aiv_panel_contract": {
                    "version": PANEL_CONTRACT_VERSION,
                },
                "server_tool_use_details": {"web_search_requests": 1},
            },
        )

        verified, reason = _panel_answer_attestation(
            answer,
            prompt_text="Что выбрать?",
            legacy_allowed=True,
        )

        self.assertFalse(verified)
        self.assertEqual(reason, "missing_request_policy")

    def test_memory_response_with_retrieval_telemetry_is_rejected(self) -> None:
        attestation = attest_web_response(
            requested_model="openai/gpt-5.4",
            response_model="openai/gpt-5.4",
            policy=WebSearchPolicy.FORBIDDEN,
            usage={"server_tool_use": {"web_search_requests": 1}},
            annotations=[],
            citations=[],
            router_metadata={
                "pipeline": [
                    {
                        "type": "plugin",
                        "name": "web-search",
                        "data": {"results": 3},
                    }
                ]
            },
        )

        self.assertFalse(attestation["metric_eligible"])
        self.assertIn(
            "web_search_used_while_forbidden",
            attestation["violations"],
        )
        self.assertIn(
            "router_retrieval_stage_while_forbidden",
            attestation["violations"],
        )

    def test_perplexity_native_search_requires_native_citation_evidence(
        self,
    ) -> None:
        confirmed = attest_web_response(
            requested_model="perplexity/sonar-pro-search",
            response_model="perplexity/sonar-pro-search",
            policy=WebSearchPolicy.NATIVE_REQUIRED,
            usage={},
            annotations=[
                {
                    "type": "url_citation",
                    "url_citation": {"url": "https://source.example"},
                }
            ],
            citations=[{"url": "https://source.example"}],
            router_metadata={},
        )
        unconfirmed = attest_web_response(
            requested_model="perplexity/sonar-pro-search",
            response_model="perplexity/sonar-pro-search",
            policy=WebSearchPolicy.NATIVE_REQUIRED,
            usage={},
            annotations=[],
            citations=[],
            router_metadata={},
        )

        self.assertTrue(confirmed["metric_eligible"])
        self.assertFalse(unconfirmed["metric_eligible"])
        self.assertIn(
            "perplexity_native_citation_not_confirmed",
            unconfirmed["violations"],
        )

    def test_required_request_has_one_forced_server_web_tool(self) -> None:
        fields, _policy = web_request_policy(
            model="openai/gpt-5.4",
            policy=WebSearchPolicy.REQUIRED,
        )

        self.assertEqual(fields["tool_choice"], "required")
        self.assertEqual(len(fields["tools"]), 1)
        self.assertEqual(
            fields["tools"][0]["type"],
            "openrouter:web_search",
        )

    def test_panel_request_hash_depends_on_policy_and_attestation_contract(
        self,
    ) -> None:
        web_hash = _panel_request_sha256(
            prompt_text="Какие решения выбрать?",
            mode="web",
            provider_key="openai",
            model="test/model",
        )
        memory_hash = _panel_request_sha256(
            prompt_text="Какие решения выбрать?",
            mode="memory",
            provider_key="openai",
            model="test/model",
        )
        with patch(
            "app.services.analyzer.WEB_ATTESTATION_VERSION",
            "changed-attestation-contract",
        ):
            changed_attestation_hash = _panel_request_sha256(
                prompt_text="Какие решения выбрать?",
                mode="web",
                provider_key="openai",
                model="test/model",
            )

        self.assertNotEqual(web_hash, memory_hash)
        self.assertNotEqual(web_hash, changed_attestation_hash)

    def test_strict_schemas_avoid_unsupported_array_cardinality(self) -> None:
        self.assertNotIn("minItems", str(PROMPT_SET_SCHEMA))
        self.assertNotIn("maxItems", str(PROMPT_SET_SCHEMA))
        self.assertNotIn("minItems", str(FINAL_REPORT_SCHEMA))
        self.assertNotIn("maxItems", str(FINAL_REPORT_SCHEMA))
        self.assertNotIn("minItems", str(ILLUSTRATION_CONCEPTS_SCHEMA))
        self.assertNotIn("maxItems", str(ILLUSTRATION_CONCEPTS_SCHEMA))
        self.assertNotIn("minItems", str(ILLUSTRATION_QA_SCHEMA))
        self.assertNotIn("maxItems", str(ILLUSTRATION_QA_SCHEMA))

    def test_analytics_and_visual_concepts_have_separate_validation(self) -> None:
        valid = {"sections": [{}], "actions": [{}]}
        self.assertEqual(_validate_final_report(valid), [])
        self.assertTrue(
            _validate_final_report({**valid, "illustrations": [{}, {}, {}]})
        )

        concepts = {
            "illustrations": [
                {
                    "role": role,
                    "title": f"Схема {index}",
                    "caption": "Вывод по рассчитанным данным",
                    "alt_text": "Описание схемы",
                    "core_claim": "Подтверждённый вывод",
                    "evidence_paths": [
                        {
                            "technical_access": "/technical/score",
                            "competitive_visibility": "/discovery/portfolio/web/score",
                            "web_memory_gap": "/brand_knowledge/memory/specific_rate",
                        }[role]
                    ],
                    "context_for_image": "Specific market and product context.",
                    "creative_brief": {
                        "visual_thesis": f"Distinct visual idea {index}",
                        "scene": "An authored editorial scene.",
                        "composition": "A bold asymmetric composition.",
                        "materials_and_light": "Tactile materials and dramatic light.",
                        "emotional_tone": "Confident and precise.",
                        "target_treatment": "One decisive accent.",
                        "diversity_move": f"Unique spatial move {index}.",
                    },
                }
                for index, role in enumerate(
                    (
                        "technical_access",
                        "competitive_visibility",
                        "web_memory_gap",
                    ),
                    start=1,
                )
            ]
        }
        self.assertEqual(_validate_illustration_concepts(concepts), [])
        self.assertTrue(
            _validate_illustration_concepts(
                {"illustrations": concepts["illustrations"][:2]}
            )
        )
        expressive = {
            "illustrations": [
                {
                    **item,
                    "creative_brief": {
                        **item["creative_brief"],
                        "scene": "Isometric exploded editorial collage.",
                    },
                }
                for item in concepts["illustrations"]
            ]
        }
        self.assertEqual(_validate_illustration_concepts(expressive), [])

    def test_illustration_concepts_accept_valid_section_root_pointer(self) -> None:
        concepts = {
            "illustrations": [
                {
                    "role": role,
                    "title": f"Схема {index}",
                    "caption": "Вывод по рассчитанным данным",
                    "alt_text": "Описание схемы",
                    "core_claim": "Подтверждённый вывод",
                    "evidence_paths": [path],
                    "context_for_image": "Specific market and product context.",
                    "creative_brief": {
                        "visual_thesis": f"Distinct root idea {index}",
                        "scene": "An authored editorial scene.",
                        "composition": "A bold asymmetric composition.",
                        "materials_and_light": "Tactile materials and dramatic light.",
                        "emotional_tone": "Confident and precise.",
                        "target_treatment": "One decisive accent.",
                        "diversity_move": f"Unique root move {index}.",
                    },
                }
                for index, (role, path) in enumerate(
                    (
                        ("technical_access", "/technical"),
                        ("competitive_visibility", "/competitors"),
                        ("web_memory_gap", "/brand_knowledge"),
                    ),
                    start=1,
                )
            ]
        }
        self.assertEqual(
            _validate_illustration_concepts(
                concepts,
                {
                    "technical": {"score": 95},
                    "competitors": [],
                    "brand_knowledge": {"memory": {}},
                },
            ),
            [],
        )

        hallucinated = copy.deepcopy(concepts)
        hallucinated["illustrations"][0]["caption"] = (
            "Техническая доступность составляет 77%."
        )
        numeric_errors = _validate_illustration_concepts(
            hallucinated,
            {
                "technical": {"score": 95},
                "competitors": [],
                "brand_knowledge": {"memory": {}},
            },
        )
        self.assertTrue(any("77" in item for item in numeric_errors))

        hallucinated_title = copy.deepcopy(concepts)
        hallucinated_title["illustrations"][0]["title"] = "Доступность 99%"
        title_errors = _validate_illustration_concepts(
            hallucinated_title,
            {
                "technical": {"score": 95},
                "competitors": [],
                "brand_knowledge": {"memory": {}},
            },
        )
        self.assertTrue(any("99" in item for item in title_errors))

        unpaired_report = {
            "technical": {"score": 95},
            "competitors": [],
            "discovery": {
                "paired_web_lift": {"n_pairs": 0},
                "parent": {"memory": {"data_state": "unavailable"}},
                "portfolio": {"memory": {"data_state": "unavailable"}},
            },
            "brand_knowledge": {
                "memory": {"data_state": "unavailable"}
            },
        }
        normalized = _normalize_unpaired_memory_illustration(
            concepts,
            unpaired_report,
        )
        memory_concept = normalized["illustrations"][2]
        self.assertNotRegex(memory_concept["caption"], r"\d")
        self.assertIn("оценить нельзя", memory_concept["caption"])
        self.assertEqual(
            _validate_illustration_concepts(normalized, unpaired_report),
            [],
        )

    def test_final_report_accepts_as_many_sections_as_opus_needs(self) -> None:
        expansive = {
            "sections": [{} for _ in range(12)],
            "actions": [{} for _ in range(14)],
        }
        self.assertEqual(_validate_final_report(expansive), [])

    def test_image_cache_is_bound_to_the_exact_visual_concept(self) -> None:
        illustration = ReportIllustration(
            run_id="run-id",
            sequence=1,
            title="Схема",
            caption="Подпись",
            alt_text="Описание",
            generation_prompt="concept-v2",
            model=settings.OPENROUTER_IMAGE_MODEL,
            file_url="/static/generated/run-id/01.png",
            usage_json={
                "generation_version": ILLUSTRATION_GENERATION_VERSION,
            },
        )
        self.assertTrue(
            _illustration_cache_matches(illustration, "concept-v2")
        )
        self.assertFalse(
            _illustration_cache_matches(illustration, "concept-v3")
        )

    def test_competitive_image_prompt_never_exposes_brand_as_text(self) -> None:
        prompt = _illustration_prompt(
            {
                "core_claim": "RW+ is visible in the evaluated field.",
                "context_for_image": "RW+ works in a complex market.",
                "creative_brief": {
                    "visual_thesis": "Show RW+ through one decisive accent.",
                },
                "fact_contract": {
                    "brand": {
                        "name": "RW+",
                        "category": "RW+ product analytics",
                        "products": ["RW+ visibility audit"],
                        "positioning": "RW+ measures AI visibility",
                    }
                },
            },
            brand_name="RW+",
            sequence=2,
        )
        self.assertNotIn("RW+", prompt)
        self.assertIn("the evaluated brand", prompt)
        self.assertIn("art direction is intentionally open", prompt)
        self.assertIn("depth, texture, light", prompt)

    def test_short_brand_name_does_not_corrupt_words(self) -> None:
        prompt = _illustration_prompt(
            {
                "core_claim": "AI is visible in retail analytics.",
                "context_for_image": "A retail media planning room.",
                "creative_brief": {
                    "visual_thesis": "Show retail campaign planning.",
                },
                "fact_contract": {
                    "brand": {
                        "name": "AI",
                        "category": "retail media",
                        "products": ["retail analytics"],
                    }
                },
            },
            brand_name="AI",
            sequence=2,
        )
        self.assertNotIn("AI is visible", prompt)
        self.assertIn("retail analytics", prompt)
        self.assertNotIn("retthe analyzed clientl", prompt)

    def test_quality_review_requires_context_specificity_score(self) -> None:
        self.assertIn(
            "Проверка качества не оценила контекстную специфичность.",
            _illustration_review_errors(
                {
                    "usable": True,
                    "facts_grounded": True,
                    "claim_readable": True,
                    "unsupported_assertions": [],
                    "visible_text_problems": [],
                    "hard_blockers": [],
                }
            ),
        )

    def test_generation_concept_never_rewrites_report_facts(
        self,
    ) -> None:
        source = {
            "role": "competitive_visibility",
            "core_claim": "Show the measured competitive position.",
        }
        normalized = _illustration_generation_concept(
            source,
            sequence=2,
            fact_context={
                "competitors": [{"name": "Alternative", "mention_share": 18.0}],
                "discovery": {"portfolio": {"web": {"score": 25.0}}},
            },
        )
        self.assertEqual(
            normalized["core_claim"],
            "Show the measured competitive position.",
        )
        self.assertEqual(normalized["fact_contract"]["competitors"][0]["name"], "Alternative")
        self.assertNotIn("visual_node_count_policy", normalized)

    def test_llm_cache_is_bound_to_input_model_and_layer_version(self) -> None:
        artifact = RunArtifact(
            run_id="run-id",
            stage_key="report",
            artifact_key="final_report",
            status="completed",
            model=ANALYSIS_MODEL,
            prompt_version=FINAL_REPORT_VERSION,
            input_json={"technical_score": 70},
            output_json={"sections": [{}], "actions": [{}]},
        )
        self.assertTrue(
            _artifact_cache_matches(
                artifact,
                input_json={"technical_score": 70},
                model=ANALYSIS_MODEL,
                prompt_version=FINAL_REPORT_VERSION,
            )
        )
        self.assertFalse(
            _artifact_cache_matches(
                artifact,
                input_json={"technical_score": 75},
                model=ANALYSIS_MODEL,
                prompt_version=FINAL_REPORT_VERSION,
            )
        )
        self.assertFalse(
            _artifact_cache_matches(
                artifact,
                input_json={"technical_score": 70},
                model=PROCESSING_MODEL,
                prompt_version=FINAL_REPORT_VERSION,
            )
        )

    def test_unknown_rendering_is_not_treated_as_client_side_failure(self) -> None:
        assessment = _rendering_assessment(
            Counter({"server_rendered": 2, "unknown": 1})
        )
        self.assertEqual(assessment["ratio"], 1)
        self.assertEqual(assessment["evaluated_pages"], 2)
        self.assertEqual(assessment["unknown_pages"], 1)
        self.assertIn("определить не удалось", assessment["conclusion"])
        self.assertNotIn("JavaScript", assessment["conclusion"])

        with_client_shell = _rendering_assessment(
            Counter({"server_rendered": 2, "client_rendered_shell": 1})
        )
        self.assertAlmostEqual(with_client_shell["ratio"], 2 / 3)
        self.assertIn("JavaScript", with_client_shell["conclusion"])


class ProcessingModelTests(unittest.IsolatedAsyncioTestCase):
    def test_brand_diagnostic_prompts_must_name_the_known_brand(self) -> None:
        prompts = [
            {
                "prompt_key": f"u-{index}",
                "intent_class": intent,
                "role": "unbranded_discovery",
                "text": (
                    f"Какие сервисы стоит выбрать для задачи № {index}? "
                    "Назовите конкретные варианты."
                ),
                "rationale": f"Проверяет сценарий {intent}.",
                "choice_request": True,
            }
            for index, intent in enumerate(
                ("I", "E", "T", "NB", "NAV", "TR"),
                start=1,
            )
        ]
        prompts.extend(
            {
                "prompt_key": f"b-{index}",
                "intent_class": intent,
                "role": "brand_diagnostic",
                "text": f"Что известно о RW+ для задачи № {index}?",
                "rationale": "Проверяет знание бренда.",
                "choice_request": False,
            }
            for index, intent in enumerate(("I", "E", "TR"), start=1)
        )
        profile = {"brand_name": "RW+", "brand_aliases": ["Realweb Plus"]}

        self.assertEqual(_validate_prompt_set({"prompts": prompts}, profile), [])

        prompts[2]["text"] = (
            "Кого пригласить в тендер, чтобы получить сильное предложение?"
        )
        self.assertEqual(_validate_prompt_set({"prompts": prompts}, profile), [])

        prompts[-1]["text"] = "Что известно об этом поставщике?"
        errors = _validate_prompt_set({"prompts": prompts}, profile)
        self.assertTrue(
            any("не содержит" in error and "Брендовый" in error for error in errors)
        )

    def test_prompt_semantic_review_rejects_mislabeled_intent(self) -> None:
        prompts = [
            {
                "prompt_key": f"u-{intent.lower()}",
                "intent_class": intent,
                "role": "unbranded_discovery",
            }
            for intent in ("I", "E", "T", "NB", "NAV", "TR")
        ]
        checks = [
            {
                "prompt_key": item["prompt_key"],
                "declared_intent": item["intent_class"],
                "dominant_intent": item["intent_class"],
                "matches": True,
                "grounded_in_research": True,
                "supporting_evidence": ["https://research.example/market"],
                "unsupported_assumptions": [],
                "reason": "Соответствует.",
                "fix_instruction": "",
            }
            for item in prompts
        ]
        need_based = next(
            check for check in checks if check["declared_intent"] == "NB"
        )
        need_based["dominant_intent"] = "NAV"
        need_based["matches"] = False
        need_based["reason"] = "Запрос ищет площадку, а не решает задачу."
        need_based["fix_instruction"] = "Сформулируйте боль или ограничение."
        review = {
            "verdict": "revise",
            "summary": "Один класс подменён.",
            "checks": checks,
        }

        errors = _prompt_review_errors(review, {"prompts": prompts})

        self.assertEqual(len(errors), 1)
        self.assertIn("u-nb", errors[0])
        self.assertIn("классу NB", errors[0])
        self.assertIn("боль или ограничение", errors[0])

    def test_prompt_review_schema_checks_declared_and_dominant_intent(self) -> None:
        check_properties = (
            PROMPT_SET_REVIEW_SCHEMA["properties"]["checks"]["items"]["properties"]
        )
        self.assertEqual(
            check_properties["declared_intent"]["enum"],
            ["I", "E", "T", "NB", "NAV", "TR"],
        )
        self.assertEqual(
            check_properties["dominant_intent"]["enum"],
            ["I", "E", "T", "NB", "NAV", "TR"],
        )
        self.assertIn("grounded_in_research", check_properties)
        self.assertIn("supporting_evidence", check_properties)
        self.assertIn("unsupported_assumptions", check_properties)

    async def test_site_classification_stays_on_analysis_model(self) -> None:
        context = {
            "requested_site": {
                "domain": "example.com",
                "url": "https://example.com/",
            },
            "pages": [],
        }
        with patch(
            "app.services.analyzer._structured_artifact",
            new_callable=AsyncMock,
            return_value={"brand_name": "Example"},
        ) as structured:
            await _classify_site("run-id", context)

        structured.assert_awaited_once()
        self.assertEqual(structured.await_args.kwargs["model"], ANALYSIS_MODEL)
        self.assertEqual(structured.await_args.kwargs["reasoning_effort"], "high")
        self.assertEqual(structured.await_args.kwargs["user_payload"], context)

    async def test_site_profile_explicitly_models_market_and_customer_choice(self) -> None:
        required = set(SITE_PROFILE_SCHEMA["required"])
        self.assertTrue(
            {
                "topics",
                "market",
                "business_model",
                "customer_jobs",
                "decision_criteria",
            }.issubset(required)
        )

    async def test_market_research_precedes_prompts_and_requires_attested_web(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "site_type": "Сайт продукта",
            "category": "Аналитика",
            "topics": ["AI visibility"],
            "market": "B2B-аналитика",
            "business_model": "Подписка",
            "products": ["Платформа"],
            "audiences": ["Руководители маркетинга"],
            "customer_jobs": ["Сравнить поставщиков"],
            "decision_criteria": ["Точность"],
            "geography": ["Россия"],
            "language": "ru",
            "positioning": "Аналитическая платформа",
            "entity_scope": [],
            "evidence": ["Описание на главной странице"],
            "uncertainties": [],
            "confidence": "high",
        }
        site_context = {
            "requested_site": {
                "domain": "example.com",
                "url": "https://example.com/",
            },
            "pages": [
                {
                    "url": "https://example.com/",
                    "title": "Example",
                    "meta_description": "Аналитическая платформа",
                    "main_text": "Example помогает измерять видимость.",
                }
            ],
        }
        ready = _ready_market_research()
        parsed = {
            key: value
            for key, value in ready.items()
            if key not in {"sufficiency", "web_evidence"}
        }
        response = SimpleNamespace(
            parsed=parsed,
            text=json.dumps(parsed, ensure_ascii=False),
            usage={
                "_aiv_web_attestation": ready["web_evidence"]["attestation"],
                "_aiv_response_annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": citation,
                    }
                    for citation in ready["web_evidence"]["citations"]
                ],
            },
            web_attestation=ready["web_evidence"]["attestation"],
            citations=ready["web_evidence"]["citations"],
        )
        with (
            patch(
                "app.services.analyzer._cached_market_research",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            result = await _market_research(
                "run-id",
                profile,
                site_context,
            )

        self.assertEqual(result["status"], "ready")
        request = chat_mock.await_args.kwargs
        self.assertEqual(request["model"], ANALYSIS_MODEL)
        self.assertEqual(request["web_policy"], WebSearchPolicy.REQUIRED)
        request_payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(request_payload["requested_site"], site_context["requested_site"])
        self.assertEqual(request_payload["site_profile"], profile)
        self.assertEqual(request_payload["site_evidence"], site_context["pages"])
        final_write = save_artifact.await_args.kwargs
        self.assertEqual(final_write["artifact_key"], "market_research")
        self.assertEqual(final_write["prompt_version"], MARKET_RESEARCH_VERSION)
        self.assertEqual(final_write["status"], "completed")
        self.assertEqual(
            final_write["output_json"]["site_confirmed"]["primary_brand"],
            profile["brand_name"],
        )

    def test_market_research_blocks_empty_jobs_and_criteria(self) -> None:
        research = _ready_market_research()
        external = research["external_market_research"]
        external["customer_jobs"] = []
        external["decision_criteria"] = []
        gate = _market_research_sufficiency(
            research,
            {"brand_name": "Example"},
            requested_site={"domain": "example.com"},
            site_evidence=[{"url": "https://example.com/"}],
            web_attestation=research["web_evidence"]["attestation"],
            citation_urls=[
                item["url"] for item in research["web_evidence"]["citations"]
            ],
        )

        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(
            any("задачи аудитории" in issue for issue in gate["blocking_issues"])
        )
        self.assertTrue(
            any("критерии выбора" in issue for issue in gate["blocking_issues"])
        )

    async def test_unattested_market_research_never_reaches_prompt_model(
        self,
    ) -> None:
        research = _ready_market_research()
        research["web_evidence"]["attestation"]["state"] = "violated"
        research["web_evidence"]["attestation"]["metric_eligible"] = False
        with (
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
            self.assertRaises(MarketResearchGateError),
        ):
            await _generate_prompt_set(
                "run-id",
                {"brand_name": "Example", "brand_aliases": []},
                market_research=research,
            )

        chat_mock.assert_not_awaited()

    async def test_invalid_site_profile_never_starts_market_research(
        self,
    ) -> None:
        with (
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            self.assertRaises(MarketResearchGateError),
        ):
            await _market_research(
                "run-id",
                {"brand_name": "", "evidence": []},
                {
                    "requested_site": {"domain": "example.com"},
                    "pages": [],
                },
            )

        chat_mock.assert_not_awaited()
        self.assertEqual(save_artifact.await_args.kwargs["status"], "failed")
        self.assertEqual(
            save_artifact.await_args.kwargs["output_json"]["status"],
            "blocked",
        )

    async def test_prompt_generation_receives_site_and_profile_unchanged(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "site_type": "Сайт продукта",
            "category": "Аналитика",
            "topics": ["Исследования"],
            "market": "B2B SaaS",
            "business_model": "Подписка",
            "products": ["Платформа"],
            "audiences": ["Маркетологи"],
            "customer_jobs": ["Сравнить видимость бренда"],
            "decision_criteria": ["Точность", "Скорость"],
            "geography": ["Россия"],
            "language": "ru",
            "positioning": "Аналитическая платформа",
            "evidence": ["Описание на главной"],
            "uncertainties": [],
            "confidence": "high",
        }
        requested_site = {
            "domain": "example.com",
            "url": "https://example.com/",
        }
        market_research = _ready_market_research()
        prompts = []
        for index, intent in enumerate(("I", "E", "T", "NB", "NAV", "TR"), start=1):
            prompts.append(
                {
                    "prompt_key": f"u-{index}",
                    "intent_class": intent,
                    "role": "unbranded_discovery",
                    "text": (
                        f"Какие сервисы стоит выбрать для задачи № {index}? "
                        "Назовите конкретные варианты."
                    ),
                    "rationale": "Проверяет пользовательскую задачу.",
                    "choice_request": True,
                }
            )
        for index, intent in enumerate(("I", "E", "TR"), start=1):
            prompts.append(
                {
                    "prompt_key": f"b-{index}",
                    "intent_class": intent,
                    "role": "brand_diagnostic",
                    "text": f"Что известно про Example: вопрос № {index}?",
                    "rationale": "Проверяет знание бренда.",
                    "choice_request": False,
                }
            )
        response = SimpleNamespace(
            parsed={"prompts": prompts},
            text=json.dumps({"prompts": prompts}, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ) as semantic_review,
        ):
            result = await _generate_prompt_set(
                "run-id",
                profile,
                requested_site,
                market_research=market_research,
            )

        self.assertEqual(result, {"prompts": prompts})
        request = chat_mock.await_args.kwargs
        self.assertEqual(request["model"], ANALYSIS_MODEL)
        self.assertEqual(request["reasoning_effort"], "high")
        system = request["messages"][0]["content"]
        self.assertIn("NB — Need Based", system)
        self.assertIn("NAV — Navigation", system)
        self.assertIn("TR — Trend-Driven", system)
        self.assertIn("NB нельзя подменять навигацией", system)
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["requested_site"], requested_site)
        self.assertEqual(payload["site_profile"], profile)
        self.assertEqual(payload["market_research"], market_research)
        self.assertRegex(payload["market_research_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(save_artifact.await_args.kwargs["input_json"], payload)
        semantic_review.assert_awaited_once_with(
            "run-id",
            profile,
            {"prompts": prompts},
            market_research,
        )
        self.assertEqual(
            save_artifact.await_args.kwargs["prompt_version"],
            PROMPT_SET_VERSION,
        )

    async def test_prompt_generation_retries_after_semantic_critic(self) -> None:
        profile = {"brand_name": "Example", "brand_aliases": []}
        market_research = _ready_market_research()

        def prompt_set(suffix: str) -> dict[str, object]:
            prompts: list[dict[str, object]] = []
            for index, intent in enumerate(
                ("I", "E", "T", "NB", "NAV", "TR"),
                start=1,
            ):
                prompts.append(
                    {
                        "prompt_key": f"u-{intent.lower()}",
                        "intent_class": intent,
                        "role": "unbranded_discovery",
                        "text": (
                            f"Какие варианты подходят для сценария {index} "
                            f"{suffix}? Назовите конкретные решения."
                        ),
                        "rationale": f"Проверяет класс {intent}.",
                        "choice_request": True,
                    }
                )
            for index, intent in enumerate(("I", "E", "TR"), start=1):
                prompts.append(
                    {
                        "prompt_key": f"b-{index}",
                        "intent_class": intent,
                        "role": "brand_diagnostic",
                        "text": f"Что известно об Example для вопроса {index} {suffix}?",
                        "rationale": "Проверяет знание бренда.",
                        "choice_request": False,
                    }
                )
            return {"prompts": prompts}

        first = prompt_set("первая версия")
        corrected = prompt_set("исправленная версия")
        chat_results = [
            SimpleNamespace(
                parsed=value,
                text=json.dumps(value, ensure_ascii=False),
                usage={"total_tokens": 1},
            )
            for value in (first, corrected)
        ]
        critic_error = (
            "u-nb должен соответствовать классу NB: "
            "сформулируйте задачу, боль или ограничение"
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=chat_results,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                side_effect=[[critic_error], []],
            ) as semantic_review,
        ):
            result = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=market_research,
            )

        self.assertEqual(result, corrected)
        self.assertEqual(chat_mock.await_count, 2)
        self.assertEqual(semantic_review.await_count, 2)
        retry_payload = json.loads(
            chat_mock.await_args_list[1].kwargs["messages"][1]["content"]
        )
        self.assertEqual(
            retry_payload["validation_errors_to_fix"],
            [critic_error],
        )

    async def test_market_research_digest_invalidates_cached_prompt_set(
        self,
    ) -> None:
        profile = {"brand_name": "Example", "brand_aliases": []}

        def prompt_set(suffix: str) -> dict[str, object]:
            prompts: list[dict[str, object]] = [
                {
                    "prompt_key": f"u-{intent.lower()}",
                    "intent_class": intent,
                    "role": "unbranded_discovery",
                    "text": (
                        f"Какие решения подходят для задачи {intent} {suffix}? "
                        "Назовите конкретные варианты."
                    ),
                    "rationale": f"Проверяет {intent}.",
                    "choice_request": True,
                }
                for intent in ("I", "E", "T", "NB", "NAV", "TR")
            ]
            prompts.extend(
                {
                    "prompt_key": f"b-{index}",
                    "intent_class": intent,
                    "role": "brand_diagnostic",
                    "text": f"Что известно об Example для задачи {index} {suffix}?",
                    "rationale": "Проверяет знание бренда.",
                    "choice_request": False,
                }
                for index, intent in enumerate(("I", "E", "TR"), start=1)
            )
            return {"prompts": prompts}

        cached = prompt_set("из кэша")
        regenerated = prompt_set("после изменения")
        research_before = _ready_market_research(
            first_job="Сравнить поставщиков аналитики"
        )
        research_after = json.loads(
            json.dumps(research_before, ensure_ascii=False)
        )
        research_after["external_market_research"]["customer_jobs"] = [
            "Проверить полноту рыночных данных"
        ]
        seen_inputs: list[dict[str, object]] = []

        async def fake_artifact_output(
            _run_id: str,
            _artifact_key: str,
            **kwargs: object,
        ) -> dict[str, object] | None:
            input_json = kwargs["input_json"]
            seen_inputs.append(input_json)
            job = input_json["market_research"]["external_market_research"][
                "customer_jobs"
            ][0]
            return cached if job == "Сравнить поставщиков аналитики" else None

        response = SimpleNamespace(
            parsed=regenerated,
            text=json.dumps(regenerated, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new=fake_artifact_output,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._review_prompt_set_semantics",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            first = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research_before,
            )
            second = await _generate_prompt_set(
                "run-id",
                profile,
                market_research=research_after,
            )

        self.assertEqual(first, cached)
        self.assertEqual(second, regenerated)
        self.assertEqual(chat_mock.await_count, 1)
        self.assertNotEqual(
            seen_inputs[0]["market_research_digest"],
            seen_inputs[1]["market_research_digest"],
        )

    async def test_prompt_reviewer_sees_sources_and_rejects_assumption(
        self,
    ) -> None:
        research = _ready_market_research()
        prompts = [
            {
                "prompt_key": f"u-{intent.lower()}",
                "intent_class": intent,
                "role": "unbranded_discovery",
                "text": f"Какие варианты подходят для задачи {intent}?",
            }
            for intent in ("I", "E", "T", "NB", "NAV", "TR")
        ]
        checks = [
            {
                "prompt_key": item["prompt_key"],
                "declared_intent": item["intent_class"],
                "dominant_intent": item["intent_class"],
                "matches": True,
                "grounded_in_research": True,
                "supporting_evidence": ["https://research.example/market"],
                "unsupported_assumptions": [],
                "reason": "Соответствует.",
                "fix_instruction": "",
            }
            for item in prompts
        ]
        checks[0]["grounded_in_research"] = False
        checks[0]["supporting_evidence"] = []
        checks[0]["unsupported_assumptions"] = [
            "Исследование не подтверждает эту задачу аудитории."
        ]
        checks[0]["fix_instruction"] = "Используйте подтверждённую customer job."
        review = {
            "verdict": "revise",
            "summary": "Есть неподтверждённое допущение.",
            "checks": checks,
        }
        with patch(
            "app.services.analyzer._structured_artifact",
            new_callable=AsyncMock,
            return_value=review,
        ) as structured:
            errors = await _review_prompt_set_semantics(
                "run-id",
                {"brand_name": "Example"},
                {"prompts": prompts},
                research,
            )

        self.assertTrue(any("неподтверждённое" in error for error in errors))
        payload = structured.await_args.kwargs["user_payload"]
        self.assertEqual(payload["market_research"], research)
        self.assertRegex(payload["market_research_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            payload["market_research"]["sources"][0]["confidence"],
            "high",
        )
        self.assertEqual(
            payload["market_research"]["web_evidence"]["citations"][0]["url"],
            "https://research.example/market",
        )

    async def test_entity_catalog_uses_processing_model(self) -> None:
        with patch(
            "app.services.analyzer._processing_artifact",
            new_callable=AsyncMock,
            return_value={"entities": []},
        ) as processing:
            await _entity_catalog(
                "run-id",
                {"brand_name": "Example", "brand_aliases": [], "products": []},
                [],
            )

        processing.assert_awaited_once()

    async def test_entity_catalog_extracts_in_small_batches_before_merge(self) -> None:
        answers = [
            {
                "answer_id": index,
                "answer": f"Named entity {index}",
            }
            for index in range(17)
        ]
        active = 0
        max_active = 0
        release = asyncio.Event()
        merge_payload: dict[str, object] = {}
        merge_version = ""
        chunk_artifact_keys: list[str] = []
        chunk_versions: list[str] = []

        async def fake_processing(
            run_id: str,
            **kwargs: object,
        ) -> dict[str, object]:
            nonlocal active, max_active, merge_version
            self.assertEqual(run_id, "run-id")
            if kwargs["artifact_key"] == "entity_catalog":
                merge_payload.update(
                    kwargs["user_payload"]  # type: ignore[arg-type]
                )
                merge_version = str(kwargs["prompt_version"])
                return {
                    "target_aliases": ["merged"],
                    "entities": [],
                    "uncertainties": [],
                }

            chunk = kwargs["user_payload"]["answers"]  # type: ignore[index]
            self.assertLessEqual(len(chunk), 8)
            chunk_artifact_keys.append(str(kwargs["artifact_key"]))
            chunk_versions.append(str(kwargs["prompt_version"]))
            first_answer_id = int(chunk[0]["answer_id"])
            active += 1
            max_active = max(max_active, active)
            if active == PROCESSING_BATCH_CONCURRENCY:
                release.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=1)
                # Finish later chunks first to prove that merge order does not
                # depend on request completion order.
                await asyncio.sleep((17 - first_answer_id) / 1000)
                return {
                    "target_aliases": [str(first_answer_id)],
                    "entities": [],
                    "uncertainties": [],
                }
            finally:
                active -= 1

        with patch(
            "app.services.analyzer._processing_artifact",
            new=fake_processing,
        ):
            result = await _entity_catalog(
                "run-id",
                {
                    "brand_name": "Example",
                    "brand_aliases": [],
                    "products": [],
                    "topics": ["перформанс-маркетинг"],
                },
                answers,
            )

        self.assertEqual(result["target_aliases"], ["merged"])
        self.assertEqual(max_active, PROCESSING_BATCH_CONCURRENCY)
        self.assertEqual(len(set(chunk_artifact_keys)), 3)
        self.assertEqual(
            set(chunk_versions),
            {ENTITY_CATALOG_CHUNK_VERSION},
        )
        self.assertEqual(merge_version, ENTITY_CATALOG_VERSION)
        self.assertEqual(
            merge_payload["target"]["topics"],  # type: ignore[index]
            ["перформанс-маркетинг"],
        )
        self.assertEqual(
            [
                catalog["target_aliases"][0]
                for catalog in merge_payload["chunk_catalogs"]  # type: ignore[index]
            ],
            ["0", "8", "16"],
        )

    async def test_annotation_requests_are_parallel_but_writes_stay_ordered(
        self,
    ) -> None:
        profile = {"brand_name": "Example", "brand_aliases": []}
        catalog = {"target_aliases": ["Example"], "entities": []}
        annotation_context_hash = _annotation_context_sha256(profile, catalog)
        pending = [
            {
                "answer_id": answer_id,
                "mode": "web",
                "system": "Test",
                "answer_model": "test/model",
                "answer_sha256": f"sha-{answer_id}",
                "scenario": f"Scenario {answer_id}",
                "scenario_role": "unbranded_discovery",
                "intent_class": "I",
                "answer": f"Answer {answer_id}",
            }
            for answer_id in range(1, 26)
        ]
        active_requests = 0
        max_active_requests = 0
        release = asyncio.Event()

        async def fake_processing(
            run_id: str,
            **kwargs: object,
        ) -> dict[str, object]:
            nonlocal active_requests, max_active_requests
            self.assertEqual(run_id, "run-id")
            batch = kwargs["user_payload"]["answers"]  # type: ignore[index]
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            if active_requests == PROCESSING_BATCH_CONCURRENCY:
                release.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=1)
                return {
                    "answers": [
                        {
                            "answer_id": item["answer_id"],
                            "valid": True,
                            "target_mentioned": False,
                            "target_position": None,
                            "target_role": "absent",
                            "sentiment": "unknown",
                            "entity_mentions": [],
                            "brand_answer": {
                                "directness": "not_applicable",
                                "specificity": "not_applicable",
                                "supported_facets": [],
                                "contradictions": [],
                            },
                            "evidence": [],
                            "uncertainties": [],
                        }
                        for item in batch
                    ]
                }
            finally:
                active_requests -= 1

        save_order: list[int] = []
        active_saves = 0
        max_active_saves = 0

        async def fake_save(
            run_id: str,
            annotations: list[dict[str, object]],
            allowed_ids: set[int],
        ) -> int:
            nonlocal active_saves, max_active_saves
            self.assertEqual(run_id, "run-id")
            active_saves += 1
            max_active_saves = max(max_active_saves, active_saves)
            try:
                save_order.append(min(allowed_ids))
                await asyncio.sleep(0)
                return len(annotations)
            finally:
                active_saves -= 1

        completed_rows = [
            (
                SimpleNamespace(
                    response_text=item["answer"],
                    model=item["answer_model"],
                ),
                SimpleNamespace(
                    annotation_json={
                        "_annotation_version": ANNOTATION_VERSION,
                        "_answer_sha256": hashlib.sha256(
                            item["answer"].encode("utf-8")
                        ).hexdigest(),
                        "_answer_model": item["answer_model"],
                        "_annotation_input_sha256": annotation_context_hash,
                    }
                ),
            )
            for item in pending
        ]

        class FakeResult:
            def __init__(
                self,
                *,
                rows: list[tuple[int]] | None = None,
            ) -> None:
                self._rows = rows or []

            def all(self) -> list[tuple[SimpleNamespace, SimpleNamespace]]:
                return self._rows

        class FakeSession:
            def __init__(self) -> None:
                self.execute_count = 0

            async def execute(self, statement: object) -> FakeResult:
                del statement
                self.execute_count += 1
                return FakeResult(rows=completed_rows)

        class FakeSessionContext:
            def __init__(self) -> None:
                self.session = FakeSession()

            async def __aenter__(self) -> FakeSession:
                return self.session

            async def __aexit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                del exc_type, exc, traceback

        with (
            patch(
                "app.services.analyzer._unannotated_answers",
                new=AsyncMock(return_value=pending),
            ),
            patch(
                "app.services.analyzer._processing_artifact",
                new=fake_processing,
            ),
            patch(
                "app.services.analyzer._save_annotations",
                new=fake_save,
            ),
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ) as progress,
            patch(
                "app.services.analyzer.SessionLocal",
                new=lambda: FakeSessionContext(),
            ),
        ):
            await _annotate_answers(
                "run-id",
                profile,
                catalog,
            )

        self.assertEqual(
            max_active_requests,
            PROCESSING_BATCH_CONCURRENCY,
        )
        self.assertEqual(max_active_saves, 1)
        self.assertEqual(save_order, [1, 11, 21])
        self.assertEqual(
            [call.kwargs["percent"] for call in progress.await_args_list],
            [76, 79, 80],
        )

    async def test_answer_processing_uses_terra_with_high_reasoning(self) -> None:
        with patch(
            "app.services.analyzer._structured_artifact",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ) as structured:
            result = await _processing_artifact(
                "run-id",
                stage_key="knowledge_gap",
                artifact_key="annotations_test",
                schema={"type": "object"},
                schema_name="annotations_test",
                system="Разметь ответы.",
                user_payload={"answers": []},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(PROCESSING_MODEL, "openai/gpt-5.6-terra")
        self.assertEqual(structured.await_args.kwargs["model"], PROCESSING_MODEL)
        self.assertEqual(structured.await_args.kwargs["reasoning_effort"], "high")

    async def test_saved_answer_reprocessing_never_calls_the_model_panel(self) -> None:
        with (
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.analyzer._technical_summary",
                new=AsyncMock(return_value={"score": 90}),
            ),
            patch(
                "app.services.analyzer._review_technical_summary",
                new=AsyncMock(return_value={"findings": []}),
            ),
            patch(
                "app.services.analyzer._site_context",
                new=AsyncMock(return_value={"requested_site": {}}),
            ),
            patch(
                "app.services.analyzer._classify_site",
                new=AsyncMock(return_value={"brand_name": "Example"}),
            ),
            patch(
                "app.services.analyzer._finish_saved_answer_analysis",
                new_callable=AsyncMock,
            ) as finish,
            patch(
                "app.services.analyzer._generate_prompt_set",
                new_callable=AsyncMock,
            ) as generate_prompts,
            patch(
                "app.services.analyzer._run_panel",
                new_callable=AsyncMock,
            ) as run_panel,
        ):
            await reprocess_saved_answers("run-id")

        finish.assert_awaited_once()
        self.assertFalse(
            finish.await_args.kwargs["regenerate_illustrations"]
        )
        generate_prompts.assert_not_awaited()
        run_panel.assert_not_awaited()

    async def test_visual_concepts_use_strong_model_with_independent_artifact(self) -> None:
        concepts = {
            "illustrations": [
                {
                    "role": role,
                    "title": f"Схема {index}",
                    "caption": "Вывод по рассчитанным данным",
                    "alt_text": "Описание схемы",
                    "core_claim": "Подтверждённый вывод",
                    "evidence_paths": [
                        {
                            "technical_access": "/technical/score",
                            "competitive_visibility": "/discovery/portfolio/web/score",
                            "web_memory_gap": "/brand_knowledge/memory/specific_rate",
                        }[role]
                    ],
                    "context_for_image": "Specific market and product context.",
                    "creative_brief": {
                        "visual_thesis": f"Distinct concept {index}",
                        "scene": "A bold editorial scene.",
                        "composition": "Asymmetric spatial hierarchy.",
                        "materials_and_light": "Tactile materials and strong light.",
                        "emotional_tone": "Confident.",
                        "target_treatment": "A single decisive accent.",
                        "diversity_move": f"Different visual move {index}.",
                    },
                }
                for index, role in enumerate(
                    (
                        "technical_access",
                        "competitive_visibility",
                        "web_memory_gap",
                    ),
                    start=1,
                )
            ]
        }
        response = SimpleNamespace(
            parsed=concepts,
            text=json.dumps(concepts, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            result = await _illustration_concepts(
                "run-id",
                {
                    "brand": {"name": "Example"},
                    "technical": {"score": 90},
                    "discovery": {"portfolio": {"web": {"score": 30}}},
                    "brand_knowledge": {"memory": {"specific_rate": 60}},
                },
                [{"target_role": "recommended"}],
            )

        self.assertEqual(result, concepts["illustrations"])
        request = chat_mock.await_args.kwargs
        self.assertEqual(request["model"], ILLUSTRATION_CONCEPT_MODEL)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(
            request["response_schema"],
            ILLUSTRATION_CONCEPTS_SCHEMA,
        )
        system_prompt = request["messages"][0]["content"]
        self.assertIn("report_data — единственный актуальный снимок", system_prompt)
        self.assertIn("n_pairs=0", system_prompt)
        self.assertIn("observed_difference=null", system_prompt)
        completed = save_artifact.await_args_list[-1].kwargs
        self.assertEqual(completed["artifact_key"], "illustration_concepts")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["model"], ILLUSTRATION_CONCEPT_MODEL)

    async def test_image_quality_review_is_multimodal_terra_high(self) -> None:
        review = {
            "usable": True,
            "inferred_message": "Схема ясно показывает сравнение.",
            "facts_grounded": True,
            "claim_readable": True,
            "unsupported_assertions": [],
            "visible_text_problems": [],
            "hard_blockers": [],
            "scores": {
                "context_specificity": 4,
                "visual_story": 4,
                "distinctiveness": 4,
                "hierarchy": 4,
                "craft": 4,
                "richness": 4,
            },
            "strengths": ["Ясная композиция."],
            "improvements": [],
            "retry_instruction": "",
        }
        response = SimpleNamespace(
            parsed=review,
            text=json.dumps(review, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        generation_prompt = "PRIVATE_GENERATION_PROMPT_MARKER"
        concept = {"role": "competitive_visibility"}
        fact_context = {"web_visibility": {"score": 0}}
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
        ):
            result = await _review_illustration(
                "run-id",
                sequence=2,
                concept=concept,
                fact_context=fact_context,
                generation_prompt=generation_prompt,
                image_content=b"fake-png-bytes",
                media_type="image/png",
            )

        self.assertEqual(result, review)
        request = chat_mock.await_args.kwargs
        self.assertEqual(request["model"], PROCESSING_MODEL)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["response_schema"], ILLUSTRATION_QA_SCHEMA)
        content = request["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        payload = json.loads(content[0]["text"])
        self.assertEqual(payload["concept"], concept)
        self.assertEqual(payload["fact_context"], fact_context)
        self.assertNotIn("generation_prompt", payload)
        self.assertNotIn(generation_prompt, content[0]["text"])
        self.assertEqual(
            payload["generation_prompt_sha256"],
            hashlib.sha256(generation_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertTrue(payload["quality_requirements"]["facts_grounded"])
        self.assertTrue(
            payload["quality_requirements"]["no_unsupported_assertions"]
        )
        self.assertTrue(
            content[1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        encoded_image = content[1]["image_url"]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded_image), b"fake-png-bytes")

    async def test_image_quality_cache_and_request_are_bound_to_prompt_hash(
        self,
    ) -> None:
        review = {
            "usable": True,
            "inferred_message": "Фактическая схема.",
            "facts_grounded": True,
            "claim_readable": True,
            "unsupported_assertions": [],
            "visible_text_problems": [],
            "hard_blockers": [],
            "scores": {
                "context_specificity": 4,
                "visual_story": 4,
                "distinctiveness": 4,
                "hierarchy": 4,
                "craft": 4,
                "richness": 4,
            },
            "strengths": [],
            "improvements": [],
            "retry_instruction": "",
        }
        response = SimpleNamespace(
            parsed=review,
            text=json.dumps(review, ensure_ascii=False),
            usage={"total_tokens": 1},
        )
        prompts = [
            "PRIVATE_PROMPT_ALPHA",
            "PRIVATE_PROMPT_BETA",
        ]
        with (
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ) as artifact_output,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                return_value=response,
            ) as chat_mock,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ),
        ):
            for generation_prompt in prompts:
                await _review_illustration(
                    "run-id",
                    sequence=1,
                    concept={"role": "technical_access"},
                    fact_context={"technical": {"score": 95}},
                    generation_prompt=generation_prompt,
                    image_content=b"same-image",
                    media_type="image/png",
                )

        cache_inputs = [
            call.kwargs["input_json"]
            for call in artifact_output.await_args_list
        ]
        request_inputs = [
            json.loads(call.kwargs["messages"][1]["content"][0]["text"])
            for call in chat_mock.await_args_list
        ]
        expected_hashes = [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in prompts
        ]
        self.assertEqual(
            [
                payload["generation_prompt_sha256"]
                for payload in cache_inputs
            ],
            expected_hashes,
        )
        self.assertEqual(request_inputs, cache_inputs)
        self.assertNotEqual(cache_inputs[0], cache_inputs[1])
        for prompt, request_payload in zip(prompts, request_inputs):
            self.assertNotIn("generation_prompt", request_payload)
            self.assertNotIn(
                prompt,
                json.dumps(request_payload, ensure_ascii=False),
            )

    async def test_rejected_image_is_regenerated_with_review_feedback(self) -> None:
        rejected = {
            "usable": False,
            "inferred_message": "Декоративная сцена без ясного вывода.",
            "facts_grounded": True,
            "claim_readable": False,
            "unsupported_assertions": [],
            "visible_text_problems": [],
            "hard_blockers": ["Главный вывод не считывается."],
            "scores": {
                "context_specificity": 2,
                "visual_story": 1,
                "distinctiveness": 3,
                "hierarchy": 1,
                "craft": 3,
                "richness": 3,
            },
            "strengths": ["Выразительная фактура."],
            "improvements": ["Нужен ясный смысловой центр."],
            "retry_instruction": (
                "Use a clear paired contrast with a stronger focal point."
            ),
        }
        approved = {
            "usable": True,
            "inferred_message": "Два режима сопоставлены через единый образ.",
            "facts_grounded": True,
            "claim_readable": True,
            "unsupported_assertions": [],
            "visible_text_problems": [],
            "hard_blockers": [],
            "scores": {
                "context_specificity": 4,
                "visual_story": 5,
                "distinctiveness": 4,
                "hierarchy": 5,
                "craft": 4,
                "richness": 4,
            },
            "strengths": ["Сильный смысловой центр."],
            "improvements": [],
            "retry_instruction": "",
        }
        images = [
            SimpleNamespace(
                content=b"first",
                extension="png",
                media_type="image/png",
                usage={"cost": 0.1},
            ),
            SimpleNamespace(
                content=b"second",
                extension="png",
                media_type="image/png",
                usage={"cost": 0.1},
            ),
        ]
        with (
            patch(
                "app.services.analyzer.generate_image",
                new=AsyncMock(side_effect=images),
            ) as generate_mock,
            patch(
                "app.services.analyzer._review_illustration",
                new=AsyncMock(side_effect=[rejected, approved]),
            ),
        ):
            image, review, accepted_prompt, attempts = (
                await _generate_reviewed_image(
                    "run-id",
                    sequence=3,
                    concept={"role": "web_memory_gap"},
                    fact_context={"knowledge_gap": 0},
                    base_prompt="Build a literal comparison.",
                )
            )

        self.assertEqual(image.content, b"second")
        self.assertEqual(review["inferred_message"], approved["inferred_message"])
        self.assertEqual(len(review["_candidate_history"]), 2)
        self.assertEqual(attempts, 2)
        self.assertEqual(generate_mock.await_count, 2)
        self.assertIn("QUALITY REVIEW REJECTED", accepted_prompt)
        self.assertIn("clear paired contrast", accepted_prompt)
        self.assertTrue(_illustration_review_errors(rejected))
        self.assertEqual(_illustration_review_errors(approved), [])

    async def test_third_candidate_is_used_only_after_two_rejections(self) -> None:
        rejected = {
            "usable": False,
            "inferred_message": "Смысл искажён.",
            "facts_grounded": False,
            "claim_readable": False,
            "unsupported_assertions": ["Добавлена неподтверждённая сущность."],
            "visible_text_problems": [],
            "hard_blockers": ["Главный вывод противоречит данным."],
            "scores": {
                "context_specificity": 2,
                "visual_story": 2,
                "distinctiveness": 3,
                "hierarchy": 2,
                "craft": 3,
                "richness": 3,
            },
            "strengths": [],
            "improvements": ["Собрать новую метафору."],
            "retry_instruction": "Invent a different fact-grounded scene.",
        }
        approved = {
            **rejected,
            "usable": True,
            "inferred_message": "Фактический контраст считывается.",
            "facts_grounded": True,
            "claim_readable": True,
            "unsupported_assertions": [],
            "visible_text_problems": ["Незначительная нечитаемая фактура."],
            "hard_blockers": [],
            "scores": {
                **rejected["scores"],
                "context_specificity": 5,
            },
        }
        images = [
            SimpleNamespace(
                content=f"candidate-{index}".encode(),
                extension="png",
                media_type="image/png",
                usage={"cost": 0.1},
            )
            for index in range(1, 4)
        ]
        with (
            patch(
                "app.services.analyzer.generate_image",
                new=AsyncMock(side_effect=images),
            ) as generate_mock,
            patch(
                "app.services.analyzer._review_illustration",
                new=AsyncMock(side_effect=[rejected, rejected, approved]),
            ),
        ):
            image, review, _accepted_prompt, attempts = (
                await _generate_reviewed_image(
                    "run-id",
                    sequence=1,
                    concept={"role": "technical_access"},
                    fact_context={"technical": {"score": 95}},
                    base_prompt="Show server-readable content.",
                )
            )

        self.assertEqual(image.content, b"candidate-3")
        self.assertEqual(attempts, 3)
        self.assertEqual(generate_mock.await_count, 3)
        self.assertEqual(len(review["_candidate_history"]), 3)
        self.assertEqual(_illustration_review_errors(approved), [])


class IllustrationRoleConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_roles_are_bounded_parallel_with_ordered_output_and_writes(
        self,
    ) -> None:
        rows: dict[int, ReportIllustration] = {}
        active_generations = 0
        max_active_generations = 0
        active_commits = 0
        max_active_commits = 0
        generation_pair_started = asyncio.Event()
        progress_percents: list[int] = []

        class FakeResult:
            def __init__(self, row: ReportIllustration | None) -> None:
                self.row = row

            def scalar_one_or_none(self) -> ReportIllustration | None:
                return self.row

            def scalar_one(self) -> ReportIllustration:
                if self.row is None:
                    raise AssertionError("Illustration row was not created")
                return self.row

        class FakeSession:
            async def execute(self, statement: object) -> FakeResult:
                params = statement.compile().params  # type: ignore[attr-defined]
                sequence = next(
                    int(value)
                    for key, value in params.items()
                    if key.startswith("sequence_")
                )
                return FakeResult(rows.get(sequence))

            def add(self, illustration: ReportIllustration) -> None:
                rows[illustration.sequence] = illustration

            async def commit(self) -> None:
                nonlocal active_commits, max_active_commits
                active_commits += 1
                max_active_commits = max(max_active_commits, active_commits)
                try:
                    await asyncio.sleep(0)
                finally:
                    active_commits -= 1

        class FakeSessionContext:
            async def __aenter__(self) -> FakeSession:
                return FakeSession()

            async def __aexit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                del exc_type, exc, traceback

        async def fake_generate_reviewed_image(
            run_id: str,
            **kwargs: object,
        ) -> tuple[SimpleNamespace, dict[str, object], str, int]:
            nonlocal active_generations, max_active_generations
            self.assertEqual(run_id, "run-id")
            sequence = int(kwargs["sequence"])
            active_generations += 1
            max_active_generations = max(
                max_active_generations,
                active_generations,
            )
            if active_generations == ILLUSTRATION_ROLE_CONCURRENCY:
                generation_pair_started.set()
            try:
                await asyncio.wait_for(generation_pair_started.wait(), timeout=1)
                await asyncio.sleep({1: 0.03, 2: 0.01, 3: 0.0}[sequence])
                return (
                    SimpleNamespace(
                        content=f"image-{sequence}".encode(),
                        extension="png",
                        media_type="image/png",
                        usage={"cost": sequence},
                    ),
                    {"usable": True, "_candidate_history": []},
                    f"accepted-{sequence}",
                    2,
                )
            finally:
                active_generations -= 1

        async def fake_update_progress(
            run_id: str,
            *,
            percent: int,
            **kwargs: object,
        ) -> None:
            del kwargs
            self.assertEqual(run_id, "run-id")
            progress_percents.append(percent)
            await asyncio.sleep(0)

        concepts = [
            {
                "role": role,
                "title": f"Иллюстрация {sequence}",
                "caption": "Подпись",
                "alt_text": "Описание",
                "evidence_paths": [],
            }
            for sequence, role in enumerate(
                (
                    "technical_access",
                    "competitive_visibility",
                    "web_memory_gap",
                ),
                start=1,
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch(
                    "app.services.analyzer.GENERATED_DIR",
                    Path(temporary_directory),
                ),
                patch(
                    "app.services.analyzer.SessionLocal",
                    new=lambda: FakeSessionContext(),
                ),
                patch(
                    "app.services.analyzer._illustration_fact_context",
                    return_value={},
                ),
                patch(
                    "app.services.analyzer._illustration_generation_concept",
                    side_effect=lambda concept, **_kwargs: concept,
                ),
                patch(
                    "app.services.analyzer._illustration_prompt",
                    side_effect=lambda _concept, **kwargs: (
                        f"prompt-{kwargs['sequence']}"
                    ),
                ),
                patch(
                    "app.services.analyzer._generate_reviewed_image",
                    new=fake_generate_reviewed_image,
                ),
                patch(
                    "app.services.analyzer.update_progress",
                    new=fake_update_progress,
                ),
            ):
                result = await _generate_illustrations(
                    "run-id",
                    brand_name="Example",
                    concepts=concepts,
                    public_report={},
                )

            generated = Path(temporary_directory) / "run-id"
            self.assertEqual(
                sorted(path.name for path in generated.iterdir()),
                ["01.png", "02.png", "03.png"],
            )

        self.assertEqual(ILLUSTRATION_ROLE_CONCURRENCY, 2)
        self.assertEqual(max_active_generations, 2)
        self.assertEqual(max_active_commits, 1)
        self.assertEqual([item["sequence"] for item in result], [1, 2, 3])
        self.assertEqual(sorted(rows), [1, 2, 3])
        self.assertEqual(progress_percents, [91, 93, 95, 97])


class ParallelReportBranchTests(unittest.IsolatedAsyncioTestCase):
    async def test_analytics_and_visual_branch_start_in_parallel(self) -> None:
        analytics_started = asyncio.Event()
        visuals_started = asyncio.Event()

        async def final_report(*_args: object) -> dict[str, object]:
            analytics_started.set()
            await asyncio.wait_for(visuals_started.wait(), timeout=0.5)
            return {"sections": [{}], "actions": [{}]}

        async def illustration_concepts(*_args: object) -> list[dict[str, str]]:
            visuals_started.set()
            await asyncio.wait_for(analytics_started.wait(), timeout=0.5)
            return [{"title": "Схема"} for _ in range(3)]

        illustrations = [
            {"sequence": index, "file_url": f"/static/{index}.png"}
            for index in range(1, 4)
        ]
        with (
            patch(
                "app.services.analyzer._final_report",
                new=AsyncMock(side_effect=final_report),
            ),
            patch(
                "app.services.analyzer._illustration_concepts",
                new=AsyncMock(side_effect=illustration_concepts),
            ),
            patch(
                "app.services.analyzer._generate_illustrations",
                new=AsyncMock(return_value=illustrations),
            ),
            patch(
                "app.services.analyzer.update_progress",
                new_callable=AsyncMock,
            ),
        ):
            final, generated = await asyncio.wait_for(
                _run_report_branches(
                    "run-id",
                    public_report={"brand": {"name": "Example"}},
                    evidence=[],
                    answer_corpus={"manifest": {}, "answers": []},
                    brand_name="Example",
                ),
                timeout=1,
            )

        self.assertEqual(final["sections"], [{}])
        self.assertEqual(generated, illustrations)


class ReusedIllustrationMetadataTests(unittest.IsolatedAsyncioTestCase):
    def test_reuse_refreshes_copy_but_keeps_alt_for_saved_bitmap(self) -> None:
        saved = [
            {
                "sequence": 1,
                "title": "Старая подпись",
                "caption": "Без веба конкретны 4 из 12 ответов; разрыв 22,3.",
                "alt_text": "Старое описание",
                "file_url": "/static/generated/run-id/01.png",
                "generation_prompt": "Не должен попасть в публичный отчёт",
            }
        ]
        refreshed = [
            {
                "title": "Сравнение памяти пока недоступно",
                "caption": (
                    "Сохранённый корпус не позволяет честно сопоставить "
                    "ответы с веб-поиском и без него."
                ),
                "alt_text": "Иллюстрация с пометкой о недостатке данных.",
            }
        ]

        result = _reuse_saved_illustration_assets(saved, refreshed)

        self.assertEqual(
            result,
            [
                {
                    "sequence": 1,
                    "title": "Сравнение памяти пока недоступно",
                    "caption": (
                        "Сохранённый корпус не позволяет честно сопоставить "
                        "ответы с веб-поиском и без него."
                    ),
                    "alt_text": "Старое описание",
                    "file_url": "/static/generated/run-id/01.png",
                }
            ],
        )
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("4 из 12", serialized)
        self.assertNotIn("22,3", serialized)
        self.assertNotIn("generation_prompt", serialized)

        fallback = _reuse_saved_illustration_assets(
            [{"sequence": 1, "file_url": "/static/generated/run-id/01.png"}],
            refreshed,
        )
        self.assertEqual(
            fallback[0]["alt_text"],
            "Иллюстрация с пометкой о недостатке данных.",
        )

    async def test_reanalysis_refreshes_copy_without_generating_images(self) -> None:
        analytics_started = asyncio.Event()
        concepts_started = asyncio.Event()

        async def final_report(*_args: object) -> dict[str, object]:
            analytics_started.set()
            await asyncio.wait_for(concepts_started.wait(), timeout=0.5)
            return {"sections": [{}], "actions": [{}]}

        async def illustration_concepts(*_args: object) -> list[dict[str, str]]:
            concepts_started.set()
            await asyncio.wait_for(analytics_started.wait(), timeout=0.5)
            return [
                {
                    "title": "Актуальный вывод",
                    "caption": "Подпись из пересчитанных метрик.",
                    "alt_text": "Актуальное описание.",
                }
            ]

        with (
            patch(
                "app.services.analyzer._final_report",
                new=AsyncMock(side_effect=final_report),
            ),
            patch(
                "app.services.analyzer._illustration_concepts",
                new=AsyncMock(side_effect=illustration_concepts),
            ) as concepts,
            patch(
                "app.services.analyzer._generate_illustrations",
                new_callable=AsyncMock,
            ) as generate,
        ):
            final, reused = await asyncio.wait_for(
                _run_reused_report_branches(
                    "run-id",
                    public_report={"brand_knowledge": {"memory": {}}},
                    evidence=[],
                    answer_corpus={"manifest": {}, "answers": []},
                    saved_illustrations=[
                        {
                            "sequence": 1,
                            "file_url": "/static/generated/run-id/01.png",
                        }
                    ],
                ),
                timeout=1,
            )

        self.assertEqual(final["sections"], [{}])
        self.assertEqual(reused[0]["title"], "Актуальный вывод")
        self.assertEqual(
            reused[0]["file_url"],
            "/static/generated/run-id/01.png",
        )
        concepts.assert_awaited_once()
        generate.assert_not_awaited()

    async def test_reanalysis_uses_number_free_copy_fallback(self) -> None:
        saved = [
            {
                "sequence": sequence,
                "file_url": f"/static/generated/run-id/0{sequence}.png",
            }
            for sequence in range(1, 4)
        ]
        with (
            patch(
                "app.services.analyzer._final_report",
                new=AsyncMock(return_value={"sections": [{}], "actions": [{}]}),
            ),
            patch(
                "app.services.analyzer._illustration_concepts",
                new=AsyncMock(
                    side_effect=OpenRouterError("unsupported number 100")
                ),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
        ):
            final, reused = await _run_reused_report_branches(
                "run-id",
                public_report={},
                evidence=[],
                answer_corpus={"manifest": {}, "answers": []},
                saved_illustrations=saved,
            )

        self.assertEqual(final["sections"], [{}])
        self.assertEqual(len(reused), 3)
        self.assertTrue(all(item["file_url"] for item in reused))
        serialized = json.dumps(reused, ensure_ascii=False)
        self.assertNotIn("100", serialized)
        fallback_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key")
            == "illustration_concepts_fallback"
        ]
        self.assertEqual(len(fallback_writes), 1)
        self.assertEqual(fallback_writes[0]["status"], "completed")


class FinalAnswerCorpusTests(unittest.TestCase):
    def _rows(
        self,
        count: int = 13,
        *,
        last_suffix: str = "LAST-SENTINEL",
    ) -> list[tuple[ModelAnswer, VisibilityPrompt, AnswerAnnotation]]:
        rows: list[
            tuple[ModelAnswer, VisibilityPrompt, AnswerAnnotation]
        ] = []
        providers = ("openai", "gemini", "perplexity", "deepseek", "claude")
        intents = ("I", "E", "T", "NB", "NAV", "TR")
        for index in range(count):
            suffix = last_suffix if index == count - 1 else "END"
            answer_text = f"FULL-{index}-" + ("x" * (5000 + index)) + suffix
            answer_sha256 = hashlib.sha256(answer_text.encode()).hexdigest()
            answer = ModelAnswer(
                id=index + 1,
                run_id="run-id",
                prompt_id=index + 1,
                provider_key=providers[index % len(providers)],
                model="test/model",
                mode="web" if index % 2 == 0 else "memory",
                status="completed",
                response_text=answer_text,
                citations_json=[{"url": f"https://source.example/{index}"}],
                usage_json={
                    "_aiv_panel_contract": {
                        "version": PANEL_CONTRACT_VERSION,
                        "request_sha256": f"request-{index}",
                    }
                },
            )
            prompt = VisibilityPrompt(
                id=index + 1,
                run_id="run-id",
                prompt_key=f"prompt-{index}",
                intent_class=intents[index % len(intents)],
                role=(
                    "brand_diagnostic"
                    if index >= 15
                    else "unbranded_discovery"
                ),
                text=f"Пользовательский сценарий № {index}",
                sequence=index + 1,
            )
            annotation = AnswerAnnotation(
                answer_id=index + 1,
                annotation_json={
                    "_annotation_version": ANNOTATION_VERSION,
                    "_answer_sha256": answer_sha256,
                    "_answer_model": "test/model",
                    "_annotation_input_sha256": "annotation-context",
                    "valid": True,
                    "target_mentioned": index % 3 == 0,
                    "target_position": 1 if index % 3 == 0 else None,
                    "target_role": (
                        "recommended" if index % 3 == 0 else "absent"
                    ),
                    "sentiment": "positive" if index % 3 == 0 else "unknown",
                    "evidence": [],
                    "uncertainties": [],
                },
            )
            rows.append((answer, prompt, annotation))
        return rows

    @staticmethod
    def _expected_cells(
        metric_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            {
                "prompt_id": row["prompt_id"],
                "provider_key": row["provider_key"],
                "model": row["model"],
                "mode": row["mode"],
            }
            for row in metric_rows
        ]

    def test_more_than_twelve_answers_keep_last_full_sentinel(self) -> None:
        corpus = _full_answer_corpus_items(self._rows())

        self.assertEqual(len(corpus), 13)
        self.assertTrue(corpus[-1]["answer_text"].endswith("LAST-SENTINEL"))
        self.assertEqual(corpus[-1]["answer_id"], 13)
        self.assertEqual(corpus[-1]["provider_key"], "perplexity")
        self.assertEqual(corpus[-1]["model"], "test/model")
        self.assertEqual(corpus[-1]["citations"][0]["url"], "https://source.example/12")
        self.assertEqual(
            corpus[-1]["provenance"]["raw_answer_sha256"],
            hashlib.sha256(corpus[-1]["answer_text"].encode()).hexdigest(),
        )
        for item in corpus:
            self.assertGreater(len(item["answer_text"]), 5000)
            self.assertTrue(item["scenario"].startswith("Пользовательский сценарий"))

    def test_changing_thirteenth_answer_invalidates_manifest_and_final_input(
        self,
    ) -> None:
        original_models = self._rows()
        changed_models = self._rows(last_suffix="CHANGED-LAST-SENTINEL")
        original_rows = _rows_from_full_answer_models(original_models)
        changed_rows = _rows_from_full_answer_models(changed_models)
        expected = self._expected_cells(original_rows)
        original_manifest = _final_corpus_manifest(
            original_rows,
            expected_cells=expected,
        )
        changed_manifest = _final_corpus_manifest(
            changed_rows,
            expected_cells=expected,
        )
        original_payload = _final_report_payload(
            {"brand": {"name": "Example"}},
            {
                "manifest": original_manifest,
                "answers": _full_answer_corpus_items(original_models),
            },
        )
        changed_payload = _final_report_payload(
            {"brand": {"name": "Example"}},
            {
                "manifest": changed_manifest,
                "answers": _full_answer_corpus_items(changed_models),
            },
        )

        self.assertNotEqual(
            original_manifest["observed_cells_sha256"],
            changed_manifest["observed_cells_sha256"],
        )
        self.assertNotEqual(original_manifest["digest"], changed_manifest["digest"])
        self.assertNotEqual(original_payload, changed_payload)
        cached_artifact = RunArtifact(
            run_id="run-id",
            stage_key="report",
            artifact_key="final_report",
            status="completed",
            model=ANALYSIS_MODEL,
            prompt_version=FINAL_REPORT_VERSION,
            input_json=original_payload,
            output_json={"headline": "Cached"},
        )
        self.assertFalse(
            _artifact_cache_matches(
                cached_artifact,
                input_json=changed_payload,
                model=ANALYSIS_MODEL,
                prompt_version=FINAL_REPORT_VERSION,
            )
        )

        changed_id_rows = [dict(row) for row in original_rows]
        changed_id_rows[-1]["answer_id"] = 13_000
        changed_id_manifest = _final_corpus_manifest(
            changed_id_rows,
            expected_cells=expected,
        )
        self.assertNotEqual(
            original_manifest["observed_cells_sha256"],
            changed_id_manifest["observed_cells_sha256"],
        )
        self.assertNotEqual(
            original_manifest["digest"],
            changed_id_manifest["digest"],
        )

    def test_manifest_mismatch_is_never_complete(self) -> None:
        models = self._rows(count=2)
        metric_rows = _rows_from_full_answer_models(models)
        expected = [
            *self._expected_cells(metric_rows),
            {
                "prompt_id": 99,
                "provider_key": "claude",
                "model": "test/model",
                "mode": "memory",
            },
        ]

        manifest = _final_corpus_manifest(
            metric_rows,
            expected_cells=expected,
        )

        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["expected_count"], 3)
        self.assertEqual(manifest["observed_count"], 2)
        self.assertEqual(manifest["missing_cells"][0]["prompt_id"], 99)

    def test_manifest_accepts_explicit_invalid_annotation_as_observation(
        self,
    ) -> None:
        models = self._rows(count=1)
        models[0][2].annotation_json = {
            **models[0][2].annotation_json,
            "valid": False,
        }
        metric_rows = _rows_from_full_answer_models(models)
        manifest = _final_corpus_manifest(
            metric_rows,
            expected_cells=self._expected_cells(metric_rows),
        )

        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["invalid_cells"], [])

    def test_final_selection_is_deterministic_and_respects_attestation(
        self,
    ) -> None:
        models = self._rows(count=20)
        items = _full_answer_corpus_items(models)
        metric_rows = _rows_from_full_answer_models(models)
        manifest = _final_corpus_manifest(
            metric_rows,
            expected_cells=self._expected_cells(metric_rows),
        )

        selected, selection_manifest = _select_final_answer_context(
            items,
            corpus_manifest=manifest,
        )
        reversed_selected, reversed_manifest = _select_final_answer_context(
            list(reversed(items)),
            corpus_manifest=manifest,
        )

        self.assertLessEqual(len(selected), FINAL_CONTEXT_MAX_ANSWERS)
        self.assertTrue(selection_manifest["coverage_complete"])
        self.assertEqual(
            [item["answer_id"] for item in selected],
            [item["answer_id"] for item in reversed_selected],
        )
        self.assertEqual(
            selection_manifest["digest"],
            reversed_manifest["digest"],
        )
        self.assertEqual(
            selection_manifest["selected_full_text_count"]
            + selection_manifest["selected_metadata_only_count"],
            len(selected),
        )
        for item in selected:
            if item["context_access"] == "full_text":
                self.assertGreater(len(item["answer_text"]), 5000)
                self.assertTrue(
                    item["answer_text"].endswith(("END", "LAST-SENTINEL"))
                )
                self.assertEqual(item["verified_mode"], item["requested_mode"])
            else:
                self.assertEqual(item["context_access"], "metadata_only")
                self.assertIsNone(item["verified_mode"])
                self.assertNotIn("answer_text", item)
                self.assertNotIn("annotation", item)
                self.assertNotIn("citations", item)

    def test_final_selection_finds_feasible_exact_cover_after_greedy_trap(
        self,
    ) -> None:
        specs = [
            (1, "p1", 0, "web", "T", "b", "missing", False, False, "absent", "unknown"),
            (2, "p1", 0, "memory", "T", "u", "missing", True, True, "absent", "mixed"),
            (3, "p1", 1, "web", "T", "b", "legacy", True, False, "recommended", "positive"),
            (4, "p1", 1, "memory", "E", "b", "legacy", True, True, "absent", "unknown"),
            (5, "p2", 0, "web", "I", "u", "ok", True, False, "mentioned", "positive"),
            (6, "p2", 1, "web", "E", "b", "ok", False, True, "absent", "unknown"),
        ]
        answers = []
        for (
            answer_id,
            provider,
            prompt_id,
            mode,
            intent,
            role,
            evidence_state,
            valid,
            target_mentioned,
            target_role,
            sentiment,
        ) in specs:
            answers.append(
                {
                    "answer_id": answer_id,
                    "prompt_id": prompt_id,
                    "provider_key": provider,
                    "mode": mode,
                    "intent_class": intent,
                    "scenario_role": role,
                    "scenario_sequence": prompt_id + 1,
                    "answer_text": f"answer-{answer_id}",
                    "citations": [],
                    "annotation": {
                        "valid": valid,
                        "target_mentioned": target_mentioned,
                        "target_role": target_role,
                        "sentiment": sentiment,
                        "evidence": [],
                        "uncertainties": [],
                    },
                    "panel_evidence": {
                        "reason": evidence_state,
                        "sha256": f"pe-{answer_id}",
                    },
                    "provenance": {
                        "raw_answer_sha256": f"raw-{answer_id}",
                        "annotation_sha256": f"ann-{answer_id}",
                    },
                }
            )

        selected, selection_manifest = _select_final_answer_context(
            answers,
            corpus_manifest={
                "digest": "full-corpus",
                "critic_rows_sha256": "critic-rows",
            },
            max_answers=4,
        )

        self.assertEqual(
            sorted(item["answer_id"] for item in selected),
            [2, 3, 5, 6],
        )
        self.assertTrue(selection_manifest["coverage_complete"])
        self.assertEqual(selection_manifest["selected_count"], 4)
        self.assertEqual(
            selection_manifest["full_corpus_critic_rows_sha256"],
            "critic-rows",
        )

    def test_final_selection_prefers_short_equivalent_full_answer(self) -> None:
        def answer(answer_id: int, text: str, evidence: list[str]) -> dict[str, Any]:
            return {
                "answer_id": answer_id,
                "prompt_id": 1,
                "provider_key": "openai",
                "mode": "web",
                "intent_class": "I",
                "scenario_role": "unbranded_discovery",
                "scenario_sequence": 1,
                "answer_text": text,
                "citations": [],
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "evidence": evidence,
                    "uncertainties": [],
                },
                "panel_evidence": {
                    "reason": "verified",
                    "sha256": f"pe-{answer_id}",
                },
                "provenance": {
                    "raw_answer_sha256": f"raw-{answer_id}",
                    "annotation_sha256": f"ann-{answer_id}",
                },
            }

        short = answer(1, "Краткий полный ответ.", [])
        long = answer(2, "Д" * 20_000, ["Формально богаче"])
        selected, _manifest = _select_final_answer_context(
            [long, short],
            corpus_manifest={"digest": "full-corpus"},
            max_answers=1,
        )

        self.assertEqual([item["answer_id"] for item in selected], [1])
        self.assertEqual(selected[0]["answer_text"], short["answer_text"])

    def test_public_methodology_does_not_claim_unavailable_memory_comparison(
        self,
    ) -> None:
        unavailable = {
            "score": None,
            "specific_rate": None,
            "state": "unknown",
            "data_state": "unavailable",
            "valid_answers": 0,
        }
        web = {
            "score": 42.0,
            "specific_rate": 42.0,
            "state": "visible",
            "data_state": "complete",
            "valid_answers": 5,
        }
        metrics = {
            "parent_discovery": {"web": web, "memory": unavailable},
            "portfolio_visibility": {"web": web, "memory": unavailable},
            "brand_knowledge": {
                "web": web,
                "memory": unavailable,
                "providers": [],
            },
            "paired_web_lift": {"n_pairs": 0},
            "model_consistency": None,
            "providers": [],
            "intents": [],
            "sentiment": {},
            "web": web,
            "memory": unavailable,
            "competitors": [],
            "quality": {"panel_evidence": {}},
            "metric_note": "Экспресс-снимок.",
        }
        report = _build_public_report(
            profile={"brand_name": "Example"},
            technical={
                "state": "complete",
                "score": 90,
                "coverage": {
                    "coverage_state": "complete",
                    "evaluated_pages": 2,
                },
            },
            technical_review={},
            metrics=metrics,
        )

        methodology = report["methodology"]
        self.assertNotIn(
            "сравнивает ответы с веб-поиском и без него",
            methodology["summary"],
        )
        self.assertFalse(
            any(
                "без веб-поиска" in item["name"]
                for item in methodology["modes"]
            )
        )
        self.assertIn("не участвует в процентах", methodology["offline_limit"])

    def test_public_report_preserves_observational_rates_but_withholds_facets(
        self,
    ) -> None:
        web = {
            "score": 62.5,
            "mention_rate": 75.0,
            "specific_rate": 50.0,
            "state": "visible",
            "data_state": "complete",
            "valid_answers": 4,
            "expected_answers": 4,
        }
        observational_memory = {
            "score": 37.5,
            "mention_rate": 25.0,
            "specific_rate": 50.0,
            "state": "visible",
            "data_state": "limited",
            "valid_answers": 4,
            "expected_answers": 4,
            "evidence_state": "legacy_observational",
            "observational_answers": 4,
            "strictly_attested_answers": 0,
            "strict_no_web_verified": False,
            "limitation_reason": LEGACY_MEMORY_OBSERVATION_REASON,
            "provenance": {
                "cohort_sha256": "observational-cohort-sha256",
                "source": "saved_model_answers",
            },
            "facets": {
                "identity": 75.0,
                "offering": 50.0,
            },
        }
        knowledge_provider = {
            "provider_key": "openai",
            "web": copy.deepcopy(web),
            "memory": copy.deepcopy(observational_memory),
        }
        discovery_provider = {
            "provider_key": "openai",
            "brand_knowledge": {
                "web": copy.deepcopy(web),
                "memory": copy.deepcopy(observational_memory),
            },
        }
        metrics = {
            "parent_discovery": {
                "web": copy.deepcopy(web),
                "memory": copy.deepcopy(observational_memory),
            },
            "portfolio_visibility": {
                "web": copy.deepcopy(web),
                "memory": copy.deepcopy(observational_memory),
            },
            "brand_knowledge": {
                "web": copy.deepcopy(web),
                "memory": copy.deepcopy(observational_memory),
                "providers": [knowledge_provider],
            },
            "paired_web_lift": {
                "n_pairs": 4,
                "data_state": "limited",
                "causal_interpretation_allowed": False,
            },
            "model_consistency": 0.75,
            "providers": [discovery_provider],
            "intents": [],
            "sentiment": {},
            "web": copy.deepcopy(web),
            "memory": copy.deepcopy(observational_memory),
            "competitors": [],
            "quality": {
                "state": "limited",
                "legacy_observational_answers": 4,
                "panel_evidence": {},
            },
            "metric_note": "Экспресс-снимок.",
        }
        original_metrics = copy.deepcopy(metrics)

        report = _build_public_report(
            profile={"brand_name": "Example"},
            technical={
                "state": "complete",
                "score": 90,
                "coverage": {
                    "coverage_state": "complete",
                    "evaluated_pages": 2,
                },
            },
            technical_review={},
            metrics=metrics,
        )

        public_memory = report["brand_knowledge"]["memory"]
        self.assertEqual(public_memory["score"], 37.5)
        self.assertEqual(public_memory["mention_rate"], 25.0)
        self.assertEqual(public_memory["specific_rate"], 50.0)
        self.assertEqual(public_memory["valid_answers"], 4)
        self.assertEqual(public_memory["expected_answers"], 4)
        self.assertEqual(
            public_memory["evidence_state"],
            "legacy_observational",
        )
        self.assertEqual(public_memory["observational_answers"], 4)
        self.assertEqual(public_memory["strictly_attested_answers"], 0)
        self.assertFalse(public_memory["strict_no_web_verified"])
        self.assertEqual(
            public_memory["provenance"],
            observational_memory["provenance"],
        )
        self.assertNotIn("facets", public_memory)
        self.assertTrue(public_memory["qualitative_context_withheld"])

        public_knowledge_provider_memory = (
            report["brand_knowledge"]["providers"][0]["memory"]
        )
        self.assertNotIn("facets", public_knowledge_provider_memory)
        self.assertTrue(
            public_knowledge_provider_memory["qualitative_context_withheld"]
        )
        public_discovery_provider_memory = (
            report["discovery"]["providers"][0]["brand_knowledge"]["memory"]
        )
        self.assertNotIn("facets", public_discovery_provider_memory)
        self.assertTrue(
            public_discovery_provider_memory["qualitative_context_withheld"]
        )

        self.assertEqual(
            report["key_metrics"]["brand_knowledge"]["label"],
            "Конкретика без зафиксированного веб-поиска",
        )
        observational_mode = report["methodology"]["modes"][1]
        self.assertEqual(
            observational_mode["name"],
            "Безбрендовые сценарии · без зафиксированного веб-поиска",
        )
        self.assertIn(
            "Исторический наблюдательный срез",
            observational_mode["description"],
        )
        self.assertIn(
            CANONICAL_OBSERVATIONAL_MEMORY_LIMITATION,
            report["methodology"]["offline_limit"],
        )
        self.assertEqual(metrics, original_metrics)

    def test_public_report_preserves_unavailable_portfolio_scope(self) -> None:
        unavailable_portfolio = {
            "score": None,
            "mention_rate": None,
            "mention_count": None,
            "state": "unknown",
            "data_state": "unavailable",
            "valid_answers": 6,
            "expected_answers": 6,
            "unavailable_reason": "target_portfolio_unconfirmed",
        }
        available = {
            "score": 40.0,
            "mention_rate": 40.0,
            "specific_rate": 40.0,
            "state": "visible",
            "data_state": "complete",
            "valid_answers": 6,
            "expected_answers": 6,
        }
        portfolio_scope = {
            "state": "unavailable",
            "candidate_entities": 12,
            "confirmed_entities": 0,
            "rejected_entities": 12,
            "reason": "target_portfolio_unconfirmed",
        }
        metrics = {
            "parent_discovery": {"web": available, "memory": available},
            "portfolio_visibility": {
                "web": unavailable_portfolio,
                "memory": unavailable_portfolio,
            },
            "portfolio_scope": portfolio_scope,
            "brand_knowledge": {
                "web": available,
                "memory": available,
                "providers": [],
            },
            "paired_web_lift": {
                "n_pairs": 6,
                "parent": {
                    "web": available,
                    "memory": available,
                    "score_lift": 0.0,
                },
                "portfolio": {
                    "web": unavailable_portfolio,
                    "memory": unavailable_portfolio,
                    "score_lift": None,
                    "data_state": "unavailable",
                    "state": "unknown",
                    "unavailable_reason": "target_portfolio_unconfirmed",
                },
            },
            "model_consistency": None,
            "providers": [],
            "intents": [],
            "sentiment": {},
            "web": unavailable_portfolio,
            "memory": unavailable_portfolio,
            "competitors": [],
            "quality": {"panel_evidence": {}},
            "metric_note": "Экспресс-снимок.",
        }

        report = _build_public_report(
            profile={"brand_name": "Example"},
            technical={
                "state": "complete",
                "score": 90,
                "coverage": {
                    "coverage_state": "complete",
                    "evaluated_pages": 2,
                },
            },
            technical_review={},
            metrics=metrics,
        )

        self.assertEqual(report["portfolio_scope"], portfolio_scope)
        self.assertEqual(
            report["discovery"]["portfolio_scope"],
            portfolio_scope,
        )
        key_metric = report["key_metrics"]["portfolio_capture"]
        self.assertIsNone(key_metric["value"])
        self.assertEqual(key_metric["state"], "unavailable")
        self.assertEqual(key_metric["data_state"], "unavailable")
        self.assertEqual(
            key_metric["unavailable_reason"],
            "target_portfolio_unconfirmed",
        )
        consistency = report["discovery"]["model_consistency"]
        self.assertEqual(consistency["data_state"], "unavailable")
        self.assertEqual(
            consistency["unavailable_reason"],
            "target_portfolio_unconfirmed",
        )
        self.assertNotIn(
            "или его предложение",
            report["methodology"]["modes"][0]["description"],
        )
        self.assertIn(
            "не заменены нулём",
            report["methodology"]["portfolio_scope_limit"],
        )

    def test_final_selection_fails_if_required_states_exceed_limit(self) -> None:
        answers = [
            {
                "answer_id": index,
                "prompt_id": index,
                "provider_key": "openai",
                "mode": "web",
                "intent_class": "I",
                "scenario_role": "unbranded_discovery",
                "scenario_sequence": index,
                "answer_text": f"Ответ {index}",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_role": "absent",
                    "sentiment": "unknown",
                },
                "panel_evidence": {
                    "reason": f"evidence-state-{index}",
                    "sha256": f"sha-{index}",
                },
                "provenance": {
                    "raw_answer_sha256": f"raw-{index}",
                    "annotation_sha256": f"annotation-{index}",
                },
            }
            for index in range(13)
        ]

        with self.assertRaisesRegex(
            OpenRouterError,
            "cannot cover required evidence strata",
        ):
            _select_final_answer_context(
                answers,
                corpus_manifest={"digest": "full-corpus"},
                max_answers=12,
            )

    def test_corpus_ids_and_raw_hashes_equal_critic_manifest(self) -> None:
        models = self._rows()
        critic_rows = _rows_from_full_answer_models(models)
        expected = self._expected_cells(critic_rows)
        critic_manifest = _final_corpus_manifest(
            critic_rows,
            expected_cells=expected,
        )
        final_manifest = _final_corpus_manifest(
            _rows_from_full_answer_models(models),
            expected_cells=expected,
        )
        corpus = _full_answer_corpus_items(models)

        self.assertTrue(critic_manifest["complete"])
        self.assertEqual(final_manifest["digest"], critic_manifest["digest"])
        self.assertEqual(
            final_manifest["critic_rows_sha256"],
            critic_manifest["critic_rows_sha256"],
        )
        self.assertEqual(
            final_manifest["answer_ids"],
            [item["answer_id"] for item in corpus],
        )
        self.assertEqual(
            [
                item["raw_answer_sha256"]
                for item in final_manifest["observed_cells"]
            ],
            [
                item["provenance"]["raw_answer_sha256"]
                for item in corpus
            ],
        )

    def test_final_context_preflight_is_explicit_and_never_truncates(self) -> None:
        payload = {"answer_corpus": [{"answer_text": "я" * 100}]}
        serialized_bytes = len(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

        preflight = _final_input_preflight(payload, token_budget=1)

        self.assertEqual(preflight["state"], "limited")
        self.assertFalse(preflight["accepted"])
        self.assertEqual(
            preflight["estimated_input_tokens"],
            (serialized_bytes + 2) // 3,
        )

        estimated = preflight["estimated_input_tokens"]
        reserved = _final_input_preflight(
            payload,
            token_budget=estimated + 9,
            reserve_tokens=10,
        )
        self.assertFalse(reserved["accepted"])
        self.assertEqual(reserved["repair_reserve_tokens"], 10)


class FinalReportPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_full_corpus_fails_before_opus_and_is_recorded(
        self,
    ) -> None:
        answer_corpus = {
            "manifest": {"digest": "corpus-digest"},
            "answers": [{"answer_text": "я" * 100}],
        }
        with (
            patch.object(settings, "FINAL_INPUT_TOKEN_BUDGET", 1),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
            ) as artifact_output,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "exceeds the configured context budget",
            ):
                await _final_report(
                    "run-id",
                    {"brand": {"name": "Example"}},
                    answer_corpus,
                )

        artifact_output.assert_not_awaited()
        chat_mock.assert_not_awaited()
        saved = save_artifact.await_args.kwargs
        self.assertEqual(saved["artifact_key"], "final_report_preflight")
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["output_json"]["state"], "limited")
        self.assertEqual(saved["output_json"]["input_token_budget"], 1)
        self.assertGreater(
            saved["output_json"]["estimated_input_tokens"],
            1,
        )


class FinalReportStructureRepairTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _payload() -> dict[str, Any]:
        return {
            "report_data": {"brand": {"name": "Example"}},
            "answer_selection_manifest": {"digest": "selection"},
            "answer_corpus_manifest": {"digest": "corpus"},
            "selected_full_answers": [],
        }

    @staticmethod
    def _candidate(marker: str) -> dict[str, Any]:
        return {
            "headline": marker,
            "headline_emphasis": [],
            "sections": [{"heading": "Раздел", "body": marker}],
            "actions": [{"title": "Действие", "why": marker}],
        }

    async def test_structure_repair_keeps_semantic_repair_budget(self) -> None:
        payload = self._payload()
        empty_author = {
            "headline": "Пустой авторский ответ",
            "sections": [],
            "actions": [],
        }
        structurally_repaired = self._candidate("Структура восстановлена")
        semantically_repaired = self._candidate("Семантика исправлена")
        chat_results = [
            SimpleNamespace(parsed=empty_author, text="empty", usage={}),
            SimpleNamespace(
                parsed=structurally_repaired,
                text="structured",
                usage={"attempt": "structure"},
            ),
            SimpleNamespace(
                parsed=semantically_repaired,
                text="semantic",
                usage={"attempt": "semantic"},
            ),
        ]
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=chat_results,
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    {"verdict": "revise"},
                    {"verdict": "pass"},
                ],
            ) as semantic_review,
            patch(
                "app.services.analyzer.report_semantic_blockers",
                side_effect=[["Исправьте семантическую ошибку."], []],
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, semantically_repaired)
        self.assertEqual(final_chat.await_count, 3)
        self.assertEqual(semantic_review.await_count, 2)
        structure_payload = json.loads(
            final_chat.await_args_list[1].kwargs["messages"][1]["content"]
        )
        self.assertEqual(
            structure_payload["structure_validation_errors_to_fix"],
            [
                "В отчёте должен быть хотя бы один содержательный раздел.",
                "В отчёте должно быть хотя бы одно приоритетное действие.",
            ],
        )
        self.assertEqual(structure_payload["rejected_report"], empty_author)
        self.assertNotIn("semantic_review_to_fix", structure_payload)
        semantic_payload = json.loads(
            final_chat.await_args_list[2].kwargs["messages"][1]["content"]
        )
        self.assertEqual(
            semantic_payload["rejected_report"],
            structurally_repaired,
        )
        self.assertEqual(
            semantic_payload["semantic_review_to_fix"]["verdict"],
            "revise",
        )
        self.assertNotIn(
            "structure_validation_errors_to_fix",
            semantic_payload,
        )
        author_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key")
            == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
        ]
        self.assertEqual(
            [item["status"] for item in author_writes],
            ["running", "completed"],
        )
        self.assertEqual(
            author_writes[-1]["output_json"],
            structurally_repaired,
        )

    async def test_second_structurally_empty_response_fails_closed(self) -> None:
        payload = self._payload()
        empty = {
            "headline": "Структура отсутствует",
            "sections": [],
            "actions": [],
        }
        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(parsed=empty, text="one", usage={}),
                    SimpleNamespace(parsed=empty, text="two", usage={}),
                ],
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
            ) as semantic_review,
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "хотя бы один содержательный раздел",
            ):
                await _final_report(
                    "run-id",
                    payload["report_data"],
                    {"manifest": {"digest": "corpus"}, "answers": [{}]},
                )

        self.assertEqual(final_chat.await_count, 2)
        semantic_review.assert_not_awaited()
        author_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key")
            == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
        ]
        self.assertEqual(
            [item["status"] for item in author_writes],
            ["running", "failed"],
        )
        self.assertFalse(
            any(item.get("status") == "completed" for item in author_writes)
        )
        final_writes = [
            call.kwargs
            for call in save_artifact.await_args_list
            if call.kwargs.get("artifact_key") == "final_report"
        ]
        self.assertEqual(final_writes[-1]["status"], "failed")

    async def test_valid_author_candidate_cache_avoids_author_chat(self) -> None:
        payload = self._payload()
        cached_candidate = self._candidate("Кандидат из кэша")

        async def artifact_output(
            _run_id: str,
            artifact_key: str,
            **_kwargs: Any,
        ) -> dict[str, Any] | None:
            if artifact_key == FINAL_REPORT_AUTHOR_ARTIFACT_KEY:
                return copy.deepcopy(cached_candidate)
            return None

        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(side_effect=artifact_output),
            ) as artifact_cache,
            patch(
                "app.services.analyzer._save_artifact",
                new_callable=AsyncMock,
            ) as save_artifact,
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                return_value={"verdict": "pass"},
            ),
            patch(
                "app.services.analyzer.report_semantic_blockers",
                return_value=[],
            ),
        ):
            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, cached_candidate)
        final_chat.assert_not_awaited()
        author_cache_calls = [
            call
            for call in artifact_cache.await_args_list
            if call.args[1] == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
        ]
        self.assertEqual(len(author_cache_calls), 1)
        author_cache_call = author_cache_calls[0]
        self.assertEqual(author_cache_call.kwargs["input_json"], payload)
        self.assertEqual(author_cache_call.kwargs["model"], ANALYSIS_MODEL)
        self.assertEqual(
            author_cache_call.kwargs["prompt_version"],
            FINAL_REPORT_VERSION,
        )
        self.assertFalse(
            any(
                call.kwargs.get("artifact_key")
                == FINAL_REPORT_AUTHOR_ARTIFACT_KEY
                for call in save_artifact.await_args_list
            )
        )

    async def test_semantic_block_invalidates_author_cache_for_next_run(
        self,
    ) -> None:
        payload = self._payload()
        blocked_candidate = self._candidate("Заблокированный кандидат")
        replacement_candidate = self._candidate("Новый кандидат")
        artifacts: dict[str, dict[str, Any]] = {}

        async def artifact_output(
            _run_id: str,
            artifact_key: str,
            **kwargs: Any,
        ) -> dict[str, Any] | None:
            artifact = artifacts.get(artifact_key)
            if not artifact or artifact.get("status") != "completed":
                return None
            if artifact.get("input_json") != kwargs.get("input_json"):
                return None
            if artifact.get("model") != kwargs.get("model"):
                return None
            if artifact.get("prompt_version") != kwargs.get(
                "prompt_version"
            ):
                return None
            return copy.deepcopy(artifact.get("output_json"))

        async def save_artifact(
            _run_id: str,
            **kwargs: Any,
        ) -> None:
            artifacts[str(kwargs["artifact_key"])] = copy.deepcopy(kwargs)

        with (
            patch(
                "app.services.analyzer._final_report_payload",
                return_value=payload,
            ),
            patch(
                "app.services.analyzer._artifact_output",
                new=AsyncMock(side_effect=artifact_output),
            ),
            patch(
                "app.services.analyzer._save_artifact",
                new=AsyncMock(side_effect=save_artifact),
            ),
            patch(
                "app.services.analyzer.chat",
                new_callable=AsyncMock,
                side_effect=[
                    SimpleNamespace(
                        parsed=blocked_candidate,
                        text="blocked",
                        usage={"candidate": 1},
                    ),
                    SimpleNamespace(
                        parsed=replacement_candidate,
                        text="replacement",
                        usage={"candidate": 2},
                    ),
                ],
            ) as final_chat,
            patch(
                "app.services.analyzer._final_report_semantic_review_artifact",
                new_callable=AsyncMock,
                side_effect=[
                    {"verdict": "block"},
                    {"verdict": "pass"},
                ],
            ) as semantic_review,
            patch(
                "app.services.analyzer.report_semantic_blockers",
                side_effect=[["Неподтверждённое утверждение."], []],
            ),
        ):
            with self.assertRaisesRegex(
                OpenRouterError,
                "semantic gate blocked publication",
            ):
                await _final_report(
                    "run-id",
                    payload["report_data"],
                    {"manifest": {"digest": "corpus"}, "answers": [{}]},
                )

            invalidated = artifacts[FINAL_REPORT_AUTHOR_ARTIFACT_KEY]
            self.assertEqual(invalidated["status"], "failed")
            self.assertEqual(
                invalidated["output_json"]["state"],
                "semantic_block",
            )
            self.assertEqual(
                invalidated["output_json"]["semantic_verdict"],
                "block",
            )
            self.assertEqual(
                invalidated["output_json"]["blockers"],
                ["Неподтверждённое утверждение."],
            )

            result = await _final_report(
                "run-id",
                payload["report_data"],
                {"manifest": {"digest": "corpus"}, "answers": [{}]},
            )

        self.assertEqual(result, replacement_candidate)
        self.assertEqual(final_chat.await_count, 2)
        self.assertEqual(semantic_review.await_count, 2)
        reviewed_candidates = [
            call.kwargs["candidate_report"]
            for call in semantic_review.await_args_list
        ]
        self.assertEqual(
            reviewed_candidates,
            [blocked_candidate, replacement_candidate],
        )


class FinalAnswerCorpusDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()

    async def test_full_db_corpus_matches_critic_gate_beyond_twelve_rows(
        self,
    ) -> None:
        run_id = f"test-final-corpus-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            for prompt_index in range(2):
                prompt = VisibilityPrompt(
                    run_id=run_id,
                    prompt_key=f"prompt-{prompt_index}",
                    intent_class="I",
                    role="unbranded_discovery",
                    text=f"Сценарий {prompt_index}",
                    rationale="Проверка полного корпуса.",
                    sequence=prompt_index + 1,
                )
                session.add(prompt)
                await session.flush()
                for mode in ("web", "memory"):
                    for panel in panel_models():
                        model = (
                            panel.model
                            if mode == "web"
                            else panel.memory_model
                        )
                        if model is None:
                            continue
                        is_last = (
                            prompt_index == 1
                            and mode == "memory"
                            and panel.key == "claude"
                        )
                        answer_text = (
                            f"Ответ {prompt_index}/{panel.key}/{mode} "
                            + ("LAST-DB-SENTINEL" if is_last else "END")
                        )
                        raw_sha256 = hashlib.sha256(
                            answer_text.encode("utf-8")
                        ).hexdigest()
                        usage_json, citations_json = _attested_panel_usage(
                            prompt_text=prompt.text,
                            mode=mode,
                            provider_key=panel.key,
                            model=model,
                        )
                        answer = ModelAnswer(
                            run_id=run_id,
                            prompt_id=prompt.id,
                            provider_key=panel.key,
                            model=model,
                            mode=mode,
                            status="completed",
                            response_text=answer_text,
                            citations_json=citations_json,
                            usage_json=usage_json,
                        )
                        session.add(answer)
                        await session.flush()
                        session.add(
                            AnswerAnnotation(
                                answer_id=answer.id,
                                annotation_json={
                                    "_annotation_version": ANNOTATION_VERSION,
                                    "_answer_sha256": raw_sha256,
                                    "_answer_model": model,
                                    "_annotation_input_sha256": "context",
                                    "valid": True,
                                    "target_mentioned": False,
                                },
                            )
                        )
            await session.commit()

        try:
            critic_rows = await _metric_rows(
                run_id,
                annotation_input_sha256="context",
            )
            expected_cells = await _expected_corpus_cells(
                run_id,
                critic_rows,
            )
            critic_manifest = _final_corpus_manifest(
                critic_rows,
                expected_cells=expected_cells,
            )
            critic_gate = {
                "passed": True,
                "corpus_manifest": critic_manifest,
                "provenance": {
                    "rows_sha256": critic_manifest["critic_rows_sha256"],
                },
            }
            async with SessionLocal() as session:
                persisted_rows = (
                    await session.execute(
                        select(ModelAnswer, VisibilityPrompt, AnswerAnnotation)
                        .join(
                            VisibilityPrompt,
                            ModelAnswer.prompt_id == VisibilityPrompt.id,
                        )
                        .outerjoin(
                            AnswerAnnotation,
                            AnswerAnnotation.answer_id == ModelAnswer.id,
                        )
                        .where(ModelAnswer.run_id == run_id)
                    )
                ).all()
            persisted_manifest = _final_corpus_manifest(
                _rows_from_full_answer_models(list(persisted_rows)),
                expected_cells=expected_cells,
            )
            self.assertEqual(
                persisted_manifest,
                critic_manifest,
                msg=json.dumps(
                    {
                        "critic": critic_manifest,
                        "persisted": persisted_manifest,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            corpus = await _full_answer_context(
                run_id,
                critic_gate=critic_gate,
                critic_rows=critic_rows,
                expected_corpus_cells=expected_cells,
            )

            self.assertTrue(corpus["manifest"]["complete"])
            self.assertEqual(len(corpus["answers"]), 18)
            self.assertEqual(
                corpus["manifest"]["digest"],
                critic_manifest["digest"],
            )
            self.assertTrue(
                any(
                    item["answer_text"].endswith("LAST-DB-SENTINEL")
                    for item in corpus["answers"]
                )
            )
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()


class ContentExtractionTests(unittest.TestCase):
    def test_schema_org_microdata_types_support_http_and_https(self) -> None:
        html = """
        <html><body>
          <main>
            <section itemscope itemtype="http://schema.org/Organization">
              <article
                itemscope
                itemtype="https://schema.org/Product https://example.com/Other"
              ></article>
              <div itemscope itemtype="HTTPS://SCHEMA.ORG/Service"></div>
            </section>
          </main>
        </body></html>
        """

        signals = extract_text_signals(html, "text/html")

        self.assertEqual(
            signals["structured_data_types"],
            ["Organization", "Product", "Service"],
        )

    def test_microdata_types_are_deduplicated_with_json_ld(self) -> None:
        html = """
        <html><head>
          <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": ["Organization", "https://schema.org/Product"]
            }
          </script>
        </head><body>
          <main
            itemscope
            itemtype="http://schema.org/Organization https://schema.org/Product"
          >
            <div itemscope itemtype="https://schema.org/LocalBusiness"></div>
            <div itemscope itemtype="http://schema.org/LocalBusiness"></div>
          </main>
        </body></html>
        """

        signals = extract_text_signals(html, "text/html")

        self.assertEqual(
            signals["structured_data_types"],
            ["Organization", "Product", "LocalBusiness"],
        )

    def test_login_widget_does_not_override_real_article_content(self) -> None:
        article = " ".join(
            ["Подробное описание продукта, условий и сценариев применения."] * 40
        )
        html = f"""
        <html><head><title>Продукт</title></head><body>
          <header><form class="login-form"><input type="password"><button>Войти</button></form></header>
          <main><article><h1>Продукт для бизнеса</h1><p>{article}</p></article></main>
        </body></html>
        """
        signals = extract_text_signals(html, "text/html")
        self.assertTrue(signals["auth_form_present"])
        self.assertFalse(signals["looks_like_login_wall"])
        self.assertGreater(signals["main_content_length"], 800)

    def test_header_login_form_does_not_hide_short_product_page(self) -> None:
        product_copy = " ".join(
            [
                "Сервис помогает команде сравнить варианты и выбрать решение.",
                "На странице описаны возможности, условия и следующий шаг.",
                "Пользователь может изучить предложение без регистрации.",
                "Доступны примеры применения и ответы на частые вопросы.",
            ]
        )
        html = f"""
        <html><head><title>Продукт</title></head><body>
          <header>
            <form class="login-form">
              <input type="password"><button>Войти</button>
            </form>
          </header>
          <main><h1>Сервис для продуктовой команды</h1><p>{product_copy}</p></main>
        </body></html>
        """
        signals = extract_text_signals(html, "text/html")
        self.assertTrue(signals["auth_form_present"])
        self.assertGreater(signals["main_content_length"], 180)
        self.assertLess(signals["main_content_length"], 500)
        self.assertFalse(signals["looks_like_login_wall"])

    def test_real_login_wall_is_detected(self) -> None:
        html = """
        <html><head><title>Вход</title></head><body>
          <main><h1>Войдите, чтобы продолжить</h1>
            <form class="login-form"><input name="email"><input type="password"></form>
          </main>
        </body></html>
        """
        signals = extract_text_signals(html, "text/html")
        self.assertTrue(signals["auth_form_present"])
        self.assertTrue(signals["looks_like_login_wall"])

    def test_csr_shell_and_hybrid_ssr_are_distinguished(self) -> None:
        csr = """
        <html><body><div id="root"></div>
        <script src="a.js"></script><script src="b.js"></script><script src="c.js"></script>
        </body></html>
        """
        csr_signals = extract_text_signals(csr, "text/html")
        self.assertEqual(csr_signals["render_strategy"], "client_rendered_shell")

        text_body = " ".join(["Сервер уже отдал содержательный текст страницы."] * 40)
        ssr = f"""
        <html><body><div id="__next"><main><h1>Услуга</h1><p>{text_body}</p></main></div>
        <script id="__NEXT_DATA__" type="application/json">{{}}</script>
        </body></html>
        """
        ssr_signals = extract_text_signals(ssr, "text/html")
        self.assertEqual(ssr_signals["render_strategy"], "hybrid_ssr_hydration")

    def test_short_confirmation_page_is_not_invented_as_geo_block(self) -> None:
        html = """
        <html><head><title>Спасибо</title></head><body>
          <div class="wrapper">%s</div>
          <main><h1>Ваша заявка отправлена</h1>
            <p>Мы скоро свяжемся с вами.</p></main>
        </body></html>
        """ % ("<span></span>" * 500)
        signals = extract_text_signals(html, "text/html")
        self.assertTrue(signals["looks_disproportionate_wrapper"])
        self.assertFalse(signals["looks_like_geo_block"])


class TechnicalProbeOutcomeTests(unittest.TestCase):
    @staticmethod
    def _probe(**overrides):
        values = {
            "error_class": None,
            "http_status": 200,
            "challenge_detected": False,
            "content_signals": {},
            "content_extractable_text_length": 500,
            "body_looks_empty": False,
            "response_size_bytes": 10_000,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_transport_and_transient_http_failures_are_unknown(self) -> None:
        self.assertEqual(
            _probe_access_outcome(
                self._probe(
                    error_class="connect_timeout",
                    http_status=None,
                )
            ),
            "unknown",
        )
        self.assertEqual(
            _probe_access_outcome(self._probe(http_status=503)),
            "unknown",
        )
        self.assertEqual(
            _probe_access_outcome(self._probe(http_status=429)),
            "unknown",
        )

    def test_confirmed_http_block_is_not_unknown(self) -> None:
        self.assertEqual(
            _probe_access_outcome(self._probe(http_status=403)),
            "blocked",
        )
        self.assertEqual(
            _probe_access_outcome(
                self._probe(http_status=503, challenge_detected=True)
            ),
            "blocked",
        )

    def test_large_client_and_redirect_shells_are_not_readable_content(self) -> None:
        for signal in (
            {"looks_like_spa_shell": True},
            {"looks_like_redirect_shell": True},
            {"render_strategy": "client_rendered_shell"},
        ):
            with self.subTest(signal=signal):
                self.assertEqual(
                    _probe_access_outcome(
                        self._probe(
                            content_signals=signal,
                            content_extractable_text_length=0,
                            body_looks_empty=False,
                        )
                    ),
                    "blocked",
                )

    def test_server_content_remains_available(self) -> None:
        self.assertEqual(
            _probe_access_outcome(self._probe()),
            "available",
        )

    def test_truncated_readable_prefix_is_not_compared_to_full_control(self) -> None:
        self.assertEqual(
            _probe_access_outcome(
                self._probe(
                    response_size_bytes=768 * 1024,
                    content_extractable_text_length=250,
                ),
                baseline_length=18_000,
            ),
            "available",
        )
        self.assertEqual(
            _probe_access_outcome(
                self._probe(
                    content_signals={"_body_truncated": True},
                    content_extractable_text_length=0,
                    body_looks_empty=True,
                ),
                baseline_length=18_000,
            ),
            "unknown",
        )


class DeterministicMetricTests(unittest.TestCase):
    def test_evidence_must_be_an_exact_contiguous_raw_substring(self) -> None:
        raw = "**Jois**\nквартиры с зелёными террасами"

        self.assertTrue(_evidence_is_literal(raw, "Jois"))
        self.assertTrue(
            _evidence_is_literal(raw, "квартиры с зелёными террасами")
        )
        self.assertFalse(_evidence_is_literal(raw, "JOIS"))
        self.assertFalse(
            _evidence_is_literal(
                raw,
                "**Jois** квартиры с зелёными террасами",
            )
        )
        # Literal containment and entity completeness are separate guards:
        # a short slice is technically raw, but cannot prove a declared alias.
        self.assertTrue(_evidence_is_literal(raw, "Joi"))
        self.assertFalse(
            _evidence_contains_complete_alias(
                "Joi",
                [("Jois", "standalone")],
            )
        )

    def test_only_profile_confirmed_owners_extend_attribution_scope(self) -> None:
        profile = {
            "brand_name": "ЖК «Джойс»",
            "brand_aliases": ["JOIS"],
            "entity_scope": [
                {
                    "canonical_name": "MR Group",
                    "aliases": ["МР Групп"],
                    "entity_type": "business_unit",
                    "relationship": "operated_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": "Онлайн-покупка MR Group",
                    "aliases": ["онлайн-покупка"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                },
                {
                    "canonical_name": "Uncertain Owner",
                    "aliases": [],
                    "entity_type": "business_unit",
                    "relationship": "operated_by",
                    "commercially_relevant": True,
                    "confidence": "low",
                },
            ],
        }

        aliases = _attribution_owner_aliases(
            profile,
            {"target_aliases": ["ЖК JOIS"], "entities": []},
        )

        self.assertIn("JOIS", aliases)
        self.assertIn("MR Group", aliases)
        self.assertNotIn("Онлайн-покупка MR Group", aliases)
        self.assertNotIn("Uncertain Owner", aliases)

    def test_contextual_catalog_does_not_synthesize_generic_token_aliases(
        self,
    ) -> None:
        entries = _entity_alias_entries(
            {
                "canonical_name": "Онлайн-покупка MR Group",
                "aliases": [
                    {
                        "value": "онлайн-покупка",
                        "match_policy": "requires_target_attribution",
                    }
                ],
                "mention_policy": "requires_target_attribution",
            },
            {"brand_name": "ЖК «Джойс»"},
            excluded_aliases=["MR Group"],
        )
        values = {alias for alias, _policy in entries}

        self.assertIn("Онлайн-покупка MR Group", values)
        self.assertIn("онлайн-покупка", values)
        self.assertNotIn("Group", values)
        self.assertNotIn("Онлайн", values)
        self.assertNotIn("покупка", values)

    def test_markdown_owner_scope_repairs_jois_portfolio_evidence(self) -> None:
        profile = {
            "brand_name": "ЖК «Джойс»",
            "brand_aliases": ["JOIS", "ЖК JOIS"],
            "entity_scope": [
                {
                    "canonical_name": "MR Group",
                    "aliases": [],
                    "entity_type": "business_unit",
                    "relationship": "operated_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        cases = [
            (
                "Квартиры с приватными террасами ЖК «Джойс»",
                ["квартиры с приватными террасами", "зелёными террасами"],
                (
                    "- JOIS (MR Group). В продаже есть квартиры с "
                    "приватными террасами."
                ),
            ),
            (
                "Двухуровневые пентхаусы ЖК «Джойс»",
                ["двухуровневые пентхаусы"],
                (
                    "**2. ЖК «Джойс» (JOIS)**\n"
                    "- Локация: Москва\n"
                    "- Двухуровневые пентхаусы с потолками до 6 метров"
                ),
            ),
            (
                "MR Base",
                ["White Box", "предчистовая отделка"],
                (
                    "### 3. ЖК JOIS (Застройщик: MR Group)\n"
                    "* **Репутация застройщика:** **MR Group** — лидер рынка.\n\n"
                    "**Почему выигрывает:**\n"
                    "* **Архитектура:** башни с террасами.\n"
                    "* **Качество отделки:** Квартиры сдаются в "
                    "предчистовой отделке White Box."
                ),
            ),
            (
                "Рассрочка 0% от MR Group",
                ["Рассрочка 0%"],
                (
                    "#### 3. MR Group (сегмент MR Premium)\n"
                    "* **JOIS**, SLAVA, MИRА:\n"
                    "  * *Условия:* Рассрочка 0% с ПВ от 30%."
                ),
            ),
            (
                "Онлайн-покупка MR Group",
                ["онлайн-покупки", "мобильное приложение"],
                (
                    "- **MR Group** — у компании есть собственный сценарий "
                    "«онлайн-покупки» через **мобильное приложение**."
                ),
            ),
        ]

        for canonical, aliases, raw in cases:
            with self.subTest(canonical=canonical):
                scoped_profile = {
                    **profile,
                    "entity_scope": [
                        *profile["entity_scope"],
                        {
                            "canonical_name": canonical,
                            "aliases": aliases,
                            "entity_type": "service",
                            "relationship": "offered_by",
                            "commercially_relevant": True,
                            "confidence": "high",
                        },
                    ],
                }
                catalog = {
                    "target_aliases": ["JOIS", "ЖК JOIS"],
                    "entities": [
                        {
                            "canonical_name": canonical,
                            "aliases": [
                                {
                                    "value": alias,
                                    "match_policy": (
                                        "requires_target_attribution"
                                    ),
                                }
                                for alias in aliases
                            ],
                            "category": "target",
                            "target_relationship": "portfolio_entity",
                            "commercially_relevant": True,
                            "mention_policy": "requires_target_attribution",
                        }
                    ],
                }
                reconciled = _reconcile_annotation(
                    {
                        "answer_id": 1,
                        "valid": True,
                        "target_mentioned": False,
                        "target_position": None,
                        "target_role": "absent",
                        "sentiment": "unknown",
                        "entity_mentions": [],
                        "brand_answer": {
                            "directness": "not_applicable",
                            "specificity": "not_applicable",
                            "supported_facets": [],
                            "contradictions": [],
                        },
                        "evidence": ["fabricated summary"],
                        "uncertainties": [],
                    },
                    {
                        "answer": raw,
                        "answer_sha256": "hash",
                        "answer_model": "provider/model",
                    },
                    scoped_profile,
                    catalog,
                )
                mention = next(
                    item
                    for item in reconciled["entity_mentions"]
                    if item["canonical_name"] == canonical
                )
                self.assertTrue(mention["attributed_to_target"])
                self.assertIn(mention["evidence"], raw)
                self.assertEqual(reconciled["evidence"], [])

    def test_answer_claims_cannot_expand_profile_confirmed_portfolio(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "products": ["Campaign 360"],
            "entity_scope": [],
        }
        catalog = {
            "target_aliases": ["Example"],
            "entities": [
                {
                    "canonical_name": "Campaign 360",
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
                {
                    "canonical_name": "Invented Suite",
                    "aliases": [],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
            ],
        }

        scoped = _scope_entity_catalog_to_profile(catalog, profile)

        confirmed, rejected = scoped["entities"]
        self.assertTrue(confirmed["_profile_membership_confirmed"])
        self.assertEqual(confirmed["category"], "target")
        self.assertFalse(rejected["_profile_membership_confirmed"])
        self.assertEqual(rejected["category"], "other")
        self.assertEqual(rejected["target_relationship"], "unrelated")
        self.assertFalse(rejected["commercially_relevant"])
        self.assertNotIn("_profile_membership_confirmed", catalog["entities"][0])

    def test_profile_target_wins_same_name_competitor_collision(self) -> None:
        """A later competitor duplicate must not shadow a site-owned unit."""

        profile = {
            "brand_name": "Northstar Group",
            "brand_aliases": [],
            "products": [],
            "entity_scope": [
                {
                    "canonical_name": (
                        "Облачная платформа аналитики "
                        "(бизнес-юнит Northstar Group)"
                    ),
                    "aliases": [
                        "Orbit Cloud",
                        "облачная платформа аналитики",
                    ],
                    "relationship": "owned_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        catalog = {
            "target_aliases": ["Northstar Group"],
            "entities": [
                {
                    "canonical_name": "Orbit Cloud",
                    "aliases": [
                        "orbitcloud.io",
                        {
                            "value": "облачная платформа аналитики",
                            "match_policy": "requires_target_attribution",
                        },
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
                {
                    "canonical_name": "Orbit Cloud",
                    "aliases": [],
                    "category": "competitor",
                    "target_relationship": "competitor",
                    "commercially_relevant": True,
                    "mention_policy": "standalone",
                },
            ],
        }
        rows = [
            {
                "answer_id": 81,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 17,
                "prompt_key": "nav-cloud",
                "intent_class": "NAV",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Для этой задачи подходит Orbit Cloud.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "Orbit Cloud",
                            "position": 1,
                            "role": "recommended",
                            # The raw answer does not name the parent. Site
                            # membership plus a standalone product name is the
                            # portfolio proof; this flag correctly stays false.
                            "attributed_to_target": False,
                            "evidence": "Orbit Cloud",
                        }
                    ],
                },
            }
        ]

        scoped = _scope_entity_catalog_to_profile(catalog, profile)
        metrics = _compute_metrics(rows, profile, catalog)

        self.assertEqual(len(scoped["entities"]), 1)
        entity = scoped["entities"][0]
        self.assertEqual(entity["category"], "target")
        self.assertEqual(entity["target_relationship"], "portfolio_entity")
        self.assertTrue(entity["_profile_membership_confirmed"])
        self.assertEqual(
            metrics["portfolio_visibility"]["web"]["mention_count"],
            1,
        )
        self.assertEqual(
            metrics["portfolio_visibility"]["web"]["mentioned_entities"],
            [
                {
                    "name": "Orbit Cloud",
                    "answer_count": 1,
                    "answer_rate": 100.0,
                }
            ],
        )
        self.assertFalse(
            any(
                item["name"] == "Orbit Cloud"
                and item["relationship"] == "competitor"
                for item in metrics["competitors"]
            )
        )

    def test_invalid_answers_do_not_affect_rates_competitors_or_pairs(
        self,
    ) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [],
                },
            },
            {
                "answer_id": 2,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": False,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [],
                },
            },
            {
                "answer_id": 3,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 2,
                "prompt_key": "u-2",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": False,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [
                        {
                            "canonical_name": "Phantom",
                            "position": 2,
                            "role": "recommended",
                        }
                    ],
                },
            },
            {
                "answer_id": 4,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 2,
                "prompt_key": "u-2",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [],
                },
            },
        ]

        metrics = _compute_metrics(
            rows,
            {"brand_name": "Цель"},
            {
                "entities": [
                    {
                        "canonical_name": "Phantom",
                        "category": "competitor",
                    }
                ]
            },
        )

        web = metrics["parent_discovery"]["web"]
        self.assertEqual(web["expected_answers"], 2)
        self.assertEqual(web["annotated_answers"], 2)
        self.assertEqual(web["valid_answers"], 1)
        self.assertEqual(web["coverage_rate"], 50.0)
        self.assertEqual(web["mention_rate"], 0.0)
        self.assertEqual(web["mention_count"], 0)
        self.assertEqual(web["data_state"], "limited")
        self.assertEqual(
            [row["name"] for row in metrics["competitors"]],
            ["Цель"],
        )
        self.assertEqual(metrics["competitors"][0]["total_answers"], 1)
        self.assertEqual(metrics["paired_web_lift"]["n_pairs"], 0)
        self.assertIsNone(
            metrics["paired_web_lift"]["parent"]["score_lift"]
        )
        self.assertIsNone(metrics["knowledge_gap"])
        self.assertEqual(metrics["quality"]["state"], "limited")
        self.assertEqual(metrics["quality"]["coverage_rate"], 50.0)

    def test_entity_ranking_deduplicates_case_variants_per_answer(
        self,
    ) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "MGCom",
                            "position": 1,
                            "role": "mentioned",
                        },
                        {
                            "canonical_name": "mgcom",
                            "position": 2,
                            "role": "mentioned",
                        },
                    ],
                },
            },
        ]

        metrics = _compute_metrics(
            rows,
            {"brand_name": "Цель"},
            {
                "entities": [
                    {
                        "canonical_name": "MGCom",
                        "category": "competitor",
                    }
                ]
            },
        )

        self.assertEqual(
            [row["name"] for row in metrics["competitors"]],
            ["Цель", "MGCom"],
        )
        self.assertEqual(metrics["competitors"][1]["mention_count"], 1)
        self.assertEqual(metrics["competitors"][1]["total_answers"], 1)
        self.assertEqual(metrics["competitors"][1]["mention_share"], 100.0)

    def test_no_valid_answers_leave_metrics_unknown_with_zero_coverage(
        self,
    ) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": False,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [
                        {
                            "canonical_name": "Phantom",
                            "position": 2,
                            "role": "recommended",
                        }
                    ],
                },
            },
            {
                "answer_id": 2,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 2,
                "prompt_key": "u-2",
                "intent_class": "E",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {},
            },
            {
                "answer_id": 3,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 3,
                "prompt_key": "b-1",
                "intent_class": "TR",
                "role": "brand_diagnostic",
                "status": "completed",
                "citations_count": 1,
                "annotation": {
                    "valid": False,
                    "target_mentioned": True,
                    "target_position": None,
                    "target_role": "mentioned",
                    "sentiment": "positive",
                    "entity_mentions": [],
                    "brand_answer": {
                        "directness": "direct",
                        "specificity": "specific",
                        "supported_facets": ["identity"],
                        "contradictions": [],
                    },
                },
            },
        ]

        metrics = _compute_metrics(
            rows,
            {"brand_name": "Цель"},
            {
                "entities": [
                    {
                        "canonical_name": "Phantom",
                        "category": "competitor",
                    }
                ]
            },
        )

        discovery = metrics["parent_discovery"]["web"]
        self.assertEqual(discovery["annotated_answers"], 1)
        self.assertEqual(discovery["valid_answers"], 0)
        self.assertEqual(discovery["coverage_rate"], 0.0)
        self.assertEqual(discovery["data_state"], "limited")
        self.assertEqual(discovery["state"], "unknown")
        self.assertIsNone(discovery["score"])
        self.assertIsNone(discovery["mention_rate"])
        self.assertIsNone(discovery["mention_count"])
        self.assertEqual(metrics["competitors"], [])

        knowledge = metrics["brand_knowledge"]["web"]
        self.assertEqual(knowledge["annotated_answers"], 1)
        self.assertEqual(knowledge["valid_answers"], 0)
        self.assertEqual(knowledge["coverage_rate"], 0.0)
        self.assertEqual(knowledge["data_state"], "limited")
        self.assertEqual(knowledge["state"], "unknown")
        self.assertIsNone(knowledge["answer_rate"])
        self.assertIsNone(knowledge["answer_count"])
        self.assertIsNone(knowledge["specific_rate"])
        self.assertIsNone(knowledge["specific_count"])

        self.assertEqual(metrics["paired_web_lift"]["n_pairs"], 0)
        self.assertEqual(metrics["quality"]["state"], "limited")
        self.assertEqual(metrics["quality"]["coverage_rate"], 0.0)

    def test_parent_mention_without_portfolio_entity_stays_parent_only(
        self,
    ) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [
                        {
                            "canonical_name": "RW+",
                            "position": 1,
                            "role": "recommended",
                        }
                    ],
                },
            },
            {
                "answer_id": 2,
                "mode": "web",
                "provider_key": "gemini",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [],
                },
            },
            {
                "answer_id": 3,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [
                        {
                            "canonical_name": "RW+",
                            "position": 1,
                            "role": "recommended",
                        }
                    ],
                },
            },
        ]
        metrics = _compute_metrics(
            rows,
            {"brand_name": "RW+"},
            {
                "entities": [
                    {
                        "canonical_name": "RW+",
                        "category": "target",
                        "target_relationship": "exact_target",
                        "commercially_relevant": True,
                    }
                ]
            },
        )

        self.assertEqual(
            metrics["parent_discovery"]["web"]["mention_count"],
            1,
        )
        self.assertEqual(metrics["parent_discovery"]["web"]["score"], 50.0)
        portfolio = metrics["portfolio_visibility"]["web"]
        self.assertIsNone(portfolio["mention_count"])
        self.assertIsNone(portfolio["score"])
        self.assertEqual(portfolio["data_state"], "unavailable")
        self.assertEqual(
            portfolio["unavailable_reason"],
            "target_portfolio_unconfirmed",
        )
        self.assertEqual(portfolio["mentioned_entities"], [])
        self.assertEqual(metrics["portfolio_scope"]["state"], "unavailable")
        for mode in ("web", "memory"):
            overall = metrics["portfolio_visibility"][mode]
            self.assertIsNone(overall["score"])
            self.assertIsNone(overall["mention_rate"])
            self.assertIsNone(overall["mention_count"])
            self.assertEqual(overall["data_state"], "unavailable")
            self.assertEqual(
                overall["unavailable_reason"],
                "target_portfolio_unconfirmed",
            )
            self.assertIs(metrics[mode], overall)

        chatgpt = next(
            provider
            for provider in metrics["providers"]
            if provider["name"] == "ChatGPT"
        )
        self.assertEqual(chatgpt["parent_discovery"]["score"], 100.0)
        self.assertIsNone(chatgpt["portfolio_capture"]["score"])
        for provider in metrics["providers"]:
            self.assertIsNone(provider["portfolio_capture"]["score"])
            self.assertEqual(
                provider["portfolio_capture"]["unavailable_reason"],
                "target_portfolio_unconfirmed",
            )
        intent_i = next(
            item for item in metrics["intents"] if item["intent"] == "I"
        )
        self.assertIsNone(intent_i["portfolio_capture"]["score"])
        for intent in metrics["intents"]:
            self.assertIsNone(intent["portfolio_capture"]["score"])
            self.assertEqual(
                intent["portfolio_capture"]["unavailable_reason"],
                "target_portfolio_unconfirmed",
            )
        self.assertEqual(metrics["paired_web_lift"]["n_pairs"], 1)
        self.assertEqual(
            metrics["paired_web_lift"]["portfolio"]["score_lift"],
            None,
        )
        paired_portfolio = metrics["paired_web_lift"]["portfolio"]
        self.assertEqual(paired_portfolio["data_state"], "unavailable")
        self.assertEqual(
            paired_portfolio["unavailable_reason"],
            "target_portfolio_unconfirmed",
        )
        for mode in ("web", "memory"):
            self.assertIsNone(paired_portfolio[mode]["score"])
            self.assertEqual(
                paired_portfolio[mode]["unavailable_reason"],
                "target_portfolio_unconfirmed",
            )
        self.assertIsNone(metrics["knowledge_gap"])
        self.assertIsNone(metrics["model_consistency"])
        self.assertFalse(
            any(
                row.get("relationship") == "portfolio"
                for row in metrics["competitors"]
            )
        )
        self.assertEqual(
            [row["relationship"] for row in metrics["competitors"]],
            ["parent"],
        )

    def test_portfolio_metrics_count_only_portfolio_entity_mentions(
        self,
    ) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "T",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "RW+ и Realweb названы среди вариантов.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": 1,
                    "target_role": "recommended",
                    "sentiment": "positive",
                    "entity_mentions": [
                        {
                            "canonical_name": "RW+",
                            "position": 1,
                            "role": "recommended",
                        },
                        {
                            "canonical_name": "Realweb",
                            "position": 4,
                            "role": "mentioned",
                        },
                    ],
                },
            }
        ]
        metrics = _compute_metrics(
            rows,
            {
                "brand_name": "RW+",
                "entity_scope": [
                    {
                        "canonical_name": "Realweb",
                        "aliases": [],
                        "relationship": "owned_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    }
                ],
            },
            {
                "entities": [
                    {
                        "canonical_name": "RW+",
                        "category": "target",
                        "target_relationship": "exact_target",
                        "commercially_relevant": True,
                    },
                    {
                        "canonical_name": "Realweb",
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    },
                ]
            },
        )

        self.assertEqual(metrics["parent_discovery"]["web"]["score"], 100.0)
        portfolio = metrics["portfolio_visibility"]["web"]
        self.assertEqual(portfolio["mention_count"], 1)
        self.assertEqual(portfolio["mention_rate"], 100.0)
        self.assertEqual(portfolio["top3_count"], 0)
        self.assertEqual(portfolio["recommendation_count"], 0)
        self.assertEqual(portfolio["score"], 25.0)
        self.assertEqual(
            portfolio["mentioned_entities"],
            [
                {
                    "name": "Realweb",
                    "answer_count": 1,
                    "answer_rate": 100.0,
                }
            ],
        )
        self.assertEqual(
            [row["relationship"] for row in metrics["competitors"]],
            ["parent", "portfolio"],
        )

    def test_metrics_are_computed_from_atomic_annotations(self) -> None:
        rows = []
        for index, intent in enumerate(("I", "E", "T", "NB", "NAV", "TR"), start=1):
            rows.append(
                {
                    "answer_id": index,
                    "mode": "web",
                    "provider_key": "openai",
                    "prompt_id": index,
                    "prompt_key": f"web-{intent}",
                    "intent_class": intent,
                    "role": "unbranded_discovery",
                    "annotation": {
                        "valid": True,
                        "target_mentioned": True,
                        "target_position": 1,
                        "target_role": "recommended",
                        "sentiment": "positive",
                        "entity_mentions": [
                            {
                                "canonical_name": "Конкурент",
                                "position": 2,
                                "role": "recommended",
                            }
                        ],
                    },
                }
            )
            rows.append(
                {
                    "answer_id": index + 10,
                    "mode": "memory",
                    "provider_key": "openai",
                    "prompt_id": index,
                    "prompt_key": f"memory-{intent}",
                    "intent_class": intent,
                    "role": "unbranded_discovery",
                    "annotation": {
                        "valid": True,
                        "target_mentioned": False,
                        "target_position": None,
                        "target_role": "absent",
                        "sentiment": "unknown",
                        "entity_mentions": [],
                    },
                }
            )
        metrics = _compute_metrics(
            rows,
            {"brand_name": "Цель"},
            {
                "entities": [
                    {
                        "canonical_name": "Конкурент",
                        "category": "competitor",
                    }
                ]
            },
        )
        self.assertEqual(metrics["parent_discovery"]["web"]["score"], 100.0)
        self.assertIsNone(metrics["web"]["score"])
        self.assertIsNone(metrics["memory"]["score"])
        self.assertIsNone(metrics["knowledge_gap"])
        self.assertEqual(metrics["web"]["data_state"], "unavailable")
        self.assertEqual(
            metrics["web"]["unavailable_reason"],
            "target_portfolio_unconfirmed",
        )
        self.assertEqual(metrics["portfolio_scope"]["state"], "unavailable")
        self.assertEqual(metrics["portfolio_scope"]["confirmed_entities"], 0)
        self.assertEqual(metrics["competitors"][0]["name"], "Цель")
        self.assertEqual(metrics["competitors"][1]["name"], "Конкурент")

    def test_brand_knowledge_facets_have_stable_tie_order(self) -> None:
        metrics = _compute_metrics(
            [
                {
                    "answer_id": 1,
                    "mode": "web",
                    "provider_key": "openai",
                    "prompt_id": 1,
                    "prompt_key": "brand-1",
                    "intent_class": "NB",
                    "role": "brand_diagnostic",
                    "status": "completed",
                    "answer_text": "Конкретный ответ о бренде.",
                    "citations_count": 0,
                    "annotation": {
                        "valid": True,
                        "target_mentioned": True,
                        "target_position": None,
                        "target_role": "mentioned",
                        "sentiment": "neutral",
                        "entity_mentions": [],
                        "brand_answer": {
                            "directness": "direct",
                            "specificity": "specific",
                            "supported_facets": [
                                "portfolio",
                                "offering",
                                "identity",
                            ],
                            "contradictions": [],
                        },
                    },
                }
            ],
            {"brand_name": "Example"},
            {"target_aliases": ["Example"], "entities": []},
        )

        self.assertEqual(
            [
                item["name"]
                for item in metrics["brand_knowledge"]["web"]["facets"]
            ],
            ["identity", "offering", "portfolio"],
        )

    def test_legacy_memory_observation_is_counted_but_stays_limited(self) -> None:
        metrics = _compute_metrics(
            [
                {
                    "answer_id": 1,
                    "mode": "memory",
                    "provider_key": "openai",
                    "prompt_id": 1,
                    "prompt_key": "brand-memory",
                    "intent_class": "I",
                    "role": "brand_diagnostic",
                    "status": "completed",
                    "answer_text": "Example — аналитическая платформа.",
                    "citations_count": 0,
                    "metric_eligible": True,
                    "context_eligible": False,
                    "metric_evidence_state": "legacy_observational",
                    "web_attestation_reason": (
                        LEGACY_MEMORY_OBSERVATION_REASON
                    ),
                    "annotation": {
                        "valid": True,
                        "target_mentioned": True,
                        "target_position": None,
                        "target_role": "mentioned",
                        "sentiment": "neutral",
                        "entity_mentions": [],
                        "brand_answer": {
                            "directness": "direct",
                            "specificity": "specific",
                            "supported_facets": ["identity", "offering"],
                            "contradictions": [],
                        },
                    },
                }
            ],
            {"brand_name": "Example"},
            {"target_aliases": ["Example"], "entities": []},
        )

        memory = metrics["brand_knowledge"]["memory"]
        self.assertEqual(memory["specific_rate"], 100.0)
        self.assertEqual(memory["valid_answers"], 1)
        self.assertEqual(memory["data_state"], "limited")
        self.assertEqual(memory["evidence_state"], "legacy_observational")
        self.assertEqual(memory["observational_answers"], 1)
        self.assertEqual(memory["strictly_attested_answers"], 0)
        self.assertFalse(memory["strict_no_web_verified"])
        self.assertEqual(metrics["quality"]["state"], "limited")
        self.assertEqual(
            metrics["quality"]["legacy_observational_answers"],
            1,
        )

    def test_parent_portfolio_knowledge_and_paired_lift_are_separate(self) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Realweb — один из рекомендуемых вариантов.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "Realweb",
                            "position": None,
                            "role": "mentioned",
                        }
                    ],
                },
            },
            {
                "answer_id": 2,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [],
                },
            },
            {
                "answer_id": 3,
                "mode": "web",
                "provider_key": "perplexity",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [],
                },
            },
            {
                "answer_id": 4,
                "mode": "memory",
                "provider_key": "openai",
                "prompt_id": 2,
                "prompt_key": "b-1",
                "intent_class": "TR",
                "role": "brand_diagnostic",
                "status": "completed",
                "citations_count": 0,
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": None,
                    "target_role": "mentioned",
                    "sentiment": "neutral",
                    "entity_mentions": [],
                    "brand_answer": {
                        "directness": "direct",
                        "specificity": "specific",
                        "supported_facets": ["identity", "portfolio"],
                        "contradictions": [],
                    },
                },
            },
        ]
        metrics = _compute_metrics(
            rows,
            {
                "brand_name": "RW+",
                "entity_scope": [
                    {
                        "canonical_name": "Realweb",
                        "aliases": [],
                        "relationship": "owned_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    }
                ],
            },
            {
                "entities": [
                    {
                        "canonical_name": "Realweb",
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    }
                ]
            },
        )
        self.assertEqual(metrics["parent_discovery"]["web"]["score"], 0.0)
        self.assertEqual(metrics["portfolio_visibility"]["web"]["score"], 12.5)
        self.assertEqual(
            metrics["portfolio_visibility"]["web"]["mentioned_entities"],
            [
                {
                    "name": "Realweb",
                    "answer_count": 1,
                    "answer_rate": 50.0,
                }
            ],
        )
        self.assertEqual(metrics["brand_knowledge"]["memory"]["specific_rate"], 100.0)
        self.assertEqual(metrics["paired_web_lift"]["n_pairs"], 1)
        self.assertEqual(
            metrics["paired_web_lift"]["portfolio"]["score_lift"],
            25.0,
        )
        self.assertEqual(metrics["competitors"][0]["name"], "RW+")
        self.assertEqual(metrics["competitors"][0]["mention_count"], 0)
        self.assertEqual(metrics["competitors"][1]["relationship"], "portfolio")

        observational_rows = copy.deepcopy(rows)
        observational_rows[1].update(
            {
                "metric_eligible": True,
                "context_eligible": False,
                "metric_evidence_state": "legacy_observational",
                "web_attestation_reason": LEGACY_MEMORY_OBSERVATION_REASON,
            }
        )
        observational = _compute_metrics(
            observational_rows,
            {
                "brand_name": "RW+",
                "entity_scope": [
                    {
                        "canonical_name": "Realweb",
                        "aliases": [],
                        "relationship": "owned_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    }
                ],
            },
            {
                "entities": [
                    {
                        "canonical_name": "Realweb",
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    }
                ]
            },
        )
        paired = observational["paired_web_lift"]
        self.assertEqual(paired["data_state"], "limited")
        self.assertFalse(paired["causal_interpretation_allowed"])
        self.assertIsNone(paired["parent"]["score_lift"])
        self.assertEqual(paired["parent"]["observed_difference"], 0.0)
        self.assertEqual(
            paired["parent"]["observed_difference_metric"],
            "mention_rate_percentage_points",
        )
        self.assertIsNone(paired["portfolio"]["score_lift"])
        self.assertEqual(
            paired["portfolio"]["observed_difference"],
            100.0,
        )
        self.assertEqual(
            paired["portfolio"]["observed_difference_metric"],
            "mention_rate_percentage_points",
        )
        self.assertIsNone(observational["knowledge_gap"])
        self.assertEqual(observational["knowledge_gap_state"], "unknown")

    def test_profile_must_confirm_each_one_word_contextual_alias(self) -> None:
        profile = {
            "brand_name": "Realweb",
            "entity_scope": [
                {
                    "canonical_name": "DOOH Realweb",
                    "aliases": ["programmatic DOOH"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "DOOH Realweb",
                    "aliases": [
                        {
                            "value": "аналитика",
                            "match_policy": "requires_target_attribution",
                        }
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ]
        }
        source = {
            "answer_id": 120,
            "valid": True,
            "target_mentioned": True,
            "target_position": None,
            "target_role": "mentioned",
            "sentiment": "neutral",
            "entity_mentions": [
                {
                    "canonical_name": "DOOH Realweb",
                    "position": None,
                    "role": "mentioned",
                    "attributed_to_target": True,
                    "evidence": "Realweb предлагает аналитику",
                }
            ],
            "brand_answer": {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
            "evidence": [],
            "uncertainties": [],
        }

        reconciled = _reconcile_annotation(
            source,
            {
                "answer": "Realweb предлагает аналитику.",
                "answer_sha256": "hash",
                "answer_model": "provider/model",
                "scenario_role": "unbranded_discovery",
            },
            profile,
            catalog,
        )

        self.assertFalse(
            any(
                mention["canonical_name"] == "DOOH Realweb"
                for mention in reconciled["entity_mentions"]
            )
        )

    def test_service_word_before_solutions_is_not_a_competing_owner(self) -> None:
        answer = (
            "1. Realweb (Санкт-Петербург, Москва)\n"
            "- Сильная сторона: одно из наиболее универсальных "
            "digital-агентств на рынке. Хорошо сочетает "
            "performance-маркетинг, медиазакупку, аналитику, "
            "собственные технологические решения и стратегическую экспертизу."
        )
        entity = {
            "canonical_name": "Исследования и аналитика Realweb",
            "aliases": [
                {
                    "value": "аналитика",
                    "match_policy": "requires_target_attribution",
                }
            ],
            "mention_policy": "requires_target_attribution",
        }

        self.assertTrue(
            _portfolio_entity_is_grounded(
                answer,
                entity,
                ["Realweb"],
                {
                    "canonical_name": entity["canonical_name"],
                    "attributed_to_target": True,
                    "evidence": answer,
                },
            )
        )
        self.assertFalse(
            _portfolio_entity_is_grounded(
                "### Realweb\n- Сильная сторона: Okkam предлагает аналитику.",
                entity,
                ["Realweb"],
                {
                    "canonical_name": entity["canonical_name"],
                    "attributed_to_target": True,
                    "evidence": (
                        "### Realweb\n- Сильная сторона: "
                        "Okkam предлагает аналитику."
                    ),
                },
            )
        )

    def test_branded_not_applicable_is_repaired_from_literal_portfolio_fact(
        self,
    ) -> None:
        profile = {
            "brand_name": "Realweb",
            "brand_aliases": ["Реалвеб"],
            "products": ["цифровая наружная реклама"],
            "entity_scope": [
                {
                    "canonical_name": "DOOH Realweb",
                    "aliases": ["цифровая наружная реклама"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        catalog = {
            "entities": [
                {
                    "canonical_name": "DOOH Realweb",
                    "aliases": [
                        {
                            "value": "DOOH",
                            "match_policy": "requires_target_attribution",
                        }
                    ],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ]
        }
        raw = (
            "Сравнение агентств.\n"
            "- **Realweb** — закупка DOOH через рекламные платформы.\n"
            "- iConText — отдельная строка сравнения."
        )
        source = {
            "answer_id": 121,
            "valid": True,
            "target_mentioned": True,
            "target_position": None,
            "target_role": "mentioned",
            "sentiment": "positive",
            "entity_mentions": [],
            "brand_answer": {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
            "evidence": [],
            "uncertainties": [],
        }

        branded = _reconcile_annotation(
            source,
            {
                "answer": raw,
                "answer_sha256": "hash",
                "answer_model": "provider/model",
                "scenario_role": "brand_diagnostic",
            },
            profile,
            catalog,
        )

        dooh = next(
            item
            for item in branded["entity_mentions"]
            if item["canonical_name"] == "DOOH Realweb"
        )
        self.assertTrue(dooh["attributed_to_target"])
        self.assertIn(dooh["evidence"], raw)
        self.assertIn("Realweb", dooh["evidence"])
        self.assertIn("DOOH", dooh["evidence"])
        self.assertEqual(branded["brand_answer"]["directness"], "partial")
        self.assertEqual(branded["brand_answer"]["specificity"], "specific")
        self.assertEqual(
            branded["brand_answer"]["supported_facets"],
            ["offering", "portfolio"],
        )
        self.assertTrue(
            any(
                "недопустимый not_applicable" in note
                for note in branded["_reconciliation_notes"]
            )
        )

        unbranded = _reconcile_annotation(
            {
                **source,
                "brand_answer": {
                    "directness": "direct",
                    "specificity": "specific",
                    "supported_facets": ["offering"],
                    "contradictions": [],
                },
            },
            {
                "answer": raw,
                "answer_sha256": "hash",
                "answer_model": "provider/model",
                "scenario_role": "unbranded_discovery",
            },
            profile,
            catalog,
        )
        self.assertEqual(
            unbranded["brand_answer"],
            {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
        )

        competitor_only = _reconcile_annotation(
            source,
            {
                "answer": (
                    "Realweb участвует в сравнении.\n"
                    "- **iConText** — закупка DOOH через свою платформу."
                ),
                "answer_sha256": "hash",
                "answer_model": "provider/model",
                "scenario_role": "brand_diagnostic",
            },
            profile,
            catalog,
        )
        competitor_dooh = next(
            (
                item
                for item in competitor_only["entity_mentions"]
                if item["canonical_name"] == "DOOH Realweb"
            ),
            None,
        )
        self.assertTrue(
            competitor_dooh is None
            or competitor_dooh["attributed_to_target"] is False
        )
        self.assertEqual(
            competitor_only["brand_answer"]["specificity"],
            "not_applicable",
        )

    def test_brand_diagnostic_not_applicable_reaches_critic_warning(self) -> None:
        warnings = _deterministic_annotation_warnings(
            profile={"brand_name": "Example"},
            catalog={"entities": []},
            rows=[
                {
                    "answer_id": 121,
                    "mode": "web",
                    "role": "brand_diagnostic",
                    "status": "completed",
                    "answer_text": "Example — конкретное описание.",
                    "annotation": {
                        "valid": True,
                        "brand_answer": {
                            "directness": "not_applicable",
                            "specificity": "not_applicable",
                            "supported_facets": [],
                            "contradictions": [],
                        },
                    },
                }
            ],
        )

        self.assertEqual(len(warnings), 1)
        self.assertEqual(
            warnings[0]["code"],
            "brand_knowledge_false_negative",
        )
        self.assertEqual(warnings[0]["severity"], "important")
        self.assertEqual(warnings[0]["answer_ids"], [121])

    def test_entity_graph_rules_are_independent_of_the_rw_plus_case(self) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "T",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Orbit Cloud и BluePeak подходят для задачи.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "Orbit Cloud",
                            "position": 1,
                            "role": "recommended",
                        },
                        {
                            "canonical_name": "BluePeak",
                            "position": 2,
                            "role": "recommended",
                        },
                    ],
                },
            }
        ]
        metrics = _compute_metrics(
            rows,
            {
                "brand_name": "Northstar Group",
                "entity_scope": [
                    {
                        "canonical_name": "Orbit Cloud",
                        "aliases": [],
                        "relationship": "operated_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    }
                ],
            },
            {
                "entities": [
                    {
                        "canonical_name": "Orbit Cloud",
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    },
                    {
                        "canonical_name": "BluePeak",
                        "category": "competitor",
                        "target_relationship": "competitor",
                        "commercially_relevant": True,
                    },
                ]
            },
        )

        self.assertEqual(metrics["parent_discovery"]["web"]["mention_count"], 0)
        self.assertEqual(metrics["portfolio_visibility"]["web"]["mention_count"], 1)
        self.assertEqual(metrics["competitors"][0]["name"], "Northstar Group")
        self.assertEqual(metrics["competitors"][1]["name"], "Orbit Cloud")
        self.assertEqual(metrics["competitors"][2]["name"], "BluePeak")

    def test_literal_portfolio_alias_repairs_a_missed_llm_mention(self) -> None:
        reconciled = _reconcile_annotation(
            {
                "answer_id": 63,
                "valid": True,
                "target_mentioned": False,
                "target_position": None,
                "target_role": "absent",
                "sentiment": "unknown",
                "entity_mentions": [],
                "brand_answer": {
                    "directness": "not_applicable",
                    "specificity": "not_applicable",
                    "supported_facets": [],
                    "contradictions": [],
                },
                "evidence": [],
                "uncertainties": [],
            },
            {
                "answer": "Для этой задачи можно рассмотреть Centra.",
                "answer_sha256": "hash",
                "answer_model": "provider/model",
            },
            {
                "brand_name": "RW+",
                "brand_aliases": [],
                "entity_scope": [
                    {
                        "canonical_name": "Centra",
                        "aliases": [],
                        "relationship": "owned_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    }
                ],
            },
            {
                "target_aliases": ["RW+"],
                "entities": [
                    {
                        "canonical_name": "Centra",
                        "aliases": ["Centra"],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    }
                ],
            },
        )
        self.assertEqual(
            reconciled["entity_mentions"][0]["canonical_name"],
            "Centra",
        )
        self.assertTrue(reconciled["_reconciliation_notes"])
        self.assertEqual(reconciled["_annotation_input_sha256"], "")

    def test_generic_service_requires_explicit_target_annotation(self) -> None:
        item = {
            "answer_id": 64,
            "valid": True,
            "target_mentioned": False,
            "target_position": None,
            "target_role": "absent",
            "sentiment": "unknown",
            "entity_mentions": [],
            "brand_answer": {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
            "evidence": [],
            "uncertainties": [],
        }
        profile = {
            "brand_name": "Realweb",
            "brand_aliases": ["Реалвеб"],
            "entity_scope": [
                {
                    "canonical_name": "DOOH",
                    "aliases": ["цифровая наружная реклама"],
                    "entity_type": "service",
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        catalog = {
            "target_aliases": ["Realweb", "Реалвеб"],
            "entities": [
                {
                    "canonical_name": "DOOH",
                    "aliases": ["цифровая наружная реклама"],
                    "category": "target",
                    "target_relationship": "portfolio_entity",
                    "commercially_relevant": True,
                    "mention_policy": "requires_target_attribution",
                }
            ],
        }

        generic_only = _reconcile_annotation(
            item,
            {
                "answer": "Для кампании подойдёт DOOH.",
                "answer_sha256": "hash-1",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )
        attributed = _reconcile_annotation(
            {
                **item,
                "entity_mentions": [
                    {
                        "canonical_name": "DOOH",
                        "position": None,
                        "role": "mentioned",
                        "attributed_to_target": True,
                        "evidence": "Realweb предлагает DOOH",
                    }
                ],
            },
            {
                "answer": "Realweb предлагает DOOH.",
                "answer_sha256": "hash-2",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )

        self.assertEqual(generic_only["entity_mentions"], [])
        self.assertEqual(
            [value["canonical_name"] for value in attributed["entity_mentions"]],
            ["DOOH"],
        )

    def test_reconciliation_clears_target_attribution_outside_site_portfolio(
        self,
    ) -> None:
        answer_text = (
            "Omni360 — независимая DSP-платформа. "
            "Programmatic использует DSP и DMP."
        )
        reconciled = _reconcile_annotation(
            {
                "answer_id": 120,
                "valid": True,
                "target_mentioned": False,
                "target_position": None,
                "target_role": "absent",
                "sentiment": "unknown",
                "entity_mentions": [
                    {
                        "canonical_name": "Omni360",
                        "position": 1,
                        "role": "mentioned",
                        "attributed_to_target": True,
                        "evidence": "Omni360 — независимая DSP-платформа",
                    },
                    {
                        "canonical_name": "Example DSP",
                        "position": None,
                        "role": "mentioned",
                        "attributed_to_target": True,
                        "evidence": "Programmatic использует DSP",
                    },
                ],
                "brand_answer": {
                    "directness": "not_applicable",
                    "specificity": "not_applicable",
                    "supported_facets": [],
                    "contradictions": [],
                },
                "evidence": [],
                "uncertainties": [],
            },
            {
                "answer": answer_text,
                "answer_sha256": "hash",
                "answer_model": "provider/model",
            },
            {
                "brand_name": "Example",
                "brand_aliases": [],
                "entity_scope": [],
            },
            {
                "target_aliases": ["Example"],
                "entities": [
                    {
                        "canonical_name": "Omni360",
                        "aliases": [],
                        "category": "competitor",
                        "target_relationship": "competitor",
                    },
                    {
                        "canonical_name": "Example DSP",
                        "aliases": [
                            {
                                "value": "DSP",
                                "match_policy": "requires_target_attribution",
                            }
                        ],
                        "category": "other",
                        "target_relationship": "unrelated",
                        "commercially_relevant": False,
                        "_profile_membership_confirmed": False,
                    },
                ],
            },
        )

        self.assertTrue(
            all(
                mention["attributed_to_target"] is False
                for mention in reconciled["entity_mentions"]
            )
        )
        self.assertEqual(reconciled["target_mentioned"], False)
        self.assertTrue(
            any(
                "вне подтверждённого портфеля" in note
                for note in reconciled["_reconciliation_notes"]
            )
        )

    def test_portfolio_metrics_ground_generic_services_in_raw_answer(self) -> None:
        rows = [
            {
                "answer_id": 1,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "DOOH помогает охватить аудиторию в городе.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "DOOH",
                            "position": None,
                            "role": "mentioned",
                            "attributed_to_target": False,
                            "evidence": "DOOH помогает охватить аудиторию",
                        }
                    ],
                },
            },
            {
                "answer_id": 2,
                "mode": "web",
                "provider_key": "gemini",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Realweb предлагает DOOH для городских кампаний.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": None,
                    "target_role": "mentioned",
                    "sentiment": "neutral",
                    "entity_mentions": [
                        {
                            "canonical_name": "DOOH",
                            "position": None,
                            "role": "mentioned",
                            "attributed_to_target": True,
                            "evidence": "Realweb предлагает DOOH",
                        }
                    ],
                },
            },
            {
                "answer_id": 3,
                "mode": "web",
                "provider_key": "claude",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Для автоматизации можно рассмотреть Garpun.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "Garpun",
                            "position": None,
                            "role": "mentioned",
                        }
                    ],
                },
            },
        ]
        metrics = _compute_metrics(
            rows,
            {
                "brand_name": "Realweb",
                "brand_aliases": ["Реалвеб"],
                "entity_scope": [
                    {
                        "canonical_name": "DOOH",
                        "aliases": [],
                        "entity_type": "service",
                        "relationship": "offered_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    },
                    {
                        "canonical_name": "Garpun",
                        "aliases": [],
                        "entity_type": "platform",
                        "relationship": "owned_by",
                        "commercially_relevant": True,
                        "confidence": "high",
                    },
                ],
            },
            {
                "target_aliases": ["Realweb", "Реалвеб"],
                "entities": [
                    {
                        "canonical_name": "Realweb",
                        "aliases": ["Реалвеб"],
                        "category": "target",
                        "target_relationship": "exact_target",
                        "commercially_relevant": True,
                    },
                    {
                        "canonical_name": "DOOH",
                        "aliases": [],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "requires_target_attribution",
                    },
                    {
                        "canonical_name": "Garpun",
                        "aliases": [],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    },
                ],
            },
        )

        portfolio = metrics["portfolio_visibility"]["web"]
        self.assertEqual(portfolio["mention_count"], 2)
        self.assertEqual(portfolio["mention_rate"], 66.7)
        self.assertEqual(
            [item["name"] for item in portfolio["mentioned_entities"]],
            ["DOOH", "Garpun"],
        )

    def test_generic_cooccurrence_is_not_reconciled_or_grounded(self) -> None:
        profile = {
            "brand_name": "Realweb",
            "brand_aliases": [],
            "entity_scope": [
                {
                    "canonical_name": "DOOH",
                    "aliases": [],
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        entity = {
            "canonical_name": "DOOH",
            "aliases": [],
            "category": "target",
            "target_relationship": "portfolio_entity",
            "commercially_relevant": True,
            "mention_policy": "requires_target_attribution",
        }
        catalog = {
            "target_aliases": ["Realweb"],
            "entities": [entity],
        }
        base_annotation = {
            "answer_id": 65,
            "valid": True,
            "target_mentioned": False,
            "target_position": None,
            "target_role": "absent",
            "sentiment": "unknown",
            "entity_mentions": [],
            "brand_answer": {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
            "evidence": [],
            "uncertainties": [],
        }
        false_relations = [
            "Realweb и Okkam сравнили рынок DOOH.",
            "Для DOOH рекомендуют Okkam, а Realweb силён в performance.",
        ]

        for answer_text in false_relations:
            with self.subTest(answer_text=answer_text):
                reconciled = _reconcile_annotation(
                    base_annotation,
                    {
                        "answer": answer_text,
                        "answer_sha256": "hash",
                        "answer_model": "provider/model",
                    },
                    profile,
                    catalog,
                )
                self.assertEqual(reconciled["entity_mentions"], [])
                self.assertFalse(
                    _portfolio_entity_is_grounded(
                        answer_text,
                        entity,
                        ["Realweb", "Риалвеб"],
                        {
                            "canonical_name": "DOOH",
                            "attributed_to_target": True,
                            "evidence": answer_text,
                        },
                    )
                )

    def test_competitor_service_attribution_does_not_leak_to_target(
        self,
    ) -> None:
        answer_text = (
            "Okkam предлагает DOOH; Realweb предлагает аналитику."
        )
        generic_entity = {
            "aliases": [],
            "mention_policy": "requires_target_attribution",
        }

        self.assertFalse(
            _portfolio_entity_is_grounded(
                answer_text,
                {**generic_entity, "canonical_name": "DOOH"},
                ["Realweb"],
                {
                    "attributed_to_target": True,
                    "evidence": answer_text,
                },
            )
        )
        self.assertTrue(
            _portfolio_entity_is_grounded(
                answer_text,
                {
                    **generic_entity,
                    "canonical_name": "аналитика",
                    "aliases": ["аналитику"],
                },
                ["Realweb"],
                {
                    "attributed_to_target": True,
                    "evidence": "Realweb предлагает аналитику",
                },
            )
        )

    def test_prose_offer_is_not_mistaken_for_a_structured_label(self) -> None:
        answer_text = (
            "OMD, Media Instinct, Realweb и другие агентства "
            "предлагают закупку programmatic DOOH как сервис."
        )
        entity = {
            "canonical_name": "DOOH Realweb",
            "aliases": ["programmatic DOOH"],
            "mention_policy": "requires_target_attribution",
        }

        self.assertTrue(
            _portfolio_entity_is_grounded(
                answer_text,
                entity,
                ["Realweb"],
                {
                    "attributed_to_target": True,
                    "evidence": answer_text,
                },
            )
        )

    def test_profile_confirmed_contextual_aliases_support_inflections(
        self,
    ) -> None:
        entity = {
            "canonical_name": "Стратегия и продвижение Example",
            "aliases": [
                {
                    "value": "стратегия",
                    "match_policy": "requires_target_attribution",
                }
            ],
            "mention_policy": "requires_target_attribution",
        }
        answer_text = (
            "Example предлагает разработку стратегий и ведение кампаний."
        )

        self.assertTrue(
            _portfolio_entity_is_grounded(
                answer_text,
                entity,
                ["Example"],
                {
                    "attributed_to_target": True,
                    "evidence": answer_text,
                },
            )
        )
        competitor_text = (
            "Okkam предлагает разработку стратегий, "
            "а Example упомянут для сравнения."
        )
        self.assertFalse(
            _portfolio_entity_is_grounded(
                competitor_text,
                entity,
                ["Example"],
                {
                    "attributed_to_target": True,
                    "evidence": competitor_text,
                },
            )
        )

    def test_target_owned_headings_and_explicit_site_claims_are_grounded(
        self,
    ) -> None:
        entity = {
            "canonical_name": "DOOH Example",
            "aliases": [
                {
                    "value": "DOOH",
                    "match_policy": "requires_target_attribution",
                },
                {
                    "value": "цифровая наружная реклама",
                    "match_policy": "requires_target_attribution",
                },
            ],
            "mention_policy": "requires_target_attribution",
        }
        positives = [
            "У Example:\n- DOOH — отдельное направление.",
            "На сайте Example указано: DOOH · первое место в рейтинге.",
            "Example — агентство с хорошими DOOH-кейсами.",
        ]
        for answer_text in positives:
            with self.subTest(answer_text=answer_text):
                self.assertTrue(
                    _portfolio_entity_is_grounded(
                        answer_text,
                        entity,
                        ["Example"],
                        {
                            "attributed_to_target": True,
                            "evidence": answer_text,
                        },
                    )
                )

        self.assertFalse(
            _portfolio_entity_is_grounded(
                "Example закупает рекламу для клиентов.",
                entity,
                ["Example"],
                {
                    "attributed_to_target": True,
                    "evidence": "Example закупает рекламу для клиентов.",
                },
            )
        )

    def test_structured_target_field_attributes_generic_service(self) -> None:
        entity = {
            "canonical_name": "DOOH",
            "aliases": ["цифровая наружная реклама"],
            "mention_policy": "requires_target_attribution",
        }
        structured_fragments = [
            "### Realweb\n**Сильная сторона:** DOOH и performance.",
            "• Realweb\nСпециализация — performance и DOOH",
            "Realweb\nПодходит для: DOOH-задач",
            "Realweb — Хорошо сочетает performance и DOOH",
            "1. Realweb\nЭкспертиза: programmatic, DOOH",
            "Realweb\nКомпетенции:\n- DOOH",
            "• Realweb\n- Предлагает планирование и закупку DOOH.",
            (
                "Realweb (Риалвеб)\n"
                "В чём сила: медиазакупка и DOOH."
            ),
            (
                "Realweb. Сильная сторона: универсальное агентство. "
                "Хорошо сочетает performance и DOOH."
            ),
            (
                "Realweb (Риалвеб)\n"
                "Аналитика и технологии: DOOH-дашборды."
            ),
            (
                "Realweb\n"
                "Аналитика и ПО: собственная DOOH-платформа."
            ),
            (
                "Realweb\n"
                "Сильная сторона: performance\n"
                "Компетенции: цифровая наружная реклама"
            ),
        ]

        for answer_text in structured_fragments:
            with self.subTest(answer_text=answer_text):
                self.assertTrue(
                    _portfolio_entity_is_grounded(
                        answer_text,
                        entity,
                        ["Realweb", "Риалвеб"],
                        {
                            "attributed_to_target": True,
                            "evidence": answer_text,
                        },
                    )
                )

    def test_structured_target_field_rejects_cross_fragment_cooccurrence(
        self,
    ) -> None:
        entity = {
            "canonical_name": "DOOH",
            "aliases": [],
            "mention_policy": "requires_target_attribution",
        }
        unrelated_fragments = [
            "Realweb предлагает performance. Okkam предлагает DOOH.",
            "Realweb и DOOH представлены в одном обзоре.",
            "Realweb\n\nСильная сторона: DOOH",
            "Realweb\nOkkam\nСильная сторона: DOOH",
            "Realweb, DOOH",
            "Realweb — DOOH",
            "Realweb: DOOH",
            "Realweb. Okkam предлагает DOOH.",
            "Realweb\nСильная сторона: Okkam предлагает DOOH",
            "Realweb (Okkam)\nСпециализация: DOOH",
        ]

        for answer_text in unrelated_fragments:
            with self.subTest(answer_text=answer_text):
                self.assertFalse(
                    _portfolio_entity_is_grounded(
                        answer_text,
                        entity,
                        ["Realweb"],
                        {
                            "attributed_to_target": True,
                            "evidence": answer_text,
                        },
                    )
                )

    def test_reconciliation_repairs_only_literal_attribution_evidence(
        self,
    ) -> None:
        profile = {
            "brand_name": "Realweb",
            "brand_aliases": ["Риалвеб"],
            "entity_scope": [
                {
                    "canonical_name": "DOOH Realweb",
                    "aliases": ["DOOH"],
                    "relationship": "offered_by",
                    "commercially_relevant": True,
                    "confidence": "high",
                }
            ],
        }
        entity = {
            "canonical_name": "DOOH Realweb",
            "aliases": [
                {
                    "value": "DOOH",
                    "match_policy": "requires_target_attribution",
                }
            ],
            "category": "target",
            "target_relationship": "portfolio_entity",
            "commercially_relevant": True,
            "mention_policy": "requires_target_attribution",
        }
        catalog = {
            "target_aliases": ["Realweb"],
            "entities": [entity],
        }
        answer = (
            "• Realweb\n"
            "  - Предлагает планирование и закупку programmatic DOOH.\n"
            "  - Работает как интегратор."
        )
        annotation = {
            "answer_id": 82,
            "valid": True,
            "target_mentioned": True,
            "target_position": 1,
            "target_role": "recommended",
            "sentiment": "positive",
            "entity_mentions": [
                {
                    "canonical_name": "DOOH Realweb",
                    "position": 1,
                    "role": "recommended",
                    "attributed_to_target": True,
                    "evidence": (
                        "Realweb — предлагает планирование и закупку "
                        "programmatic DOOH."
                    ),
                }
            ],
            "evidence": [],
            "uncertainties": [],
            "brand_answer": {
                "directness": "not_applicable",
                "specificity": "not_applicable",
                "supported_facets": [],
                "contradictions": [],
            },
        }

        reconciled = _reconcile_annotation(
            annotation,
            {
                "answer": answer,
                "answer_sha256": "hash",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )
        mention = reconciled["entity_mentions"][0]
        self.assertTrue(mention["attributed_to_target"])
        self.assertIn(mention["evidence"], answer)
        self.assertTrue(
            _portfolio_entity_is_grounded(
                answer,
                entity,
                ["Realweb", "Риалвеб"],
                mention,
            )
        )

        false_reconciled = _reconcile_annotation(
            {
                **annotation,
                "entity_mentions": [
                    {
                        **annotation["entity_mentions"][0],
                        "evidence": "Realweb, DOOH",
                    }
                ],
            },
            {
                "answer": "Realweb, DOOH",
                "answer_sha256": "hash",
                "answer_model": "provider/model",
            },
            profile,
            catalog,
        )
        self.assertFalse(
            false_reconciled["entity_mentions"][0][
                "attributed_to_target"
            ]
        )

    def test_portfolio_grounding_fails_closed_without_raw_answer(self) -> None:
        self.assertFalse(
            _portfolio_entity_is_grounded(
                "",
                {
                    "canonical_name": "Garpun",
                    "aliases": ["Гарпун"],
                    "mention_policy": "standalone",
                },
                ["Realweb"],
                {
                    "canonical_name": "Garpun",
                    "attributed_to_target": False,
                    "evidence": "Garpun",
                },
            )
        )

    def test_alias_match_policy_is_per_alias_and_backwards_compatible(
        self,
    ) -> None:
        entity = {
            "canonical_name": "Campaign 360",
            "aliases": [
                {
                    "value": "campaign",
                    "match_policy": "requires_target_attribution",
                },
                "Campaign360",
            ],
            "mention_policy": "standalone",
        }

        self.assertFalse(
            _portfolio_entity_is_grounded(
                "This campaign improved the result.",
                entity,
                ["Realweb"],
                {
                    "attributed_to_target": False,
                    "evidence": "campaign",
                },
            )
        )
        self.assertTrue(
            _portfolio_entity_is_grounded(
                "Campaign 360 improved the result.",
                entity,
                ["Realweb"],
            )
        )
        self.assertTrue(
            _portfolio_entity_is_grounded(
                "Campaign360 improved the result.",
                entity,
                ["Realweb"],
            )
        )

    def test_declared_policy_is_not_overridden_by_entity_type(self) -> None:
        self.assertEqual(
            _portfolio_mention_policy(
                {
                    "canonical_name": "Analytics",
                    "mention_policy": "requires_target_attribution",
                },
                {
                    "entity_scope": [
                        {
                            "canonical_name": "Analytics",
                            "entity_type": "product",
                        }
                    ]
                },
            ),
            "requires_target_attribution",
        )
        self.assertEqual(
            _portfolio_mention_policy(
                {
                    "canonical_name": "Centra",
                    "mention_policy": "standalone",
                },
                {
                    "entity_scope": [
                        {
                            "canonical_name": "Centra",
                            "entity_type": "business_unit",
                        }
                    ]
                },
            ),
            "standalone",
        )

    def test_unconfirmed_catalog_candidate_cannot_enter_portfolio(self) -> None:
        rows = [
            {
                "answer_id": 70,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": "Рассмотрите Garpun.",
                "annotation": {
                    "valid": True,
                    "target_mentioned": False,
                    "target_position": None,
                    "target_role": "absent",
                    "sentiment": "unknown",
                    "entity_mentions": [
                        {
                            "canonical_name": "Garpun",
                            "position": 1,
                            "role": "recommended",
                        }
                    ],
                },
            }
        ]
        metrics = _compute_metrics(
            rows,
            {
                "brand_name": "Realweb",
                "entity_scope": [],
                "products": [],
            },
            {
                "entities": [
                    {
                        "canonical_name": "Garpun",
                        "aliases": [],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "standalone",
                    }
                ]
            },
        )

        self.assertIsNone(
            metrics["portfolio_visibility"]["web"]["mention_count"]
        )
        self.assertEqual(
            metrics["portfolio_visibility"]["web"]["mentioned_entities"],
            [],
        )
        self.assertEqual(
            metrics["portfolio_visibility"]["web"]["data_state"],
            "unavailable",
        )
        self.assertEqual(
            metrics["portfolio_scope"],
            {
                "state": "unavailable",
                "candidate_entities": 1,
                "confirmed_entities": 0,
                "rejected_entities": 1,
                "reason": "target_portfolio_unconfirmed",
            },
        )

    def test_multiple_profile_products_count_one_answer_once(self) -> None:
        answer_text = "Realweb предлагает: DOOH, programmatic, аналитика."
        entity_names = ["DOOH", "programmatic", "аналитика"]
        rows = [
            {
                "answer_id": 71,
                "mode": "web",
                "provider_key": "openai",
                "prompt_id": 1,
                "prompt_key": "u-1",
                "intent_class": "I",
                "role": "unbranded_discovery",
                "status": "completed",
                "answer_text": answer_text,
                "annotation": {
                    "valid": True,
                    "target_mentioned": True,
                    "target_position": None,
                    "target_role": "mentioned",
                    "sentiment": "neutral",
                    "entity_mentions": [
                        {
                            "canonical_name": name,
                            "position": None,
                            "role": "mentioned",
                            "attributed_to_target": True,
                            "evidence": answer_text,
                        }
                        for name in entity_names
                    ],
                },
            }
        ]
        metrics = _compute_metrics(
            rows,
            {
                "brand_name": "Realweb",
                "brand_aliases": [],
                "entity_scope": [],
                "products": entity_names,
            },
            {
                "target_aliases": ["Realweb"],
                "entities": [
                    {
                        "canonical_name": name,
                        "aliases": [],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                        "commercially_relevant": True,
                        "mention_policy": "requires_target_attribution",
                    }
                    for name in entity_names
                ],
            },
        )
        portfolio = metrics["portfolio_visibility"]["web"]

        self.assertEqual(portfolio["mention_count"], 1)
        self.assertEqual(
            {item["name"] for item in portfolio["mentioned_entities"]},
            set(entity_names),
        )
        self.assertTrue(
            all(
                item["answer_count"] == 1
                for item in portfolio["mentioned_entities"]
            )
        )

    def test_alias_policy_change_invalidates_annotation_context(self) -> None:
        profile = {"brand_name": "Realweb"}
        standalone = {
            "target_aliases": ["Realweb"],
            "entities": [
                {
                    "canonical_name": "Campaign 360",
                    "aliases": [
                        {
                            "value": "campaign",
                            "match_policy": "standalone",
                        }
                    ],
                }
            ],
        }
        contextual = {
            "target_aliases": ["Realweb"],
            "entities": [
                {
                    "canonical_name": "Campaign 360",
                    "aliases": [
                        {
                            "value": "campaign",
                            "match_policy": "requires_target_attribution",
                        }
                    ],
                }
            ],
        }

        self.assertNotEqual(
            _annotation_context_sha256(profile, standalone),
            _annotation_context_sha256(profile, contextual),
        )

    def test_target_false_positive_without_literal_alias_is_removed(self) -> None:
        reconciled = _reconcile_annotation(
            {
                "answer_id": 101,
                "valid": True,
                "target_mentioned": True,
                "target_position": 3,
                "target_role": "recommended",
                "sentiment": "positive",
                "entity_mentions": [
                    {
                        "canonical_name": "Realweb",
                        "position": 3,
                        "role": "recommended",
                    },
                    {
                        "canonical_name": "Centra",
                        "position": 4,
                        "role": "mentioned",
                    },
                ],
                "brand_answer": {
                    "directness": "not_applicable",
                    "specificity": "not_applicable",
                    "supported_facets": [],
                    "contradictions": [],
                },
                "evidence": [
                    "«наравне с Realweb»",
                    "«Centra названа среди платформ»",
                ],
                "uncertainties": [],
            },
            {
                "answer": "Среди платформ в ответе названа Centra.",
                "answer_sha256": "hash",
                "answer_model": "provider/model",
            },
            {
                "brand_name": "Realweb",
                "brand_aliases": ["Реалвеб", "Риалвеб"],
            },
            {
                "target_aliases": ["Realweb", "Реалвеб", "Риалвеб"],
                "entities": [
                    {
                        "canonical_name": "Realweb",
                        "aliases": ["Реалвеб", "Риалвеб"],
                        "category": "target",
                        "target_relationship": "exact_target",
                    },
                    {
                        "canonical_name": "Centra",
                        "aliases": ["Centra"],
                        "category": "target",
                        "target_relationship": "portfolio_entity",
                    },
                ],
            },
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertIsNone(reconciled["target_position"])
        self.assertEqual(reconciled["target_role"], "absent")
        self.assertEqual(reconciled["sentiment"], "unknown")
        self.assertEqual(
            [item["canonical_name"] for item in reconciled["entity_mentions"]],
            ["Centra"],
        )
        self.assertEqual(
            reconciled["evidence"],
            [],
        )
        self.assertTrue(
            any(
                "Снято неподтверждённое упоминание" in note
                for note in reconciled["_reconciliation_notes"]
            )
        )

    def test_alias_inside_url_or_email_is_not_a_brand_mention(self) -> None:
        answer_text = (
            "Источник: https://example.ru/rating-mgcom-realweb-arrowmedia/ "
            "и почта tender-realweb@example.ru."
        )
        reconciled = _reconcile_annotation(
            {
                "valid": True,
                "target_mentioned": True,
                "target_position": 1,
                "target_role": "mentioned",
                "sentiment": "neutral",
                "entity_mentions": [],
                "brand_answer": {},
                "evidence": [],
                "uncertainties": [],
            },
            {
                "answer": answer_text,
                "answer_sha256": "hash",
                "answer_model": "provider/model",
            },
            {"brand_name": "Realweb", "brand_aliases": []},
            {"target_aliases": ["Realweb"], "entities": []},
        )

        self.assertFalse(reconciled["target_mentioned"])
        self.assertEqual(reconciled["target_role"], "absent")

    def test_headline_emphasis_is_exact_non_overlapping_and_lossless(self) -> None:
        report = _sanitize_headline_emphasis(
            {
                "headline": "Бренд знают, но редко предлагают без подсказки",
                "headline_emphasis": [
                    "знают",
                    "редко предлагают",
                    "перефразированного текста здесь нет",
                ],
            }
        )
        self.assertEqual(
            report["headline_emphasis"],
            ["знают", "редко предлагают"],
        )


class DatabaseSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()

    async def test_legacy_memory_observation_requires_the_complete_clean_cohort(
        self,
    ) -> None:
        run_id = f"test-legacy-memory-cohort-{uuid.uuid4()}"
        prompt_rows: list[VisibilityPrompt] = []
        prompt_payload: list[dict[str, object]] = []
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.completed,
                    config_json={"pipeline_version": "aiv-2026-07"},
                )
            )
            for sequence in range(1, 10):
                payload = {
                    "prompt_key": f"legacy-{sequence}",
                    "intent_class": ("I", "E", "T")[sequence % 3],
                    "role": (
                        "brand_diagnostic"
                        if sequence > 6
                        else "unbranded_discovery"
                    ),
                    "text": f"Исторический сценарий {sequence}",
                    "rationale": f"Причина {sequence}",
                }
                prompt_payload.append(payload)
                prompt = VisibilityPrompt(
                    run_id=run_id,
                    sequence=sequence,
                    **payload,
                )
                session.add(prompt)
                prompt_rows.append(prompt)
            await session.flush()
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="prompts",
                    artifact_key="prompt_set",
                    status="completed",
                    prompt_version="aiv-2026-07-30",
                    output_json={"prompts": prompt_payload},
                )
            )
            first_answer: ModelAnswer | None = None
            for prompt in prompt_rows:
                for provider_key, model in LEGACY_MEMORY_MODELS.items():
                    usage: dict[str, object] = {"total_tokens": 10}
                    if first_answer is None:
                        usage["_aiv_panel_contract"] = {
                            "version": LEGACY_PANEL_CONTRACT_VERSION,
                            "request_sha256": _legacy_panel_request_sha256(
                                prompt_text=prompt.text,
                                mode="memory",
                                provider_key=provider_key,
                                model=model,
                            ),
                        }
                    answer = ModelAnswer(
                        run_id=run_id,
                        prompt_id=prompt.id,
                        provider_key=provider_key,
                        model=model,
                        mode="memory",
                        status="completed",
                        response_text=f"Ответ {prompt.sequence} {provider_key}",
                        citations_json=[],
                        usage_json=usage,
                    )
                    session.add(answer)
                    if first_answer is None:
                        first_answer = answer
            await session.commit()

        try:
            contract = await _legacy_panel_run_contract(run_id)
            self.assertTrue(contract["eligible"])
            self.assertTrue(contract["memory_observation_eligible"])
            self.assertEqual(contract["memory_cell_count"], 36)
            self.assertEqual(contract["memory_observation_reasons"], [])

            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer)
                        .where(ModelAnswer.run_id == run_id)
                        .order_by(ModelAnswer.id)
                    )
                ).scalars().first()
                assert answer is not None
                usage = dict(answer.usage_json or {})
                provenance = dict(usage["_aiv_panel_contract"])
                provenance["request_sha256"] = "corrupt"
                usage["_aiv_panel_contract"] = provenance
                answer.usage_json = usage
                await session.commit()
            contract = await _legacy_panel_run_contract(run_id)
            self.assertFalse(contract["memory_observation_eligible"])
            self.assertIn(
                "legacy_request_hash_mismatch",
                contract["memory_observation_reasons"],
            )

            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer)
                        .where(ModelAnswer.run_id == run_id)
                        .order_by(ModelAnswer.id)
                    )
                ).scalars().first()
                assert answer is not None
                await session.delete(answer)
                await session.commit()
            contract = await _legacy_panel_run_contract(run_id)
            self.assertFalse(contract["memory_observation_eligible"])
            self.assertIn(
                "legacy_memory_cell_count_mismatch",
                contract["memory_observation_reasons"],
            )
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_sqlite_foreign_keys_are_enabled(self) -> None:
        async with SessionLocal() as session:
            enabled = (await session.execute(text("PRAGMA foreign_keys"))).scalar_one()
        self.assertEqual(enabled, 1)

    async def test_site_context_carries_the_user_domain_explicitly(self) -> None:
        run_id = f"test-context-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            session.add(
                SitePage(
                    run_id=run_id,
                    url="https://example.com/",
                    page_kind="home",
                    text_length=12,
                    main_text="Пример сайта",
                )
            )
            await session.commit()
        try:
            context = await _site_context(run_id)
            self.assertEqual(
                context["requested_site"],
                {
                    "domain": "example.com",
                    "url": "https://example.com/",
                },
            )
            self.assertEqual(context["pages"][0]["url"], "https://example.com/")
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_catalog_and_annotation_inputs_cover_both_modes_and_long_answers(
        self,
    ) -> None:
        run_id = f"test-answer-inputs-{uuid.uuid4()}"
        long_answer = "Начало. " + ("x" * 7_500) + " UNIQUE_TAIL"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения выбрать?",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            for mode in ("web", "memory"):
                answer = ModelAnswer(
                    run_id=run_id,
                    prompt_id=prompt.id,
                    provider_key=f"provider-{mode}",
                    model="test/model",
                    mode=mode,
                    status="completed",
                    response_text=long_answer,
                )
                session.add(answer)
                await session.flush()
                session.add(
                    AnswerAnnotation(
                        answer_id=answer.id,
                        annotation_json={
                            "valid": True,
                            "_annotation_version": ANNOTATION_VERSION,
                            "_answer_sha256": hashlib.sha256(
                                long_answer.encode("utf-8")
                            ).hexdigest(),
                            "_answer_model": "test/model",
                            "_annotation_input_sha256": "old-context",
                        },
                    )
                )
            await session.commit()

        try:
            catalog_answers = await _answers_for_catalog(run_id)
            self.assertEqual(
                {item["mode"] for item in catalog_answers},
                {"web", "memory"},
            )
            self.assertTrue(
                all("UNIQUE_TAIL" in item["answer"] for item in catalog_answers)
            )

            self.assertEqual(
                await _unannotated_answers(
                    run_id,
                    annotation_input_sha256="old-context",
                ),
                [],
            )
            stale = await _unannotated_answers(
                run_id,
                annotation_input_sha256="new-context",
            )
            self.assertEqual(len(stale), 2)
            self.assertTrue(all("UNIQUE_TAIL" in item["answer"] for item in stale))
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_metric_rows_fail_closed_on_stale_or_ineligible_annotations(
        self,
    ) -> None:
        run_id = f"test-metric-provenance-{uuid.uuid4()}"
        profile = {"brand_name": "Example"}
        catalog = {"target_aliases": ["Example"], "entities": []}
        context_sha256 = _annotation_context_sha256(profile, catalog)
        answer_specs = [
            ("current", "completed", "Example указан.", "test/model", "current"),
            ("stale-raw", "completed", "Новый raw.", "test/model", "stale_raw"),
            ("stale-model", "completed", "Example указан снова.", "test/model", "stale_model"),
            ("stale-context", "completed", "Example назван.", "test/model", "stale_context"),
            ("failed", "failed", "Example в старом ответе.", "test/model", "current"),
            ("empty", "completed", "   ", "test/model", "current"),
        ]
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения выбрать?",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            for provider, status, raw, model, stale_kind in answer_specs:
                usage, citations = _attested_panel_usage(
                    prompt_text=prompt.text,
                    mode="web",
                    provider_key=provider,
                    model=model,
                )
                answer = ModelAnswer(
                    run_id=run_id,
                    prompt_id=prompt.id,
                    provider_key=provider,
                    model=model,
                    mode="web",
                    status=status,
                    response_text=raw,
                    citations_json=citations,
                    usage_json=usage,
                )
                session.add(answer)
                await session.flush()
                annotation_model = (
                    "old/model" if stale_kind == "stale_model" else model
                )
                annotation_context = (
                    "old-context"
                    if stale_kind == "stale_context"
                    else context_sha256
                )
                annotation_raw = (
                    "Старый raw." if stale_kind == "stale_raw" else raw
                )
                session.add(
                    AnswerAnnotation(
                        answer_id=answer.id,
                        annotation_json={
                            "valid": True,
                            "target_mentioned": True,
                            "target_position": 1,
                            "target_role": "recommended",
                            "sentiment": "positive",
                            "entity_mentions": [],
                            "_annotation_version": ANNOTATION_VERSION,
                            "_answer_sha256": hashlib.sha256(
                                annotation_raw.encode("utf-8")
                            ).hexdigest(),
                            "_answer_model": annotation_model,
                            "_annotation_input_sha256": annotation_context,
                        },
                    )
                )
            await session.commit()

        try:
            rows = await _metric_rows(
                run_id,
                annotation_input_sha256=context_sha256,
            )
            by_provider = {row["provider_key"]: row for row in rows}
            self.assertEqual(
                by_provider["current"]["annotation_state"],
                "current",
            )
            for provider in ("stale-raw", "stale-model", "stale-context"):
                self.assertEqual(
                    by_provider[provider]["annotation_state"],
                    "missing_or_stale",
                )
                self.assertEqual(by_provider[provider]["annotation"], {})
            for provider in ("failed", "empty"):
                self.assertEqual(
                    by_provider[provider]["annotation_state"],
                    "ineligible",
                )
                self.assertEqual(by_provider[provider]["annotation"], {})

            metrics = _compute_metrics(rows, profile, catalog)
            discovery = metrics["parent_discovery"]["web"]
            self.assertEqual(discovery["expected_answers"], 6)
            self.assertEqual(discovery["completed_answers"], 4)
            self.assertEqual(discovery["annotated_answers"], 1)
            self.assertEqual(discovery["valid_answers"], 1)
            self.assertEqual(discovery["mention_count"], 1)
            self.assertTrue(ANNOTATION_VERSION.endswith("annotations-v14"))
            self.assertTrue(METRICS_VERSION.endswith("metrics-v17"))
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_unattested_mode_is_limited_unknown_and_not_metric_eligible(
        self,
    ) -> None:
        run_id = f"test-unattested-mode-{uuid.uuid4()}"
        profile = {"brand_name": "Example"}
        catalog = {"target_aliases": ["Example"], "entities": []}
        context_sha256 = _annotation_context_sha256(profile, catalog)
        raw = "Example рекомендован."
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения выбрать?",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status="completed",
                response_text=raw,
                usage_json={"total_tokens": 10},
            )
            session.add(answer)
            await session.flush()
            session.add(
                AnswerAnnotation(
                    answer_id=answer.id,
                    annotation_json={
                        "valid": True,
                        "target_mentioned": True,
                        "target_position": 1,
                        "target_role": "recommended",
                        "sentiment": "positive",
                        "entity_mentions": [],
                        "_annotation_version": ANNOTATION_VERSION,
                        "_answer_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                        "_answer_model": "test/model",
                        "_annotation_input_sha256": context_sha256,
                    },
                )
            )
            await session.commit()

        try:
            with patch(
                "app.services.analyzer.panel_models",
                return_value=(
                    PanelModel(
                        key="openai",
                        label="ChatGPT",
                        model="test/model",
                        memory_model="test/model",
                    ),
                ),
            ):
                jobs = await _ensure_answer_rows(
                    run_id,
                    [prompt],
                    "web",
                )
            self.assertEqual(jobs, [])
            async with SessionLocal() as session:
                saved = (
                    await session.execute(
                        select(ModelAnswer).where(
                            ModelAnswer.run_id == run_id
                        )
                    )
                ).scalar_one()
            self.assertEqual(saved.status, "completed")
            self.assertEqual(saved.response_text, raw)

            rows = await _metric_rows(
                run_id,
                annotation_input_sha256=context_sha256,
            )
            self.assertEqual(rows[0]["annotation_state"], "current")
            self.assertFalse(rows[0]["metric_eligible"])
            self.assertFalse(rows[0]["web_attested"])
            self.assertEqual(
                rows[0]["web_attestation_reason"],
                "legacy_run_contract_unverified",
            )

            metrics = _compute_metrics(rows, profile, catalog)
            discovery = metrics["parent_discovery"]["web"]
            self.assertEqual(discovery["data_state"], "unavailable")
            self.assertEqual(discovery["state"], "unknown")
            self.assertEqual(discovery["valid_answers"], 0)
            self.assertIsNone(discovery["coverage_rate"])
            self.assertIsNone(discovery["mention_rate"])
            self.assertEqual(metrics["quality"]["state"], "limited")
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_completed_panel_evidence_is_append_only(self) -> None:
        run_id = f"test-panel-contract-{uuid.uuid4()}"
        panel = PanelModel(
            key="openai",
            label="ChatGPT",
            model="test/model",
            memory_model="test/model",
        )
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения выбрать?",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            usage, citations = _attested_panel_usage(
                prompt_text=prompt.text,
                mode="web",
                provider_key="openai",
                model="test/model",
            )
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status="completed",
                response_text="Сохранённый ответ",
                citations_json=citations,
                usage_json={"total_tokens": 10, **usage},
            )
            session.add(answer)
            await session.flush()
            session.add(
                AnswerAnnotation(
                    answer_id=answer.id,
                    annotation_json={"valid": True},
                )
            )
            await session.commit()

        try:
            async with SessionLocal() as session:
                prompt = (
                    await session.execute(
                        select(VisibilityPrompt).where(
                            VisibilityPrompt.run_id == run_id
                        )
                    )
                ).scalar_one()
            with patch(
                "app.services.analyzer.panel_models",
                return_value=(panel,),
            ):
                jobs = await _ensure_answer_rows(run_id, [prompt], "web")
            self.assertEqual(jobs, [])

            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
                provenance = answer.usage_json["_aiv_panel_contract"]
            self.assertEqual(provenance["version"], PANEL_CONTRACT_VERSION)
            self.assertTrue(provenance["request_sha256"])
            self.assertTrue(provenance["request_policy_sha256"])
            self.assertEqual(
                provenance["attestation_version"],
                WEB_ATTESTATION_VERSION,
            )
            self.assertTrue(
                provenance["web_attestation"]["metric_eligible"]
            )

            with (
                patch(
                    "app.services.analyzer.panel_models",
                    return_value=(panel,),
                ),
                patch(
                    "app.services.analyzer._panel_system",
                    return_value="Изменённый системный контракт",
                ),
            ):
                jobs = await _ensure_answer_rows(run_id, [prompt], "web")
            self.assertEqual(jobs, [])
            async with SessionLocal() as session:
                answer = (
                    await session.execute(
                        select(ModelAnswer).where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
                annotation_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(AnswerAnnotation)
                        .where(AnswerAnnotation.answer_id == answer.id)
                    )
                ).scalar_one()
            self.assertEqual(answer.status, "completed")
            self.assertEqual(answer.response_text, "Сохранённый ответ")
            self.assertEqual(annotation_count, 1)
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_changed_prompt_discards_stale_answers_and_annotations(self) -> None:
        run_id = f"test-prompts-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                )
            )
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="u-1",
                intent_class="I",
                role="unbranded_discovery",
                text="Старый пользовательский запрос",
                rationale="Старое основание",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status="completed",
                response_text="Ответ на старый запрос",
            )
            session.add(answer)
            await session.flush()
            old_answer_id = answer.id
            session.add(
                AnswerAnnotation(
                    answer_id=answer.id,
                    annotation_json={"valid": True},
                )
            )
            await session.commit()

        try:
            persisted = await _persist_prompts(
                run_id,
                {
                    "prompts": [
                        {
                            "prompt_key": "u-1",
                            "intent_class": "I",
                            "role": "unbranded_discovery",
                            "text": "Новый пользовательский запрос",
                            "rationale": "Новое основание",
                        }
                    ]
                },
            )
            self.assertEqual(persisted[0].text, "Новый пользовательский запрос")
            async with SessionLocal() as session:
                answer_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(ModelAnswer)
                        .where(ModelAnswer.run_id == run_id)
                    )
                ).scalar_one()
                annotation_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(AnswerAnnotation)
                        .where(AnswerAnnotation.answer_id == old_answer_id)
                    )
                ).scalar_one()
            self.assertEqual(answer_count, 0)
            self.assertEqual(annotation_count, 0)
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_run_delete_cascades_through_new_pipeline_tables(self) -> None:
        run_id = f"test-{uuid.uuid4()}"
        async with SessionLocal() as session:
            run = Run(
                id=run_id,
                domain="example.com",
                status=RunStatus.completed,
                config_json={},
                progress_current=100,
                progress_total=100,
                progress_percent=100,
            )
            session.add(run)
            prompt = VisibilityPrompt(
                run_id=run_id,
                prompt_key="i",
                intent_class="I",
                role="unbranded_discovery",
                text="Какие решения выбрать?",
                sequence=1,
            )
            session.add(prompt)
            await session.flush()
            answer = ModelAnswer(
                run_id=run_id,
                prompt_id=prompt.id,
                provider_key="openai",
                model="test/model",
                mode="web",
                status="completed",
                response_text="Ответ",
            )
            session.add(answer)
            await session.flush()
            session.add(AnswerAnnotation(answer_id=answer.id, annotation_json={"valid": True}))
            session.add(
                SitePage(
                    run_id=run_id,
                    url="https://example.com/",
                    page_kind="home",
                    text_length=10,
                )
            )
            session.add(
                RunArtifact(
                    run_id=run_id,
                    stage_key="report",
                    artifact_key="final",
                    status="completed",
                )
            )
            session.add(
                ReportIllustration(
                    run_id=run_id,
                    sequence=1,
                    title="Схема",
                    caption="Подпись",
                    alt_text="Описание",
                    generation_prompt="prompt",
                    model="image/model",
                )
            )
            session.add(
                DomainProbe(
                    run_id=run_id,
                    domain="example.com",
                    user_agent_label="GPTBot",
                    user_agent_string="GPTBot",
                    target_url="https://example.com/",
                    probe_type=ProbeType.main_page,
                    challenge_detected=False,
                    body_looks_empty=False,
                )
            )
            session.add(
                RobotsRule(
                    run_id=run_id,
                    domain="example.com",
                    bot_name="GPTBot",
                    rule="allow_all",
                )
            )
            await session.commit()
            await session.execute(delete(Run).where(Run.id == run_id))
            await session.commit()

            for model in (
                DomainProbe,
                RobotsRule,
                SitePage,
                RunArtifact,
                VisibilityPrompt,
                ModelAnswer,
                ReportIllustration,
            ):
                count = (
                    await session.execute(
                        select(func.count())
                        .select_from(model)
                        .where(model.run_id == run_id)
                    )
                ).scalar_one()
                self.assertEqual(count, 0, model.__name__)

            annotation_count = (
                await session.execute(
                    select(func.count())
                    .select_from(AnswerAnnotation)
                    .join(ModelAnswer)
                    .where(ModelAnswer.run_id == run_id)
                )
            ).scalar_one()
            self.assertEqual(annotation_count, 0, AnswerAnnotation.__name__)


class PublicApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await init_db()
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_create_run_accepts_only_one_domain_and_uses_fixed_config(self) -> None:
        domain = f"api-{uuid.uuid4().hex}.example.com"
        with patch("app.routes.runs.coordinator.wake") as wake:
            response = await self.client.post(
                "/api/runs",
                json={"domain": f"https://www.{domain}/path"},
            )
        self.assertEqual(response.status_code, 200)
        run_id = response.json()["run_id"]
        wake.assert_called_once_with()
        try:
            async with SessionLocal() as session:
                run = (
                    await session.execute(select(Run).where(Run.id == run_id))
                ).scalar_one()
                self.assertEqual(run.domain, domain)
                self.assertEqual(run.config_json["domains"], [domain])
                self.assertEqual(
                    run.config_json["user_agents"],
                    list(crawler.AUDIT_USER_AGENTS),
                )
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_create_run_rejects_hidden_analysis_options(self) -> None:
        with patch("app.routes.runs.coordinator.wake") as wake:
            response = await self.client.post(
                "/api/runs",
                json={
                    "domain": "example.com",
                    "model": "provider/model",
                    "domains": ["example.com", "example.org"],
                },
            )
        self.assertEqual(response.status_code, 422)
        wake.assert_not_called()

    async def test_concurrent_retry_claims_the_run_only_once(self) -> None:
        run_id = f"test-retry-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.failed,
                    config_json={},
                )
            )
            await session.commit()
        try:
            with patch("app.routes.runs.coordinator.wake") as wake:
                first, second = await asyncio.gather(
                    self.client.post(f"/api/runs/{run_id}/retry"),
                    self.client.post(f"/api/runs/{run_id}/retry"),
                )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(wake.call_count, 2)
            async with SessionLocal() as session:
                status, resume_count = (
                    await session.execute(
                        select(Run.status, Run.resume_count).where(
                            Run.id == run_id
                        )
                    )
                ).one()
            self.assertEqual(status, RunStatus.pending)
            self.assertEqual(resume_count, 1)
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_public_history_returns_every_run_newest_first(self) -> None:
        first_id = f"test-lookup-a-{uuid.uuid4()}"
        second_id = f"test-lookup-b-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=first_id,
                    domain="first.example",
                    status=RunStatus.completed,
                    config_json={},
                )
            )
            await session.commit()
            session.add(
                Run(
                    id=second_id,
                    domain="second.example",
                    status=RunStatus.completed,
                    config_json={},
                )
            )
            await session.commit()
        try:
            lookup = await self.client.post(
                "/api/runs/lookup",
                json={"ids": [second_id]},
            )
            self.assertEqual(lookup.status_code, 200)
            self.assertEqual(
                [item["id"] for item in lookup.json()],
                [second_id],
            )
            history = await self.client.get("/api/runs")
            self.assertEqual(history.status_code, 200)
            history_ids = [item["id"] for item in history.json()]
            self.assertIn(first_id, history_ids)
            self.assertIn(second_id, history_ids)
            self.assertLess(
                history_ids.index(second_id),
                history_ids.index(first_id),
            )
        finally:
            async with SessionLocal() as session:
                await session.execute(
                    delete(Run).where(Run.id.in_([first_id, second_id]))
                )
                await session.commit()

    async def test_public_detail_hides_internal_pipeline_data(self) -> None:
        run_id = f"test-public-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.completed,
                    config_json={"secret_internal": True},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    report_json={"brand": {"name": "Example"}},
                )
            )
            await session.commit()
        try:
            response = await self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            for forbidden in (
                "config_json",
                "probes",
                "robots_rules",
                "model_answers",
                "visibility_prompts",
                "share_token",
            ):
                self.assertNotIn(forbidden, body)
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

    async def test_share_token_resolves_safe_report_view(self) -> None:
        run_id = f"test-share-{uuid.uuid4()}"
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    domain="example.com",
                    status=RunStatus.completed,
                    config_json={},
                    progress_current=100,
                    progress_total=100,
                    progress_percent=100,
                    analysis_markdown="# Отчёт",
                )
            )
            await session.commit()
        try:
            generated = await self.client.post(f"/api/runs/{run_id}/share")
            self.assertEqual(generated.status_code, 200)
            token = generated.json()["share_token"]
            shared = await self.client.get(f"/api/shared/{token}")
            self.assertEqual(shared.status_code, 200)
            self.assertEqual(shared.json()["domain"], "example.com")
            self.assertNotIn("share_token", shared.json())
        finally:
            async with SessionLocal() as session:
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()


if __name__ == "__main__":
    unittest.main()
